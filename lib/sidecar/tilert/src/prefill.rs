// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-token prefill on vLLM's native gRPC server.
//!
//! vLLM runs TileRT's KV connector, which pushes a request's prompt KV to the
//! decode server named in that request's `kv_transfer_params`. The connector
//! ignores requests that do not name one.

use std::time::Duration;

use dynamo_backend_common::{BackendError, DynamoError};
use dynamo_sidecar_common::v14::{GrpcChannelPool, status_to_dynamo, stream_status_to_dynamo};
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcEndpoint, GrpcTransportConfig, cannot_connect,
    connection_timeout, json_to_struct_v14, stream_incomplete,
};
use tokio::time::{Instant, sleep_until, timeout_at};
use tonic_health_v14::pb::health_check_response::ServingStatus;
use tonic_health_v14::pb::{HealthCheckRequest, health_client::HealthClient};
use tonic_v14::Code;
use tonic_v14::transport::Channel;
use vllm_proto as pb;

use crate::error::backend_error;

const INFERENCE_SERVICE: &str = "vllm.Inference";
// The request's slot stays taken while its abort is in flight.
const ABORT_RPC_DEADLINE: Duration = Duration::from_secs(5);

#[derive(Clone, Debug)]
pub(crate) struct PrefillRequest {
    pub request_id: String,
    pub token_ids: Vec<u32>,
    pub temperature: f32,
    pub seed: Option<i64>,
    pub ignore_eos: bool,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub decode_kv_host: String,
    pub decode_ctrl_port: u16,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct PrefillOutcome {
    /// The sampled token, absent when the first token ended the reply.
    pub first_token: Option<u32>,
    /// The EOS or stop token that ended the reply, when vLLM reports it.
    pub stop_token: Option<u32>,
    pub prompt_tokens: Option<u32>,
    pub cached_tokens: Option<u32>,
}

pub(crate) fn build_request(request: PrefillRequest) -> Result<pb::GenerateRequest, DynamoError> {
    let kv_transfer_params = json_to_struct_v14(
        serde_json::json!({
            "tilert_host": request.decode_kv_host,
            "tilert_ctrl_port": request.decode_ctrl_port,
        }),
        "kv_transfer_params",
    )?;
    Ok(pb::GenerateRequest {
        request_id: request.request_id,
        prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
            ids: request.token_ids,
        })),
        temperature: Some(request.temperature),
        // The decode stage samples without top-k or top-p. Left unset, vLLM
        // would sample the first token with the model's generation defaults.
        sampling: Some(pb::RandomSampling {
            num_sequences: 1,
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: None,
            seed: request.seed,
        }),
        decoding: Some(pb::DecodingParameters {
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.repetition_penalty.unwrap_or(0.0),
            ..Default::default()
        }),
        stopping: Some(pb::StoppingCriteria {
            max_new_tokens: 1,
            ignore_eos: request.ignore_eos,
            ..Default::default()
        }),
        response: Some(pb::ResponseOptions {
            output_text: Some(false),
            output_token_ids: true,
            ..Default::default()
        }),
        kv: Some(pb::KvCacheParameters {
            kv_transfer_params: Some(kv_transfer_params),
            ..Default::default()
        }),
        ..Default::default()
    })
}

/// Folds the response stream of a one-token request.
#[derive(Default)]
pub(crate) struct OutcomeBuilder {
    outcome: PrefillOutcome,
    finished: bool,
}

impl OutcomeBuilder {
    pub(crate) fn push(&mut self, response: pb::GenerateResponse) -> Result<(), DynamoError> {
        if let Some(prompt) = response.prompt_info {
            self.outcome.prompt_tokens = Some(prompt.num_prompt_tokens);
        }
        let Some(output) = response.outputs else {
            return Ok(());
        };
        if self.outcome.first_token.is_none() {
            self.outcome.first_token = output.token_ids.first().copied();
        }
        let Some(finish) = output.finish_info else {
            return Ok(());
        };
        if pb::finish_info::FinishReason::try_from(finish.finish_reason)
            == Ok(pb::finish_info::FinishReason::Aborted)
        {
            return Err(backend_error(
                BackendError::Cancelled,
                "vLLM aborted the prefill request",
            ));
        }
        self.outcome.cached_tokens = finish.num_cached_tokens;
        self.outcome.stop_token = match finish.stop_reason {
            Some(pb::finish_info::StopReason::StopTokenId(id))
            | Some(pb::finish_info::StopReason::EosTokenId(id)) => Some(id),
            _ => None,
        };
        self.finished = true;
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<PrefillOutcome, DynamoError> {
        if !self.finished {
            return Err(stream_incomplete(
                "vLLM prefill stream ended before a finish message",
            ));
        }
        Ok(self.outcome)
    }
}

pub(crate) struct PrefillClient {
    pool: GrpcChannelPool,
}

impl PrefillClient {
    pub(crate) async fn connect(
        endpoint: &GrpcEndpoint,
        transport: GrpcTransportConfig,
        deadline: Instant,
    ) -> Result<Self, DynamoError> {
        let pool = timeout_at(
            deadline,
            GrpcChannelPool::connect("vLLM", endpoint, transport, false),
        )
        .await
        .map_err(|_| {
            connection_timeout(format!(
                "vLLM gRPC connection to {endpoint} exceeded the startup deadline"
            ))
        })??;
        let client = Self { pool };
        client
            .wait_serving(deadline, transport.retry_interval)
            .await?;
        Ok(client)
    }

    async fn wait_serving(
        &self,
        deadline: Instant,
        retry_interval: Duration,
    ) -> Result<(), DynamoError> {
        loop {
            let mut health = HealthClient::new(self.pool.next_channel());
            let check = health.check(HealthCheckRequest {
                service: INFERENCE_SERVICE.to_string(),
            });
            let last = match timeout_at(deadline, check).await {
                Ok(Ok(response)) => {
                    let status = ServingStatus::try_from(response.into_inner().status)
                        .unwrap_or(ServingStatus::Unknown);
                    if status == ServingStatus::Serving {
                        return Ok(());
                    }
                    format!("reported {}", status.as_str_name())
                }
                Ok(Err(status))
                    if matches!(status.code(), Code::NotFound | Code::Unimplemented) =>
                {
                    return Err(cannot_connect(format!(
                        "{INFERENCE_SERVICE} is unavailable through the standard gRPC health API: {status}"
                    )));
                }
                Ok(Err(status)) => format!("health check failed: {status}"),
                Err(_) => "health check exceeded the startup deadline".to_string(),
            };
            let now = Instant::now();
            if now >= deadline {
                return Err(cannot_connect(format!(
                    "{INFERENCE_SERVICE} did not become SERVING before the startup deadline: {last}"
                )));
            }
            sleep_until((now + retry_interval).min(deadline)).await;
        }
    }

    pub(crate) async fn prefill(
        &self,
        request: PrefillRequest,
    ) -> Result<PrefillOutcome, DynamoError> {
        let mut client = pb::inference_client::InferenceClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        let mut stream = client
            .generate_stream(build_request(request)?)
            .await
            .map(tonic_v14::Response::into_inner)
            .map_err(|status| stream_status_to_dynamo("GenerateStream", status))?;
        let mut outcome = OutcomeBuilder::default();
        while let Some(response) = stream
            .message()
            .await
            .map_err(|status| stream_status_to_dynamo("GenerateStream", status))?
        {
            outcome.push(response)?;
        }
        outcome.finish()
    }

    pub(crate) fn health_channel(&self) -> Channel {
        self.pool.next_channel()
    }

    /// vLLM does not notice a dropped response stream, so a cancelled prefill
    /// has to be aborted by id.
    pub(crate) async fn abort(&self, rid: &str) {
        let mut client = pb::control_client::ControlClient::new(self.pool.next_channel());
        let call = client.abort(pb::AbortRequest {
            request_ids: vec![rid.to_string()],
        });
        match tokio::time::timeout(ABORT_RPC_DEADLINE, call).await {
            Ok(Ok(_)) => {}
            Ok(Err(status)) => {
                let error = status_to_dynamo("Abort", status);
                tracing::warn!(rid, %error, "vLLM Abort failed");
            }
            Err(_) => tracing::warn!(rid, "vLLM Abort timed out"),
        }
    }
}

/// Follows vLLM's overall gRPC health (`""`) until vLLM stops serving, and
/// returns why.
pub(crate) async fn watch_health(channel: Channel, open_timeout: Duration) -> String {
    let request = HealthCheckRequest {
        service: String::new(),
    };
    let mut client = HealthClient::new(channel);
    let mut stream = match tokio::time::timeout(open_timeout, client.watch(request)).await {
        Ok(Ok(response)) => response.into_inner(),
        Ok(Err(status)) => return format!("health watch failed: {status}"),
        Err(_) => return format!("health watch did not open in {open_timeout:?}"),
    };
    loop {
        match stream.message().await {
            Ok(Some(response)) => {
                let status =
                    ServingStatus::try_from(response.status).unwrap_or(ServingStatus::Unknown);
                if status != ServingStatus::Serving {
                    return format!("health reported {}", status.as_str_name());
                }
            }
            Ok(None) => return "health stream ended".to_string(),
            Err(status) => return format!("health stream failed: {status}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::is_kind;

    fn response(
        token_ids: Vec<u32>,
        finish: Option<pb::FinishInfo>,
        prompt_tokens: Option<u32>,
    ) -> pb::GenerateResponse {
        pb::GenerateResponse {
            prompt_info: prompt_tokens.map(|num_prompt_tokens| pb::PromptInfo {
                num_prompt_tokens,
                ..Default::default()
            }),
            outputs: Some(pb::SequenceOutput {
                token_ids,
                finish_info: finish,
                ..Default::default()
            }),
        }
    }

    #[test]
    fn outcome_collects_first_token_and_cached_count() {
        let mut builder = OutcomeBuilder::default();
        builder.push(response(vec![], None, Some(10))).unwrap();
        builder
            .push(response(
                vec![42],
                Some(pb::FinishInfo {
                    num_output_tokens: 1,
                    finish_reason: pb::finish_info::FinishReason::Length as i32,
                    num_cached_tokens: Some(8),
                    ..Default::default()
                }),
                None,
            ))
            .unwrap();
        assert_eq!(
            builder.finish().unwrap(),
            PrefillOutcome {
                first_token: Some(42),
                stop_token: None,
                prompt_tokens: Some(10),
                cached_tokens: Some(8),
            }
        );
    }

    #[test]
    fn outcome_reports_the_eos_that_ended_an_empty_reply() {
        let mut builder = OutcomeBuilder::default();
        builder
            .push(response(
                vec![],
                Some(pb::FinishInfo {
                    finish_reason: pb::finish_info::FinishReason::Stop as i32,
                    stop_reason: Some(pb::finish_info::StopReason::EosTokenId(1)),
                    ..Default::default()
                }),
                None,
            ))
            .unwrap();
        let outcome = builder.finish().unwrap();
        assert_eq!(outcome.first_token, None);
        assert_eq!(outcome.stop_token, Some(1));
    }

    #[test]
    fn unfinished_and_aborted_streams_are_errors() {
        let mut builder = OutcomeBuilder::default();
        builder.push(response(vec![5], None, None)).unwrap();
        let unfinished = builder.finish().unwrap_err();
        assert!(is_kind(&unfinished, BackendError::StreamIncomplete));

        let mut builder = OutcomeBuilder::default();
        let aborted = builder
            .push(response(
                vec![],
                Some(pb::FinishInfo {
                    finish_reason: pb::finish_info::FinishReason::Aborted as i32,
                    ..Default::default()
                }),
                None,
            ))
            .unwrap_err();
        assert!(is_kind(&aborted, BackendError::Cancelled));
    }
}
