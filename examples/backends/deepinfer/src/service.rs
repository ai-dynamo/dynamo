// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_trtllm_sidecar::proto as pb;
use futures::Stream;
use prost_types::value::Kind;
use serde_json::{Value, json};
use std::{pin::Pin, sync::Arc};
use tonic::{Request, Response, Status};

use crate::bridge::Bridge;

type RpcStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

#[derive(Clone)]
pub struct Service {
    bridge: Arc<Bridge>,
    model: String,
    context_length: u32,
}

impl Service {
    pub fn new(bridge: Arc<Bridge>, model: String, context_length: u32) -> Self {
        Self {
            bridge,
            model,
            context_length,
        }
    }

    #[allow(clippy::result_large_err)]
    fn prepare(&self, request: &pb::GenerateRequest) -> Result<(Value, u32), Status> {
        if request.request_id.is_empty() || request.model.is_empty() {
            return Err(Status::invalid_argument(
                "request_id and model are required",
            ));
        }
        let Some(pb::generate_request::Input::TokenIds(input)) = &request.input else {
            return Err(Status::invalid_argument(
                "this server requires token_ids input",
            ));
        };
        let prompt_tokens = u32::try_from(input.ids.len())
            .map_err(|_| Status::invalid_argument("prompt too long"))?;
        if prompt_tokens == 0
            || prompt_tokens >= self.context_length
            || input
                .ids
                .iter()
                .any(|id| *id >= self.bridge.ready.vocab_size || *id > i32::MAX as u32)
        {
            return Err(Status::invalid_argument(
                "empty, oversized, or out-of-vocabulary prompt",
            ));
        }
        if !request.media.is_empty()
            || !request.lora_name.is_empty()
            || request.kv.is_some()
            || request.guided.is_some()
        {
            return Err(Status::unimplemented(
                "this aggregated text example does not implement media, LoRA, KV handoff, or guided decoding",
            ));
        }
        if request
            .extra
            .as_ref()
            .and_then(|v| v.fields.get("context_only"))
            .is_some_and(|v| matches!(v.kind, Some(Kind::BoolValue(true))))
        {
            return Err(Status::unimplemented(
                "context-only execution is not implemented",
            ));
        }
        let sampling = request.sampling.unwrap_or_default();
        if sampling.num_sequences.is_some_and(|n| n != 1)
            || sampling.min_p.is_some_and(|n| n != 0.0)
            || sampling.frequency_penalty.is_some_and(|n| n != 0.0)
            || sampling.presence_penalty.is_some_and(|n| n != 0.0)
            || sampling.repetition_penalty.is_some_and(|n| n != 1.0)
        {
            return Err(Status::unimplemented(
                "multiple sequences and non-default sampling penalties are unsupported",
            ));
        }
        let (temperature, top_p, top_k) = (
            sampling.temperature.unwrap_or(1.0),
            sampling.top_p.unwrap_or(1.0),
            sampling.top_k.unwrap_or(-1),
        );
        if !(temperature.is_finite()
            && temperature >= 0.0
            && top_p.is_finite()
            && 0.0 < top_p
            && top_p <= 1.0)
            || (top_k != -1 && top_k < 1)
        {
            return Err(Status::invalid_argument(
                "invalid temperature, top_p, or top_k",
            ));
        }
        let stopping = request.stopping.clone().unwrap_or_default();
        if stopping.min_tokens.is_some_and(|n| n > 0)
            || stopping.include_stop_in_output == Some(true)
        {
            return Err(Status::unimplemented(
                "min_tokens and include_stop_in_output are unsupported",
            ));
        }
        let max_tokens = stopping
            .max_tokens
            .unwrap_or(self.context_length - prompt_tokens);
        if max_tokens == 0 || max_tokens > self.context_length - prompt_tokens {
            return Err(Status::invalid_argument(
                "max_tokens must fit the remaining context",
            ));
        }
        let mut stop_ids = Vec::new();
        for condition in stopping.conditions {
            match condition.condition {
                Some(pb::stop_condition::Condition::StopTokenId(id)) => stop_ids.push(id),
                _ => {
                    return Err(Status::unimplemented(
                        "only token-ID stop conditions are supported",
                    ));
                }
            }
        }
        if !stopping.ignore_eos.unwrap_or(false) {
            stop_ids.push(self.bridge.ready.eos_token_id);
        }
        if let Some(response) = &request.response {
            let valid_candidates = response.output_candidates.as_ref().is_none_or(|c| {
                matches!(
                    c.selection,
                    Some(pb::candidate_token_selection::Selection::TopN(1))
                )
            });
            if response.return_prompt_logprobs == Some(true) || !valid_candidates {
                return Err(Status::unimplemented(
                    "only selected-token logprobs (top_n=1) are supported",
                ));
            }
        }
        Ok((
            json!({"type":"generate", "token_ids":input.ids, "stop_token_ids":stop_ids,
            "sampling":{"temperature":temperature, "top_p":top_p, "top_k":top_k,
                "max_tokens":max_tokens, "ignore_eos":stopping.ignore_eos.unwrap_or(false), "seq_seed":sampling.seed}}),
            prompt_tokens,
        ))
    }
}

#[tonic::async_trait]
impl pb::inference_server::Inference for Service {
    type GenerateStream = RpcStream<pb::GenerateResponse>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let request = request.into_inner();
        let (command, prompt_tokens) = self.prepare(&request)?;
        let mut session = self.bridge.submit(&request.request_id, command)?;
        let request_id = request.request_id;
        tracing::info!(%request_id, prompt_tokens, "OpenEngine Generate admitted");
        Ok(Response::new(Box::pin(async_stream::try_stream! {
            let mut completion_tokens = 0;
            loop {
                let output = tokio::select! {
                    biased;
                    _ = session.cancelled.cancelled() => None,
                    output = session.receiver.recv() => Some(output),
                };
                let Some(output) = output else {
                    let failure = session.failure.lock().clone();
                    if let Some(message) = failure { Err(Status::unavailable(message))?; }
                    yield terminal(&request_id, pb::FinishReason::Cancelled, None, prompt_tokens, completion_tokens);
                    break;
                };
                let output = output.ok_or_else(|| Status::unavailable("engine output closed before terminal"))?;
                if let Some(message) = output.error { Err(Status::invalid_argument(message))?; }
                if let Some(token_id) = output.token_id {
                    completion_tokens += 1;
                    yield pb::GenerateResponse {
                        request_id: request_id.clone(), usage: None,
                        event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
                            output_index: Some(0), text: String::new(),
                            tokens: vec![pb::TokenInfo { token_id, logprob: output.logprob, ..Default::default() }],
                        })),
                    };
                }
                if output.finished {
                    let reason = match output.finish_reason.as_str() {
                        "stop" => pb::FinishReason::Stop,
                        "length" => pb::FinishReason::Length,
                        "abort" => pb::FinishReason::Cancelled,
                        other => Err(Status::internal(format!("DeepInfer terminal error: {other}")))?,
                    };
                    yield terminal(&request_id, reason, output.stop_token_id, prompt_tokens, completion_tokens);
                    break;
                }
            }
        })))
    }
}

fn terminal(
    request_id: &str,
    reason: pb::FinishReason,
    stop_id: Option<u32>,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: request_id.to_owned(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: reason as i32,
                message: String::new(),
                stop_match: stop_id.map(|id| pb::StopMatch {
                    r#match: Some(pb::stop_match::Match::StopTokenId(id)),
                }),
            },
        )),
        usage: Some(pb::Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cached_prompt_tokens: Some(0),
            reasoning_tokens: None,
        }),
    }
}

#[tonic::async_trait]
impl pb::control_server::Control for Service {
    async fn get_server_info(
        &self,
        _: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        Ok(Response::new(pb::ServerInfo {
            engine_name: "deepinfer".into(),
            engine_version: "example".into(),
            engine_role: pb::EngineRole::Aggregated as i32,
            instance_id: "deepinfer-openengine".into(),
            supported_models: vec![self.model.clone()],
            schema_revision: 1,
            minimum_client_revision: 1,
            schema_release: "b5f2bd93721f7b888d3e2440679e0ae7012939d1".into(),
            ..Default::default()
        }))
    }

    async fn get_model_info(
        &self,
        _: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        Ok(Response::new(pb::ModelInfo {
            model_id: self.model.clone(),
            served_model_name: self.model.clone(),
            max_context_length: Some(self.context_length),
            max_output_tokens: Some(self.context_length),
            supports_text_input: Some(false),
            supports_token_ids_input: Some(true),
            reasoning_parser: "deepseek_v41".into(),
            tool_call_parser: "deepseek_v41".into(),
            supports_lora: Some(false),
            supports_multimodal: Some(false),
            generation: Some(pb::GenerationCapabilities {
                max_num_sequences: Some(1),
                ..Default::default()
            }),
            ..Default::default()
        }))
    }

    async fn get_load(
        &self,
        _: Request<pb::GetLoadRequest>,
    ) -> Result<Response<pb::LoadInfo>, Status> {
        Ok(Response::new(pb::LoadInfo {
            instance_id: "deepinfer-openengine".into(),
            running_requests: Some(self.bridge.active()),
            ..Default::default()
        }))
    }

    async fn health(
        &self,
        request: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        if request.into_inner().include_inference_probe {
            return unsupported();
        }
        Ok(Response::new(pb::HealthResponse {
            state: if self.bridge.alive() {
                pb::HealthState::Ready
            } else {
                pb::HealthState::NotReady
            } as i32,
            checks: vec![],
        }))
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        let aborted = match request.into_inner().target {
            Some(pb::abort_request::Target::RequestId(id)) => self.bridge.abort(Some(&id)),
            Some(pb::abort_request::Target::AllRequests(_)) => self.bridge.abort(None),
            Some(pb::abort_request::Target::KvSession(_)) => return unsupported(),
            None => return Err(Status::invalid_argument("Abort requires a target")),
        };
        Ok(Response::new(pb::AbortResponse {
            status: if aborted {
                pb::AbortStatus::Aborted
            } else {
                pb::AbortStatus::AlreadyFinished
            } as i32,
            message: String::new(),
        }))
    }

    async fn load_lora(
        &self,
        _: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        unsupported()
    }
    async fn unload_lora(
        &self,
        _: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        unsupported()
    }
    async fn list_loras(
        &self,
        _: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        unsupported()
    }
    async fn get_kv_event_sources(
        &self,
        _: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        unsupported()
    }
    type SubscribeKvEventsStream = RpcStream<pb::SubscribeKvEventsResponse>;
    async fn subscribe_kv_events(
        &self,
        _: Request<pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        unsupported()
    }
}

#[allow(clippy::result_large_err)]
fn unsupported<T>() -> Result<Response<T>, Status> {
    Err(Status::unimplemented(
        "not implemented by the DeepInfer example",
    ))
}
