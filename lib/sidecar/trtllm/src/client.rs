// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin client for TensorRT-LLM's OpenEngine gRPC services (`openengine.v1`).

use std::sync::Arc;
use std::time::Duration;

use tokio::time::{Instant, sleep_until, timeout_at};

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, GrpcEndpoint, GrpcTransportConfig,
    connection_timeout,
};
use tonic::transport::Channel;

pub(crate) use dynamo_sidecar_common::{
    cancelled, engine_shutdown, invalid_argument, status_to_dynamo,
};

use crate::proto as pb;
use crate::proto::control_client::ControlClient;
use crate::proto::inference_client::InferenceClient;

/// Deadline for the one-shot `Control.Abort`, so a connected-but-unresponsive
/// server cannot hang cancellation. Startup is bounded by the operator's
/// `--grpc-startup-deadline-secs` instead.
const RPC_TIMEOUT: Duration = Duration::from_secs(30);
const LOAD_RPC_TIMEOUT: Duration = Duration::from_secs(2);

/// The engine's limits, resolved at startup from `--context-length` and
/// `Control.GetModelInfo`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ModelLimits {
    /// Maximum input + output tokens; backs both the registered context window
    /// and the default `max_tokens`. Absent when neither source supplied one,
    /// which leaves requests that omit `max_tokens` to be rejected.
    pub(crate) context_length: Option<u32>,
    /// Cap on generated tokens, when the engine reports one. A context window
    /// alone can imply a larger budget than the engine will accept.
    pub(crate) max_output_tokens: Option<u32>,
}

const TARGET_DP_RANK_METADATA_KEY: &str = "openengine-target-dp-rank";

#[derive(Clone)]
pub(crate) struct TrtllmClient {
    pool: Arc<GrpcChannelPool>,
}

impl TrtllmClient {
    pub(crate) async fn connect(
        endpoint: &GrpcEndpoint,
        transport: GrpcTransportConfig,
    ) -> Result<Self, DynamoError> {
        let pool = GrpcChannelPool::connect("TensorRT-LLM", endpoint, transport).await?;
        Ok(Self {
            pool: Arc::new(pool),
        })
    }

    pub(crate) fn connection_count(&self) -> usize {
        self.pool.len()
    }

    fn inference(&self) -> InferenceClient<Channel> {
        InferenceClient::new(self.pool.next_channel())
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
    }

    fn control(&self) -> ControlClient<Channel> {
        ControlClient::new(self.pool.next_channel())
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
    }

    pub(crate) async fn generate(
        &self,
        request: pb::GenerateRequest,
        target_dp_rank: Option<u32>,
    ) -> Result<tonic::Streaming<pb::GenerateResponse>, DynamoError> {
        let mut request = tonic::Request::new(request);
        if let Some(rank) = target_dp_rank {
            request.metadata_mut().insert(
                TARGET_DP_RANK_METADATA_KEY,
                tonic::metadata::MetadataValue::from(rank),
            );
        }
        self.inference()
            .generate(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("Generate", status))
    }

    /// Reads the server's advertised limits, keeping positive values.
    ///
    /// An engine started without `--max_seq_len` leaves `max_context_length`
    /// unset rather than substituting its `max_input_len` default (measured
    /// against TensorRT-LLM main at 8bbaf66bd5), so `None` here means "the
    /// server did not say", and `--context-length` supplies it instead.
    async fn get_model_info(&self, model: &str) -> Result<ModelLimits, tonic::Status> {
        let info = self
            .control()
            .get_model_info(pb::GetModelInfoRequest {
                model: model.to_string(),
            })
            .await?
            .into_inner();
        Ok(ModelLimits {
            context_length: info.max_context_length.filter(|len| *len > 0),
            max_output_tokens: info.max_output_tokens.filter(|cap| *cap > 0),
        })
    }

    /// One `GetModelInfo` call, for when `--context-length` already supplies the
    /// window and the engine is consulted only to cross-check it and to learn
    /// its output cap.
    pub(crate) async fn model_limits(&self, model: &str) -> Result<ModelLimits, DynamoError> {
        tokio::time::timeout(RPC_TIMEOUT, self.get_model_info(model))
            .await
            .map_err(|_| {
                connection_timeout(format!(
                    "GetModelInfo did not respond within {RPC_TIMEOUT:?}"
                ))
            })?
            .map_err(|status| status_to_dynamo("GetModelInfo", status))
    }

    pub(crate) async fn server_info(&self) -> Result<Option<pb::ServerInfo>, DynamoError> {
        let response = tokio::time::timeout(
            RPC_TIMEOUT,
            self.control().get_server_info(pb::GetServerInfoRequest {}),
        )
        .await
        .map_err(|_| {
            connection_timeout(format!(
                "GetServerInfo did not respond within {RPC_TIMEOUT:?}"
            ))
        })?;
        match response {
            Ok(response) => Ok(Some(response.into_inner())),
            Err(status) if status.code() == tonic::Code::Unimplemented => Ok(None),
            Err(status) => Err(status_to_dynamo("GetServerInfo", status)),
        }
    }

    pub(crate) async fn kv_event_sources(&self) -> Result<Vec<pb::KvEventSource>, DynamoError> {
        tokio::time::timeout(
            RPC_TIMEOUT,
            self.control()
                .get_kv_event_sources(pb::GetKvEventSourcesRequest {
                    data_parallel_ranks: Vec::new(),
                }),
        )
        .await
        .map_err(|_| {
            connection_timeout(format!(
                "GetKvEventSources did not respond within {RPC_TIMEOUT:?}"
            ))
        })?
        .map(tonic::Response::into_inner)
        .map(|response| response.sources)
        .or_else(|status| {
            if status.code() == tonic::Code::Unimplemented {
                Ok(Vec::new())
            } else {
                Err(status)
            }
        })
        .map_err(|status| status_to_dynamo("GetKvEventSources", status))
    }

    pub(crate) async fn get_load(
        &self,
        include_per_rank: bool,
    ) -> Result<pb::LoadInfo, DynamoError> {
        tokio::time::timeout(
            LOAD_RPC_TIMEOUT,
            self.control()
                .get_load(pb::GetLoadRequest { include_per_rank }),
        )
        .await
        .map_err(|_| {
            connection_timeout(format!(
                "GetLoad did not respond within {LOAD_RPC_TIMEOUT:?}"
            ))
        })?
        .map(tonic::Response::into_inner)
        .or_else(|status| {
            if status.code() == tonic::Code::Unimplemented {
                Ok(pb::LoadInfo::default())
            } else {
                Err(status)
            }
        })
        .map_err(|status| status_to_dynamo("GetLoad", status))
    }

    /// Polls `GetModelInfo` until the engine reports a usable context length or
    /// `deadline` passes.
    ///
    /// Used only when no `--context-length` was supplied, which makes the engine
    /// the sole source. TensorRT-LLM binds its gRPC port before the model
    /// finishes loading, so the early calls can fail outright or answer without
    /// limits; waiting is what lets a large engine finish loading. An answer
    /// that says the request itself is wrong ends the wait immediately -- no
    /// amount of waiting fixes a model the server does not serve, or a server
    /// with no Control service.
    pub(crate) async fn wait_for_model_limits(
        &self,
        model: &str,
        deadline: Instant,
        retry_interval: Duration,
    ) -> Result<ModelLimits, DynamoError> {
        let mut last = "it never answered".to_string();
        loop {
            match timeout_at(deadline, self.get_model_info(model)).await {
                Ok(Ok(limits)) if limits.context_length.is_some() => return Ok(limits),
                Ok(Ok(_)) => last = "it reported no max_context_length".to_string(),
                Ok(Err(status)) if answers_the_request_is_wrong(&status) => {
                    return Err(status_to_dynamo("GetModelInfo", status));
                }
                Ok(Err(status)) => last = format!("{}: {}", status.code(), status.message()),
                Err(_) => break,
            }
            let next_attempt = Instant::now() + retry_interval;
            if next_attempt >= deadline {
                break;
            }
            tracing::debug!(model, reason = %last, "waiting for TensorRT-LLM GetModelInfo");
            sleep_until(next_attempt).await;
        }
        Err(connection_timeout(format!(
            "TensorRT-LLM did not report a model context length before the gRPC startup \
             deadline ({last}). Raise --grpc-startup-deadline-secs if the engine is still \
             loading, or pin the window with --context-length."
        )))
    }

    pub(crate) async fn abort(&self, request_id: String) -> Result<(), DynamoError> {
        let response = tokio::time::timeout(
            RPC_TIMEOUT,
            self.control().abort(pb::AbortRequest {
                target: Some(pb::abort_request::Target::RequestId(request_id)),
            }),
        )
        .await
        .map_err(|_| connection_timeout(format!("Abort did not respond within {RPC_TIMEOUT:?}")))?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("Abort", status))?;
        // A request already gone (finished/never seen) is not an error; only an
        // unspecified status is protocol drift.
        match pb::AbortStatus::try_from(response.status) {
            Ok(pb::AbortStatus::Aborted) | Ok(pb::AbortStatus::AlreadyFinished) => Ok(()),
            _ => Err(protocol_error(format!(
                "TensorRT-LLM returned an unexpected abort status: {}",
                response.message
            ))),
        }
    }
}

/// Whether the server answered that the request itself is wrong -- a model it
/// does not serve, or no Control service at all. Waiting cannot change any of
/// these, so a startup probe stops rather than burning its whole deadline.
fn answers_the_request_is_wrong(status: &tonic::Status) -> bool {
    matches!(
        status.code(),
        tonic::Code::InvalidArgument | tonic::Code::NotFound | tonic::Code::Unimplemented
    )
}

pub(crate) fn protocol_error(message: impl Into<String>) -> DynamoError {
    dynamo_sidecar_common::protocol_error("TensorRT-LLM", message)
}

/// A generation failure the engine reported in-band via an `EngineError` event
/// (as opposed to a transport/validation gRPC status).
pub(crate) fn engine_error(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::Unknown))
        .message(message)
        .build()
}
