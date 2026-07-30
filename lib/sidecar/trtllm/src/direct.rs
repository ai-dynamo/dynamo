// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-side direct-gRPC dispatch for TensorRT-LLM's `TrtllmService`.
//!
//! [`GrpcDispatch`] fills `PushRouter`'s transport seam
//! ([`dynamo_runtime::pipeline::StreamingDispatch`]) so the frontend dispatches
//! the final hop straight to each stock TensorRT-LLM container's gRPC server —
//! keeping the router's instance selection, occupancy, fault detection, and
//! migration, and swapping only the transport below the seam. It reuses the
//! sidecar's request/response translation ([`build_generate_request`] /
//! [`ResponseState`]) and gRPC client ([`TrtllmClient`]) verbatim, one channel
//! pool per discovered worker instance.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{
    DynamoError, ErrorType, LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest, usage,
};
use dynamo_runtime::{
    component::{Instance, TransportType},
    discovery::EndpointInstanceId,
    engine::AsyncEngineContext,
    pipeline::{
        AddressedRequest, AsyncEngineContextProvider, Error, ManyIn, ManyOut, ResponseStream,
        SingleIn, StreamingDispatch,
    },
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};
use dynamo_llm::discovery::{DirectDispatchProvider, LlmStreamingDispatch};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use parking_lot::Mutex;

use crate::client::TrtllmClient;
use crate::convert::{ResponseState, build_generate_request};

/// `direct_backend` name this crate registers and handles. The engine writes it
/// into the model card's `runtime_data`; the provider's
/// [`DirectDispatchProvider::backend`] returns it — single source so they can't
/// diverge.
pub const TRTLLM_BACKEND: &str = "trtllm";

fn top_level(kind: ErrorType, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(kind)
        .message(message)
        .build()
}

/// Map a raw gRPC `tonic::Status` to a **top-level** [`ErrorType`].
///
/// This is the load-bearing difference from the request-plane sidecar's
/// [`dynamo_sidecar_common::status_to_dynamo`], which produces
/// `ErrorType::Backend(..)`. `PushRouter`'s `is_inhibited` / `is_migratable`
/// match top-level variants only, so a `Backend(..)`-nested error would silently
/// disable `report_instance_down` / overload / migration on the direct hop.
fn status_to_top_level(rpc: &str, status: tonic::Status) -> DynamoError {
    let kind = match status.code() {
        // Transport faults → report_instance_down + migrate.
        tonic::Code::Unavailable => ErrorType::CannotConnect,
        tonic::Code::DeadlineExceeded => ErrorType::ConnectionTimeout,
        // Backpressure → mark_overloaded_immediate (NOT migratable), HTTP 529.
        tonic::Code::ResourceExhausted => ErrorType::ResourceExhausted,
        // Client cancel → not migratable, no down-report.
        tonic::Code::Cancelled => ErrorType::Cancelled,
        // Engine rejected the request → surfaced, no down-report.
        tonic::Code::InvalidArgument
        | tonic::Code::NotFound
        | tonic::Code::OutOfRange
        | tonic::Code::FailedPrecondition
        | tonic::Code::AlreadyExists => ErrorType::InvalidArgument,
        _ => ErrorType::Unknown,
    };
    top_level(
        kind,
        format!("{rpc}: {} ({:?})", status.message(), status.code()),
    )
}

/// A [`StreamingDispatch`] that dials TensorRT-LLM's `TrtllmService` gRPC
/// directly, one connection pool per discovered worker instance.
pub struct GrpcDispatch {
    transport: GrpcTransportConfig,
    /// Resolved model context length, used to fill a default `max_tokens` when a
    /// request omits one (mirrors the sidecar engine). Sourced from the MDC.
    context_length: Option<u32>,
    /// Per-instance gRPC channel pools, keyed by discovery `instance_id`. Built
    /// lazily from the address carried on each `AddressedRequest`.
    clients: Mutex<HashMap<u64, Arc<TrtllmClient>>>,
}

impl GrpcDispatch {
    pub fn new(transport: GrpcTransportConfig, context_length: Option<u32>) -> Self {
        Self {
            transport,
            context_length,
            clients: Mutex::new(HashMap::new()),
        }
    }

    /// Get or lazily open the gRPC channel pool for a worker instance.
    async fn client_for(
        &self,
        instance_id: u64,
        address: &str,
    ) -> Result<Arc<TrtllmClient>, DynamoError> {
        {
            let guard = self.clients.lock();
            if let Some(client) = guard.get(&instance_id) {
                return Ok(client.clone());
            }
        }
        // Connect outside the lock; a rare concurrent double-connect is resolved
        // by `or_insert` keeping the first pool (the loser's pool is dropped).
        let endpoint = GrpcEndpoint::parse(address, "direct gRPC endpoint")?;
        let client = Arc::new(
            TrtllmClient::connect(&endpoint, self.transport)
                .await
                .map_err(|e| {
                    top_level(ErrorType::CannotConnect, format!("connect {address}: {e}"))
                })?,
        );
        Ok(self
            .clients
            .lock()
            .entry(instance_id)
            .or_insert(client)
            .clone())
    }
}

#[async_trait]
impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for GrpcDispatch {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<PreprocessedRequest>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (addressed, handle) = request.into_parts();
        let ctx: Arc<dyn AsyncEngineContext> = handle.context();
        let (req, address, instance) = addressed.into_parts();
        let request_id = ctx.id().to_string();
        // Reject a missing instance rather than collapsing to pool key 0, which
        // would bind unrelated addresses together.
        let instance = instance.ok_or_else(|| {
            top_level(
                ErrorType::InvalidArgument,
                "direct gRPC dispatch requires a resolved worker instance",
            )
        })?;
        // Shed a non-gRPC worker registered under this direct model.
        // `CannotConnect` is inhibited, so report_instance_down drops it.
        if !matches!(instance.transport, TransportType::Grpc(_)) {
            tracing::warn!(
                instance_id = instance.instance_id,
                transport = ?instance.transport,
                "non-gRPC worker registered under a direct-gRPC model; shedding \
                 (do not mix --direct and request-plane workers for one model)"
            );
            return Err(top_level(
                ErrorType::CannotConnect,
                "worker is not a direct-gRPC instance",
            )
            .into());
        }
        let instance_id = instance.instance_id;

        let client = self.client_for(instance_id, &address).await?;
        let proto = build_generate_request(&req, &request_id, self.context_length)?;
        let mut stream = client
            .generate_raw(proto)
            .await
            .map_err(|status| status_to_top_level("Generate", status))?;

        tracing::debug!(
            model = %req.model,
            input_tokens = req.token_ids.len(),
            request_id = %request_id,
            "direct gRPC dispatch"
        );

        let mut state = ResponseState::new(&req);
        let stream_ctx = ctx.clone();
        // On cancel, ask TensorRT-LLM to abort early (as the sidecar engine does);
        // dropping the stream alone only sends gRPC CANCELLED.
        let abort_client = client;
        let mapped = async_stream::stream! {
            loop {
                tokio::select! {
                    biased;
                    _ = stream_ctx.stopped() => {
                        if let Err(error) = abort_client.abort(request_id).await {
                            tracing::debug!(%error, "TensorRT-LLM Abort RPC failed on cancel");
                        }
                        yield Annotated::from_data(
                            LLMEngineOutput::cancelled()
                                .with_usage(usage(state.prompt_tokens(), state.completion_tokens())),
                        );
                        break;
                    }
                    message = stream.message() => match message {
                        Ok(Some(response)) => match state.convert(response) {
                            Ok(Some(output)) => {
                                let terminal = output.finish_reason.is_some();
                                yield Annotated::from_data(output);
                                if terminal {
                                    break;
                                }
                            }
                            Ok(None) => {}
                            // Engine errors stay Backend-nested: not transport
                            // faults, so they must not report_instance_down.
                            Err(error) => {
                                yield Annotated::from_err(error);
                                break;
                            }
                        },
                        Ok(None) => {
                            yield Annotated::from_err(top_level(
                                ErrorType::Disconnected,
                                "Generate ended before a terminal response",
                            ));
                            break;
                        }
                        Err(status) => {
                            yield Annotated::from_err(status_to_top_level("Generate", status));
                            break;
                        }
                    }
                }
            }
        };
        Ok(ResponseStream::new(Box::pin(mapped), ctx))
    }

    async fn generate_bidirectional(
        &self,
        _instance: Instance,
        _address: String,
        _input: ManyIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        Err(anyhow::anyhow!(
            "TensorRT-LLM direct gRPC dispatch does not support bidirectional streaming"
        ))
    }

    async fn on_instance_removed(&self, id: &EndpointInstanceId) {
        if self.clients.lock().remove(&id.instance_id).is_some() {
            tracing::debug!(
                instance_id = id.instance_id,
                "evicted direct-gRPC channel pool for removed worker"
            );
        }
    }
}

/// Composition-root provider that builds a [`GrpcDispatch`] for models whose
/// worker advertises `runtime_data["direct_backend"] = "trtllm"`. Register it
/// with `dynamo_llm::discovery::register_direct_dispatch_provider` in the
/// frontend composition root before serving.
pub struct TrtllmDirectDispatchProvider {
    transport: GrpcTransportConfig,
    /// One shared [`GrpcDispatch`] per model, keyed by `card.name()`. The router
    /// rebuilds its dispatch on every worker-set change, but the per-endpoint
    /// instance-removal watcher is first-wins and lives for the whole runtime —
    /// so a fresh dispatch per rebuild would never receive `on_instance_removed`
    /// and its channel pools would leak. Reusing one dispatch keeps the watcher
    /// wired to the live pool, mirroring the request plane's per-DRT shared state.
    dispatches: Mutex<HashMap<String, Arc<GrpcDispatch>>>,
}

impl TrtllmDirectDispatchProvider {
    pub fn new() -> Self {
        Self {
            transport: GrpcTransportConfig::default(),
            dispatches: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for TrtllmDirectDispatchProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DirectDispatchProvider for TrtllmDirectDispatchProvider {
    fn backend(&self) -> &str {
        TRTLLM_BACKEND
    }

    async fn build(&self, card: &ModelDeploymentCard) -> anyhow::Result<LlmStreamingDispatch> {
        let dispatch: LlmStreamingDispatch = self
            .dispatches
            .lock()
            .entry(card.name().to_string())
            .or_insert_with(|| {
                Arc::new(GrpcDispatch::new(
                    self.transport,
                    card.runtime_config.context_length,
                ))
            })
            .clone();
        Ok(dispatch)
    }
}

#[cfg(test)]
mod tests {
    use super::status_to_top_level;
    use dynamo_backend_common::ErrorType;

    #[test]
    fn status_maps_to_top_level_error_types() {
        // These MUST be top-level variants (not Backend(..)) or PushRouter's
        // report_instance_down / overload / migration silently stop firing.
        for (code, expected) in [
            (tonic::Code::Unavailable, ErrorType::CannotConnect),
            (tonic::Code::DeadlineExceeded, ErrorType::ConnectionTimeout),
            (tonic::Code::ResourceExhausted, ErrorType::ResourceExhausted),
            (tonic::Code::Cancelled, ErrorType::Cancelled),
            (tonic::Code::InvalidArgument, ErrorType::InvalidArgument),
            (tonic::Code::Internal, ErrorType::Unknown),
        ] {
            let err = status_to_top_level("Generate", tonic::Status::new(code, "x"));
            assert_eq!(err.error_type(), expected, "code {code:?}");
        }
    }
}
