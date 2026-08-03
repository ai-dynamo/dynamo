// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-side direct-gRPC dispatch for vLLM's native `Generate` service.
//!
//! [`GrpcDispatch`] fills `PushRouter`'s transport seam
//! ([`dynamo_runtime::pipeline::StreamingDispatch`]) so the frontend dispatches
//! the final hop straight to each stock vLLM (`vllm-rs`) container's gRPC server
//! — keeping the router's instance selection, occupancy, fault detection, and
//! migration, and swapping only the transport below the seam. It reuses the
//! sidecar's request/response translation ([`build_generate_request`] /
//! [`ResponseState`]) and gRPC client ([`VllmClient`]) verbatim, one channel
//! pool per discovered worker instance.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{
    DisaggregationMode, DynamoError, ErrorType, LLMEngineOutput, LLMEngineOutputExt,
    PreprocessedRequest, usage,
};
use dynamo_llm::discovery::{DirectDispatchProvider, LlmStreamingDispatch};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::worker_type::WorkerType;
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
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use parking_lot::Mutex;

use crate::client::VllmClient;
use crate::convert::{ResponseState, build_generate_request};

/// `direct_backend` name this crate registers and handles. The engine writes it
/// into the model card's `runtime_data`; the provider's
/// [`DirectDispatchProvider::backend`] returns it — single source so they can't
/// diverge.
pub const VLLM_BACKEND: &str = "vllm";

fn top_level(kind: ErrorType, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(kind)
        .message(message)
        .build()
}

/// Terminal output emitted when a request is cancelled, carrying the usage
/// accumulated so far (mirrors the request-plane sidecar engine).
fn cancelled(state: &ResponseState) -> LLMEngineOutput {
    LLMEngineOutput::cancelled().with_usage(usage(
        state.prompt_tokens(),
        state.reported_completion_tokens(),
    ))
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

/// A [`StreamingDispatch`] that dials vLLM's `Generate` gRPC service directly,
/// one connection pool per discovered worker instance.
pub struct GrpcDispatch {
    transport: GrpcTransportConfig,
    /// Disaggregation role this dispatch drives. `build_kv_parameters` switches on
    /// it: a prefill dispatch stamps `do_remote_decode`, a decode dispatch reads
    /// the request's `prefill_result` KV payload, aggregated passes neither.
    mode: DisaggregationMode,
    /// Per-instance gRPC channel pools, keyed by discovery `instance_id`. Built
    /// lazily from the address carried on each `AddressedRequest`.
    clients: Mutex<HashMap<u64, Arc<VllmClient>>>,
}

impl GrpcDispatch {
    pub fn new(transport: GrpcTransportConfig, mode: DisaggregationMode) -> Self {
        Self {
            transport,
            mode,
            clients: Mutex::new(HashMap::new()),
        }
    }

    /// Get or lazily open the gRPC channel pool for a worker instance.
    async fn client_for(
        &self,
        instance_id: u64,
        address: &str,
    ) -> Result<Arc<VllmClient>, DynamoError> {
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
            VllmClient::connect(&endpoint, self.transport)
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

        // Capture request facts for tracing before `build_generate_request`
        // consumes the request; `ResponseState` borrows it just before the move.
        let model = req.model.clone();
        let input_tokens = req.token_ids.len();
        let mut state = ResponseState::new(&req, self.mode);
        let proto = build_generate_request(req, request_id.clone(), self.mode)?;
        let mut stream = client
            .generate_stream_raw(proto)
            .await
            .map_err(|status| status_to_top_level("GenerateStream", status))?;

        tracing::debug!(
            model = %model,
            input_tokens,
            request_id = %request_id,
            "direct gRPC dispatch"
        );

        let stream_ctx = ctx.clone();
        // vLLM's gRPC has no Abort RPC; on cancel, dropping the stream sends gRPC
        // CANCELLED to the server. We just emit the terminal cancelled chunk with
        // accumulated usage (mirrors the request-plane sidecar engine).
        let mapped = async_stream::stream! {
            loop {
                tokio::select! {
                    biased;
                    _ = stream_ctx.stopped() => {
                        yield Annotated::from_data(cancelled(&state));
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
                                "GenerateStream ended before a terminal response",
                            ));
                            break;
                        }
                        Err(status) => {
                            yield Annotated::from_err(status_to_top_level("GenerateStream", status));
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
            "vLLM direct gRPC dispatch does not support bidirectional streaming"
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

/// Per-(model, role) cache of direct dispatches, keyed by
/// `(card.name(), card.worker_type)`.
type DispatchCache = Mutex<HashMap<(String, Option<WorkerType>), Arc<GrpcDispatch>>>;

/// Composition-root provider that builds a [`GrpcDispatch`] for models whose
/// worker advertises `runtime_data["direct_backend"] = "vllm"`. Register it with
/// `dynamo_llm::discovery::register_direct_dispatch_provider` in the frontend
/// composition root before serving.
pub struct VllmDirectDispatchProvider {
    transport: GrpcTransportConfig,
    /// One shared [`GrpcDispatch`] per (model, role), keyed by
    /// `(card.name(), card.worker_type)`. The router rebuilds its dispatch on
    /// every worker-set change, but the per-endpoint instance-removal watcher is
    /// first-wins and lives for the whole runtime — so a fresh dispatch per
    /// rebuild would never receive `on_instance_removed` and its channel pools
    /// would leak. Reusing one dispatch keeps the watcher wired to the live pool,
    /// mirroring the request plane's per-DRT shared state. The role is part of the
    /// key because a prefill card and a decode card share a model name but drive
    /// different [`DisaggregationMode`]s.
    dispatches: DispatchCache,
}

impl VllmDirectDispatchProvider {
    pub fn new() -> Self {
        Self {
            transport: GrpcTransportConfig::default(),
            dispatches: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for VllmDirectDispatchProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DirectDispatchProvider for VllmDirectDispatchProvider {
    fn backend(&self) -> &str {
        VLLM_BACKEND
    }

    async fn build(&self, card: &ModelDeploymentCard) -> anyhow::Result<LlmStreamingDispatch> {
        let mode = dispatch_mode(card);
        let transport = self.transport;
        let dispatch: LlmStreamingDispatch = self
            .dispatches
            .lock()
            .entry((card.name().to_string(), card.worker_type))
            .or_insert_with(|| Arc::new(GrpcDispatch::new(transport, mode)))
            .clone();
        Ok(dispatch)
    }
}

/// Resolve the [`DisaggregationMode`] a card's role drives. vLLM transfers KV via
/// `prefill_result` (NixlConnector), so no bootstrap endpoint is needed — only
/// the mode differs between prefill and decode dispatches.
fn dispatch_mode(card: &ModelDeploymentCard) -> DisaggregationMode {
    match card.worker_type {
        Some(WorkerType::Prefill) => DisaggregationMode::Prefill,
        Some(WorkerType::Decode) => DisaggregationMode::Decode,
        _ => DisaggregationMode::Aggregated,
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
            let err = status_to_top_level("GenerateStream", tonic::Status::new(code, "x"));
            assert_eq!(err.error_type(), expected, "code {code:?}");
        }
    }
}
