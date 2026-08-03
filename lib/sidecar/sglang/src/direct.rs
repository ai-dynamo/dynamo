// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-side direct-gRPC dispatch for SGLang's native `SglangService`.
//!
//! [`GrpcDispatch`] fills `PushRouter`'s transport seam
//! ([`dynamo_runtime::pipeline::StreamingDispatch`]) so the frontend dispatches
//! the final hop straight to each stock SGLang container's gRPC server — keeping
//! the router's instance selection, occupancy, fault detection, and migration,
//! and swapping only the transport below the seam. It reuses the sidecar's
//! request/response translation ([`build_generate_request`] and the
//! `protocol.rs` helpers) and its gRPC connection [`Pool`] verbatim, one pool
//! per discovered worker instance.
//!
//! Disaggregation-aware: each dispatch is built for one role (aggregated,
//! prefill, or decode) from the model card's `worker_type`, and drives
//! [`build_generate_request`] with that [`DisaggregationMode`]. A prefill
//! dispatch also carries the discovery-resolved bootstrap host/port as a
//! fallback; in practice the frontend's `PrefillRouter` stamps
//! `request.bootstrap_info` upfront, which `resolve_disaggregated_params` prefers.

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
use parking_lot::Mutex;
use tokio::time::Instant;

use crate::args::{TransportConfig, normalize_endpoint};
use crate::client::{self, Pool};
use crate::proto as pb;
use crate::protocol::{
    build_generate_request, engine_data_from_meta, extract_logprobs, meta_u32, output_ids_to_u32,
    terminal_from_meta,
};

/// `direct_backend` name this crate registers and handles. The engine writes it
/// into the model card's `runtime_data`; the provider's
/// [`DirectDispatchProvider::backend`] returns it — single source so they can't
/// diverge.
pub const SGLANG_BACKEND: &str = "sglang";

fn top_level(kind: ErrorType, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(kind)
        .message(message)
        .build()
}

/// Map a raw gRPC `tonic::Status` to a **top-level** [`ErrorType`].
///
/// This is the load-bearing difference from the request-plane sidecar's
/// [`crate::client::status_to_dynamo`], which produces `ErrorType::Backend(..)`.
/// `PushRouter`'s `is_inhibited` / `is_migratable` match top-level variants only,
/// so a `Backend(..)`-nested error would silently disable `report_instance_down`
/// / overload / migration on the direct hop.
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

/// A [`StreamingDispatch`] that dials SGLang's `SglangService` gRPC directly, one
/// connection [`Pool`] per discovered worker instance.
pub struct GrpcDispatch {
    transport: TransportConfig,
    /// Disaggregation role this dispatch drives (`build_generate_request` switches
    /// the KV-transfer handoff on it).
    mode: DisaggregationMode,
    /// Prefill bootstrap host/port from discovery — a fallback for a prefill
    /// dispatch when a request arrives without `bootstrap_info`. `None` for
    /// decode / aggregated dispatches.
    bootstrap_host: Option<String>,
    bootstrap_port: Option<u16>,
    /// Per-instance gRPC connection pools, keyed by discovery `instance_id`.
    /// Built lazily from the address carried on each `AddressedRequest`.
    clients: Mutex<HashMap<u64, Arc<Pool>>>,
}

impl GrpcDispatch {
    pub fn new(
        transport: TransportConfig,
        mode: DisaggregationMode,
        bootstrap_host: Option<String>,
        bootstrap_port: Option<u16>,
    ) -> Self {
        Self {
            transport,
            mode,
            bootstrap_host,
            bootstrap_port,
            clients: Mutex::new(HashMap::new()),
        }
    }

    /// Get or lazily open the gRPC connection pool for a worker instance.
    async fn client_for(&self, instance_id: u64, address: &str) -> Result<Arc<Pool>, DynamoError> {
        {
            let guard = self.clients.lock();
            if let Some(pool) = guard.get(&instance_id) {
                return Ok(pool.clone());
            }
        }
        // Connect outside the lock; a rare concurrent double-connect is resolved
        // by `or_insert` keeping the first pool (the loser's pool is dropped).
        let normalized =
            normalize_endpoint(address).map_err(|e| top_level(ErrorType::InvalidArgument, e))?;
        let deadline = Instant::now() + self.transport.connect_timeout;
        let pool = Arc::new(
            Pool::connect(
                &normalized,
                &self.transport,
                self.transport.connections,
                deadline,
            )
            .await
            .map_err(|e| top_level(ErrorType::CannotConnect, format!("connect {address}: {e}")))?,
        );
        Ok(self
            .clients
            .lock()
            .entry(instance_id)
            .or_insert(pool)
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

        let pool = self.client_for(instance_id, &address).await?;
        let mut grpc_client = pool.stream_client();

        let prompt_tokens = req.token_ids.len() as u32;
        let return_tokens_as_ids = req
            .output_options
            .return_tokens_as_token_ids
            .unwrap_or(false);
        // Drive the request for this dispatch's role; `resolve_disaggregated_params`
        // prefers the request's own `bootstrap_info` / `prefill_result` and only
        // falls back to the dispatch's discovery-resolved host/port.
        let proto = build_generate_request(
            &req,
            &request_id,
            self.mode,
            self.bootstrap_host.as_deref(),
            self.bootstrap_port,
        )?;

        let mut stream = grpc_client
            .generate(proto)
            .await
            .map_err(|status| status_to_top_level("Generate", status))?
            .into_inner();

        tracing::debug!(
            model = %req.model,
            input_tokens = req.token_ids.len(),
            request_id = %request_id,
            "direct gRPC dispatch"
        );

        let stream_ctx = ctx.clone();
        // On cancel, ask SGLang to abort early (as the sidecar engine does);
        // dropping the stream alone only sends gRPC CANCELLED.
        let abort_pool = pool.clone();
        let abort_id = request_id.clone();
        let abort_timeout = self.transport.connect_timeout;
        let mapped = async_stream::stream! {
            let mut generated = 0_u32;
            let mut observed_prompt_tokens = prompt_tokens;
            let mut logprob_offset = 0_usize;
            // SGLang's native gRPC streams `output_ids` CUMULATIVELY (the full
            // sequence-so-far each chunk), so track how many tokens we've already
            // emitted and forward only the tail. Without this the client sees the
            // whole prefix repeated on every chunk and the completion-token count
            // balloons past the requested `max_tokens`.
            let mut token_offset = 0_usize;
            loop {
                tokio::select! {
                    biased;
                    _ = stream_ctx.stopped() => {
                        let mut control = abort_pool.control_client();
                        let request = pb::AbortRequest { rid: abort_id.clone(), abort_all: false };
                        if let Err(error) = client::abort(&mut control, request, abort_timeout).await {
                            tracing::debug!(%error, "SGLang Abort RPC failed on cancel");
                        }
                        yield Annotated::from_data(
                            LLMEngineOutput::cancelled()
                                .with_usage(usage(observed_prompt_tokens, generated)),
                        );
                        break;
                    }
                    message = stream.message() => {
                        let response = match message {
                            Ok(Some(response)) => response,
                            Ok(None) => {
                                yield Annotated::from_err(top_level(
                                    ErrorType::Disconnected,
                                    "Generate ended before a terminal response",
                                ));
                                break;
                            }
                            // Transport-level status: map to a top-level error so
                            // the router can report_instance_down / migrate.
                            Err(status) => {
                                yield Annotated::from_err(status_to_top_level("Generate", status));
                                break;
                            }
                        };

                        if let Some(value) = meta_u32(&response.meta_info, "prompt_tokens") {
                            observed_prompt_tokens = value;
                        }
                        // Cumulative → delta: slice off the tokens we have not
                        // forwarded yet. `saturating_sub`-style guard in case a
                        // chunk ever carries fewer ids than the last (it should
                        // not for a monotonic stream).
                        let cumulative = match output_ids_to_u32(&response.output_ids) {
                            Ok(ids) => ids,
                            Err(error) => {
                                yield Annotated::from_err(error);
                                break;
                            }
                        };
                        let token_ids: Vec<u32> = if token_offset < cumulative.len() {
                            cumulative[token_offset..].to_vec()
                        } else {
                            Vec::new()
                        };
                        token_offset = cumulative.len();
                        let (log_probs, top_logprobs, next_offset) = match extract_logprobs(
                            &response.meta_info,
                            logprob_offset,
                            return_tokens_as_ids,
                        ) {
                            Ok(values) => values,
                            Err(error) => {
                                yield Annotated::from_err(error);
                                break;
                            }
                        };
                        logprob_offset = next_offset;

                        generated = generated.saturating_add(token_ids.len() as u32);
                        if response.finished {
                            let mut terminal = match terminal_from_meta(
                                &response.meta_info,
                                observed_prompt_tokens,
                                generated,
                            ) {
                                Ok(terminal) => terminal,
                                Err(error) => {
                                    yield Annotated::from_err(error);
                                    break;
                                }
                            };
                            let engine_data = match engine_data_from_meta(&response.meta_info, true) {
                                Ok(engine_data) => engine_data,
                                Err(error) => {
                                    yield Annotated::from_err(error);
                                    break;
                                }
                            };
                            terminal.token_ids = token_ids;
                            terminal.log_probs = log_probs;
                            terminal.top_logprobs = top_logprobs;
                            terminal.engine_data = engine_data;
                            yield Annotated::from_data(terminal);
                            break;
                        }

                        if !token_ids.is_empty() {
                            let engine_data = match engine_data_from_meta(&response.meta_info, false) {
                                Ok(engine_data) => engine_data,
                                Err(error) => {
                                    yield Annotated::from_err(error);
                                    break;
                                }
                            };
                            yield Annotated::from_data(LLMEngineOutput {
                                token_ids,
                                log_probs,
                                top_logprobs,
                                engine_data,
                                ..Default::default()
                            });
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
            "SGLang direct gRPC dispatch does not support bidirectional streaming"
        ))
    }

    async fn on_instance_removed(&self, id: &EndpointInstanceId) {
        if self.clients.lock().remove(&id.instance_id).is_some() {
            tracing::debug!(
                instance_id = id.instance_id,
                "evicted direct-gRPC connection pool for removed worker"
            );
        }
    }
}

/// Per-(model, role) cache of direct dispatches, keyed by
/// `(card.name(), card.worker_type)`.
type DispatchCache = Mutex<HashMap<(String, Option<WorkerType>), Arc<GrpcDispatch>>>;

/// Composition-root provider that builds a [`GrpcDispatch`] for models whose
/// worker advertises `runtime_data["direct_backend"] = "sglang"`. Register it
/// with `dynamo_llm::discovery::register_direct_dispatch_provider` in the
/// frontend composition root before serving.
pub struct SglangDirectDispatchProvider {
    transport: TransportConfig,
    /// One shared [`GrpcDispatch`] per (model, role), keyed by
    /// `(card.name(), card.worker_type)`. The router rebuilds its dispatch on
    /// every worker-set change, but the per-endpoint instance-removal watcher is
    /// first-wins and lives for the whole runtime — so a fresh dispatch per
    /// rebuild would never receive `on_instance_removed` and its connection pools
    /// would leak. Reusing one dispatch keeps the watcher wired to the live pool,
    /// mirroring the request plane's per-DRT shared state. The role is part of the
    /// key because a prefill card and a decode card share a model name but need
    /// different [`DisaggregationMode`]s (and bootstrap endpoints).
    dispatches: DispatchCache,
}

impl SglangDirectDispatchProvider {
    pub fn new() -> Self {
        Self {
            transport: TransportConfig::default(),
            dispatches: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for SglangDirectDispatchProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DirectDispatchProvider for SglangDirectDispatchProvider {
    fn backend(&self) -> &str {
        SGLANG_BACKEND
    }

    async fn build(&self, card: &ModelDeploymentCard) -> anyhow::Result<LlmStreamingDispatch> {
        let (mode, bootstrap_host, bootstrap_port) = dispatch_role(card);
        let transport = self.transport.clone();
        let dispatch: LlmStreamingDispatch = self
            .dispatches
            .lock()
            .entry((card.name().to_string(), card.worker_type))
            .or_insert_with(|| {
                Arc::new(GrpcDispatch::new(
                    transport,
                    mode,
                    bootstrap_host,
                    bootstrap_port,
                ))
            })
            .clone();
        Ok(dispatch)
    }
}

/// Resolve the [`DisaggregationMode`] and prefill bootstrap fallback for a card's
/// role. A prefill card carries its discovery-published bootstrap host/port on
/// `runtime_config.disaggregated_endpoint`; decode / aggregated cards carry none.
fn dispatch_role(card: &ModelDeploymentCard) -> (DisaggregationMode, Option<String>, Option<u16>) {
    match card.worker_type {
        Some(WorkerType::Prefill) => {
            let endpoint = card.runtime_config.disaggregated_endpoint.as_ref();
            let host = endpoint.and_then(|e| e.bootstrap_host.clone());
            let port = endpoint.and_then(|e| e.bootstrap_port);
            (DisaggregationMode::Prefill, host, port)
        }
        Some(WorkerType::Decode) => (DisaggregationMode::Decode, None, None),
        _ => (DisaggregationMode::Aggregated, None, None),
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
