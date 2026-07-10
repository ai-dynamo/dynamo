// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `VllmSidecarEngine` — a [`LLMEngine`] that drives an out-of-process vLLM
//! engine over the OpenEngine v1 gRPC contract.
//!
//! # Endpoint-only configuration
//!
//! The sidecar takes a single engine-specific input: the OpenEngine endpoint.
//! Everything else — model identity, disaggregation role, parallelism, KV
//! block sizing, and context length — is **discovered** from the engine over
//! `GetEngineInfo` / `GetModelInfo`. There is no `--model` or
//! `--disaggregation-mode`.
//!
//! # Two-phase discovery
//!
//! [`run`](dynamo_backend_common::run) requires the [`WorkerConfig`]
//! (namespace / component / disaggregation role) to be built *synchronously*
//! in `from_args`, before the async [`start`](LLMEngine::start) runs. So:
//!
//! 1. **`from_args` (bootstrap):** on a throwaway current-thread runtime,
//!    connect and call discovery to learn the role + model, derive the
//!    component, and build `WorkerConfig`. The bootstrap channel is **dropped**
//!    — it is bound to that temporary runtime's reactor and cannot be reused.
//! 2. **`start` (worker runtime):** reconnect on the worker's runtime, poll
//!    `Health` until `READY` (the engine may still be loading), re-run
//!    discovery, validate the role is unchanged, and return the full
//!    [`EngineConfig`].

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, DisaggregationMode, DynamoError, EngineConfig, GenerateContext,
    HEALTH_CHECK_KEY, KvEventSource, LLMEngine, LLMEngineOutput, LLMEngineOutputExt, LoraAdapter,
    LlmRegistration, MultimodalData, PreprocessedRequest, WorkerConfig, usage,
};
use futures::stream::BoxStream;
use tokio::sync::{OnceCell, watch};
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use crate::args::{Args, TransportConfig, normalize_endpoint};
use crate::client::{self, Client, Discovery, Pool};
use crate::proto as pb;

/// A Dynamo backend that proxies inference to a vLLM OpenEngine server.
pub struct VllmSidecarEngine {
    /// Normalised gRPC endpoint (e.g. `http://127.0.0.1:50051`).
    endpoint: String,
    /// Connect / readiness tunables.
    transport: TransportConfig,
    /// Role discovered at bootstrap; drives `generate()` dispatch and is
    /// re-validated against the live engine in `start()`.
    disaggregation_mode: DisaggregationMode,
    /// Connection pool, set once in `start()`. Streaming calls round-robin
    /// across it; control RPCs use its stable connection.
    pool: OnceCell<Pool>,
    /// Primary model name accepted by the OpenEngine server.
    served_model_name: OnceCell<String>,
    /// Cancels in-flight `generate()` streams on `cleanup()`.
    cancel: CancellationToken,
    /// Set when the out-of-process OpenEngine server can no longer serve.
    fatal: watch::Sender<Option<String>>,
}

impl VllmSidecarEngine {
    /// Direct constructor. The public entry point is [`from_args`](Self::from_args);
    /// this exists for programmatic / test construction.
    pub(crate) fn new(
        endpoint: impl Into<String>,
        transport: TransportConfig,
        disaggregation_mode: DisaggregationMode,
    ) -> Self {
        let (fatal, _) = watch::channel(None);
        Self {
            endpoint: endpoint.into(),
            transport,
            disaggregation_mode,
            pool: OnceCell::new(),
            served_model_name: OnceCell::new(),
            cancel: CancellationToken::new(),
            fatal,
        }
    }

    /// Parse CLI args, bootstrap-discover the engine role + model, and build the
    /// pair `run()` consumes.
    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = match argv {
            Some(a) => <Args as clap::Parser>::try_parse_from(a),
            None => <Args as clap::Parser>::try_parse(),
        }
        .map_err(|e| client::invalid_arg(e.to_string()))?;

        let endpoint = normalize_endpoint(&args.openengine_endpoint);
        let transport = args.transport();

        let discovery = bootstrap_discover(&endpoint, &transport)?;
        let role = engine_role(&discovery);
        let disaggregation_mode = role_to_mode(role);

        let served_model_name = (!discovery.model.served_model_name.is_empty())
            .then(|| discovery.model.served_model_name.clone());

        tracing::info!(
            %endpoint,
            role = ?role,
            model = %discovery.model.model_id,
            "vllm sidecar bootstrapped engine discovery"
        );

        let config = WorkerConfig {
            namespace: args.namespace,
            component: component_for_role(role).to_string(),
            endpoint: args.endpoint,
            endpoint_types: args.endpoint_types,
            custom_jinja_template: args.custom_jinja_template,
            disaggregation_mode,
            model_name: discovery.model.model_id.clone(),
            served_model_name,
            ..Default::default()
        };

        let engine = Self::new(endpoint, transport, disaggregation_mode);
        Ok((engine, config))
    }

    /// Poll `Health` until the engine reports `READY` or the deadline elapses.
    /// Transient RPC errors are treated as "not ready yet" and retried — the
    /// engine may still be loading the model when we reconnect.
    async fn await_ready(&self, client: &mut Client) -> Result<(), DynamoError> {
        let deadline = Instant::now() + self.transport.deadline;
        let request = || pb::HealthRequest {
            include_inference_probe: false,
            model: String::new(),
            role: pb::EngineRole::Unspecified as i32,
        };

        loop {
            let outcome = client.health(request()).await;
            let retry_msg = match outcome {
                Ok(resp) => {
                    let state = pb::HealthState::try_from(resp.into_inner().state)
                        .unwrap_or(pb::HealthState::Unspecified);
                    match state {
                        pb::HealthState::Ready => return Ok(()),
                        pb::HealthState::Draining => {
                            return Err(client::engine_shutdown("engine is draining"));
                        }
                        other => format!("engine not ready (state {other:?})"),
                    }
                }
                Err(status) => format!("Health RPC failed: {}", status.message()),
            };

            if Instant::now() >= deadline {
                return Err(client::engine_shutdown(format!(
                    "engine did not reach READY within {:?}: {retry_msg}",
                    self.transport.deadline
                )));
            }
            tokio::time::sleep(self.transport.poll_interval).await;
        }
    }
}

#[async_trait]
impl LLMEngine for VllmSidecarEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.pool.initialized() {
            return Err(client::engine_shutdown("vllm sidecar already started"));
        }

        let pool =
            Pool::connect(&self.endpoint, &self.transport, self.transport.connections).await?;
        let mut control = pool.control_client();
        self.await_ready(&mut control).await?;
        let discovery = client::discover(&mut control).await?;

        // The role is the engine's authoritative `kv_role`; it must not have
        // flipped between bootstrap and now (that would mean the worker
        // registered under the wrong component).
        let observed = role_to_mode(engine_role(&discovery));
        if observed != self.disaggregation_mode {
            return Err(client::invalid_arg(format!(
                "engine role changed since bootstrap: registered as {:?} but engine now reports {:?}",
                self.disaggregation_mode, observed
            )));
        }

        let pool_size = pool.len();
        self.pool
            .set(pool)
            .map_err(|_| client::engine_shutdown("vllm sidecar already started"))?;

        let config = build_engine_config(&discovery);
        let served_model_name = config
            .served_model_name
            .clone()
            .unwrap_or_else(|| config.model.clone());
        self.served_model_name
            .set(served_model_name)
            .map_err(|_| client::engine_shutdown("vllm sidecar model already initialized"))?;
        tracing::info!(
            model = %config.model,
            context_length = ?config.llm.as_ref().and_then(|llm| llm.context_length),
            kv_cache_block_size = ?config.llm.as_ref().and_then(|llm| llm.kv_cache_block_size),
            connections = pool_size,
            "vllm sidecar started"
        );
        Ok(config)
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let mut client = self
            .pool
            .get()
            .map(Pool::stream_client)
            .ok_or_else(|| client::engine_shutdown("generate called before start"))?;

        // Decode workers must receive the prefill peer's handoff (forwarded by
        // the frontend's PrefillRouter). Fail loudly if it is missing.
        if self.disaggregation_mode.is_decode() && request.prefill_result.is_none() {
            return Err(client::invalid_arg(
                "vllm sidecar decode worker received request with no prefill_result; \
                 expected the frontend to forward disaggregated_params from a prefill peer",
            ));
        }

        let is_prefill = self.disaggregation_mode.is_prefill();
        let prompt_len = request.token_ids.len() as u32;
        let mut grpc_req = build_generate_request(&request, ctx.id(), is_prefill)?;
        grpc_req.model = self
            .served_model_name
            .get()
            .ok_or_else(|| client::engine_shutdown("generate called before model discovery"))?
            .clone();
        let cancel = self.cancel.clone();
        let fatal = self.fatal.clone();
        let current_failure = {
            let fatal_rx = self.fatal.subscribe();
            fatal_rx.borrow().as_ref().cloned()
        };

        Ok(Box::pin(async_stream::stream! {
            if let Some(reason) = current_failure {
                yield Err(client::engine_shutdown(reason));
                return;
            }

            // Pre-flight: honour an already-cancelled request before opening the
            // RPC, so a stopped context never streams a single token.
            if ctx.is_stopped() || cancel.is_cancelled() {
                yield Ok(LLMEngineOutput::cancelled().with_usage(usage(prompt_len, 0)));
                return;
            }

            let open = tokio::select! {
                biased;
                _ = ctx.stopped() => None,
                _ = cancel.cancelled() => None,
                res = client.generate(grpc_req) => Some(res),
            };
            let Some(open) = open else {
                yield Ok(LLMEngineOutput::cancelled().with_usage(usage(prompt_len, 0)));
                return;
            };
            let mut stream = match open {
                Ok(resp) => resp.into_inner(),
                Err(status) => {
                    if is_fatal_generate_status(status.code()) {
                        let err = fatal_generate_error(status);
                        signal_engine_failure(&fatal, err.message().to_string());
                        yield Err(err);
                    } else {
                        yield Err(client::status_to_dynamo("Generate", status));
                    }
                    return;
                }
            };

            // `generated` is our own running count of forwarded token IDs; it,
            // not the engine's self-report, defines the terminal
            // completion_tokens so `sum(chunk.token_ids) == completion_tokens`
            // holds by construction.
            let mut generated: u32 = 0;
            let mut prompt_tokens = prompt_len;

            loop {
                tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        yield Ok(LLMEngineOutput::cancelled()
                            .with_usage(usage(prompt_tokens, generated)));
                        break;
                    }
                    _ = cancel.cancelled() => {
                        yield Ok(LLMEngineOutput::cancelled()
                            .with_usage(usage(prompt_tokens, generated)));
                        break;
                    }
                    msg = stream.message() => {
                        match msg {
                            Ok(Some(resp)) => {
                                if let Some(u) = resp.usage.as_ref()
                                    && u.prompt_tokens != 0
                                {
                                    prompt_tokens = u.prompt_tokens;
                                }
                                match resp.event {
                                    Some(pb::generate_response::Event::Token(t)) => {
                                        // Prefill workers exist only to populate the KV cache and
                                        // hand it off; the decode peer regenerates from the
                                        // transferred KV, so the prefill's sampled token is
                                        // discarded. Suppress token outputs here so the FIRST
                                        // output the PrefillRouter observes is the terminal one
                                        // carrying `disaggregated_params` — the router reads
                                        // `first_output` only.
                                        if is_prefill || t.token_ids.is_empty() {
                                            continue;
                                        }
                                        generated += t.token_ids.len() as u32;
                                        yield Ok(LLMEngineOutput {
                                            token_ids: t.token_ids,
                                            ..Default::default()
                                        });
                                    }
                                    Some(pb::generate_response::Event::Finished(f)) => {
                                        let reason = pb::FinishReason::try_from(f.reason)
                                            .unwrap_or(pb::FinishReason::Unspecified);
                                        yield Ok(finish_output(
                                            reason, prompt_tokens, generated, None,
                                        ));
                                        break;
                                    }
                                    Some(pb::generate_response::Event::PrefillReady(p)) => {
                                        let disagg = p.kv_session.map(kv_session_to_disagg_json);
                                        yield Ok(finish_output(
                                            pb::FinishReason::Stop,
                                            prompt_tokens,
                                            generated,
                                            disagg,
                                        ));
                                        break;
                                    }
                                    Some(pb::generate_response::Event::Error(e)) => {
                                        yield Err(client::engine_error_to_dynamo(&e));
                                        break;
                                    }
                                    None => continue,
                                }
                            }
                            Ok(None) => {
                                let err = client::engine_shutdown(
                                    "engine closed the Generate stream before a terminal event",
                                );
                                signal_engine_failure(&fatal, err.message().to_string());
                                yield Err(err);
                                break;
                            }
                            Err(status) => {
                                if is_fatal_generate_status(status.code()) {
                                    let err = fatal_generate_error(status);
                                    signal_engine_failure(&fatal, err.message().to_string());
                                    yield Err(err);
                                } else {
                                    yield Err(client::status_to_dynamo("Generate", status));
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        let Some(mut client) = self.pool.get().map(Pool::control_client) else {
            return;
        };
        let req = pb::AbortRequest {
            request_id: ctx.id().to_string(),
            kv_session: None,
            abort_all: false,
        };
        // Idempotent on the engine side (unknown id → ABORTED); failures here
        // are best-effort and swallowed.
        if let Err(status) = client.abort(req).await {
            tracing::debug!(error = %status.message(), request_id = ctx.id(), "vllm sidecar: abort RPC failed (ignored)");
        }
    }

    async fn load_lora(&self, adapter: LoraAdapter) -> Result<LoraAdapter, DynamoError> {
        let mut client = self
            .pool
            .get()
            .map(Pool::lora_client)
            .ok_or_else(|| client::engine_shutdown("load_lora called before start"))?;
        let response = client
            .load_lora(pb::LoadLoraRequest {
                adapter: Some(pb::LoraAdapter {
                    lora_id: adapter.id,
                    lora_name: adapter.name,
                    source_path: adapter.path,
                }),
            })
            .await
            .map_err(|status| client::status_to_dynamo("LoadLora", status))?
            .into_inner()
            .adapter
            .ok_or_else(|| client::engine_shutdown("LoadLora returned no adapter"))?;
        Ok(lora_from_proto(response))
    }

    async fn unload_lora(&self, name: &str) -> Result<LoraAdapter, DynamoError> {
        let mut client = self
            .pool
            .get()
            .map(Pool::lora_client)
            .ok_or_else(|| client::engine_shutdown("unload_lora called before start"))?;
        let response = client
            .unload_lora(pb::UnloadLoraRequest {
                lora_name: name.to_string(),
            })
            .await
            .map_err(|status| client::status_to_dynamo("UnloadLora", status))?
            .into_inner()
            .adapter
            .ok_or_else(|| client::engine_shutdown("UnloadLora returned no adapter"))?;
        Ok(lora_from_proto(response))
    }

    async fn list_loras(&self) -> Result<Vec<LoraAdapter>, DynamoError> {
        let mut client = self
            .pool
            .get()
            .map(Pool::lora_client)
            .ok_or_else(|| client::engine_shutdown("list_loras called before start"))?;
        let response = client
            .list_loras(pb::ListLorasRequest {})
            .await
            .map_err(|status| client::status_to_dynamo("ListLoras", status))?
            .into_inner();
        Ok(response.adapters.into_iter().map(lora_from_proto).collect())
    }

    async fn begin_drain(&self) -> Result<(), DynamoError> {
        let Some(mut client) = self.pool.get().map(Pool::control_client) else {
            return Ok(());
        };
        let deadline_ms = self.transport.deadline.as_millis().min(u32::MAX as u128) as u32;
        let req = pb::DrainRequest {
            stop_accepting_new_requests: true,
            deadline_ms,
            abort_after_deadline: false,
        };
        match client.drain(req).await {
            Ok(resp) => {
                let mut stream = resp.into_inner();
                loop {
                    match stream.message().await {
                        Ok(Some(_)) => continue,
                        Ok(None) => break,
                        Err(status) => {
                            tracing::warn!(error = %status.message(), "vllm sidecar: drain stream error (ignored)");
                            break;
                        }
                    }
                }
            }
            Err(status) => {
                tracing::warn!(error = %status.message(), "vllm sidecar: drain RPC failed (ignored)");
            }
        }
        Ok(())
    }

    async fn watch(&self) -> Result<(), DynamoError> {
        let Some(pool) = self.pool.get() else {
            return std::future::pending::<Result<(), DynamoError>>().await;
        };

        let mut fatal_rx = self.fatal.subscribe();
        let mut client = pool.control_client();
        let mut interval = tokio::time::interval(self.transport.poll_interval);
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            if let Some(reason) = fatal_rx.borrow().as_ref().cloned() {
                return Err(client::engine_shutdown(reason));
            }

            tokio::select! {
                changed = fatal_rx.changed() => {
                    if changed.is_err() {
                        return Err(client::engine_shutdown(
                            "vllm sidecar fatal watcher closed",
                        ));
                    }
                }
                _ = interval.tick() => {
                    match tokio::time::timeout(
                        self.transport.connect_timeout,
                        client.health(health_request()),
                    )
                    .await
                    {
                        Ok(Ok(resp)) => {
                            let state = pb::HealthState::try_from(resp.into_inner().state)
                                .unwrap_or(pb::HealthState::Unspecified);
                            match state {
                                pb::HealthState::Ready => {}
                                pb::HealthState::Draining => {
                                    return Err(client::engine_shutdown(
                                        "engine reported DRAINING after startup",
                                    ));
                                }
                                other => {
                                    return Err(client::engine_shutdown(format!(
                                        "engine health state {other:?} after startup"
                                    )));
                                }
                            }
                        }
                        Ok(Err(status)) => {
                            return Err(client::engine_shutdown(format!(
                                "OpenEngine Health RPC failed after startup: {} ({:?})",
                                status.message(),
                                status.code(),
                            )));
                        }
                        Err(_) => {
                            return Err(client::engine_shutdown(format!(
                                "OpenEngine Health RPC timed out after {:?}",
                                self.transport.connect_timeout,
                            )));
                        }
                    }
                }
            }
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        // Cancels in-flight `generate()` streams (they select on this token).
        // Idempotent (CancellationToken::cancel) and null-safe (no resources
        // to free on a partially-started engine; the pool's channels close when
        // it drops).
        self.cancel.cancel();
        tracing::info!("vllm sidecar: cleanup invoked");
        Ok(())
    }

    async fn kv_event_sources(&self) -> Result<Vec<KvEventSource>, DynamoError> {
        let Some(mut client) = self.pool.get().map(Pool::control_client) else {
            return Ok(Vec::new());
        };
        let resp = client
            .get_kv_event_sources(pb::GetKvEventSourcesRequest {
                data_parallel_ranks: Vec::new(),
            })
            .await
            .map_err(|s| client::status_to_dynamo("GetKvEventSources", s))?
            .into_inner();

        // Surface only ZMQ publishers; the Dynamo KV router subscribes to those
        // directly. Other transports are not yet routable.
        let sources = resp
            .sources
            .into_iter()
            .filter(|s| s.transport == "zmq")
            .filter_map(|s| {
                let e = s.endpoint_addr?;
                let proto = if e.protocol.is_empty() {
                    "tcp"
                } else {
                    &e.protocol
                };
                Some(KvEventSource::Zmq {
                    endpoint: format!("{proto}://{}:{}", e.host, e.port),
                    topic: s.topic,
                    dp_rank: s.data_parallel_rank,
                })
            })
            .collect();
        Ok(sources)
    }

    async fn health_check_payload(&self) -> Result<Option<serde_json::Value>, DynamoError> {
        // Canary round-tripped through generate() by the runtime's
        // HealthCheckManager: a single greedy token.
        let mut payload = serde_json::json!({
            "token_ids": [1],
            "stop_conditions": {"max_tokens": 1, "ignore_eos": true},
            "sampling_options": {"temperature": 0.0},
        });
        // Decode-mode generate() rejects requests without prefill_result;
        // synthesize an empty handoff so the canary clears the precondition.
        if self.disaggregation_mode.is_decode() {
            payload["prefill_result"] = serde_json::json!({"disaggregated_params": {}});
        }
        payload[HEALTH_CHECK_KEY] = serde_json::Value::Bool(true);
        Ok(Some(payload))
    }
}

// ============================================================================
// Discovery → config helpers
// ============================================================================

/// Bootstrap discovery on a throwaway current-thread runtime. The connection
/// is intentionally dropped with the runtime — `start()` reconnects on the
/// worker runtime.
fn bootstrap_discover(
    endpoint: &str,
    transport: &TransportConfig,
) -> Result<Discovery, DynamoError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| client::engine_shutdown(format!("bootstrap runtime: {e}")))?;
    rt.block_on(async {
        let mut client = client::connect(endpoint, transport).await?;
        client::discover(&mut client).await
    })
}

fn engine_role(discovery: &Discovery) -> pb::EngineRole {
    pb::EngineRole::try_from(discovery.engine.role).unwrap_or(pb::EngineRole::Aggregated)
}

fn role_to_mode(role: pb::EngineRole) -> DisaggregationMode {
    match role {
        pb::EngineRole::Prefill => DisaggregationMode::Prefill,
        pb::EngineRole::Decode => DisaggregationMode::Decode,
        _ => DisaggregationMode::Aggregated,
    }
}

fn health_request() -> pb::HealthRequest {
    pb::HealthRequest {
        include_inference_probe: false,
        model: String::new(),
        role: pb::EngineRole::Unspecified as i32,
    }
}

fn signal_engine_failure(fatal: &watch::Sender<Option<String>>, reason: impl Into<String>) {
    let _ = fatal.send(Some(reason.into()));
}

fn is_fatal_generate_status(code: tonic::Code) -> bool {
    matches!(
        code,
        tonic::Code::Unknown
            | tonic::Code::Unavailable
            | tonic::Code::Internal
            | tonic::Code::DeadlineExceeded
            | tonic::Code::DataLoss
            | tonic::Code::Aborted
    )
}

fn fatal_generate_error(status: tonic::Status) -> DynamoError {
    client::engine_shutdown(format!(
        "OpenEngine Generate RPC failed: {} ({:?})",
        status.message(),
        status.code(),
    ))
}

/// Prefill workers register under the `prefill` component (targeted by the
/// frontend's PrefillRouter); aggregated and decode workers use `backend`.
fn component_for_role(role: pb::EngineRole) -> &'static str {
    match role {
        pb::EngineRole::Prefill => "prefill",
        _ => "backend",
    }
}

fn build_engine_config(discovery: &Discovery) -> EngineConfig {
    let model = &discovery.model;
    let parallelism = discovery.engine.parallelism.clone().unwrap_or_default();
    let served_model_name =
        (!model.served_model_name.is_empty()).then(|| model.served_model_name.clone());

    // Proto3 scalar `0` is "unset"; gate each optional field so we advertise
    // `None` rather than a bogus zero.
    EngineConfig {
        model: model.model_id.clone(),
        served_model_name,
        runtime_data: Default::default(),
        llm: Some(LlmRegistration {
            context_length: (model.max_context_length != 0).then_some(model.max_context_length),
            kv_cache_block_size: (model.kv_block_size != 0).then_some(model.kv_block_size),
            total_kv_blocks: (model.total_kv_blocks != 0).then_some(model.total_kv_blocks),
            max_num_seqs: (model.max_running_requests != 0).then_some(model.max_running_requests),
            max_num_batched_tokens: (model.max_batched_tokens != 0)
                .then_some(model.max_batched_tokens),
            data_parallel_size: (parallelism.data_parallel_size != 0)
                .then_some(parallelism.data_parallel_size),
            data_parallel_start_rank: (parallelism.data_parallel_start_rank != 0)
                .then_some(parallelism.data_parallel_start_rank),
            supports_lora: model.supports_lora,
            // vLLM's KV transport (NixlConnector) is internal — no
            // Dynamo-level bootstrap host/port handshake.
            bootstrap_host: None,
            bootstrap_port: None,
        }),
    }
}

fn lora_from_proto(adapter: pb::LoraAdapter) -> LoraAdapter {
    LoraAdapter {
        id: adapter.lora_id,
        name: adapter.lora_name,
        path: adapter.source_path,
    }
}

// ============================================================================
// Request building + terminal mapping
// ============================================================================

pub(crate) fn build_generate_request(
    request: &PreprocessedRequest,
    request_id: &str,
    is_prefill: bool,
) -> Result<pb::GenerateRequest, DynamoError> {
    let sampling = &request.sampling_options;
    // Prefill only needs to populate the KV cache for the prompt: cap to one
    // token regardless of the client's request.
    let max_tokens = if is_prefill {
        1
    } else {
        request.stop_conditions.max_tokens.unwrap_or(0)
    };

    let proto_sampling = pb::SamplingParams {
        temperature: sampling.temperature.unwrap_or(0.0) as f64,
        top_p: sampling.top_p.unwrap_or(0.0) as f64,
        top_k: sampling.top_k.filter(|v| *v > 0).unwrap_or(0),
        frequency_penalty: sampling.frequency_penalty.unwrap_or(0.0) as f64,
        presence_penalty: sampling.presence_penalty.unwrap_or(0.0) as f64,
        max_tokens,
        seed: sampling
            .seed
            .filter(|v| *v > 0)
            .map(|v| v as u64)
            .unwrap_or(0),
        ignore_eos: request.stop_conditions.ignore_eos.unwrap_or(false),
    };

    let mut stop = Vec::new();
    if let Some(strings) = &request.stop_conditions.stop {
        for text in strings {
            stop.push(pb::StopCondition {
                condition: Some(pb::stop_condition::Condition::StopText(text.clone())),
            });
        }
    }
    if let Some(ids) = &request.stop_conditions.stop_token_ids {
        for id in ids {
            stop.push(pb::StopCondition {
                condition: Some(pb::stop_condition::Condition::StopTokenId(*id)),
            });
        }
    }

    // Decode-role disaggregation: lift the prefill peer's handoff into a KV
    // session the engine forwards to its connector.
    let kv_session = request
        .prefill_result
        .as_ref()
        .map(|pr| disagg_json_to_kv_session(&pr.disaggregated_params, request_id));

    // Forward the KV router's forced DP-rank decision so the engine pins this
    // request to the rank that holds its prefix. Without it, vLLM internal-DP
    // load-balances and scatters the prefix across ranks, so a child block's
    // parent lands under a different rank → `parent_block_not_found` in the
    // indexer. The sidecar advertises one KV-event source per *engine-local*
    // rank (`GetKvEventSources`), so the router indexes — and decides — in the
    // engine-local rank space; the value is forwarded as-is. The in-process
    // backend reads the same `routing.dp_rank` for both prefill and decode
    // (`dynamo.common.backend.dp_rank.forced_dp_rank`); `prefill_dp_rank` is
    // never populated by the router.
    let data_parallel_rank = request.routing.as_ref().and_then(|r| r.dp_rank);

    let media = build_media(request)?;
    let lora_name = request
        .routing
        .as_ref()
        .and_then(|routing| routing.lora_name.clone())
        .unwrap_or_default();

    Ok(pb::GenerateRequest {
        request_id: request_id.to_string(),
        model: request.model.clone(),
        input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
            ids: request.token_ids.clone(),
        })),
        sampling: Some(proto_sampling),
        stop,
        stream: true,
        media,
        lora_name,
        data_parallel_rank,
        kv_session,
        metadata: Default::default(),
        // openengine.v1 additive request fields. Guided decoding and logprobs
        // are not mapped yet.
        ..Default::default()
    })
}

/// Map a media-map key (`image_url`/`video_url`/`audio_url`) to its proto
/// modality. Unknown keys fall back to `Unspecified`, which the engine treats
/// as image.
fn modality_for_key(key: &str) -> pb::Modality {
    match key {
        "image_url" => pb::Modality::Image,
        "video_url" => pb::Modality::Video,
        "audio_url" => pb::Modality::Audio,
        _ => pb::Modality::Unspecified,
    }
}

/// A URL / `data:` string becomes a `data_uri` source when it is a data URI,
/// otherwise a plain `url` source the engine fetches.
fn media_source_from_str(s: &str) -> pb::media_item::Source {
    if s.starts_with("data:") {
        pb::media_item::Source::DataUri(s.to_string())
    } else {
        pb::media_item::Source::Url(s.to_string())
    }
}

/// Build the proto `media` list from the request's multimodal map.
///
/// The sidecar runs the frontend in URL-passthrough mode (`media_decoder:
/// null`), so each item is a URL or `data:` URI the engine fetches and
/// preprocesses. A pre-decoded RDMA descriptor (`MultimodalData::Decoded`) is
/// rejected fail-closed: the sidecar has no NIXL agent to dereference it.
///
/// Items are emitted in a fixed modality order (image, then video, then audio)
/// so the i-th item aligns with the i-th placeholder marker for single-modality
/// prompts.
fn build_media(request: &PreprocessedRequest) -> Result<Vec<pb::MediaItem>, DynamoError> {
    let Some(map) = request.multi_modal_data.as_ref() else {
        return Ok(Vec::new());
    };
    let mut media = Vec::new();
    for key in ["image_url", "video_url", "audio_url"] {
        let Some(items) = map.get(key) else { continue };
        let modality = modality_for_key(key);
        for item in items {
            let source = match item {
                MultimodalData::Url(u) => media_source_from_str(u.as_str()),
                MultimodalData::RawUrl(s) => media_source_from_str(s),
                MultimodalData::Decoded(_) => {
                    return Err(client::invalid_arg(
                        "vllm-sidecar received a pre-decoded RDMA media descriptor; the \
                         sidecar has no NIXL agent to dereference it. Run the frontend in \
                         URL-passthrough mode (set the model's media_decoder to null).",
                    ));
                }
            };
            media.push(pb::MediaItem {
                modality: modality as i32,
                source: Some(source),
                mime_type: String::new(),
                uuid: String::new(),
            });
        }
    }
    Ok(media)
}

fn finish_output(
    reason: pb::FinishReason,
    prompt_tokens: u32,
    generated: u32,
    disaggregated_params: Option<serde_json::Value>,
) -> LLMEngineOutput {
    let mut out = match reason {
        pb::FinishReason::Length => LLMEngineOutput::length(),
        pb::FinishReason::Cancelled => LLMEngineOutput::cancelled(),
        pb::FinishReason::Error => {
            LLMEngineOutput::error("engine reported error finish reason".to_string())
        }
        // STOP and UNSPECIFIED both map to a normal stop terminal.
        _ => LLMEngineOutput::stop(),
    }
    .with_usage(usage(prompt_tokens, generated));
    out.disaggregated_params = disaggregated_params;
    out
}

// ============================================================================
// KV session <-> disaggregated_params encoding
// ============================================================================

/// Encode a prefill `KvSessionRef` into the JSON the frontend's PrefillRouter
/// forwards to the decode peer (round-trips with
/// [`disagg_json_to_kv_session`]).
///
/// The sidecar carries typed connector handoff metadata as native JSON.
pub(crate) fn kv_session_to_disagg_json(session: pb::KvSessionRef) -> serde_json::Value {
    let mut obj = serde_json::json!({
        "session_id": session.session_id,
        "transfer_backend": session.transfer_backend,
        "dp_rank": session.dp_rank,
    });
    if let Some(s) = session.attributes_struct.as_ref() {
        obj["attributes_struct"] = prost_struct_to_json(s);
    }
    obj
}

/// Reconstruct a `KvSessionRef` from the prefill peer's forwarded JSON.
pub(crate) fn disagg_json_to_kv_session(
    params: &serde_json::Value,
    request_id: &str,
) -> pb::KvSessionRef {
    let obj = params.as_object();
    let get_str = |key: &str| {
        obj.and_then(|o| o.get(key))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    };
    let session_id = get_str("session_id").unwrap_or_else(|| request_id.to_string());
    let transfer_backend = get_str("transfer_backend").unwrap_or_default();
    let dp_rank = obj
        .and_then(|o| o.get("dp_rank"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let attributes_struct = obj
        .and_then(|o| o.get("attributes_struct"))
        .and_then(json_to_prost_struct);
    pb::KvSessionRef {
        session_id,
        transfer_backend,
        endpoints: Vec::new(),
        dp_rank,
        attributes_struct,
    }
}

/// Convert a JSON object into a `google.protobuf.Struct`. Non-object inputs
/// yield `None`.
pub(crate) fn json_to_prost_struct(value: &serde_json::Value) -> Option<prost_types::Struct> {
    match value {
        serde_json::Value::Object(map) => Some(prost_types::Struct {
            fields: map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_prost_value(v)))
                .collect(),
        }),
        _ => None,
    }
}

fn json_to_prost_value(value: &serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(prost_types::NullValue::NullValue as i32),
        serde_json::Value::Bool(b) => Kind::BoolValue(*b),
        serde_json::Value::Number(n) => Kind::NumberValue(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => Kind::StringValue(s.clone()),
        serde_json::Value::Array(arr) => Kind::ListValue(prost_types::ListValue {
            values: arr.iter().map(json_to_prost_value).collect(),
        }),
        serde_json::Value::Object(map) => Kind::StructValue(prost_types::Struct {
            fields: map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_prost_value(v)))
                .collect(),
        }),
    };
    prost_types::Value { kind: Some(kind) }
}

/// Convert a `google.protobuf.Struct` back into a JSON object.
pub(crate) fn prost_struct_to_json(s: &prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        s.fields
            .iter()
            .map(|(k, v)| (k.clone(), prost_value_to_json(v)))
            .collect(),
    )
}

fn prost_value_to_json(value: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match &value.kind {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(Kind::NumberValue(n)) => number_to_json(*n),
        Some(Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(Kind::ListValue(l)) => {
            serde_json::Value::Array(l.values.iter().map(prost_value_to_json).collect())
        }
        Some(Kind::StructValue(s)) => prost_struct_to_json(s),
    }
}

/// `google.protobuf.Struct` numbers are IEEE-754 doubles; recover integral
/// values as JSON integers so connector params (`remote_port`, `tp_size`, …)
/// keep their integer type through the pass-through.
fn number_to_json(n: f64) -> serde_json::Value {
    if n.is_finite() && n.fract() == 0.0 && n >= i64::MIN as f64 && n <= i64::MAX as f64 {
        serde_json::Value::Number((n as i64).into())
    } else {
        serde_json::Number::from_f64(n)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null)
    }
}
