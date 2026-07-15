// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `VllmSidecarEngine` — a [`LLMEngine`] that drives an out-of-process vLLM
//! engine over vLLM's gRPC API.
//!
//! # Minimal configuration
//!
//! The sidecar takes a single engine-specific input: the gRPC endpoint.
//! Model identity, parallelism, KV block sizing, and context length are
//! discovered through `GetServerInfo` / `GetModelInfo`. The Dynamo topology
//! role is configured explicitly because it is not engine metadata.
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
    PreprocessedRequest, WorkerConfig, usage,
};
use futures::stream::BoxStream;
use tokio::sync::OnceCell;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use crate::args::{Args, TransportConfig, normalize_endpoint};
use crate::client::{self, Health, Pool};
use crate::discovery::{
    BootstrapIdentity, bootstrap_discover, build_engine_config, component_for_mode,
    lora_from_proto, nonempty, validate_discovery,
};
use crate::proto as pb;
use crate::request::{build_generate_request, finish_output};
use crate::wire::{GenerateEvent, validate_generate_response};

/// A Dynamo backend that proxies inference to vLLM's gRPC server.
pub struct VllmSidecarEngine {
    /// Normalised gRPC endpoint (e.g. `http://127.0.0.1:50051`).
    endpoint: String,
    /// Connect / readiness tunables.
    transport: TransportConfig,
    /// Role discovered at bootstrap; drives `generate()` dispatch and is
    /// re-validated against the live engine in `start()`.
    disaggregation_mode: DisaggregationMode,
    /// WorkerConfig-critical identity captured during bootstrap discovery.
    bootstrap_identity: Option<BootstrapIdentity>,
    /// Connection pool, set once in `start()`. Streaming calls round-robin
    /// across it; control RPCs use its stable connection.
    pool: OnceCell<Pool>,
    /// Primary model name accepted by the gRPC server.
    served_model_name: OnceCell<String>,
    /// Cancels in-flight `generate()` streams on `cleanup()`.
    cancel: CancellationToken,
}

impl VllmSidecarEngine {
    /// Direct constructor. The public entry point is [`from_args`](Self::from_args);
    /// this exists for programmatic / test construction.
    pub(crate) fn new(
        endpoint: impl Into<String>,
        transport: TransportConfig,
        disaggregation_mode: DisaggregationMode,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            transport,
            disaggregation_mode,
            bootstrap_identity: None,
            pool: OnceCell::new(),
            served_model_name: OnceCell::new(),
            cancel: CancellationToken::new(),
        }
    }

    fn with_bootstrap_identity(mut self, identity: BootstrapIdentity) -> Self {
        self.bootstrap_identity = Some(identity);
        self
    }

    pub(crate) async fn drain_engine(&self) -> Result<bool, DynamoError> {
        let Some(mut client) = self.pool.get().map(Pool::control_client) else {
            return Ok(true);
        };
        let response = tokio::time::timeout(
            self.transport.connect_timeout,
            client.drain(pb::DrainRequest {}),
        )
        .await
        .map_err(|_| {
            client::engine_shutdown(format!(
                "Drain RPC timed out after {:?}",
                self.transport.connect_timeout
            ))
        })?
        .map_err(|status| client::status_to_dynamo("Drain", status))?
        .into_inner();
        match pb::DrainState::try_from(response.state).unwrap_or(pb::DrainState::Unspecified) {
            pb::DrainState::Complete => Ok(true),
            pb::DrainState::InProgress => Ok(false),
            state => Err(client::engine_shutdown(format!(
                "Drain returned invalid state {state:?}: {}",
                response.message
            ))),
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

        let endpoint = normalize_endpoint(&args.grpc_endpoint).map_err(client::invalid_arg)?;
        let transport = args.transport();

        let discovery = bootstrap_discover(&endpoint, &transport)?;
        validate_discovery(&discovery)?;
        let disaggregation_mode = args.disaggregation_mode;

        let served_model_name = (!discovery.model.served_model_name.is_empty())
            .then(|| discovery.model.served_model_name.clone());

        tracing::info!(
            %endpoint,
            role = ?disaggregation_mode,
            model = %discovery.model.model_id,
            "vllm sidecar bootstrapped engine discovery"
        );

        let config = WorkerConfig {
            namespace: args.namespace,
            component: component_for_mode(disaggregation_mode).to_string(),
            endpoint: args.endpoint,
            endpoint_types: args.endpoint_types,
            custom_jinja_template: args.custom_jinja_template,
            disaggregation_mode,
            model_name: discovery.model.model_id.clone(),
            served_model_name,
            reasoning_parser: nonempty(&discovery.model.reasoning_parser),
            tool_call_parser: nonempty(&discovery.model.tool_call_parser),
            ..Default::default()
        };

        let identity = BootstrapIdentity::from_discovery(&discovery);
        let engine =
            Self::new(endpoint, transport, disaggregation_mode).with_bootstrap_identity(identity);
        Ok((engine, config))
    }

    /// Poll canonical gRPC health until the Generate service is serving.
    /// Transient RPC errors are treated as "not ready yet" and retried — the
    /// engine may still be loading the model when we reconnect.
    async fn await_ready(&self, client: &mut Health, deadline: Instant) -> Result<(), DynamoError> {
        let request = || tonic_health::pb::HealthCheckRequest {
            service: "vllm.Generate".to_string(),
        };

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(client::engine_shutdown(format!(
                    "engine did not reach READY within {:?}",
                    self.transport.deadline
                )));
            }
            let outcome = tokio::time::timeout(
                remaining.min(self.transport.connect_timeout),
                client.check(request()),
            )
            .await;
            let retry_msg = match outcome {
                Ok(Ok(resp)) => {
                    let state = tonic_health::pb::health_check_response::ServingStatus::try_from(
                        resp.into_inner().status,
                    )
                    .unwrap_or(tonic_health::pb::health_check_response::ServingStatus::Unknown);
                    match state {
                        tonic_health::pb::health_check_response::ServingStatus::Serving => {
                            return Ok(());
                        }
                        other => format!("engine not ready (state {other:?})"),
                    }
                }
                Ok(Err(status)) => format!("Health RPC failed: {}", status.message()),
                Err(_) => "Health RPC timed out".to_string(),
            };

            if Instant::now() >= deadline {
                return Err(client::engine_shutdown(format!(
                    "engine did not reach READY within {:?}: {retry_msg}",
                    self.transport.deadline
                )));
            }
            tokio::time::sleep(
                self.transport
                    .poll_interval
                    .min(deadline.saturating_duration_since(Instant::now())),
            )
            .await;
        }
    }
}

#[async_trait]
impl LLMEngine for VllmSidecarEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.pool.initialized() {
            return Err(client::engine_shutdown("vllm sidecar already started"));
        }

        let startup_deadline = Instant::now() + self.transport.deadline;
        let pool = tokio::time::timeout(
            self.transport.deadline,
            Pool::connect(&self.endpoint, &self.transport, self.transport.connections),
        )
        .await
        .map_err(|_| client::engine_shutdown("connection pool startup timed out"))??;
        let mut health = pool.health_client();
        self.await_ready(&mut health, startup_deadline).await?;
        let mut control = pool.control_client();
        let discovery = client::discover(
            &mut control,
            startup_deadline.saturating_duration_since(Instant::now()),
        )
        .await?;

        validate_discovery(&discovery)?;
        if let Some(identity) = &self.bootstrap_identity {
            identity.validate(&discovery)?;
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

        Ok(Box::pin(async_stream::stream! {
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
                res = client.generate_stream(grpc_req) => Some(res),
            };
            let Some(open) = open else {
                yield Ok(LLMEngineOutput::cancelled().with_usage(usage(prompt_len, 0)));
                return;
            };
            let mut stream = match open {
                Ok(resp) => resp.into_inner(),
                Err(status) => {
                    yield Err(client::status_to_dynamo("GenerateStream", status));
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
                                let response = match validate_generate_response(resp, is_prefill) {
                                    Ok(response) => response,
                                    Err(error) => {
                                        yield Err(error);
                                        break;
                                    }
                                };
                                if let Some(value) = response.prompt_tokens
                                    && value != 0
                                {
                                    prompt_tokens = value;
                                }
                                let mut terminal = false;
                                for event in response.events {
                                    match event {
                                        GenerateEvent::Token { token_ids, logprobs } => {
                                            generated += token_ids.len() as u32;
                                            yield Ok(LLMEngineOutput {
                                                token_ids,
                                                log_probs: logprobs,
                                                ..Default::default()
                                            });
                                        }
                                        GenerateEvent::Finished(reason) => {
                                            yield Ok(finish_output(
                                                reason, prompt_tokens, generated, None,
                                            ));
                                            terminal = true;
                                        }
                                        GenerateEvent::PrefillReady(disagg) => {
                                            yield Ok(finish_output(
                                                pb::finish_info::FinishReason::Stop,
                                                prompt_tokens,
                                                generated,
                                                Some(disagg),
                                            ));
                                            terminal = true;
                                        }
                                    }
                                }
                                if terminal {
                                    break;
                                }
                            }
                            Ok(None) => {
                                let err = client::status_to_dynamo(
                                    "GenerateStream",
                                    tonic::Status::internal(
                                    "engine closed the Generate stream before a terminal event",
                                    ),
                                );
                                yield Err(err);
                                break;
                            }
                            Err(status) => {
                                yield Err(client::status_to_dynamo("GenerateStream", status));
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
            request_ids: vec![ctx.id().to_string()],
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
            .map(Pool::control_client)
            .ok_or_else(|| client::engine_shutdown("load_lora called before start"))?;
        let requested = adapter.clone();
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
        let loaded = lora_from_proto(response)?;
        if loaded != requested {
            return Err(client::engine_shutdown(
                "LoadLora returned an adapter identity different from the request",
            ));
        }
        Ok(loaded)
    }

    async fn unload_lora(&self, name: &str) -> Result<LoraAdapter, DynamoError> {
        let mut client = self
            .pool
            .get()
            .map(Pool::control_client)
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
        let adapter = lora_from_proto(response)?;
        if adapter.name != name {
            return Err(client::engine_shutdown(
                "UnloadLora returned an adapter name different from the request",
            ));
        }
        Ok(adapter)
    }

    async fn list_loras(&self) -> Result<Vec<LoraAdapter>, DynamoError> {
        let mut client = self
            .pool
            .get()
            .map(Pool::control_client)
            .ok_or_else(|| client::engine_shutdown("list_loras called before start"))?;
        let response = client
            .list_loras(pb::ListLorasRequest {})
            .await
            .map_err(|status| client::status_to_dynamo("ListLoras", status))?
            .into_inner();
        let mut names = std::collections::HashSet::new();
        let mut ids = std::collections::HashSet::new();
        let mut paths = std::collections::HashSet::new();
        let mut adapters = Vec::with_capacity(response.adapters.len());
        for adapter in response.adapters {
            let adapter = lora_from_proto(adapter)?;
            if !names.insert(adapter.name.clone())
                || !ids.insert(adapter.id)
                || !paths.insert(adapter.path.clone())
            {
                return Err(client::engine_shutdown(
                    "ListLoras returned duplicate or conflicting adapter identities",
                ));
            }
            adapters.push(adapter);
        }
        Ok(adapters)
    }

    async fn begin_drain(&self) -> Result<Option<bool>, DynamoError> {
        self.drain_engine().await.map(Some)
    }

    async fn is_quiescent(&self) -> Result<Option<bool>, DynamoError> {
        self.drain_engine().await.map(Some)
    }

    async fn watch(&self) -> Result<(), DynamoError> {
        let Some(pool) = self.pool.get() else {
            return std::future::pending::<Result<(), DynamoError>>().await;
        };

        let mut client = pool.health_client();
        let mut interval = tokio::time::interval(self.transport.poll_interval);
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            interval.tick().await;
            match tokio::time::timeout(
                self.transport.connect_timeout,
                client.check(tonic_health::pb::HealthCheckRequest {
                    service: "vllm.Generate".to_string(),
                }),
            )
            .await
            {
                Ok(Ok(resp)) => {
                    let state = tonic_health::pb::health_check_response::ServingStatus::try_from(
                        resp.into_inner().status,
                    )
                    .unwrap_or(tonic_health::pb::health_check_response::ServingStatus::Unknown);
                    match state {
                        tonic_health::pb::health_check_response::ServingStatus::Serving => {}
                        other => {
                            return Err(client::engine_shutdown(format!(
                                "engine health state {other:?} after startup"
                            )));
                        }
                    }
                }
                Ok(Err(status)) => {
                    return Err(client::engine_shutdown(format!(
                        "vLLM gRPC Health RPC failed after startup: {} ({:?})",
                        status.message(),
                        status.code(),
                    )));
                }
                Err(_) => {
                    return Err(client::engine_shutdown(format!(
                        "vLLM gRPC Health RPC timed out after {:?}",
                        self.transport.connect_timeout,
                    )));
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
            .get_kv_event_sources(pb::GetKvEventSourcesRequest {})
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
                    dp_rank: s.data_parallel_rank.unwrap_or_default(),
                    image_token_id: None,
                })
            })
            .collect();
        Ok(sources)
    }

    async fn health_check_payload(&self) -> Result<Option<serde_json::Value>, DynamoError> {
        if self.disaggregation_mode.is_decode() {
            return Ok(None);
        }
        // Canary round-tripped through generate() by the runtime's
        // HealthCheckManager: a single greedy token.
        let mut payload = serde_json::json!({
            "token_ids": [1],
            "stop_conditions": {"max_tokens": 1, "ignore_eos": true},
            "sampling_options": {"temperature": 0.0},
        });
        payload[HEALTH_CHECK_KEY] = serde_json::Value::Bool(true);
        Ok(Some(payload))
    }
}
