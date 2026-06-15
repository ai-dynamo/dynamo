// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `SglangSidecarEngine` — a [`LLMEngine`] that drives an out-of-process SGLang
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
    HEALTH_CHECK_KEY, KvEventSource, LLMEngine, LLMEngineOutput, LLMEngineOutputExt, MultimodalData,
    PreprocessedRequest, TopLogprob, WorkerConfig, usage,
};
use futures::stream::BoxStream;
use tokio::sync::OnceCell;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::args::{Args, TransportConfig, normalize_endpoint};
use crate::client::{self, Client, Discovery, Pool};
use crate::proto as pb;

/// A Dynamo backend that proxies inference to a SGLang OpenEngine server.
pub struct SglangSidecarEngine {
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
    /// Cancels in-flight `generate()` streams on `cleanup()`.
    cancel: CancellationToken,
}

impl SglangSidecarEngine {
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
            pool: OnceCell::new(),
            cancel: CancellationToken::new(),
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
            "sglang sidecar bootstrapped engine discovery"
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
            // Response parsers the engine advertises (empty when unset). Written
            // into the worker's ModelRuntimeConfig so the frontend applies
            // tool-call / reasoning parsing to this sidecar-served model.
            reasoning_parser: (!discovery.model.reasoning_parser.is_empty())
                .then(|| discovery.model.reasoning_parser.clone()),
            tool_call_parser: (!discovery.model.tool_call_parser.is_empty())
                .then(|| discovery.model.tool_call_parser.clone()),
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
impl LLMEngine for SglangSidecarEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.pool.initialized() {
            return Err(client::engine_shutdown("sglang sidecar already started"));
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
            .map_err(|_| client::engine_shutdown("sglang sidecar already started"))?;

        let config = build_engine_config(&discovery);
        tracing::info!(
            model = %config.model,
            context_length = ?config.context_length,
            kv_cache_block_size = ?config.kv_cache_block_size,
            connections = pool_size,
            "sglang sidecar started"
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

        // Decode workers must receive the prefill peer's handoff. SGLang's
        // PrefillRouter uses the Bootstrap path (router-assigned room in
        // `bootstrap_info`, sent to prefill + decode concurrently); the Completed
        // path (`prefill_result.disaggregated_params`) is also accepted.
        if self.disaggregation_mode.is_decode()
            && request.prefill_result.is_none()
            && request.bootstrap_info.is_none()
        {
            return Err(client::invalid_arg(
                "sglang sidecar decode worker received request with neither bootstrap_info \
                 nor prefill_result; expected the frontend's PrefillRouter to forward one",
            ));
        }

        let is_prefill = self.disaggregation_mode.is_prefill();
        let prompt_len = request.token_ids.len() as u32;
        let grpc_req = build_generate_request(&request, ctx.id(), is_prefill)?;
        let cancel = self.cancel.clone();

        Ok(Box::pin(async_stream::stream! {
            // Pre-flight: honour an already-cancelled request before opening the
            // RPC, so a stopped context never streams a single token.
            if ctx.is_stopped() || cancel.is_cancelled() {
                yield Ok(LLMEngineOutput::cancelled().with_usage(usage(prompt_len, 0)));
                return;
            }

            // TTFT-tail per-hop tracing: stamp request dispatch and first-token
            // receipt at the sidecar to split the outbound wrapper (everything
            // up to + including the SGLang OpenEngine server + gRPC-return) from the downstream
            // sidecar -> frontend -> client delivery path.
            let __t0 = std::time::Instant::now();
            let mut __first_token = true;
            tracing::info!(request_id = %ctx.id(), "sc.send");
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
                    yield Err(client::status_to_dynamo("Generate", status));
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
                                        if __first_token {
                                            __first_token = false;
                                            tracing::info!(
                                                request_id = %ctx.id(),
                                                elapsed_ms = __t0.elapsed().as_millis() as u64,
                                                "sc.first_token"
                                            );
                                        }
                                        generated += t.token_ids.len() as u32;
                                        // Chosen-token logprob per output token.
                                        let log_probs = (!t.logprobs.is_empty()).then(|| {
                                            t.logprobs.iter().map(|lp| lp.logprob).collect::<Vec<f64>>()
                                        });
                                        // Top-K alternatives per output token.
                                        let top_logprobs = (!t.top_logprobs.is_empty()).then(|| {
                                            t.top_logprobs
                                                .iter()
                                                .map(|tl| {
                                                    tl.entries
                                                        .iter()
                                                        .map(|lp| TopLogprob {
                                                            rank: lp.rank,
                                                            token_id: lp.token_id,
                                                            token: (!lp.token.is_empty())
                                                                .then(|| lp.token.clone()),
                                                            logprob: lp.logprob,
                                                            bytes: None,
                                                        })
                                                        .collect::<Vec<_>>()
                                                })
                                                .collect::<Vec<_>>()
                                        });
                                        yield Ok(LLMEngineOutput {
                                            token_ids: t.token_ids,
                                            log_probs,
                                            top_logprobs,
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
                                yield Err(client::engine_shutdown(
                                    "engine closed the Generate stream before a terminal event",
                                ));
                                break;
                            }
                            Err(status) => {
                                yield Err(client::status_to_dynamo("Generate", status));
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
            tracing::debug!(error = %status.message(), request_id = ctx.id(), "sglang sidecar: abort RPC failed (ignored)");
        }
    }

    async fn drain(&self) -> Result<(), DynamoError> {
        let Some(mut client) = self.pool.get().map(Pool::control_client) else {
            return Ok(());
        };
        let deadline_ms = self
            .transport
            .deadline
            .as_millis()
            .min(u32::MAX as u128) as u32;
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
                            tracing::warn!(error = %status.message(), "sglang sidecar: drain stream error (ignored)");
                            break;
                        }
                    }
                }
            }
            Err(status) => {
                tracing::warn!(error = %status.message(), "sglang sidecar: drain RPC failed (ignored)");
            }
        }
        Ok(())
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        // Cancels in-flight `generate()` streams (they select on this token).
        // Idempotent (CancellationToken::cancel) and null-safe (no resources
        // to free on a partially-started engine; the pool's channels close when
        // it drops).
        self.cancel.cancel();
        tracing::info!("sglang sidecar: cleanup invoked");
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
            .map(|s| {
                // Prefer the connectable `endpoint_addr` (a routable host:port)
                // over the legacy `endpoint` string, which the engine often
                // sets to a bind wildcard (e.g. `tcp://*:5557`) that the KV
                // router cannot dial from another node.
                let endpoint = match s.endpoint_addr {
                    Some(e) => {
                        let proto = if e.protocol.is_empty() { "tcp" } else { &e.protocol };
                        format!("{proto}://{}:{}", e.host, e.port)
                    }
                    None => s.endpoint,
                };
                KvEventSource::Zmq {
                    endpoint,
                    topic: s.topic,
                    dp_rank: s.data_parallel_rank,
                }
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

    // SGLang disaggregation keys KV transfer on a (host, port, room) triple. A
    // prefill engine advertises its bootstrap host/port as the first
    // `KvConnectorInfo.local_endpoints` entry; the frontend's PrefillRouter
    // needs them in `EngineConfig` to drive the decode peer's connect. Empty
    // (aggregated / decode) → `None`.
    let bootstrap = discovery
        .engine
        .kv_connector
        .as_ref()
        .and_then(|kv| kv.local_endpoints.first())
        .filter(|e| !e.host.is_empty() && e.port != 0);
    let bootstrap_host = bootstrap.map(|e| e.host.clone());
    let bootstrap_port = bootstrap.map(|e| e.port as u16);

    // Proto3 scalar `0` is "unset"; gate each optional field so we advertise
    // `None` rather than a bogus zero.
    EngineConfig {
        model: model.model_id.clone(),
        served_model_name,
        context_length: (model.max_context_length != 0).then_some(model.max_context_length),
        kv_cache_block_size: (model.kv_block_size != 0).then_some(model.kv_block_size),
        total_kv_blocks: (model.total_kv_blocks != 0).then_some(model.total_kv_blocks),
        max_num_seqs: (model.max_running_requests != 0).then_some(model.max_running_requests),
        max_num_batched_tokens: (model.max_batched_tokens != 0).then_some(model.max_batched_tokens),
        data_parallel_size: (parallelism.data_parallel_size != 0)
            .then_some(parallelism.data_parallel_size),
        data_parallel_start_rank: (parallelism.data_parallel_start_rank != 0)
            .then_some(parallelism.data_parallel_start_rank),
        bootstrap_host,
        bootstrap_port,
        runtime_data: Default::default(),
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

    // Disaggregation handoff → KV session the engine forwards to its connector.
    // SGLang's Bootstrap path (router-assigned `bootstrap_info`, sent to both
    // prefill and decode) takes precedence; fall back to the Completed path
    // (`prefill_result.disaggregated_params`). Aggregated → None.
    let kv_session = request
        .bootstrap_info
        .as_ref()
        .map(|bi| bootstrap_info_to_kv_session(bi, request_id))
        .or_else(|| {
            request
                .prefill_result
                .as_ref()
                .map(|pr| disagg_json_to_kv_session(&pr.disaggregated_params, request_id))
        });

    // Forward the KV router's forced DP-rank decision so the engine pins this
    // request to the rank that holds its prefix. Without it, SGLang internal-DP
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

    // Constrained / guided decoding. Validated by the frontend; the engine's
    // grammar backend enforces it during sampling. At most one guide is sent.
    let guided = request.sampling_options.guided_decoding.as_ref().and_then(|g| {
        use pb::guided_decoding::Guide;
        let guide = if let Some(json) = &g.json {
            Some(Guide::JsonSchema(json_value_to_string(json)))
        } else if let Some(regex) = &g.regex {
            Some(Guide::Regex(regex.clone()))
        } else if let Some(grammar) = &g.grammar {
            Some(Guide::EbnfGrammar(grammar.clone()))
        } else {
            g.structural_tag
                .as_ref()
                .map(|t| Guide::StructuralTag(json_value_to_string(t)))
        };
        guide.map(|guide| pb::GuidedDecoding {
            guide: Some(guide),
            backend: g.backend.clone().unwrap_or_default(),
        })
    });

    // Per-request LoRA adapter (router/preprocessor sets it on routing hints).
    let lora_name = request
        .routing
        .as_ref()
        .and_then(|r| r.lora_name.clone())
        .unwrap_or_default();

    // Logprobs: `logprobs` is the top-alternatives count; `prompt_logprobs`
    // asks the engine to also score the prompt (offset 0). Absent → completion
    // tokens only (engine default, signalled by logprob_start_len = -1).
    let opts = &request.output_options;
    let return_logprobs = opts.logprobs.is_some() || opts.prompt_logprobs.is_some();
    let top_logprobs = opts.logprobs.unwrap_or(0);
    let logprob_start_len = if opts.prompt_logprobs.is_some() { 0 } else { -1 };

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
        guided,
        lora_name,
        return_logprobs,
        top_logprobs,
        logprob_start_len,
        data_parallel_rank,
        kv_session,
        metadata: Default::default(),
    })
}

/// A guided-decoding constraint can arrive as a JSON string or a JSON
/// object/value; flatten to the string SGLang's grammar backend expects.
fn json_value_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
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
                        "sglang-sidecar received a pre-decoded RDMA media descriptor; the \
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

/// Build a `KvSessionRef` from the frontend's router-assigned `BootstrapInfo`
/// (SGLang Bootstrap path). The `(host, port, room)` triple is carried in
/// `attributes_struct` so the engine maps it back to SGLang bootstrap kwargs.
fn bootstrap_info_to_kv_session(
    bi: &dynamo_backend_common::BootstrapInfo,
    request_id: &str,
) -> pb::KvSessionRef {
    let mut fields = std::collections::BTreeMap::new();
    fields.insert(
        "bootstrap_host".to_string(),
        prost_types::Value {
            kind: Some(prost_types::value::Kind::StringValue(bi.bootstrap_host.clone())),
        },
    );
    fields.insert(
        "bootstrap_port".to_string(),
        prost_types::Value {
            kind: Some(prost_types::value::Kind::NumberValue(bi.bootstrap_port as f64)),
        },
    );
    fields.insert(
        "bootstrap_room".to_string(),
        prost_types::Value {
            kind: Some(prost_types::value::Kind::NumberValue(bi.bootstrap_room as f64)),
        },
    );
    pb::KvSessionRef {
        session_id: request_id.to_string(),
        transfer_backend: "nixl".to_string(),
        endpoints: Vec::new(),
        dp_rank: 0,
        attributes: std::collections::HashMap::new(),
        attributes_struct: Some(prost_types::Struct {
            fields: fields.into_iter().collect(),
        }),
    }
}

/// Encode a prefill `KvSessionRef` into the JSON the frontend's PrefillRouter
/// forwards to the decode peer (round-trips with
/// [`disagg_json_to_kv_session`]).
///
/// The sidecar is an opaque pass-through for the connector's handoff metadata:
/// the typed `attributes_struct` is carried as native JSON (types preserved),
/// with the legacy string-map `attributes` carried alongside only if set.
pub(crate) fn kv_session_to_disagg_json(session: pb::KvSessionRef) -> serde_json::Value {
    let mut obj = serde_json::json!({
        "session_id": session.session_id,
        "transfer_backend": session.transfer_backend,
        "dp_rank": session.dp_rank,
    });
    if let Some(s) = session.attributes_struct.as_ref() {
        obj["attributes_struct"] = prost_struct_to_json(s);
    }
    if !session.attributes.is_empty() {
        obj["attributes"] = serde_json::json!(session.attributes);
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
    let attributes = obj
        .and_then(|o| o.get("attributes"))
        .and_then(|v| v.as_object())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default();

    pb::KvSessionRef {
        session_id,
        transfer_backend,
        endpoints: Vec::new(),
        dp_rank,
        attributes,
        attributes_struct,
    }
}

/// Convert a JSON object into a `google.protobuf.Struct`. Non-object inputs
/// yield `None`.
pub(crate) fn json_to_prost_struct(value: &serde_json::Value) -> Option<prost_types::Struct> {
    match value {
        serde_json::Value::Object(map) => Some(prost_types::Struct {
            fields: map.iter().map(|(k, v)| (k.clone(), json_to_prost_value(v))).collect(),
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
            fields: map.iter().map(|(k, v)| (k.clone(), json_to_prost_value(v))).collect(),
        }),
    };
    prost_types::Value { kind: Some(kind) }
}

/// Convert a `google.protobuf.Struct` back into a JSON object.
pub(crate) fn prost_struct_to_json(s: &prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        s.fields.iter().map(|(k, v)| (k.clone(), prost_value_to_json(v))).collect(),
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
