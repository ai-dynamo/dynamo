// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `SglangSmgSidecarEngine` — an [`LLMEngine`] that drives upstream SGLang's
//! SMG `sglang.grpc.scheduler.SglangScheduler` service.
//!
//! The sidecar is Dynamo-only Rust. It never imports SGLang; it connects to an
//! already-started `python -m sglang.launch_server --grpc-mode` process, reads
//! model/server metadata, and registers itself as aggregated, prefill, or
//! decode according to the upstream engine's own `disaggregation_mode`.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::net::UdpSocket;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, BootstrapInfo, DisaggregationMode, DynamoError, EngineConfig,
    GenerateContext, HEALTH_CHECK_KEY, LLMEngine, LLMEngineOutput, LLMEngineOutputExt,
    LlmRegistration, ModelInput, PreprocessedRequest, TopLogprob, WorkerConfig, usage,
};
use dynamo_llm::protocols::common::preprocessor::{MultimodalData, MultimodalDataMap};
use futures::stream::BoxStream;
use prost_types::value::Kind;
use tokio::sync::OnceCell;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::args::{Args, TransportConfig, normalize_endpoint};
use crate::client::{self, Client, Discovery, Pool};
use crate::proto::scheduler as pb;

const FAKE_BOOTSTRAP_HOST: &str = "__dynamo_health_check__";
const MAX_SMG_BOOTSTRAP_ROOM: u64 = i32::MAX as u64;

/// A Dynamo backend that proxies inference to an upstream SGLang SMG server.
pub struct SglangSmgSidecarEngine {
    /// Normalised gRPC endpoint (e.g. `http://127.0.0.1:30000`).
    endpoint: String,
    /// Connect / readiness tunables.
    transport: TransportConfig,
    /// Role discovered at bootstrap; re-validated in `start()`.
    disaggregation_mode: DisaggregationMode,
    /// Connection pool, set once in `start()`.
    pool: OnceCell<Pool>,
    /// Live metadata used by request lowering after `start()`.
    metadata: OnceCell<RuntimeMetadata>,
    /// Cancels in-flight `generate()` streams on `cleanup()`.
    cancel: CancellationToken,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RuntimeMetadata {
    pub(crate) data_parallel_size: Option<u32>,
    pub(crate) bootstrap_host: Option<String>,
    pub(crate) bootstrap_port: Option<u16>,
}

impl SglangSmgSidecarEngine {
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
            metadata: OnceCell::new(),
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

        let endpoint = normalize_endpoint(&args.smg_endpoint);
        let transport = args.transport();

        let discovery = bootstrap_discover(&endpoint, &transport)?;
        validate_discovery(&discovery)?;
        let disaggregation_mode = discover_disaggregation_mode(&discovery)?;
        let served_model_name = served_model_name(&discovery.model);

        tracing::info!(
            %endpoint,
            mode = %disaggregation_mode,
            model = %discovery.model.model_path,
            served_model_name = ?served_model_name,
            "SGLang SMG sidecar bootstrapped engine discovery"
        );

        let (tool_call_parser, reasoning_parser) = if disaggregation_mode.is_prefill() {
            (None, None)
        } else {
            (args.dyn_tool_call_parser, args.dyn_reasoning_parser)
        };

        let config = WorkerConfig {
            namespace: args.namespace,
            component: component_for_mode(disaggregation_mode).to_string(),
            endpoint: args.endpoint,
            endpoint_types: args.endpoint_types,
            custom_jinja_template: args.custom_jinja_template,
            disaggregation_mode,
            model_name: discovery.model.model_path.clone(),
            served_model_name,
            model_input: ModelInput::Tokens,
            tool_call_parser,
            reasoning_parser,
            exclude_tools_when_tool_choice_none: args.exclude_tools_when_tool_choice_none,
            enable_kv_routing: false,
            ..Default::default()
        };

        let engine = Self::new(endpoint, transport, disaggregation_mode);
        Ok((engine, config))
    }

    /// Poll SMG `HealthCheck` until the engine reports healthy or the deadline
    /// elapses. Transient RPC errors are retried because the SGLang process may
    /// still be loading the model.
    async fn await_ready(&self, client: &mut Client) -> Result<(), DynamoError> {
        let deadline = Instant::now() + self.transport.deadline;
        loop {
            let outcome = client.health_check(pb::HealthCheckRequest {}).await;
            let retry_msg = match outcome {
                Ok(resp) => {
                    let resp = resp.into_inner();
                    if resp.healthy {
                        return Ok(());
                    }
                    if resp.message.is_empty() {
                        "engine health check returned unhealthy".to_string()
                    } else {
                        format!("engine unhealthy: {}", resp.message)
                    }
                }
                Err(status) => format!("HealthCheck RPC failed: {}", status.message()),
            };

            if Instant::now() >= deadline {
                return Err(client::engine_shutdown(format!(
                    "SMG SGLang engine did not become healthy within {:?}: {retry_msg}",
                    self.transport.deadline
                )));
            }
            tokio::time::sleep(self.transport.poll_interval).await;
        }
    }
}

#[async_trait]
impl LLMEngine for SglangSmgSidecarEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.pool.initialized() {
            return Err(client::engine_shutdown(
                "SGLang SMG sidecar already started",
            ));
        }

        let pool =
            Pool::connect(&self.endpoint, &self.transport, self.transport.connections).await?;
        let mut control = pool.control_client();
        self.await_ready(&mut control).await?;
        let discovery = client::discover(&mut control).await?;
        validate_discovery(&discovery)?;

        let observed = discover_disaggregation_mode(&discovery)?;
        if observed != self.disaggregation_mode {
            return Err(client::invalid_arg(format!(
                "SGLang SMG engine role changed since bootstrap: registered as {:?} but engine now reports {:?}",
                self.disaggregation_mode, observed
            )));
        }

        let metadata = build_runtime_metadata(&discovery, observed);
        if observed.is_prefill()
            && (metadata.bootstrap_host.is_none() || metadata.bootstrap_port.is_none())
        {
            return Err(client::invalid_arg(
                "SGLang SMG prefill worker did not advertise a bootstrap host/port through GetServerInfo",
            ));
        }

        let pool_size = pool.len();
        self.metadata
            .set(metadata)
            .map_err(|_| client::engine_shutdown("SGLang SMG sidecar already started"))?;
        self.pool
            .set(pool)
            .map_err(|_| client::engine_shutdown("SGLang SMG sidecar already started"))?;

        let config = build_engine_config(&discovery, observed);
        tracing::info!(
            model = %config.model,
            mode = %observed,
            context_length = ?config.llm.as_ref().and_then(|llm| llm.context_length),
            data_parallel_size = ?config.llm.as_ref().and_then(|llm| llm.data_parallel_size),
            connections = pool_size,
            "SGLang SMG sidecar started"
        );
        Ok(config)
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        validate_supported_request(&request)?;

        let metadata = self
            .metadata
            .get()
            .cloned()
            .ok_or_else(|| client::engine_shutdown("generate called before start"))?;
        let mut client = self
            .pool
            .get()
            .map(Pool::stream_client)
            .ok_or_else(|| client::engine_shutdown("generate called before start"))?;

        if self.disaggregation_mode.is_decode()
            && request.prefill_result.is_none()
            && request.bootstrap_info.is_none()
            && !request.is_probe
        {
            return Err(client::invalid_arg(
                "SGLang SMG decode worker received request with neither bootstrap_info nor prefill_result",
            ));
        }

        let is_prefill = self.disaggregation_mode.is_prefill();
        let prompt_len = request.token_ids.len() as u32;
        let (grpc_req, prefill_disagg_json) =
            build_generate_request(&request, ctx.id(), is_prefill, &metadata)?;
        let cancel = self.cancel.clone();

        Ok(Box::pin(async_stream::stream! {
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
                    yield Err(client::status_to_dynamo("Generate", status));
                    return;
                }
            };

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
                                match resp.response {
                                    Some(pb::generate_response::Response::Chunk(chunk)) => {
                                        if chunk.prompt_tokens > 0 {
                                            prompt_tokens = chunk.prompt_tokens as u32;
                                        }
                                        if is_prefill || chunk.token_ids.is_empty() {
                                            continue;
                                        }
                                        generated = generated.saturating_add(chunk.token_ids.len() as u32);
                                        yield Ok(chunk_output(chunk));
                                    }
                                    Some(pb::generate_response::Response::Complete(done)) => {
                                        if done.prompt_tokens > 0 {
                                            prompt_tokens = done.prompt_tokens as u32;
                                        }
                                        if !is_prefill && generated == 0 && !done.output_ids.is_empty() {
                                            generated = generated.saturating_add(done.output_ids.len() as u32);
                                            yield Ok(LLMEngineOutput {
                                                token_ids: done.output_ids.clone(),
                                                index: Some(done.index),
                                                ..Default::default()
                                            });
                                        }
                                        let mut out = finish_output(
                                            &done.finish_reason,
                                            prompt_tokens,
                                            generated,
                                        );
                                        out.index = Some(done.index);
                                        if is_prefill {
                                            out.disaggregated_params = prefill_disagg_json.clone();
                                        }
                                        yield Ok(out);
                                        break;
                                    }
                                    Some(pb::generate_response::Response::Error(err)) => {
                                        yield Err(client::generate_error_to_dynamo(&err));
                                        break;
                                    }
                                    None => continue,
                                }
                            }
                            Ok(None) => {
                                yield Err(client::engine_shutdown(
                                    "SMG SGLang engine closed the Generate stream before a complete event",
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
            reason: "cancelled by Dynamo".to_string(),
        };
        if let Err(status) = client.abort(req).await {
            tracing::debug!(
                error = %status.message(),
                request_id = ctx.id(),
                "SGLang SMG sidecar: abort RPC failed (ignored)"
            );
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.cancel.cancel();
        tracing::info!("SGLang SMG sidecar: cleanup invoked");
        Ok(())
    }

    async fn health_check_payload(&self) -> Result<Option<serde_json::Value>, DynamoError> {
        if !matches!(self.disaggregation_mode, DisaggregationMode::Aggregated) {
            // Disaggregated probes need a real bootstrap handoff. The runtime
            // canary cannot synthesize one that is valid for both SGLang
            // transfer backends and topology-routed workers.
            return Ok(None);
        }

        let mut payload = serde_json::json!({
            "token_ids": [1],
            "stop_conditions": {"max_tokens": 1, "ignore_eos": true},
            "sampling_options": {"temperature": 0.0},
        });
        payload[HEALTH_CHECK_KEY] = serde_json::Value::Bool(true);
        Ok(Some(payload))
    }
}

// ============================================================================
// Discovery / config helpers
// ============================================================================

fn bootstrap_discover(
    endpoint: &str,
    transport: &TransportConfig,
) -> Result<Discovery, DynamoError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| client::engine_shutdown(format!("bootstrap runtime: {e}")))?;
    rt.block_on(async {
        let deadline = Instant::now() + transport.deadline;
        loop {
            match client::connect(endpoint, transport).await {
                Ok(mut c) => match client::discover(&mut c).await {
                    Ok(discovery) => return Ok(discovery),
                    Err(err) if Instant::now() < deadline => {
                        tracing::debug!(error = %err, "SGLang SMG bootstrap discovery not ready yet");
                    }
                    Err(err) => return Err(err),
                },
                Err(err) if Instant::now() < deadline => {
                    tracing::debug!(error = %err, "SGLang SMG bootstrap connect not ready yet");
                }
                Err(err) => return Err(err),
            }
            tokio::time::sleep(transport.poll_interval).await;
        }
    })
}

fn validate_discovery(discovery: &Discovery) -> Result<(), DynamoError> {
    if discovery.model.model_path.is_empty() {
        return Err(client::invalid_arg(
            "SMG GetModelInfo returned an empty model_path",
        ));
    }
    if !discovery.model.is_generation {
        return Err(client::invalid_arg(format!(
            "SGLang SMG sidecar supports generation models only; model `{}` reports is_generation=false",
            discovery.model.model_path
        )));
    }
    Ok(())
}

fn served_model_name(model: &pb::GetModelInfoResponse) -> Option<String> {
    (!model.served_model_name.is_empty()).then(|| model.served_model_name.clone())
}

fn discover_disaggregation_mode(discovery: &Discovery) -> Result<DisaggregationMode, DynamoError> {
    let raw = field_string(
        &discovery.server,
        &[
            "disaggregation_mode",
            "disaggregationMode",
            "kv_role",
            "role",
        ],
    )
    .unwrap_or_else(|| "null".to_string());

    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "null" | "none" | "agg" | "aggregated" | "worker" | "backend" => {
            Ok(DisaggregationMode::Aggregated)
        }
        "prefill" | "kv_producer" => Ok(DisaggregationMode::Prefill),
        "decode" | "kv_consumer" => Ok(DisaggregationMode::Decode),
        other => Err(client::invalid_arg(format!(
            "SGLang SMG sidecar does not understand disaggregation mode {other:?}"
        ))),
    }
}

fn component_for_mode(mode: DisaggregationMode) -> &'static str {
    if mode.is_prefill() {
        "prefill"
    } else {
        "backend"
    }
}

fn build_runtime_metadata(discovery: &Discovery, mode: DisaggregationMode) -> RuntimeMetadata {
    let data_parallel_size = field_u32(&discovery.server, &["dp_size", "data_parallel_size"]);
    let (bootstrap_host, bootstrap_port) = if mode.is_prefill() {
        discover_bootstrap_address(&discovery.server)
    } else {
        (None, None)
    };
    RuntimeMetadata {
        data_parallel_size,
        bootstrap_host,
        bootstrap_port,
    }
}

fn build_engine_config(discovery: &Discovery, mode: DisaggregationMode) -> EngineConfig {
    let metadata = build_runtime_metadata(discovery, mode);
    let context_length = positive_i32_as_u32(discovery.model.max_context_length)
        .or_else(|| positive_i32_as_u32(discovery.model.max_req_input_len))
        .or_else(|| field_u32(&discovery.server, &["context_length", "max_context_length"]));
    let kv_cache_block_size = field_u32(&discovery.server, &["page_size", "kv_cache_block_size"]);
    let total_kv_blocks = field_u64(
        &discovery.server,
        &["total_kv_blocks", "total_num_gpu_blocks", "num_gpu_blocks"],
    );
    let max_num_seqs = field_u64(
        &discovery.server,
        &[
            "max_running_requests",
            "max_num_seqs",
            "max_concurrent_requests",
        ],
    );
    let max_num_batched_tokens = field_u64(
        &discovery.server,
        &["max_total_tokens", "max_num_batched_tokens"],
    );
    let data_parallel_start_rank = field_u32(
        &discovery.server,
        &["data_parallel_start_rank", "dp_start_rank"],
    );

    let mut runtime_data = HashMap::new();
    if !discovery.server.sglang_version.is_empty() {
        runtime_data.insert(
            "sglang_version".to_string(),
            serde_json::Value::String(discovery.server.sglang_version.clone()),
        );
    }
    if !discovery.server.server_type.is_empty() {
        runtime_data.insert(
            "server_type".to_string(),
            serde_json::Value::String(discovery.server.server_type.clone()),
        );
    }
    runtime_data.insert(
        "disaggregation_mode".to_string(),
        serde_json::Value::String(mode.to_string()),
    );

    EngineConfig {
        model: discovery.model.model_path.clone(),
        served_model_name: served_model_name(&discovery.model),
        runtime_data,
        llm: Some(LlmRegistration {
            context_length,
            kv_cache_block_size,
            total_kv_blocks,
            max_num_seqs,
            max_num_batched_tokens,
            data_parallel_size: metadata.data_parallel_size,
            data_parallel_start_rank,
            bootstrap_host: metadata.bootstrap_host,
            bootstrap_port: metadata.bootstrap_port,
        }),
    }
}

fn positive_i32_as_u32(value: i32) -> Option<u32> {
    (value > 0).then(|| value as u32)
}

fn discover_bootstrap_address(server: &pb::GetServerInfoResponse) -> (Option<String>, Option<u16>) {
    let port = field_u32(server, &["disaggregation_bootstrap_port", "bootstrap_port"])
        .and_then(|p| u16::try_from(p).ok())
        .filter(|p| *p != 0);
    let host = field_string(
        server,
        &[
            "bootstrap_host",
            "disaggregation_bootstrap_host",
            "advertised_host",
            "host_ip",
            "host",
        ],
    )
    .and_then(|h| {
        let trimmed = h.trim();
        if trimmed.is_empty() || matches!(trimmed, "0.0.0.0" | "::" | "[::]") {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
    .or_else(local_ip_auto);
    (host, port)
}

fn local_ip_auto() -> Option<String> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    Some(socket.local_addr().ok()?.ip().to_string())
}

fn field_string(server: &pb::GetServerInfoResponse, keys: &[&str]) -> Option<String> {
    for source in [&server.server_args, &server.scheduler_info] {
        let Some(s) = source.as_ref() else { continue };
        for key in keys {
            let Some(value) = s.fields.get(*key) else {
                continue;
            };
            if let Some(s) = prost_value_as_string(value) {
                return Some(s);
            }
        }
    }
    None
}

fn field_u32(server: &pb::GetServerInfoResponse, keys: &[&str]) -> Option<u32> {
    field_u64(server, keys).and_then(|v| u32::try_from(v).ok())
}

fn field_u64(server: &pb::GetServerInfoResponse, keys: &[&str]) -> Option<u64> {
    for source in [&server.server_args, &server.scheduler_info] {
        let Some(s) = source.as_ref() else { continue };
        for key in keys {
            let Some(value) = s.fields.get(*key) else {
                continue;
            };
            if let Some(n) = prost_value_as_u64(value) {
                return Some(n);
            }
        }
    }
    None
}

fn prost_value_as_string(value: &prost_types::Value) -> Option<String> {
    match value.kind.as_ref()? {
        Kind::StringValue(s) => Some(s.clone()),
        Kind::NumberValue(n) if n.is_finite() => Some(number_to_json(*n).to_string()),
        Kind::BoolValue(b) => Some(b.to_string()),
        _ => None,
    }
}

fn prost_value_as_u64(value: &prost_types::Value) -> Option<u64> {
    match value.kind.as_ref()? {
        Kind::NumberValue(n) if n.is_finite() && *n >= 0.0 => Some(*n as u64),
        Kind::StringValue(s) => s.parse().ok(),
        _ => None,
    }
}

// ============================================================================
// Request building + response mapping
// ============================================================================

pub(crate) fn validate_supported_request(request: &PreprocessedRequest) -> Result<(), DynamoError> {
    if request.prompt_embeds.is_some() {
        return Err(unsupported("prompt embeddings"));
    }
    if request.mm_processor_kwargs.is_some() {
        return Err(unsupported("multimodal processor kwargs"));
    }
    if request.stop_conditions.max_thinking_tokens.is_some() {
        return Err(unsupported("thinking-token budget"));
    }

    let sampling = &request.sampling_options;
    if sampling.n.unwrap_or(1) > 1 {
        return Err(unsupported("multi-output sampling (n > 1)"));
    }
    if sampling.best_of.unwrap_or(1) > 1 {
        return Err(unsupported("best_of > 1"));
    }
    if sampling.use_beam_search.unwrap_or(false) {
        return Err(unsupported("beam search"));
    }
    if sampling.length_penalty.is_some() {
        return Err(unsupported("length_penalty"));
    }
    if let Some(guided) = sampling.guided_decoding.as_ref() {
        if guided.choice.as_ref().is_some_and(|c| !c.is_empty()) {
            return Err(unsupported("guided decoding choice constraint"));
        }
        if guided.backend.is_some() {
            return Err(unsupported("guided decoding backend override"));
        }
        if guided.whitespace_pattern.is_some() {
            return Err(unsupported("guided decoding whitespace_pattern"));
        }
    }
    Ok(())
}

fn unsupported(feature: &str) -> DynamoError {
    client::invalid_arg(format!(
        "SGLang SMG sidecar supports aggregated and disaggregated text/token generation with URL multimodal passthrough; unsupported feature: {feature}"
    ))
}

pub(crate) fn build_generate_request(
    request: &PreprocessedRequest,
    request_id: &str,
    is_prefill: bool,
    metadata: &RuntimeMetadata,
) -> Result<(pb::GenerateRequest, Option<serde_json::Value>), DynamoError> {
    validate_supported_request(request)?;

    let disaggregated_params =
        build_disaggregated_params(request, request_id, is_prefill, metadata)?;
    let prefill_disagg_json = if is_prefill {
        disaggregated_params
            .as_ref()
            .map(disaggregated_params_to_json)
    } else {
        None
    };

    let data_parallel_rank = routed_dp_rank(request, is_prefill)
        .map(|r| i32::try_from(r).unwrap_or(i32::MAX))
        .unwrap_or(-1);

    Ok((
        pb::GenerateRequest {
            request_id: request_id.to_string(),
            tokenized: Some(pb::TokenizedInput {
                original_text: String::new(),
                input_ids: request.token_ids.clone(),
            }),
            mm_inputs: build_multimodal_inputs(request)?,
            sampling_params: Some(build_sampling_params(request, is_prefill)?),
            return_logprob: request.output_options.logprobs.is_some()
                || request.output_options.prompt_logprobs.is_some(),
            logprob_start_len: if request.output_options.prompt_logprobs.is_some() {
                0
            } else {
                -1
            },
            top_logprobs_num: request
                .output_options
                .logprobs
                .or(request.output_options.prompt_logprobs)
                .map(|v| i32::try_from(v).unwrap_or(i32::MAX))
                .unwrap_or(0),
            token_ids_logprob: Vec::new(),
            return_hidden_states: false,
            disaggregated_params,
            custom_logit_processor: String::new(),
            timestamp: None,
            log_metrics: true,
            input_embeds: Vec::new(),
            lora_id: request
                .routing
                .as_ref()
                .and_then(|r| r.lora_name.clone())
                .unwrap_or_default(),
            data_parallel_rank,
            stream: true,
        },
        prefill_disagg_json,
    ))
}

fn routed_dp_rank(request: &PreprocessedRequest, is_prefill: bool) -> Option<u32> {
    request.routing.as_ref().and_then(|r| {
        if is_prefill {
            r.prefill_dp_rank.or(r.dp_rank)
        } else {
            r.dp_rank
        }
    })
}

fn build_sampling_params(
    request: &PreprocessedRequest,
    is_prefill: bool,
) -> Result<pb::SamplingParams, DynamoError> {
    let sampling = &request.sampling_options;
    let stop = &request.stop_conditions;
    let max_new_tokens = if is_prefill {
        Some(1)
    } else {
        stop.max_tokens
            .map(|v| i32::try_from(v).map_err(|_| client::invalid_arg("max_tokens exceeds int32")))
            .transpose()?
    };

    let mut stop_token_ids = Vec::new();
    if let Some(ids) = &stop.stop_token_ids {
        stop_token_ids.extend(ids.iter().copied());
    }
    if let Some(ids) = &stop.stop_token_ids_hidden {
        stop_token_ids.extend(ids.iter().copied());
    }

    Ok(pb::SamplingParams {
        temperature: sampling.temperature.unwrap_or(1.0),
        top_p: sampling.top_p.unwrap_or(1.0),
        top_k: sampling.top_k.unwrap_or(-1),
        min_p: sampling.min_p.unwrap_or(0.0),
        frequency_penalty: sampling.frequency_penalty.unwrap_or(0.0),
        presence_penalty: sampling.presence_penalty.unwrap_or(0.0),
        repetition_penalty: sampling.repetition_penalty.unwrap_or(1.0),
        max_new_tokens,
        stop: stop.stop.clone().unwrap_or_default(),
        stop_token_ids,
        skip_special_tokens: request.output_options.skip_special_tokens.unwrap_or(true),
        spaces_between_special_tokens: true,
        constraint: build_guided_constraint(sampling)?,
        n: i32::from(sampling.n.unwrap_or(1)),
        min_new_tokens: stop
            .min_tokens
            .map(|v| i32::try_from(v).unwrap_or(i32::MAX))
            .unwrap_or(0),
        ignore_eos: stop.ignore_eos.unwrap_or(false),
        no_stop_trim: sampling.include_stop_str_in_output.unwrap_or(false),
        stream_interval: None,
        logit_bias: HashMap::new(),
        custom_params: request.extra_args.as_ref().and_then(json_to_prost_struct),
    })
}

fn build_guided_constraint(
    sampling: &dynamo_backend_common::SamplingOptions,
) -> Result<Option<pb::sampling_params::Constraint>, DynamoError> {
    let Some(guided) = sampling.guided_decoding.as_ref() else {
        return Ok(None);
    };

    if let Some(json) = guided.json.as_ref() {
        let schema = match json {
            serde_json::Value::String(s) => s.clone(),
            value => serde_json::to_string(value).map_err(|e| {
                client::invalid_arg(format!("failed to serialize guided JSON schema: {e}"))
            })?,
        };
        return Ok(Some(pb::sampling_params::Constraint::JsonSchema(schema)));
    }
    if let Some(regex) = guided.regex.as_ref() {
        return Ok(Some(pb::sampling_params::Constraint::Regex(regex.clone())));
    }
    if let Some(grammar) = guided.grammar.as_ref() {
        return Ok(Some(pb::sampling_params::Constraint::EbnfGrammar(
            grammar.clone(),
        )));
    }
    if let Some(tag) = guided.structural_tag.as_ref() {
        let tag = match tag {
            serde_json::Value::String(s) => s.clone(),
            value => serde_json::to_string(value).map_err(|e| {
                client::invalid_arg(format!("failed to serialize structural_tag: {e}"))
            })?,
        };
        return Ok(Some(pb::sampling_params::Constraint::StructuralTag(tag)));
    }

    Ok(None)
}

fn build_multimodal_inputs(
    request: &PreprocessedRequest,
) -> Result<Option<pb::MultimodalInputs>, DynamoError> {
    let Some(map) = request.multi_modal_data.as_ref() else {
        return Ok(None);
    };

    let mut mm = pb::MultimodalInputs {
        image_urls: Vec::new(),
        video_urls: Vec::new(),
        audio_urls: Vec::new(),
        processed_features: None,
        image_data: Vec::new(),
        video_data: Vec::new(),
        audio_data: Vec::new(),
        modalities: Vec::new(),
    };

    push_media_urls(map, &mut mm.image_urls, "image", &["image_url", "image"])?;
    push_media_urls(map, &mut mm.video_urls, "video", &["video_url", "video"])?;
    push_media_urls(map, &mut mm.audio_urls, "audio", &["audio_url", "audio"])?;

    if mm.image_urls.is_empty() && mm.video_urls.is_empty() && mm.audio_urls.is_empty() {
        Ok(None)
    } else {
        Ok(Some(mm))
    }
}

fn push_media_urls(
    map: &MultimodalDataMap,
    target: &mut Vec<String>,
    modality: &str,
    keys: &[&str],
) -> Result<(), DynamoError> {
    for key in keys {
        let Some(items) = map.get(*key) else { continue };
        for item in items {
            match item {
                MultimodalData::Url(u) => target.push(u.as_str().to_string()),
                MultimodalData::RawUrl(s) => target.push(s.clone()),
                MultimodalData::Decoded(_) => {
                    return Err(client::invalid_arg(format!(
                        "SGLang SMG sidecar received a pre-decoded RDMA {modality} descriptor; run the frontend in URL-passthrough mode"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn build_disaggregated_params(
    request: &PreprocessedRequest,
    request_id: &str,
    is_prefill: bool,
    metadata: &RuntimeMetadata,
) -> Result<Option<pb::DisaggregatedParams>, DynamoError> {
    if let Some(info) = request.bootstrap_info.as_ref() {
        return Ok(Some(bootstrap_info_to_params(
            info,
            metadata.data_parallel_size,
        )?));
    }
    if let Some(prefill) = request.prefill_result.as_ref() {
        return Ok(Some(disagg_json_to_params(
            &prefill.disaggregated_params,
            metadata.data_parallel_size,
        )?));
    }
    if is_prefill {
        let (Some(host), Some(port)) = (
            metadata.bootstrap_host.as_ref(),
            metadata.bootstrap_port.as_ref(),
        ) else {
            return Err(client::invalid_arg(
                "SGLang SMG prefill worker has no bootstrap host/port",
            ));
        };
        let room = fallback_room(request_id);
        return Ok(Some(pb::DisaggregatedParams {
            bootstrap_host: host.clone(),
            bootstrap_port: i32::from(*port),
            bootstrap_room: normalize_bootstrap_room(room, metadata.data_parallel_size)?,
        }));
    }
    if request.is_probe {
        return Ok(Some(pb::DisaggregatedParams {
            bootstrap_host: FAKE_BOOTSTRAP_HOST.to_string(),
            bootstrap_port: 0,
            bootstrap_room: 0,
        }));
    }
    Ok(None)
}

fn bootstrap_info_to_params(
    info: &BootstrapInfo,
    dp_size: Option<u32>,
) -> Result<pb::DisaggregatedParams, DynamoError> {
    Ok(pb::DisaggregatedParams {
        bootstrap_host: info.bootstrap_host.clone(),
        bootstrap_port: i32::from(info.bootstrap_port),
        bootstrap_room: normalize_bootstrap_room(info.bootstrap_room, dp_size)?,
    })
}

fn disagg_json_to_params(
    value: &serde_json::Value,
    dp_size: Option<u32>,
) -> Result<pb::DisaggregatedParams, DynamoError> {
    let obj = value.as_object().ok_or_else(|| {
        client::invalid_arg("disaggregated_params must be a JSON object with bootstrap metadata")
    })?;
    let host = obj
        .get("bootstrap_host")
        .and_then(|v| v.as_str())
        .ok_or_else(|| client::invalid_arg("disaggregated_params missing bootstrap_host"))?;
    let port = obj
        .get("bootstrap_port")
        .and_then(json_u64)
        .and_then(|p| i32::try_from(p).ok())
        .ok_or_else(|| client::invalid_arg("disaggregated_params missing bootstrap_port"))?;
    let room = obj
        .get("bootstrap_room")
        .and_then(json_u64)
        .ok_or_else(|| client::invalid_arg("disaggregated_params missing bootstrap_room"))?;

    Ok(pb::DisaggregatedParams {
        bootstrap_host: host.to_string(),
        bootstrap_port: port,
        bootstrap_room: normalize_bootstrap_room(room, dp_size)?,
    })
}

fn disaggregated_params_to_json(params: &pb::DisaggregatedParams) -> serde_json::Value {
    serde_json::json!({
        "bootstrap_host": params.bootstrap_host.clone(),
        "bootstrap_port": params.bootstrap_port,
        "bootstrap_room": params.bootstrap_room,
    })
}

fn fallback_room(request_id: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    request_id.hash(&mut hasher);
    hasher.finish()
}

fn normalize_bootstrap_room(room: u64, dp_size: Option<u32>) -> Result<i32, DynamoError> {
    let span = match dp_size {
        Some(size) if size > 0 => {
            let size = u64::from(size);
            (MAX_SMG_BOOTSTRAP_ROOM / size).saturating_mul(size).max(1)
        }
        _ => MAX_SMG_BOOTSTRAP_ROOM,
    };
    let room = room % span;
    i32::try_from(room).map_err(|_| {
        client::invalid_arg(format!(
            "bootstrap_room {room} cannot fit SGLang SMG int32 field"
        ))
    })
}

fn json_u64(value: &serde_json::Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|v| u64::try_from(v).ok()))
        .or_else(|| value.as_str().and_then(|s| s.parse().ok()))
}

fn chunk_output(chunk: pb::GenerateStreamChunk) -> LLMEngineOutput {
    let (log_probs, top_logprobs) = map_output_logprobs(chunk.output_logprobs.as_ref());
    LLMEngineOutput {
        token_ids: chunk.token_ids,
        log_probs,
        top_logprobs,
        index: Some(chunk.index),
        ..Default::default()
    }
}

fn map_output_logprobs(
    logprobs: Option<&pb::OutputLogProbs>,
) -> (Option<Vec<f64>>, Option<Vec<Vec<TopLogprob>>>) {
    let Some(logprobs) = logprobs else {
        return (None, None);
    };
    let chosen = (!logprobs.token_logprobs.is_empty()).then(|| {
        logprobs
            .token_logprobs
            .iter()
            .map(|v| f64::from(*v))
            .collect::<Vec<_>>()
    });
    let top = (!logprobs.top_logprobs.is_empty()).then(|| {
        logprobs
            .top_logprobs
            .iter()
            .map(|entry| {
                entry
                    .values
                    .iter()
                    .zip(entry.token_ids.iter())
                    .enumerate()
                    .filter_map(|(rank, (logprob, token_id))| {
                        Some(TopLogprob {
                            rank: u32::try_from(rank).ok()?,
                            token_id: u32::try_from(*token_id).ok()?,
                            token: None,
                            logprob: f64::from(*logprob),
                            bytes: None,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });
    (chosen, top)
}

fn finish_output(reason: &str, prompt_tokens: u32, generated: u32) -> LLMEngineOutput {
    let normalized = reason.to_ascii_lowercase();
    match normalized.as_str() {
        "length" => LLMEngineOutput::length(),
        "abort" | "aborted" | "cancelled" | "canceled" => LLMEngineOutput::cancelled(),
        "error" => LLMEngineOutput::error("engine reported error finish reason".to_string()),
        _ => LLMEngineOutput::stop(),
    }
    .with_usage(usage(prompt_tokens, generated))
}

// ============================================================================
// JSON <-> google.protobuf.Struct helpers
// ============================================================================

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

fn number_to_json(n: f64) -> serde_json::Value {
    if n.is_finite() && n.fract() == 0.0 && n >= i64::MIN as f64 && n <= i64::MAX as f64 {
        serde_json::Value::Number((n as i64).into())
    } else {
        serde_json::Number::from_f64(n)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null)
    }
}
