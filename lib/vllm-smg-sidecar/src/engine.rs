// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `VllmSmgSidecarEngine` — an [`LLMEngine`] that drives upstream vLLM
//! `serve --grpc` over SMG's `vllm.grpc.engine.VllmEngine` service.
//!
//! This is intentionally smaller than the OpenEngine sidecar. SMG's upstream
//! vLLM servicer provides generation, health, abort, and coarse model/server
//! metadata, but not enough stable surface for Dynamo-equivalent disaggregated
//! serving, KV-event routing, or URL-passthrough multimodal inputs. The first
//! implementation is therefore aggregated text/token generation only.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, DisaggregationMode, DynamoError, EngineConfig, GenerateContext,
    HEALTH_CHECK_KEY, LLMEngine, LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest,
    WorkerConfig, usage,
};
use futures::stream::BoxStream;
use tokio::sync::{OnceCell, watch};
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use crate::args::{Args, TransportConfig, normalize_endpoint};
use crate::client::{self, Client, Discovery, Pool};
use crate::proto::engine as pb;

/// A Dynamo backend that proxies inference to an upstream vLLM SMG server.
pub struct VllmSmgSidecarEngine {
    /// Normalised gRPC endpoint (e.g. `http://127.0.0.1:50051`).
    endpoint: String,
    /// Connect / readiness tunables.
    transport: TransportConfig,
    /// Connection pool, set once in `start()`.
    pool: OnceCell<Pool>,
    /// Cancels in-flight `generate()` streams on `cleanup()`.
    cancel: CancellationToken,
    /// Set when the out-of-process SMG server can no longer serve.
    fatal: watch::Sender<Option<String>>,
}

impl VllmSmgSidecarEngine {
    /// Direct constructor. The public entry point is [`from_args`](Self::from_args);
    /// this exists for programmatic / test construction.
    pub(crate) fn new(endpoint: impl Into<String>, transport: TransportConfig) -> Self {
        let (fatal, _) = watch::channel(None);
        Self {
            endpoint: endpoint.into(),
            transport,
            pool: OnceCell::new(),
            cancel: CancellationToken::new(),
            fatal,
        }
    }

    /// Parse CLI args, bootstrap-discover the engine metadata, and build the
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

        let served_model_name = (!discovery.model.served_model_name.is_empty())
            .then(|| discovery.model.served_model_name.clone());

        tracing::info!(
            %endpoint,
            model = %discovery.model.model_path,
            served_model_name = ?served_model_name,
            kv_role = %discovery.server.kv_role,
            "vllm SMG sidecar bootstrapped engine discovery"
        );

        let config = WorkerConfig {
            namespace: args.namespace,
            component: "backend".to_string(),
            endpoint: args.endpoint,
            endpoint_types: args.endpoint_types,
            custom_jinja_template: args.custom_jinja_template,
            disaggregation_mode: DisaggregationMode::Aggregated,
            model_name: discovery.model.model_path.clone(),
            served_model_name,
            ..Default::default()
        };

        let engine = Self::new(endpoint, transport);
        Ok((engine, config))
    }

    /// Poll SMG `HealthCheck` until the engine reports healthy or the deadline
    /// elapses. Transient RPC errors are retried because the vLLM process may
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
                    "SMG vLLM engine did not become healthy within {:?}: {retry_msg}",
                    self.transport.deadline
                )));
            }
            tokio::time::sleep(self.transport.poll_interval).await;
        }
    }
}

#[async_trait]
impl LLMEngine for VllmSmgSidecarEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.pool.initialized() {
            return Err(client::engine_shutdown("vllm SMG sidecar already started"));
        }

        let pool =
            Pool::connect(&self.endpoint, &self.transport, self.transport.connections).await?;
        let mut control = pool.control_client();
        self.await_ready(&mut control).await?;
        let discovery = client::discover(&mut control).await?;
        validate_discovery(&discovery)?;

        let pool_size = pool.len();
        self.pool
            .set(pool)
            .map_err(|_| client::engine_shutdown("vllm SMG sidecar already started"))?;

        let config = build_engine_config(&discovery);
        tracing::info!(
            model = %config.model,
            context_length = ?config.context_length,
            data_parallel_size = ?config.data_parallel_size,
            connections = pool_size,
            "vllm SMG sidecar started"
        );
        Ok(config)
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        validate_supported_request(&request)?;

        let mut client = self
            .pool
            .get()
            .map(Pool::stream_client)
            .ok_or_else(|| client::engine_shutdown("generate called before start"))?;

        let prompt_len = request.token_ids.len() as u32;
        let grpc_req = build_generate_request(&request, ctx.id())?;
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
                                        if chunk.prompt_tokens != 0 {
                                            prompt_tokens = chunk.prompt_tokens;
                                        }
                                        if chunk.token_ids.is_empty() {
                                            continue;
                                        }
                                        generated = generated.saturating_add(chunk.token_ids.len() as u32);
                                        let mut out = LLMEngineOutput {
                                            token_ids: chunk.token_ids,
                                            ..Default::default()
                                        };
                                        out.index = Some(chunk.index);
                                        yield Ok(out);
                                    }
                                    Some(pb::generate_response::Response::Complete(done)) => {
                                        if done.prompt_tokens != 0 {
                                            prompt_tokens = done.prompt_tokens;
                                        }
                                        let mut out = finish_output(
                                            &done.finish_reason,
                                            prompt_tokens,
                                            generated,
                                        );
                                        out.index = Some(done.index);
                                        yield Ok(out);
                                        break;
                                    }
                                    None => continue,
                                }
                            }
                            Ok(None) => {
                                let err = client::engine_shutdown(
                                    "SMG vLLM engine closed the Generate stream before a complete event",
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
            request_ids: vec![ctx.id().to_string()],
        };
        if let Err(status) = client.abort(req).await {
            tracing::debug!(
                error = %status.message(),
                request_id = ctx.id(),
                "vllm SMG sidecar: abort RPC failed (ignored)"
            );
        }
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
                            "vllm SMG sidecar fatal watcher closed",
                        ));
                    }
                }
                _ = interval.tick() => {
                    match tokio::time::timeout(
                        self.transport.connect_timeout,
                        client.health_check(pb::HealthCheckRequest {}),
                    )
                    .await
                    {
                        Ok(Ok(resp)) => {
                            let resp = resp.into_inner();
                            if !resp.healthy {
                                let message = if resp.message.is_empty() {
                                    "SMG HealthCheck returned unhealthy".to_string()
                                } else {
                                    format!("SMG HealthCheck returned unhealthy: {}", resp.message)
                                };
                                return Err(client::engine_shutdown(message));
                            }
                        }
                        Ok(Err(status)) => {
                            return Err(client::engine_shutdown(format!(
                                "SMG HealthCheck RPC failed after startup: {} ({:?})",
                                status.message(),
                                status.code(),
                            )));
                        }
                        Err(_) => {
                            return Err(client::engine_shutdown(format!(
                                "SMG HealthCheck RPC timed out after {:?}",
                                self.transport.connect_timeout,
                            )));
                        }
                    }
                }
            }
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.cancel.cancel();
        tracing::info!("vllm SMG sidecar: cleanup invoked");
        Ok(())
    }

    async fn health_check_payload(&self) -> Result<Option<serde_json::Value>, DynamoError> {
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
        let mut client = client::connect(endpoint, transport).await?;
        client::discover(&mut client).await
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
            "SMG vLLM sidecar supports generation models only; model `{}` reports is_generation=false",
            discovery.model.model_path
        )));
    }

    match discovery.server.kv_role.as_str() {
        "" | "kv_both" => Ok(()),
        "kv_producer" | "kv_consumer" => Err(client::invalid_arg(format!(
            "SMG vLLM sidecar v1 supports aggregated serving only; engine reported kv_role={}",
            discovery.server.kv_role
        ))),
        other => Err(client::invalid_arg(format!(
            "SMG vLLM sidecar v1 does not understand kv_role={other:?}"
        ))),
    }
}

fn build_engine_config(discovery: &Discovery) -> EngineConfig {
    let model = &discovery.model;
    let served_model_name =
        (!model.served_model_name.is_empty()).then(|| model.served_model_name.clone());
    let context_length = if model.max_context_length != 0 {
        Some(model.max_context_length)
    } else if model.max_req_input_len > 0 {
        Some(model.max_req_input_len as u32)
    } else {
        None
    };

    EngineConfig {
        model: model.model_path.clone(),
        served_model_name,
        context_length,
        data_parallel_size: (discovery.server.data_parallel_size > 0)
            .then_some(discovery.server.data_parallel_size as u32),
        // SMG vLLM does not expose stable KV block capacity / block size in
        // the upstream servicer, so leave KV-aware scheduling hints unset.
        kv_cache_block_size: None,
        total_kv_blocks: None,
        max_num_seqs: None,
        max_num_batched_tokens: None,
        data_parallel_start_rank: None,
        bootstrap_host: None,
        bootstrap_port: None,
        runtime_data: Default::default(),
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
        "SMG Generate RPC failed: {} ({:?})",
        status.message(),
        status.code(),
    ))
}

// ============================================================================
// Request building + response mapping
// ============================================================================

pub(crate) fn validate_supported_request(request: &PreprocessedRequest) -> Result<(), DynamoError> {
    if request.prompt_embeds.is_some() {
        return Err(unsupported("prompt embeddings"));
    }
    if request.multi_modal_data.is_some() || request.mm_processor_kwargs.is_some() {
        return Err(unsupported("multimodal execution"));
    }
    if request.prefill_result.is_some() || request.bootstrap_info.is_some() {
        return Err(unsupported("disaggregated prefill/decode handoff"));
    }
    if request.output_options.logprobs.is_some() || request.output_options.prompt_logprobs.is_some()
    {
        return Err(unsupported("logprobs"));
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
        if guided.backend.is_some() {
            return Err(unsupported("guided decoding backend override"));
        }
        if guided.whitespace_pattern.is_some() {
            return Err(unsupported("guided decoding whitespace_pattern"));
        }
    }
    if let Some(routing) = request.routing.as_ref() {
        if routing.dp_rank.is_some() || routing.prefill_dp_rank.is_some() {
            return Err(unsupported("KV-aware DP-rank routing"));
        }
    }

    Ok(())
}

fn unsupported(feature: &str) -> DynamoError {
    client::invalid_arg(format!(
        "SMG vLLM sidecar v1 supports aggregated text/token generation only; unsupported feature: {feature}"
    ))
}

pub(crate) fn build_generate_request(
    request: &PreprocessedRequest,
    request_id: &str,
) -> Result<pb::GenerateRequest, DynamoError> {
    validate_supported_request(request)?;

    Ok(pb::GenerateRequest {
        request_id: request_id.to_string(),
        input: Some(pb::generate_request::Input::Tokenized(pb::TokenizedInput {
            original_text: String::new(),
            input_ids: request.token_ids.clone(),
        })),
        sampling_params: Some(build_sampling_params(request)?),
        stream: true,
        kv_transfer_params: None,
        mm_inputs: None,
        kv_transfer_params_json: None,
        data_parallel_rank: None,
    })
}

fn build_sampling_params(request: &PreprocessedRequest) -> Result<pb::SamplingParams, DynamoError> {
    let sampling = &request.sampling_options;
    let stop = &request.stop_conditions;

    let seed = match sampling.seed {
        Some(seed) => Some(i32::try_from(seed).map_err(|_| {
            client::invalid_arg(format!(
                "SMG seed is int32 but request seed {seed} is outside int32 range"
            ))
        })?),
        None => None,
    };

    let mut stop_token_ids = Vec::new();
    if let Some(ids) = &stop.stop_token_ids {
        stop_token_ids.extend(ids.iter().copied());
    }
    if let Some(ids) = &stop.stop_token_ids_hidden {
        stop_token_ids.extend(ids.iter().copied());
    }

    Ok(pb::SamplingParams {
        temperature: sampling.temperature,
        top_p: sampling.top_p.unwrap_or(0.0),
        top_k: sampling.top_k.filter(|v| *v > 0).unwrap_or(0) as u32,
        min_p: sampling.min_p.unwrap_or(0.0),
        frequency_penalty: sampling.frequency_penalty.unwrap_or(0.0),
        presence_penalty: sampling.presence_penalty.unwrap_or(0.0),
        repetition_penalty: sampling.repetition_penalty.unwrap_or(0.0),
        max_tokens: stop.max_tokens,
        min_tokens: stop.min_tokens.unwrap_or(0),
        stop: stop.stop.clone().unwrap_or_default(),
        stop_token_ids,
        skip_special_tokens: request.output_options.skip_special_tokens.unwrap_or(true),
        spaces_between_special_tokens: true,
        ignore_eos: stop.ignore_eos.unwrap_or(false),
        n: sampling.n.map(u32::from).unwrap_or(1),
        logprobs: None,
        prompt_logprobs: None,
        seed,
        include_stop_str_in_output: sampling.include_stop_str_in_output.unwrap_or(false),
        logit_bias: Default::default(),
        truncate_prompt_tokens: None,
        constraint: build_guided_constraint(sampling)?,
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
    if let Some(choice) = guided.choice.as_ref()
        && !choice.is_empty()
    {
        return Ok(Some(pb::sampling_params::Constraint::Choice(
            pb::ChoiceConstraint {
                choices: choice.clone(),
            },
        )));
    }
    if let Some(grammar) = guided.grammar.as_ref() {
        return Ok(Some(pb::sampling_params::Constraint::Grammar(
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
