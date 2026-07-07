// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo backend for SGLang's native `sglang.runtime.v1` gRPC server.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, DisaggregationMode, DynamoError, EngineConfig, GenerateContext, LLMEngine,
    LLMEngineOutput, LLMEngineOutputExt, LlmRegistration, ModelInput, PreprocessedRequest,
    StopReason, TopLogprob, WorkerConfig, usage,
};
use futures::stream::BoxStream;
use serde_json::Value;
use tokio::sync::OnceCell;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::args::{Args, TransportConfig, normalize_endpoint};
use crate::client::{self, Client, Discovery, Pool};
use crate::proto as pb;

pub struct SglangSidecarEngine {
    endpoint: String,
    transport: TransportConfig,
    disaggregation_mode: DisaggregationMode,
    bootstrap_host: Option<String>,
    bootstrap_port: Option<u16>,
    pool: OnceCell<Pool>,
    cancel: CancellationToken,
}

impl SglangSidecarEngine {
    pub(crate) fn new(
        endpoint: impl Into<String>,
        transport: TransportConfig,
        disaggregation_mode: DisaggregationMode,
        bootstrap_host: Option<String>,
        bootstrap_port: Option<u16>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            transport,
            disaggregation_mode,
            bootstrap_host,
            bootstrap_port,
            pool: OnceCell::new(),
            cancel: CancellationToken::new(),
        }
    }

    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = match argv {
            Some(args) => <Args as clap::Parser>::try_parse_from(args),
            None => <Args as clap::Parser>::try_parse(),
        }
        .map_err(|err| client::invalid_arg(err.to_string()))?;

        let endpoint = normalize_endpoint(&args.sglang_endpoint);
        let transport = args.transport();
        let discovery = bootstrap_discover(&endpoint, &transport)?;
        let disaggregation_mode = discovery_mode(&discovery)?;
        let bootstrap_host = if disaggregation_mode.is_prefill() {
            resolve_bootstrap_host(args.bootstrap_host.as_deref(), &endpoint, &discovery)?
        } else {
            None
        };
        let bootstrap_port = if disaggregation_mode.is_prefill() {
            discovery_bootstrap_port(&discovery)?
        } else {
            None
        };

        tracing::info!(
            %endpoint,
            mode = ?disaggregation_mode,
            model = %discovery.model_path,
            "sglang sidecar bootstrapped native gRPC discovery"
        );

        let config = WorkerConfig {
            namespace: args.namespace,
            component: component_for_mode(disaggregation_mode).to_string(),
            endpoint: args.endpoint,
            endpoint_types: args.endpoint_types,
            custom_jinja_template: args.custom_jinja_template,
            disaggregation_mode,
            model_name: discovery.model_path.clone(),
            served_model_name: discovery.served_model_name.clone(),
            model_input: ModelInput::Tokens,
            reasoning_parser: discovery_string(&discovery.server_info, "reasoning_parser"),
            tool_call_parser: discovery_string(&discovery.server_info, "tool_call_parser"),
            ..Default::default()
        };

        Ok((
            Self::new(
                endpoint,
                transport,
                disaggregation_mode,
                bootstrap_host,
                bootstrap_port,
            ),
            config,
        ))
    }

    async fn await_ready(&self, client: &mut Client) -> Result<(), DynamoError> {
        let deadline = Instant::now() + self.transport.deadline;
        loop {
            let retry_message = match client.health_check(pb::HealthCheckRequest {}).await {
                Ok(response) => {
                    if response.into_inner().healthy {
                        return Ok(());
                    }
                    "SGLang reported unhealthy".to_string()
                }
                Err(status) => format!("HealthCheck RPC failed: {}", status.message()),
            };
            if Instant::now() >= deadline {
                return Err(client::engine_shutdown(format!(
                    "SGLang did not become healthy within {:?}: {retry_message}",
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
        let observed_mode = discovery_mode(&discovery)?;
        if observed_mode != self.disaggregation_mode {
            return Err(client::invalid_arg(format!(
                "SGLang role changed since bootstrap: registered as {:?}, now reports {:?}",
                self.disaggregation_mode, observed_mode
            )));
        }

        let config = build_engine_config(
            &discovery,
            self.disaggregation_mode,
            self.bootstrap_host.clone(),
            self.bootstrap_port,
        )?;
        let connection_count = pool.len();
        self.pool
            .set(pool)
            .map_err(|_| client::engine_shutdown("sglang sidecar already started"))?;
        tracing::info!(
            model = %config.model,
            mode = ?self.disaggregation_mode,
            connections = connection_count,
            "sglang sidecar started"
        );
        Ok(config)
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let mut grpc_client = self
            .pool
            .get()
            .map(Pool::stream_client)
            .ok_or_else(|| client::engine_shutdown("generate called before start"))?;

        let prompt_tokens = request.token_ids.len() as u32;
        let return_tokens_as_ids = request
            .output_options
            .return_tokens_as_token_ids
            .unwrap_or(false);
        let grpc_request = build_generate_request(
            &request,
            ctx.id(),
            self.disaggregation_mode,
            self.bootstrap_host.as_deref(),
            self.bootstrap_port,
        )?;
        let prefill_handoff = if self.disaggregation_mode.is_prefill() {
            grpc_request
                .disaggregated_params
                .as_ref()
                .map(disaggregated_params_to_json)
        } else {
            None
        };
        let cancel = self.cancel.clone();
        let is_prefill = self.disaggregation_mode.is_prefill();

        Ok(Box::pin(async_stream::stream! {
            if ctx.is_stopped() || cancel.is_cancelled() {
                yield Ok(LLMEngineOutput::cancelled().with_usage(usage(prompt_tokens, 0)));
                return;
            }

            tracing::debug!(request_id = %ctx.id(), "sending request to SGLang gRPC");
            let opened = tokio::select! {
                biased;
                _ = ctx.stopped() => None,
                _ = cancel.cancelled() => None,
                response = grpc_client.generate(grpc_request) => Some(response),
            };
            let Some(opened) = opened else {
                yield Ok(LLMEngineOutput::cancelled().with_usage(usage(prompt_tokens, 0)));
                return;
            };
            let mut stream = match opened {
                Ok(response) => response.into_inner(),
                Err(status) => {
                    yield Err(client::status_to_dynamo("Generate", status));
                    return;
                }
            };

            let mut generated = 0_u32;
            let mut observed_prompt_tokens = prompt_tokens;
            let mut logprob_offset = 0_usize;
            loop {
                tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        yield Ok(LLMEngineOutput::cancelled()
                            .with_usage(usage(observed_prompt_tokens, generated)));
                        break;
                    }
                    _ = cancel.cancelled() => {
                        yield Ok(LLMEngineOutput::cancelled()
                            .with_usage(usage(observed_prompt_tokens, generated)));
                        break;
                    }
                    message = stream.message() => {
                        let response = match message {
                            Ok(Some(response)) => response,
                            Ok(None) => {
                                yield Err(client::engine_shutdown(
                                    "SGLang closed Generate before a finished response",
                                ));
                                break;
                            }
                            Err(status) => {
                                yield Err(client::status_to_dynamo("Generate", status));
                                break;
                            }
                        };

                        if let Some(value) = meta_u32(&response.meta_info, "prompt_tokens") {
                            observed_prompt_tokens = value;
                        }
                        let token_ids = match output_ids_to_u32(&response.output_ids) {
                            Ok(ids) => ids,
                            Err(err) => {
                                yield Err(err);
                                break;
                            }
                        };
                        let (log_probs, top_logprobs, next_offset) =
                            match extract_logprobs(
                                &response.meta_info,
                                logprob_offset,
                                return_tokens_as_ids,
                            ) {
                                Ok(values) => values,
                                Err(err) => {
                                    yield Err(err);
                                    break;
                                }
                            };
                        logprob_offset = next_offset;

                        if is_prefill {
                            if response.finished {
                                let mut terminal = terminal_from_meta(
                                    &response.meta_info,
                                    observed_prompt_tokens,
                                    0,
                                );
                                terminal.disaggregated_params = prefill_handoff.clone();
                                yield Ok(terminal);
                                break;
                            }
                            continue;
                        }

                        generated = generated.saturating_add(token_ids.len() as u32);
                        if response.finished {
                            let mut terminal = terminal_from_meta(
                                &response.meta_info,
                                observed_prompt_tokens,
                                generated,
                            );
                            terminal.token_ids = token_ids;
                            terminal.log_probs = log_probs;
                            terminal.top_logprobs = top_logprobs;
                            terminal.engine_data = engine_data_from_meta(&response.meta_info);
                            yield Ok(terminal);
                            break;
                        }

                        if !token_ids.is_empty() {
                            yield Ok(LLMEngineOutput {
                                token_ids,
                                log_probs,
                                top_logprobs,
                                engine_data: engine_data_from_meta(&response.meta_info),
                                ..Default::default()
                            });
                        }
                    }
                }
            }
        }))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        let Some(mut grpc_client) = self.pool.get().map(Pool::control_client) else {
            return;
        };
        let request = pb::AbortRequest {
            rid: ctx.id().to_string(),
            abort_all: false,
        };
        if let Err(status) = grpc_client.abort(request).await {
            tracing::debug!(
                request_id = ctx.id(),
                error = %status.message(),
                "SGLang Abort RPC failed"
            );
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.cancel.cancel();
        tracing::info!("sglang sidecar shutdown complete");
        Ok(())
    }
}

fn bootstrap_discover(
    endpoint: &str,
    transport: &TransportConfig,
) -> Result<Discovery, DynamoError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| client::engine_shutdown(format!("bootstrap runtime: {err}")))?;
    runtime.block_on(async {
        let mut grpc_client = client::connect(endpoint, transport).await?;
        client::discover(&mut grpc_client).await
    })
}

fn discovery_mode(discovery: &Discovery) -> Result<DisaggregationMode, DynamoError> {
    match discovery
        .server_info
        .get("disaggregation_mode")
        .and_then(Value::as_str)
        .unwrap_or("null")
    {
        "null" | "agg" | "aggregated" => Ok(DisaggregationMode::Aggregated),
        "prefill" => Ok(DisaggregationMode::Prefill),
        "decode" => Ok(DisaggregationMode::Decode),
        mode => Err(client::protocol_error(format!(
            "unsupported SGLang disaggregation_mode `{mode}`"
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

fn discovery_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|entry| !entry.is_empty())
        .map(str::to_string)
}

fn discovery_bootstrap_port(discovery: &Discovery) -> Result<Option<u16>, DynamoError> {
    client::json_u64(&discovery.server_info, "disaggregation_bootstrap_port")
        .map(|port| {
            u16::try_from(port).map_err(|_| {
                client::protocol_error(format!(
                    "SGLang disaggregation_bootstrap_port is out of range: {port}"
                ))
            })
        })
        .transpose()
        .and_then(|port| {
            port.filter(|port| *port != 0).map_or_else(
                || {
                    Err(client::protocol_error(
                        "prefill SGLang server did not report disaggregation_bootstrap_port",
                    ))
                },
                |port| Ok(Some(port)),
            )
        })
}

fn resolve_bootstrap_host(
    explicit: Option<&str>,
    endpoint: &str,
    discovery: &Discovery,
) -> Result<Option<String>, DynamoError> {
    if let Some(host) = explicit.filter(|host| !host.trim().is_empty()) {
        return Ok(Some(host.trim().to_string()));
    }
    let from_dist = discovery
        .server_info
        .get("dist_init_addr")
        .and_then(Value::as_str)
        .and_then(host_from_address);
    let from_endpoint = url::Url::parse(endpoint)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string));
    from_dist.or(from_endpoint).map(Some).ok_or_else(|| {
        client::invalid_arg("could not derive a prefill bootstrap host; set --bootstrap-host")
    })
}

fn host_from_address(address: &str) -> Option<String> {
    let candidate = if address.contains("://") {
        address.to_string()
    } else {
        format!("tcp://{address}")
    };
    url::Url::parse(&candidate)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
}

fn build_engine_config(
    discovery: &Discovery,
    mode: DisaggregationMode,
    bootstrap_host: Option<String>,
    bootstrap_port: Option<u16>,
) -> Result<EngineConfig, DynamoError> {
    let page_size = client::json_u32(&discovery.server_info, "page_size");
    let max_total_tokens = client::json_u64(&discovery.server_info, "max_total_num_tokens");
    let total_kv_blocks = match (max_total_tokens, page_size) {
        (Some(tokens), Some(page_size)) if page_size > 0 => {
            Some(tokens.saturating_add(u64::from(page_size) - 1) / u64::from(page_size))
        }
        _ => None,
    };
    let dp_size = client::json_u32(&discovery.server_info, "dp_size")
        .unwrap_or(1)
        .max(1);
    let max_num_seqs =
        client::json_u64(&discovery.server_info, "max_running_requests").map(|value| {
            if dp_size > 1 {
                value / u64::from(dp_size)
            } else {
                value
            }
        });
    let max_num_batched_tokens =
        client::json_u64(&discovery.server_info, "max_prefill_tokens").or(max_total_tokens);

    let enable_dp_attention = discovery
        .server_info
        .get("enable_dp_attention")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let nnodes = client::json_u32(&discovery.server_info, "nnodes")
        .unwrap_or(1)
        .max(1);
    let node_rank = client::json_u32(&discovery.server_info, "node_rank").unwrap_or(0);
    let (data_parallel_start_rank, data_parallel_size) = if enable_dp_attention && dp_size > 1 {
        let local_size = (dp_size / nnodes).max(1);
        (Some(node_rank.saturating_mul(local_size)), Some(local_size))
    } else {
        (Some(0), Some(1))
    };

    if mode.is_prefill() && (bootstrap_host.is_none() || bootstrap_port.is_none()) {
        return Err(client::protocol_error(
            "prefill SGLang discovery did not provide a usable bootstrap address",
        ));
    }

    let mut runtime_data = HashMap::new();
    runtime_data.insert(
        "grpc_service".to_string(),
        Value::String("sglang.runtime.v1.SglangService".to_string()),
    );

    Ok(EngineConfig {
        model: discovery.model_path.clone(),
        served_model_name: discovery.served_model_name.clone(),
        runtime_data,
        llm: Some(LlmRegistration {
            context_length: discovery.max_model_len,
            kv_cache_block_size: page_size,
            total_kv_blocks,
            max_num_seqs,
            max_num_batched_tokens,
            data_parallel_size,
            data_parallel_start_rank,
            bootstrap_host: mode.is_prefill().then_some(bootstrap_host).flatten(),
            bootstrap_port: mode.is_prefill().then_some(bootstrap_port).flatten(),
        }),
    })
}

pub(crate) fn build_generate_request(
    request: &PreprocessedRequest,
    request_id: &str,
    mode: DisaggregationMode,
    bootstrap_host: Option<&str>,
    bootstrap_port: Option<u16>,
) -> Result<pb::GenerateRequest, DynamoError> {
    validate_request(request)?;
    let input_ids = request
        .token_ids
        .iter()
        .map(|token| {
            i32::try_from(*token)
                .map_err(|_| client::invalid_arg(format!("token id {token} does not fit in i32")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let max_new_tokens = if mode.is_prefill() {
        Some(1)
    } else {
        request
            .stop_conditions
            .max_tokens
            .map(i32::try_from)
            .transpose()
            .map_err(|_| client::invalid_arg("max_tokens does not fit in i32"))?
    };
    let min_new_tokens = request
        .stop_conditions
        .min_tokens
        .map(i32::try_from)
        .transpose()
        .map_err(|_| client::invalid_arg("min_tokens does not fit in i32"))?;

    let mut stop_token_ids = Vec::new();
    for tokens in [
        request.stop_conditions.stop_token_ids.as_ref(),
        request.stop_conditions.stop_token_ids_hidden.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        for token in tokens {
            let token = i32::try_from(*token).map_err(|_| {
                client::invalid_arg(format!("stop token id {token} does not fit in i32"))
            })?;
            if !stop_token_ids.contains(&token) {
                stop_token_ids.push(token);
            }
        }
    }

    let guided = request.sampling_options.guided_decoding.as_ref();
    let sampling_params = pb::SamplingParams {
        temperature: request.sampling_options.temperature,
        top_p: request.sampling_options.top_p,
        top_k: request.sampling_options.top_k,
        min_p: request.sampling_options.min_p,
        frequency_penalty: request.sampling_options.frequency_penalty,
        presence_penalty: request.sampling_options.presence_penalty,
        repetition_penalty: request.sampling_options.repetition_penalty,
        max_new_tokens,
        min_new_tokens,
        stop: request.stop_conditions.stop.clone().unwrap_or_default(),
        stop_token_ids,
        ignore_eos: request.stop_conditions.ignore_eos,
        n: request.sampling_options.n.map(i32::from),
        json_schema: guided
            .and_then(|value| value.json.as_ref())
            .map(json_value_to_string),
        regex: guided.and_then(|value| value.regex.clone()),
    };

    let output_options = &request.output_options;
    let return_logprob =
        output_options.logprobs.is_some() || output_options.prompt_logprobs.is_some();
    let top_logprobs_num = output_options
        .logprobs
        .unwrap_or(0)
        .max(output_options.prompt_logprobs.unwrap_or(0));
    let top_logprobs_num = i32::try_from(top_logprobs_num)
        .map_err(|_| client::invalid_arg("requested logprobs does not fit in i32"))?;
    let logprob_start_len = output_options.prompt_logprobs.map(|_| 0).unwrap_or(-1);
    let routed_dp_rank = request
        .routing
        .as_ref()
        .and_then(|routing| routing.dp_rank)
        .map(i32::try_from)
        .transpose()
        .map_err(|_| client::invalid_arg("routed dp_rank does not fit in i32"))?;
    let lora_path = request
        .routing
        .as_ref()
        .and_then(|routing| routing.lora_name.clone());

    let mut trace_headers = HashMap::new();
    dynamo_runtime::logging::inject_trace_headers_into_map(&mut trace_headers);

    Ok(pb::GenerateRequest {
        input_ids,
        sampling_params: Some(sampling_params),
        stream: Some(true),
        return_logprob: Some(return_logprob),
        top_logprobs_num: Some(top_logprobs_num),
        logprob_start_len: Some(logprob_start_len),
        rid: Some(request_id.to_string()),
        lora_path,
        routing_key: request.mdc_sum.clone(),
        routed_dp_rank,
        trace_headers,
        session_id: None,
        disaggregated_params: resolve_disaggregated_params(
            request,
            mode,
            bootstrap_host,
            bootstrap_port,
        )?,
    })
}

fn validate_request(request: &PreprocessedRequest) -> Result<(), DynamoError> {
    if request.token_ids.is_empty() {
        return Err(client::invalid_arg("token_ids must not be empty"));
    }
    if request.prompt_embeds.is_some() {
        return Err(client::invalid_arg(
            "prompt_embeds are not supported by SGLang's native gRPC proto",
        ));
    }
    if request.multi_modal_data.is_some() || request.mm_processor_kwargs.is_some() {
        return Err(client::invalid_arg(
            "multimodal payloads are not supported by SGLang's native Generate RPC",
        ));
    }
    if request.sampling_options.n.unwrap_or(1) != 1 {
        return Err(client::invalid_arg("n must be 1 for the SGLang sidecar"));
    }
    if request.sampling_options.best_of.unwrap_or(1) != 1 {
        return Err(client::invalid_arg(
            "best_of is not represented by SGLang's native gRPC proto",
        ));
    }
    if request.sampling_options.use_beam_search.unwrap_or(false) {
        return Err(client::invalid_arg(
            "beam search is not represented by SGLang's native gRPC proto",
        ));
    }
    if let Some(penalty) = request.sampling_options.length_penalty
        && (penalty - 1.0).abs() > f32::EPSILON
    {
        return Err(client::invalid_arg(
            "length_penalty is not represented by SGLang's native gRPC proto",
        ));
    }
    if request.sampling_options.seed.is_some() {
        return Err(client::invalid_arg(
            "seed is not represented by SGLang's native gRPC proto",
        ));
    }
    if request.stop_conditions.max_thinking_tokens.is_some() {
        return Err(client::invalid_arg(
            "max_thinking_tokens is not represented by SGLang's native gRPC proto",
        ));
    }
    if request
        .sampling_options
        .include_stop_str_in_output
        .unwrap_or(false)
    {
        return Err(client::invalid_arg(
            "include_stop_str_in_output is not represented by SGLang's native gRPC proto",
        ));
    }
    if request
        .stop_conditions
        .stop_token_ids_visible
        .as_ref()
        .is_some_and(|tokens| !tokens.is_empty())
    {
        return Err(client::invalid_arg(
            "visible stop-token semantics are not represented by SGLang's native gRPC proto",
        ));
    }
    if let Some(guided) = request.sampling_options.guided_decoding.as_ref()
        && (guided
            .choice
            .as_ref()
            .is_some_and(|value| !value.is_empty())
            || guided.grammar.is_some()
            || guided
                .backend
                .as_ref()
                .is_some_and(|value| !value.is_empty())
            || guided.whitespace_pattern.is_some()
            || guided.structural_tag.is_some())
    {
        return Err(client::invalid_arg(
            "the native SGLang gRPC proto currently supports only JSON-schema and regex guided decoding",
        ));
    }
    if request
        .routing
        .as_ref()
        .and_then(|routing| routing.priority)
        .unwrap_or(0)
        != 0
    {
        return Err(client::invalid_arg(
            "engine priority is not represented by SGLang's native gRPC proto",
        ));
    }
    Ok(())
}

fn resolve_disaggregated_params(
    request: &PreprocessedRequest,
    mode: DisaggregationMode,
    bootstrap_host: Option<&str>,
    bootstrap_port: Option<u16>,
) -> Result<Option<pb::DisaggregatedParams>, DynamoError> {
    if mode == DisaggregationMode::Aggregated {
        return Ok(None);
    }
    if let Some(info) = request.bootstrap_info.as_ref() {
        return bootstrap_values_to_proto(
            &info.bootstrap_host,
            u64::from(info.bootstrap_port),
            info.bootstrap_room,
        )
        .map(Some);
    }
    if let Some(prefill) = request.prefill_result.as_ref() {
        return disaggregated_json_to_proto(&prefill.disaggregated_params).map(Some);
    }
    if mode.is_prefill() {
        let host = bootstrap_host.ok_or_else(|| {
            client::invalid_arg("prefill request has no bootstrap host from discovery")
        })?;
        let port = bootstrap_port.ok_or_else(|| {
            client::invalid_arg("prefill request has no bootstrap port from discovery")
        })?;
        let room = rand::random::<u64>() & (i64::MAX as u64);
        return bootstrap_values_to_proto(host, u64::from(port), room).map(Some);
    }
    Err(client::invalid_arg(
        "decode request has neither bootstrap_info nor prefill_result",
    ))
}

fn disaggregated_json_to_proto(value: &Value) -> Result<pb::DisaggregatedParams, DynamoError> {
    let host = value
        .get("bootstrap_host")
        .and_then(Value::as_str)
        .ok_or_else(|| client::invalid_arg("disaggregated_params.bootstrap_host is missing"))?;
    let port = value
        .get("bootstrap_port")
        .and_then(Value::as_u64)
        .ok_or_else(|| client::invalid_arg("disaggregated_params.bootstrap_port is missing"))?;
    let room = value
        .get("bootstrap_room")
        .and_then(Value::as_u64)
        .ok_or_else(|| client::invalid_arg("disaggregated_params.bootstrap_room is missing"))?;
    bootstrap_values_to_proto(host, port, room)
}

fn bootstrap_values_to_proto(
    host: &str,
    port: u64,
    room: u64,
) -> Result<pb::DisaggregatedParams, DynamoError> {
    if host.trim().is_empty() {
        return Err(client::invalid_arg("bootstrap_host must not be empty"));
    }
    let bootstrap_port = i32::try_from(port)
        .map_err(|_| client::invalid_arg(format!("bootstrap_port is out of range: {port}")))?;
    let bootstrap_room = i64::try_from(room).map_err(|_| {
        client::invalid_arg(format!(
            "bootstrap_room must fit SGLang's signed int64 field: {room}"
        ))
    })?;
    Ok(pb::DisaggregatedParams {
        bootstrap_host: host.to_string(),
        bootstrap_port,
        bootstrap_room,
    })
}

fn disaggregated_params_to_json(params: &pb::DisaggregatedParams) -> Value {
    serde_json::json!({
        "bootstrap_host": params.bootstrap_host,
        "bootstrap_port": params.bootstrap_port,
        "bootstrap_room": params.bootstrap_room,
    })
}

fn json_value_to_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        value => value.to_string(),
    }
}

fn output_ids_to_u32(ids: &[i32]) -> Result<Vec<u32>, DynamoError> {
    ids.iter()
        .map(|id| {
            u32::try_from(*id).map_err(|_| {
                client::protocol_error(format!("SGLang returned a negative token id: {id}"))
            })
        })
        .collect()
}

fn meta_value(meta: &HashMap<String, String>, key: &str) -> Option<Value> {
    meta.get(key)
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
}

fn meta_u32(meta: &HashMap<String, String>, key: &str) -> Option<u32> {
    meta_value(meta, key)
        .and_then(|value| value.as_u64())
        .and_then(|value| u32::try_from(value).ok())
}

fn terminal_from_meta(
    meta: &HashMap<String, String>,
    prompt_tokens: u32,
    generated: u32,
) -> LLMEngineOutput {
    let finish = meta_value(meta, "finish_reason").unwrap_or(Value::Null);
    let finish_type = finish
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| finish.as_str())
        .unwrap_or("stop");
    let mut output = match finish_type {
        "length" => LLMEngineOutput::length(),
        "abort" | "cancelled" => LLMEngineOutput::cancelled(),
        "error" => LLMEngineOutput::error(
            finish
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("SGLang generation error")
                .to_string(),
        ),
        _ => LLMEngineOutput::stop(),
    }
    .with_usage(usage(prompt_tokens, generated));
    output.stop_reason = finish.get("matched").and_then(|matched| match matched {
        Value::String(value) => Some(StopReason::String(value.clone())),
        Value::Number(value) => value.as_i64().map(StopReason::Int),
        _ => None,
    });
    output
}

fn engine_data_from_meta(meta: &HashMap<String, String>) -> Option<Value> {
    meta_value(meta, "routed_experts")
        .map(|routed_experts| serde_json::json!({"routed_experts": routed_experts}))
}

type ExtractedLogprobs = (Option<Vec<f64>>, Option<Vec<Vec<TopLogprob>>>, usize);

fn extract_logprobs(
    meta: &HashMap<String, String>,
    offset: usize,
    return_tokens_as_ids: bool,
) -> Result<ExtractedLogprobs, DynamoError> {
    let Some(Value::Array(all_logprobs)) = meta_value(meta, "output_token_logprobs") else {
        return Ok((None, None, offset));
    };
    if offset >= all_logprobs.len() {
        return Ok((None, None, all_logprobs.len()));
    }

    let mut log_probs = Vec::with_capacity(all_logprobs.len() - offset);
    for entry in &all_logprobs[offset..] {
        let value = entry
            .as_array()
            .and_then(|parts| parts.first())
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                client::protocol_error("invalid output_token_logprobs entry from SGLang")
            })?;
        log_probs.push(value);
    }

    let top_logprobs = match meta_value(meta, "output_top_logprobs") {
        Some(Value::Array(all_top)) => {
            let mut positions = Vec::new();
            for position in all_top.iter().skip(offset) {
                let Some(entries) = position.as_array() else {
                    positions.push(Vec::new());
                    continue;
                };
                let mut mapped = Vec::with_capacity(entries.len());
                for (index, entry) in entries.iter().enumerate() {
                    let parts = entry.as_array().ok_or_else(|| {
                        client::protocol_error("invalid output_top_logprobs entry from SGLang")
                    })?;
                    let logprob = parts.first().and_then(Value::as_f64).ok_or_else(|| {
                        client::protocol_error("missing top-logprob value from SGLang")
                    })?;
                    let token_id = parts.get(1).and_then(Value::as_u64).ok_or_else(|| {
                        client::protocol_error("missing top-logprob token id from SGLang")
                    })?;
                    let token_id = u32::try_from(token_id).map_err(|_| {
                        client::protocol_error("top-logprob token id does not fit u32")
                    })?;
                    let token = if return_tokens_as_ids {
                        Some(format!("token_id:{token_id}"))
                    } else {
                        parts.get(2).and_then(Value::as_str).map(str::to_string)
                    };
                    mapped.push(TopLogprob {
                        rank: u32::try_from(index + 1).unwrap_or(u32::MAX),
                        token_id,
                        token,
                        logprob,
                        bytes: None,
                    });
                }
                positions.push(mapped);
            }
            Some(positions)
        }
        _ => None,
    };

    Ok((Some(log_probs), top_logprobs, all_logprobs.len()))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_backend_common::{
        BootstrapInfo, DisaggregationMode, FinishReason, OutputOptions, PrefillResult,
        PreprocessedRequest, SamplingOptions, StopConditions,
    };
    use serde_json::json;

    use super::{
        build_generate_request, disaggregated_params_to_json, extract_logprobs, terminal_from_meta,
    };

    fn request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("Qwen/Qwen3-0.6B".to_string())
            .token_ids(vec![1, 2, 3])
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .stop_conditions(StopConditions {
                max_tokens: Some(8),
                ..Default::default()
            })
            .build()
            .unwrap()
    }

    #[test]
    fn request_maps_native_fields_and_full_width_room() {
        let mut request = request();
        request.bootstrap_info = Some(BootstrapInfo {
            bootstrap_host: "prefill".to_string(),
            bootstrap_port: 5000,
            bootstrap_room: i64::MAX as u64,
            handoff_id: None,
        });
        let mapped =
            build_generate_request(&request, "rid-1", DisaggregationMode::Decode, None, None)
                .unwrap();
        assert_eq!(mapped.input_ids, vec![1, 2, 3]);
        assert_eq!(mapped.rid.as_deref(), Some("rid-1"));
        assert_eq!(mapped.sampling_params.unwrap().max_new_tokens, Some(8));
        assert_eq!(
            mapped.disaggregated_params.unwrap().bootstrap_room,
            i64::MAX
        );
    }

    #[test]
    fn prefill_clamps_generation_and_creates_handoff() {
        let mapped = build_generate_request(
            &request(),
            "rid-2",
            DisaggregationMode::Prefill,
            Some("prefill"),
            Some(5001),
        )
        .unwrap();
        assert_eq!(mapped.sampling_params.unwrap().max_new_tokens, Some(1));
        assert_eq!(mapped.disaggregated_params.unwrap().bootstrap_port, 5001);
    }

    #[test]
    fn prefill_handoff_round_trips_to_decode_request() {
        let prefill = build_generate_request(
            &request(),
            "rid-prefill",
            DisaggregationMode::Prefill,
            Some("prefill.internal"),
            Some(5001),
        )
        .unwrap();
        let handoff = prefill.disaggregated_params.unwrap();

        let mut decode_request = request();
        decode_request.prefill_result = Some(PrefillResult {
            disaggregated_params: disaggregated_params_to_json(&handoff),
            prompt_tokens_details: None,
        });
        let decode = build_generate_request(
            &decode_request,
            "rid-decode",
            DisaggregationMode::Decode,
            None,
            None,
        )
        .unwrap();

        assert_eq!(decode.disaggregated_params, Some(handoff));
    }

    #[test]
    fn logprobs_are_sliced_from_cumulative_metadata() {
        let meta = HashMap::from([
            (
                "output_token_logprobs".to_string(),
                json!([[-0.1, 10, "a"], [-0.2, 11, "b"]]).to_string(),
            ),
            (
                "output_top_logprobs".to_string(),
                json!([[[-0.1, 10, "a"]], [[-0.2, 11, "b"]]]).to_string(),
            ),
        ]);
        let (logprobs, top, next) = extract_logprobs(&meta, 1, false).unwrap();
        assert_eq!(logprobs.unwrap(), vec![-0.2]);
        assert_eq!(top.unwrap()[0][0].token_id, 11);
        assert_eq!(next, 2);
    }

    #[test]
    fn terminal_maps_finish_reason_and_usage() {
        let meta = HashMap::from([(
            "finish_reason".to_string(),
            json!({"type": "length"}).to_string(),
        )]);
        let terminal = terminal_from_meta(&meta, 4, 3);
        assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
        assert_eq!(terminal.completion_usage.unwrap().total_tokens, 7);
    }

    #[test]
    fn decode_requires_rendezvous_params() {
        assert!(
            build_generate_request(&request(), "rid-3", DisaggregationMode::Decode, None, None,)
                .is_err()
        );
    }

    #[test]
    fn room_above_signed_int64_is_rejected() {
        let mut request = request();
        request.bootstrap_info = Some(BootstrapInfo {
            bootstrap_host: "prefill".to_string(),
            bootstrap_port: 5000,
            bootstrap_room: i64::MAX as u64 + 1,
            handoff_id: None,
        });
        assert!(
            build_generate_request(&request, "rid-4", DisaggregationMode::Decode, None, None,)
                .is_err()
        );
    }
}
