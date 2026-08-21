// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo backend for SGLang's native `sglang.runtime.v1` gRPC server.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, DisaggregationMode, DynamoError, EngineConfig, GenerateContext,
    KvEventSource, LLMEngine, LLMEngineOutput, LLMEngineOutputExt, LlmRegistration, ModelInput,
    PreprocessedRequest, WorkerConfig, usage,
};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use futures::stream::BoxStream;
use serde_json::Value;
use tokio::sync::OnceCell;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::args::Args;
use crate::client::{self, Client, Discovery, Pool};
use crate::native_http::{self, NativeHttp};
use crate::proto as pb;
use crate::protocol::{
    build_generate_request, disaggregated_params_to_json, engine_data_from_meta, extract_logprobs,
    meta_u32, output_ids_to_u32, terminal_from_meta,
};

pub struct SglangSidecarEngine {
    endpoint: GrpcEndpoint,
    transport: GrpcTransportConfig,
    disaggregation_mode: DisaggregationMode,
    bootstrap_host: Option<String>,
    bootstrap_port: Option<u16>,
    native_http: Option<NativeHttp>,
    state: OnceCell<StartedState>,
    cancel: CancellationToken,
}

struct StartedState {
    pool: Pool,
    kv_event_sources: Vec<DiscoveredKvEventSource>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DiscoveredKvEventSource {
    endpoint: String,
    topic: String,
    dp_rank: u32,
}

impl SglangSidecarEngine {
    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = match argv {
            Some(args) => <Args as clap::Parser>::try_parse_from(args)
                .map_err(|err| client::invalid_arg(err.to_string()))?,
            None => <Args as clap::Parser>::parse(),
        };

        if args.sidecar.common.route_to_encoder {
            return Err(client::invalid_arg(
                "route-to-encoder is not supported by the SGLang sidecar",
            ));
        }

        let endpoint = args.sidecar.grpc_endpoint;
        let transport = args.sidecar.grpc.config();
        let discovery = bootstrap_discover(&endpoint, &transport)?;
        let disaggregation_mode = discovery_mode(&discovery)?;
        let bootstrap_host = if disaggregation_mode.is_prefill() {
            resolve_bootstrap_host(
                args.bootstrap_host.as_deref(),
                endpoint.as_str(),
                &discovery,
            )?
        } else {
            None
        };
        let bootstrap_port = if disaggregation_mode.is_prefill() {
            discovery_bootstrap_port(&discovery)?
        } else {
            None
        };
        let native_http =
            NativeHttp::discover(&endpoint, &discovery, transport.connect_attempt_timeout)?;

        tracing::info!(
            %endpoint,
            mode = ?disaggregation_mode,
            model = %discovery.model_path,
            "sglang sidecar bootstrapped native gRPC discovery"
        );

        let common = args.sidecar.common;
        let config = WorkerConfig {
            namespace: common.namespace,
            component: if disaggregation_mode == DisaggregationMode::Aggregated {
                common.component
            } else {
                disaggregation_mode.discovery_component().to_string()
            },
            endpoint: common.endpoint,
            endpoint_types: common.endpoint_types,
            custom_jinja_template: common.custom_jinja_template,
            disaggregation_mode,
            model_name: discovery.tokenizer_path.clone(),
            served_model_name: discovery.served_model_name.clone(),
            model_input: ModelInput::Tokens,
            reasoning_parser: common
                .dyn_reasoning_parser
                .or_else(|| discovery_string(&discovery.server_info, "reasoning_parser")),
            tool_call_parser: common
                .dyn_tool_call_parser
                .or_else(|| discovery_string(&discovery.server_info, "tool_call_parser")),
            exclude_tools_when_tool_choice_none: common.exclude_tools_when_tool_choice_none,
            route_to_encoder: false,
            enable_rl: common.enable_rl,
            ..Default::default()
        };

        Ok((
            Self {
                endpoint,
                transport,
                disaggregation_mode,
                bootstrap_host,
                bootstrap_port,
                native_http,
                state: OnceCell::new(),
                cancel: CancellationToken::new(),
            },
            config,
        ))
    }

    async fn await_ready(&self, client: &mut Client, deadline: Instant) -> Result<(), DynamoError> {
        loop {
            let retry_message = match client::health_check(client, deadline).await {
                Ok(healthy) => {
                    if healthy {
                        return Ok(());
                    }
                    "SGLang reported unhealthy".to_string()
                }
                Err(error) => format!("HealthCheck RPC failed: {error}"),
            };
            if Instant::now() >= deadline {
                return Err(client::engine_shutdown(format!(
                    "SGLang did not become healthy within {:?}: {retry_message}",
                    self.transport.startup_deadline
                )));
            }
            tokio::time::sleep_until(
                (Instant::now() + self.transport.retry_interval).min(deadline),
            )
            .await;
        }
    }
}

#[async_trait]
impl LLMEngine for SglangSidecarEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.state.initialized() {
            return Err(client::engine_shutdown("sglang sidecar already started"));
        }

        let deadline = Instant::now() + self.transport.startup_deadline;
        let pool = Pool::connect(&self.endpoint, &self.transport, deadline).await?;
        let mut control = pool.control_client();
        self.await_ready(&mut control, deadline).await?;
        let discovery = client::discover(&mut control, deadline).await?;
        let observed_mode = discovery_mode(&discovery)?;
        if observed_mode != self.disaggregation_mode {
            return Err(client::invalid_arg(format!(
                "SGLang role changed since bootstrap: registered as {:?}, now reports {:?}",
                self.disaggregation_mode, observed_mode
            )));
        }

        let mut config = build_engine_config(
            &discovery,
            self.disaggregation_mode,
            self.bootstrap_host.clone(),
            self.bootstrap_port,
        )?;
        if self.native_http.is_some() {
            config
                .runtime_data
                .insert("sglang_generate".into(), true.into());
        }
        let kv_event_sources = discover_kv_event_sources(&discovery, &config, &self.endpoint)?;
        let connection_count = pool.len();
        let kv_event_source_count = kv_event_sources.len();
        self.state
            .set(StartedState {
                pool,
                kv_event_sources,
            })
            .map_err(|_| client::engine_shutdown("sglang sidecar already started"))?;
        tracing::info!(
            model = %config.model,
            mode = ?self.disaggregation_mode,
            connections = connection_count,
            kv_event_sources = kv_event_source_count,
            "sglang sidecar started"
        );
        Ok(config)
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let state = self
            .state
            .get()
            .ok_or_else(|| client::engine_shutdown("generate called before start"))?;
        if let Some(native_request) = native_http::request(
            &request,
            ctx.id(),
            self.disaggregation_mode,
            self.bootstrap_host.as_deref(),
            self.bootstrap_port,
        )? {
            let native_http = self.native_http.clone().ok_or_else(|| {
                client::invalid_arg(
                    "native SGLang Generate is unavailable because GetServerInfo did not report an HTTP port",
                )
            })?;
            return Ok(native_http.generate(native_request, ctx, self.cancel.clone()));
        }
        let mut grpc_client = state.pool.stream_client();

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
                        let (log_probs, top_logprobs) =
                            match extract_logprobs(&response.meta_info, return_tokens_as_ids) {
                                Ok(values) => values,
                                Err(err) => {
                                    yield Err(err);
                                    break;
                                }
                            };

                        if is_prefill {
                            if response.finished {
                                let mut terminal = match terminal_from_meta(
                                    &response.meta_info,
                                    observed_prompt_tokens,
                                    0,
                                ) {
                                    Ok(terminal) => terminal,
                                    Err(error) => {
                                        yield Err(error);
                                        break;
                                    }
                                };
                                terminal.disaggregated_params = prefill_handoff.clone();
                                yield Ok(terminal);
                                break;
                            }
                            continue;
                        }

                        generated = generated.saturating_add(token_ids.len() as u32);
                        if response.finished {
                            let mut terminal = match terminal_from_meta(
                                &response.meta_info,
                                observed_prompt_tokens,
                                generated,
                            ) {
                                Ok(terminal) => terminal,
                                Err(error) => {
                                    yield Err(error);
                                    break;
                                }
                            };
                            let engine_data = match engine_data_from_meta(&response.meta_info, true) {
                                Ok(engine_data) => engine_data,
                                Err(error) => {
                                    yield Err(error);
                                    break;
                                }
                            };
                            terminal.token_ids = token_ids;
                            terminal.log_probs = log_probs;
                            terminal.top_logprobs = top_logprobs;
                            terminal.engine_data = engine_data;
                            yield Ok(terminal);
                            break;
                        }

                        if !token_ids.is_empty() {
                            let engine_data = match engine_data_from_meta(&response.meta_info, false) {
                                Ok(engine_data) => engine_data,
                                Err(error) => {
                                    yield Err(error);
                                    break;
                                }
                            };
                            yield Ok(LLMEngineOutput {
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
        }))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        let Some(mut grpc_client) = self.state.get().map(|state| state.pool.control_client())
        else {
            return;
        };
        let request = pb::AbortRequest {
            rid: ctx.id().to_string(),
            abort_all: false,
        };
        if let Err(error) = client::abort(
            &mut grpc_client,
            request,
            self.transport.connect_attempt_timeout,
        )
        .await
        {
            tracing::debug!(
                request_id = ctx.id(),
                %error,
                "SGLang Abort RPC failed"
            );
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.cancel.cancel();
        tracing::info!("sglang sidecar shutdown complete");
        Ok(())
    }

    async fn kv_event_sources(&self) -> Result<Vec<KvEventSource>, DynamoError> {
        let state = self
            .state
            .get()
            .ok_or_else(|| client::engine_shutdown("sglang sidecar is not started"))?;
        Ok(state
            .kv_event_sources
            .iter()
            .map(|source| KvEventSource::Zmq {
                endpoint: source.endpoint.clone(),
                topic: source.topic.clone(),
                dp_rank: source.dp_rank,
            })
            .collect())
    }
}

fn bootstrap_discover(
    endpoint: &GrpcEndpoint,
    transport: &GrpcTransportConfig,
) -> Result<Discovery, DynamoError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| client::engine_shutdown(format!("bootstrap runtime: {err}")))?;
    runtime.block_on(async {
        let deadline = Instant::now() + transport.startup_deadline;
        let mut grpc_client = client::connect(endpoint, transport, deadline).await?;
        client::discover(&mut grpc_client, deadline).await
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
    let local_host = dynamo_runtime::utils::local_ip_for_advertise();
    resolve_bootstrap_host_with_local(explicit, endpoint, discovery, &local_host)
}

fn resolve_bootstrap_host_with_local(
    explicit: Option<&str>,
    endpoint: &str,
    discovery: &Discovery,
    local_host: &str,
) -> Result<Option<String>, DynamoError> {
    if let Some(host) = explicit.filter(|host| !host.trim().is_empty()) {
        return Ok(Some(host.trim().to_string()));
    }
    let from_server = discovery
        .server_info
        .get("host")
        .and_then(Value::as_str)
        .filter(|host| is_routable_host(host));
    if let Some(host) = from_server {
        return Ok(Some(host.trim().to_string()));
    }
    let from_dist = discovery
        .server_info
        .get("dist_init_addr")
        .and_then(Value::as_str)
        .and_then(host_from_address)
        .filter(|host| is_routable_host(host));
    if let Some(host) = from_dist {
        return Ok(Some(host));
    }
    if is_routable_host(local_host) {
        return Ok(Some(local_host.to_string()));
    }
    let from_endpoint = url::Url::parse(endpoint)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
        .filter(|host| is_routable_host(host));
    from_endpoint.map(Some).ok_or_else(|| {
        client::invalid_arg(
            "could not derive a routable prefill bootstrap host; set --bootstrap-host",
        )
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

fn is_routable_host(host: &str) -> bool {
    let host = host.trim().trim_matches(&['[', ']'][..]);
    if host.is_empty()
        || host.eq_ignore_ascii_case("localhost")
        || host.to_ascii_lowercase().ends_with(".localhost")
    {
        return false;
    }
    host.parse::<IpAddr>()
        .map(|address| !address.is_loopback() && !address.is_unspecified())
        .unwrap_or(true)
}

fn discover_kv_event_sources(
    discovery: &Discovery,
    engine_config: &EngineConfig,
    grpc_endpoint: &GrpcEndpoint,
) -> Result<Vec<DiscoveredKvEventSource>, DynamoError> {
    let Some(descriptor) = discovery.server_info.get("kv_events") else {
        return Ok(Vec::new());
    };
    if descriptor.is_null() {
        return Ok(Vec::new());
    }
    let descriptor = descriptor.as_object().ok_or_else(|| {
        client::protocol_error("SGLang GetServerInfo.kv_events must be an object or null")
    })?;

    let publisher = descriptor
        .get("publisher")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            client::protocol_error("SGLang GetServerInfo.kv_events.publisher must be a string")
        })?;
    if publisher != "zmq" {
        return Err(client::protocol_error(format!(
            "unsupported SGLang KV-event publisher `{publisher}`; expected `zmq`"
        )));
    }
    let endpoint_host = descriptor
        .get("endpoint_host")
        .and_then(Value::as_str)
        .filter(|host| !host.trim().is_empty())
        .ok_or_else(|| {
            client::protocol_error(
                "SGLang GetServerInfo.kv_events.endpoint_host must be a non-empty string",
            )
        })?;
    let base_port = descriptor
        .get("endpoint_port_base")
        .and_then(Value::as_u64)
        .and_then(|port| u16::try_from(port).ok())
        .filter(|port| *port != 0)
        .ok_or_else(|| {
            client::protocol_error(
                "SGLang GetServerInfo.kv_events.endpoint_port_base must be in 1..=65535",
            )
        })?;
    let topic = descriptor
        .get("topic")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            client::protocol_error("SGLang GetServerInfo.kv_events.topic must be a string")
        })?;
    let block_size = descriptor
        .get("block_size")
        .and_then(Value::as_u64)
        .and_then(|size| u32::try_from(size).ok())
        .filter(|size| *size != 0)
        .ok_or_else(|| {
            client::protocol_error(
                "SGLang GetServerInfo.kv_events.block_size must be a positive uint32",
            )
        })?;
    let reported_dp_size = descriptor
        .get("dp_size")
        .and_then(Value::as_u64)
        .and_then(|size| u32::try_from(size).ok())
        .filter(|size| *size != 0)
        .ok_or_else(|| {
            client::protocol_error(
                "SGLang GetServerInfo.kv_events.dp_size must be a positive uint32",
            )
        })?;

    let llm = engine_config.llm.as_ref().ok_or_else(|| {
        client::protocol_error("SGLang KV events require an LLM engine registration")
    })?;
    if llm.kv_cache_block_size != Some(block_size) {
        return Err(client::protocol_error(format!(
            "SGLang KV-event block size {block_size} does not match the registered engine block size {:?}",
            llm.kv_cache_block_size
        )));
    }
    let dp_start = llm.data_parallel_start_rank.unwrap_or(0);
    let dp_size = llm.data_parallel_size.unwrap_or(1);
    if reported_dp_size != dp_size {
        return Err(client::protocol_error(format!(
            "SGLang KV-event discovery reports {reported_dp_size} DP ranks but this sidecar registered {dp_size}"
        )));
    }
    let dp_end = dp_start.checked_add(dp_size).ok_or_else(|| {
        client::protocol_error("SGLang data-parallel rank range overflows uint32")
    })?;
    if dp_end > reported_dp_size {
        return Err(client::protocol_error(format!(
            "SGLang KV-event discovery does not cover registered DP rank range {dp_start}..{dp_end}"
        )));
    }
    let nnodes = client::json_u32(&discovery.server_info, "nnodes")
        .unwrap_or(1)
        .max(1);
    if nnodes > 1 && dp_size > 1 {
        return Err(client::protocol_error(format!(
            "SGLang KV events for multi-node DP are unsupported because GetServerInfo does not map DP ranks to publisher hosts (nnodes={nnodes}, dp_size={dp_size})"
        )));
    }

    let connect_host = kv_event_connect_host(endpoint_host, grpc_endpoint)?;
    let mut sources = Vec::with_capacity(dp_size as usize);
    for dp_rank in dp_start..dp_end {
        let port = u32::from(base_port)
            .checked_add(dp_rank)
            .filter(|port| *port <= u32::from(u16::MAX))
            .ok_or_else(|| {
                client::protocol_error(format!(
                    "SGLang KV-event port overflows 65535 for base port {base_port} and DP rank {dp_rank}"
                ))
            })?;
        sources.push(DiscoveredKvEventSource {
            endpoint: format!("tcp://{connect_host}:{port}"),
            topic: topic.to_string(),
            dp_rank,
        });
    }

    tracing::info!(
        sources = sources.len(),
        block_size,
        base_port,
        topic,
        "discovered SGLang ZMQ KV-event sources"
    );
    Ok(sources)
}

fn kv_event_connect_host(
    endpoint_host: &str,
    grpc_endpoint: &GrpcEndpoint,
) -> Result<String, DynamoError> {
    let endpoint_host = endpoint_host.trim();
    let bare_host = endpoint_host.trim_matches(&['[', ']'][..]);
    if matches!(bare_host, "*" | "0.0.0.0" | "::") {
        return Ok(grpc_endpoint.authority_host().to_string());
    }
    if let Ok(address) = bare_host.parse::<IpAddr>() {
        return Ok(match address {
            IpAddr::V4(address) => address.to_string(),
            IpAddr::V6(address) => format!("[{address}]"),
        });
    }
    if bare_host.contains([':', '/', '\\']) || bare_host.chars().any(char::is_whitespace) {
        return Err(client::protocol_error(format!(
            "invalid SGLang KV-event endpoint host `{endpoint_host}`"
        )));
    }
    Ok(bare_host.to_string())
}

fn build_engine_config(
    discovery: &Discovery,
    mode: DisaggregationMode,
    bootstrap_host: Option<String>,
    bootstrap_port: Option<u16>,
) -> Result<EngineConfig, DynamoError> {
    let page_size = client::json_u32(&discovery.server_info, "page_size");
    let dcp_size = client::json_u32(&discovery.server_info, "dcp_size")
        .unwrap_or(1)
        .max(1);
    let kv_cache_block_size = page_size.map(|size| size.saturating_mul(dcp_size));
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
    let (data_parallel_start_rank, data_parallel_size) = if enable_dp_attention && dp_size > 1 {
        // Native gRPC is exposed by the rank-zero frontend for the complete
        // multi-node SGLang endpoint, so one sidecar registers every DP rank.
        (Some(0), Some(dp_size))
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
        model_aliases: Vec::new(),
        runtime_data,
        llm: Some(LlmRegistration {
            context_length: discovery.max_model_len,
            kv_cache_block_size,
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

#[cfg(test)]
mod tests {
    use dynamo_sidecar_common::GrpcEndpoint;
    use serde_json::json;

    use super::{
        DisaggregationMode, DiscoveredKvEventSource, Discovery, build_engine_config,
        discover_kv_event_sources, resolve_bootstrap_host_with_local,
    };

    fn discovery(server_info: serde_json::Value) -> Discovery {
        Discovery {
            model_path: "model".to_string(),
            tokenizer_path: "tokenizer".to_string(),
            served_model_name: None,
            max_model_len: None,
            model_info: json!({}),
            server_info,
        }
    }

    #[test]
    fn explicit_bootstrap_host_takes_precedence() {
        let host = resolve_bootstrap_host_with_local(
            Some("prefill.example"),
            "http://127.0.0.1:30001",
            &discovery(json!({"dist_init_addr": "10.0.0.1:20000"})),
            "10.0.0.2",
        )
        .unwrap();
        assert_eq!(host.as_deref(), Some("prefill.example"));
    }

    #[test]
    fn server_host_precedes_dist_init_addr() {
        let host = resolve_bootstrap_host_with_local(
            None,
            "http://127.0.0.1:30001",
            &discovery(json!({
                "host": "10.0.0.1",
                "dist_init_addr": "10.0.0.2:20000"
            })),
            "10.0.0.3",
        )
        .unwrap();
        assert_eq!(host.as_deref(), Some("10.0.0.1"));
    }

    #[test]
    fn wildcard_server_host_uses_dist_init_addr() {
        let host = resolve_bootstrap_host_with_local(
            None,
            "http://127.0.0.1:30001",
            &discovery(json!({
                "host": "0.0.0.0",
                "dist_init_addr": "10.0.0.1:20000"
            })),
            "10.0.0.2",
        )
        .unwrap();
        assert_eq!(host.as_deref(), Some("10.0.0.1"));
    }

    #[test]
    fn loopback_endpoint_uses_routable_local_address() {
        let host = resolve_bootstrap_host_with_local(
            None,
            "http://127.0.0.1:30001",
            &discovery(json!({})),
            "10.0.0.2",
        )
        .unwrap();
        assert_eq!(host.as_deref(), Some("10.0.0.2"));
    }

    #[test]
    fn loopback_only_discovery_requires_override() {
        let error = resolve_bootstrap_host_with_local(
            None,
            "http://localhost:30001",
            &discovery(json!({"dist_init_addr": "0.0.0.0:20000"})),
            "127.0.0.1",
        )
        .unwrap_err();
        assert!(error.to_string().contains("--bootstrap-host"));
    }

    #[test]
    fn multi_node_grpc_endpoint_registers_all_dp_ranks() {
        let config = build_engine_config(
            &discovery(json!({
                "dp_size": 16,
                "enable_dp_attention": true,
                "nnodes": 4,
                "node_rank": 0,
            })),
            DisaggregationMode::Decode,
            None,
            None,
        )
        .unwrap();
        let registration = config.llm.unwrap();

        assert_eq!(registration.data_parallel_start_rank, Some(0));
        assert_eq!(registration.data_parallel_size, Some(16));
    }

    #[test]
    fn dcp_registers_logical_kv_block_size() {
        let config = build_engine_config(
            &discovery(json!({
                "page_size": 64,
                "dcp_size": 8,
                "max_total_num_tokens": 1024,
            })),
            DisaggregationMode::Decode,
            None,
            None,
        )
        .unwrap();
        let registration = config.llm.unwrap();

        assert_eq!(registration.kv_cache_block_size, Some(512));
        assert_eq!(registration.total_kv_blocks, Some(16));
    }

    #[test]
    fn discovers_ranked_kv_event_sources_from_server_info() {
        let discovery = discovery(json!({
            "page_size": 64,
            "dcp_size": 2,
            "dp_size": 2,
            "enable_dp_attention": true,
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "topic": "kv",
                "block_size": 128,
                "dp_size": 2
            }
        }));
        let config =
            build_engine_config(&discovery, DisaggregationMode::Aggregated, None, None).unwrap();
        let endpoint = GrpcEndpoint::parse("http://worker.example:30001", "test").unwrap();

        let sources = discover_kv_event_sources(&discovery, &config, &endpoint).unwrap();

        assert_eq!(
            sources,
            [
                DiscoveredKvEventSource {
                    endpoint: "tcp://worker.example:5557".to_string(),
                    topic: "kv".to_string(),
                    dp_rank: 0,
                },
                DiscoveredKvEventSource {
                    endpoint: "tcp://worker.example:5558".to_string(),
                    topic: "kv".to_string(),
                    dp_rank: 1,
                },
            ]
        );
    }
}
