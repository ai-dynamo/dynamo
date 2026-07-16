// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo backend for SGLang's native `sglang.runtime.v1` gRPC server.

use std::collections::HashMap;
use std::future::Future;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, DisaggregationMode, DynamoError, EngineConfig, GenerateContext, LLMEngine,
    LLMEngineOutput, LLMEngineOutputExt, LlmRegistration, ModelInput, PreprocessedRequest,
    WorkerConfig, usage,
};
use futures::stream::BoxStream;
use serde_json::Value;
use tokio::sync::OnceCell;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::args::{Args, TransportConfig, normalize_endpoint};
use crate::client::{self, Client, Discovery, Pool};
use crate::proto as pb;
use crate::protocol::{
    build_generate_request, disaggregated_params_to_json, engine_data_from_meta, extract_logprobs,
    meta_u32, output_ids_to_u32, terminal_from_meta,
};

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
            Some(args) => <Args as clap::Parser>::try_parse_from(args)
                .map_err(|err| client::invalid_arg(err.to_string()))?,
            None => <Args as clap::Parser>::parse(),
        };

        let endpoint = normalize_endpoint(&args.sglang_endpoint).map_err(client::invalid_arg)?;
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
            model_name: discovery.tokenizer_path.clone(),
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
                    self.transport.deadline
                )));
            }
            tokio::time::sleep_until((Instant::now() + self.transport.poll_interval).min(deadline))
                .await;
        }
    }
}

#[async_trait]
impl LLMEngine for SglangSidecarEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.pool.initialized() {
            return Err(client::engine_shutdown("sglang sidecar already started"));
        }

        let deadline = Instant::now() + self.transport.deadline;
        let pool = Pool::connect(
            &self.endpoint,
            &self.transport,
            self.transport.connections,
            deadline,
        )
        .await?;
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
        let (_, data_parallel_start_rank, data_parallel_size) =
            data_parallel_scope(&discovery.server_info);
        let effective_loads = wait_for_effective_loads(
            |deadline| {
                let mut control = control.clone();
                async move { client::get_load(&mut control, deadline).await }
            },
            deadline,
            self.transport.poll_interval,
            data_parallel_start_rank,
            data_parallel_size,
        )
        .await;

        let config = build_engine_config(
            &discovery,
            effective_loads.as_deref(),
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
        let Some(mut grpc_client) = self.pool.get().map(Pool::control_client) else {
            return;
        };
        let request = pb::AbortRequest {
            rid: ctx.id().to_string(),
            abort_all: false,
        };
        if let Err(error) =
            client::abort(&mut grpc_client, request, self.transport.connect_timeout).await
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
        let deadline = Instant::now() + transport.deadline;
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

fn data_parallel_scope(server_info: &Value) -> (u32, u32, u32) {
    let dp_size = client::json_u32(server_info, "dp_size").unwrap_or(1).max(1);
    let enable_dp_attention = server_info
        .get("enable_dp_attention")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let nnodes = client::json_u32(server_info, "nnodes").unwrap_or(1).max(1);
    let node_rank = client::json_u32(server_info, "node_rank").unwrap_or(0);
    let (data_parallel_start_rank, data_parallel_size) = if enable_dp_attention && dp_size > 1 {
        let local_size = (dp_size / nnodes).max(1);
        (node_rank.saturating_mul(local_size), local_size)
    } else {
        (0, dp_size)
    };
    (dp_size, data_parallel_start_rank, data_parallel_size)
}

async fn wait_for_effective_loads<F, Fut>(
    mut get_load: F,
    deadline: Instant,
    poll_interval: Duration,
    data_parallel_start_rank: u32,
    data_parallel_size: u32,
) -> Option<Vec<Value>>
where
    F: FnMut(Instant) -> Fut,
    Fut: Future<Output = Result<Vec<Value>, DynamoError>>,
{
    loop {
        if Instant::now() >= deadline {
            tracing::warn!("SGLang effective capacity was unavailable before the startup deadline");
            return None;
        }

        if let Ok(loads) = get_load(deadline).await
            && effective_engine_max_num_seqs(
                Some(&loads),
                data_parallel_start_rank,
                data_parallel_size,
            )
            .is_some()
        {
            return Some(loads);
        }

        tokio::time::sleep_until((Instant::now() + poll_interval).min(deadline)).await;
    }
}

fn effective_engine_max_num_seqs(
    loads: Option<&[Value]>,
    data_parallel_start_rank: u32,
    data_parallel_size: u32,
) -> Option<u64> {
    let loads = loads?;
    let end_rank = data_parallel_start_rank.checked_add(data_parallel_size)?;
    let mut smallest = None;

    for rank in data_parallel_start_rank..end_rank {
        let mut matching = loads
            .iter()
            .filter(|load| client::json_u32(load, "dp_rank") == Some(rank));
        let capacity = matching
            .next()
            .and_then(|load| client::json_u64(load, "max_running_requests"))
            .filter(|capacity| *capacity > 0)?;
        if matching.next().is_some() {
            return None;
        }
        smallest = Some(smallest.map_or(capacity, |current: u64| current.min(capacity)));
    }

    smallest.map(|capacity| capacity.saturating_mul(u64::from(data_parallel_size)))
}

fn build_engine_config(
    discovery: &Discovery,
    effective_loads: Option<&[Value]>,
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
    let (dp_size, data_parallel_start_rank, data_parallel_size) =
        data_parallel_scope(&discovery.server_info);
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

    let engine_max_num_seqs = effective_engine_max_num_seqs(
        effective_loads,
        data_parallel_start_rank,
        data_parallel_size,
    );

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
            engine_max_num_seqs,
            max_num_batched_tokens,
            data_parallel_size: Some(data_parallel_size),
            data_parallel_start_rank: Some(data_parallel_start_rank),
            bootstrap_host: mode.is_prefill().then_some(bootstrap_host).flatten(),
            bootstrap_port: mode.is_prefill().then_some(bootstrap_port).flatten(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::future::ready;
    use std::time::Duration;

    use dynamo_backend_common::{DisaggregationMode, DynamoError};
    use serde_json::json;
    use tokio::time::Instant;

    use super::{
        Discovery, build_engine_config, resolve_bootstrap_host_with_local, wait_for_effective_loads,
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
    fn dist_init_addr_precedes_local_address() {
        let host = resolve_bootstrap_host_with_local(
            None,
            "http://127.0.0.1:30001",
            &discovery(json!({"dist_init_addr": "10.0.0.1:20000"})),
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
    fn engine_config_normalizes_max_num_seqs_to_local_dp_scope() {
        let loads = vec![
            json!({"dp_rank": 2, "max_running_requests": 12}),
            json!({"dp_rank": 3, "max_running_requests": 12}),
        ];
        let config = build_engine_config(
            &discovery(json!({
                "max_running_requests": 48,
                "dp_size": 4,
                "enable_dp_attention": true,
                "nnodes": 2,
                "node_rank": 1
            })),
            Some(&loads),
            DisaggregationMode::Aggregated,
            None,
            None,
        )
        .unwrap();
        let llm = config
            .llm
            .expect("SGLang sidecar must register LLM capacity");

        assert_eq!(llm.max_num_seqs, Some(12));
        assert_eq!(llm.data_parallel_size, Some(2));
        assert_eq!(llm.data_parallel_start_rank, Some(2));
        assert_eq!(llm.engine_max_num_seqs, Some(24));
    }

    #[test]
    fn engine_config_reports_all_ranks_owned_by_ordinary_dp() {
        let loads = (0..4)
            .map(|dp_rank| json!({"dp_rank": dp_rank, "max_running_requests": 12}))
            .collect::<Vec<_>>();
        let llm = build_engine_config(
            &discovery(json!({"max_running_requests": 48, "dp_size": 4})),
            Some(&loads),
            DisaggregationMode::Aggregated,
            None,
            None,
        )
        .unwrap()
        .llm
        .unwrap();

        assert_eq!(
            (
                llm.data_parallel_start_rank,
                llm.data_parallel_size,
                llm.engine_max_num_seqs,
            ),
            (Some(0), Some(4), Some(48)),
        );
    }

    #[tokio::test]
    async fn startup_waits_for_complete_effective_capacity() {
        let mut samples = VecDeque::from([
            vec![json!({"dp_rank": 0, "max_running_requests": 0})],
            vec![
                json!({"dp_rank": 0, "max_running_requests": 12}),
                json!({"dp_rank": 1, "max_running_requests": 10}),
            ],
        ]);
        let loads = wait_for_effective_loads(
            |_| ready(Ok::<_, DynamoError>(samples.pop_front().unwrap())),
            Instant::now() + Duration::from_secs(1),
            Duration::ZERO,
            0,
            2,
        )
        .await
        .expect("a complete positive capacity snapshot");

        let llm = build_engine_config(
            &discovery(json!({"max_running_requests": 24, "dp_size": 2})),
            Some(&loads),
            DisaggregationMode::Aggregated,
            None,
            None,
        )
        .unwrap()
        .llm
        .unwrap();

        assert_eq!(llm.engine_max_num_seqs, Some(20));
        assert!(samples.is_empty());
    }

    #[test]
    fn effective_capacity_requires_all_owned_ranks_and_uses_smallest() {
        let loads = |capacities: &[u64]| {
            capacities
                .iter()
                .enumerate()
                .map(|(dp_rank, capacity)| {
                    json!({"dp_rank": dp_rank, "max_running_requests": capacity})
                })
                .collect::<Vec<_>>()
        };
        let complete = loads(&[11, 10, 9, 10]);
        let incomplete = loads(&[12, 12, 12]);

        for (loads, expected) in [
            (Some(complete.as_slice()), Some(36)),
            (None, None),
            (Some(incomplete.as_slice()), None),
        ] {
            assert_eq!(super::effective_engine_max_num_seqs(loads, 0, 4), expected);
        }
    }
}
