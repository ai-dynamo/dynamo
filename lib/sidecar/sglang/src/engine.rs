// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo backend for SGLang's native `sglang.api.v1` gRPC server.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, DisaggregationMode, DynamoError, EngineConfig, GenerateContext,
    KvEventSource, LLMEngine, LLMEngineOutput, LLMEngineOutputExt, LlmRegistration, ModelInput,
    PreprocessedRequest, RuntimeConfig, WorkerConfig, usage,
};
use dynamo_llm::local_model::runtime_config::{
    TOKEN_BUDGET_RUNTIME_KEY, TOOL_CALL_STRUCTURAL_TAG_REASONING_GATE_RUNTIME_KEY, TokenBudget,
};
use dynamo_sidecar_common::{
    EngineBootstrapResult, GrpcEndpoint, GrpcTransportConfig, SidecarStartupError,
};
use futures::stream::BoxStream;
use serde_json::Value;
use tokio::sync::OnceCell;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::args::Args;
use crate::client::{self, AbortRpc, Client, GRPC_SERVICE, NodeMetadata, Pool, WORKER_GROUP_KEY};
use crate::discovery::{self, EngineInfo, sglang_eagle_enabled};
use crate::native_http::{self, NativeHttp};
use crate::proto as pb;
use crate::protocol::{FrameOutput, ResponseConverter, build_generate_request, validate_dp_rank};

const RETRY_LOG_INTERVAL: Duration = Duration::from_secs(30);

pub struct SglangSidecarEngine {
    endpoint: GrpcEndpoint,
    transport: GrpcTransportConfig,
    disaggregation_mode: DisaggregationMode,
    bootstrap_host: Option<String>,
    bootstrap_port: Option<u16>,
    native_http_max_event_bytes: usize,
    state: OnceCell<StartedState>,
    cancel: CancellationToken,
}

struct StartedState {
    pool: Pool,
    native_http: Option<NativeHttp>,
    kv_event_sources: Vec<DiscoveredKvEventSource>,
    combined_token_limit: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DiscoveredKvEventSource {
    endpoint: String,
    topic: String,
    dp_rank: u32,
}

impl SglangSidecarEngine {
    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        match argv {
            Some(argv) => Self::try_from_args(argv).map_err(SidecarStartupError::into_dynamo),
            None => Self::from_parsed(<Args as clap::Parser>::parse()),
        }
    }

    /// Parse injected arguments while retaining Clap's structured exit error.
    ///
    /// Embedded callers use this to distinguish help and version output from
    /// Dynamo startup failures without changing `from_args`'s error contract.
    pub fn try_from_args(argv: Vec<String>) -> Result<(Self, WorkerConfig), SidecarStartupError> {
        let args = <Args as clap::Parser>::try_parse_from(argv)?;
        Self::from_parsed(args).map_err(Into::into)
    }

    /// Parse CLI arguments without connecting; discovery runs after probe startup.
    pub fn from_cli() -> Result<
        (
            RuntimeConfig,
            impl std::future::Future<Output = EngineBootstrapResult<Self>>,
        ),
        DynamoError,
    > {
        let args = <Args as clap::Parser>::parse();
        Self::validate_args(&args)?;
        Ok((
            args.sidecar.common.runtime.clone(),
            Self::from_parsed_async(args, false),
        ))
    }

    /// Parse embedded launcher arguments now, then discover metadata after the
    /// shared sidecar runner has started probes and connected the runtime.
    pub fn try_from_args_async(
        argv: Vec<String>,
    ) -> Result<
        (
            RuntimeConfig,
            impl std::future::Future<Output = EngineBootstrapResult<Self>>,
        ),
        SidecarStartupError,
    > {
        let args = <Args as clap::Parser>::try_parse_from(argv)?;
        Self::validate_args(&args)?;
        Ok((
            args.sidecar.common.runtime.clone(),
            Self::from_parsed_async(args, false),
        ))
    }

    fn from_parsed(args: Args) -> Result<(Self, WorkerConfig), DynamoError> {
        Self::validate_args(&args)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| client::engine_shutdown(format!("bootstrap runtime: {error}")))?;
        runtime.block_on(Self::from_parsed_async(args, true))
    }

    pub(crate) fn validate_args(args: &Args) -> Result<(), DynamoError> {
        if args.sidecar.common.route_to_encoder {
            return Err(client::invalid_arg(
                "route-to-encoder is not supported by the SGLang sidecar",
            ));
        }
        Ok(())
    }

    async fn from_parsed_async(
        args: Args,
        bootstrap: bool,
    ) -> Result<(Self, WorkerConfig), DynamoError> {
        let discovery = client::bootstrap_discover(
            &args.sidecar.grpc_endpoint,
            &args.sidecar.grpc.config(),
            bootstrap,
        )
        .await?;
        Self::from_discovered(args, discovery)
    }

    pub(crate) fn from_discovered(
        args: Args,
        discovery: EngineInfo,
    ) -> Result<(Self, WorkerConfig), DynamoError> {
        Self::validate_args(&args)?;
        let endpoint = args.sidecar.grpc_endpoint;
        let transport = args.sidecar.grpc.config();
        let disaggregation_mode = discovery.disaggregation_mode;
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
        tracing::info!(
            %endpoint,
            mode = ?disaggregation_mode,
            model = %discovery.model_path,
            "sglang sidecar bootstrapped native gRPC discovery"
        );

        let native_http_max_event_bytes = args.native_http_max_event_bytes.get();
        let common = args.sidecar.common;
        let structural_tag = common.structural_tag(true);
        let config = WorkerConfig {
            runtime: common.runtime,
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
                .or_else(|| discovery.reasoning_parser.clone()),
            tool_call_parser: common
                .dyn_tool_call_parser
                .or_else(|| discovery.tool_call_parser.clone()),
            structural_tag,
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
                native_http_max_event_bytes,
                state: OnceCell::new(),
                cancel: CancellationToken::new(),
            },
            config,
        ))
    }

    async fn await_ready(&self, client: &mut Client, deadline: Instant) -> Result<(), DynamoError> {
        let started = Instant::now();
        let mut attempt = 0_u64;
        let mut last_logged_at: Option<Instant> = None;
        loop {
            attempt += 1;
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
            let now = Instant::now();
            if last_logged_at.is_none_or(|last| now.duration_since(last) >= RETRY_LOG_INTERVAL) {
                // WARN, not silent: this loop previously logged nothing at all on a
                // failed attempt, so a SGLang engine that's slow (or never becomes)
                // healthy produced zero visible output anywhere for up to
                // startup_deadline (default 300s) -- indistinguishable from a hang.
                // Rate-limited like GrpcChannelPool::connect_until_ready: a normal
                // slow startup retries every retry_interval (default 1s) and would
                // otherwise spam hundreds of lines.
                tracing::warn!(
                    attempt,
                    elapsed = ?started.elapsed(),
                    remaining = ?deadline.saturating_duration_since(now),
                    retry_interval = ?self.transport.retry_interval,
                    reason = %retry_message,
                    "SGLang not healthy yet; retrying"
                );
                last_logged_at = Some(now);
            }
            tokio::time::sleep_until((now + self.transport.retry_interval).min(deadline)).await;
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
        let discovery = discovery::engine_info(client::discover(&mut control, deadline).await?)?;
        let observed_mode = discovery.disaggregation_mode;
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
        // Before native HTTP readiness, which may spend the rest of the
        // startup deadline: the worker group's DNS lookup needs some of it.
        let kv_event_sources = discover_kv_event_sources(&discovery, &mut config, deadline).await?;
        let native_http = match NativeHttp::discover(
            &self.endpoint,
            &discovery,
            AbortRpc::new(control),
            self.transport.connect_attempt_timeout,
            self.native_http_max_event_bytes,
        )? {
            Some(native_http) => {
                match native_http
                    .await_ready(deadline, self.transport.retry_interval)
                    .await
                {
                    Ok(()) => Some(native_http),
                    Err(error) => {
                        tracing::warn!(
                            %error,
                            "SGLang native HTTP generation is unavailable; continuing with gRPC"
                        );
                        None
                    }
                }
            }
            None => None,
        };
        if native_http.is_some() {
            config
                .runtime_data
                .insert("sglang_generate".into(), true.into());
        }
        let connection_count = pool.len();
        let kv_event_source_count = kv_event_sources.len();
        let combined_token_limit = combined_token_limit(&discovery);
        if lacks_draft_reserve(&discovery) {
            tracing::warn!(
                "SGLang runs EAGLE speculative decoding but reports no draft-token reserve \
                 (GetServerInfo.num_reserved_tokens); it predates the api.v1 discovery fields \
                 listed in lib/sidecar/sglang/README.md, so no token budget is published"
            );
        }
        if combined_token_limit.is_none() {
            tracing::warn!("requests without max_tokens keep SGLang's 128-token default");
        }
        self.state
            .set(StartedState {
                pool,
                native_http,
                kv_event_sources,
                combined_token_limit,
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
        mut request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let state = self
            .state
            .get()
            .ok_or_else(|| client::engine_shutdown("generate called before start"))?;
        validate_dp_rank(&request, self.disaggregation_mode)?;
        if let Some(native_request) = native_http::request(
            &request,
            ctx.id(),
            self.disaggregation_mode,
            self.bootstrap_host.as_deref(),
            self.bootstrap_port,
        )? {
            let native_http = state.native_http.clone().ok_or_else(|| {
                client::invalid_arg(
                    "native SGLang Generate is unavailable because no ready incremental HTTP endpoint was discovered",
                )
            })?;
            return Ok(native_http.generate(native_request, ctx, self.cancel.child_token()));
        }
        if request.stop_conditions.max_tokens.is_none() {
            request.stop_conditions.max_tokens = omitted_max_tokens_default(
                state.combined_token_limit,
                request.token_ids.len(),
                request.stop_conditions.min_tokens,
            );
        }
        let mut grpc_client = state.pool.stream_client();

        let prompt_tokens = request.token_ids.len() as u32;
        let return_tokens_as_ids = request
            .output_options
            .return_tokens_as_token_ids
            .unwrap_or(false);
        let (grpc_request, rendezvous) = build_generate_request(
            &request,
            ctx.id(),
            self.disaggregation_mode,
            self.bootstrap_host.as_deref(),
            self.bootstrap_port,
        )?;
        let is_prefill = self.disaggregation_mode.is_prefill();
        let mut prefill_handoff = if is_prefill { rendezvous } else { None };
        let cancel = self.cancel.child_token();
        let mut converter = ResponseConverter::new(
            is_prefill,
            return_tokens_as_ids,
            request.stop_conditions,
            prompt_tokens,
        );

        Ok(Box::pin(async_stream::stream! {
            if ctx.is_stopped() || cancel.is_cancelled() {
                yield Ok(LLMEngineOutput::cancelled().with_usage(usage(prompt_tokens, 0)));
                return;
            }
            let shutdown = cancel.clone();
            let stopped_ctx = ctx.inner_arc();
            let mut request_cancellation = Box::pin(async move { stopped_ctx.stopped().await });
            let mut shutdown_cancellation = Box::pin(async move { cancel.cancelled().await });

            tracing::debug!(request_id = %ctx.id(), "sending request to SGLang gRPC");
            let opened = tokio::select! {
                biased;
                _ = &mut request_cancellation => None,
                _ = &mut shutdown_cancellation => None,
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
            if is_prefill {
                let Some(handoff) = prefill_handoff.take() else {
                    yield Err(client::protocol_error(
                        "SGLang gRPC prefill request is missing disaggregated params",
                    ));
                    return;
                };
                // Publish the handoff only after the gRPC transport opens the
                // response stream so decode can rendezvous while prefill runs.
                yield Ok(LLMEngineOutput {
                    disaggregated_params: Some(handoff),
                    ..Default::default()
                });
            }

            loop {
                tokio::select! {
                    biased;
                    _ = &mut request_cancellation => {
                        yield Ok(LLMEngineOutput::cancelled().with_usage(converter.usage()));
                        break;
                    }
                    _ = &mut shutdown_cancellation => {
                        yield Ok(LLMEngineOutput::cancelled().with_usage(converter.usage()));
                        break;
                    }
                    message = stream.message() => {
                        let item = match message {
                            Ok(Some(item)) => item,
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
                        match item.item {
                            Some(pb::generate_stream_item::Item::Frame(frame)) => {
                                match converter.frame(frame) {
                                    Ok(FrameOutput::Skip) => {}
                                    Ok(FrameOutput::Delta(output)) => yield Ok(output),
                                    Ok(FrameOutput::Terminal(output)) => {
                                        yield Ok(output);
                                        break;
                                    }
                                    Err(error) => {
                                        yield Err(error);
                                        break;
                                    }
                                }
                            }
                            // SGLang ends a request's stream after its error item.
                            Some(pb::generate_stream_item::Item::Error(error)) => {
                                let is_cancelled = ctx.is_stopped() || shutdown.is_cancelled();
                                yield converter.stream_error(&error, is_cancelled);
                                break;
                            }
                            None => {
                                yield Err(client::protocol_error(
                                    "SGLang Generate stream item has neither a frame nor an error",
                                ));
                                break;
                            }
                        }
                    }
                }
            }
        }))
    }

    /// The Abort RPC also cancels requests opened through native HTTP.
    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        let Some(state) = self.state.get() else {
            return;
        };
        AbortRpc::new(state.pool.control_client())
            .abort(ctx.id(), self.transport.connect_attempt_timeout)
            .await;
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
                heartbeat_timeout: None,
                endpoint: source.endpoint.clone(),
                topic: source.topic.clone(),
                dp_rank: source.dp_rank,
                image_token_id: None,
            })
            .collect())
    }
}

fn discovery_bootstrap_port(discovery: &EngineInfo) -> Result<Option<u16>, DynamoError> {
    discovery
        .disaggregation_bootstrap_port
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
    discovery: &EngineInfo,
) -> Result<Option<String>, DynamoError> {
    let local_host = dynamo_runtime::utils::local_ip_for_advertise();
    resolve_bootstrap_host_with_local(explicit, endpoint, discovery, &local_host)
}

fn resolve_bootstrap_host_with_local(
    explicit: Option<&str>,
    endpoint: &str,
    discovery: &EngineInfo,
    local_host: &str,
) -> Result<Option<String>, DynamoError> {
    if let Some(host) = explicit.filter(|host| !host.trim().is_empty()) {
        return Ok(Some(host.trim().to_string()));
    }
    let from_server = discovery
        .host
        .as_deref()
        .filter(|host| is_routable_host(host));
    if let Some(host) = from_server {
        return Ok(Some(host.trim().to_string()));
    }
    let from_dist = discovery
        .dist_init_addr
        .as_deref()
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

/// The engine's own KV-event publishers. Each must publish at the registered
/// block size; the worker group is published for multi-node deployments.
async fn discover_kv_event_sources(
    discovery: &EngineInfo,
    config: &mut EngineConfig,
    deadline: Instant,
) -> Result<Vec<DiscoveredKvEventSource>, DynamoError> {
    let metadata: &NodeMetadata = discovery.kv_events.as_ref().map_err(Clone::clone)?;
    let llm = config
        .llm
        .as_ref()
        .ok_or_else(|| client::invalid_arg("KV sources require LLM registration"))?;
    metadata
        .validate_registration(llm.kv_cache_block_size)
        .map_err(|error| client::invalid_arg(error.to_string()))?;
    if let Some(group_id) = metadata
        .worker_group_id(deadline)
        .await
        .map_err(|error| client::invalid_arg(error.to_string()))?
    {
        config
            .runtime_data
            .insert(WORKER_GROUP_KEY.into(), group_id.into());
    }
    let sources: Vec<_> = metadata
        .kv_event_sources
        .iter()
        .map(|source| DiscoveredKvEventSource {
            endpoint: source.endpoint.clone(),
            topic: source.topic.clone(),
            dp_rank: source.dp_rank,
        })
        .collect();
    if !sources.is_empty() {
        tracing::info!(
            sources = sources.len(),
            "discovered SGLang ZMQ KV-event sources"
        );
    }
    Ok(sources)
}

/// Context length minus the speculative draft-token reserve, which SGLang
/// counts against the context once `max_new_tokens` is set. `None` when the
/// reserve is unknown.
fn combined_token_limit(discovery: &EngineInfo) -> Option<u32> {
    discovery
        .context_length?
        .checked_sub(discovery.num_reserved_tokens?)
}

/// EAGLE reserves at least one draft-token slot per request, so a server that
/// reports none predates the field and the reserve is unknown.
fn lacks_draft_reserve(discovery: &EngineInfo) -> bool {
    sglang_eagle_enabled(discovery.speculative_algorithm.as_deref())
        && discovery.num_reserved_tokens.is_none()
}

/// SGLang holds a request's grammar until its reasoning parser sees the end of thinking.
fn sglang_gates_reasoning(discovery: &EngineInfo) -> bool {
    discovery.reasoning_parser.is_some() && !discovery.skip_tokenizer_init
}

/// SGLang rejects token overflow unless `allow_auto_truncate` is set.
fn sglang_token_budget(discovery: &EngineInfo) -> Option<TokenBudget> {
    let truncates = discovery.allow_auto_truncate;
    Some(TokenBudget {
        combined_limit: combined_token_limit(discovery)?,
        reject_prompt_overflow: !truncates,
        reject_total_overflow: !truncates,
    })
}

/// SGLang's gRPC bridge turns an unset `max_new_tokens` into its 128-token
/// default, so ask for the room left in the context. Leave it unset when that
/// room cannot hold `min_tokens`, and SGLang clamps the request itself.
fn omitted_max_tokens_default(
    combined_token_limit: Option<u32>,
    prompt_len: usize,
    min_tokens: Option<u32>,
) -> Option<u32> {
    combined_token_limit?
        .checked_sub(u32::try_from(prompt_len).ok()?)
        .filter(|remaining| *remaining >= min_tokens.unwrap_or(0).max(1))
}

fn build_engine_config(
    discovery: &EngineInfo,
    mode: DisaggregationMode,
    bootstrap_host: Option<String>,
    bootstrap_port: Option<u16>,
) -> Result<EngineConfig, DynamoError> {
    let page_size = discovery.page_size;
    let kv_cache_block_size = page_size.map(|size| size.saturating_mul(discovery.dcp_size));
    let max_total_tokens = discovery.max_total_num_tokens;
    let total_kv_blocks = match (max_total_tokens, page_size) {
        (Some(tokens), Some(page_size)) if page_size > 0 => {
            Some(tokens.saturating_add(u64::from(page_size) - 1) / u64::from(page_size))
        }
        _ => None,
    };
    let max_num_seqs = discovery.max_num_seqs;
    let max_num_batched_tokens = discovery.max_prefill_tokens.or(max_total_tokens);

    if mode.is_prefill() && (bootstrap_host.is_none() || bootstrap_port.is_none()) {
        return Err(client::protocol_error(
            "prefill SGLang discovery did not provide a usable bootstrap address",
        ));
    }

    let enable_eagle = sglang_eagle_enabled(discovery.speculative_algorithm.as_deref());

    let mut runtime_data = HashMap::new();
    runtime_data.insert(
        "grpc_service".to_string(),
        Value::String(GRPC_SERVICE.to_string()),
    );
    if let Some(total_tokens) = discovery.native_offloading_capacity {
        runtime_data.insert(
            "native_offloading_capacity".to_string(),
            serde_json::json!({"total_tokens": total_tokens}),
        );
    }
    if sglang_gates_reasoning(discovery) {
        runtime_data.insert(
            TOOL_CALL_STRUCTURAL_TAG_REASONING_GATE_RUNTIME_KEY.to_string(),
            Value::Bool(true),
        );
    }
    if let Some(token_budget) = sglang_token_budget(discovery) {
        runtime_data.insert(
            TOKEN_BUDGET_RUNTIME_KEY.to_string(),
            serde_json::to_value(token_budget).expect("token budget is serializable"),
        );
    }

    Ok(EngineConfig {
        model: discovery.model_path.clone(),
        served_model_name: discovery.served_model_name.clone(),
        model_aliases: Vec::new(),
        runtime_data,
        llm: Some(LlmRegistration {
            context_length: discovery.context_length,
            kv_cache_block_size,
            total_kv_blocks,
            max_num_seqs,
            max_num_batched_tokens,
            max_gpu_lora_count: None,
            // Discovery rejects deployments with more than one DP rank.
            data_parallel_size: Some(1),
            data_parallel_start_rank: Some(0),
            enable_eagle,
            bootstrap_host: mode.is_prefill().then_some(bootstrap_host).flatten(),
            bootstrap_port: mode.is_prefill().then_some(bootstrap_port).flatten(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;
    use tokio::time::Instant;

    use clap::Parser;
    use dynamo_backend_common::{BackendError, DynamoError, ErrorType, ModelInput};
    use serde_json::json;

    use super::{
        Args, DisaggregationMode, DiscoveredKvEventSource, EngineConfig, EngineInfo,
        SglangSidecarEngine, WORKER_GROUP_KEY, build_engine_config, combined_token_limit,
        discover_kv_event_sources, discovery_bootstrap_port, lacks_draft_reserve,
        omitted_max_tokens_default, resolve_bootstrap_host_with_local, sglang_eagle_enabled,
    };
    use crate::discovery;
    use crate::proto as pb;

    /// The discovery RPC responses of one engine.
    struct Reported {
        model: pb::GetModelInfoResponse,
        server: pb::GetServerInfoResponse,
        deployment: pb::DeploymentInfo,
    }

    fn try_discovery(update: impl FnOnce(&mut Reported)) -> Result<EngineInfo, DynamoError> {
        let mut reported = Reported {
            model: pb::GetModelInfoResponse {
                model_path: "model".into(),
                tokenizer_path: "tokenizer".into(),
                ..Default::default()
            },
            server: pb::GetServerInfoResponse {
                incremental_streaming_output: true,
                ..Default::default()
            },
            deployment: pb::DeploymentInfo {
                nnodes: 1,
                dp_size: 1,
                dcp_size: 1,
                ..Default::default()
            },
        };
        update(&mut reported);
        let Reported {
            model,
            mut server,
            deployment,
        } = reported;
        server.deployment_info = Some(deployment);
        discovery::engine_info(discovery::parse_discovery(model, server, Vec::new())?)
    }

    /// The engine's view of discovery responses that `update` rewrites.
    fn discovery(update: impl FnOnce(&mut Reported)) -> EngineInfo {
        try_discovery(update).unwrap()
    }

    fn args() -> Args {
        Args::try_parse_from([
            "sidecar",
            "--grpc-endpoint",
            "worker.example:30001",
            "--namespace",
            "test-namespace",
            "--component",
            "custom-component",
            "--endpoint",
            "tokens",
            "--endpoint-types",
            "completions",
            "--custom-jinja-template",
            "template.jinja",
            "--exclude-tools-when-tool-choice-none",
            "false",
            "--enable-rl",
            "--grpc-connections",
            "3",
            "--grpc-connect-attempt-timeout-secs",
            "7",
            "--grpc-retry-interval-secs",
            "2",
            "--grpc-startup-deadline-secs",
            "11",
        ])
        .unwrap()
    }

    #[test]
    fn worker_preserves_options_and_uses_discovered_identity_and_role() {
        for (mode, component) in [("aggregated", "custom-component"), ("decode", "backend")] {
            let mut args = args();
            args.sidecar.common.disaggregation_mode = DisaggregationMode::Prefill;
            let mut discovery = discovery(|reported| {
                reported.server.disaggregation_mode = mode.into();
            });
            discovery.served_model_name = Some("public-model".into());
            let (engine, config) = SglangSidecarEngine::from_discovered(args, discovery).unwrap();

            assert_eq!(config.namespace, "test-namespace");
            assert_eq!(config.component, component);
            assert_eq!(config.endpoint, "tokens");
            assert_eq!(config.endpoint_types, "completions");
            assert_eq!(
                config.custom_jinja_template.as_deref(),
                Some(std::path::Path::new("template.jinja"))
            );
            assert_eq!(config.model_name, "tokenizer");
            assert_eq!(config.served_model_name.as_deref(), Some("public-model"));
            assert!(matches!(config.model_input, ModelInput::Tokens));
            assert!(!config.exclude_tools_when_tool_choice_none);
            assert!(!config.route_to_encoder);
            assert!(config.enable_rl);
            assert_eq!(
                config.disaggregation_mode.as_str(),
                if mode == "aggregated" {
                    "agg"
                } else {
                    "decode"
                }
            );
            assert_eq!(engine.disaggregation_mode, config.disaggregation_mode);
            assert_eq!(engine.endpoint.as_str(), "http://worker.example:30001");
            assert_eq!(engine.transport.connections.get(), 3);
            assert_eq!(engine.transport.connect_attempt_timeout.as_secs(), 7);
            assert_eq!(engine.transport.retry_interval.as_secs(), 2);
            assert_eq!(engine.transport.startup_deadline.as_secs(), 11);
            assert!(engine.bootstrap_host.is_none());
            assert!(engine.bootstrap_port.is_none());
            assert!(!engine.state.initialized());
        }
    }

    #[test]
    fn worker_parsers_prefer_cli_then_discovery() {
        for (cli_reasoning, cli_tools, expected_reasoning, expected_tools) in [
            (
                Some("cli-reasoning"),
                Some("cli-tools"),
                "cli-reasoning",
                "cli-tools",
            ),
            (Some("cli-reasoning"), None, "cli-reasoning", "native-tools"),
            (None, None, "native-reasoning", "native-tools"),
        ] {
            let mut args = args();
            args.sidecar.common.dyn_reasoning_parser = cli_reasoning.map(str::to_string);
            args.sidecar.common.dyn_tool_call_parser = cli_tools.map(str::to_string);
            let (_, config) = SglangSidecarEngine::from_discovered(
                args,
                discovery(|reported| {
                    reported.model.reasoning_parser = Some("native-reasoning".into());
                    reported.model.tool_call_parser = Some("native-tools".into());
                }),
            )
            .unwrap();
            assert_eq!(config.reasoning_parser.as_deref(), Some(expected_reasoning));
            assert_eq!(config.tool_call_parser.as_deref(), Some(expected_tools));
        }
        for native in [Some(String::new()), None] {
            let (_, config) = SglangSidecarEngine::from_discovered(
                args(),
                discovery(|reported| {
                    reported.model.reasoning_parser = native.clone();
                    reported.model.tool_call_parser = native;
                }),
            )
            .unwrap();
            assert!(config.reasoning_parser.is_none());
            assert!(config.tool_call_parser.is_none());
        }
    }

    #[test]
    fn encoder_routing_is_rejected_before_discovery() {
        let mut args = args();
        args.sidecar.common.route_to_encoder = true;
        let error = SglangSidecarEngine::from_parsed(args)
            .err()
            .expect("unsupported encoder routing");
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert!(
            error
                .to_string()
                .contains("route-to-encoder is not supported")
        );
    }

    #[test]
    fn discovery_defers_start_time_validation() {
        // start() reports these where it resolves them, so bootstrap
        // construction still succeeds.
        let info = discovery(|reported| {
            reported.server.port = 65_536;
            reported.deployment.nnodes = 2;
        });
        SglangSidecarEngine::from_discovered(args(), info.clone()).unwrap();
        build_engine_config(&info, DisaggregationMode::Aggregated, None, None).unwrap();
        let port_error = info.http_port.as_ref().unwrap().as_ref().unwrap_err();
        assert!(port_error.to_string().contains("GetServerInfo.port"));
        let kv_error = info.kv_events.as_ref().unwrap_err();
        assert!(
            kv_error
                .to_string()
                .contains("multinode deployments require dist_init_addr")
        );
    }

    #[test]
    fn bootstrap_port_requires_a_nonzero_u16() {
        for (value, expected) in [(1, 1), (65535, u16::MAX)] {
            assert_eq!(
                discovery_bootstrap_port(&discovery(|reported| {
                    reported.deployment.disaggregation_bootstrap_port = Some(value);
                }))
                .unwrap(),
                Some(expected)
            );
        }
        for value in [None, Some(0), Some(65536)] {
            let error = discovery_bootstrap_port(&discovery(|reported| {
                reported.deployment.disaggregation_bootstrap_port = value;
            }))
            .unwrap_err();
            assert_eq!(
                error.error_type(),
                ErrorType::Backend(BackendError::Unknown)
            );
            assert!(error.to_string().contains("disaggregation_bootstrap_port"));
        }
    }

    /// The bootstrap host candidates discovery reports.
    fn hosts(host: &str, dist_init_addr: Option<&str>) -> EngineInfo {
        discovery(|reported| {
            reported.server.host = host.into();
            reported.deployment.dist_init_addr = dist_init_addr.map(str::to_string);
        })
    }

    #[test]
    fn bootstrap_host_falls_back_to_remote_endpoint_and_preserves_ipv6() {
        for (info, endpoint, expected) in [
            (
                hosts("localhost", Some("worker.localhost:20000")),
                "http://remote.example:30001",
                "remote.example",
            ),
            (
                hosts("", Some("tcp://[2001:db8::1]:20000")),
                "http://127.0.0.1:30001",
                "[2001:db8::1]",
            ),
        ] {
            let host =
                resolve_bootstrap_host_with_local(Some(" "), endpoint, &info, "::1").unwrap();
            assert_eq!(host.as_deref(), Some(expected));
        }
    }

    #[test]
    fn registration_preserves_identity_and_optional_limits() {
        let mut info = discovery(|reported| {
            reported.server.max_context_length = 4096;
            reported.server.max_total_num_tokens = Some(512);
            reported.deployment.max_prefill_tokens = 128;
            reported.deployment.max_running_requests = Some(32);
        });
        info.served_model_name = Some("public-model".into());
        let config =
            build_engine_config(&info, DisaggregationMode::Aggregated, None, None).unwrap();
        assert_eq!(config.model, "model");
        assert_eq!(config.served_model_name.as_deref(), Some("public-model"));
        assert_eq!(
            config.runtime_data.get("grpc_service"),
            Some(&json!("sglang.api.v1.SglangService"))
        );
        let llm = config.llm.unwrap();
        assert_eq!(llm.context_length, Some(4096));
        assert_eq!(llm.max_num_batched_tokens, Some(128));
        assert_eq!(llm.max_num_seqs, Some(32));
        assert_eq!(llm.kv_cache_block_size, None);
        assert_eq!(llm.total_kv_blocks, None);

        // Zero is proto3's unset prefill limit.
        let info = discovery(|reported| {
            reported.server.max_total_num_tokens = Some(512);
        });
        assert_eq!(
            build_engine_config(&info, DisaggregationMode::Aggregated, None, None)
                .unwrap()
                .llm
                .unwrap()
                .max_num_batched_tokens,
            Some(512)
        );
        let llm = build_engine_config(
            &discovery(|_| {}),
            DisaggregationMode::Aggregated,
            None,
            None,
        )
        .unwrap()
        .llm
        .unwrap();
        assert_eq!(
            (
                llm.context_length,
                llm.max_num_seqs,
                llm.max_num_batched_tokens
            ),
            (None, None, None)
        );
        assert_eq!(
            (llm.data_parallel_start_rank, llm.data_parallel_size),
            (Some(0), Some(1))
        );
    }

    #[test]
    fn registration_requires_bootstrap_address_only_for_prefill() {
        for (host, port) in [
            (None, None),
            (Some("prefill.example"), None),
            (None, Some(8998)),
        ] {
            let error = build_engine_config(
                &discovery(|_| {}),
                DisaggregationMode::Prefill,
                host.map(str::to_string),
                port,
            )
            .unwrap_err();
            assert_eq!(
                error.error_type(),
                ErrorType::Backend(BackendError::Unknown)
            );
            assert!(error.to_string().contains("usable bootstrap address"));
        }
        for mode in [
            DisaggregationMode::Aggregated,
            DisaggregationMode::Decode,
            DisaggregationMode::Prefill,
        ] {
            let llm = build_engine_config(
                &discovery(|_| {}),
                mode,
                Some("prefill.example".into()),
                Some(8998),
            )
            .unwrap()
            .llm
            .unwrap();
            assert_eq!(
                llm.bootstrap_host.as_deref(),
                mode.is_prefill().then_some("prefill.example")
            );
            assert_eq!(llm.bootstrap_port, mode.is_prefill().then_some(8998));
        }
    }

    #[tokio::test]
    async fn kv_event_sources_match_registration_and_publish_the_worker_group() {
        // Node 0 of a two-node tensor-parallel deployment with one DP rank.
        let info = |block_size: u32, sources: bool| {
            discovery(|reported| {
                reported.deployment.nnodes = 2;
                reported.deployment.page_size = Some(64);
                reported.deployment.dcp_size = 2;
                reported.deployment.dist_init_addr = Some("127.0.0.1:2345".into());
                if sources {
                    reported.deployment.kv_event_sources = vec![pb::KvEventSource {
                        dp_rank: 0,
                        endpoint: "tcp://127.0.0.1:5557".into(),
                        topic: "kv".into(),
                        block_size,
                    }];
                }
            })
        };
        async fn resolve(
            info: EngineInfo,
        ) -> Result<(Vec<DiscoveredKvEventSource>, EngineConfig), DynamoError> {
            let mut config =
                build_engine_config(&info, DisaggregationMode::Aggregated, None, None).unwrap();
            discover_kv_event_sources(&info, &mut config, Instant::now() + Duration::from_secs(1))
                .await
                .map(|sources| (sources, config))
        }

        let (sources, config) = resolve(info(128, true)).await.unwrap();
        assert_eq!(
            sources,
            [DiscoveredKvEventSource {
                endpoint: "tcp://127.0.0.1:5557".to_string(),
                topic: "kv".to_string(),
                dp_rank: 0,
            }]
        );
        assert_eq!(
            config.runtime_data[WORKER_GROUP_KEY],
            "dist_init:tcp://127.0.0.1:2345"
        );

        // An empty list is authoritative and still publishes the group.
        let (sources, config) = resolve(info(128, false)).await.unwrap();
        assert!(sources.is_empty());
        assert!(config.runtime_data.contains_key(WORKER_GROUP_KEY));

        let error = resolve(info(64, true)).await.unwrap_err();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert!(
            error
                .to_string()
                .contains("block_size does not match the registration"),
            "{error}"
        );

        // A single-node engine has no worker group.
        let (_, config) = resolve(discovery(|_| {})).await.unwrap();
        assert!(!config.runtime_data.contains_key(WORKER_GROUP_KEY));
    }

    #[test]
    fn eagle_enabled_tracks_sglang_is_eagle_predicate() {
        let algorithm = |value: Option<&str>| {
            discovery(|reported| {
                reported.deployment.speculative_algorithm = value.map(str::to_string);
            })
            .speculative_algorithm
        };
        for name in [
            "EAGLE",
            "eagle",
            "EAGLE3",
            "FROZEN_KV_MTP",
            "NEXTN",
            " Eagle ",
        ] {
            assert!(
                sglang_eagle_enabled(algorithm(Some(name)).as_deref()),
                "{name:?} keys the radix cache by bigrams and must advertise enable_eagle"
            );
        }
        for name in [None, Some("NONE"), Some("NGRAM"), Some("STANDALONE")] {
            assert!(
                !sglang_eagle_enabled(algorithm(name).as_deref()),
                "{name:?} must not advertise enable_eagle"
            );
        }
    }

    #[test]
    fn combined_token_limit_subtracts_the_reported_draft_reserve() {
        let limit = |context: u32, reserved: u64| {
            combined_token_limit(&discovery(|reported| {
                reported.server.max_context_length = context;
                reported.server.num_reserved_tokens = reserved;
            }))
        };
        assert_eq!(limit(4096, 0), Some(4096));
        assert_eq!(limit(4096, 4), Some(4092));
        assert_eq!(limit(4, 5), None);
        // Zero is proto3's unset context length.
        assert_eq!(limit(0, 0), None);
    }

    /// EAGLE reserves at least one draft-token slot, so an EAGLE server that
    /// reports none predates the field. Its reserve is unknown: no budget is
    /// published and omitted max_tokens keep SGLang's default, rather than a
    /// budget the draft tokens would overrun.
    #[test]
    fn eagle_without_a_reported_draft_reserve_publishes_no_budget() {
        let info = |algorithm: Option<&str>, reserved: u64| {
            discovery(|reported| {
                reported.server.max_context_length = 4096;
                reported.server.num_reserved_tokens = reserved;
                reported.deployment.speculative_algorithm = algorithm.map(str::to_string);
            })
        };
        let budget = |info: &EngineInfo| {
            build_engine_config(info, DisaggregationMode::Aggregated, None, None)
                .unwrap()
                .runtime_data
                .remove("token_budget")
        };
        let old_eagle = info(Some("EAGLE"), 0);
        assert!(lacks_draft_reserve(&old_eagle));
        assert_eq!(combined_token_limit(&old_eagle), None);
        assert_eq!(budget(&old_eagle), None);

        assert!(!lacks_draft_reserve(&info(Some("EAGLE"), 4)));
        // Other algorithms reserve nothing, so zero is their real reserve.
        for algorithm in [None, Some("NGRAM")] {
            let info = info(algorithm, 0);
            assert!(!lacks_draft_reserve(&info), "{algorithm:?}");
            assert_eq!(combined_token_limit(&info), Some(4096), "{algorithm:?}");
            assert!(budget(&info).is_some(), "{algorithm:?}");
        }
    }

    #[test]
    fn reasoning_gate_follows_the_engine_reasoning_parser() {
        let gate = |parser: Option<&str>, skip_tokenizer_init: bool| {
            build_engine_config(
                &discovery(|reported| {
                    reported.model.reasoning_parser = parser.map(str::to_string);
                    reported.server.skip_tokenizer_init = skip_tokenizer_init;
                }),
                DisaggregationMode::Aggregated,
                None,
                None,
            )
            .unwrap()
            .runtime_data
            .remove("tool_call_structural_tag_reasoning_gate")
        };
        assert_eq!(gate(Some("kimi_k2"), false), Some(json!(true)));
        for (parser, skip_tokenizer_init) in
            [(None, false), (Some(""), false), (Some("kimi_k2"), true)]
        {
            assert_eq!(
                gate(parser, skip_tokenizer_init),
                None,
                "{parser:?} {skip_tokenizer_init}"
            );
        }
    }

    #[test]
    fn token_budget_follows_sglang_auto_truncation() {
        let budget = |context: u32, reserved: u64, truncates: bool| {
            build_engine_config(
                &discovery(|reported| {
                    reported.server.max_context_length = context;
                    reported.server.num_reserved_tokens = reserved;
                    reported.server.allow_auto_truncate = truncates;
                }),
                DisaggregationMode::Aggregated,
                None,
                None,
            )
            .unwrap()
            .runtime_data
            .remove("token_budget")
        };
        assert_eq!(
            budget(4096, 0, false),
            Some(json!({
                "combined_limit": 4096,
                "reject_prompt_overflow": true,
                "reject_total_overflow": true,
            }))
        );
        assert_eq!(
            budget(4096, 4, true),
            Some(json!({
                "combined_limit": 4092,
                "reject_prompt_overflow": false,
                "reject_total_overflow": false,
            }))
        );
        assert_eq!(budget(0, 0, false), None);
        assert_eq!(budget(4, 5, false), None);
    }

    #[test]
    fn omitted_max_tokens_default_fills_the_remaining_budget() {
        assert_eq!(omitted_max_tokens_default(Some(4096), 3, None), Some(4093));
        assert_eq!(omitted_max_tokens_default(Some(8), 3, Some(5)), Some(5));
        assert_eq!(omitted_max_tokens_default(Some(8), 3, Some(6)), None);
        assert_eq!(omitted_max_tokens_default(Some(3), 3, None), None);
        assert_eq!(omitted_max_tokens_default(None, 3, None), None);
    }

    #[test]
    fn registration_advertises_enable_eagle_for_eagle_servers() {
        let enable_eagle = |algorithm: Option<&str>| {
            build_engine_config(
                &discovery(|reported| {
                    reported.deployment.speculative_algorithm = algorithm.map(str::to_string);
                    reported.deployment.page_size = Some(256);
                }),
                DisaggregationMode::Aggregated,
                None,
                None,
            )
            .unwrap()
            .llm
            .unwrap()
            .enable_eagle
        };
        assert!(enable_eagle(Some("EAGLE")));
        assert!(!enable_eagle(None));
    }

    #[test]
    fn explicit_bootstrap_host_takes_precedence() {
        let host = resolve_bootstrap_host_with_local(
            Some("prefill.example"),
            "http://127.0.0.1:30001",
            &hosts("", Some("10.0.0.1:20000")),
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
            &hosts("10.0.0.1", Some("10.0.0.2:20000")),
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
            &hosts("0.0.0.0", Some("10.0.0.1:20000")),
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
            &hosts("", None),
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
            &hosts("", Some("0.0.0.0:20000")),
            "127.0.0.1",
        )
        .unwrap_err();
        assert!(error.to_string().contains("--bootstrap-host"));
    }

    #[test]
    fn dcp_registers_logical_kv_block_size() {
        let config = build_engine_config(
            &discovery(|reported| {
                reported.deployment.page_size = Some(64);
                reported.deployment.dcp_size = 8;
                reported.server.max_total_num_tokens = Some(1024);
            }),
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
    fn publishes_hicache_capacity() {
        let config = build_engine_config(
            &discovery(|reported| {
                reported.deployment.enable_hierarchical_cache = true;
                reported.deployment.hicache_size = 0;
                reported.deployment.hicache_ratio = Some(3.0);
                reported.deployment.hicache_write_policy = Some("write_through".into());
                reported.deployment.page_size = Some(16);
                reported.server.max_total_num_tokens = Some(100);
            }),
            DisaggregationMode::Decode,
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            config.runtime_data.get("native_offloading_capacity"),
            Some(&json!({"total_tokens": 204}))
        );
    }

    #[test]
    fn supports_deepseek_v4_hicache_capacity() {
        let capacity = |architecture: &str, host_total_tokens: Option<u64>| {
            build_engine_config(
                &discovery(|reported| {
                    reported.model.architectures = vec![architecture.to_string()];
                    reported.deployment.enable_hierarchical_cache = true;
                    reported.deployment.hicache_size = 0;
                    reported.deployment.hicache_ratio = Some(2.0);
                    reported.deployment.hicache_write_policy = Some("write_back".into());
                    reported.deployment.hicache_host_total_tokens = host_total_tokens;
                    reported.deployment.page_size = Some(256);
                    reported.server.max_total_num_tokens = Some(8192);
                }),
                DisaggregationMode::Decode,
                None,
                None,
            )
            .unwrap()
            .runtime_data
            .remove("native_offloading_capacity")
        };
        for architecture in [
            "DeepseekV4ForCausalLM",
            "DeepseekV4ForCausalLMNextN",
            "DeepseekV4ForCausalLMDSpark",
        ] {
            assert_eq!(
                capacity(architecture, None),
                Some(json!({"total_tokens": 16384})),
                "{architecture}"
            );
        }
        assert_eq!(
            capacity("DeepseekV4ForCausalLM", Some(16640)),
            Some(json!({"total_tokens": 16640}))
        );
    }
}
