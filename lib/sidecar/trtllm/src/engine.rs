// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo backend for TensorRT-LLM's OpenEngine (`openengine.v1`) gRPC server.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, ComponentSnapshot, DisaggregationMode, DynamoError, EngineConfig,
    GenerateContext, KvEventSource, LLMEngine, LLMEngineOutput, LLMEngineOutputExt,
    MetricsBindings, MetricsCtx, PreprocessedRequest, WorkerConfig, usage,
};
use dynamo_sidecar_common::{
    GrpcEndpoint, GrpcTransportConfig, SidecarStartupError, startup_deadline,
};
use futures::stream::BoxStream;
use tokio::sync::OnceCell;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

use crate::args::Args;
use crate::client::{self, ModelLimits, TrtllmClient};
use crate::convert::{ResponseState, build_generate_request_with_routing};
use crate::model::ConfiguredModel;
use crate::proto as pb;

const ALREADY_STARTED: &str = "TensorRT-LLM sidecar has already started";
const LOAD_POLL_INTERVAL: Duration = Duration::from_millis(100);
const MAX_INITIAL_LOAD_AGE: Duration = Duration::from_secs(5);
const METRICS_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);

struct RoutingState {
    dp_size: u32,
    dp_targeting: bool,
    heartbeat_timeout: Option<Duration>,
    sources: Vec<pb::KvEventSource>,
    initial_snapshots: Vec<ComponentSnapshot>,
}

/// Terminal output emitted when a request is cancelled, carrying the usage
/// accumulated so far.
fn cancelled(state: &ResponseState) -> LLMEngineOutput {
    LLMEngineOutput::cancelled().with_usage(usage(state.prompt_tokens(), state.completion_tokens()))
}

pub struct TrtllmSidecarEngine {
    endpoint: GrpcEndpoint,
    transport: GrpcTransportConfig,
    model: ConfiguredModel,
    /// Disaggregation role this worker plays. Selects the `context_only` /
    /// `kv.session` divergence in `convert`.
    mode: DisaggregationMode,
    client: OnceCell<TrtllmClient>,
    /// Engine limits resolved at `start` from `--context-length` and
    /// `Control.GetModelInfo`, so `generate` can derive a default `max_tokens`
    /// for requests that omit one.
    limits: OnceCell<ModelLimits>,
    routing: OnceCell<RoutingState>,
    metrics_task: Arc<Mutex<Option<JoinHandle<()>>>>,
    cancel: CancellationToken,
}

impl TrtllmSidecarEngine {
    pub(crate) fn new(
        endpoint: GrpcEndpoint,
        transport: GrpcTransportConfig,
        model: ConfiguredModel,
        mode: DisaggregationMode,
    ) -> Self {
        Self {
            endpoint,
            transport,
            model,
            mode,
            client: OnceCell::new(),
            limits: OnceCell::new(),
            routing: OnceCell::new(),
            metrics_task: Arc::new(Mutex::new(None)),
            cancel: CancellationToken::new(),
        }
    }

    pub fn from_env() -> Result<(Self, WorkerConfig), DynamoError> {
        Self::from_parsed(<Args as clap::Parser>::parse())
    }

    pub fn from_args(argv: Vec<String>) -> Result<(Self, WorkerConfig), DynamoError> {
        Self::try_from_args(argv).map_err(SidecarStartupError::into_dynamo)
    }

    /// Parse injected arguments while retaining Clap's structured exit error.
    ///
    /// Embedded callers use this to distinguish help and version output from
    /// Dynamo startup failures without changing `from_args`'s error contract.
    pub fn try_from_args(argv: Vec<String>) -> Result<(Self, WorkerConfig), SidecarStartupError> {
        let args = <Args as clap::Parser>::try_parse_from(argv)?;
        Self::from_parsed(args).map_err(Into::into)
    }

    fn from_parsed(args: Args) -> Result<(Self, WorkerConfig), DynamoError> {
        if args.model_path.trim().is_empty() {
            return Err(client::invalid_argument("model-path must not be empty"));
        }
        let mode = args.sidecar.common.disaggregation_mode;
        if mode.is_encode() {
            return Err(client::invalid_argument(
                "encode mode is not supported by the TensorRT-LLM sidecar",
            ));
        }
        if args.sidecar.common.route_to_encoder {
            return Err(client::invalid_argument(
                "route-to-encoder is not supported by the TensorRT-LLM sidecar",
            ));
        }

        let endpoint = args.sidecar.grpc_endpoint;
        let transport = args.sidecar.grpc.config();
        let model = ConfiguredModel {
            source: args.model_path,
            // Absent unless `--context-length` supplied one; `start` falls back
            // to the server's `Control.GetModelInfo` report.
            context_length: args.context_length,
            kv_cache_block_size: None,
            total_kv_blocks: None,
            max_num_seqs: None,
            max_num_batched_tokens: None,
            data_parallel_size: None,
            data_parallel_start_rank: None,
        };
        let engine = Self::new(endpoint, transport, model.clone(), mode);
        let config = WorkerConfig {
            namespace: args.sidecar.common.namespace,
            // Every disaggregated role registers under its own component so
            // the frontend can target each separately; only an aggregated
            // worker uses the operator-configured one.
            component: if mode == DisaggregationMode::Aggregated {
                args.sidecar.common.component
            } else {
                mode.discovery_component().to_string()
            },
            endpoint: args.sidecar.common.endpoint,
            endpoint_types: args.sidecar.common.endpoint_types,
            custom_jinja_template: args.sidecar.common.custom_jinja_template,
            model_name: model.source.clone(),
            served_model_name: None,
            tool_call_parser: args.sidecar.common.dyn_tool_call_parser,
            reasoning_parser: args.sidecar.common.dyn_reasoning_parser,
            exclude_tools_when_tool_choice_none: args
                .sidecar
                .common
                .exclude_tools_when_tool_choice_none,
            // Framework wiring is available; discovered source declarations
            // independently decide whether any event or metrics work starts.
            enable_kv_routing: true,
            disaggregation_mode: mode,
            route_to_encoder: false,
            enable_rl: args.sidecar.common.enable_rl,
            ..Default::default()
        };
        Ok((engine, config))
    }
}

#[async_trait]
impl LLMEngine for TrtllmSidecarEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.client.initialized() {
            return Err(client::engine_shutdown(ALREADY_STARTED));
        }
        tracing::info!(
            endpoint = %self.endpoint,
            connections = self.transport.connections.get(),
            "connecting to TensorRT-LLM gRPC"
        );
        let client = TrtllmClient::connect(&self.endpoint, self.transport).await?;
        let connection_count = client.connection_count();

        // `--context-length` wins over what the engine reports, and is the
        // only source when the engine reports nothing usable. The resolved
        // value backs both the registered window and the default-`max_tokens`
        // path in `convert::max_tokens`.
        let mut model = self.model.clone();
        let limits = match model.context_length {
            // Configured: the engine is consulted once, to cross-check the
            // value and to learn its output cap. It may not answer at all --
            // an older server has no Control service -- and that must not stop
            // a worker whose window the operator already supplied.
            Some(configured) => {
                let reported = match client.model_limits(&model.source).await {
                    Ok(reported) => reported,
                    Err(error) => {
                        tracing::warn!(
                            %error,
                            configured_context_length = configured,
                            "Control.GetModelInfo failed; using the configured --context-length"
                        );
                        ModelLimits::default()
                    }
                };
                if let Some(engine_context_length) = reported.context_length
                    && engine_context_length != configured
                {
                    tracing::warn!(
                        configured_context_length = configured,
                        engine_context_length,
                        "--context-length disagrees with the context length TensorRT-LLM \
                         reported; using the configured --context-length"
                    );
                }
                ModelLimits {
                    context_length: Some(configured),
                    ..reported
                }
            }
            // Nothing configured: the engine is the only source, and it binds
            // its port before the model finishes loading, so wait for it.
            None => {
                match client
                    .wait_for_model_limits(
                        &model.source,
                        startup_deadline(self.transport.startup_deadline)?,
                        self.transport.retry_interval,
                    )
                    .await
                {
                    Ok(limits) => limits,
                    // Registering without a window is still useful: requests
                    // that carry their own `max_tokens` are served, and only
                    // the ones that omit it are rejected.
                    Err(error) => {
                        tracing::warn!(
                            %error,
                            "no context length is available; requests that omit max_tokens \
                             will be rejected. Supply --context-length."
                        );
                        ModelLimits::default()
                    }
                }
            }
        };
        model.context_length = limits.context_length;
        let _ = self.limits.set(limits);

        {
            let routing_deadline = startup_deadline(self.transport.startup_deadline)?;
            let routing_discovery = async {
                let server = client.server_info().await?.unwrap_or_default();
                let metadata = server.extra.unwrap_or_default().fields;
                let dp_targeting = matches!(
                    metadata
                        .get("trtllm_supports_dp_rank_targeting")
                        .and_then(|value| value.kind.as_ref()),
                    Some(prost_types::value::Kind::BoolValue(true))
                );
                let heartbeat_timeout =
                    metadata
                        .get("kv_event_heartbeat_interval_ms")
                        .and_then(|value| match value.kind {
                            Some(prost_types::value::Kind::NumberValue(ms))
                                if ms.is_finite() && ms > 0.0 && ms <= u32::MAX as f64 =>
                            {
                                Some(Duration::from_millis(ms as u64 * 3))
                            }
                            _ => None,
                        });
                let sources = client.kv_event_sources().await?;
                let events_enabled = !sources.is_empty();
                let load = client.get_load(true).await?;
                let load_enabled = load.used_kv_blocks.is_some()
                    || load.total_kv_blocks.is_some()
                    || !load.ranks.is_empty();
                let parallelism = server.parallelism.unwrap_or_default();
                let dp_size = parallelism.data_parallel_size.unwrap_or(1).max(1);
                let capacity = server.capacity.unwrap_or_default();
                model.kv_cache_block_size = capacity.kv_block_size.filter(|value| *value > 0);
                model.total_kv_blocks = capacity.total_kv_blocks.filter(|value| *value > 0);
                model.max_num_seqs = capacity.max_running_requests.filter(|value| *value > 0);
                model.max_num_batched_tokens =
                    capacity.max_batched_tokens.filter(|value| *value > 0);
                model.data_parallel_size = Some(dp_size);
                model.data_parallel_start_rank =
                    Some(parallelism.data_parallel_start_rank.unwrap_or(0));
                if (events_enabled || load_enabled)
                    && (model.kv_cache_block_size.is_none() || model.total_kv_blocks.is_none())
                {
                    return Err(client::protocol_error(
                        "KV routing requires positive kv_block_size and total_kv_blocks",
                    ));
                }

                if events_enabled {
                    validate_kv_sources(&sources, dp_size)?;
                }
                let initial_snapshots = if load_enabled {
                    load_snapshots(load, dp_size)?
                } else {
                    Vec::new()
                };
                tracing::info!(
                    events_enabled,
                    load_enabled,
                    dp_targeting,
                    dp_size,
                    "Discovered OpenEngine routing capabilities"
                );
                Ok::<RoutingState, DynamoError>(RoutingState {
                    dp_size,
                    dp_targeting,
                    heartbeat_timeout,
                    sources,
                    initial_snapshots,
                })
            };
            let routing = tokio::time::timeout_at(routing_deadline, routing_discovery)
                .await
                .map_err(|_| {
                    client::protocol_error(
                        "KV routing discovery exceeded the gRPC startup deadline",
                    )
                })??;
            self.routing
                .set(routing)
                .map_err(|_| client::engine_shutdown(ALREADY_STARTED))?;
        }

        self.client
            .set(client)
            .map_err(|_| client::engine_shutdown(ALREADY_STARTED))?;
        tracing::info!(
            endpoint = %self.endpoint,
            connections = connection_count,
            model = %model.source,
            context_length = ?limits.context_length,
            max_output_tokens = ?limits.max_output_tokens,
            "TensorRT-LLM gRPC is ready"
        );
        Ok(model.engine_config())
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let client = self
            .client
            .get()
            .ok_or_else(|| client::engine_shutdown("TensorRT-LLM sidecar is not started"))?;
        let request_id = ctx.id().to_string();
        let proto_request = build_generate_request_with_routing(
            &request,
            &request_id,
            &self.model.source,
            self.limits.get().copied(),
            self.mode,
            self.routing
                .get()
                .is_some_and(|routing| routing.dp_targeting),
        )?;
        let target_dp_rank = request.routing.as_ref().and_then(|routing| {
            if self.mode.is_prefill() {
                routing.prefill_dp_rank.or(routing.dp_rank)
            } else {
                routing.dp_rank
            }
        });
        if let Some(rank) = target_dp_rank {
            let dp_size = self.routing.get().map_or(1, |routing| routing.dp_size);
            if rank >= dp_size {
                return Err(client::invalid_argument(format!(
                    "data-parallel rank {rank} is outside the engine range 0..{dp_size}",
                )));
            }
        }
        let mut state = ResponseState::new(&request, self.mode);
        let cancel = self.cancel.clone();
        // A decode request that took a handoff has KV transferred into it, and
        // the transceiver releases those blocks when the engine finishes the
        // request -- not when the client goes away. Dropping the stream on
        // cancellation would strand the prefill worker's blocks. Defer only
        // until the first token proves the transfer landed; deferring past
        // that would let a cancelled request generate its whole budget with no
        // consumer. A decode-mode request without a handoff ran locally and
        // has nothing to strand.
        let defer_request_cancellation = self.mode.is_decode() && request.prefill_result.is_some();
        let stopped_ctx = ctx.inner_arc();
        // Hoisted: `stopped()` is an async-trait method, so re-creating it per
        // streamed chunk costs a boxed future and a waker registration on every
        // token.
        let mut request_cancellation = Box::pin(async move { stopped_ctx.stopped().await });
        let shutdown = cancel.clone();
        let mut shutdown_cancellation = Box::pin(async move { shutdown.cancelled().await });

        // The same deferral applies here, not just to the streaming loop below.
        // `generate` sends the request and then awaits response headers, so
        // losing this race can drop a request the engine has already accepted
        // and begun pulling KV for; and an already-stopped context would skip
        // the dispatch entirely, leaving the prefill worker's blocks with no
        // decode leg to claim them. Both strand exactly what the deferral
        // exists to protect. Shutdown still wins -- the process is going away.
        let stream = tokio::select! {
            biased;
            _ = &mut request_cancellation, if !defer_request_cancellation => None,
            _ = &mut shutdown_cancellation => None,
            result = client.generate(proto_request, target_dp_rank) => Some(result?),
        };
        let Some(mut stream) = stream else {
            let output = cancelled(&state);
            return Ok(Box::pin(futures::stream::once(async move { Ok(output) })));
        };

        Ok(Box::pin(async_stream::stream! {
            let mut transfer_settled = false;
            loop {
                tokio::select! {
                    biased;
                    _ = &mut request_cancellation, if !defer_request_cancellation || transfer_settled => {
                        yield Ok(cancelled(&state));
                        break;
                    }
                    _ = &mut shutdown_cancellation => {
                        yield Ok(cancelled(&state));
                        break;
                    }
                    message = stream.message() => {
                        match message {
                            Ok(Some(response)) => match state.convert(response) {
                                Ok(Some(output)) => {
                                    transfer_settled |= !output.token_ids.is_empty();
                                    let terminal = output.finish_reason.is_some();
                                    yield Ok(output);
                                    if terminal {
                                        break;
                                    }
                                }
                                Ok(None) => {}
                                Err(error) => {
                                    yield Err(error);
                                    break;
                                }
                            },
                            Ok(None) => {
                                yield Err(client::protocol_error(
                                    "Generate ended before a terminal response",
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
        let Some(client) = self.client.get() else {
            return;
        };
        if let Err(error) = client.abort(ctx.id().to_string()).await {
            // Escaped: the message embeds the engine's gRPC status text, and
            // this site logs at the default level, so raw newlines from the
            // peer would let it forge what look like separate log records.
            tracing::warn!(
                request_id = ctx.id(),
                error = %error.to_string().escape_debug(),
                "TensorRT-LLM Control.Abort failed"
            );
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.cancel.cancel();
        let task = self
            .metrics_task
            .lock()
            .expect("metrics task mutex poisoned")
            .take();
        if let Some(task) = task {
            let mut task = task;
            if timeout(METRICS_SHUTDOWN_TIMEOUT, &mut task).await.is_err() {
                task.abort();
            }
        }
        tracing::info!("TensorRT-LLM sidecar shutdown complete");
        Ok(())
    }

    async fn kv_event_sources(&self) -> Result<Vec<KvEventSource>, DynamoError> {
        let routing = self
            .routing
            .get()
            .ok_or_else(|| client::engine_shutdown("TensorRT-LLM sidecar is not started"))?;
        routing
            .sources
            .iter()
            .map(|source| to_kv_event_source(source, routing.heartbeat_timeout))
            .collect()
    }

    async fn setup_metrics(&self, _ctx: MetricsCtx<'_>) -> Result<MetricsBindings, DynamoError> {
        let routing = self
            .routing
            .get()
            .ok_or_else(|| client::engine_shutdown("TensorRT-LLM sidecar is not started"))?;
        if routing.initial_snapshots.is_empty() {
            return Ok(MetricsBindings::default());
        }
        let client = self
            .client
            .get()
            .ok_or_else(|| client::engine_shutdown("TensorRT-LLM sidecar is not started"))?
            .clone();
        let dp_size = routing.dp_size;
        let initial = routing.initial_snapshots.clone();
        let cancel = self.cancel.clone();
        let task_slot = Arc::clone(&self.metrics_task);

        let bindings = MetricsBindings {
            dp_ranks: (0..dp_size).collect(),
            on_publisher_ready: Some(Box::new(move |publisher| {
                for snapshot in initial {
                    publisher.publish(snapshot.dp_rank, snapshot);
                }
                let task = tokio::spawn(async move {
                    let mut interval = tokio::time::interval(LOAD_POLL_INTERVAL);
                    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    let mut warned = false;
                    loop {
                        tokio::select! {
                            biased;
                            _ = cancel.cancelled() => break,
                            _ = interval.tick() => {
                                let load = tokio::select! {
                                    biased;
                                    _ = cancel.cancelled() => break,
                                    load = client.get_load(true) => load,
                                };
                                match load.and_then(|load| {
                                    load_snapshots(load, dp_size)
                                }) {
                                    Ok(snapshots) => {
                                        warned = false;
                                        for snapshot in snapshots {
                                            publisher.publish(snapshot.dp_rank, snapshot);
                                        }
                                    }
                                    Err(error) if !warned => {
                                        warned = true;
                                        tracing::warn!(%error, "GetLoad poll failed; suppressing repeated warnings");
                                    }
                                    Err(error) => tracing::debug!(%error, "GetLoad poll failed"),
                                }
                            }
                        }
                    }
                });
                *task_slot.lock().expect("metrics task mutex poisoned") = Some(task);
                Ok(())
            })),
        };
        Ok(bindings)
    }
}

fn validate_kv_sources(sources: &[pb::KvEventSource], dp_size: u32) -> Result<(), DynamoError> {
    let mut ranks = HashSet::new();
    for source in sources {
        if source.transport != "zmq" || source.encoding != "msgpack" {
            return Err(client::protocol_error(
                "KV routing requires ZMQ sources with msgpack encoding",
            ));
        }
        if source.schema_version != Some(1) {
            return Err(client::protocol_error(
                "GetKvEventSources returned an unsupported schema_version",
            ));
        }
        let rank = source.data_parallel_rank.ok_or_else(|| {
            client::protocol_error("GetKvEventSources omitted data_parallel_rank")
        })?;
        if rank >= dp_size || !ranks.insert(rank) {
            return Err(client::protocol_error(format!(
                "GetKvEventSources returned invalid or duplicate rank {rank}",
            )));
        }
        let endpoint = source
            .endpoint_addr
            .as_ref()
            .ok_or_else(|| client::protocol_error("GetKvEventSources omitted endpoint_addr"))?;
        if endpoint.protocol != "tcp"
            || endpoint.host.trim().is_empty()
            || matches!(endpoint.host.as_str(), "*" | "0.0.0.0" | "::")
            || endpoint.port == 0
        {
            return Err(client::protocol_error(
                "GetKvEventSources returned an invalid TCP endpoint",
            ));
        }
    }
    if ranks.len() != dp_size as usize {
        return Err(client::protocol_error(format!(
            "GetKvEventSources returned {} sources for {dp_size} data-parallel ranks",
            ranks.len(),
        )));
    }
    Ok(())
}

fn to_kv_event_source(
    source: &pb::KvEventSource,
    heartbeat_timeout: Option<Duration>,
) -> Result<KvEventSource, DynamoError> {
    let endpoint = source
        .endpoint_addr
        .as_ref()
        .ok_or_else(|| client::protocol_error("GetKvEventSources omitted endpoint_addr"))?;
    let host = if endpoint.host.contains(':') {
        format!("[{}]", endpoint.host)
    } else {
        endpoint.host.clone()
    };
    Ok(KvEventSource::Zmq {
        heartbeat_timeout,
        endpoint: format!("tcp://{host}:{}", endpoint.port),
        topic: source.topic.clone(),
        dp_rank: source.data_parallel_rank.ok_or_else(|| {
            client::protocol_error("GetKvEventSources omitted data_parallel_rank")
        })?,
    })
}

fn load_snapshots(load: pb::LoadInfo, dp_size: u32) -> Result<Vec<ComponentSnapshot>, DynamoError> {
    let timestamp = load
        .timestamp_unix_nanos
        .ok_or_else(|| client::protocol_error("GetLoad omitted timestamp_unix_nanos"))?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| client::protocol_error(error.to_string()))?
        .as_nanos();
    let timestamp = u128::from(timestamp);
    let future_tolerance = Duration::from_secs(1).as_nanos();
    if now.saturating_sub(timestamp) > MAX_INITIAL_LOAD_AGE.as_nanos()
        || timestamp.saturating_sub(now) > future_tolerance
    {
        return Err(client::protocol_error("GetLoad returned a stale snapshot"));
    }
    let mut ranks = HashSet::new();
    let mut snapshots = Vec::with_capacity(load.ranks.len());
    for rank in load.ranks {
        let dp_rank = rank
            .data_parallel_rank
            .ok_or_else(|| client::protocol_error("GetLoad rank omitted data_parallel_rank"))?;
        let used = rank
            .used_kv_blocks
            .ok_or_else(|| client::protocol_error("GetLoad rank omitted used_kv_blocks"))?;
        let total = rank
            .total_kv_blocks
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                client::protocol_error("GetLoad rank omitted positive total_kv_blocks")
            })?;
        if dp_rank >= dp_size || !ranks.insert(dp_rank) || used > total {
            return Err(client::protocol_error(format!(
                "GetLoad returned invalid rank {dp_rank} counters",
            )));
        }
        snapshots.push(ComponentSnapshot {
            kv_used_blocks: used,
            kv_total_blocks: total,
            gpu_cache_usage: used as f32 / total as f32,
            kv_cache_hit_rate: None,
            dp_rank,
        });
    }
    if ranks.len() != dp_size as usize {
        return Err(client::protocol_error(format!(
            "GetLoad returned {} snapshots for {dp_size} data-parallel ranks",
            ranks.len(),
        )));
    }
    Ok(snapshots)
}
