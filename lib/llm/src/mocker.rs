// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mocker module - runtime integration for the mock scheduler.
//!
//! The core mocker logic lives in the `dynamo-mocker` crate.
//! This module provides the runtime-dependent engine wrapper.

mod handoff;
mod metrics;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use crate::backend::ExecutionContext;
use crate::kv_router::publisher::{KvEventPublisher, KvEventSourceConfig, WorkerMetricsPublisher};
use crate::protocols::TokenIdType;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use anyhow::{Context, Result, bail};
use dynamo_kv_router::protocols::{KvCacheEvent, StorageTier};
use dynamo_mocker::common::handoff::HandoffId;
use dynamo_mocker::common::protocols::{
    DirectRequest, KvCacheEventSink, KvEventPublishers, MockEngineArgs, RawKvEventSink,
};
use dynamo_mocker::live::{LiveEngine, LiveEngineConfig};
use dynamo_mocker::loadgen::{OUTPUT_REPLAY_ID_ANNOTATION_KEY, effective_replay_key};
use dynamo_mocker::services::bootstrap::{
    BootstrapIdentity, BootstrapParticipantRole, BootstrapServer, BootstrapServerConfig,
    ParticipantRegistration, connect_to_prefill,
};
use dynamo_mocker::services::zmq_events::ZmqKvEventSink;
use dynamo_protocols::types::{CompletionUsage, PromptTokensDetails};
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::{
    component::Endpoint,
    engine::{AsyncEngineContext, AsyncEngineContextProvider},
    pipeline::{AsyncEngine, Error, ManyOut, ResponseStream, SingleIn, async_trait},
    traits::DistributedRuntimeProvider,
};
use futures::StreamExt;
use rand::Rng;
use serde::Deserialize;
use tokio::sync::{OnceCell, Semaphore, mpsc, oneshot, watch};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use uuid::Uuid;

use self::handoff::{
    HandoffControl, SourceHandoffManager, SourceRegistration, cancel_destination,
    live_handoff_boundary, order_for_engine, run_destination_session,
};
use self::metrics::NativeMockerMetrics;

pub const MOCKER_COMPONENT: &str = "mocker";

#[derive(Debug, Clone, Deserialize)]
struct ResponseReplayTraceRow {
    #[serde(default)]
    request_id: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default, alias = "output_tokens")]
    output_length: Option<usize>,
    #[serde(default)]
    output_token_ids: Option<Vec<TokenIdType>>,
}

#[derive(Debug, Clone, Default)]
struct ResponseReplayTable {
    rows: HashMap<String, Vec<TokenIdType>>,
}

impl ResponseReplayTable {
    fn from_path(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open response replay trace {}", path.display()))?;
        let reader = BufReader::new(file);
        let mut rows = HashMap::new();
        let mut session_turns: HashMap<String, usize> = HashMap::new();

        for (line_index, line) in reader.lines().enumerate() {
            let line = line.with_context(|| {
                format!(
                    "failed to read line {} from response replay trace {}",
                    line_index + 1,
                    path.display()
                )
            })?;
            if line.trim().is_empty() {
                continue;
            }

            let row: ResponseReplayTraceRow = serde_json::from_str(&line).with_context(|| {
                format!(
                    "failed to parse line {} from response replay trace {}",
                    line_index + 1,
                    path.display()
                )
            })?;
            let turn_index = row
                .session_id
                .as_ref()
                .map(|session_id| {
                    let entry = session_turns.entry(session_id.clone()).or_default();
                    let turn_index = *entry;
                    *entry += 1;
                    turn_index
                })
                .unwrap_or(0);

            let Some(output_token_ids) = row.output_token_ids else {
                continue;
            };
            let output_length = row.output_length.ok_or_else(|| {
                anyhow::anyhow!(
                    "response replay trace line {} has output_token_ids but no output_length",
                    line_index + 1
                )
            })?;
            if output_length != output_token_ids.len() {
                bail!(
                    "response replay trace line {} output_length {} does not match output_token_ids length {}",
                    line_index + 1,
                    output_length,
                    output_token_ids.len()
                );
            }

            let key = effective_replay_key(
                row.request_id.as_deref(),
                row.session_id.as_deref(),
                turn_index,
                line_index,
            );
            if rows.insert(key.clone(), output_token_ids).is_some() {
                bail!(
                    "response replay trace line {} duplicates output_replay_id key {}",
                    line_index + 1,
                    key
                );
            }
        }

        Ok(Self { rows })
    }

    fn get(&self, key: &str) -> Option<Vec<TokenIdType>> {
        self.rows.get(key).cloned()
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.rows.len()
    }
}

/// Wrapper to adapt KvEventPublisher to the KvCacheEventSink trait
struct KvEventSinkAdapter(KvEventPublisher);

impl KvCacheEventSink for KvEventSinkAdapter {
    fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
        self.0
            .publish(event)
            .map_err(|e| anyhow::anyhow!("Failed to send KV event: {}", e))
    }

    fn publish_with_storage_tier(
        &self,
        event: KvCacheEvent,
        storage_tier: StorageTier,
    ) -> anyhow::Result<()> {
        self.0
            .publish_with_storage_tier(event, storage_tier)
            .map_err(|e| anyhow::anyhow!("Failed to send KV event: {}", e))
    }

    fn publish_batch_with_storage_tiers(
        &self,
        events: Vec<(KvCacheEvent, StorageTier)>,
    ) -> anyhow::Result<()> {
        self.0
            .publish_batch_with_storage_tiers(events)
            .map_err(|e| anyhow::anyhow!("Failed to send KV event batch: {}", e))
    }
}

/// Cumulative usage snapshot carrying the scheduler's admission cache truth
/// in `prompt_tokens_details.cached_tokens`.
fn usage_with_cached_tokens(
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: usize,
) -> CompletionUsage {
    // Saturate rather than panic on pathological token counts.
    fn to_u32(value: usize) -> u32 {
        value.try_into().unwrap_or(u32::MAX)
    }
    CompletionUsage {
        prompt_tokens: to_u32(prompt_tokens),
        completion_tokens: to_u32(completion_tokens),
        total_tokens: to_u32(prompt_tokens.saturating_add(completion_tokens)),
        prompt_tokens_details: Some(PromptTokensDetails {
            audio_tokens: None,
            cached_tokens: Some(to_u32(cached_tokens)),
        }),
        completion_tokens_details: None,
    }
}

fn generate_random_token() -> TokenIdType {
    let mut rng = rand::rng();
    rng.random_range(1000..2000)
}

async fn wait_for_no_bootstrap_handoff_delay(
    is_prefill: bool,
    has_handoff_session: bool,
    delay_ms: Option<f64>,
) {
    if let Some(delay) = no_bootstrap_handoff_delay(is_prefill, has_handoff_session, delay_ms) {
        tokio::time::sleep(delay).await;
    }
}

fn no_bootstrap_handoff_delay(
    is_prefill: bool,
    has_handoff_session: bool,
    delay_ms: Option<f64>,
) -> Option<Duration> {
    if !is_prefill || has_handoff_session {
        return None;
    }
    let delay_ms = delay_ms?;
    Some(Duration::from_secs_f64(delay_ms.max(0.0) / 1000.0))
}

async fn send_response(
    stream_tx: &mpsc::UnboundedSender<LLMEngineOutput>,
    output: LLMEngineOutput,
    context: &Arc<dyn AsyncEngineContext>,
) -> bool {
    tokio::select! {
        biased;
        _ = stream_tx.closed() => false,
        _ = context.stopped() => {
            let _ = stream_tx.send(LLMEngineOutput::cancelled());
            false
        }
        result = async { stream_tx.send(output) } => result.is_ok(),
    }
}

struct MockerExecutionContext {
    engines: OnceCell<Vec<LiveEngine>>,
    handoff_session_permits: OnceCell<Vec<Arc<Semaphore>>>,
    startup_state: watch::Sender<StartupState>,
    engine_args: MockEngineArgs,
    response_replay_table: Option<ResponseReplayTable>,
    unset_dp_rank_counter: AtomicU32,
    /// Bootstrap server for prefill workers in disaggregated mode
    bootstrap_server: Arc<OnceCell<Arc<BootstrapServer>>>,
    source_handoff_manager: OnceCell<SourceHandoffManager>,
    handoff_shutdown: CancellationToken,
    metrics_shutdown: CancellationToken,
    handoff_tasks: TaskTracker,
    metrics_tasks: TaskTracker,
    native_metrics: Arc<NativeMockerMetrics>,
    /// Keep ZMQ relay publishers alive until their engines finish shutting down.
    _relay_publishers: OnceCell<Vec<Option<KvEventPublisher>>>,
    /// Forward pass metrics publisher (kept alive for the engine lifetime).
    _fpm_publisher: OnceCell<crate::fpm_publisher::FpmDirectPublisher>,
}

struct PreparedBootstrap {
    server: Arc<BootstrapServer>,
    max_sessions: usize,
    cancel: Option<CancellationToken>,
}

impl PreparedBootstrap {
    async fn shutdown(mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel.cancel();
        }
        self.server.wait_closed().await;
    }
}

impl Drop for PreparedBootstrap {
    fn drop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel.cancel();
        }
    }
}

#[derive(Clone, Debug)]
enum StartupState {
    Starting,
    Ready,
    Failed(Arc<str>),
}

impl MockerExecutionContext {
    fn new(engine_args: MockEngineArgs) -> Self {
        let (startup_state, _) = watch::channel(StartupState::Starting);
        let native_metrics = NativeMockerMetrics::new(engine_args.engine_type, engine_args.dp_size)
            .expect("mocker native metrics collectors should be valid");
        let response_replay_table = engine_args
            .response_replay_trace_path
            .as_deref()
            .map(|path| {
                ResponseReplayTable::from_path(path).unwrap_or_else(|error| {
                    panic!(
                        "failed to load response replay trace {}: {error:#}",
                        path.display()
                    )
                })
            });
        if let Some(table) = response_replay_table.as_ref() {
            tracing::info!(
                rows = table.rows.len(),
                "loaded response replay token table"
            );
        }
        Self {
            engines: OnceCell::new(),
            handoff_session_permits: OnceCell::new(),
            startup_state,
            engine_args,
            response_replay_table,
            unset_dp_rank_counter: AtomicU32::new(0),
            bootstrap_server: Arc::new(OnceCell::new()),
            source_handoff_manager: OnceCell::new(),
            handoff_shutdown: CancellationToken::new(),
            metrics_shutdown: CancellationToken::new(),
            handoff_tasks: TaskTracker::new(),
            metrics_tasks: TaskTracker::new(),
            native_metrics,
            _relay_publishers: OnceCell::new(),
            _fpm_publisher: OnceCell::new(),
        }
    }

    fn resolve_dp_rank(&self, request: &PreprocessedRequest) -> u32 {
        if let Some(dp_rank) = request.routing.as_ref().and_then(|routing| routing.dp_rank) {
            return dp_rank;
        }

        self.unset_dp_rank_counter.fetch_add(1, Ordering::Relaxed) % self.engine_args.dp_size
    }

    async fn prepare_bootstrap(&self) -> Result<Option<PreparedBootstrap>> {
        if !self.engine_args.is_prefill() {
            return Ok(None);
        }
        let Some(port) = self.engine_args.bootstrap_port else {
            return Ok(None);
        };
        let max_sessions = self
            .engine_args
            .effective_handoff_capacity()
            .checked_mul(self.engine_args.dp_size as usize)
            .expect("mocker handoff session limit overflow");
        let cancel = self.handoff_shutdown.child_token();
        let server = BootstrapServer::start(
            port,
            cancel.clone(),
            BootstrapServerConfig {
                max_pending_connections: max_sessions,
                ..BootstrapServerConfig::default()
            },
        )
        .await?;
        Ok(Some(PreparedBootstrap {
            server,
            max_sessions,
            cancel: Some(cancel),
        }))
    }

    fn commit_bootstrap(&self, mut prepared: PreparedBootstrap) {
        prepared.cancel.take();
        let server = Arc::clone(&prepared.server);
        let max_sessions = prepared.max_sessions;
        drop(prepared);
        let incoming_rx = server
            .take_incoming_receiver()
            .expect("new bootstrap server must own its incoming receiver");
        let manager = SourceHandoffManager::start(
            incoming_rx,
            max_sessions,
            Duration::from_millis(self.engine_args.handoff_session_timeout_ms),
            self.handoff_shutdown.clone(),
        );
        assert!(
            self.source_handoff_manager.set(manager).is_ok(),
            "source handoff manager initialized more than once"
        );
        assert!(
            self.bootstrap_server.set(server.clone()).is_ok(),
            "bootstrap server initialized more than once"
        );
        tracing::info!(
            port = server.port(),
            "Bootstrap server started for prefill worker"
        );
    }

    async fn start(self: Arc<Self>, endpoint: dynamo_runtime::component::Endpoint) -> Result<()> {
        let result = Arc::clone(&self).start_inner(endpoint).await;
        match &result {
            Ok(()) => {
                self.startup_state.send_replace(StartupState::Ready);
            }
            Err(error) => {
                self.set_startup_failure(format!("{error:#}"));
            }
        }
        result
    }

    async fn start_inner(
        self: Arc<Self>,
        endpoint: dynamo_runtime::component::Endpoint,
    ) -> Result<()> {
        let component = endpoint.component().clone();
        // Use primary_token() instead of child_token() so the mocker continues running
        // during graceful shutdown (Phase 1/2) and only stops in Phase 3.
        // child_token() is a child of endpoint_shutdown_token which is cancelled in Phase 1.
        // primary_token() is only cancelled in Phase 3, after waiting for inflight requests.
        let primary_token = component.drt().primary_token();
        self.native_metrics
            .register(component.get_metrics_registry())?;

        // Simulate engine startup time if configured
        if let Some(startup_time_secs) = self.engine_args.startup_time {
            tracing::info!("Simulating engine startup time: {:.2}s", startup_time_secs);
            tokio::time::sleep(Duration::from_secs_f64(startup_time_secs)).await;
            tracing::info!("Engine startup simulation completed");
        }

        let kv_endpoint = if self.engine_args.needs_kv_publisher() {
            tracing::info!(
                "Initializing KV event publisher with block_size {}, enable_local_indexer={}",
                self.engine_args.block_size,
                self.engine_args.enable_local_indexer
            );
            Some(&endpoint)
        } else {
            None
        };
        let mut prepared_bootstrap = self.prepare_bootstrap().await?;

        // Create FPM publisher upfront and get per-dp-rank sink handles.
        let worker_id = component.drt().connection_id().to_string();
        let (fpm_publisher, fpm_sinks) = match crate::fpm_publisher::FpmDirectPublisher::new(
            endpoint.clone(),
            worker_id,
            self.engine_args.dp_size,
        )
        .await
        {
            Ok((publisher, sinks)) => (Some(publisher), sinks),
            Err(e) => {
                tracing::error!("Failed to start FPM publisher: {e}");
                (
                    None,
                    (0..self.engine_args.dp_size)
                        .map(|_| dynamo_mocker::common::protocols::FpmPublisher::default())
                        .collect(),
                )
            }
        };

        let (engines, relay_publishers, handoff_session_permits) =
            match self.start_engines(kv_endpoint, fpm_sinks).await {
                Ok(started) => started,
                Err(error) => {
                    if let Some(prepared) = prepared_bootstrap.take() {
                        prepared.shutdown().await;
                    }
                    return Err(error);
                }
            };

        if let Err(error) = Self::start_metrics_publishing(
            &engines,
            endpoint,
            self.native_metrics.clone(),
            self.metrics_shutdown.clone(),
            self.metrics_tasks.clone(),
        )
        .await
        {
            Self::shutdown_engines(&engines).await;
            if let Some(prepared) = prepared_bootstrap.take() {
                prepared.shutdown().await;
            }
            return Err(error);
        }

        assert!(
            self.engines.set(engines).is_ok(),
            "live Mocker engines initialized more than once"
        );
        assert!(
            self._relay_publishers.set(relay_publishers).is_ok(),
            "mocker relay publishers initialized more than once"
        );
        assert!(
            self.handoff_session_permits
                .set(handoff_session_permits)
                .is_ok(),
            "mocker handoff permits initialized more than once"
        );
        if let Some(publisher) = fpm_publisher {
            assert!(
                self._fpm_publisher.set(publisher).is_ok(),
                "mocker FPM publisher initialized more than once"
            );
        }
        if let Some(prepared) = prepared_bootstrap.take() {
            self.commit_bootstrap(prepared);
        }

        let shutdown = Arc::clone(&self);
        tokio::spawn(async move {
            primary_token.cancelled().await;
            shutdown.handoff_shutdown.cancel();
            shutdown.handoff_tasks.close();
            if let Some(manager) = shutdown.source_handoff_manager.get().cloned() {
                manager.wait_closed().await;
            }
            if let Some(server) = shutdown.bootstrap_server.get().cloned() {
                server.wait_closed().await;
            }
            shutdown.handoff_tasks.wait().await;
            if let Some(engines) = shutdown.engines.get() {
                Self::shutdown_engines(engines).await;
            }
            shutdown.metrics_shutdown.cancel();
            shutdown.metrics_tasks.close();
            shutdown.metrics_tasks.wait().await;
        });

        Ok(())
    }

    async fn shutdown_engines(engines: &[LiveEngine]) {
        for result in futures::future::join_all(engines.iter().map(LiveEngine::shutdown)).await {
            if let Err(error) = result {
                tracing::error!(%error, "failed to shut down live Mocker engine");
            }
        }
    }

    fn set_startup_failure(&self, error: impl Into<Arc<str>>) {
        self.startup_state
            .send_replace(StartupState::Failed(error.into()));
    }

    async fn wait_for_startup(&self) -> Result<()> {
        let mut state = self.startup_state.subscribe();
        loop {
            let current = state.borrow().clone();
            match current {
                StartupState::Starting => {}
                StartupState::Ready => return Ok(()),
                StartupState::Failed(error) => {
                    bail!("mocker startup failed: {error}");
                }
            }
            state
                .changed()
                .await
                .map_err(|_| anyhow::anyhow!("mocker startup state channel closed"))?;
        }
    }

    async fn engine(&self, dp_rank: usize) -> Result<LiveEngine> {
        if let Some(engines) = self.engines.get() {
            return Ok(engines[dp_rank].clone());
        }

        self.wait_for_startup().await?;
        Ok(self
            .engines
            .get()
            .ok_or_else(|| anyhow::anyhow!("mocker reported ready without live engines"))?[dp_rank]
            .clone())
    }

    async fn handoff_session_permit(&self, dp_rank: usize) -> Result<Arc<Semaphore>> {
        if let Some(permits) = self.handoff_session_permits.get() {
            return Ok(permits[dp_rank].clone());
        }

        self.wait_for_startup().await?;
        Ok(self
            .handoff_session_permits
            .get()
            .ok_or_else(|| anyhow::anyhow!("mocker reported ready without handoff permits"))?
            [dp_rank]
            .clone())
    }

    async fn start_engines(
        &self,
        endpoint: Option<&dynamo_runtime::component::Endpoint>,
        fpm_sinks: Vec<dynamo_mocker::common::protocols::FpmPublisher>,
    ) -> Result<(
        Vec<LiveEngine>,
        Vec<Option<KvEventPublisher>>,
        Vec<Arc<Semaphore>>,
    )> {
        let args = &self.engine_args;
        let mut engine_configs = Vec::with_capacity(args.dp_size as usize);
        let mut relay_publishers = Vec::with_capacity(args.dp_size as usize);
        let mut handoff_session_permits = Vec::with_capacity(args.dp_size as usize);

        for (dp_rank, fpm_publisher) in (0..args.dp_size).zip(fpm_sinks) {
            let (kv_event_publishers, relay_publisher): (
                KvEventPublishers,
                Option<KvEventPublisher>,
            ) = match endpoint {
                Some(endpoint) if args.zmq_kv_events_port.is_some() => {
                    let zmq_port = args.zmq_kv_events_port.unwrap() + dp_rank as u16;
                    let replay_port = args.zmq_replay_port.map(|p| p + dp_rank as u16);
                    match ZmqKvEventSink::new(
                        zmq_port,
                        replay_port,
                        dp_rank,
                        args.block_size as u32,
                    )
                    .await
                    {
                        Ok(sink) => {
                            let source_config = Some(KvEventSourceConfig::Zmq {
                                endpoint: format!("tcp://127.0.0.1:{zmq_port}"),
                                topic: String::new(),
                                image_token_id: None,
                            });
                            match KvEventPublisher::new_with_local_indexer(
                                endpoint.clone(),
                                args.block_size as u32,
                                source_config,
                                args.enable_local_indexer,
                                dp_rank,
                                None,
                            ) {
                                Ok(publisher) => (
                                    KvEventPublishers::new(
                                        None,
                                        Some(Arc::new(sink) as Arc<dyn RawKvEventSink>),
                                    ),
                                    Some(publisher),
                                ),
                                Err(e) => {
                                    tracing::error!(
                                        "Failed to create KV event relay for dp_rank {dp_rank}: {e}"
                                    );
                                    (KvEventPublishers::default(), None)
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to create ZMQ KV event sink for dp_rank {dp_rank}: {e}"
                            );
                            (KvEventPublishers::default(), None)
                        }
                    }
                }
                Some(endpoint) => {
                    match KvEventPublisher::new_with_local_indexer(
                        endpoint.clone(),
                        args.block_size as u32,
                        None,
                        args.enable_local_indexer,
                        dp_rank,
                        None,
                    ) {
                        Ok(publisher) => (
                            KvEventPublishers::new(
                                Some(Arc::new(KvEventSinkAdapter(publisher))
                                    as Arc<dyn KvCacheEventSink>),
                                None,
                            ),
                            None,
                        ),
                        Err(e) => {
                            tracing::error!(
                                "Failed to create KV event publisher for dp_rank {dp_rank}: {e}"
                            );
                            (KvEventPublishers::default(), None)
                        }
                    }
                }
                None => (KvEventPublishers::default(), None),
            };

            engine_configs.push(LiveEngineConfig {
                kv_event_publishers,
                fpm_publisher,
            });
            relay_publishers.push(relay_publisher);
            handoff_session_permits
                .push(Arc::new(Semaphore::new(args.effective_handoff_capacity())));
        }
        // One logical worker owns one attention-DP generalized engine. The
        // returned rank-scoped LiveEngine handles share its single actor and
        // grouped pass barrier.
        let engines = LiveEngine::start_grouped_with_configs(args.clone(), engine_configs)?;
        Ok((engines, relay_publishers, handoff_session_permits))
    }

    /// Start background tasks to publish metrics on change
    async fn start_metrics_publishing(
        engines: &[LiveEngine],
        endpoint: Endpoint,
        native_metrics: Arc<NativeMockerMetrics>,
        cancel_token: CancellationToken,
        tasks: TaskTracker,
    ) -> Result<()> {
        let metrics_publisher = Arc::new(WorkerMetricsPublisher::new()?);

        if let Err(e) = metrics_publisher.create_endpoint(endpoint).await {
            tracing::error!("Metrics endpoint failed: {e}");
        }
        for engine in engines {
            let mut metrics_rx = engine.metrics_receiver();
            let publisher = metrics_publisher.clone();
            let native_metrics = native_metrics.clone();
            let cancel_token = cancel_token.clone();

            tasks.spawn(async move {
                loop {
                    tokio::select! {
                        // Watch for metrics changes
                        Ok(_) = metrics_rx.changed() => {
                            // Get the latest metrics
                            let metrics = metrics_rx.borrow().clone();
                            native_metrics.update_scheduler_snapshot(&metrics);

                            // Publish metrics using flat API
                            if let Err(e) = publisher.publish(
                                Some(metrics.dp_rank),
                                None,
                                Some(metrics.active_decode_blocks),
                            ) {
                                tracing::warn!("Failed to publish metrics for DP rank {}: {e}", metrics.dp_rank);
                            } else {
                                tracing::debug!(
                                    dp_rank = metrics.dp_rank,
                                    active_decode_blocks = metrics.active_decode_blocks,
                                    total_blocks = metrics.total_blocks,
                                    gpu_cache_usage_perc = metrics.gpu_cache_usage_perc,
                                    "published mocker load metrics"
                                );
                            }
                        }
                        _ = cancel_token.cancelled() => {
                            tracing::debug!("Metrics publishing cancelled");
                            break;
                        }
                    }
                }
            });
        }
        tracing::info!("Metrics background tasks started");
        Ok(())
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for MockerExecutionContext
{
    async fn generate(
        &self,
        input: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (request, ctx) = input.into_parts();
        let request_start = Instant::now();

        let dp_rank = self.resolve_dp_rank(&request);

        // Validate dp_rank
        if dp_rank >= self.engine_args.dp_size {
            return Err(Error::msg(format!(
                "dp_rank {} is out of bounds for dp_size {}",
                dp_rank, self.engine_args.dp_size
            )));
        }
        let engine = self
            .engine(dp_rank as usize)
            .await
            .map_err(|error| Error::msg(error.to_string()))?;

        let request_uuid = ctx.id().parse().unwrap_or(Uuid::new_v4());
        let is_prefill = self.engine_args.is_prefill();
        let requested_max_output_tokens = if is_prefill {
            1
        } else {
            request
                .stop_conditions
                .max_tokens
                .ok_or_else(|| Error::msg("max_output_tokens must be specified for mocker"))?
                as usize
        };
        let replay_key = (!is_prefill)
            .then(|| request.get_annotation_value(OUTPUT_REPLAY_ID_ANNOTATION_KEY))
            .flatten();
        let planned_output_token_ids = replay_key.as_deref().and_then(|key| {
            let Some(table) = self.response_replay_table.as_ref() else {
                tracing::warn!(
                    replay_key = key,
                    "request asked for output token replay but mocker has no response replay trace"
                );
                return None;
            };
            match table.get(key) {
                Some(tokens) => Some(tokens),
                None => {
                    tracing::warn!(
                        replay_key = key,
                        "request asked for output token replay but key was not found"
                    );
                    None
                }
            }
        });
        let has_planned_output_tokens = planned_output_token_ids.is_some();
        let max_output_tokens = planned_output_token_ids
            .as_ref()
            .map_or(requested_max_output_tokens, Vec::len);
        let effective_max_output_tokens =
            self.engine_args
                .max_model_len
                .map_or(max_output_tokens, |max_model_len| {
                    max_output_tokens.min(max_model_len.saturating_sub(request.token_ids.len()))
                });
        let native_timing = self
            .native_metrics
            .request_timing(&request.model, dp_rank, is_prefill, request_start)
            .await;

        let prompt_tokens_count = request.token_ids.len();
        // Convert PreprocessedRequest to DirectRequest for scheduler
        let direct_request = DirectRequest {
            tokens: request.token_ids.clone(),
            max_output_tokens,
            output_token_ids: planned_output_token_ids.clone(),
            uuid: Some(request_uuid),
            dp_rank,
            arrival_timestamp_ms: request.request_timestamp_ms,
            ..Default::default()
        };

        let (stream_tx, stream_rx) = mpsc::unbounded_channel::<LLMEngineOutput>();

        let handoff_id = request
            .bootstrap_info
            .as_ref()
            .and_then(|info| info.handoff_id);
        let has_handoff_session = handoff_id.is_some();
        if request.bootstrap_info.is_some()
            && (self.engine_args.is_prefill() || self.engine_args.is_decode())
            && handoff_id.is_none()
        {
            return Err(Error::msg("disaggregated mocker requires a handoff ID"));
        }

        let handoff_cancel = CancellationToken::new();
        let mut source_completion_rx = None;
        let mut destination_error_rx = None;
        let mut destination_cleanup: Option<HandoffControl> = None;
        let mut live_request;

        if let Some(handoff_id) = handoff_id {
            let bootstrap_info = request
                .bootstrap_info
                .as_ref()
                .expect("mocker handoff metadata requires bootstrap info");
            let handoff_id = HandoffId::from(handoff_id);
            let identity = BootstrapIdentity {
                handoff_id,
                bootstrap_room: bootstrap_info.bootstrap_room,
                request_id: request_uuid,
            };
            let order = match order_for_engine(self.engine_args.engine_type) {
                Ok(order) => order,
                Err(error) => {
                    return Err(Error::msg(error.to_string()));
                }
            };
            let session_permit = match self
                .handoff_session_permit(dp_rank as usize)
                .await
                .map_err(|error| Error::msg(error.to_string()))?
                .try_acquire_owned()
            {
                Ok(permit) => permit,
                Err(_) => {
                    return Err(Error::msg(format!(
                        "mocker handoff session limit reached for DP rank {dp_rank}"
                    )));
                }
            };
            let (registration, prepared_request) = engine
                .prepare_request(direct_request)
                .map_err(|error| Error::msg(error.to_string()))?;
            live_request = prepared_request;
            let (control, lifecycle) = match engine.register_handoff(handoff_id) {
                Ok(boundary) => boundary,
                Err(error) => {
                    return Err(Error::msg(error.to_string()));
                }
            };
            let (control, lifecycle) = live_handoff_boundary(control, lifecycle, registration);

            if self.engine_args.is_prefill() {
                let Some(manager) = self.source_handoff_manager.get() else {
                    return Err(Error::msg("source handoff manager is not initialized"));
                };
                let (completion_tx, completion_rx) = oneshot::channel();
                if let Err(error) = manager.try_register(SourceRegistration {
                    identity,
                    order,
                    engine_type: self.engine_args.engine_type,
                    control,
                    lifecycle,
                    completion_tx,
                    cancel: handoff_cancel.clone(),
                    observer: None,
                    _permit: session_permit,
                }) {
                    return Err(Error::msg(error.to_string()));
                }
                source_completion_rx = Some(completion_rx);
            } else if self.engine_args.is_decode() {
                let registration = ParticipantRegistration {
                    role: BootstrapParticipantRole::Destination,
                    dp_rank,
                    order,
                    engine_type: self.engine_args.engine_type,
                };
                let connection = match connect_to_prefill(
                    &bootstrap_info.bootstrap_host,
                    bootstrap_info.bootstrap_port,
                    identity,
                    registration,
                )
                .await
                {
                    Ok(connection) => connection,
                    Err(error) => {
                        return Err(Error::msg(format!("bootstrap connection failed: {error}")));
                    }
                };
                let (error_tx, error_rx) = mpsc::unbounded_channel();
                let session_control = control.clone();
                let session_cancel = handoff_cancel.clone();
                let session_timeout =
                    Duration::from_millis(self.engine_args.handoff_session_timeout_ms);
                let global_shutdown = self.handoff_shutdown.clone();
                self.handoff_tasks.spawn(async move {
                    let _session_permit = session_permit;
                    if let Err(error) = run_destination_session(
                        connection,
                        session_control,
                        lifecycle,
                        session_cancel,
                        session_timeout,
                        global_shutdown,
                    )
                    .await
                    {
                        let _ = error_tx.send(error.to_string());
                    }
                });
                destination_error_rx = Some(error_rx);
                destination_cleanup = Some(control);
            } else {
                return Err(Error::msg(
                    "aggregated mocker request cannot carry handoff metadata",
                ));
            }
        } else {
            live_request = engine
                .submit(direct_request)
                .await
                .map_err(|error| Error::msg(error.to_string()))?;
        }

        let async_context = ctx.context();
        let reasoning = self.engine_args.reasoning.clone();
        let handoff_session_timeout =
            Duration::from_millis(self.engine_args.handoff_session_timeout_ms);
        let mut native_timing = native_timing;
        let response_task_tracker = (source_completion_rx.is_some()
            || destination_cleanup.is_some())
        .then(|| self.handoff_tasks.clone());

        // Spawn a task to handle the complex async logic
        let response_task = async move {
            let mut token_count = 0;
            let mut cached_prefix_tokens: Option<usize> = None;
            let mut source_completion_rx = source_completion_rx;
            let mut source_handoff_complete = source_completion_rx.is_none();
            let mut destination_error_rx = destination_error_rx;
            let mut request_completed_normally = false;
            let think_len = reasoning
                .as_ref()
                .map(|cfg| cfg.num_thinking_tokens(max_output_tokens))
                .unwrap_or(0);

            loop {
                tokio::select! {
                    source_completion = async {
                        source_completion_rx
                            .as_mut()
                            .expect("guarded source completion receiver")
                            .await
                    }, if source_completion_rx.is_some() => {
                        source_completion_rx = None;
                        match source_completion {
                            Ok(Ok(())) => source_handoff_complete = true,
                            Ok(Err(error)) => {
                                let _ = send_response(
                                    &stream_tx,
                                    LLMEngineOutput::error(error),
                                    &async_context,
                                )
                                .await;
                                break;
                            }
                            Err(_) => {
                                let _ = send_response(
                                    &stream_tx,
                                    LLMEngineOutput::error(
                                        "source handoff session ended without completion".to_string(),
                                    ),
                                    &async_context,
                                )
                                .await;
                                break;
                            }
                        }
                    }
                    destination_error = async {
                        match destination_error_rx.as_mut() {
                            Some(receiver) => receiver.recv().await,
                            None => std::future::pending().await,
                        }
                    }, if destination_error_rx.is_some() => {
                        match destination_error {
                            Some(error) => {
                                let _ = send_response(
                                    &stream_tx,
                                    LLMEngineOutput::error(error),
                                    &async_context,
                                )
                                .await;
                                break;
                            }
                            None => destination_error_rx = None,
                        }
                    }
                    maybe_signal = live_request.recv() => {
                        let Some(signal) = maybe_signal else {
                            let _ = send_response(
                                &stream_tx,
                                LLMEngineOutput::error("All output transmitters closed".to_string()),
                                &async_context,
                            ).await;
                            break;
                        };

                        // A terminally rejected request never ran because it violated
                        // a worker admission limit. Emit no token and do not complete
                        // the bootstrap room; surface the rejection before any
                        // token/prefill bookkeeping.
                        if signal.rejected {
                            handoff_cancel.cancel();
                            let _ = send_response(
                                &stream_tx,
                                LLMEngineOutput::error(
                                    "request rejected: request exceeds worker admission limits".to_string(),
                                ),
                                &async_context,
                            )
                            .await;
                            break;
                        }

                        if let Some(cached) = signal.cached_tokens {
                            cached_prefix_tokens = Some(cached);
                        }

                        // Generate a token (with thinking boundaries if configured)
                        let token_id = if has_planned_output_tokens {
                            signal.token_id.unwrap_or_else(generate_random_token)
                        } else if token_count == 0 && think_len > 0 {
                            reasoning.as_ref().unwrap().start_thinking_token_id
                        } else if think_len > 0 && token_count == think_len - 1 {
                            reasoning.as_ref().unwrap().end_thinking_token_id
                        } else {
                            generate_random_token()
                        };
                        token_count += 1;

                        // The first chunk carries the admission cache truth; the
                        // final chunk repeats cumulative totals (OpenAI convention).
                        let output = LLMEngineOutput {
                            token_ids: vec![token_id],
                            disaggregated_params: is_prefill.then(|| serde_json::json!("dummy")),
                            completion_usage: signal.cached_tokens.map(|cached| {
                                usage_with_cached_tokens(prompt_tokens_count, token_count, cached)
                            }),
                            ..Default::default()
                        };

                        if signal.completed && token_count < effective_max_output_tokens {
                            let _ = send_response(
                                &stream_tx,
                                LLMEngineOutput::error(
                                    "Completion signal received before max tokens reached".to_string(),
                                ),
                                &async_context,
                            )
                            .await;
                            break;
                        }

                        if signal.completed {
                            if !send_response(&stream_tx, output, &async_context).await {
                                break;
                            }
                            native_timing.record_tokens(1);

                            let delay_completed = tokio::select! {
                                _ = wait_for_no_bootstrap_handoff_delay(
                                    is_prefill,
                                    has_handoff_session,
                                    signal.handoff_delay_ms,
                                ) => true,
                                _ = stream_tx.closed() => false,
                                _ = async_context.stopped() => {
                                    handoff_cancel.cancel();
                                    let _ = stream_tx.send(LLMEngineOutput::cancelled());
                                    false
                                }
                            };
                            if !delay_completed {
                                break;
                            }

                            if !source_handoff_complete
                                && let Some(completion_rx) = source_completion_rx.take()
                            {
                                let completion = tokio::select! {
                                    completion = completion_rx => completion,
                                    _ = stream_tx.closed() => {
                                        handoff_cancel.cancel();
                                        break;
                                    }
                                    _ = async_context.stopped() => {
                                        handoff_cancel.cancel();
                                        let _ = stream_tx.send(LLMEngineOutput::cancelled());
                                        break;
                                    }
                                };
                                match completion {
                                    Ok(Ok(())) => {}
                                    Ok(Err(error)) => {
                                        let _ = send_response(
                                            &stream_tx,
                                            LLMEngineOutput::error(error),
                                            &async_context,
                                        )
                                        .await;
                                        break;
                                    }
                                    Err(_) => {
                                        let _ = send_response(
                                            &stream_tx,
                                            LLMEngineOutput::error(
                                                "source handoff session ended without completion"
                                                    .to_string(),
                                            ),
                                            &async_context,
                                        )
                                        .await;
                                        break;
                                    }
                                }
                            }

                            let mut final_output = LLMEngineOutput::length();
                            if let Some(cached) = cached_prefix_tokens {
                                final_output.completion_usage = Some(usage_with_cached_tokens(
                                    prompt_tokens_count,
                                    token_count,
                                    cached,
                                ));
                            }
                            if !send_response(&stream_tx, final_output, &async_context).await {
                                break;
                            }
                            native_timing.record_normal_completion();
                            request_completed_normally = true;
                            break;
                        }

                        if !send_response(&stream_tx, output, &async_context).await {
                            break;
                        }
                        native_timing.record_tokens(1);
                    }

                    _ = async_context.stopped() => {
                        handoff_cancel.cancel();
                        let _ = stream_tx.send(LLMEngineOutput::cancelled());
                        break;
                    }

                    _ = stream_tx.closed() => {
                        handoff_cancel.cancel();
                        break;
                    }
                }
            }

            if !request_completed_normally {
                handoff_cancel.cancel();
                if let Some(control) = destination_cleanup.as_ref() {
                    cancel_destination(control, handoff_session_timeout).await;
                }
                let _ = live_request.cancel().await;
            }
        };
        if let Some(tasks) = response_task_tracker {
            tasks.spawn(response_task);
        } else {
            tokio::spawn(response_task);
        }

        let stream = UnboundedReceiverStream::new(stream_rx).map(Annotated::from_data);
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

/// Create a mocker engine as ExecutionContext
pub async fn make_mocker_engine(
    distributed_runtime: DistributedRuntime,
    endpoint_id: dynamo_runtime::protocols::EndpointId,
    args: MockEngineArgs,
) -> Result<ExecutionContext, Error> {
    tracing::info!("Creating mocker engine with config: {args:?}");
    let engine = Arc::new(MockerExecutionContext::new(args));
    let startup_engine = Arc::clone(&engine);
    let cancel_token = distributed_runtime.primary_token();
    tokio::spawn(async move {
        let component = loop {
            if cancel_token.is_cancelled() {
                tracing::debug!("Mocker engine startup cancelled");
                startup_engine.set_startup_failure("mocker startup was cancelled");
                return;
            }

            let ready = distributed_runtime
                .namespace(&endpoint_id.namespace)
                .and_then(|namespace| namespace.component(&endpoint_id.component))
                .ok();
            if let Some(component) = ready
                && let Ok(instances) = component.list_instances().await
                && !instances.is_empty()
            {
                break component;
            }

            tracing::debug!("Component service not available yet, retrying...");
            tokio::time::sleep(Duration::from_millis(100)).await;
        };

        tracing::debug!("Component service is now available, starting mocker engine");
        let endpoint = component.endpoint(endpoint_id.name);
        if let Err(error) = startup_engine.start(endpoint).await {
            tracing::error!("Failed to start mocker engine: {error}");
        }
    });

    Ok(engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::llm_backend::PreprocessedRequest;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_mocker::common::protocols::{MockEngineArgs, WorkerType};
    use dynamo_runtime::pipeline::{AsyncEngine, SingleIn};
    use futures::StreamExt;
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::time::Duration;

    fn prefill_request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("mock".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions {
                max_tokens: Some(1),
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .eos_token_ids(vec![])
            .annotations(vec![])
            .build()
            .unwrap()
    }

    fn decode_request(prompt_tokens: usize, max_tokens: u32) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("mock".to_string())
            .token_ids(vec![1; prompt_tokens])
            .stop_conditions(StopConditions {
                max_tokens: Some(max_tokens),
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .eos_token_ids(vec![])
            .annotations(vec![])
            .build()
            .unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn no_bootstrap_prefill_delays_terminal_finish_once() {
        let args = MockEngineArgs::builder()
            .worker_type(WorkerType::Prefill)
            .block_size(4)
            .num_gpu_blocks(64)
            .max_num_batched_tokens(Some(64))
            .speedup_ratio(1000.0)
            .kv_transfer_bandwidth(Some(1.0))
            .kv_bytes_per_token(Some(25_000_000))
            .build()
            .unwrap();
        let live = LiveEngine::start(args.clone(), 0).unwrap();
        let engine = MockerExecutionContext::new(args);
        assert!(engine.engines.set(vec![live]).is_ok());
        let mut request = prefill_request();
        request.token_ids = vec![1, 2, 3, 4];

        let mut stream = engine.generate(SingleIn::new(request)).await.unwrap();
        let token = stream.next().await.unwrap().data.unwrap();
        assert_eq!(token.token_ids.len(), 1);
        assert!(token.finish_reason.is_none());
        assert!(
            tokio::time::timeout(Duration::from_millis(99), stream.next())
                .await
                .is_err()
        );

        tokio::time::advance(Duration::from_millis(1)).await;
        let finish = stream.next().await.unwrap().data.unwrap();
        assert!(finish.token_ids.is_empty());
        assert!(finish.finish_reason.is_some());
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn context_capped_completion_maps_to_length() {
        let args = MockEngineArgs::builder()
            .max_model_len(Some(4))
            .block_size(4)
            .num_gpu_blocks(64)
            .max_num_batched_tokens(Some(64))
            .speedup_ratio(1000.0)
            .build()
            .unwrap();
        let live = LiveEngine::start(args.clone(), 0).unwrap();
        let engine = MockerExecutionContext::new(args);
        assert!(engine.engines.set(vec![live]).is_ok());

        let mut stream = engine
            .generate(SingleIn::new(decode_request(3, 4)))
            .await
            .unwrap();
        let token = stream.next().await.unwrap().data.unwrap();
        assert_eq!(token.token_ids.len(), 1);
        assert!(token.finish_reason.is_none());
        assert_eq!(
            token.completion_usage,
            Some(CompletionUsage {
                prompt_tokens: 3,
                completion_tokens: 1,
                total_tokens: 4,
                prompt_tokens_details: Some(PromptTokensDetails {
                    audio_tokens: None,
                    cached_tokens: Some(0),
                }),
                completion_tokens_details: None,
            })
        );
        let mut expected_finish = LLMEngineOutput::length();
        expected_finish.completion_usage = Some(usage_with_cached_tokens(3, 1, 0));
        assert_eq!(stream.next().await.unwrap().data.unwrap(), expected_finish);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn dropping_response_cancels_live_request_and_allows_id_reuse() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(64)
            .max_num_seqs(Some(1))
            .max_num_batched_tokens(Some(64))
            .speedup_ratio(0.1)
            .build()
            .unwrap();
        let live = LiveEngine::start(args.clone(), 0).unwrap();
        let engine = MockerExecutionContext::new(args);
        assert!(engine.engines.set(vec![live.clone()]).is_ok());
        let blocker_id = Uuid::from_u128(98);
        let mut blocker = live
            .submit(DirectRequest {
                tokens: vec![1],
                max_output_tokens: 10_000,
                output_token_ids: Some(vec![7; 10_000]),
                uuid: Some(blocker_id),
                ..Default::default()
            })
            .await
            .unwrap();
        let blocker_drain = tokio::spawn(async move { while blocker.recv().await.is_some() {} });
        let request_id = Uuid::from_u128(99);
        let input = SingleIn::with_id_and_metadata(
            decode_request(1, 100),
            request_id.to_string(),
            BTreeMap::new(),
        );
        let stream = engine.generate(input).await.unwrap();
        drop(stream);

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if live.active_request_count() == 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("dropped response must cancel its live request");
        assert!(live.cancel(blocker_id).await.unwrap());
        blocker_drain.await.unwrap();

        let input = SingleIn::with_id_and_metadata(
            decode_request(1, 1),
            request_id.to_string(),
            BTreeMap::new(),
        );
        let mut replacement = engine.generate(input).await.unwrap();
        while replacement.next().await.is_some() {}
        assert_eq!(live.active_request_count(), 0);
    }

    #[tokio::test]
    async fn unread_response_completes_without_overflow_cancellation() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(64)
            .max_num_batched_tokens(Some(64))
            .speedup_ratio(1000.0)
            .build()
            .unwrap();
        let live = LiveEngine::start(args.clone(), 0).unwrap();
        let engine = MockerExecutionContext::new(args);
        assert!(engine.engines.set(vec![live.clone()]).is_ok());

        let mut stream = engine
            .generate(SingleIn::new(decode_request(1, 32)))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while live.active_request_count() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("generation should finish while the response remains unread");

        let mut output_tokens = 0;
        let mut finish = None;
        while let Some(output) = stream.next().await {
            let output = output.data.unwrap();
            output_tokens += output.token_ids.len();
            if output.finish_reason.is_some() {
                finish = output.finish_reason;
            }
        }
        assert_eq!(output_tokens, 32);
        assert_eq!(finish, LLMEngineOutput::length().finish_reason);
    }

    #[test]
    fn unbounded_sequence_limit_uses_finite_multi_handoff_capacity() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(3)
            .max_num_seqs(None)
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.effective_handoff_capacity(), 3);
        let permits = tokio::sync::Semaphore::new(args.effective_handoff_capacity());
        let held = (0..3)
            .map(|_| permits.try_acquire().unwrap())
            .collect::<Vec<_>>();
        assert!(permits.try_acquire().is_err());
        drop(held);
    }

    #[tokio::test]
    async fn startup_failure_wakes_engine_waiters() {
        let engine = Arc::new(MockerExecutionContext::new(
            MockEngineArgs::builder().build().unwrap(),
        ));
        let waiting_engine = Arc::clone(&engine);
        let waiter = tokio::spawn(async move { waiting_engine.engine(0).await });
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());

        engine.set_startup_failure("rank initialization failed");
        let error = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("startup waiter should wake")
            .unwrap()
            .err()
            .expect("startup waiter should return the initialization failure");
        assert!(error.to_string().contains("rank initialization failed"));
    }

    #[tokio::test]
    async fn prepared_bootstrap_shutdown_releases_the_listener() {
        let occupied = std::net::TcpListener::bind(("0.0.0.0", 0)).unwrap();
        let port = occupied.local_addr().unwrap().port();
        let args = MockEngineArgs::builder()
            .worker_type(WorkerType::Prefill)
            .bootstrap_port(Some(port))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        let engine = MockerExecutionContext::new(args);

        assert!(engine.prepare_bootstrap().await.is_err());

        drop(occupied);
        let prepared = engine
            .prepare_bootstrap()
            .await
            .unwrap()
            .expect("released bootstrap port must be reusable");
        prepared.shutdown().await;
        let retry = engine
            .prepare_bootstrap()
            .await
            .unwrap()
            .expect("failed startup cleanup must release the bootstrap port");
        retry.shutdown().await;
    }

    fn write_replay_trace(lines: &[serde_json::Value]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(file, "{}", serde_json::to_string(line).unwrap()).unwrap();
        }
        file
    }

    #[test]
    fn response_replay_table_derives_keys_and_validates_lengths() {
        let file = write_replay_trace(&[
            serde_json::json!({
                "request_id": "explicit",
                "session_id": "s",
                "output_length": 2,
                "output_token_ids": [7, 8],
            }),
            serde_json::json!({
                "session_id": "s",
                "output_length": 1,
                "output_token_ids": [9],
            }),
            serde_json::json!({
                "output_length": 1,
                "output_token_ids": [10],
            }),
        ]);

        let table = ResponseReplayTable::from_path(file.path()).unwrap();
        assert_eq!(table.len(), 3);
        assert_eq!(table.get("explicit").as_deref(), Some(&[7, 8][..]));
        assert_eq!(table.get("s:1").as_deref(), Some(&[9][..]));
        assert_eq!(table.get("line:2").as_deref(), Some(&[10][..]));

        let invalid = write_replay_trace(&[serde_json::json!({
            "output_length": 2,
            "output_token_ids": [1],
        })]);
        let err = ResponseReplayTable::from_path(invalid.path()).unwrap_err();
        assert!(
            err.to_string()
                .contains("output_length 2 does not match output_token_ids length 1"),
            "{err:#}"
        );
    }
}
