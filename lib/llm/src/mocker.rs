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
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use crate::backend::ExecutionContext;
use crate::kv_router::publisher::{
    KvEventPublisher, KvEventPublisherShutdown, KvEventPublisherShutdownOutcome,
    KvEventSourceConfig, ValkeyWorkerEventLease, ValkeyWorkerRegistration, WorkerMetricsPublisher,
    shutdown_publishers_and_wait,
};
use crate::protocols::TokenIdType;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use anyhow::{Context, Result, bail};
use dashmap::DashMap;
use dynamo_kv_router::protocols::{KvCacheEvent, StorageTier};
use dynamo_mocker::common::handoff::HandoffId;
use dynamo_mocker::common::protocols::{
    DirectRequest, KvCacheEventSink, KvEventPublishers, MockEngineArgs, OutputSignal,
    RawKvEventSink,
};
use dynamo_mocker::engine::create_engine;
use dynamo_mocker::loadgen::{OUTPUT_REPLAY_ID_ANNOTATION_KEY, effective_replay_key};
use dynamo_mocker::scheduler::{SchedulerCommandEnvelope, SchedulerHandle};
use dynamo_mocker::services::bootstrap::{
    BootstrapIdentity, BootstrapParticipantRole, BootstrapServer, BootstrapServerConfig,
    ParticipantRegistration, connect_to_prefill,
};
use dynamo_mocker::services::zmq_events::ZmqKvEventSink;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::{
    component::Component,
    engine::AsyncEngineContextProvider,
    pipeline::{AsyncEngine, Error, ManyOut, ResponseStream, SingleIn, async_trait},
    traits::DistributedRuntimeProvider,
};
use futures::StreamExt;
use rand::Rng;
use serde::Deserialize;
use tokio::sync::{Mutex, Notify, OnceCell, Semaphore, mpsc, oneshot, watch};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use uuid::Uuid;

use self::handoff::{
    HandoffEventRegistry, SourceHandoffManager, SourceRegistration, cancel_destination,
    order_for_engine, run_destination_session,
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
struct KvEventSinkAdapter {
    publisher: KvEventPublisher,
    terminal: AtomicBool,
}

impl KvEventSinkAdapter {
    fn new(publisher: KvEventPublisher) -> Self {
        Self {
            publisher,
            terminal: AtomicBool::new(false),
        }
    }

    fn record_result(&self, result: anyhow::Result<()>) -> anyhow::Result<()> {
        if result.is_err() {
            // KvEventPublisher only rejects input after its channel closes or
            // its direct-Valkey integrity gate permanently fences the worker.
            self.terminal.store(true, Ordering::Relaxed);
        }
        result
    }
}

impl KvCacheEventSink for KvEventSinkAdapter {
    fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
        self.record_result(
            self.publisher
                .publish(event)
                .map_err(|e| anyhow::anyhow!("Failed to send KV event: {}", e)),
        )
    }

    fn is_closed(&self) -> bool {
        self.terminal.load(Ordering::Relaxed)
    }

    fn publish_with_storage_tier(
        &self,
        event: KvCacheEvent,
        storage_tier: StorageTier,
    ) -> anyhow::Result<()> {
        self.record_result(
            self.publisher
                .publish_with_storage_tier(event, storage_tier)
                .map_err(|e| anyhow::anyhow!("Failed to send KV event: {}", e)),
        )
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

/// AsyncEngine wrapper around the Scheduler that generates random character tokens
pub struct MockEngine {
    active_requests: Arc<DashMap<Uuid, mpsc::UnboundedSender<OutputSignal>>>,
    request_senders: OnceCell<Vec<mpsc::UnboundedSender<DirectRequest>>>,
    command_senders: OnceCell<Vec<mpsc::Sender<SchedulerCommandEnvelope>>>,
    handoff_session_permits: OnceCell<Vec<Arc<Semaphore>>>,
    senders_ready: Notify,
    engine_args: MockEngineArgs,
    response_replay_table: Option<ResponseReplayTable>,
    unset_dp_rank_counter: AtomicU32,
    /// Bootstrap server for prefill workers in disaggregated mode
    bootstrap_server: Arc<OnceCell<Arc<BootstrapServer>>>,
    source_handoff_manager: OnceCell<SourceHandoffManager>,
    handoff_events: HandoffEventRegistry,
    handoff_shutdown: CancellationToken,
    scheduler_shutdown: CancellationToken,
    handoff_tasks: TaskTracker,
    scheduler_tasks: TaskTracker,
    native_metrics: Arc<NativeMockerMetrics>,
    /// Keep schedulers alive so their CancelGuards don't fire prematurely.
    _schedulers: OnceCell<Vec<Box<dyn SchedulerHandle>>>,
    /// Forward pass metrics publisher (kept alive for the engine lifetime).
    _fpm_publisher: OnceCell<crate::fpm_publisher::FpmDirectPublisher>,
    /// One process-level owner lease shared by every DP-rank publisher.
    _valkey_registration: Arc<Mutex<Option<ValkeyWorkerRegistration>>>,
}

struct PreparedBootstrap {
    server: Arc<BootstrapServer>,
    max_sessions: usize,
}

struct PreparedSchedulers {
    schedulers: Vec<Box<dyn SchedulerHandle>>,
    kv_publisher_shutdowns: Vec<KvEventPublisherShutdown>,
    request_senders: Vec<mpsc::UnboundedSender<DirectRequest>>,
    command_senders: Vec<mpsc::Sender<SchedulerCommandEnvelope>>,
    handoff_session_permits: Vec<Arc<Semaphore>>,
}

struct MockerSchedulerStartupFailure {
    error: anyhow::Error,
    publisher_outcome: KvEventPublisherShutdownOutcome,
}

fn startup_rollback_can_unregister(outcome: KvEventPublisherShutdownOutcome) -> bool {
    outcome == KvEventPublisherShutdownOutcome::Drained
}

impl MockEngine {
    /// Create a new MockEngine with the given parameters
    pub fn new(engine_args: MockEngineArgs) -> Self {
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
            active_requests: Arc::new(DashMap::new()),
            request_senders: OnceCell::new(),
            command_senders: OnceCell::new(),
            handoff_session_permits: OnceCell::new(),
            senders_ready: Notify::new(),
            engine_args,
            response_replay_table,
            unset_dp_rank_counter: AtomicU32::new(0),
            bootstrap_server: Arc::new(OnceCell::new()),
            source_handoff_manager: OnceCell::new(),
            handoff_events: HandoffEventRegistry::default(),
            handoff_shutdown: CancellationToken::new(),
            scheduler_shutdown: CancellationToken::new(),
            handoff_tasks: TaskTracker::new(),
            scheduler_tasks: TaskTracker::new(),
            native_metrics,
            _schedulers: OnceCell::new(),
            _fpm_publisher: OnceCell::new(),
            _valkey_registration: Arc::new(Mutex::new(None)),
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
        let server = BootstrapServer::start(
            port,
            self.handoff_shutdown.clone(),
            BootstrapServerConfig {
                max_pending_connections: max_sessions,
                ..BootstrapServerConfig::default()
            },
        )
        .await?;
        Ok(Some(PreparedBootstrap {
            server,
            max_sessions,
        }))
    }

    fn commit_bootstrap(&self, prepared: PreparedBootstrap) {
        let PreparedBootstrap {
            server,
            max_sessions,
        } = prepared;
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

    async fn rollback_registration(
        &self,
        registration: &mut Option<ValkeyWorkerRegistration>,
        publisher_outcome: KvEventPublisherShutdownOutcome,
    ) {
        let Some(mut registration) = registration.take() else {
            return;
        };
        if startup_rollback_can_unregister(publisher_outcome) {
            if let Err(error) = registration.shutdown().await {
                tracing::warn!(
                    error = %error,
                    "Failed to roll back mocker Valkey registration; lease expiry remains the backstop"
                );
            }
        } else {
            tracing::warn!(
                ?publisher_outcome,
                "Skipping explicit Valkey unregister after ambiguous startup drain; lease expiry remains the backstop"
            );
        }
    }

    async fn rollback_uncommitted_bootstrap(&self, prepared: Option<&PreparedBootstrap>) {
        self.handoff_shutdown.cancel();
        if let Some(prepared) = prepared {
            prepared.server.wait_closed().await;
        }
    }

    async fn rollback_prepared_schedulers(
        &self,
        publishers: &[KvEventPublisherShutdown],
    ) -> KvEventPublisherShutdownOutcome {
        self.scheduler_shutdown.cancel();
        self.scheduler_tasks.close();
        self.scheduler_tasks.wait().await;
        shutdown_publishers_and_wait(publishers).await
    }

    pub async fn start(&self, component: Component) -> Result<()> {
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

        let kv_component = if self.engine_args.needs_kv_publisher() {
            tracing::info!(
                "Initializing KV event publisher with block_size {}, enable_local_indexer={}",
                self.engine_args.block_size,
                self.engine_args.enable_local_indexer
            );
            Some(&component)
        } else {
            None
        };
        let registration_ranks = (0..self.engine_args.dp_size).collect::<Vec<_>>();
        let mut valkey_registration = ValkeyWorkerRegistration::register_from_env(
            &component,
            None,
            self.engine_args.block_size as u32,
            &registration_ranks,
        )
        .await?;
        let valkey_event_lease = valkey_registration
            .as_ref()
            .map(ValkeyWorkerRegistration::event_lease);
        let prepared_bootstrap = match self.prepare_bootstrap().await {
            Ok(prepared) => prepared,
            Err(error) => {
                self.rollback_registration(
                    &mut valkey_registration,
                    KvEventPublisherShutdownOutcome::Drained,
                )
                .await;
                return Err(error.context("failed to prepare mocker bootstrap service"));
            }
        };

        // Create FPM publisher upfront and get per-dp-rank sink handles.
        let worker_id = component.drt().connection_id().to_string();
        let (fpm_publisher, fpm_sinks) = match crate::fpm_publisher::FpmDirectPublisher::new(
            component.clone(),
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

        let prepared_schedulers = match self
            .start_schedulers(
                kv_component,
                self.scheduler_shutdown.clone(),
                fpm_sinks,
                valkey_event_lease,
            )
            .await
        {
            Ok(prepared) => prepared,
            Err(failure) => {
                self.rollback_uncommitted_bootstrap(prepared_bootstrap.as_ref())
                    .await;
                self.rollback_registration(&mut valkey_registration, failure.publisher_outcome)
                    .await;
                return Err(failure.error.context("failed to start mocker schedulers"));
            }
        };

        if let Err(error) = Self::start_metrics_publishing(
            &prepared_schedulers.schedulers,
            component.clone(),
            self.native_metrics.clone(),
            self.scheduler_shutdown.clone(),
            self.scheduler_tasks.clone(),
        )
        .await
        {
            self.rollback_uncommitted_bootstrap(prepared_bootstrap.as_ref())
                .await;
            let publisher_outcome = self
                .rollback_prepared_schedulers(&prepared_schedulers.kv_publisher_shutdowns)
                .await;
            self.rollback_registration(&mut valkey_registration, publisher_outcome)
                .await;
            return Err(error.context("failed to start mocker metrics publishing"));
        }

        if let Some(prepared) = prepared_bootstrap {
            self.commit_bootstrap(prepared);
        }
        self.request_senders
            .set(prepared_schedulers.request_senders)
            .expect("mocker request senders initialized more than once");
        self.command_senders
            .set(prepared_schedulers.command_senders)
            .expect("mocker command senders initialized more than once");
        self.handoff_session_permits
            .set(prepared_schedulers.handoff_session_permits)
            .expect("mocker handoff permits initialized more than once");
        self._schedulers
            .set(prepared_schedulers.schedulers)
            .unwrap_or_else(|_| panic!("mocker schedulers initialized more than once"));
        if let Some(publisher) = fpm_publisher {
            self._fpm_publisher
                .set(publisher)
                .unwrap_or_else(|_| panic!("mocker FPM publisher initialized more than once"));
        }
        *self._valkey_registration.lock().await = valkey_registration;
        self.senders_ready.notify_waiters();

        let handoff_shutdown = self.handoff_shutdown.clone();
        let scheduler_shutdown = self.scheduler_shutdown.clone();
        let handoff_tasks = self.handoff_tasks.clone();
        let scheduler_tasks = self.scheduler_tasks.clone();
        let source_manager = self.source_handoff_manager.get().cloned();
        let bootstrap_server = self.bootstrap_server.get().cloned();
        let valkey_registration = self._valkey_registration.clone();
        let kv_publisher_shutdowns = prepared_schedulers.kv_publisher_shutdowns;
        tokio::spawn(async move {
            primary_token.cancelled().await;
            handoff_shutdown.cancel();
            handoff_tasks.close();
            if let Some(manager) = source_manager {
                manager.wait_closed().await;
            }
            if let Some(server) = bootstrap_server {
                server.wait_closed().await;
            }
            handoff_tasks.wait().await;
            scheduler_shutdown.cancel();
            scheduler_tasks.close();
            scheduler_tasks.wait().await;
            let publisher_outcome = shutdown_publishers_and_wait(&kv_publisher_shutdowns).await;
            let registration = valkey_registration.lock().await.take();
            if let Some(mut registration) = registration {
                if publisher_outcome == KvEventPublisherShutdownOutcome::Drained {
                    if let Err(error) = registration.shutdown().await {
                        tracing::warn!(
                            error = %error,
                            "Failed to unregister mocker Valkey lease; expiry remains the backstop"
                        );
                    }
                } else {
                    tracing::warn!(
                        ?publisher_outcome,
                        "Skipping mocker Valkey unregister after forced publisher shutdown; lease expiry remains the backstop"
                    );
                }
            }
        });

        Ok(())
    }

    /// Send a request to the appropriate scheduler, waiting for initialization if needed.
    pub async fn direct(&self, request: DirectRequest, dp_rank: usize) {
        let sender = self.request_sender(dp_rank).await;
        let _ = sender.send(request);
    }

    async fn request_sender(&self, dp_rank: usize) -> mpsc::UnboundedSender<DirectRequest> {
        if let Some(senders) = self.request_senders.get() {
            return senders[dp_rank].clone();
        }

        // Register the waiter *before* re-checking to avoid a TOCTOU race
        // where `start_schedulers` sets + notifies between our check and subscribe.
        let notified = self.senders_ready.notified();
        if let Some(senders) = self.request_senders.get() {
            return senders[dp_rank].clone();
        }
        notified.await;

        let senders = self
            .request_senders
            .get()
            .expect("must be set after notify");
        senders[dp_rank].clone()
    }

    async fn command_sender(&self, dp_rank: usize) -> mpsc::Sender<SchedulerCommandEnvelope> {
        if let Some(senders) = self.command_senders.get() {
            return senders[dp_rank].clone();
        }
        let notified = self.senders_ready.notified();
        if let Some(senders) = self.command_senders.get() {
            return senders[dp_rank].clone();
        }
        notified.await;
        self.command_senders
            .get()
            .expect("scheduler command senders must be initialized before notification")[dp_rank]
            .clone()
    }

    async fn handoff_session_permit(&self, dp_rank: usize) -> Arc<Semaphore> {
        if let Some(permits) = self.handoff_session_permits.get() {
            return permits[dp_rank].clone();
        }
        let notified = self.senders_ready.notified();
        if let Some(permits) = self.handoff_session_permits.get() {
            return permits[dp_rank].clone();
        }
        notified.await;
        self.handoff_session_permits
            .get()
            .expect("handoff session permits must be initialized before notification")[dp_rank]
            .clone()
    }

    /// Create schedulers and spawn their background tasks for distributing token notifications.
    async fn start_schedulers(
        &self,
        component: Option<&Component>,
        cancel_token: CancellationToken,
        fpm_sinks: Vec<dynamo_mocker::common::protocols::FpmPublisher>,
        valkey_event_lease: Option<ValkeyWorkerEventLease>,
    ) -> std::result::Result<PreparedSchedulers, MockerSchedulerStartupFailure> {
        let args = &self.engine_args;
        let require_valkey_publisher = valkey_event_lease.is_some();
        let mut schedulers = Vec::<Box<dyn SchedulerHandle>>::new();
        let mut kv_publisher_shutdowns = Vec::with_capacity(args.dp_size as usize);
        let mut senders = Vec::with_capacity(args.dp_size as usize);
        let mut command_senders = Vec::with_capacity(args.dp_size as usize);
        let mut handoff_session_permits = Vec::with_capacity(args.dp_size as usize);

        for (dp_rank, fpm_publisher) in (0..args.dp_size).zip(fpm_sinks) {
            let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();
            let mut fatal_publisher_error = None;

            let (kv_event_publishers, relay_publisher, publisher_shutdown): (
                KvEventPublishers,
                Option<KvEventPublisher>,
                Option<KvEventPublisherShutdown>,
            ) = match component {
                Some(comp) if args.zmq_kv_events_port.is_some() => {
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
                            match KvEventPublisher::new_with_local_indexer_and_worker_id_and_valkey_lease(
                                comp.clone(),
                                None,
                                args.block_size as u32,
                                source_config,
                                args.enable_local_indexer,
                                dp_rank,
                                None,
                                valkey_event_lease.clone(),
                            ) {
                                Ok(publisher) => {
                                    let shutdown = publisher.shutdown_handle();
                                    (
                                        KvEventPublishers::new(
                                            None,
                                            Some(Arc::new(sink) as Arc<dyn RawKvEventSink>),
                                        ),
                                        Some(publisher),
                                        Some(shutdown),
                                    )
                                }
                                Err(e) => {
                                    tracing::error!(
                                        "Failed to create KV event relay for dp_rank {dp_rank}: {e}"
                                    );
                                    if require_valkey_publisher {
                                        fatal_publisher_error = Some(anyhow::anyhow!(
                                            "failed to create required Valkey KV event relay for DP rank {dp_rank}: {e:#}"
                                        ));
                                    }
                                    (KvEventPublishers::default(), None, None)
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to create ZMQ KV event sink for dp_rank {dp_rank}: {e}"
                            );
                            if require_valkey_publisher {
                                fatal_publisher_error = Some(anyhow::anyhow!(
                                    "failed to create required ZMQ KV event sink for DP rank {dp_rank}: {e:#}"
                                ));
                            }
                            (KvEventPublishers::default(), None, None)
                        }
                    }
                }
                Some(comp) => {
                    match KvEventPublisher::new_with_local_indexer_and_worker_id_and_valkey_lease(
                        comp.clone(),
                        None,
                        args.block_size as u32,
                        None,
                        args.enable_local_indexer,
                        dp_rank,
                        None,
                        valkey_event_lease.clone(),
                    ) {
                        Ok(publisher) => {
                            let shutdown = publisher.shutdown_handle();
                            (
                                KvEventPublishers::new(
                                    Some(Arc::new(KvEventSinkAdapter::new(publisher))
                                        as Arc<dyn KvCacheEventSink>),
                                    None,
                                ),
                                None,
                                Some(shutdown),
                            )
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to create KV event publisher for dp_rank {dp_rank}: {e}"
                            );
                            if require_valkey_publisher {
                                fatal_publisher_error = Some(anyhow::anyhow!(
                                    "failed to create required Valkey KV event publisher for DP rank {dp_rank}: {e:#}"
                                ));
                            }
                            (KvEventPublishers::default(), None, None)
                        }
                    }
                }
                None => (KvEventPublishers::default(), None, None),
            };
            if let Some(error) = fatal_publisher_error {
                cancel_token.cancel();
                self.scheduler_tasks.close();
                self.scheduler_tasks.wait().await;
                let publisher_outcome = shutdown_publishers_and_wait(&kv_publisher_shutdowns).await;
                return Err(MockerSchedulerStartupFailure {
                    error,
                    publisher_outcome,
                });
            }
            if let Some(shutdown) = publisher_shutdown {
                kv_publisher_shutdowns.push(shutdown);
            }

            let mut scheduler = create_engine(
                args.clone(),
                dp_rank,
                Some(output_tx),
                kv_event_publishers,
                Some(cancel_token.clone()),
                fpm_publisher,
            );

            senders.push(scheduler.request_sender());
            command_senders.push(scheduler.command_sender());
            handoff_session_permits
                .push(Arc::new(Semaphore::new(args.effective_handoff_capacity())));
            let mut lifecycle_rx = scheduler
                .take_lifecycle_receiver()
                .expect("new scheduler must expose one lifecycle receiver");
            schedulers.push(scheduler);

            let active_requests_clone = self.active_requests.clone();
            let cancel_token_cloned = cancel_token.clone();
            let handoff_events = self.handoff_events.clone();

            self.scheduler_tasks.spawn({
                let cancel_token = cancel_token.clone();
                async move {
                    loop {
                        tokio::select! {
                            biased;
                            _ = cancel_token.cancelled() => break,
                            event = lifecycle_rx.recv() => {
                                let Some(event) = event else {
                                    break;
                                };
                                handoff_events.deliver(event).await;
                            }
                        }
                    }
                }
            });

            self.scheduler_tasks.spawn(async move {
                // Keep the relay publisher alive for the lifetime of this task.
                // Dropping it would cancel its background ZMQ→NATS relay tasks.
                let _relay_publisher = relay_publisher;

                loop {
                    tokio::select! {
                        signal_result = output_rx.recv() => {
                            let Some(output_batch) = signal_result else {
                                break; // Channel closed
                            };

                            for signal in output_batch {
                                if let Some(request_tx) = active_requests_clone.get(&signal.uuid) {
                                    let _ = request_tx.send(signal);
                                }
                            }
                        }
                        _ = cancel_token_cloned.cancelled() => {
                            tracing::info!("Scheduler output task cancelled, clearing active requests");
                            active_requests_clone.clear();
                            break;
                        }
                    }
                }
            });
        }

        Ok(PreparedSchedulers {
            schedulers,
            kv_publisher_shutdowns,
            request_senders: senders,
            command_senders,
            handoff_session_permits,
        })
    }

    /// Start background tasks to publish metrics on change
    async fn start_metrics_publishing(
        schedulers: &[Box<dyn SchedulerHandle>],
        component: Component,
        native_metrics: Arc<NativeMockerMetrics>,
        cancel_token: CancellationToken,
        tasks: TaskTracker,
    ) -> Result<()> {
        let metrics_publisher = Arc::new(WorkerMetricsPublisher::new()?);

        if let Err(e) = metrics_publisher.create_endpoint(component).await {
            tracing::error!("Metrics endpoint failed: {e}");
        }
        for scheduler in schedulers.iter() {
            let mut metrics_rx = scheduler.metrics_receiver();
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
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<LLMEngineOutput>, Error> for MockEngine {
    async fn generate(
        &self,
        input: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<LLMEngineOutput>, Error> {
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

        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<OutputSignal>();
        self.active_requests.insert(request_uuid, request_tx);

        // Create a simple channel for the stream
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
            self.active_requests.remove(&request_uuid);
            return Err(Error::msg("disaggregated mocker requires a handoff ID"));
        }

        let handoff_cancel = CancellationToken::new();
        let mut source_completion_rx = None;
        let mut destination_error_rx = None;
        let mut destination_cleanup = None;

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
                    self.active_requests.remove(&request_uuid);
                    return Err(Error::msg(error.to_string()));
                }
            };
            let session_permit = match self
                .handoff_session_permit(dp_rank as usize)
                .await
                .try_acquire_owned()
            {
                Ok(permit) => permit,
                Err(_) => {
                    self.active_requests.remove(&request_uuid);
                    return Err(Error::msg(format!(
                        "mocker handoff session limit reached for DP rank {dp_rank}"
                    )));
                }
            };
            let command_tx = self.command_sender(dp_rank as usize).await;
            let lifecycle = match self.handoff_events.register(handoff_id) {
                Ok(lifecycle) => lifecycle,
                Err(error) => {
                    self.active_requests.remove(&request_uuid);
                    return Err(Error::msg(error.to_string()));
                }
            };

            if self.engine_args.is_prefill() {
                let Some(manager) = self.source_handoff_manager.get() else {
                    self.active_requests.remove(&request_uuid);
                    return Err(Error::msg("source handoff manager is not initialized"));
                };
                let (completion_tx, completion_rx) = oneshot::channel();
                if let Err(error) = manager.try_register(SourceRegistration {
                    identity,
                    order,
                    engine_type: self.engine_args.engine_type,
                    request: direct_request,
                    command_tx,
                    lifecycle,
                    completion_tx,
                    cancel: handoff_cancel.clone(),
                    observer: None,
                    _permit: session_permit,
                }) {
                    self.active_requests.remove(&request_uuid);
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
                        self.active_requests.remove(&request_uuid);
                        return Err(Error::msg(format!("bootstrap connection failed: {error}")));
                    }
                };
                let (error_tx, error_rx) = mpsc::unbounded_channel();
                let session_command_tx = command_tx.clone();
                let session_cancel = handoff_cancel.clone();
                let session_timeout =
                    Duration::from_millis(self.engine_args.handoff_session_timeout_ms);
                let global_shutdown = self.handoff_shutdown.clone();
                self.handoff_tasks.spawn(async move {
                    let _session_permit = session_permit;
                    if let Err(error) = run_destination_session(
                        connection,
                        direct_request,
                        session_command_tx,
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
                destination_cleanup = Some((command_tx, handoff_id));
            } else {
                self.active_requests.remove(&request_uuid);
                return Err(Error::msg(
                    "aggregated mocker request cannot carry handoff metadata",
                ));
            }
        } else {
            self.direct(direct_request, dp_rank as usize).await;
        }

        let active_requests = self.active_requests.clone();
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
                                let _ = stream_tx.send(LLMEngineOutput::error(error));
                                break;
                            }
                            Err(_) => {
                                let _ = stream_tx.send(LLMEngineOutput::error(
                                    "source handoff session ended without completion".to_string(),
                                ));
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
                                let _ = stream_tx.send(LLMEngineOutput::error(error));
                                break;
                            }
                            None => destination_error_rx = None,
                        }
                    }
                    maybe_signal = request_rx.recv() => {
                        let Some(signal) = maybe_signal else {
                            let _ = stream_tx.send(LLMEngineOutput::error("All output transmitters closed".to_string()));
                            break;
                        };

                        // A terminally rejected request never ran because it violated
                        // a worker admission limit. Emit no token and do not complete
                        // the bootstrap room; surface the rejection before any
                        // token/prefill bookkeeping.
                        if signal.rejected {
                            handoff_cancel.cancel();
                            let _ = stream_tx.send(LLMEngineOutput::error(
                                "request rejected: request exceeds worker admission limits".to_string(),
                            ));
                            break;
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

                        let output = LLMEngineOutput {
                            token_ids: vec![token_id],
                            disaggregated_params: is_prefill.then(|| serde_json::json!("dummy")),
                            ..Default::default()
                        };

                        if signal.completed && token_count < effective_max_output_tokens {
                            let _ = stream_tx.send(LLMEngineOutput::error("Completion signal received before max tokens reached".to_string()));
                            break;
                        }

                        if signal.completed {
                            if stream_tx.send(output).is_err() {
                                tracing::error!("Output stream receiver closed.");
                                break;
                            }
                            native_timing.record_tokens(1);

                            wait_for_no_bootstrap_handoff_delay(
                                is_prefill,
                                has_handoff_session,
                                signal.handoff_delay_ms,
                            )
                            .await;

                            if !source_handoff_complete
                                && let Some(completion_rx) = source_completion_rx.take()
                            {
                                let completion = tokio::select! {
                                    completion = completion_rx => completion,
                                    _ = async_context.stopped() => {
                                        handoff_cancel.cancel();
                                        let _ = stream_tx.send(LLMEngineOutput::cancelled());
                                        break;
                                    }
                                };
                                match completion {
                                    Ok(Ok(())) => {}
                                    Ok(Err(error)) => {
                                        let _ = stream_tx.send(LLMEngineOutput::error(error));
                                        break;
                                    }
                                    Err(_) => {
                                        let _ = stream_tx.send(LLMEngineOutput::error(
                                            "source handoff session ended without completion".to_string(),
                                        ));
                                        break;
                                    }
                                }
                            }

                            if stream_tx.send(LLMEngineOutput::length()).is_err() {
                                tracing::error!("Output stream receiver closed.");
                                break;
                            }
                            native_timing.record_normal_completion();
                            request_completed_normally = true;
                            break;
                        }

                        if stream_tx.send(output).is_err() {
                            tracing::error!("Output stream receiver closed.");
                            break;
                        }
                        native_timing.record_tokens(1);
                    }

                    _ = async_context.stopped() => {
                        handoff_cancel.cancel();
                        let _ = stream_tx.send(LLMEngineOutput::cancelled());
                        break;
                    }
                }
            }

            if !request_completed_normally {
                handoff_cancel.cancel();
                if let Some((command_tx, handoff_id)) = destination_cleanup.as_ref() {
                    cancel_destination(command_tx, *handoff_id, handoff_session_timeout).await;
                }
            }

            active_requests.remove(&request_uuid);
        };
        if let Some(tasks) = response_task_tracker {
            tasks.spawn(response_task);
        } else {
            tokio::spawn(response_task);
        }

        let stream = UnboundedReceiverStream::new(stream_rx);
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

pub struct AnnotatedMockEngine {
    inner: Arc<MockEngine>,
    startup: watch::Receiver<MockEngineStartup>,
}

#[derive(Clone, Debug)]
enum MockEngineStartup {
    Pending,
    Ready,
    Failed(Arc<str>),
    Cancelled,
}

async fn wait_for_mock_engine_startup(
    startup: &watch::Receiver<MockEngineStartup>,
) -> Result<(), Error> {
    let mut startup = startup.clone();
    loop {
        let state = startup.borrow_and_update().clone();
        match state {
            MockEngineStartup::Pending => {}
            MockEngineStartup::Ready => return Ok(()),
            MockEngineStartup::Failed(error) => {
                return Err(Error::msg(format!("mocker engine startup failed: {error}")));
            }
            MockEngineStartup::Cancelled => {
                return Err(Error::msg("mocker engine startup was cancelled"));
            }
        }
        startup.changed().await.map_err(|_| {
            Error::msg("mocker engine startup task exited without publishing a result")
        })?;
    }
}

impl AnnotatedMockEngine {
    pub fn new(
        inner: MockEngine,
        distributed_runtime: DistributedRuntime,
        endpoint_id: dynamo_runtime::protocols::EndpointId,
    ) -> Self {
        let inner = Arc::new(inner);
        let inner_clone = inner.clone();
        let (startup_tx, startup) = watch::channel(MockEngineStartup::Pending);

        // Start background task to wait for component service and start the engine
        let cancel_token = distributed_runtime.primary_token();
        tokio::spawn(async move {
            let component = loop {
                if cancel_token.is_cancelled() {
                    tracing::debug!("Mocker engine startup cancelled");
                    let _ = startup_tx.send(MockEngineStartup::Cancelled);
                    return;
                }

                let ready = distributed_runtime
                    .namespace(&endpoint_id.namespace)
                    .and_then(|ns| ns.component(&endpoint_id.component))
                    .ok();

                if let Some(comp) = ready
                    && let Ok(instances) = comp.list_instances().await
                    && !instances.is_empty()
                {
                    break comp;
                }

                tracing::debug!("Component service not available yet, retrying...");
                tokio::time::sleep(Duration::from_millis(100)).await;
            };

            tracing::debug!("Component service is now available, starting mocker engine");
            match inner_clone.start(component).await {
                Ok(()) => {
                    let _ = startup_tx.send(MockEngineStartup::Ready);
                }
                Err(error) => {
                    let error: Arc<str> = Arc::from(format!("{error:#}"));
                    tracing::error!(error = %error, "Failed to start mocker engine");
                    let _ = startup_tx.send(MockEngineStartup::Failed(error));
                }
            }
        });

        Self { inner, startup }
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for AnnotatedMockEngine
{
    async fn generate(
        &self,
        input: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        wait_for_mock_engine_startup(&self.startup).await?;
        let stream = self.inner.generate(input).await?;
        let context = stream.context();

        // Convert stream of LLMEngineOutput to Annotated<LLMEngineOutput>
        let annotated_stream = stream.map(Annotated::from_data);

        Ok(ResponseStream::new(Box::pin(annotated_stream), context))
    }
}

/// Create a mocker engine as ExecutionContext
pub async fn make_mocker_engine(
    distributed_runtime: DistributedRuntime,
    endpoint_id: dynamo_runtime::protocols::EndpointId,
    args: MockEngineArgs,
) -> Result<ExecutionContext, Error> {
    // Create the mocker engine
    tracing::info!("Creating mocker engine with config: {args:?}");
    let annotated_engine =
        AnnotatedMockEngine::new(MockEngine::new(args), distributed_runtime, endpoint_id);

    Ok(Arc::new(annotated_engine))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::llm_backend::PreprocessedRequest;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_mocker::common::protocols::{MockEngineArgs, OutputSignal, WorkerType};
    use dynamo_runtime::pipeline::{AsyncEngine, SingleIn};
    use futures::StreamExt;
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

    #[tokio::test]
    async fn startup_failure_rejects_requests_without_waiting_for_senders() {
        let (startup_tx, startup_rx) = watch::channel(MockEngineStartup::Pending);
        startup_tx
            .send(MockEngineStartup::Failed(Arc::from(
                "lease acquisition failed",
            )))
            .unwrap();

        let error = tokio::time::timeout(
            Duration::from_millis(100),
            wait_for_mock_engine_startup(&startup_rx),
        )
        .await
        .expect("startup failure must not hang")
        .unwrap_err();

        assert!(error.to_string().contains("lease acquisition failed"));
    }

    #[tokio::test]
    async fn startup_readiness_unblocks_only_after_success() {
        let (startup_tx, startup_rx) = watch::channel(MockEngineStartup::Pending);
        let waiter = tokio::spawn(async move { wait_for_mock_engine_startup(&startup_rx).await });
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());

        startup_tx.send(MockEngineStartup::Ready).unwrap();
        waiter.await.unwrap().unwrap();
    }

    #[test]
    fn startup_rollback_unregisters_only_after_proven_drain() {
        assert!(startup_rollback_can_unregister(
            KvEventPublisherShutdownOutcome::Drained
        ));
        assert!(!startup_rollback_can_unregister(
            KvEventPublisherShutdownOutcome::Forced
        ));
        assert!(!startup_rollback_can_unregister(
            KvEventPublisherShutdownOutcome::TimedOut
        ));
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
            .build()
            .unwrap();
        let engine = MockEngine::new(args);
        let (request_tx, mut request_rx) = tokio::sync::mpsc::unbounded_channel();
        engine.request_senders.set(vec![request_tx]).unwrap();

        let mut stream = engine
            .generate(SingleIn::new(prefill_request()))
            .await
            .unwrap();
        let request = request_rx.recv().await.unwrap();
        let request_id = request.uuid.unwrap();
        engine
            .active_requests
            .get(&request_id)
            .unwrap()
            .send(OutputSignal {
                uuid: request_id,
                token_id: None,
                completed: true,
                rejected: false,
                handoff_delay_ms: Some(100.0),
            })
            .unwrap();

        let token = stream.next().await.unwrap();
        assert_eq!(token.token_ids.len(), 1);
        assert!(token.finish_reason.is_none());
        assert!(
            tokio::time::timeout(Duration::from_millis(99), stream.next())
                .await
                .is_err()
        );

        tokio::time::advance(Duration::from_millis(1)).await;
        let finish = stream.next().await.unwrap();
        assert!(finish.token_ids.is_empty());
        assert!(finish.finish_reason.is_some());
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn context_capped_completion_maps_to_length() {
        let args = MockEngineArgs::builder()
            .max_model_len(Some(4))
            .build()
            .unwrap();
        let engine = MockEngine::new(args);
        let (request_tx, mut request_rx) = tokio::sync::mpsc::unbounded_channel();
        engine.request_senders.set(vec![request_tx]).unwrap();

        let mut stream = engine
            .generate(SingleIn::new(decode_request(3, 4)))
            .await
            .unwrap();
        let request = request_rx.recv().await.unwrap();
        assert_eq!(request.max_output_tokens, 4);
        let request_id = request.uuid.unwrap();
        engine
            .active_requests
            .get(&request_id)
            .unwrap()
            .send(OutputSignal {
                uuid: request_id,
                token_id: Some(42),
                completed: true,
                rejected: false,
                handoff_delay_ms: None,
            })
            .unwrap();

        let token = stream.next().await.unwrap();
        assert_eq!(token.token_ids.len(), 1);
        assert!(token.finish_reason.is_none());
        assert_eq!(stream.next().await.unwrap(), LLMEngineOutput::length());
        assert!(stream.next().await.is_none());
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
    async fn bootstrap_bind_failure_leaves_startup_state_retryable() {
        let occupied = std::net::TcpListener::bind(("0.0.0.0", 0)).unwrap();
        let port = occupied.local_addr().unwrap().port();
        let args = MockEngineArgs::builder()
            .worker_type(WorkerType::Prefill)
            .bootstrap_port(Some(port))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        let engine = MockEngine::new(args);

        assert!(engine.prepare_bootstrap().await.is_err());
        assert!(engine.request_senders.get().is_none());
        assert!(engine.command_senders.get().is_none());
        assert!(engine.handoff_session_permits.get().is_none());
        assert!(engine._schedulers.get().is_none());
        assert!(!engine.handoff_shutdown.is_cancelled());
        assert!(!engine.scheduler_shutdown.is_cancelled());

        drop(occupied);
        let prepared = engine
            .prepare_bootstrap()
            .await
            .unwrap()
            .expect("released bootstrap port must be reusable");
        engine.handoff_shutdown.cancel();
        prepared.server.wait_closed().await;
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
