// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use rmp_serde as rmps;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use zeromq::{Socket, SocketRecv, SubSocket};

use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;
use dynamo_runtime::{
    component::{Component, Namespace},
    transports::nats::{NatsQueue, Slug},
};

/// Helper function to create a KV stream name from a component and subject.
///
/// Generates a slugified stream name in the format:
/// `namespace-{namespace}-component-{component}-{subject}`
fn create_kv_stream_name(component: &Component, subject: &str) -> String {
    Slug::slugify(&format!(
        "namespace.{}.component.{}.{}",
        component.namespace().name(),
        component.name(),
        subject
    ))
    .to_string()
    .replace("_", "-")
}

pub use dynamo_kv_router::zmq_wire::create_stored_blocks;
use dynamo_kv_router::zmq_wire::*;

use crate::kv_router::{
    KV_EVENT_SUBJECT, KV_METRICS_SUBJECT, WORKER_KV_INDEXER_BUFFER_SIZE,
    indexer::{KvIndexerMetrics, LocalKvIndexer},
    protocols::*,
    worker_query::start_worker_kv_query_endpoint,
};
use dynamo_runtime::config::environment_names::nats as env_nats;

// Error handling configuration for ZMQ operations
const INITIAL_BACKOFF_MS: u64 = 10;
const MAX_BACKOFF_MS: u64 = 5000;
const MAX_CONSECUTIVE_ERRORS: u32 = 10;
const MAX_BACKOFF_EXPONENT: u32 = 8; // Cap at 2^8 = 256x multiplier to prevent overflow

// Batching configuration
const MAX_BATCHING_TIMEOUT_MS: u64 = 15_000; // 15 seconds, prevents misconfiguration
pub const DEFAULT_BATCHING_TIMEOUT_MS: Option<u64> = None; // disabled by default
const DEFAULT_MAX_BATCH_BLOCKS: usize = 128; // Max blocks to batch before flushing

// ---------------------------------------------------------------------------
// Engines dropped events metric
// ---------------------------------------------------------------------------

use std::sync::OnceLock;

use dynamo_runtime::metrics::prometheus_names::kv_publisher;

/// Metrics for the KV publisher, created via the MetricsHierarchy API.
/// This provides automatic `dynamo_namespace`, `dynamo_component`, and other
/// hierarchy labels for free.
pub struct KvPublisherMetrics {
    /// Total number of raw events dropped by engines before reaching publisher
    pub engines_dropped_events_total: prometheus::IntCounterVec,
}

static KV_PUBLISHER_METRICS: OnceLock<Arc<KvPublisherMetrics>> = OnceLock::new();

impl KvPublisherMetrics {
    /// Create from a Component, memoized in a static OnceLock.
    /// Uses the MetricsHierarchy API which auto-prepends `dynamo_component_`,
    /// injects hierarchy labels, and registers with the DRT `MetricsRegistry`.
    pub fn from_component(component: &Component) -> Arc<Self> {
        KV_PUBLISHER_METRICS
            .get_or_init(|| {
                let metrics = component.metrics();
                match metrics.create_intcountervec(
                    kv_publisher::ENGINES_DROPPED_EVENTS_TOTAL,
                    "Total number of raw events dropped by engines before reaching publisher (detected via event_id gaps)",
                    &["worker_id"],
                    &[],
                ) {
                    Ok(engines_dropped_events_total) => {
                        Arc::new(Self { engines_dropped_events_total })
                    }
                    Err(e) => {
                        tracing::warn!("Failed to create kv_publisher metrics from component: {}. Using unregistered metrics as fallback.", e);
                        Arc::new(Self::new_unregistered())
                    }
                }
            })
            .clone()
    }

    /// Creates unregistered metrics for use when the MetricsRegistry is not available.
    /// This is used as a fallback when metric creation fails.
    pub fn new_unregistered() -> Self {
        Self {
            engines_dropped_events_total: prometheus::IntCounterVec::new(
                prometheus::Opts::new(
                    kv_publisher::ENGINES_DROPPED_EVENTS_TOTAL,
                    "Total number of raw events dropped by engines before reaching publisher (detected via event_id gaps)",
                ),
                &["worker_id"],
            )
            .expect("failed to create engines_dropped_events_total counter"),
        }
    }

    /// Increment the engines dropped events counter by the given amount.
    pub fn increment_engines_dropped_events(&self, worker_id: u64, count: u64) {
        self.engines_dropped_events_total
            .with_label_values(&[&worker_id.to_string()])
            .inc_by(count);
    }
}

/// Get the KV publisher metrics if initialized.
fn kv_publisher_metrics() -> Option<Arc<KvPublisherMetrics>> {
    KV_PUBLISHER_METRICS.get().cloned()
}

// -------------------------------------------------------------------------
// Batching State -----------------------------------------------------------
// -------------------------------------------------------------------------

/// Accumulator for in-flight KV cache events that will be merged into a single
/// [`RouterEvent`] before being forwarded to the event sink.
#[derive(Debug)]
struct BatchingState {
    /// Block hashes accumulating for the next Removed event.
    pending_removed: Option<KvCacheRemoveData>,
    /// Blocks accumulating for the next Stored event.
    pending_stored: Option<KvCacheStoreData>,
    /// Monotonic published-batch counter. Increments by 1 per flush so downstream
    /// consumers always see consecutive event IDs, regardless of how many raw source
    /// events were merged into the batch.
    next_publish_id: u64,
    /// dp_rank of the events in the current pending batch.
    /// A change signals that the batch must be flushed before accumulating further.
    last_dp_rank: u32,
    /// When we last flushed (or initialized). Used to detect stale pending data:
    /// if a new event arrives after a long idle period (exceeding timeout),
    /// we flush immediately for lower latency on sparse important events.
    last_flush_time: Instant,
}

impl BatchingState {
    fn new() -> Self {
        Self {
            pending_removed: None,
            pending_stored: None,
            next_publish_id: 1,
            last_dp_rank: 0,
            last_flush_time: Instant::now(),
        }
    }

    fn has_pending(&self) -> bool {
        self.pending_removed.is_some() || self.pending_stored.is_some()
    }

    fn pending_block_count(&self) -> usize {
        self.pending_removed
            .as_ref()
            .map(|r| r.block_hashes.len())
            .unwrap_or(0)
            + self
                .pending_stored
                .as_ref()
                .map(|s| s.blocks.len())
                .unwrap_or(0)
    }

    /// Records that a flush just happened. Called after every flush to track
    /// idle periods for stale-data detection.
    fn record_flush_time(&mut self) {
        self.last_flush_time = Instant::now();
    }

    /// Returns the time remaining in the current batch window (zero if already elapsed).
    fn remaining_timeout(&self, timeout_ms: u64) -> Duration {
        let timeout = Duration::from_millis(timeout_ms);
        let elapsed = self.last_flush_time.elapsed();
        if elapsed >= timeout {
            Duration::ZERO
        } else {
            timeout - elapsed
        }
    }

    /// Returns `true` when the batch window has elapsed (or `timeout_ms` is zero).
    fn is_timeout_elapsed(&self, timeout_ms: u64) -> bool {
        self.remaining_timeout(timeout_ms) == Duration::ZERO
    }
}

// -------------------------------------------------------------------------
// KV Event Publishers -----------------------------------------------------
// -------------------------------------------------------------------------

/// Configure the source of KV events.
/// Currently, only ZMQ is supported.
pub enum KvEventSourceConfig {
    Zmq { endpoint: String, topic: String },
}

/// The source of KV events.
enum KvEventSource {
    Zmq {
        zmq_handle: tokio::task::JoinHandle<()>,
    },
}

impl KvEventSource {
    /// Start the event source from a [`KvEventSourceConfig`].
    fn start(
        component: Component,
        kv_block_size: u32,
        source_config: KvEventSourceConfig,
        cancellation_token: CancellationToken,
        tx: mpsc::UnboundedSender<KvCacheEvent>,
        next_event_id: Arc<AtomicU64>,
    ) -> Result<Self> {
        match source_config {
            KvEventSourceConfig::Zmq { endpoint, topic } => {
                let zmq_handle = component
                    .drt()
                    .runtime()
                    .secondary()
                    .spawn(start_zmq_listener(
                        endpoint,
                        topic,
                        tx,
                        cancellation_token.clone(),
                        kv_block_size,
                        next_event_id,
                    ));

                Ok(KvEventSource::Zmq { zmq_handle })
            }
        }
    }

    fn shutdown(&self) {
        match self {
            KvEventSource::Zmq { zmq_handle } => {
                zmq_handle.abort();
            }
        }
    }
}

/// A publisher of KV events.
pub struct KvEventPublisher {
    /// The size of the KV block.
    kv_block_size: u32,
    /// The source of KV events.
    /// Can be `None` if all events provided through [`KvEventPublisher::publish`].
    source: Option<KvEventSource>,
    /// The cancellation token.
    cancellation_token: CancellationToken,
    /// The channel to send events to.
    tx: mpsc::UnboundedSender<KvCacheEvent>,
    /// Internal monotonic event ID counter - ensures each event gets a unique, incrementing ID.
    /// Shared with the ZMQ listener (if any) to maintain consistency.
    next_event_id: Arc<AtomicU64>,
}

impl KvEventPublisher {
    pub fn new(
        component: Component,
        kv_block_size: u32,
        source_config: Option<KvEventSourceConfig>,
    ) -> Result<Self> {
        Self::new_with_local_indexer(
            component,
            kv_block_size,
            source_config,
            false,
            0,
            DEFAULT_BATCHING_TIMEOUT_MS,
        )
    }

    pub fn new_with_local_indexer(
        component: Component,
        kv_block_size: u32,
        source_config: Option<KvEventSourceConfig>,
        enable_local_indexer: bool,
        dp_rank: DpRank,
        batching_timeout_ms: Option<u64>,
    ) -> Result<Self> {
        let cancellation_token = CancellationToken::new();
        // None = disabled (flush every event); Some(0) normalised to None; Some(ms) = opt-in.
        // Cap at MAX_BATCHING_TIMEOUT_MS to prevent misconfiguration.
        let batching_timeout_ms = batching_timeout_ms
            .filter(|&ms| {
                if ms > MAX_BATCHING_TIMEOUT_MS {
                    tracing::warn!(
                        requested_ms = ms,
                        max_ms = MAX_BATCHING_TIMEOUT_MS,
                        "batching_timeout_ms too high, capping to 15s"
                    );
                }
                // if ms is 0, treat as disabled (None)
                ms > 0
            })
            .map(|ms| ms.min(MAX_BATCHING_TIMEOUT_MS));

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();

        // Infer worker_id from component's connection
        let worker_id = component.drt().connection_id();

        // Initialize the KV publisher metrics via MetricsHierarchy API
        // This provides automatic hierarchy labels (dynamo_namespace, dynamo_component, etc.)
        KvPublisherMetrics::from_component(&component);

        let component_name = component.name();
        tracing::info!(
            "Initializing KvEventPublisher for worker {worker_id} in component {component_name}"
        );

        if enable_local_indexer {
            tracing::info!(
                "LocalKvIndexer enabled for worker {worker_id} in component {component_name}"
            );
        }

        // Internal monotonic event ID counter - shared with ZMQ listener if any
        let next_event_id = Arc::new(AtomicU64::new(0));

        // Create our event source (if any)
        let mut source = None;
        if let Some(config) = source_config {
            source = Some(KvEventSource::start(
                component.clone(),
                kv_block_size,
                config,
                cancellation_token.clone(),
                tx.clone(),
                next_event_id.clone(),
            )?);
        }

        // Create local indexer if requested
        let local_indexer = if enable_local_indexer {
            let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
            Some(Arc::new(LocalKvIndexer::new(
                cancellation_token.clone(),
                kv_block_size,
                metrics,
                WORKER_KV_INDEXER_BUFFER_SIZE,
            )))
        } else {
            None
        };

        // Spawn runtime for router->local indexer comm if requested
        let _local_indexer_query_handle = local_indexer.as_ref().map(|local_indexer_ref| {
            let component = component.clone();
            let local_indexer = local_indexer_ref.clone();

            component
                .drt()
                .runtime()
                .secondary()
                .spawn(start_worker_kv_query_endpoint(
                    component,
                    worker_id,
                    dp_rank,
                    local_indexer,
                ))
        });

        let cancellation_token_clone = cancellation_token.clone();
        let local_indexer_clone = local_indexer.clone();

        if enable_local_indexer {
            // When local indexer is enabled, use the event plane directly.
            // EventPublisher handles transport selection (ZMQ or NATS) based on environment.
            // Durability is provided by the local indexer's event buffer.
            tracing::info!("Using event plane for KV event publishing (local_indexer mode)");
            let component_clone = component.clone();
            component.drt().runtime().secondary().spawn(async move {
                let event_publisher =
                    match EventPublisher::for_component(&component_clone, KV_EVENT_SUBJECT).await {
                        Ok(publisher) => publisher,
                        Err(e) => {
                            tracing::error!("Failed to create event publisher: {}", e);
                            return;
                        }
                    };

                start_event_processor(
                    EventPlanePublisher(event_publisher),
                    worker_id,
                    cancellation_token_clone,
                    rx,
                    local_indexer_clone,
                    batching_timeout_ms,
                )
                .await
            });
        } else {
            // When local indexer is disabled, use JetStream (NatsQueue) for durability.
            let stream_name = create_kv_stream_name(&component, KV_EVENT_SUBJECT);
            let nats_server = std::env::var(env_nats::NATS_SERVER)
                .unwrap_or_else(|_| "nats://localhost:4222".to_string());
            let mut nats_queue = NatsQueue::new_without_consumer(
                stream_name,
                nats_server,
                std::time::Duration::from_secs(60), // 1 minute timeout
            );

            component.drt().runtime().secondary().spawn(async move {
                if let Err(e) = nats_queue.connect().await {
                    tracing::error!("Failed to connect NatsQueue: {e}");
                    return;
                }
                start_event_processor_jetstream(
                    JetStreamPublisher(nats_queue),
                    worker_id,
                    cancellation_token_clone,
                    rx,
                    local_indexer_clone,
                    batching_timeout_ms,
                )
                .await
            });
        }

        Ok(Self {
            kv_block_size,
            source,
            cancellation_token,
            tx,
            next_event_id,
        })
    }

    pub fn publish(&self, event: KvCacheEvent) -> Result<(), mpsc::error::SendError<KvCacheEvent>> {
        self.tx.send(event)
    }

    /// Get and increment the next event ID atomically.
    /// Use this to assign monotonically increasing event IDs to events before publishing.
    pub fn next_event_id(&self) -> u64 {
        self.next_event_id.fetch_add(1, Ordering::SeqCst)
    }

    pub fn kv_block_size(&self) -> u32 {
        self.kv_block_size
    }

    pub fn shutdown(&mut self) {
        if !self.cancellation_token.is_cancelled() {
            self.cancellation_token.cancel();
        }

        if let Some(source) = self.source.take() {
            source.shutdown();
        }
    }
}

impl Drop for KvEventPublisher {
    fn drop(&mut self) {
        self.shutdown();
    }
}

use dynamo_kv_router::EventSink;

struct EventPlanePublisher(EventPublisher);

impl EventSink for EventPlanePublisher {
    fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
        self.0.publish(event)
    }
}

struct JetStreamPublisher(NatsQueue);

impl EventSink for JetStreamPublisher {
    fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
        NatsQueue::publish_event(&self.0, KV_EVENT_SUBJECT, event)
    }
}

/// Publishes a single [`KvCacheEvent`] to the event sink and, when present, the local indexer.
/// Errors are logged and swallowed so the caller loop can continue uninterrupted.
async fn emit<P: EventSink>(
    publisher: &P,
    local_indexer: &Option<Arc<LocalKvIndexer>>,
    worker_id: u64,
    event: KvCacheEvent,
) {
    let router_event = RouterEvent::new(worker_id, event);
    if let Some(indexer) = local_indexer
        && let Err(e) = indexer.apply_event_with_buffer(router_event.clone()).await
    {
        tracing::warn!(worker_id, error = %e, "Failed to apply event to local indexer");
    }
    if let Err(e) = publisher.publish_event(&router_event).await {
        tracing::error!(worker_id, error = %e, "Failed to publish event");
    }
}

impl BatchingState {
    /// Publishes any pending batch as a single [`RouterEvent`] and advances the monotonic
    /// batch ID. No-ops when nothing is pending, so callers may call unconditionally.
    async fn flush<P: EventSink + Send + Sync + 'static>(
        &mut self,
        publisher: &P,
        local_indexer: &Option<Arc<LocalKvIndexer>>,
        worker_id: u64,
    ) {
        if !self.has_pending() {
            return;
        }
        let id = self.next_publish_id;
        let dp_rank = self.last_dp_rank;
        if let Some(data) = self.pending_removed.take() {
            emit(
                publisher,
                local_indexer,
                worker_id,
                KvCacheEvent {
                    event_id: id,
                    data: KvCacheEventData::Removed(data),
                    dp_rank,
                },
            )
            .await;
        }
        if let Some(data) = self.pending_stored.take() {
            emit(
                publisher,
                local_indexer,
                worker_id,
                KvCacheEvent {
                    event_id: id,
                    data: KvCacheEventData::Stored(data),
                    dp_rank,
                },
            )
            .await;
        }
        // Consecutive batch IDs (1, 2, 3, …) keep downstream gap-detection happy.
        self.next_publish_id += 1;
        // Record when we flushed for stale-data detection on next event.
        self.record_flush_time();
    }
}

/// Batching loop: accumulates Removed/Stored events and flushes them as a single
/// [`RouterEvent`] when any of the following conditions are met:
/// - Event type switches (Removed ↔ Stored)
/// - `dp_rank` changes between consecutive events
/// - A `Stored` event's `parent_hash` breaks the sequential chain
/// - The batch window expires (`Some(timeout_ms)`; `None` = disabled, flush every event)
/// - Channel is closed or a cancellation signal is received
async fn run_event_processor_loop<P: EventSink + Send + Sync + 'static>(
    publisher: P,
    worker_id: u64,
    cancellation_token: CancellationToken,
    mut rx: mpsc::UnboundedReceiver<KvCacheEvent>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    timeout_ms: Option<u64>,
    max_batch_blocks: usize,
) {
    let mut batching_state = BatchingState::new();
    // Track last raw input event_id for gap detection (dropped events before batching).
    // The raw event_id is a globally monotonic counter assigned by the ZMQ listener,
    // so any gap here means events were silently dropped (e.g. send error on the channel).
    let mut last_raw_input_id: Option<u64> = None;

    loop {
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                tracing::info!("KV Event source received cancellation signal");
                batching_state.flush(&publisher, &local_indexer, worker_id).await;
                break;
            }
            event = rx.recv() => {
                let Some(event) = event else {
                    tracing::debug!("Event processor channel closed.");
                    batching_state.flush(&publisher, &local_indexer, worker_id).await;
                    break;
                };

                // Warn if the raw input event_id is not consecutive — events were dropped
                // (e.g. channel send error) before they reached the batching layer.
                let raw_event_id = event.event_id;
                if let Some(last_id) = last_raw_input_id
                    && raw_event_id > last_id + 1
                {
                    let gap = raw_event_id - last_id - 1;
                    tracing::warn!(
                        worker_id,
                        last_raw_input_id = last_id,
                        raw_event_id,
                        gap,
                        "Input event gap detected: raw events dropped before batching"
                    );
                    // Increment Prometheus counter for dropped events (if initialized)
                    if let Some(metrics) = kv_publisher_metrics() {
                        metrics.increment_engines_dropped_events(worker_id, gap);
                    } else {
                        tracing::warn!(
                            worker_id,
                            gap,
                            "Failed to record dropped events metric: metrics not initialized"
                        );
                    }
                }
                last_raw_input_id = Some(raw_event_id);

                tracing::trace!("Event processor for worker_id {} processing event: {:?}", worker_id, event.data);

                let dp_rank_changed = batching_state.has_pending()
                    && event.dp_rank != batching_state.last_dp_rank;

                match event.data {
                    KvCacheEventData::Removed(data) => {
                        if batching_state.pending_stored.is_some() || dp_rank_changed {
                            batching_state.flush(&publisher, &local_indexer, worker_id).await;
                        }
                        match &mut batching_state.pending_removed {
                            Some(pending) => pending.block_hashes.extend(data.block_hashes),
                            None => {
                                batching_state.pending_removed = Some(data);
                            }
                        }
                    }
                    KvCacheEventData::Stored(data) => {
                        // Flush if: type switch, dp_rank change, or the chain is broken
                        // (new event's parent_hash doesn't continue from the last stored block).
                        let should_flush = dp_rank_changed
                            || batching_state.pending_removed.is_some()
                            || batching_state.pending_stored.as_ref().is_some_and(|p| {
                                data.parent_hash != p.blocks.last().map(|b| b.block_hash)
                            });
                        if should_flush {
                            batching_state.flush(&publisher, &local_indexer, worker_id).await;
                        }
                        match &mut batching_state.pending_stored {
                            // Only extend blocks; parent_hash stays fixed from the first event.
                            Some(pending) => pending.blocks.extend(data.blocks),
                            None => {
                                batching_state.pending_stored = Some(data);
                            }
                        }
                    }
                    KvCacheEventData::Cleared => {
                        batching_state.flush(&publisher, &local_indexer, worker_id).await;
                        emit(&publisher, &local_indexer, worker_id, KvCacheEvent {
                            event_id: batching_state.next_publish_id,
                            data: KvCacheEventData::Cleared,
                            dp_rank: event.dp_rank,
                        }).await;
                        batching_state.next_publish_id += 1;
                    }
                }

                // Track dp_rank after the match so in-flight flushes use the old value.
                batching_state.last_dp_rank = event.dp_rank;

                // Flush after every event when disabled (None), or when the window has elapsed,
                // or when the batch exceeds the max block count.
                // The sleep arm only arms when batching is enabled; this covers the disabled path.
                if batching_state.has_pending()
                    && (timeout_ms.is_none_or(|ms| batching_state.is_timeout_elapsed(ms))
                        || batching_state.pending_block_count() > max_batch_blocks)
                {
                    batching_state.flush(&publisher, &local_indexer, worker_id).await;
                }
            }
            // if has some pending and has timeout, and no new events come in, then flush when timeout elapsed to prevent stale events
            _ = tokio::time::sleep(
                timeout_ms.map(|ms| batching_state.remaining_timeout(ms)).unwrap_or(Duration::from_secs(3600))
            ), if timeout_ms.is_some() && batching_state.has_pending() => {
                batching_state.flush(&publisher, &local_indexer, worker_id).await;
            }
        }
    }
}

/// Batched event processor for ephemeral transports (NATS Core / ZMQ).
async fn start_event_processor<P: EventSink + Send + Sync + 'static>(
    publisher: P,
    worker_id: u64,
    cancellation_token: CancellationToken,
    rx: mpsc::UnboundedReceiver<KvCacheEvent>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    batching_timeout_ms: Option<u64>,
) {
    run_event_processor_loop(
        publisher,
        worker_id,
        cancellation_token,
        rx,
        local_indexer,
        batching_timeout_ms,
        DEFAULT_MAX_BATCH_BLOCKS,
    )
    .await
}

/// Batched event processor using JetStream (durable).
async fn start_event_processor_jetstream<P: EventSink + Send + Sync + 'static>(
    publisher: P,
    worker_id: u64,
    cancellation_token: CancellationToken,
    rx: mpsc::UnboundedReceiver<KvCacheEvent>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    batching_timeout_ms: Option<u64>,
) {
    run_event_processor_loop(
        publisher,
        worker_id,
        cancellation_token,
        rx,
        local_indexer,
        batching_timeout_ms,
        DEFAULT_MAX_BATCH_BLOCKS,
    )
    .await
}

/// Calculate exponential backoff duration based on consecutive error count
fn calculate_backoff_ms(consecutive_errors: u32) -> u64 {
    std::cmp::min(
        INITIAL_BACKOFF_MS * 2_u64.pow(consecutive_errors.min(MAX_BACKOFF_EXPONENT)),
        MAX_BACKOFF_MS,
    )
}

pub async fn start_zmq_listener(
    zmq_endpoint: String,
    zmq_topic: String,
    tx: mpsc::UnboundedSender<KvCacheEvent>,
    cancellation_token: CancellationToken,
    kv_block_size: u32,
    next_event_id: Arc<AtomicU64>,
) {
    tracing::debug!(
        "KVEventPublisher connecting to ZMQ endpoint {} (topic '{}')",
        zmq_endpoint,
        zmq_topic
    );

    let warning_count = Arc::new(AtomicU32::new(0));

    let mut socket = SubSocket::new();

    // Subscribe to the requested topic (empty string == all topics)
    if let Err(e) = socket.subscribe(&zmq_topic).await {
        tracing::error!("Failed to subscribe on ZMQ socket: {}", e);
        return;
    }

    // Connect to the ZMQ endpoint. SGLang binds locally, Dynamo connects.
    // In multi-node setups, each node runs dynamo.sglang alongside local SGLang ranks,
    // so ZMQ connections are always local. NATS handles cross-node event distribution.
    if let Err(e) = socket.connect(&zmq_endpoint).await {
        tracing::error!("Failed to connect ZMQ SUB socket to {zmq_endpoint}: {e}");
        return;
    }

    let mut consecutive_errors = 0u32;
    #[expect(unused_assignments)]
    let mut exit_reason = "unknown";
    let mut messages_processed = 0u64;

    'main: loop {
        tokio::select! {
            biased;

            // Check for cancellation
            _ = cancellation_token.cancelled() => {
                tracing::debug!("ZMQ listener received cancellation signal");
                exit_reason = "cancellation token cancelled";
                break 'main;
            }

            // Receive message
            msg_result = socket.recv() => {
                let Ok(msg) = msg_result else {
                    let e = msg_result.unwrap_err();
                    consecutive_errors += 1;

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        tracing::error!(
                            error=%e,
                            consecutive_errors=%consecutive_errors,
                            "Too many consecutive ZMQ errors, terminating listener"
                        );
                        exit_reason = "too many consecutive errors";
                        break 'main;
                    }

                    // Simple exponential backoff with max exponent to prevent overflow
                    let backoff_ms = calculate_backoff_ms(consecutive_errors);

                    tracing::warn!(
                        error=%e,
                        consecutive_errors=%consecutive_errors,
                        backoff_ms=%backoff_ms,
                        "Error reading from ZMQ socket, applying exponential backoff"
                    );

                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    continue;
                };
                // Reset error count on successful message
                consecutive_errors = 0;

                // We expect multipart frames: [topic, seq, payload]
                let mut frames: Vec<Vec<u8>> = msg.into_vec().into_iter().map(|frame| frame.to_vec()).collect();

                if frames.len() != 3 {
                    tracing::warn!("Received unexpected ZMQ frame count: expected 3, actual {}", frames.len());
                    continue;
                }

                // Extract the payload and sequence number.
                let payload = frames.pop().unwrap();
                let seq_bytes = frames.pop().unwrap();

                if seq_bytes.len() != 8 {
                    tracing::warn!("Invalid sequence number byte length: expected 8, actual {}", seq_bytes.len());
                    continue;
                }

                // Note: We extract the engine's sequence number for logging but use our own
                // internal monotonic counter for event_id to ensure per-dp_rank monotonicity
                let engine_seq = u64::from_be_bytes(seq_bytes.try_into().unwrap());

                // Decode our batch of events.
                let batch_result = rmps::from_slice::<KvEventBatch>(&payload);
                let Ok(batch) = batch_result else {
                    let e = batch_result.unwrap_err();
                    tracing::warn!("Failed to decode KVEventBatch msgpack: {e}");
                    continue;
                };

                tracing::trace!(
                    "ZMQ listener on {} received batch with {} events (engine_seq={}, dp_rank={})",
                    zmq_endpoint,
                    batch.events.len(),
                    engine_seq,
                    batch.data_parallel_rank.unwrap_or(0)
                );

                let dp_rank = batch.data_parallel_rank.unwrap_or(0).cast_unsigned();
                for raw_event in batch.events.into_iter() {
                    // Use shared monotonic event_id counter instead of engine's sequence number
                    let event_id = next_event_id.fetch_add(1, Ordering::SeqCst);

                    let event = convert_event(raw_event, event_id, kv_block_size, dp_rank, &warning_count);
                    if tx.send(event).is_err() {
                        tracing::warn!("Failed to send message to channel - receiver dropped");
                        exit_reason = "channel receiver dropped";
                        break 'main;
                    }
                    messages_processed += 1;
                }
            }
        }
    }
    tracing::debug!(
        "ZMQ listener exiting, reason: {}, messages processed: {}",
        exit_reason,
        messages_processed
    );
}

// -------------------------------------------------------------------------
// Metrics Publishers ------------------------------------------------------
// -------------------------------------------------------------------------

/// Metrics data passed through the channel for NATS publishing
#[derive(Debug, Clone, Default, PartialEq)]
struct WorkerMetrics {
    dp_rank: DpRank,
    active_decode_blocks: u64,
}

pub struct WorkerMetricsPublisher {
    tx: tokio::sync::watch::Sender<WorkerMetrics>,
    rx: tokio::sync::watch::Receiver<WorkerMetrics>,
}

impl WorkerMetricsPublisher {
    pub fn new() -> Result<Self> {
        let (tx, rx) = tokio::sync::watch::channel(WorkerMetrics::default());
        Ok(WorkerMetricsPublisher { tx, rx })
    }

    /// Publish worker metrics for load monitoring.
    ///
    /// # Arguments
    /// * `dp_rank` - Data parallel rank of the worker (None defaults to 0)
    /// * `active_decode_blocks` - Number of active KV cache blocks
    pub fn publish(&self, dp_rank: Option<DpRank>, active_decode_blocks: u64) -> Result<()> {
        let metrics = WorkerMetrics {
            dp_rank: dp_rank.unwrap_or(0),
            active_decode_blocks,
        };
        tracing::trace!(
            "Publish metrics: dp_rank={}, active_decode_blocks={}",
            metrics.dp_rank,
            metrics.active_decode_blocks
        );
        self.tx
            .send(metrics)
            .map_err(|_| anyhow::anyhow!("metrics channel closed"))
    }

    pub async fn create_endpoint(&self, component: Component) -> Result<()> {
        let worker_id = component.drt().connection_id();
        self.start_nats_metrics_publishing(component.namespace().clone(), worker_id);
        Ok(())
    }

    /// Starts a background task to publish metrics over NATS
    ///
    /// This task monitors metric changes (specifically active_decode_blocks)
    /// and publishes stable metrics to NATS after they've been unchanged for 1ms.
    fn start_nats_metrics_publishing(&self, namespace: Namespace, worker_id: u64) {
        let nats_rx = self.rx.clone();

        tokio::spawn(async move {
            let event_publisher =
                match EventPublisher::for_namespace(&namespace, KV_METRICS_SUBJECT).await {
                    Ok(publisher) => publisher,
                    Err(e) => {
                        tracing::error!("Failed to create metrics publisher: {}", e);
                        return;
                    }
                };

            let mut rx = nats_rx;
            let mut last_metrics: Option<WorkerMetrics> = None;
            let mut pending_publish: Option<WorkerMetrics> = None;
            let mut publish_timer =
                Box::pin(tokio::time::sleep(tokio::time::Duration::from_secs(0)));
            publish_timer.as_mut().reset(tokio::time::Instant::now()); // Complete immediately

            loop {
                tokio::select! {
                    // Handle metrics changes
                    result = rx.changed() => {
                        if result.is_err() {
                            tracing::debug!(
                                "Metrics publisher sender dropped, stopping NATS background task"
                            );
                            break;
                        }

                        let metrics = rx.borrow_and_update().clone();

                        // Check if metrics have changed
                        let has_changed = last_metrics.as_ref() != Some(&metrics);

                        // If metrics changed, schedule a publish
                        if has_changed {
                            pending_publish = Some(metrics.clone());
                            last_metrics = Some(metrics);

                            // Start the 1ms timer
                            publish_timer.as_mut().reset(
                                tokio::time::Instant::now() + tokio::time::Duration::from_millis(1)
                            );
                        }
                    }
                    // Timer expired - publish if we have pending metrics
                    _ = &mut publish_timer => {
                        if let Some(metrics) = pending_publish.take() {
                            let active_load = ActiveLoad {
                                worker_id,
                                dp_rank: metrics.dp_rank,
                                active_decode_blocks: Some(metrics.active_decode_blocks),
                                active_prefill_tokens: None,
                            };

                            if let Err(e) = event_publisher.publish(&active_load).await {
                                tracing::warn!("Failed to publish metrics: {}", e);
                            }
                        }

                        // Reset timer to pending state to avoid tight loop
                        // It will be reset to 1ms when metrics actually change
                        publish_timer.as_mut().reset(
                            tokio::time::Instant::now() + tokio::time::Duration::from_secs(3600)
                        );
                    }
                }
            }
        });
    }
}

// -------------------------------------------------------------------------
// Testing -----------------------------------------------------------------
// -------------------------------------------------------------------------

#[cfg(test)]
mod test_event_processing {
    use super::*;
    use crate::kv_router::protocols::compute_block_hash_for_seq;

    // ---------------------------------------------------------------------
    // create_stored_block_from_parts --------------------------------------
    // ---------------------------------------------------------------------
    #[test]
    fn test_create_stored_block_from_parts() {
        let kv_block_size = 4;
        let token_ids = vec![10, 20, 30, 40];
        let blk_hash = 0xdead_beef;

        let stored =
            create_stored_block_from_parts(kv_block_size, blk_hash, &token_ids, None, None);

        assert_eq!(stored.block_hash.0, blk_hash);
        let expected_hash = compute_block_hash_for_seq(&token_ids, 4, None, None)[0];
        assert_eq!(stored.tokens_hash, expected_hash);
        assert!(stored.mm_extra_info.is_none());
    }

    // ---------------------------------------------------------------------
    // create_stored_blocks -------------------------------------------------
    // ---------------------------------------------------------------------
    #[test]
    fn test_create_stored_blocks_ok() {
        let kv_block_size = 4;
        // two blocks, each of size 4
        let token_ids = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let num_block_tokens = vec![4_u64, 4_u64];
        let block_hashes = vec![111_u64, 222_u64];

        let blocks = create_stored_blocks(
            kv_block_size,
            &token_ids,
            &num_block_tokens,
            &block_hashes,
            None,
            &Arc::new(AtomicU32::new(0)),
            None,
        );

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].block_hash.0, 111);
        assert_eq!(blocks[1].block_hash.0, 222);
    }

    #[test]
    fn test_create_stored_blocks_wrong_size_triggers_warning() {
        let kv_block_size = 4;
        let token_ids = vec![1, 2, 3, 4, 5, 6, 7];
        let num_block_tokens = vec![4_u64, 3_u64];
        let block_hashes = vec![111_u64, 222_u64];
        let warning_count = Arc::new(AtomicU32::new(0));

        let blocks = create_stored_blocks(
            kv_block_size,
            &token_ids,
            &num_block_tokens,
            &block_hashes,
            None,
            &warning_count,
            None,
        );

        // should early-exit as second has mismatch
        assert!(blocks.len() == 1);
        assert!(warning_count.load(Ordering::Relaxed) == 1)
    }

    // ---------------------------------------------------------------------
    // convert_event --------------------------------------------------------
    // ---------------------------------------------------------------------
    #[test]
    fn test_convert_event_block_stored() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(10), BlockHashValue::Unsigned(11)],
            parent_block_hash: Some(BlockHashValue::Unsigned(99)),
            token_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
            block_size: 4,
            medium: None,
            lora_name: None,
            block_mm_infos: None,
        };

        let out = convert_event(raw_evt, 42, kv_block_size, 0, &Arc::new(AtomicU32::new(0)));
        assert!(matches!(out.data, KvCacheEventData::Stored(_)));
    }

    #[test]
    fn test_convert_event_with_lora_name() {
        let kv_block_size = 4;
        let token_ids = vec![1, 2, 3, 4];

        let base_evt = RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(10)],
            parent_block_hash: None,
            token_ids: token_ids.clone(),
            block_size: 4,
            medium: None,
            lora_name: None,
            block_mm_infos: None,
        };
        let lora_evt = RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(10)],
            parent_block_hash: None,
            token_ids: token_ids.clone(),
            block_size: 4,
            medium: None,
            lora_name: Some("my-lora".to_string()),
            block_mm_infos: None,
        };

        let wc = Arc::new(AtomicU32::new(0));
        let base_out = convert_event(base_evt, 1, kv_block_size, 0, &wc);
        let lora_out = convert_event(lora_evt, 2, kv_block_size, 0, &wc);

        let base_hash = match &base_out.data {
            KvCacheEventData::Stored(s) => s.blocks[0].tokens_hash,
            _ => panic!("expected Stored"),
        };
        let lora_hash = match &lora_out.data {
            KvCacheEventData::Stored(s) => s.blocks[0].tokens_hash,
            _ => panic!("expected Stored"),
        };
        assert_ne!(
            base_hash, lora_hash,
            "LoRA blocks must produce distinct tokens_hash"
        );
    }

    #[test]
    fn test_convert_event_lora_name_none_is_base_model() {
        let kv_block_size = 4;
        let token_ids = vec![1, 2, 3, 4];
        let wc = Arc::new(AtomicU32::new(0));

        let evt1 = RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(10)],
            parent_block_hash: None,
            token_ids: token_ids.clone(),
            block_size: 4,
            medium: None,
            lora_name: None,
            block_mm_infos: None,
        };
        let evt2 = RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(10)],
            parent_block_hash: None,
            token_ids: token_ids.clone(),
            block_size: 4,
            medium: None,
            lora_name: None,
            block_mm_infos: None,
        };

        let out1 = convert_event(evt1, 1, kv_block_size, 0, &wc);
        let out2 = convert_event(evt2, 2, kv_block_size, 0, &wc);

        let hash1 = match &out1.data {
            KvCacheEventData::Stored(s) => s.blocks[0].tokens_hash,
            _ => panic!("expected Stored"),
        };
        let hash2 = match &out2.data {
            KvCacheEventData::Stored(s) => s.blocks[0].tokens_hash,
            _ => panic!("expected Stored"),
        };
        assert_eq!(
            hash1, hash2,
            "Two base-model events with same tokens should produce same hash"
        );
    }

    #[test]
    fn test_backward_compat_deserialize_map_with_lora_id_no_lora_name() {
        #[derive(serde::Serialize)]
        struct OldFormatEvent {
            #[serde(rename = "type")]
            event_type: &'static str,
            block_hashes: Vec<u64>,
            parent_block_hash: Option<u64>,
            token_ids: Vec<u32>,
            block_size: usize,
            lora_id: Option<u64>,
        }

        let payload = rmps::to_vec(&OldFormatEvent {
            event_type: "BlockStored",
            block_hashes: vec![42],
            parent_block_hash: None,
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
            lora_id: Some(5),
        })
        .unwrap();

        let event: RawKvEvent = rmps::from_slice(&payload).unwrap();
        let RawKvEvent::BlockStored { lora_name, .. } = event else {
            panic!("expected BlockStored");
        };
        assert!(
            lora_name.is_none(),
            "old-format payloads with lora_id but no lora_name should deserialize with lora_name=None"
        );
    }

    #[test]
    fn test_backward_compat_deserialize_seq_with_lora_id_no_lora_name() {
        let payload = rmps::to_vec(&(
            "BlockStored",
            vec![42_u64],
            None::<u64>,
            vec![1_u32, 2, 3, 4],
            4_usize,
            Some(5_u64), // lora_id at position 5
                         // no medium, no lora_name — simulating an old producer
        ))
        .unwrap();

        let event: RawKvEvent = rmps::from_slice(&payload).unwrap();
        let RawKvEvent::BlockStored { lora_name, .. } = event else {
            panic!("expected BlockStored");
        };
        assert!(
            lora_name.is_none(),
            "old seq-format payloads with lora_id should deserialize with lora_name=None"
        );
    }

    #[test]
    fn test_convert_event_block_removed() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::BlockRemoved {
            block_hashes: vec![BlockHashValue::Unsigned(123), BlockHashValue::Signed(456)],
            medium: None,
        };
        let out = convert_event(raw_evt, 7, kv_block_size, 0, &Arc::new(AtomicU32::new(0)));

        assert!(matches!(out.data, KvCacheEventData::Removed(_)));
    }

    #[test]
    fn test_convert_event_all_blocks_cleared() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::AllBlocksCleared;
        let out = convert_event(raw_evt, 1, kv_block_size, 0, &Arc::new(AtomicU32::new(0)));
        assert!(matches!(out.data, KvCacheEventData::Cleared));
    }

    #[test]
    fn test_parse_mm_hash_from_extra_key() {
        assert_eq!(
            parse_mm_hash_from_extra_key(
                "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210"
            ),
            Some(0x0123_4567_89ab_cdef)
        );
        assert_eq!(parse_mm_hash_from_extra_key("123"), None);
        assert_eq!(parse_mm_hash_from_extra_key("not_a_hash"), None);
    }

    #[test]
    fn test_extra_keys_to_block_mm_infos() {
        let mm_hash =
            "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210".to_string();
        let infos = extra_keys_to_block_mm_infos(Some(vec![
            Some(vec![mm_hash.clone()]),
            None,
            Some(vec!["invalid".to_string(), mm_hash]),
        ]))
        .expect("expected parsed MM infos");

        assert_eq!(infos.len(), 3);
        assert_eq!(
            infos[0].as_ref().unwrap().mm_objects[0].mm_hash,
            0x0123_4567_89ab_cdef
        );
        assert!(infos[1].is_none());
        assert_eq!(
            infos[2].as_ref().unwrap().mm_objects[0].mm_hash,
            0x0123_4567_89ab_cdef
        );
    }

    #[test]
    fn test_seq_block_stored_field8_supports_extra_keys() {
        let mm_hash =
            "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210".to_string();
        let extra_keys_payload = rmps::to_vec(&(
            "BlockStored",
            vec![10_u64],
            None::<u64>,
            vec![1_u32, 2, 3, 4],
            4_usize,
            None::<u64>,
            None::<String>,
            None::<String>,
            vec![Some(vec![mm_hash])],
        ))
        .unwrap();
        let extra_keys_event: RawKvEvent = rmps::from_slice(&extra_keys_payload).unwrap();
        let RawKvEvent::BlockStored {
            lora_name,
            block_mm_infos,
            ..
        } = extra_keys_event
        else {
            panic!("expected BlockStored");
        };
        assert!(lora_name.is_none());
        assert_eq!(
            block_mm_infos.unwrap()[0].as_ref().unwrap().mm_objects[0].mm_hash,
            0x0123_4567_89ab_cdef
        );
    }

    #[test]
    fn test_map_block_stored_supports_extra_keys() {
        #[derive(serde::Serialize)]
        struct MapBlockStoredEvent {
            #[serde(rename = "type")]
            event_type: &'static str,
            block_hashes: Vec<u64>,
            parent_block_hash: Option<u64>,
            token_ids: Vec<u32>,
            block_size: usize,
            lora_id: Option<u64>,
            medium: Option<String>,
            lora_name: Option<String>,
            extra_keys: Option<Vec<Option<Vec<String>>>>,
        }

        let payload = rmps::to_vec(&MapBlockStoredEvent {
            event_type: "BlockStored",
            block_hashes: vec![10],
            parent_block_hash: None,
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
            lora_id: None,
            medium: Some("GPU".to_string()),
            lora_name: None,
            extra_keys: Some(vec![Some(vec![
                "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210".to_string(),
            ])]),
        })
        .unwrap();

        let event: RawKvEvent = rmps::from_slice(&payload).unwrap();
        let RawKvEvent::BlockStored { block_mm_infos, .. } = event else {
            panic!("expected BlockStored");
        };
        assert_eq!(
            block_mm_infos.unwrap()[0].as_ref().unwrap().mm_objects[0].mm_hash,
            0x0123_4567_89ab_cdef
        );
    }
}

#[cfg(test)]
mod tests_startup_helpers {
    use super::*;
    use crate::kv_router::KvIndexer;
    use crate::kv_router::indexer::KvIndexerInterface;
    use crate::kv_router::protocols::{ExternalSequenceBlockHash, LocalBlockHash};
    use bytes::Bytes;
    use std::sync::{Arc, Mutex};
    use zeromq::{PubSocket, Socket, SocketSend, ZmqMessage};

    // Type alias to resolve clippy::type_complexity warning
    type PublishedEvents = Arc<Mutex<Vec<(String, Vec<u8>)>>>;

    //--------------------------------------------------------------------
    // A tiny stand-in for Component that just records every publish call
    //--------------------------------------------------------------------
    #[derive(Default)]
    struct MockComponent {
        published: PublishedEvents,
    }

    impl MockComponent {
        fn new() -> (Self, PublishedEvents) {
            let published = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    published: published.clone(),
                },
                published,
            )
        }
    }

    impl EventSink for MockComponent {
        fn publish_event(
            &self,
            event: &RouterEvent,
        ) -> impl Future<Output = anyhow::Result<()>> + Send {
            let bytes = rmp_serde::to_vec(event).unwrap();
            self.published
                .lock()
                .unwrap()
                .push((KV_EVENT_SUBJECT.to_string(), bytes));
            async { Ok(()) }
        }
    }

    //--------------------------------------------------------------------
    // Test start_event_processor
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_start_event_processor() {
        let (component, published) = MockComponent::new();

        let event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(1), ExternalSequenceBlockHash(2)],
            }),
            dp_rank: 0,
        };

        let token = CancellationToken::new();
        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        tx.send(event).unwrap();
        drop(tx);

        let handle = tokio::spawn(start_event_processor(
            component,
            1,
            token,
            rx,
            None,
            Some(10_000),
        ));

        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        let published = published.lock().unwrap();
        assert_eq!(published.len(), 1);
        let (subject, _) = &published[0];
        assert_eq!(subject, KV_EVENT_SUBJECT);
    }

    //--------------------------------------------------------------------
    // Test start_event_processor with local indexer
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_start_event_processor_with_local_indexer() {
        let (component, published) = MockComponent::new();

        // Create a local indexer
        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token.clone(), 4, metrics, 100));

        // Create BlockStored event
        let event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(100),
                        tokens_hash: LocalBlockHash(200),
                        mm_extra_info: None,
                    },
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(101),
                        tokens_hash: LocalBlockHash(201),
                        mm_extra_info: None,
                    },
                ],
            }),
            dp_rank: 0,
        };

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        tx.send(event).unwrap();
        drop(tx);

        // Start event processor with local indexer
        let handle = tokio::spawn(start_event_processor(
            component,
            1,
            token.clone(),
            rx,
            Some(local_indexer.clone()), // arc::clone just increments atomic counters
            Some(10_000),
        ));

        // Wait for processing
        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        // Verify event was published to NATS (same as test_start_event_processor)
        {
            let published_events = published.lock().unwrap();
            assert_eq!(published_events.len(), 1);
            let (subject, _) = &published_events[0];
            assert_eq!(subject, KV_EVENT_SUBJECT);
        } // drop lock

        // Verify event was applied to local indexer
        // We can check by querying the workers that have blocks
        let get_workers_tx = local_indexer.get_workers_sender();
        let mut found = false;
        for _ in 0..20 {
            // Try up to 20 times (200ms total)
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            get_workers_tx
                .send(crate::kv_router::indexer::GetWorkersRequest { resp: resp_tx })
                .await
                .unwrap();
            let workers: Vec<u64> = resp_rx.await.unwrap();

            if workers.contains(&1) {
                found = true;
                break;
            }

            // Wait before retrying
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        // Worker 1 should be in the set (we used worker_id=1)
        assert!(
            found,
            "Worker 1 was not found in the indexer after processing"
        );

        // Cleanup
        token.cancel();
    }

    //--------------------------------------------------------------------
    // Test BlockRemoved event with local indexer
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_event_processor_block_removed_with_local_indexer() {
        let (component, published) = MockComponent::new();

        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token.clone(), 4, metrics, 100));

        // First, store a block
        let store_event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(100),
                    tokens_hash: LocalBlockHash(200),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        };

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        tx.send(store_event).unwrap();

        // Start event processor with local indexer
        let handle = tokio::spawn(start_event_processor(
            component,
            1,
            token.clone(),
            rx,
            Some(local_indexer.clone()),
            Some(10_000),
        ));

        // Then remove same event
        let remove_event = KvCacheEvent {
            event_id: 2,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(100)],
            }),
            dp_rank: 0,
        };
        tx.send(remove_event).unwrap();
        drop(tx);

        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        // Local indexer should have no block
        let mut no_blocks = false;
        for _ in 0..20 {
            // Try up to 20 times (200ms total)
            let scores = local_indexer
                .find_matches(vec![LocalBlockHash(200)])
                .await
                .unwrap();
            if scores.scores.is_empty() {
                no_blocks = true;
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
        assert!(no_blocks, "worker should have no blocks after removal");

        // Global kvindexer should have recieved two events (create/remove)
        let published = published.lock().unwrap();
        assert_eq!(
            published.len(),
            2,
            "expected 2 published events, found {}",
            published.len()
        );

        token.cancel();
    }

    //--------------------------------------------------------------------
    // Test AllBlocksCleared event with local indexer
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_event_processor_all_blocks_cleared_with_local_indexer() {
        let (component, published) = MockComponent::new();

        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token.clone(), 4, metrics, 100));

        // Store a block
        let store_event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(100),
                    tokens_hash: LocalBlockHash(200),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        };

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        tx.send(store_event).unwrap();

        // Clear all blocks
        let clear_event = KvCacheEvent {
            event_id: 2,
            data: KvCacheEventData::Cleared,
            dp_rank: 0,
        };
        tx.send(clear_event).unwrap();
        drop(tx);

        // Create event processor and wait
        let handle = tokio::spawn(start_event_processor(
            component,
            1,
            token.clone(),
            rx,
            Some(local_indexer.clone()),
            Some(10_000),
        ));

        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        // Local indexer should have no block
        let mut no_blocks = false;
        for _ in 0..20 {
            // Try up to 20 times (200ms total)
            let scores = local_indexer
                .find_matches(vec![LocalBlockHash(200)])
                .await
                .unwrap();
            if scores.scores.is_empty() {
                no_blocks = true;
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
        assert!(no_blocks, "worker should have no blocks after clearing");

        // Global kvindexer should have recieved two events (create/remove)
        let published = published.lock().unwrap();
        assert_eq!(
            published.len(),
            2,
            "expected 2 published events, found {}",
            published.len()
        );

        token.cancel();
    }

    //--------------------------------------------------------------------
    // Test that local indexer failure doesn't break NATS publishing
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_event_processor_local_indexer_failure_continues() {
        let (component, published) = MockComponent::new();

        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token.clone(), 4, metrics, 100));

        // cancel indexer immediately to simulate failure
        token.cancel();

        let event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(1)],
            }),
            dp_rank: 0,
        };

        let new_token = CancellationToken::new();
        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        tx.send(event).unwrap();
        drop(tx);

        // Despite local indexer being cancelled, event processor should continue
        let handle = tokio::spawn(start_event_processor(
            component,
            1,
            new_token,
            rx,
            Some(local_indexer),
            Some(10_000),
        ));

        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        // Verify event was still published to NATS despite local indexer failure
        let published_events = published.lock().unwrap();
        assert_eq!(published_events.len(), 1);
    }

    //--------------------------------------------------------------------
    // Test start_zmq_listener without a real socket
    //   (feed it frames through a ZMQ PAIR tcp socket)
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_start_zmq_listener_pushes_to_channel() {
        // Prepare channel that listener should fill
        let (tx, mut rx) = mpsc::unbounded_channel::<KvCacheEvent>();

        // ZMQ TCP endpoint using localhost with fixed port
        let endpoint = "tcp://127.0.0.1:15555";
        let topic = "".to_string(); // subscribe to all

        // Publisher side - set up first
        let mut pub_socket = PubSocket::new();
        pub_socket.bind(endpoint).await.unwrap();

        // Cancellation token so we can stop the listener
        let token = dynamo_runtime::CancellationToken::new();
        // Event ID counter for the test listener
        let next_event_id = Arc::new(AtomicU64::new(0));

        // Spawn async listener (connects to publisher bound above)
        let listener_handle = tokio::spawn({
            let token = token.clone();
            start_zmq_listener(endpoint.to_string(), topic, tx, token, 4, next_event_id)
        });

        // Give time for the connection to establish
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Send synthetic 3-frame message: [topic, seq(8B), payload]
        let seq: u64 = 77;

        let events = vec![RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(42)],
            parent_block_hash: None,
            token_ids: vec![0, 1, 2, 3],
            block_size: 4,
            medium: None,
            lora_name: None,
            block_mm_infos: None,
        }];

        let batch = KvEventBatch {
            ts: 0.0,
            events,
            data_parallel_rank: Some(1),
        };

        let payload = Bytes::from(rmps::to_vec(&batch).unwrap());

        let frames = vec![
            Bytes::from(""),
            Bytes::from(seq.to_be_bytes().to_vec()),
            payload.clone(),
        ];

        // Create a proper multipart message
        let msg = ZmqMessage::try_from(frames).expect("Failed to create ZmqMessage");

        // Send the multipart message
        pub_socket.send(msg).await.unwrap();

        // Wait for message to be received
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Check that we received the message
        let event = rx.try_recv().expect("no message received");

        let KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash,
            blocks,
        }) = event.data
        else {
            panic!("expected KvCacheStoreData");
        };

        assert!(parent_hash.is_none());
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block_hash.0, 42);

        // Stop the listener
        token.cancel();
        let _ = listener_handle.await;
    }

    //--------------------------------------------------------------------
    // Test distributed recovery: Router queries worker's LocalKvIndexer after outage
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_distributed_kvindexer_recovery_from_outage() {
        let worker_1_id = 1u64;
        let block_size = 4u32;
        let token = CancellationToken::new();

        // === SETUP: Worker Components ===
        let (worker_component, worker_published) = MockComponent::new();
        let local_indexer_1 = Arc::new(LocalKvIndexer::new(
            token.clone(),
            block_size,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            100, // buffer size
        ));

        let (worker_tx, worker_rx) = mpsc::unbounded_channel::<KvCacheEvent>();

        // Start worker's event processor
        tokio::spawn(start_event_processor(
            worker_component,
            worker_1_id,
            token.clone(),
            worker_rx,
            Some(local_indexer_1.clone()),
            Some(10), // 10ms batching timeout
        ));

        // === SETUP: Router Components ===
        let router_indexer = Arc::new(KvIndexer::new(
            token.clone(),
            block_size,
            Arc::new(KvIndexerMetrics::new_unregistered()),
        ));

        // === STEP 1: Normal Operation ===
        let event_1 = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(100),
                        tokens_hash: LocalBlockHash(200),
                        mm_extra_info: None,
                    },
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(101),
                        tokens_hash: LocalBlockHash(201),
                        mm_extra_info: None,
                    },
                ],
            }),
            dp_rank: 0,
        };

        worker_tx.send(event_1.clone()).unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Simulate JetStream: forward worker's published event to router
        let (subject, bytes) = {
            let published = worker_published.lock().unwrap();
            assert_eq!(published.len(), 1, "Worker should have published 1 event");
            (published[0].0.clone(), published[0].1.clone())
        }; // drop worker_published before await
        assert_eq!(subject, KV_EVENT_SUBJECT);

        let router_event: RouterEvent = rmp_serde::from_slice(&bytes).unwrap();
        router_indexer
            .event_sender()
            .send(router_event)
            .await
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // assert: Router's indexer has event
        let get_workers_tx = router_indexer.get_workers_sender();
        let mut router_has_worker = false;
        for _ in 0..20 {
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            get_workers_tx
                .send(crate::kv_router::indexer::GetWorkersRequest { resp: resp_tx })
                .await
                .unwrap();
            let workers: Vec<u64> = resp_rx.await.unwrap();
            if workers.contains(&worker_1_id) {
                router_has_worker = true;
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
        assert!(
            router_has_worker,
            "Router should see worker 1 after normal operation"
        );

        // assert: Worker's local indexer buffered event
        let buffered = local_indexer_1.get_all_events_in_buffer();
        assert_eq!(buffered.len(), 1, "Local indexer should buffer 1 event");

        // === STEP 2 & 3: Simulate Outage - Stop forwarding to router ===
        let event_2 = KvCacheEvent {
            event_id: 2,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(100), // Shared prefix
                        tokens_hash: LocalBlockHash(200),
                        mm_extra_info: None,
                    },
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(102), // New block
                        tokens_hash: LocalBlockHash(202),
                        mm_extra_info: None,
                    },
                ],
            }),
            dp_rank: 0,
        };

        worker_tx.send(event_2.clone()).unwrap(); // send to worker but not to router
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // assert: Worker published event_2 to "NATS" (MockComponent)
        {
            let published = worker_published.lock().unwrap();
            assert_eq!(
                published.len(),
                2,
                "Worker should have published 2 events total"
            );
        }

        // assert: Worker's local indexer has both events
        let buffered = local_indexer_1.get_all_events_in_buffer();
        assert_eq!(
            buffered.len(),
            2,
            "Local indexer should have both events during outage"
        );

        // assert: Router DOESN'T have event_2
        let block_hashes_2 = vec![LocalBlockHash(200), LocalBlockHash(202)];
        let overlap = router_indexer
            .find_matches(block_hashes_2.clone())
            .await
            .unwrap();
        let router_overlap = overlap
            .scores
            .get(&crate::kv_router::protocols::WorkerWithDpRank::from_worker_id(worker_1_id))
            .copied()
            .unwrap_or(0);
        assert_eq!(
            router_overlap, 1,
            "Router should only see 1 shared block (not the new block from event_2)"
        );

        // === STEP 4 & 5: Recovery - Query worker's local indexer for missed events ===
        // In practice, the subscriber detects gaps and triggers recovery automatically.
        // Here we simulate that by querying for events after event_id=1.
        let last_known_id = 1u64; // Router only received event_1
        let response = local_indexer_1
            .get_events_in_id_range(Some(last_known_id + 1), None)
            .await;
        let missed_events = match response {
            crate::kv_router::indexer::WorkerKvQueryResponse::Events(e) => e,
            crate::kv_router::indexer::WorkerKvQueryResponse::TreeDump { events: e, .. } => e,
            crate::kv_router::indexer::WorkerKvQueryResponse::Error(message) => {
                panic!("Unexpected error response: {message}")
            }
            other => panic!("Unexpected response: {:?}", other),
        };
        assert_eq!(
            missed_events.len(),
            1,
            "Should get 1 missed event (event_2 with id=2)"
        );

        // Step 5: Apply missed events to router
        for router_event in missed_events {
            router_indexer
                .event_sender()
                .send(router_event)
                .await
                .unwrap();
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // assert: Router now has complete state
        let overlap = router_indexer.find_matches(block_hashes_2).await.unwrap();
        let router_overlap_after = overlap
            .scores
            .get(&crate::kv_router::protocols::WorkerWithDpRank::from_worker_id(worker_1_id))
            .copied()
            .unwrap_or(0);
        assert_eq!(
            router_overlap_after, 2,
            "Router should now see both blocks after recovery"
        );

        token.cancel();
    }
}

#[cfg(test)]
mod test_exponential_backoff {
    use super::*;

    #[test]
    fn test_backoff_calculation_progression() {
        // Test the exponential progression
        assert_eq!(calculate_backoff_ms(0), 10); // 10 * 2^0 = 10
        assert_eq!(calculate_backoff_ms(1), 20); // 10 * 2^1 = 20
        assert_eq!(calculate_backoff_ms(2), 40); // 10 * 2^2 = 40
        assert_eq!(calculate_backoff_ms(3), 80); // 10 * 2^3 = 80
        assert_eq!(calculate_backoff_ms(4), 160); // 10 * 2^4 = 160
        assert_eq!(calculate_backoff_ms(5), 320); // 10 * 2^5 = 320
        assert_eq!(calculate_backoff_ms(6), 640); // 10 * 2^6 = 640
        assert_eq!(calculate_backoff_ms(7), 1280); // 10 * 2^7 = 1280
        assert_eq!(calculate_backoff_ms(8), 2560); // 10 * 2^8 = 2560
    }

    #[test]
    fn test_backoff_caps_at_max_exponent() {
        // After MAX_BACKOFF_EXPONENT, should stay at 2^8 = 2560ms
        assert_eq!(calculate_backoff_ms(8), 2560);
        assert_eq!(calculate_backoff_ms(9), 2560); // Same as 8
        assert_eq!(calculate_backoff_ms(100), 2560); // Same as 8
    }

    #[test]
    fn test_backoff_never_exceeds_max() {
        // Even if we somehow had a huge exponent, never exceed MAX_BACKOFF_MS
        for i in 0..20 {
            assert!(calculate_backoff_ms(i) <= MAX_BACKOFF_MS);
        }
    }

    #[test]
    #[expect(clippy::assertions_on_constants)]
    fn test_backoff_constants_are_sane() {
        // Verify our constants make sense together
        assert!(INITIAL_BACKOFF_MS > 0);
        assert!(MAX_BACKOFF_MS > INITIAL_BACKOFF_MS);
        assert!(MAX_BACKOFF_EXPONENT <= 10); // Prevent crazy exponents
        assert!(MAX_CONSECUTIVE_ERRORS > 0);

        // Max calculated value should be less than MAX_BACKOFF_MS
        let max_calculated = INITIAL_BACKOFF_MS * 2_u64.pow(MAX_BACKOFF_EXPONENT);
        assert!(max_calculated <= MAX_BACKOFF_MS);
    }
}

#[cfg(all(test, feature = "integration"))]
mod test_integration_publisher {
    use super::*;
    use crate::kv_router::protocols::ActiveLoad;
    use dynamo_runtime::distributed_test_utils::create_test_drt_async;
    use dynamo_runtime::transports::event_plane::EventSubscriber;

    #[tokio::test]
    #[ignore] // Mark as ignored as requested, because CI's integrations still don't have NATS
    async fn test_metrics_publishing_behavior() -> Result<()> {
        // Set up runtime and namespace
        let drt = create_test_drt_async().await;
        let namespace = drt.namespace("ns2001".to_string())?;

        // Create a subscriber for the metrics events
        let mut subscriber = EventSubscriber::for_namespace(&namespace, KV_METRICS_SUBJECT)
            .await
            .unwrap()
            .typed::<ActiveLoad>();

        // Create WorkerMetricsPublisher
        let publisher = WorkerMetricsPublisher::new().unwrap();
        let worker_id = 1234;

        // Start NATS metrics publishing
        publisher.start_nats_metrics_publishing(namespace.clone(), worker_id);

        // Allow some time for the background task to start
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Test 1: Publish 10 different metrics with 0.5ms intervals
        // Only the last one should be published after 1ms of stability
        for i in 0..10 {
            publisher.publish(None, (i * 100) as u64).unwrap();
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }

        // Wait a bit more than 1ms to ensure the last metric is published
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Verify we receive exactly one event with the last metric values
        let result =
            tokio::time::timeout(tokio::time::Duration::from_millis(500), subscriber.next())
                .await
                .unwrap();

        let (_envelope, event) = result.unwrap().unwrap(); // Unwrap the Option and the Result
        assert_eq!(event.worker_id, worker_id);
        assert_eq!(event.active_decode_blocks, Some(900)); // Last value: 9 * 100
        assert_eq!(event.active_prefill_tokens, None); // Worker doesn't publish prefill tokens

        // Ensure no more events are waiting
        let no_msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(50), subscriber.next()).await;
        assert!(no_msg.is_err(), "Expected no more messages, but found one");

        // Test 2: Publish 10 more metrics with same active_decode_blocks - should not trigger publish
        for _ in 0..10 {
            publisher.publish(None, 900).unwrap(); // Keep same as last published
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }

        // Wait to ensure no events are published
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Verify no events are received
        let no_msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(50), subscriber.next()).await;
        assert!(
            no_msg.is_err(),
            "Expected no messages when load metrics don't change"
        );

        drt.shutdown();

        Ok(())
    }
}

#[cfg(test)]
mod batching_state_tests {
    use super::*;

    #[test]
    fn test_batching_state_default() {
        let state = BatchingState::new();
        assert!(!state.has_pending(), "Default state should have no pending");
        assert!(
            state.pending_removed.is_none(),
            "Default pending_removed should be None"
        );
        assert!(
            state.pending_stored.is_none(),
            "Default pending_stored should be None"
        );
    }

    #[test]
    fn test_batching_state_new() {
        let state = BatchingState::new();
        // last_flush_time should be set to approximately now
        let elapsed = state.last_flush_time.elapsed();
        assert!(
            elapsed < Duration::from_secs(1),
            "new() should create state with flush time set to approximately now"
        );
    }

    #[test]
    fn test_batching_state_pending_removed() {
        let mut state = BatchingState::new();
        assert!(!state.has_pending(), "Should not have pending initially");

        state.pending_removed = Some(KvCacheRemoveData {
            block_hashes: vec![],
        });
        assert!(
            state.has_pending(),
            "Should have pending after setting pending_removed"
        );
    }

    #[test]
    fn test_batching_state_pending_stored() {
        let mut state = BatchingState::new();
        assert!(!state.has_pending(), "Should not have pending initially");

        state.pending_stored = Some(KvCacheStoreData {
            parent_hash: None,
            blocks: vec![],
        });
        assert!(
            state.has_pending(),
            "Should have pending after setting pending_stored"
        );
    }

    #[test]
    fn test_batching_state_timeout() {
        let mut state = BatchingState::new();

        // Reset flush time to now so we can test timeout behavior
        state.record_flush_time();

        // Test that remaining returns positive initially (10ms timeout)
        let remaining_before = state.remaining_timeout(10);
        assert!(
            remaining_before.as_millis() > 0,
            "Should have remaining time initially"
        );

        // Test zero timeout returns zero
        let remaining_zero = state.remaining_timeout(0);
        assert_eq!(
            remaining_zero.as_millis(),
            0,
            "0 timeout should return zero"
        );
    }

    #[test]
    fn test_batching_state_record_flush_time() {
        let mut state = BatchingState::new();

        let initial_time = state.last_flush_time;

        state.record_flush_time();

        assert!(
            state.last_flush_time >= initial_time,
            "record_flush_time should update the time"
        );
    }

    #[test]
    fn test_batching_state_remaining_timeout() {
        let mut state = BatchingState::new();

        // Reset flush time to now so we can test timeout behavior
        state.record_flush_time();

        // Test that remaining returns positive initially (10ms timeout)
        let remaining = state.remaining_timeout(10);
        assert!(
            remaining.as_millis() > 0,
            "Should have remaining time initially"
        );

        // Test that with 0 timeout, returns zero
        let remaining_zero = state.remaining_timeout(0);
        assert_eq!(
            remaining_zero,
            Duration::ZERO,
            "0 timeout should return zero"
        );
    }

    #[test]
    fn test_batching_state_accumulate_removed() {
        let mut state = BatchingState::new();

        let first = KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(1), ExternalSequenceBlockHash(2)],
        };

        state.pending_removed = Some(first);

        if let Some(ref mut pending) = state.pending_removed {
            pending
                .block_hashes
                .extend(vec![ExternalSequenceBlockHash(3)]);
        }

        let pending = state.pending_removed.as_ref().unwrap();
        assert_eq!(
            pending.block_hashes.len(),
            3,
            "Should have accumulated 3 block hashes"
        );
    }

    #[test]
    fn test_batching_state_accumulate_stored() {
        let mut state = BatchingState::new();

        let block1 = KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(1),
            tokens_hash: LocalBlockHash(100),
            mm_extra_info: None,
        };
        let first = KvCacheStoreData {
            parent_hash: Some(ExternalSequenceBlockHash(0)),
            blocks: vec![block1],
        };

        state.pending_stored = Some(first);

        let block2 = KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(2),
            tokens_hash: LocalBlockHash(200),
            mm_extra_info: None,
        };

        if let Some(ref mut pending) = state.pending_stored {
            pending.blocks.extend(vec![block2]);
        }

        let pending = state.pending_stored.as_ref().unwrap();
        assert_eq!(pending.blocks.len(), 2, "Should have accumulated 2 blocks");
    }
}

#[cfg(test)]
mod event_processor_tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio_util::sync::CancellationToken;

    /// Mock publisher that collects published events
    #[derive(Debug, Clone)]
    struct MockPublisher {
        events: Arc<Mutex<Vec<RouterEvent>>>,
    }

    impl MockPublisher {
        fn new() -> Self {
            Self {
                events: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn get_events(&self) -> Vec<RouterEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl EventSink for MockPublisher {
        fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
            self.events.lock().unwrap().push(event.clone());
            async { Ok(()) }
        }
    }

    /// Test that pushing N removed events results in batched output
    /// Uses a 10ms timeout to ensure events are batched (events sent rapidly)
    #[tokio::test]
    async fn test_run_event_processor_loop_batches_removed_events_20() {
        test_removed_events_batching(20, Some(10)).await; // 20 events, 10ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_removed_events_10() {
        test_removed_events_batching(10, Some(10)).await; // 10 events, 10ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_removed_events_5() {
        test_removed_events_batching(5, Some(10)).await; // 5 events, 10ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_removed_events_3() {
        test_removed_events_batching(3, Some(10)).await; // 3 events, 10ms timeout
    }

    /// Helper function to test removed events batching with configurable count and timeout
    async fn test_removed_events_batching(event_count: usize, timeout_ms: Option<u64>) {
        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        for i in 0..event_count {
            let event = KvCacheEvent {
                event_id: i as u64,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
                }),
                dp_rank: 0,
            };
            tx.send(event).unwrap();
            // Yield to allow event processor to process the event
            tokio::task::yield_now().await;
        }

        // Wait for timeout to elapse so all events flush together as one batch
        // Add small buffer to ensure flush happens before channel close
        tokio::time::sleep(tokio::time::Duration::from_millis(
            timeout_ms.unwrap_or(0) + 1,
        ))
        .await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        assert!(
            !events.is_empty(),
            "Should have received at least one event"
        );

        // With a long timeout (100ms) and rapid event sending, all events should batch into few output events
        // (first event may flush separately, rest should batch together)
        assert!(
            events.len() <= 2,
            "With long timeout ({timeout_ms:?}), all {event_count} events should batch into at most 2 output events (got {})",
            events.len()
        );

        let total_hashes: usize = events
            .iter()
            .map(|e| {
                if let KvCacheEventData::Removed(data) = &e.event.data {
                    data.block_hashes.len()
                } else {
                    0
                }
            })
            .sum();
        assert_eq!(
            total_hashes, event_count,
            "All {} block hashes should be accounted for",
            event_count
        );
    }

    /// Test sequential stored events accumulate with different counts
    /// Uses a longer timeout (100ms) to ensure events have time to batch
    #[tokio::test]
    async fn test_run_event_processor_loop_batches_stored_events_20() {
        test_stored_events_batching(20, Some(100)).await; // 20 events, 100ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_stored_events_10() {
        test_stored_events_batching(10, Some(100)).await; // 10 events, 100ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_stored_events_5() {
        test_stored_events_batching(5, Some(100)).await; // 5 events, 100ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_stored_events_3() {
        test_stored_events_batching(3, Some(100)).await; // 3 events, 100ms timeout
    }

    /// Helper function to test stored events batching with configurable count and timeout
    async fn test_stored_events_batching(event_count: usize, timeout_ms: Option<u64>) {
        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        for i in 0..event_count {
            // For sequential batching, each event's parent_hash should be the previous event's block_hash
            let parent_hash = if i == 0 {
                Some(ExternalSequenceBlockHash(0)) // First event has parent_hash = 0
            } else {
                Some(ExternalSequenceBlockHash((i - 1) as u64)) // Subsequent events reference previous block
            };

            let event = KvCacheEvent {
                event_id: i as u64,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(i as u64),
                        tokens_hash: LocalBlockHash(i as u64 * 100),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            };
            tx.send(event).unwrap();
            // Small sleep to allow event processor to batch events
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }

        // Give the processor time to process all events before closing the channel
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        assert!(
            !events.is_empty(),
            "Should have received at least one event"
        );

        // With a long timeout, events should be batched. Either 1 or can be at most 2, if the first event flushes separately due to initial timestamp.
        assert!(
            events.len() <= 2,
            "With long timeout ({timeout_ms:?}) and sequential parent hashes, all {event_count} events should batch into at most 2 output events (got {})",
            events.len()
        );
        if events.len() == 2 {
            // If we got 2 events, the first one should contain only the first block, and the second should contain the rest
            if let KvCacheEventData::Stored(data) = &events[0].event.data {
                assert_eq!(
                    data.blocks.len(),
                    1,
                    "If 2 events, first event should have 1 block (got {})",
                    data.blocks.len()
                );
            } else {
                panic!("Expected Stored event");
            }
        }

        let total_blocks: usize = events
            .iter()
            .map(|e| {
                if let KvCacheEventData::Stored(data) = &e.event.data {
                    data.blocks.len()
                } else {
                    0
                }
            })
            .sum();
        assert_eq!(
            total_blocks, event_count,
            "All {} blocks should be accounted for",
            event_count
        );
    }

    /// Test non-sequential stored events trigger flush
    #[tokio::test]
    async fn test_run_event_processor_loop_non_sequential_flush() {
        let timeout_ms = Some(100); // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
            // SLEEP HERE?! so that events are not batched!
        });

        for i in 0..3 {
            let event = KvCacheEvent {
                event_id: i as u64,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: Some(ExternalSequenceBlockHash((i + 1) as u64 * 100)),
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(i as u64),
                        tokens_hash: LocalBlockHash(i as u64 * 100),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            };
            tx.send(event).unwrap();
        }

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        assert!(!events.is_empty(), "Should have received events");

        // With non-sequential parent hashes, each event should trigger a flush
        // So we expect 3 separate events
        assert_eq!(
            events.len(),
            3,
            "Non-sequential events should trigger flush, resulting in 3 separate events"
        );

        let total_blocks: usize = events
            .iter()
            .map(|e| {
                if let KvCacheEventData::Stored(data) = &e.event.data {
                    data.blocks.len()
                } else {
                    0
                }
            })
            .sum();
        assert_eq!(total_blocks, 3, "All 3 blocks should be accounted for");
    }

    /// Test that with short timeout and slow input, events are NOT batched
    /// Parametrized over different timeout values: 0ms, 0.1ms, 0.2ms
    /// All use 2ms delay between events, so each event times out before the next arrives
    #[tokio::test]
    async fn test_run_event_processor_loop_no_batching_with_slow_input_0ms() {
        test_no_batching_with_slow_input(None).await; // disabled (no timeout)
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_no_batching_with_slow_input_0_1ms() {
        test_no_batching_with_slow_input(Some(1)).await; // 1ms timeout (was 0.1ms in us)
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_no_batching_with_slow_input_0_2ms() {
        test_no_batching_with_slow_input(Some(2)).await; // 2ms timeout (was 0.2ms in us)
    }

    /// Helper function to test no batching with slow input
    async fn test_no_batching_with_slow_input(timeout_ms: Option<u64>) {
        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        // Send 5 removed events with 2ms delay between each
        // Since timeout is <= 0.2ms, each event should timeout and be sent individually
        for i in 0..5 {
            let event = KvCacheEvent {
                event_id: i as u64,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
                }),
                dp_rank: 0,
            };
            tx.send(event).unwrap();
            // Wait 2ms between events (much longer than the timeout)
            // This ensures each event times out before the next one arrives
            tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
        }

        // Give the processor time to process the last event
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        assert!(!events.is_empty(), "Should have received events");

        // With slow input (2ms delay) and short timeout, most events should be sent individually
        // We expect at least 3 separate events (showing reduced batching)
        assert!(
            events.len() >= 3,
            "With slow input (2ms delay) and timeout={timeout_ms:?}, should have at least 3 separate events (got {})",
            events.len()
        );

        let total_hashes: usize = events
            .iter()
            .map(|e| {
                if let KvCacheEventData::Removed(data) = &e.event.data {
                    data.block_hashes.len()
                } else {
                    0
                }
            })
            .sum();
        assert_eq!(
            total_hashes, 5,
            "All 5 block hashes should be accounted for"
        );
    }

    /// Test that switching between Removed and Stored events causes immediate flush
    #[tokio::test]
    async fn test_event_type_switching_causes_flush() {
        let timeout_ms = Some(100); // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        // Send a Removed event
        tx.send(KvCacheEvent {
            event_id: 0,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(0)],
            }),
            dp_rank: 0,
        })
        .unwrap();

        // Small sleep
        tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;

        // Send a Stored event (should cause flush of the Removed event)
        tx.send(KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(0)),
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(1),
                    tokens_hash: LocalBlockHash(100),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        })
        .unwrap();

        // Give time for processing
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        // Should have 2 events: one Removed, one Stored (not batched together)
        assert_eq!(
            events.len(),
            2,
            "Switching from Removed to Stored should cause immediate flush, resulting in 2 separate events"
        );
    }

    /// Test that dp_rank change causes immediate flush
    #[tokio::test]
    async fn test_dp_rank_change_causes_flush() {
        let timeout_ms = Some(100); // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        // Send events with dp_rank=0
        for i in 0..3 {
            tx.send(KvCacheEvent {
                event_id: i as u64,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
                }),
                dp_rank: 0,
            })
            .unwrap();
            tokio::task::yield_now().await;
        }

        // Send events with dp_rank=1 (should cause flush of previous batch)
        for i in 3..6 {
            tx.send(KvCacheEvent {
                event_id: i as u64,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
                }),
                dp_rank: 1,
            })
            .unwrap();
            tokio::task::yield_now().await;
        }

        // Give time for processing
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        // Should have 2 events: one for dp_rank=0 batch, one for dp_rank=1 batch
        assert_eq!(
            events.len(),
            2,
            "dp_rank change should cause immediate flush, resulting in 2 separate events"
        );

        // Verify all 6 block hashes are accounted for
        let total_hashes: usize = events
            .iter()
            .map(|e| {
                if let KvCacheEventData::Removed(data) = &e.event.data {
                    data.block_hashes.len()
                } else {
                    0
                }
            })
            .sum();
        assert_eq!(
            total_hashes, 6,
            "All 6 block hashes should be accounted for"
        );

        // Verify dp_rank is correct for each batch
        assert_eq!(
            events[0].event.dp_rank, 0,
            "First batch should have dp_rank=0"
        );
        assert_eq!(
            events[1].event.dp_rank, 1,
            "Second batch should have dp_rank=1"
        );
    }

    /// Test that flushed events have correct metadata (event_id, dp_rank)
    /// This verifies that metadata is NOT overwritten before flush
    #[tokio::test]
    async fn test_flushed_events_have_correct_metadata() {
        let timeout_ms = Some(100); // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        // Send first batch: 3 events with dp_rank=0, event_ids 10-12
        for i in 0..3 {
            tx.send(KvCacheEvent {
                event_id: 10 + i as u64,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
                }),
                dp_rank: 0,
            })
            .unwrap();
            tokio::task::yield_now().await;
        }

        // Send second batch: 2 events with dp_rank=1, event_ids 20-21
        // This should flush the first batch with dp_rank=0
        for i in 0..2 {
            tx.send(KvCacheEvent {
                event_id: 20 + i as u64,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash((i + 3) as u64)],
                }),
                dp_rank: 1,
            })
            .unwrap();
            tokio::task::yield_now().await;
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        assert_eq!(
            events.len(),
            2,
            "Should have 2 events (one per dp_rank batch)"
        );

        // First event should have dp_rank=0 and monotonic batch event_id=1
        assert_eq!(
            events[0].event.dp_rank, 0,
            "First batch should have dp_rank=0"
        );
        assert_eq!(
            events[0].event.event_id, 1,
            "First batch should have monotonic event_id=1"
        );

        // Second event should have dp_rank=1 and monotonic batch event_id=2
        assert_eq!(
            events[1].event.dp_rank, 1,
            "Second batch should have dp_rank=1"
        );
        assert_eq!(
            events[1].event.event_id, 2,
            "Second batch should have monotonic event_id=2"
        );
    }

    /// Test that events after a long idle period flush immediately (stale timer).
    /// This gives low latency for sparse important events after idle periods.
    /// After the initial stale flush, subsequent rapid events batch normally.
    #[tokio::test]
    async fn test_first_event_after_idle_flushes_immediately_then_batches() {
        let timeout_ms = Some(50); // 50ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        // Wait longer than timeout to simulate idle period (timer becomes stale)
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Send 3 events rapidly - first should flush immediately (stale timer),
        // remaining 2 should batch together
        for i in 0..3 {
            tx.send(KvCacheEvent {
                event_id: i as u64,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
                }),
                dp_rank: 0,
            })
            .unwrap();
            tokio::task::yield_now().await;
        }

        // Wait for timeout to elapse so remaining batch flushes
        tokio::time::sleep(tokio::time::Duration::from_millis(60)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        // First event flushes immediately (stale timer), remaining 2 batch together
        assert_eq!(
            events.len(),
            2,
            "First event should flush immediately (stale), remaining 2 should batch"
        );

        // First event has 1 hash, second event (batch) has 2 hashes
        let first_len = if let KvCacheEventData::Removed(data) = &events[0].event.data {
            data.block_hashes.len()
        } else {
            0
        };
        let second_len = if let KvCacheEventData::Removed(data) = &events[1].event.data {
            data.block_hashes.len()
        } else {
            0
        };
        assert_eq!(first_len, 1, "First event should have 1 hash");
        assert_eq!(second_len, 2, "Second event (batched) should have 2 hashes");
    }

    /// Test that stored events with dp_rank change have correct metadata
    #[tokio::test]
    async fn test_stored_events_with_dp_rank_change_correct_metadata() {
        let timeout_ms = Some(100); // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        // Send first batch: 2 sequential stored events with dp_rank=0, event_ids 100-101
        tx.send(KvCacheEvent {
            event_id: 100,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(0)),
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(1),
                    tokens_hash: LocalBlockHash(100),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        })
        .unwrap();
        tokio::task::yield_now().await;

        tx.send(KvCacheEvent {
            event_id: 101,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(1)),
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(2),
                    tokens_hash: LocalBlockHash(200),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        })
        .unwrap();
        tokio::task::yield_now().await;

        // Send second batch: 1 event with dp_rank=1, event_id=200
        // This should flush the first batch with dp_rank=0, event_id=101
        tx.send(KvCacheEvent {
            event_id: 200,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(0)),
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(100),
                    tokens_hash: LocalBlockHash(1000),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 1,
        })
        .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        assert_eq!(
            events.len(),
            2,
            "Should have 2 events (one per dp_rank batch)"
        );

        // First batch: dp_rank=0, monotonic event_id=1
        assert_eq!(
            events[0].event.dp_rank, 0,
            "First batch should have dp_rank=0"
        );
        assert_eq!(
            events[0].event.event_id, 1,
            "First batch should have monotonic event_id=1"
        );

        // Second batch: dp_rank=1, monotonic event_id=2
        assert_eq!(
            events[1].event.dp_rank, 1,
            "Second batch should have dp_rank=1"
        );
        assert_eq!(
            events[1].event.event_id, 2,
            "Second batch should have monotonic event_id=2"
        );

        // Verify block counts
        if let KvCacheEventData::Stored(data) = &events[0].event.data {
            assert_eq!(data.blocks.len(), 2, "First batch should have 2 blocks");
        } else {
            panic!("Expected Stored event");
        }
        if let KvCacheEventData::Stored(data) = &events[1].event.data {
            assert_eq!(data.blocks.len(), 1, "Second batch should have 1 block");
        } else {
            panic!("Expected Stored event");
        }
    }

    /// Test that extending a batch does NOT change parent_hash
    /// First event with parent_hash=None should keep it None even if subsequent events have Some(X)
    #[tokio::test]
    async fn test_batch_parent_hash_preserved_when_extending() {
        let timeout_ms = Some(100); // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        // First event: parent_hash=None, block_hash=1
        tx.send(KvCacheEvent {
            event_id: 0,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None, // Root block with no parent
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(1),
                    tokens_hash: LocalBlockHash(100),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        })
        .unwrap();
        tokio::task::yield_now().await;

        // Second event: parent_hash=Some(1), block_hash=2 (sequential)
        tx.send(KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(1)), // Points to previous block
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(2),
                    tokens_hash: LocalBlockHash(200),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        })
        .unwrap();
        tokio::task::yield_now().await;

        // Third event: parent_hash=Some(2), block_hash=3 (sequential)
        tx.send(KvCacheEvent {
            event_id: 2,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(2)),
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(3),
                    tokens_hash: LocalBlockHash(300),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        })
        .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        assert_eq!(
            events.len(),
            1,
            "All 3 sequential events should batch into 1"
        );

        // The batch should have parent_hash=None (preserved from first event)
        if let KvCacheEventData::Stored(data) = &events[0].event.data {
            assert_eq!(data.blocks.len(), 3, "Batch should have 3 blocks");
            assert_eq!(
                data.parent_hash, None,
                "Batch parent_hash should remain None (from first event), NOT overwritten by subsequent events"
            );
        } else {
            panic!("Expected Stored event");
        }
    }
}
