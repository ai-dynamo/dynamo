// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use rmp_serde as rmps;
use serde::Deserialize;
use serde::Serialize;
use serde::de::{self, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use zeromq::{Socket, SocketRecv, SubSocket};

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
const BATCH_TIMEOUT_US: u64 = 10_000;

// -------------------------------------------------------------------------
// Batching State -----------------------------------------------------------
// -------------------------------------------------------------------------

/// Encapsulates the state for dynamic event batching.
/// Tracks pending events and metadata for batch flushing.
#[derive(Debug)]
struct BatchingState {
    /// Pending Removed batch (block hashes to be removed)
    pending_removed: Option<KvCacheRemoveData>,
    /// Pending Stored batch (blocks to be stored)
    pending_stored: Option<KvCacheStoreData>,
    /// Monotonic counter for published batch IDs.
    /// Increments by 1 per flush, decoupled from raw source event IDs.
    next_publish_id: u64,
    /// Last DP rank for batch
    last_dp_rank: u32,
    /// Time of last flush (for timeout tracking)
    last_flush_time: Instant,
}

impl BatchingState {
    /// Creates a new BatchingState with current timestamp
    fn new() -> Self {
        Self {
            pending_removed: None,
            pending_stored: None,
            next_publish_id: 1,
            last_dp_rank: 0,
            last_flush_time: Instant::now(),
        }
    }

    /// Returns true if there are pending batches
    fn has_pending(&self) -> bool {
        self.pending_removed.is_some() || self.pending_stored.is_some()
    }

    /// Resets flush time to now
    fn reset_flush_time(&mut self) {
        self.last_flush_time = Instant::now();
    }

    /// Gets remaining time until timeout
    fn remaining_timeout(&self, timeout_us: u64) -> Duration {
        let timeout = Duration::from_micros(timeout_us);
        let elapsed = self.last_flush_time.elapsed();
        if elapsed >= timeout {
            Duration::ZERO
        } else {
            timeout - elapsed
        }
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
        Self::new_with_local_indexer(component, kv_block_size, source_config, false, 0, None)
    }

    pub fn new_with_local_indexer(
        component: Component,
        kv_block_size: u32,
        source_config: Option<KvEventSourceConfig>,
        enable_local_indexer: bool,
        dp_rank: DpRank,
        batching_timeout_us: Option<u64>,
    ) -> Result<Self> {
        let cancellation_token = CancellationToken::new();
        let batching_timeout_us = batching_timeout_us.unwrap_or(BATCH_TIMEOUT_US);

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();

        // Infer worker_id from component's connection
        let worker_id = component.drt().connection_id();

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
                    event_publisher,
                    worker_id,
                    cancellation_token_clone,
                    rx,
                    local_indexer_clone,
                    batching_timeout_us,
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
                    nats_queue,
                    worker_id,
                    cancellation_token_clone,
                    rx,
                    local_indexer_clone,
                    batching_timeout_us,
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

#[async_trait]
trait EventSink: Send + Sync {
    async fn publish_event(&self, event: &RouterEvent) -> Result<()>;
}

#[async_trait]
impl EventSink for EventPublisher {
    async fn publish_event(&self, event: &RouterEvent) -> Result<()> {
        self.publish(event).await
    }
}

#[async_trait]
impl EventSink for NatsQueue {
    async fn publish_event(&self, event: &RouterEvent) -> Result<()> {
        NatsQueue::publish_event(self, KV_EVENT_SUBJECT, event).await
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
    if let Some(indexer) = local_indexer {
        if let Err(e) = indexer.apply_event_with_buffer(router_event.clone()).await {
            tracing::warn!(worker_id, error = %e, "Failed to apply event to local indexer");
        }
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
    }
}

/// Common event processor loop using BatchingState.
/// Accumulates Removed events and sequential Stored events, flushing on:
/// - Event type change
/// - Timeout (configurable, default 10ms)
/// - Shutdown
async fn run_event_processor_loop<P: EventSink + Send + Sync + 'static>(
    publisher: P,
    worker_id: u64,
    cancellation_token: CancellationToken,
    mut rx: mpsc::UnboundedReceiver<KvCacheEvent>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    timeout_us: u64,
) {
    let mut batching_state = BatchingState::new();
    // Track last raw input event_id for gap detection (dropped events before batching).
    // The raw event_id is a globally monotonic counter assigned by the ZMQ listener,
    // so any gap here means events were silently dropped (e.g. send error on the channel).
    let mut last_raw_input_id: Option<u64> = None;

    loop {
        let remaining = batching_state.remaining_timeout(timeout_us);

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
                    && raw_event_id > last_id + 1 {
                        tracing::warn!(
                            worker_id,
                            last_raw_input_id = last_id,
                            raw_event_id,
                            gap = raw_event_id - last_id - 1,
                            "Input event gap detected: raw events dropped before batching"
                        );
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
                        // Start a new batch or extend the current one.
                        // The timer resets only when the first event of a new batch arrives.
                        match &mut batching_state.pending_removed {
                            Some(pending) => pending.block_hashes.extend(data.block_hashes),
                            None => {
                                batching_state.pending_removed = Some(data);
                                batching_state.reset_flush_time();
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
                                batching_state.reset_flush_time();
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

                // Flush immediately if the timeout already elapsed (handles timeout_us=0).
                if batching_state.has_pending()
                    && batching_state.last_flush_time.elapsed() >= Duration::from_micros(timeout_us)
                {
                    batching_state.flush(&publisher, &local_indexer, worker_id).await;
                }
            }
            _ = tokio::time::sleep(remaining), if timeout_us > 0 && batching_state.has_pending() => {
                // Timeout elapsed - flush pending events
                batching_state.flush(&publisher, &local_indexer, worker_id).await;
            }
        }
    }
}

/// Batched event processor for ephemeral transports (NATS Core / ZMQ).
/// This is a thin wrapper around run_event_processor_loop.
async fn start_event_processor<P: EventSink + Send + Sync + 'static>(
    publisher: P,
    worker_id: u64,
    cancellation_token: CancellationToken,
    rx: mpsc::UnboundedReceiver<KvCacheEvent>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    batching_timeout_us: u64,
) {
    run_event_processor_loop(
        publisher,
        worker_id,
        cancellation_token,
        rx,
        local_indexer,
        batching_timeout_us,
    )
    .await
}

/// Batched event processor using JetStream (durable).
/// This is a thin wrapper around run_event_processor_loop.
async fn start_event_processor_jetstream(
    publisher: NatsQueue,
    worker_id: u64,
    cancellation_token: CancellationToken,
    rx: mpsc::UnboundedReceiver<KvCacheEvent>,
    local_indexer: Option<Arc<LocalKvIndexer>>,
    batching_timeout_us: u64,
) {
    run_event_processor_loop(
        publisher,
        worker_id,
        cancellation_token,
        rx,
        local_indexer,
        batching_timeout_us,
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
    #[allow(unused_assignments)]
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

                let dp_rank = batch.data_parallel_rank.unwrap_or(0) as u32;
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

/// Convert a raw event coming from the ZMQ channel into the internal
/// [`KvCacheEvent`] representation used by the router.
fn convert_event(
    raw: RawKvEvent,
    event_id: u64,
    kv_block_size: u32,
    dp_rank: u32,
    warning_count: &Arc<AtomicU32>,
) -> KvCacheEvent {
    match raw {
        RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            lora_name,
            block_mm_infos,
            medium: _,
        } => {
            // Reject self-referencing blocks: all block hashes (including parent) must be unique.
            {
                let mut seen = HashSet::with_capacity(block_hashes.len() + 1);
                if let Some(parent) = parent_block_hash {
                    seen.insert(parent.into_u64());
                }
                let has_duplicate = block_hashes.iter().any(|h| !seen.insert(h.into_u64()));
                if has_duplicate {
                    tracing::warn!(
                        event_id,
                        "Self-referencing block detected: duplicate hash in store event; dropping"
                    );
                    return KvCacheEvent {
                        event_id,
                        data: KvCacheEventData::Cleared,
                        dp_rank,
                    };
                }
            }

            let num_block_tokens = vec![block_size as u64; block_hashes.len()];
            let block_hashes_u64: Vec<u64> = block_hashes
                .into_iter()
                .map(BlockHashValue::into_u64)
                .collect();
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent_block_hash
                        .map(BlockHashValue::into_u64)
                        .map(ExternalSequenceBlockHash::from),
                    blocks: create_stored_blocks(
                        kv_block_size,
                        &token_ids,
                        &num_block_tokens,
                        &block_hashes_u64,
                        lora_name.as_deref(),
                        warning_count,
                        block_mm_infos.as_deref(),
                    ),
                }),
                dp_rank,
            }
        }
        RawKvEvent::BlockRemoved { block_hashes, .. } => {
            let hashes = block_hashes
                .into_iter()
                .map(BlockHashValue::into_u64)
                .map(ExternalSequenceBlockHash::from)
                .collect();
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes,
                }),
                dp_rank,
            }
        }
        RawKvEvent::AllBlocksCleared => KvCacheEvent {
            event_id,
            data: KvCacheEventData::Cleared,
            dp_rank,
        },
    }
}

pub fn create_stored_block_from_parts(
    kv_block_size: u32,
    block_hash: u64,
    token_ids: &[u32],
    lora_name: Option<&str>,
    mm_extra_info: Option<BlockExtraInfo>,
) -> KvCacheStoredBlockData {
    let block_mm_infos = mm_extra_info.as_ref().map(|info| vec![Some(info.clone())]);
    let tokens_hash = compute_block_hash_for_seq(
        token_ids,
        kv_block_size,
        block_mm_infos.as_deref(),
        lora_name,
    )[0];

    tracing::trace!(
        "Creating stored block: external_block_hash={}, tokens_hash={}, token_ids={:?}, kv_block_size={}, mm_extra_info={:?}",
        block_hash,
        tokens_hash.0,
        token_ids,
        kv_block_size,
        mm_extra_info
    );
    KvCacheStoredBlockData {
        block_hash: ExternalSequenceBlockHash::from(block_hash),
        tokens_hash,
        mm_extra_info,
    }
}

pub fn create_stored_blocks(
    kv_block_size: u32,
    token_ids: &[u32],
    num_block_tokens: &[u64],
    block_hashes: &[u64],
    lora_name: Option<&str>,
    warning_count: &Arc<AtomicU32>,
    block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
) -> Vec<KvCacheStoredBlockData> {
    let mut blocks: Vec<KvCacheStoredBlockData> = Vec::new();

    let mut token_offset: usize = 0;
    for (block_idx, (num_tokens_it, block_hash_it)) in
        num_block_tokens.iter().zip(block_hashes.iter()).enumerate()
    {
        if *num_tokens_it != kv_block_size as u64 {
            if warning_count.fetch_add(1, Ordering::Relaxed) < 3 {
                tracing::warn!(
                    "Block not published. Block size must be {} tokens to be published. Block size is: {}",
                    kv_block_size,
                    *num_tokens_it
                );
            }
            break;
        }

        let tokens = &token_ids[token_offset..(token_offset + *num_tokens_it as usize)];
        let mm_extra_info = block_mm_infos
            .and_then(|infos| infos.get(block_idx))
            .and_then(|opt| opt.clone());

        blocks.push(create_stored_block_from_parts(
            kv_block_size,
            *block_hash_it,
            tokens,
            lora_name,
            mm_extra_info,
        ));
        token_offset += *num_tokens_it as usize;
    }

    blocks
}

// -------------------------------------------------------------------------
// Types mirroring the Python msgspec-defined structures -------------------
// -------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct KvEventBatch {
    ts: f64,
    events: Vec<RawKvEvent>,
    #[serde(alias = "dp_rank")]
    data_parallel_rank: Option<i32>,
}

impl<'de> Deserialize<'de> for KvEventBatch {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize from array format: [timestamp, [events], data_parallel_rank]
        let arr: (f64, Vec<RawKvEvent>, Option<i32>) = Deserialize::deserialize(deserializer)?;
        Ok(KvEventBatch {
            ts: arr.0,
            events: arr.1,
            data_parallel_rank: arr.2,
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(untagged)]
enum BlockHashValue {
    Signed(i64),
    Unsigned(u64),
}

impl BlockHashValue {
    fn into_u64(self) -> u64 {
        match self {
            BlockHashValue::Signed(v) => v as u64,
            BlockHashValue::Unsigned(v) => v,
        }
    }
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")] // msgspec encodes variant tag as a string when `tag=True`
enum RawKvEvent {
    BlockStored {
        /// Block hashes may be emitted as either signed or unsigned 64-bit values.
        /// We normalize them to `u64` while deserializing to support both producers.
        block_hashes: Vec<BlockHashValue>,
        parent_block_hash: Option<BlockHashValue>,
        token_ids: Vec<u32>,
        block_size: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
        /// LoRA adapter name for adapter-aware block hashing
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lora_name: Option<String>,
        /// Multimodal extra info for each block (length should match block_hashes)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    },
    BlockRemoved {
        block_hashes: Vec<BlockHashValue>,
        #[serde(skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
    },
    AllBlocksCleared,
}

/// Parse MM hash from extra_keys string:
/// - Only accept canonical vLLM MM identifiers (64-char hex digest)
/// - Convert by taking the first 16 hex chars as u64
fn parse_mm_hash_from_extra_key(s: &str) -> Option<u64> {
    // extra_keys mixes MM identifiers with LoRA/cache_salt/prompt-embed metadata.
    // Only MM identifiers should be mapped into BlockExtraInfo.
    if s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit()) {
        return u64::from_str_radix(&s[..16], 16).ok();
    }
    None
}

/// Convert vLLM BlockStored extra_keys to block-level MM infos.
/// extra_keys is a list aligned with blocks:
/// - None => no MM content in that block
/// - ["hash1", "hash2", ...] => one or more MM objects in that block
fn extra_keys_to_block_mm_infos(
    extra_keys: Option<Vec<Option<Vec<String>>>>,
) -> Option<Vec<Option<BlockExtraInfo>>> {
    let extra_keys = extra_keys?;
    if extra_keys.is_empty() {
        return None;
    }

    let infos: Vec<Option<BlockExtraInfo>> = extra_keys
        .into_iter()
        .map(|block_keys| {
            let mm_objects: Vec<BlockMmObjectInfo> = block_keys
                .unwrap_or_default()
                .iter()
                .filter_map(|key| parse_mm_hash_from_extra_key(key))
                .map(|mm_hash| BlockMmObjectInfo {
                    mm_hash,
                    offsets: vec![], // extra_keys does not carry offsets today
                })
                .collect();

            if mm_objects.is_empty() {
                None
            } else {
                Some(BlockExtraInfo { mm_objects })
            }
        })
        .collect();

    if infos.iter().all(|i| i.is_none()) {
        return None;
    }

    Some(infos)
}

/// Our producers use msgspec with `tag=True` and `array_like=True`, which
/// encodes each event as either a tagged map or a tagged tuple. To be tolerant of
/// additional fields that may be appended in the future, we implement a custom
/// deserializer that ignores unknown keys and any extra positional elements.
///
/// This keeps us compatible with older payloads while safely
/// accepting newer ones that include extra metadata.
impl<'de> Deserialize<'de> for RawKvEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(RawKvEventVisitor)
    }
}

struct RawKvEventVisitor;

impl<'de> Visitor<'de> for RawKvEventVisitor {
    type Value = RawKvEvent;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a kv event encoded as a tagged map or sequence")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut event_type: Option<String> = None;
        let mut block_hashes: Option<Vec<BlockHashValue>> = None;
        let mut parent_block_hash: Option<Option<BlockHashValue>> = None;
        let mut token_ids: Option<Vec<u32>> = None;
        let mut block_size: Option<usize> = None;
        let mut medium: Option<Option<String>> = None;
        let mut lora_name: Option<Option<String>> = None;
        let mut extra_keys: Option<Option<Vec<Option<Vec<String>>>>> = None;
        let mut block_mm_infos: Option<Option<Vec<Option<BlockExtraInfo>>>> = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "type" => {
                    event_type = Some(map.next_value()?);
                }
                "block_hashes" => {
                    block_hashes = Some(map.next_value()?);
                }
                "parent_block_hash" => {
                    parent_block_hash = Some(map.next_value()?);
                }
                "token_ids" => {
                    token_ids = Some(map.next_value()?);
                }
                "block_size" => {
                    block_size = Some(map.next_value()?);
                }
                "medium" => {
                    medium = Some(map.next_value()?);
                }
                "lora_name" => {
                    lora_name = Some(map.next_value()?);
                }
                "extra_keys" => {
                    extra_keys = Some(map.next_value()?);
                }
                "block_mm_infos" => {
                    block_mm_infos = Some(map.next_value()?);
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }

        match event_type.as_deref() {
            Some("BlockStored") => {
                let block_hashes =
                    block_hashes.ok_or_else(|| de::Error::missing_field("block_hashes"))?;
                let token_ids = token_ids.ok_or_else(|| de::Error::missing_field("token_ids"))?;
                let block_size =
                    block_size.ok_or_else(|| de::Error::missing_field("block_size"))?;
                let block_mm_infos = block_mm_infos
                    .unwrap_or(None)
                    .or_else(|| extra_keys_to_block_mm_infos(extra_keys.unwrap_or(None)));
                Ok(RawKvEvent::BlockStored {
                    block_hashes,
                    parent_block_hash: parent_block_hash.unwrap_or(None),
                    token_ids,
                    block_size,
                    medium: medium.unwrap_or(None),
                    lora_name: lora_name.unwrap_or(None),
                    block_mm_infos,
                })
            }
            Some("BlockRemoved") => {
                let block_hashes =
                    block_hashes.ok_or_else(|| de::Error::missing_field("block_hashes"))?;
                Ok(RawKvEvent::BlockRemoved {
                    block_hashes,
                    medium: medium.unwrap_or(None),
                })
            }
            Some("AllBlocksCleared") => Ok(RawKvEvent::AllBlocksCleared),
            Some(other) => Err(de::Error::unknown_variant(
                other,
                &["BlockStored", "BlockRemoved", "AllBlocksCleared"],
            )),
            None => Err(de::Error::missing_field("type")),
        }
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let tag: Option<String> = seq.next_element()?;
        let Some(tag) = tag else {
            return Err(de::Error::invalid_length(
                0,
                &"sequence must start with event tag",
            ));
        };

        match tag.as_str() {
            "BlockStored" => {
                let block_hashes: Vec<BlockHashValue> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &"missing block_hashes"))?;
                let parent_block_hash: Option<BlockHashValue> = seq.next_element()?.unwrap_or(None);
                let token_ids: Vec<u32> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(3, &"missing token_ids"))?;
                let block_size: usize = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(4, &"missing block_size"))?;
                // Position 5 was lora_id in older formats; consume and discard for compat
                let _lora_id: Option<u64> = seq.next_element()?.unwrap_or(None);
                let medium: Option<String> = seq.next_element()?.unwrap_or(None);
                let lora_name: Option<String> = seq.next_element()?.unwrap_or(None);
                let extra_keys: Option<Vec<Option<Vec<String>>>> =
                    seq.next_element()?.unwrap_or(None);
                let block_mm_infos: Option<Vec<Option<BlockExtraInfo>>> =
                    seq.next_element()?.unwrap_or(None);

                while seq.next_element::<IgnoredAny>()?.is_some() {}

                let block_mm_infos =
                    block_mm_infos.or_else(|| extra_keys_to_block_mm_infos(extra_keys));

                Ok(RawKvEvent::BlockStored {
                    block_hashes,
                    parent_block_hash,
                    token_ids,
                    block_size,
                    medium,
                    lora_name,
                    block_mm_infos,
                })
            }
            "BlockRemoved" => {
                let block_hashes: Vec<BlockHashValue> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &"missing block_hashes"))?;
                let medium: Option<String> = seq.next_element()?.unwrap_or(None);

                while seq.next_element::<IgnoredAny>()?.is_some() {}

                Ok(RawKvEvent::BlockRemoved {
                    block_hashes,
                    medium,
                })
            }
            "AllBlocksCleared" => {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(RawKvEvent::AllBlocksCleared)
            }
            other => Err(de::Error::unknown_variant(
                other,
                &["BlockStored", "BlockRemoved", "AllBlocksCleared"],
            )),
        }
    }
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

    #[async_trait::async_trait]
    impl EventSink for MockComponent {
        async fn publish_event(&self, event: &RouterEvent) -> anyhow::Result<()> {
            let bytes = rmp_serde::to_vec(event).unwrap();
            self.published
                .lock()
                .unwrap()
                .push((KV_EVENT_SUBJECT.to_string(), bytes));
            Ok(())
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
            BATCH_TIMEOUT_US,
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
            BATCH_TIMEOUT_US,
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
            BATCH_TIMEOUT_US,
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
            BATCH_TIMEOUT_US,
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
            BATCH_TIMEOUT_US,
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
            BATCH_TIMEOUT_US,
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
            crate::kv_router::indexer::WorkerKvQueryResponse::TreeDump(e) => e,
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
    #[allow(clippy::assertions_on_constants)]
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
        state.reset_flush_time();

        // Test that remaining returns positive initially (using 10ms = 10_000us)
        let remaining_before = state.remaining_timeout(10_000);
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
    fn test_batching_state_reset_flush_time() {
        let mut state = BatchingState::new();

        let initial_time = state.last_flush_time;

        state.reset_flush_time();

        assert!(
            state.last_flush_time >= initial_time,
            "reset_flush_time should update the time"
        );
    }

    #[test]
    fn test_batching_state_remaining_timeout() {
        let mut state = BatchingState::new();

        // Reset flush time to now so we can test timeout behavior
        state.reset_flush_time();

        // Test that remaining returns positive initially
        let remaining = state.remaining_timeout(10_000); // 10ms
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

    #[async_trait]
    impl EventSink for MockPublisher {
        async fn publish_event(&self, event: &RouterEvent) -> Result<()> {
            self.events.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    /// Test that pushing N removed events results in batched output
    /// Uses a 10ms timeout to ensure events are batched (events sent rapidly)
    #[tokio::test]
    async fn test_run_event_processor_loop_batches_removed_events_20() {
        test_removed_events_batching(20, 10_000).await; // 20 events, 20ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_removed_events_10() {
        test_removed_events_batching(10, 10_000).await; // 10 events, 10ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_removed_events_5() {
        test_removed_events_batching(5, 10_000).await; // 5 events, 10ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_removed_events_3() {
        test_removed_events_batching(3, 10_000).await; // 3 events, 10ms timeout
    }

    /// Helper function to test removed events batching with configurable count and timeout
    async fn test_removed_events_batching(event_count: usize, timeout_us: u64) {
        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(publisher_clone, 1, cancellation_token, rx, None, timeout_us)
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
        tokio::time::sleep(tokio::time::Duration::from_micros(timeout_us + 1000)).await;

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
            "With long timeout ({}us), all {} events should batch into at most 2 output events (got {})",
            timeout_us,
            event_count,
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
        test_stored_events_batching(20, 100_000).await; // 20 events, 100ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_stored_events_10() {
        test_stored_events_batching(10, 100_000).await; // 10 events, 100ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_stored_events_5() {
        test_stored_events_batching(5, 100_000).await; // 5 events, 100ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_stored_events_3() {
        test_stored_events_batching(3, 100_000).await; // 3 events, 100ms timeout
    }

    /// Helper function to test stored events batching with configurable count and timeout
    async fn test_stored_events_batching(event_count: usize, timeout_us: u64) {
        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(publisher_clone, 1, cancellation_token, rx, None, timeout_us)
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
            "With long timeout ({}us) and sequential parent hashes, all {} events should batch into at most 2 output events (got {})",
            timeout_us,
            event_count,
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
        let timeout_us = 100_000; // 100ms in microseconds

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(publisher_clone, 1, cancellation_token, rx, None, timeout_us)
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
        test_no_batching_with_slow_input(0).await; // 0ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_no_batching_with_slow_input_0_1ms() {
        test_no_batching_with_slow_input(100).await; // 0.1ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_no_batching_with_slow_input_0_2ms() {
        test_no_batching_with_slow_input(200).await; // 0.2ms timeout
    }

    /// Helper function to test no batching with slow input
    async fn test_no_batching_with_slow_input(timeout_us: u64) {
        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(publisher_clone, 1, cancellation_token, rx, None, timeout_us)
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
            "With slow input (2ms delay) and timeout={}us, should have at least 3 separate events (got {})",
            timeout_us,
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
        let timeout_us = 100_000; // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(publisher_clone, 1, cancellation_token, rx, None, timeout_us)
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
        let timeout_us = 100_000; // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(publisher_clone, 1, cancellation_token, rx, None, timeout_us)
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
        let timeout_us = 100_000; // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(publisher_clone, 1, cancellation_token, rx, None, timeout_us)
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

    /// Test that first event after idle period doesn't flush immediately
    /// This verifies the reset_flush_time() fix for new batches
    #[tokio::test]
    async fn test_first_event_after_idle_no_immediate_flush() {
        let timeout_us = 50_000; // 50ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(publisher_clone, 1, cancellation_token, rx, None, timeout_us)
                .await
        });

        // Wait longer than timeout to simulate idle period
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Send 3 events rapidly - they should batch together
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

        // Wait for timeout to elapse so batch flushes
        tokio::time::sleep(tokio::time::Duration::from_millis(60)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        // All 3 events should be batched into 1 output event
        assert_eq!(
            events.len(),
            1,
            "All 3 events should batch into 1 output event (not flush immediately due to stale timer)"
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
            total_hashes, 3,
            "All 3 block hashes should be accounted for"
        );
    }

    /// Test that stored events with dp_rank change have correct metadata
    #[tokio::test]
    async fn test_stored_events_with_dp_rank_change_correct_metadata() {
        let timeout_us = 100_000; // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(publisher_clone, 1, cancellation_token, rx, None, timeout_us)
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
        let timeout_us = 100_000; // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(publisher_clone, 1, cancellation_token, rx, None, timeout_us)
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
