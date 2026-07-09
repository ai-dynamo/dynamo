// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use uuid::Uuid;

use dynamo_kv_router::indexer::{KvIndexerMetrics, LocalKvIndexer};
use dynamo_kv_router::protocols::*;
pub use dynamo_kv_router::zmq_wire::create_stored_blocks;
#[cfg(test)]
use dynamo_kv_router::zmq_wire::*;
use dynamo_runtime::config::environment_names::nats as env_nats;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::{
    component::Component,
    transports::nats::{NatsQueue, Slug},
};

use crate::kv_router::{
    KV_EVENT_SUBJECT, WORKER_KV_INDEXER_BUFFER_SIZE, indexer::start_worker_kv_query_endpoint,
    metrics::KvPublisherMetrics,
};

mod batching;
mod configuration;
mod dedup;
mod event_processor;
mod multimodal_embedding_cache;
mod sinks;
#[cfg(test)]
mod tests;
mod worker;
mod worker_metrics;
mod zmq_listener;

#[cfg(test)]
use batching::BatchingState;
pub use configuration::{DEFAULT_BATCHING_TIMEOUT_MS, MAX_VALKEY_WORKER_RANKS};
use configuration::{
    DEFAULT_VALKEY_BATCHING_TIMEOUT_MS, DEFAULT_VALKEY_EVENT_INPUT_BUFFER_SIZE,
    MAX_BATCHING_TIMEOUT_MS, MIN_VALKEY_EVENT_INPUT_BUFFER_SIZE, VALKEY_BATCHING_TIMEOUT_ENV,
    VALKEY_EVENT_INPUT_BUFFER_SIZE_ENV, valkey_batching_timeout_ms, valkey_event_input_buffer_size,
};
#[cfg(test)]
use dedup::EventDedupFilter;
#[cfg(test)]
use event_processor::run_event_processor_loop;
use event_processor::{start_event_processor, start_event_processor_jetstream};
pub use multimodal_embedding_cache::{
    MultimodalEmbeddingCacheEvent, MultimodalEmbeddingCachePublisher,
    MultimodalEmbeddingCacheUpdate,
};
use sinks::{
    EventPlanePublisher, GcStepOutcome, QueuedValkeyPublisher, ValkeyEventPublisher,
    ValkeyPublisherIntegrity,
};
#[cfg(test)]
use worker::{
    MAX_VALKEY_GC_INSPECTION_BUDGET, VALKEY_WORKER_UNREGISTER_TIMEOUT,
    parse_valkey_gc_inspection_budget, parse_valkey_gc_interval_ms,
    parse_valkey_required_replica_acks, parse_valkey_worker_events_enabled,
    spawn_worker_lifecycle_gc, valkey_gc_initial_delay_ms, valkey_index_namespace,
};
pub use worker::{
    ValkeyWorkerConfig, ValkeyWorkerEventLease, ValkeyWorkerRegistration,
    valkey_worker_config_from_env, valkey_worker_events_enabled,
};
pub use worker_metrics::WorkerMetricsPublisher;
use zmq_listener::start_zmq_listener;

const DEFAULT_MAX_BATCH_BLOCKS: usize = 128;
const KV_EVENT_PUBLISHER_DRAIN_TIMEOUT: Duration = Duration::from_secs(3);
const KV_EVENT_PUBLISHER_FORCED_WAIT_TIMEOUT: Duration = Duration::from_millis(500);

struct PublisherInput {
    event: PlacementEvent,
}

#[derive(Clone)]
enum PlacementEventSender {
    Unbounded(mpsc::UnboundedSender<PlacementEvent>),
    DirectValkey {
        tx: mpsc::Sender<PublisherInput>,
        integrity: ValkeyPublisherIntegrity,
    },
}

enum PlacementEventReceiver {
    Unbounded(mpsc::UnboundedReceiver<PlacementEvent>),
    DirectValkey(mpsc::Receiver<PublisherInput>),
}

#[derive(Debug)]
enum PlacementEventSendError {
    Closed(PlacementEvent),
    Full(PlacementEvent),
    IntegrityFenced(PlacementEvent),
}

impl PlacementEventSendError {
    fn into_event(self) -> PlacementEvent {
        match self {
            Self::Closed(event) | Self::Full(event) | Self::IntegrityFenced(event) => event,
        }
    }

    fn is_closed(&self) -> bool {
        matches!(self, Self::Closed(_))
    }
}

impl From<mpsc::UnboundedSender<PlacementEvent>> for PlacementEventSender {
    fn from(tx: mpsc::UnboundedSender<PlacementEvent>) -> Self {
        Self::Unbounded(tx)
    }
}

impl From<mpsc::UnboundedReceiver<PlacementEvent>> for PlacementEventReceiver {
    fn from(rx: mpsc::UnboundedReceiver<PlacementEvent>) -> Self {
        Self::Unbounded(rx)
    }
}

impl PlacementEventSender {
    fn direct_valkey(
        tx: mpsc::Sender<PublisherInput>,
        integrity: ValkeyPublisherIntegrity,
    ) -> Self {
        Self::DirectValkey { tx, integrity }
    }

    fn integrity(&self) -> Option<&ValkeyPublisherIntegrity> {
        match self {
            Self::Unbounded(_) => None,
            Self::DirectValkey { integrity, .. } => Some(integrity),
        }
    }

    fn send(&self, event: PlacementEvent) -> std::result::Result<(), PlacementEventSendError> {
        match self {
            Self::Unbounded(tx) => tx
                .send(event)
                .map_err(|error| PlacementEventSendError::Closed(error.0)),
            Self::DirectValkey { tx, integrity } => {
                if !integrity.is_healthy() {
                    if let Some(metrics) = crate::kv_router::metrics::kv_publisher_metrics() {
                        metrics.increment_input_dropped_event("integrity_fenced");
                    }
                    return Err(PlacementEventSendError::IntegrityFenced(event));
                }
                let input = PublisherInput { event };
                match tx.try_send(input) {
                    Ok(()) => Ok(()),
                    Err(mpsc::error::TrySendError::Full(input)) => {
                        if let Some(metrics) = crate::kv_router::metrics::kv_publisher_metrics() {
                            metrics.increment_input_dropped_event("queue_full");
                        }
                        integrity.mark_fault("input_queue_overflow");
                        Err(PlacementEventSendError::Full(input.event))
                    }
                    Err(mpsc::error::TrySendError::Closed(input)) => {
                        if let Some(metrics) = crate::kv_router::metrics::kv_publisher_metrics() {
                            metrics.increment_input_dropped_event("channel_closed");
                        }
                        Err(PlacementEventSendError::Closed(input.event))
                    }
                }
            }
        }
    }
}

impl PlacementEventReceiver {
    async fn recv(&mut self) -> Option<PublisherInput> {
        match self {
            Self::Unbounded(rx) => rx.recv().await.map(|event| PublisherInput { event }),
            Self::DirectValkey(rx) => rx.recv().await,
        }
    }

    fn close(&mut self) {
        match self {
            Self::Unbounded(rx) => rx.close(),
            Self::DirectValkey(rx) => rx.close(),
        }
    }

    fn try_recv(&mut self) -> std::result::Result<PublisherInput, mpsc::error::TryRecvError> {
        match self {
            Self::Unbounded(rx) => rx.try_recv().map(|event| PublisherInput { event }),
            Self::DirectValkey(rx) => rx.try_recv(),
        }
    }
}

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

/// Configure the source of KV events.
/// Currently, only ZMQ is supported.
pub enum KvEventSourceConfig {
    Zmq {
        endpoint: String,
        topic: String,
        /// Model image-placeholder token id, used by the normalizer to rewrite
        /// vLLM BlockStored events to the canonical pad_value scheme. `None`
        /// for text-only / non-MM deployments (normalization is a no-op).
        image_token_id: Option<u32>,
    },
}

enum KvEventSource {
    Zmq {
        zmq_handle: tokio::task::JoinHandle<()>,
    },
}

impl KvEventSource {
    #[expect(clippy::too_many_arguments)]
    fn start(
        component: Component,
        worker_id: WorkerId,
        kv_block_size: u32,
        source_config: KvEventSourceConfig,
        cancellation_token: CancellationToken,
        task_tracker: &TaskTracker,
        tx: PlacementEventSender,
        next_event_id: Arc<AtomicU64>,
    ) -> Result<Self> {
        match source_config {
            KvEventSourceConfig::Zmq {
                endpoint,
                topic,
                image_token_id,
            } => {
                let zmq_handle =
                    component
                        .drt()
                        .runtime()
                        .secondary()
                        .spawn(task_tracker.track_future(start_zmq_listener(
                            endpoint,
                            topic,
                            worker_id,
                            tx,
                            cancellation_token.clone(),
                            kv_block_size,
                            next_event_id,
                            image_token_id,
                        )));

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
    /// The ID of the local worker emitting placement events.
    worker_id: WorkerId,
    /// The channel to send events to.
    tx: PlacementEventSender,
    /// Internal monotonic event ID counter. Shared with the ZMQ listener if present.
    next_event_id: Arc<AtomicU64>,
    /// Tracks every task which can write or forward an accepted KV event.
    task_tracker: TaskTracker,
    /// Shared registration/indexer cancellation used only after graceful drain
    /// exceeds its deadline.
    operation_cancel: Option<CancellationToken>,
}

/// Clonable lifecycle handle used by worker shutdown paths that do not own the
/// publisher itself. A [`KvEventPublisherShutdownOutcome::Drained`] result
/// guarantees no owned Valkey write remains in flight.
#[derive(Clone)]
pub struct KvEventPublisherShutdown {
    cancellation_token: CancellationToken,
    task_tracker: TaskTracker,
    operation_cancel: Option<CancellationToken>,
}

/// Result of the bounded two-stage publisher shutdown.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvEventPublisherShutdownOutcome {
    /// Every accepted event drained without cancelling indexer operations.
    Drained,
    /// Indexer operations were cancelled after the graceful drain deadline,
    /// and every tracked task then stopped within the forced-wait deadline.
    Forced,
    /// At least one tracked task remained after both deadlines.
    TimedOut,
}

impl KvEventPublisherShutdown {
    fn begin_shutdown(&self) {
        self.cancellation_token.cancel();
        self.task_tracker.close();
    }

    /// Signal shutdown without waiting. Legacy synchronous integrations may
    /// use this, while lifecycle owners should await [`Self::shutdown_and_wait`].
    pub fn request_shutdown(&self) {
        self.begin_shutdown();
    }

    fn cancel_operations(&self) {
        if let Some(cancellation) = &self.operation_cancel {
            cancellation.cancel();
        }
    }

    pub async fn shutdown_and_wait(&self) -> KvEventPublisherShutdownOutcome {
        shutdown_publishers_and_wait(std::slice::from_ref(self)).await
    }
}

/// Stop all publishers together so the graceful and forced deadlines apply to
/// the worker as a whole rather than once per data-parallel rank.
pub async fn shutdown_publishers_and_wait(
    publishers: &[KvEventPublisherShutdown],
) -> KvEventPublisherShutdownOutcome {
    shutdown_publishers_and_wait_with_timeouts(
        publishers,
        KV_EVENT_PUBLISHER_DRAIN_TIMEOUT,
        KV_EVENT_PUBLISHER_FORCED_WAIT_TIMEOUT,
    )
    .await
}

async fn shutdown_publishers_and_wait_with_timeouts(
    publishers: &[KvEventPublisherShutdown],
    drain_timeout: Duration,
    forced_wait_timeout: Duration,
) -> KvEventPublisherShutdownOutcome {
    for publisher in publishers {
        publisher.begin_shutdown();
    }
    let wait_for_publishers = || async {
        futures::future::join_all(
            publishers
                .iter()
                .map(|publisher| publisher.task_tracker.wait()),
        )
        .await;
    };
    if tokio::time::timeout(drain_timeout, wait_for_publishers())
        .await
        .is_ok()
    {
        return KvEventPublisherShutdownOutcome::Drained;
    }

    let remaining_tasks = publishers
        .iter()
        .map(|publisher| publisher.task_tracker.len())
        .sum::<usize>();
    tracing::warn!(
        remaining_tasks,
        drain_timeout_ms = drain_timeout.as_millis(),
        "KV event publisher drain timed out; cancelling Valkey indexer operations"
    );
    for publisher in publishers {
        publisher.cancel_operations();
    }
    if tokio::time::timeout(forced_wait_timeout, wait_for_publishers())
        .await
        .is_ok()
    {
        KvEventPublisherShutdownOutcome::Forced
    } else {
        let remaining_tasks = publishers
            .iter()
            .map(|publisher| publisher.task_tracker.len())
            .sum::<usize>();
        tracing::error!(
            remaining_tasks,
            forced_wait_timeout_ms = forced_wait_timeout.as_millis(),
            "KV event publisher tasks did not stop after operation cancellation"
        );
        KvEventPublisherShutdownOutcome::TimedOut
    }
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
        Self::new_with_local_indexer_and_worker_id(
            component,
            None,
            kv_block_size,
            source_config,
            enable_local_indexer,
            dp_rank,
            batching_timeout_ms,
        )
    }

    pub fn new_with_local_indexer_and_worker_id(
        component: Component,
        worker_id: Option<WorkerId>,
        kv_block_size: u32,
        source_config: Option<KvEventSourceConfig>,
        enable_local_indexer: bool,
        dp_rank: DpRank,
        batching_timeout_ms: Option<u64>,
    ) -> Result<Self> {
        Self::new_with_local_indexer_and_worker_id_and_valkey_lease(
            component,
            worker_id,
            kv_block_size,
            source_config,
            enable_local_indexer,
            dp_rank,
            batching_timeout_ms,
            None,
        )
    }

    #[expect(clippy::too_many_arguments)]
    pub fn new_with_local_indexer_and_worker_id_and_valkey_lease(
        component: Component,
        worker_id: Option<WorkerId>,
        kv_block_size: u32,
        source_config: Option<KvEventSourceConfig>,
        enable_local_indexer: bool,
        dp_rank: DpRank,
        batching_timeout_ms: Option<u64>,
        valkey_event_lease: Option<ValkeyWorkerEventLease>,
    ) -> Result<Self> {
        let cancellation_token = CancellationToken::new();
        let task_tracker = TaskTracker::new();
        let batching_timeout_ms = batching_timeout_ms
            .filter(|&ms| {
                if ms > MAX_BATCHING_TIMEOUT_MS {
                    tracing::warn!(
                        requested_ms = ms,
                        max_ms = MAX_BATCHING_TIMEOUT_MS,
                        "batching_timeout_ms too high, capping to 15s"
                    );
                }
                ms > 0
            })
            .map(|ms| ms.min(MAX_BATCHING_TIMEOUT_MS));

        let direct_valkey = valkey_worker_events_enabled()?;
        if direct_valkey && valkey_event_lease.is_none() {
            anyhow::bail!(
                "direct Valkey KV events require an awaited ValkeyWorkerRegistration lease"
            );
        }
        if !direct_valkey && valkey_event_lease.is_some() {
            anyhow::bail!(
                "a Valkey worker event lease was provided while direct Valkey KV events are disabled"
            );
        }

        let configured_valkey_event_input_buffer_size =
            std::env::var(VALKEY_EVENT_INPUT_BUFFER_SIZE_ENV).ok();
        if direct_valkey
            && configured_valkey_event_input_buffer_size
                .as_deref()
                .is_some_and(|value| {
                    value
                        .trim()
                        .parse::<usize>()
                        .map_or(true, |size| size < MIN_VALKEY_EVENT_INPUT_BUFFER_SIZE)
                })
        {
            tracing::warn!(
                env = VALKEY_EVENT_INPUT_BUFFER_SIZE_ENV,
                value = ?configured_valkey_event_input_buffer_size,
                default_events = DEFAULT_VALKEY_EVENT_INPUT_BUFFER_SIZE,
                min_events = MIN_VALKEY_EVENT_INPUT_BUFFER_SIZE,
                "Invalid Valkey event input buffer size; using the default"
            );
        }
        let valkey_event_input_buffer_size =
            valkey_event_input_buffer_size(configured_valkey_event_input_buffer_size.as_deref());

        let (tx, rx) = if let Some(lease) = valkey_event_lease.as_ref() {
            let (tx, rx) = mpsc::channel::<PublisherInput>(valkey_event_input_buffer_size);
            (
                PlacementEventSender::direct_valkey(tx, lease.integrity.clone()),
                PlacementEventReceiver::DirectValkey(rx),
            )
        } else {
            let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
            (
                PlacementEventSender::Unbounded(tx),
                PlacementEventReceiver::Unbounded(rx),
            )
        };
        let worker_id = worker_id.unwrap_or_else(|| component.drt().connection_id());

        let _ = KvPublisherMetrics::from_component(&component);

        let component_name = component.name();
        tracing::info!(
            event_input_buffer_size = valkey_event_input_buffer_size,
            "Initializing KvEventPublisher for worker {worker_id} in component {component_name}"
        );

        if enable_local_indexer {
            tracing::info!(
                "LocalKvIndexer enabled for worker {worker_id} in component {component_name}"
            );
        }

        let next_event_id = Arc::new(AtomicU64::new(0));

        let mut source = None;
        if let Some(config) = source_config {
            source = Some(KvEventSource::start(
                component.clone(),
                worker_id,
                kv_block_size,
                config,
                cancellation_token.clone(),
                &task_tracker,
                tx.clone(),
                next_event_id.clone(),
            )?);
        }

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
        let mut publisher_operation_cancel = None;

        if let Some(valkey_event_lease) = valkey_event_lease {
            publisher_operation_cancel = Some(valkey_event_lease.operation_cancel.clone());
            let configured_valkey_batching_timeout =
                std::env::var(VALKEY_BATCHING_TIMEOUT_ENV).ok();
            if batching_timeout_ms.is_none()
                && configured_valkey_batching_timeout
                    .as_deref()
                    .is_some_and(|value| value.trim().parse::<u64>().is_err())
            {
                tracing::warn!(
                    env = VALKEY_BATCHING_TIMEOUT_ENV,
                    value = ?configured_valkey_batching_timeout,
                    default_ms = DEFAULT_VALKEY_BATCHING_TIMEOUT_MS,
                    "Invalid Valkey event batching timeout; using the default"
                );
            }
            let direct_batching_timeout_ms = valkey_batching_timeout_ms(
                batching_timeout_ms,
                configured_valkey_batching_timeout.as_deref(),
            );
            let valkey_publisher = ValkeyEventPublisher::new(valkey_event_lease.integrity.clone());
            let recovery = valkey_publisher.clone();
            tracing::info!(
                batching_timeout_ms = ?direct_batching_timeout_ms,
                "Publishing GPU KV events directly to the shared Valkey index"
            );
            let publisher = QueuedValkeyPublisher::new(valkey_publisher);

            // Start the authoritative writer independently of an optional
            // compatibility relay. A relay reconnects in its own task, so an
            // unavailable NATS/ZMQ broker cannot strand direct GPU KV events.
            component
                .drt()
                .runtime()
                .secondary()
                .spawn(task_tracker.track_future(async move {
                    event_processor::start_direct_valkey_event_processor(
                        publisher,
                        recovery,
                        worker_id,
                        cancellation_token_clone,
                        rx,
                        local_indexer_clone,
                        direct_batching_timeout_ms,
                    )
                    .await
                }));
        } else if enable_local_indexer {
            tracing::info!("Using event plane for KV event publishing (local_indexer mode)");
            let component_clone = component.clone();
            let publisher_task = async move {
                let event_publisher_result = tokio::select! {
                    _ = cancellation_token_clone.cancelled() => return,
                    result = dynamo_runtime::transports::event_plane::EventPublisher::for_component(
                        &component_clone,
                        KV_EVENT_SUBJECT,
                    ) => result,
                };
                let event_publisher = match event_publisher_result {
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
            };
            component
                .drt()
                .runtime()
                .secondary()
                .spawn(task_tracker.track_future(publisher_task));
        } else {
            let stream_name = create_kv_stream_name(&component, KV_EVENT_SUBJECT);
            let nats_server = std::env::var(env_nats::NATS_SERVER)
                .unwrap_or_else(|_| "nats://localhost:4222".to_string());
            let mut nats_queue = NatsQueue::new_without_consumer(
                stream_name,
                nats_server,
                std::time::Duration::from_secs(60),
            );

            component
                .drt()
                .runtime()
                .secondary()
                .spawn(task_tracker.track_future(async move {
                    let connect_result = tokio::select! {
                        _ = cancellation_token_clone.cancelled() => return,
                        result = nats_queue.connect() => result,
                    };
                    if let Err(e) = connect_result {
                        tracing::error!("Failed to connect NatsQueue: {e}");
                        return;
                    }
                    start_event_processor_jetstream(
                        nats_queue,
                        worker_id,
                        cancellation_token_clone,
                        rx,
                        local_indexer_clone,
                        batching_timeout_ms,
                    )
                    .await
                }));
        }

        Ok(Self {
            kv_block_size,
            source,
            cancellation_token,
            worker_id,
            tx,
            next_event_id,
            task_tracker,
            operation_cancel: publisher_operation_cancel,
        })
    }

    pub fn publish(&self, event: KvCacheEvent) -> Result<(), mpsc::error::SendError<KvCacheEvent>> {
        let placement_event = PlacementEvent::local_gpu(self.worker_id, event);
        match self.tx.send(placement_event) {
            Ok(()) => Ok(()),
            Err(error) => Err(mpsc::error::SendError(error.into_event().event)),
        }
    }

    pub fn publish_with_storage_tier(
        &self,
        event: KvCacheEvent,
        storage_tier: StorageTier,
    ) -> Result<(), mpsc::error::SendError<KvCacheEvent>> {
        let placement_event = PlacementEvent::new(
            Placement::local_worker(self.worker_id, event.dp_rank, storage_tier),
            event,
        );
        match self.tx.send(placement_event) {
            Ok(()) => Ok(()),
            Err(error) => Err(mpsc::error::SendError(error.into_event().event)),
        }
    }

    pub fn next_event_id(&self) -> u64 {
        self.next_event_id.fetch_add(1, Ordering::SeqCst)
    }

    pub fn kv_block_size(&self) -> u32 {
        self.kv_block_size
    }

    pub fn shutdown_handle(&self) -> KvEventPublisherShutdown {
        KvEventPublisherShutdown {
            cancellation_token: self.cancellation_token.clone(),
            task_tracker: self.task_tracker.clone(),
            operation_cancel: self.operation_cancel.clone(),
        }
    }

    pub fn shutdown(&mut self) {
        if !self.cancellation_token.is_cancelled() {
            self.cancellation_token.cancel();
        }

        if let Some(source) = self.source.take() {
            source.shutdown();
        }
        self.task_tracker.close();
    }
}

impl Drop for KvEventPublisher {
    fn drop(&mut self) {
        self.shutdown();
    }
}
