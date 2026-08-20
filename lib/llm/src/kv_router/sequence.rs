// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-specific glue for [`ActiveSequencesMultiWorker`].
//!
//! This module provides the concrete [`SequencePublisher`] and [`SequenceSubscriber`]
//! implementations that wire the runtime-agnostic business logic (in `dynamo_kv_router`)
//! to the configured event transport and Prometheus metrics.

mod direct_zmq;

pub use dynamo_kv_router::multi_worker_sequence::{
    ActiveSequencesMultiWorker, SequenceError, SequencePublishQueueError, SequencePublisher,
    SequenceRequest, SequenceSubscriber,
};
use dynamo_kv_router::protocols::{
    ActiveLoad, ActiveSequenceEvent, ActiveSequenceEventBatch, MAX_REPLICA_BATCH_DURATION,
    MAX_REPLICA_BATCH_EVENTS, WorkerWithDpRank,
};
pub use dynamo_kv_router::sequence::{ActiveSequences, RequestId};

use anyhow::Result;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::{
    EventPublisher, EventSubscriber, EventTransportKind, TypedEventSubscriber,
};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;

use super::metrics::{RouterWorkerStatusMetrics, WORKER_LOAD_METRICS};
use crate::kv_router::{ACTIVE_SEQUENCES_SUBJECT, KV_METRICS_SUBJECT};
use crate::local_model::runtime_config::ModelRuntimeConfig;
#[cfg(test)]
use dynamo_kv_router::protocols::PrefillLoadHint;
#[cfg(test)]
use dynamo_runtime::transports::event_plane::MsgpackCodec;

// Match the existing standalone replica-sync queue. Lifecycle callers enqueue without awaiting;
// if the queue is full, the newest event is dropped without blocking the local mutation.
const REPLICA_EVENT_CHANNEL_CAPACITY: usize = 100_000;

const WORKER_LOAD_EVENT_CHANNEL_CAPACITY: usize = 100_000;

const WORKER_LOAD_DROP_LOG_INTERVAL: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveSequenceEventWireFormat {
    Singleton,
    Batch,
}

fn active_sequence_event_wire_format(
    transport_kind: EventTransportKind,
) -> ActiveSequenceEventWireFormat {
    match transport_kind {
        EventTransportKind::Nats => ActiveSequenceEventWireFormat::Singleton,
        EventTransportKind::Zmq => ActiveSequenceEventWireFormat::Batch,
    }
}

struct ActiveSequenceEventSender {
    event_tx: mpsc::Sender<ActiveSequenceEvent>,
    cancellation_token: CancellationToken,
}

impl ActiveSequenceEventSender {
    fn channel(
        capacity: usize,
        cancellation_token: CancellationToken,
    ) -> (Self, mpsc::Receiver<ActiveSequenceEvent>) {
        let (event_tx, event_rx) = mpsc::channel(capacity);
        (
            Self {
                event_tx,
                cancellation_token,
            },
            event_rx,
        )
    }

    fn enqueue(&self, event: ActiveSequenceEvent) -> anyhow::Result<()> {
        match self.event_tx.try_send(event) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(event)) => {
                Err(SequencePublishQueueError::full(event, self.event_tx.max_capacity()).into())
            }
            Err(mpsc::error::TrySendError::Closed(event)) => {
                Err(SequencePublishQueueError::closed(
                    event,
                    self.event_tx.max_capacity(),
                    self.cancellation_token.is_cancelled(),
                )
                .into())
            }
        }
    }
}

fn active_sequence_event_channel(
    enabled: bool,
    capacity: usize,
    cancellation_token: &CancellationToken,
) -> Option<(
    ActiveSequenceEventSender,
    mpsc::Receiver<ActiveSequenceEvent>,
)> {
    enabled.then(|| ActiveSequenceEventSender::channel(capacity, cancellation_token.child_token()))
}

#[derive(Default)]
struct WorkerLoadDropLog {
    last_logged_at: Option<std::time::Instant>,
    suppressed_since_last_log: u64,
}

impl WorkerLoadDropLog {
    fn record(&mut self, now: std::time::Instant) -> Option<u64> {
        let should_log = self.last_logged_at.is_none_or(|last_logged_at| {
            now.saturating_duration_since(last_logged_at) >= WORKER_LOAD_DROP_LOG_INTERVAL
        });
        if should_log {
            let dropped = self.suppressed_since_last_log.saturating_add(1);
            self.last_logged_at = Some(now);
            self.suppressed_since_last_log = 0;
            Some(dropped)
        } else {
            self.suppressed_since_last_log = self.suppressed_since_last_log.saturating_add(1);
            None
        }
    }
}

struct WorkerLoadEventSender {
    load_tx: mpsc::Sender<ActiveLoad>,
    cancellation_token: CancellationToken,
    drop_log: Mutex<WorkerLoadDropLog>,
}

impl WorkerLoadEventSender {
    fn channel(
        capacity: usize,
        cancellation_token: CancellationToken,
    ) -> (Self, mpsc::Receiver<ActiveLoad>) {
        let (load_tx, load_rx) = mpsc::channel(capacity);
        (
            Self {
                load_tx,
                cancellation_token,
                drop_log: Mutex::new(WorkerLoadDropLog::default()),
            },
            load_rx,
        )
    }

    fn enqueue(&self, load: ActiveLoad) -> bool {
        match self.load_tx.try_send(load) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(load)) => {
                self.record_drop("queue full", &load);
                false
            }
            Err(mpsc::error::TrySendError::Closed(load)) => {
                self.record_drop("queue closed", &load);
                false
            }
        }
    }

    fn record_drop(&self, reason: &'static str, load: &ActiveLoad) {
        if self.cancellation_token.is_cancelled() {
            tracing::trace!(
                worker_id = load.worker_id,
                dp_rank = load.dp_rank,
                reason,
                "Dropped worker-load snapshot during shutdown"
            );
            return;
        }
        let dropped = self
            .drop_log
            .lock()
            .expect("worker-load drop log mutex poisoned")
            .record(std::time::Instant::now());
        if let Some(dropped) = dropped {
            tracing::warn!(
                dropped,
                capacity = self.load_tx.max_capacity(),
                worker_id = load.worker_id,
                dp_rank = load.dp_rank,
                reason,
                "Dropped worker-load snapshots without publishing them"
            );
        }
    }
}

/// Concrete [`SequencePublisher`] backed by the runtime event plane and Prometheus gauges.
pub struct RuntimeSequencePublisher {
    event_sender: Option<ActiveSequenceEventSender>,
    load_sender: WorkerLoadEventSender,
    _load_publisher_task: AbortOnDropHandle<()>,
    worker_status_metrics: Arc<RouterWorkerStatusMetrics>,
}

impl SequencePublisher for RuntimeSequencePublisher {
    fn enqueue_event(&self, event: ActiveSequenceEvent) -> anyhow::Result<()> {
        let Some(event_sender) = &self.event_sender else {
            return Ok(());
        };
        event_sender.enqueue(event)
    }

    fn publish_load(&self, load: ActiveLoad) {
        self.load_sender.enqueue(load);
    }

    // The default batch method routes every snapshot through this FIFO.

    fn observe_load(
        &self,
        worker: &WorkerWithDpRank,
        worker_type: &str,
        blocks: usize,
        tokens: usize,
    ) {
        WORKER_LOAD_METRICS.observe(
            worker.worker_id,
            worker.dp_rank,
            worker_type,
            blocks,
            tokens,
        );
    }

    fn observe_worker_registered(&self, worker: &WorkerWithDpRank, worker_type: &str) {
        self.worker_status_metrics
            .set_registered(worker.worker_id, worker.dp_rank, worker_type);
    }

    fn observe_worker_removed(&self, worker: &WorkerWithDpRank, worker_type: &str) {
        self.worker_status_metrics
            .remove_worker(worker.worker_id, worker.dp_rank, worker_type);
    }
}

trait SingletonEventPublisher: Send + Sync {
    fn publish_event(
        &self,
        event: &ActiveSequenceEvent,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;
}

impl SingletonEventPublisher for EventPublisher {
    async fn publish_event(&self, event: &ActiveSequenceEvent) -> anyhow::Result<()> {
        self.publish(event).await
    }
}

async fn run_replica_singleton_publisher<P: SingletonEventPublisher>(
    publisher: P,
    mut event_rx: mpsc::Receiver<ActiveSequenceEvent>,
    cancellation_token: CancellationToken,
) {
    loop {
        let event = tokio::select! {
            _ = cancellation_token.cancelled() => break,
            event = event_rx.recv() => match event {
                Some(event) => event,
                None => break,
            },
        };
        // Replica sync is best-effort, so cancellation drops an in-flight publish rather than
        // delaying shutdown on transport backpressure.
        let publish_result = tokio::select! {
            _ = cancellation_token.cancelled() => break,
            result = publisher.publish_event(&event) => result,
        };
        if let Err(error) = publish_result {
            tracing::error!(
                request_id = %event.request_id,
                worker = ?event.worker,
                error = %error,
                "Failed to publish active-sequence replica event"
            );
        }
    }
}

trait LoadEventPublisher: Send + Sync {
    fn publish_load_event(
        &self,
        load: &ActiveLoad,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;
}

impl LoadEventPublisher for EventPublisher {
    async fn publish_load_event(&self, load: &ActiveLoad) -> anyhow::Result<()> {
        self.publish(load).await
    }
}

async fn run_worker_load_publisher<P: LoadEventPublisher>(
    publisher: P,
    mut load_rx: mpsc::Receiver<ActiveLoad>,
    cancellation_token: CancellationToken,
) {
    loop {
        let load = tokio::select! {
            _ = cancellation_token.cancelled() => break,
            load = load_rx.recv() => match load {
                Some(load) => load,
                None => break,
            },
        };
        let publish_result = tokio::select! {
            _ = cancellation_token.cancelled() => break,
            result = publisher.publish_load_event(&load) => result,
        };
        if let Err(error) = publish_result {
            tracing::trace!(
                worker_id = load.worker_id,
                dp_rank = load.dp_rank,
                error = %error,
                "Failed to publish ActiveLoad"
            );
        }
    }
}

async fn publish_replica_batch(publisher: &EventPublisher, events: Vec<ActiveSequenceEvent>) {
    let batch = ActiveSequenceEventBatch { events };
    let first_request_id = &batch
        .events
        .first()
        .expect("replica batch must contain an event")
        .request_id;
    let last_request_id = &batch
        .events
        .last()
        .expect("replica batch must contain an event")
        .request_id;

    if let Err(error) = publisher.publish(&batch).await {
        tracing::error!(
            event_count = batch.events.len(),
            first_request_id = %first_request_id,
            last_request_id = %last_request_id,
            error = %error,
            "Failed to publish active-sequence replica batch"
        );
    }
}

async fn collect_replica_batch(
    first_event: ActiveSequenceEvent,
    event_rx: &mut mpsc::Receiver<ActiveSequenceEvent>,
    cancellation_token: &CancellationToken,
) -> (Vec<ActiveSequenceEvent>, bool) {
    let mut events = Vec::with_capacity(MAX_REPLICA_BATCH_EVENTS);
    events.push(first_event);
    let deadline = Instant::now() + MAX_REPLICA_BATCH_DURATION;
    let flush_timer = tokio::time::sleep_until(deadline);
    tokio::pin!(flush_timer);

    while events.len() < MAX_REPLICA_BATCH_EVENTS {
        tokio::select! {
            _ = cancellation_token.cancelled() => return (events, true),
            _ = &mut flush_timer => break,
            event = event_rx.recv() => match event {
                Some(event) => events.push(event),
                None => return (events, true),
            },
        }
    }

    (events, false)
}

async fn run_replica_batch_publisher(
    publisher: EventPublisher,
    mut event_rx: mpsc::Receiver<ActiveSequenceEvent>,
    cancellation_token: CancellationToken,
) {
    loop {
        let first_event = tokio::select! {
            _ = cancellation_token.cancelled() => break,
            event = event_rx.recv() => match event {
                Some(event) => event,
                None => break,
            },
        };
        let (events, stop_after_flush) =
            collect_replica_batch(first_event, &mut event_rx, &cancellation_token).await;
        publish_replica_batch(&publisher, events).await;
        if stop_after_flush {
            break;
        }
    }
}

enum ActiveSequenceEventSubscriber {
    Nats(TypedEventSubscriber<ActiveSequenceEvent>),
    Zmq(TypedEventSubscriber<ActiveSequenceEventBatch>),
}

/// Concrete [`SequenceSubscriber`] backed by the configured runtime event transport.
pub struct RuntimeSequenceSubscriber {
    inner: ActiveSequenceEventSubscriber,
    pending: VecDeque<ActiveSequenceEvent>,
}

impl RuntimeSequenceSubscriber {
    pub(crate) async fn for_endpoint(endpoint: &Endpoint) -> Result<Self> {
        let transport_kind = endpoint.drt().default_event_transport_kind();
        let subscriber = EventSubscriber::for_endpoint_with_transport(
            endpoint,
            ACTIVE_SEQUENCES_SUBJECT,
            transport_kind,
        )
        .await?;
        let inner = match active_sequence_event_wire_format(transport_kind) {
            ActiveSequenceEventWireFormat::Singleton => {
                ActiveSequenceEventSubscriber::Nats(subscriber.typed::<ActiveSequenceEvent>())
            }
            ActiveSequenceEventWireFormat::Batch => {
                ActiveSequenceEventSubscriber::Zmq(subscriber.typed::<ActiveSequenceEventBatch>())
            }
        };
        Ok(Self {
            inner,
            pending: VecDeque::new(),
        })
    }
}

impl SequenceSubscriber for RuntimeSequenceSubscriber {
    async fn next_event(&mut self) -> Option<anyhow::Result<ActiveSequenceEvent>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(Ok(event));
            }
            match &mut self.inner {
                ActiveSequenceEventSubscriber::Nats(subscriber) => {
                    return match subscriber.next().await? {
                        Ok((_envelope, event)) => Some(Ok(event)),
                        Err(error) => Some(Err(error)),
                    };
                }
                ActiveSequenceEventSubscriber::Zmq(subscriber) => match subscriber.next().await? {
                    Ok((_envelope, batch)) => self.pending.extend(batch.events),
                    Err(error) => return Some(Err(error)),
                },
            }
        }
    }

    fn poll_next_event(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<anyhow::Result<ActiveSequenceEvent>>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }
            match &mut self.inner {
                ActiveSequenceEventSubscriber::Nats(subscriber) => {
                    return match subscriber.poll_next(cx) {
                        Poll::Ready(Some(Ok((_envelope, event)))) => Poll::Ready(Some(Ok(event))),
                        Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(error))),
                        Poll::Ready(None) => Poll::Ready(None),
                        Poll::Pending => Poll::Pending,
                    };
                }
                ActiveSequenceEventSubscriber::Zmq(subscriber) => match subscriber.poll_next(cx) {
                    Poll::Ready(Some(Ok((_envelope, batch)))) => self.pending.extend(batch.events),
                    Poll::Ready(Some(Err(error))) => return Poll::Ready(Some(Err(error))),
                    Poll::Ready(None) => return Poll::Ready(None),
                    Poll::Pending => return Poll::Pending,
                },
            }
        }
    }
}

/// Type alias for the runtime-wired multi-worker sequence tracker.
pub type ActiveSequencesMulti = ActiveSequencesMultiWorker<RuntimeSequencePublisher>;

/// Convenience async constructor that creates the event-plane publishers/subscribers
/// and returns an `Arc<ActiveSequencesMulti>` with replica sync already running.
pub async fn create_multi_worker_sequences(
    endpoint: Endpoint,
    block_size: usize,
    workers_with_configs: HashMap<u64, ModelRuntimeConfig>,
    replica_sync: bool,
    router_id: u64,
    worker_type: &'static str,
    cancellation_token: CancellationToken,
) -> Result<Arc<ActiveSequencesMulti>> {
    let transport_kind = endpoint.drt().default_event_transport_kind();
    let event_sender = if let Some((event_sender, event_rx)) = active_sequence_event_channel(
        replica_sync,
        REPLICA_EVENT_CHANNEL_CAPACITY,
        &cancellation_token,
    ) {
        let publisher_cancellation_token = event_sender.cancellation_token.clone();
        let event_publisher = EventPublisher::for_endpoint_with_transport(
            &endpoint,
            ACTIVE_SEQUENCES_SUBJECT,
            transport_kind,
        )
        .await?;
        match active_sequence_event_wire_format(transport_kind) {
            ActiveSequenceEventWireFormat::Singleton => {
                tokio::spawn(run_replica_singleton_publisher(
                    event_publisher,
                    event_rx,
                    publisher_cancellation_token,
                ));
            }
            ActiveSequenceEventWireFormat::Batch => {
                tokio::spawn(run_replica_batch_publisher(
                    event_publisher,
                    event_rx,
                    publisher_cancellation_token,
                ));
            }
        }
        Some(event_sender)
    } else {
        None
    };
    let (load_sender, load_rx) = WorkerLoadEventSender::channel(
        WORKER_LOAD_EVENT_CHANNEL_CAPACITY,
        cancellation_token.child_token(),
    );
    let load_publisher_cancellation_token = load_sender.cancellation_token.clone();
    let metrics_publisher = EventPublisher::for_endpoint(&endpoint, KV_METRICS_SUBJECT).await?;
    let load_publisher_task = AbortOnDropHandle::new(tokio::spawn(run_worker_load_publisher(
        metrics_publisher,
        load_rx,
        load_publisher_cancellation_token,
    )));
    let worker_status_metrics = RouterWorkerStatusMetrics::from_component(endpoint.component());

    let publisher = RuntimeSequencePublisher {
        event_sender,
        load_sender,
        _load_publisher_task: load_publisher_task,
        worker_status_metrics,
    };

    let dp_range: HashMap<u64, (u32, u32)> = workers_with_configs
        .into_iter()
        .map(|(id, config)| {
            (
                id,
                (config.data_parallel_start_rank, config.data_parallel_size),
            )
        })
        .collect();

    let multi_worker = ActiveSequencesMultiWorker::new(
        publisher,
        block_size,
        dp_range,
        replica_sync,
        router_id,
        worker_type,
    );

    let arc = Arc::new(multi_worker);

    if replica_sync {
        let direct_config = direct_zmq::DirectZmqSequenceConfig::from_env();
        if direct_config.should_use_direct(transport_kind) {
            let _direct_zmq_task = direct_zmq::start(
                endpoint,
                arc.clone(),
                direct_config.rcvhwm,
                cancellation_token.child_token(),
            )
            .await?;
        } else {
            let subscriber = RuntimeSequenceSubscriber::for_endpoint(&endpoint).await?;
            arc.start_replica_sync(subscriber, cancellation_token.child_token());
        }
    }

    arc.start_periodic_force_expiry_across_all_workers(cancellation_token.child_token());

    Ok(arc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::ActiveSequenceEventData;
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use tokio::time::Instant;

    fn tracking_hint(tokens: usize) -> Option<PrefillLoadHint> {
        Some(PrefillLoadHint {
            initial_effective_prefill_tokens: tokens,
            expected_prefill_duration: None,
        })
    }

    fn free_event(request_id: impl Into<String>) -> ActiveSequenceEvent {
        ActiveSequenceEvent {
            request_id: request_id.into(),
            worker: WorkerWithDpRank::new(1, 0),
            data: ActiveSequenceEventData::Free,
            router_id: 7,
            lora_name: None,
        }
    }

    fn add_event(request_id: impl Into<String>) -> ActiveSequenceEvent {
        ActiveSequenceEvent {
            request_id: request_id.into(),
            worker: WorkerWithDpRank::new(1, 0),
            data: ActiveSequenceEventData::AddRequest {
                token_sequence: None,
                track_prefill_tokens: false,
                expected_output_tokens: None,
                prefill_load_hint: None,
            },
            router_id: 7,
            lora_name: None,
        }
    }

    fn mark_event(request_id: impl Into<String>) -> ActiveSequenceEvent {
        ActiveSequenceEvent {
            request_id: request_id.into(),
            worker: WorkerWithDpRank::new(1, 0),
            data: ActiveSequenceEventData::MarkPrefillCompleted,
            router_id: 7,
            lora_name: None,
        }
    }

    struct BlockingSingletonPublisher {
        attempted_tx: mpsc::UnboundedSender<&'static str>,
        release_add: Arc<tokio::sync::Notify>,
        active: Arc<std::sync::atomic::AtomicUsize>,
        max_active: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl SingletonEventPublisher for BlockingSingletonPublisher {
        async fn publish_event(&self, event: &ActiveSequenceEvent) -> anyhow::Result<()> {
            let event_name = match &event.data {
                ActiveSequenceEventData::AddRequest { .. } => "add",
                ActiveSequenceEventData::MarkPrefillCompleted => "mark",
                ActiveSequenceEventData::Free => "free",
            };
            let active = self
                .active
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;
            self.max_active
                .fetch_max(active, std::sync::atomic::Ordering::SeqCst);
            self.attempted_tx.send(event_name).unwrap();

            if event_name == "add" {
                self.release_add.notified().await;
            }

            self.active
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

            if event_name == "mark" {
                anyhow::bail!("synthetic singleton publish failure");
            }
            Ok(())
        }
    }

    #[test]
    fn active_sequence_publish_sender_preserves_lifecycle_order() {
        let (sender, mut event_rx) =
            ActiveSequenceEventSender::channel(3, CancellationToken::new());
        sender.enqueue(add_event("ordered")).unwrap();
        sender.enqueue(mark_event("ordered")).unwrap();
        sender.enqueue(free_event("ordered")).unwrap();

        assert!(matches!(
            event_rx.try_recv().unwrap().data,
            ActiveSequenceEventData::AddRequest { .. }
        ));
        assert!(matches!(
            event_rx.try_recv().unwrap().data,
            ActiveSequenceEventData::MarkPrefillCompleted
        ));
        assert!(matches!(
            event_rx.try_recv().unwrap().data,
            ActiveSequenceEventData::Free
        ));
    }

    #[test]
    fn active_sequence_publish_sender_drops_newest_when_full() {
        let (sender, mut event_rx) =
            ActiveSequenceEventSender::channel(1, CancellationToken::new());
        sender.enqueue(add_event("accepted")).unwrap();

        let error = sender
            .enqueue(free_event("dropped"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("queue full"));
        assert!(error.contains("request_id=dropped"));
        assert!(error.contains("capacity=1"));
        assert_eq!(event_rx.len(), 1);
        assert_eq!(event_rx.try_recv().unwrap().request_id, "accepted");
    }

    #[test]
    fn active_sequence_publish_channel_is_absent_when_replica_sync_disabled() {
        assert!(active_sequence_event_channel(false, 1, &CancellationToken::new()).is_none());
    }

    #[test]
    fn active_sequence_publish_sender_classifies_closed_queue_by_cancellation() {
        let cancellation_token = CancellationToken::new();
        let (sender, event_rx) = ActiveSequenceEventSender::channel(1, cancellation_token.clone());
        drop(event_rx);

        let unexpected = sender.enqueue(free_event("unexpected")).unwrap_err();
        assert!(matches!(
            unexpected.downcast_ref::<SequencePublishQueueError>(),
            Some(SequencePublishQueueError::Closed {
                during_shutdown: false,
                ..
            })
        ));

        cancellation_token.cancel();
        let shutdown = sender.enqueue(free_event("shutdown")).unwrap_err();
        assert!(matches!(
            shutdown.downcast_ref::<SequencePublishQueueError>(),
            Some(SequencePublishQueueError::Closed {
                during_shutdown: true,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn active_sequence_singleton_publisher_serializes_and_stops_on_cancellation() {
        let (attempted_tx, mut attempted_rx) = mpsc::unbounded_channel();
        let release_add = Arc::new(tokio::sync::Notify::new());
        let active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let max_active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let publisher = BlockingSingletonPublisher {
            attempted_tx,
            release_add: Arc::clone(&release_add),
            active,
            max_active: Arc::clone(&max_active),
        };
        let (event_tx, event_rx) = mpsc::channel(3);
        event_tx.send(add_event("ordered")).await.unwrap();
        event_tx.send(mark_event("ordered")).await.unwrap();
        event_tx.send(free_event("ordered")).await.unwrap();

        let cancellation_token = CancellationToken::new();
        let task = tokio::spawn(run_replica_singleton_publisher(
            publisher,
            event_rx,
            cancellation_token.clone(),
        ));

        let first = tokio::time::timeout(std::time::Duration::from_secs(1), attempted_rx.recv())
            .await
            .expect("AddRequest publish should start")
            .expect("attempt channel should remain open");
        assert_eq!(first, "add");
        assert!(attempted_rx.try_recv().is_err());
        assert_eq!(max_active.load(std::sync::atomic::Ordering::SeqCst), 1);

        release_add.notify_one();
        let mut attempted = vec![first];
        for _ in 0..2 {
            attempted.push(
                tokio::time::timeout(std::time::Duration::from_secs(1), attempted_rx.recv())
                    .await
                    .expect("all queued publishes should be attempted")
                    .expect("attempt channel should remain open"),
            );
        }

        assert_eq!(attempted, ["add", "mark", "free"]);
        assert_eq!(max_active.load(std::sync::atomic::Ordering::SeqCst), 1);

        event_tx.send(add_event("blocked")).await.unwrap();
        let blocked = tokio::time::timeout(std::time::Duration::from_secs(1), attempted_rx.recv())
            .await
            .expect("blocked AddRequest publish should start")
            .expect("attempt channel should remain open");
        assert_eq!(blocked, "add");

        cancellation_token.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await
            .expect("singleton publisher should stop after cancellation")
            .expect("singleton publisher task should not panic");
    }

    fn active_load(worker_id: u64, blocking: bool) -> ActiveLoad {
        ActiveLoad {
            worker_id,
            dp_rank: 0,
            active_decode_blocks: Some(worker_id),
            active_prefill_tokens: None,
            kv_used_blocks: blocking.then_some(1),
        }
    }

    struct BlockingLoadPublisher {
        attempted_tx: mpsc::UnboundedSender<u64>,
        release: Arc<tokio::sync::Notify>,
        active: Arc<std::sync::atomic::AtomicUsize>,
        max_active: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl LoadEventPublisher for BlockingLoadPublisher {
        async fn publish_load_event(&self, load: &ActiveLoad) -> anyhow::Result<()> {
            let active = self
                .active
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;
            self.max_active
                .fetch_max(active, std::sync::atomic::Ordering::SeqCst);
            self.attempted_tx.send(load.worker_id).unwrap();

            if load.kv_used_blocks == Some(1) {
                self.release.notified().await;
            }

            self.active
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn worker_load_sender_drops_newest_when_full() {
        let (sender, mut load_rx) = WorkerLoadEventSender::channel(2, CancellationToken::new());

        assert!(sender.enqueue(active_load(0, false)));
        assert!(sender.enqueue(active_load(1, false)));
        assert!(!sender.enqueue(active_load(2, false)));
        assert!(!sender.enqueue(active_load(3, false)));

        assert_eq!(load_rx.len(), 2);
        assert_eq!(load_rx.try_recv().unwrap().worker_id, 0);
        assert_eq!(load_rx.try_recv().unwrap().worker_id, 1);
        assert!(load_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn worker_load_publisher_bounds_work_and_stops_on_cancellation() {
        let (attempted_tx, mut attempted_rx) = mpsc::unbounded_channel();
        let release = Arc::new(tokio::sync::Notify::new());
        let max_active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let publisher = BlockingLoadPublisher {
            attempted_tx,
            release: Arc::clone(&release),
            active: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            max_active: Arc::clone(&max_active),
        };

        let capacity = 2;
        let cancellation_token = CancellationToken::new();
        let (sender, load_rx) =
            WorkerLoadEventSender::channel(capacity, cancellation_token.child_token());
        let task = tokio::spawn(run_worker_load_publisher(
            publisher,
            load_rx,
            cancellation_token.clone(),
        ));

        assert!(sender.enqueue(active_load(0, true)));
        let first = tokio::time::timeout(std::time::Duration::from_secs(1), attempted_rx.recv())
            .await
            .expect("first load publish should start")
            .expect("attempt channel should remain open");
        assert_eq!(first, 0);

        for worker_id in 1..=capacity as u64 {
            assert!(sender.enqueue(active_load(worker_id, false)));
        }
        for worker_id in (capacity as u64 + 1)..=(capacity as u64 + 3) {
            assert!(!sender.enqueue(active_load(worker_id, false)));
        }
        // Give a concurrent publish time to violate the single-in-flight invariant.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(
            attempted_rx.try_recv().is_err(),
            "queued loads must not be published while an earlier publish is pending"
        );
        assert_eq!(max_active.load(std::sync::atomic::Ordering::SeqCst), 1);

        release.notify_one();
        let mut attempted = vec![first];
        for _ in 0..capacity {
            attempted.push(
                tokio::time::timeout(std::time::Duration::from_secs(1), attempted_rx.recv())
                    .await
                    .expect("accepted loads should be attempted once released")
                    .expect("attempt channel should remain open"),
            );
        }
        assert_eq!(attempted, [0, 1, 2]);
        assert_eq!(max_active.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(attempted_rx.try_recv().is_err());

        assert!(sender.enqueue(active_load(9, true)));
        let blocked = tokio::time::timeout(std::time::Duration::from_secs(1), attempted_rx.recv())
            .await
            .expect("blocked load publish should start")
            .expect("attempt channel should remain open");
        assert_eq!(blocked, 9);

        cancellation_token.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await
            .expect("load publisher should stop after cancellation")
            .expect("load publisher task should not panic");
    }

    #[tokio::test(start_paused = true)]
    async fn active_sequence_batch_collection_uses_time_and_count_caps() {
        let (event_tx, mut event_rx) = mpsc::channel(MAX_REPLICA_BATCH_EVENTS + 1);
        for request_id in 0..100 {
            event_tx
                .send(free_event(format!("free-{request_id}")))
                .await
                .unwrap();
        }

        let first = event_rx.recv().await.unwrap();
        let start = Instant::now();
        let (events, stop) =
            collect_replica_batch(first, &mut event_rx, &CancellationToken::new()).await;
        assert!(!stop);
        assert_eq!(events.len(), 100);
        assert_eq!(Instant::now() - start, MAX_REPLICA_BATCH_DURATION);
        let payload = MsgpackCodec
            .encode_payload(&ActiveSequenceEventBatch { events })
            .unwrap();
        let decoded: ActiveSequenceEventBatch = MsgpackCodec.decode_payload(&payload).unwrap();
        assert_eq!(decoded.events.len(), 100);
        for (request_id, event) in decoded.events.iter().enumerate() {
            assert_eq!(event.request_id, format!("free-{request_id}"));
        }

        for request_id in 0..=MAX_REPLICA_BATCH_EVENTS {
            event_tx
                .send(free_event(format!("count-{request_id}")))
                .await
                .unwrap();
        }
        let first = event_rx.recv().await.unwrap();
        let start = Instant::now();
        let (events, stop) =
            collect_replica_batch(first, &mut event_rx, &CancellationToken::new()).await;
        assert!(!stop);
        assert_eq!(events.len(), MAX_REPLICA_BATCH_EVENTS);
        assert_eq!(Instant::now(), start);
        assert_eq!(event_rx.len(), 1);

        let last = event_rx.recv().await.unwrap();
        let start = Instant::now();
        let (remaining, stop) =
            collect_replica_batch(last, &mut event_rx, &CancellationToken::new()).await;
        assert!(!stop);
        assert_eq!(remaining.len(), 1);
        assert_eq!(Instant::now() - start, MAX_REPLICA_BATCH_DURATION);
    }

    #[test]
    fn active_sequence_wire_format_uses_singletons_only_for_nats() {
        assert_eq!(
            active_sequence_event_wire_format(EventTransportKind::Nats),
            ActiveSequenceEventWireFormat::Singleton
        );
        assert_eq!(
            active_sequence_event_wire_format(EventTransportKind::Zmq),
            ActiveSequenceEventWireFormat::Batch
        );

        let event = free_event("request");
        let singleton_payload = MsgpackCodec.encode_payload(&event).unwrap();
        let decoded_singleton: ActiveSequenceEvent =
            MsgpackCodec.decode_payload(&singleton_payload).unwrap();
        assert_eq!(decoded_singleton.request_id, "request");
        assert!(
            MsgpackCodec
                .decode_payload::<ActiveSequenceEventBatch>(&singleton_payload)
                .is_err()
        );

        let batch_payload = MsgpackCodec
            .encode_payload(&ActiveSequenceEventBatch {
                events: vec![event],
            })
            .unwrap();
        let decoded_batch: ActiveSequenceEventBatch =
            MsgpackCodec.decode_payload(&batch_payload).unwrap();
        assert_eq!(decoded_batch.events[0].request_id, "request");
        assert!(
            MsgpackCodec
                .decode_payload::<ActiveSequenceEvent>(&batch_payload)
                .is_err()
        );
    }

    #[tokio::test]
    async fn active_sequence_replica_sync_isolated_by_endpoint() -> Result<()> {
        let runtime = Runtime::from_current()?;
        let distributed =
            DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
        let namespace = distributed.namespace(format!(
            "active-sequence-endpoint-isolation-{}",
            uuid::Uuid::new_v4()
        ))?;
        let component = namespace.component("workers")?;
        let endpoint_a = component.endpoint("generate-a");
        let endpoint_b = component.endpoint("generate-b");
        let workers = HashMap::from([(0, ModelRuntimeConfig::new())]);

        let cancel = CancellationToken::new();
        let sequences_a = create_multi_worker_sequences(
            endpoint_a.clone(),
            4,
            workers.clone(),
            true,
            1,
            crate::discovery::WORKER_TYPE_DECODE,
            cancel.child_token(),
        )
        .await?;
        let sequences_a_peer = create_multi_worker_sequences(
            endpoint_a,
            4,
            workers.clone(),
            true,
            3,
            crate::discovery::WORKER_TYPE_DECODE,
            cancel.child_token(),
        )
        .await?;
        let sequences_b = create_multi_worker_sequences(
            endpoint_b,
            4,
            workers,
            true,
            2,
            crate::discovery::WORKER_TYPE_DECODE,
            cancel.child_token(),
        )
        .await?;

        let worker = WorkerWithDpRank::new(0, 0);
        tokio::time::timeout(tokio::time::Duration::from_secs(5), async {
            for request_index in 0..100 {
                if sequences_a_peer.active_blocks()[&worker] > 0 {
                    break;
                }

                sequences_a.add_request(
                    SequenceRequest {
                        request_id: format!("endpoint-a-request-{request_index}"),
                        token_sequence: Some(vec![1, 2, 3, 4]),
                        track_prefill_tokens: true,
                        expected_output_tokens: None,
                        prefill_load_hint: tracking_hint(4),
                        worker,
                        lora_name: None,
                    },
                    Instant::now(),
                )?;
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            }

            anyhow::ensure!(sequences_a_peer.active_blocks()[&worker] > 0);
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        assert!(sequences_a.active_blocks()[&worker] > 0);
        assert!(sequences_a_peer.active_blocks()[&worker] > 0);
        let leaked_to_b = tokio::time::timeout(tokio::time::Duration::from_millis(250), async {
            loop {
                if sequences_b.active_blocks()[&worker] > 0 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await;
        assert!(
            leaked_to_b.is_err(),
            "endpoint B received endpoint A sequence state"
        );
        assert_eq!(sequences_b.active_blocks()[&worker], 0);
        cancel.cancel();
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_multi_worker_cross_instance_sync() -> Result<()> {
        dynamo_runtime::logging::init();

        let block_size = 4;

        let runtime = Runtime::from_current()?;
        let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

        let namespace = distributed.namespace("test_cross_instance_sync")?;
        let endpoint = namespace.component("sequences")?.endpoint("generate");

        let mut workers_with_configs = HashMap::new();

        let mut config_worker_0 = crate::local_model::runtime_config::ModelRuntimeConfig::new();
        config_worker_0.data_parallel_size = 2;
        workers_with_configs.insert(0, config_worker_0);

        let config_worker_1 = crate::local_model::runtime_config::ModelRuntimeConfig::new();
        workers_with_configs.insert(1, config_worker_1);

        let seq_manager_1 = create_multi_worker_sequences(
            endpoint.clone(),
            block_size,
            workers_with_configs.clone(),
            true,
            1,
            crate::discovery::WORKER_TYPE_DECODE,
            CancellationToken::new(),
        )
        .await?;
        let seq_manager_2 = create_multi_worker_sequences(
            endpoint,
            block_size,
            workers_with_configs,
            true,
            2,
            crate::discovery::WORKER_TYPE_DECODE,
            CancellationToken::new(),
        )
        .await?;

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        let decay_now = Instant::now();

        seq_manager_1.add_request(
            SequenceRequest {
                request_id: "request_0".to_string(),
                token_sequence: Some(vec![0, 1, 2]),
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: tracking_hint(12),
                worker: WorkerWithDpRank::new(0, 0),
                lora_name: None,
            },
            decay_now,
        )?;

        seq_manager_1.add_request(
            SequenceRequest {
                request_id: "request_1".to_string(),
                token_sequence: Some(vec![3, 4]),
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: tracking_hint(8),
                worker: WorkerWithDpRank::new(0, 1),
                lora_name: None,
            },
            decay_now,
        )?;

        seq_manager_2.add_request(
            SequenceRequest {
                request_id: "request_2".to_string(),
                token_sequence: Some(vec![0, 1, 2, 3]),
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: tracking_hint(16),
                worker: WorkerWithDpRank::new(1, 0),
                lora_name: None,
            },
            decay_now,
        )?;

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        let blocks_phase1 = seq_manager_1.active_blocks();
        let tokens_phase1 = seq_manager_1.active_tokens(Instant::now());

        let worker_0_dp0 = WorkerWithDpRank::new(0, 0);
        let worker_0_dp1 = WorkerWithDpRank::new(0, 1);
        let worker_1_dp0 = WorkerWithDpRank::new(1, 0);

        assert_eq!(
            blocks_phase1[&worker_0_dp0], 3,
            "Worker 0 dp_rank 0 should have 3 active blocks (from request_0)"
        );
        assert_eq!(
            blocks_phase1[&worker_0_dp1], 2,
            "Worker 0 dp_rank 1 should have 2 active blocks (from request_1)"
        );
        assert_eq!(
            blocks_phase1[&worker_1_dp0], 4,
            "Worker 1 dp_rank 0 should have 4 active blocks (from request_2 added by seq_manager_2)"
        );
        assert_eq!(
            tokens_phase1[&worker_0_dp0], 12,
            "Worker 0 dp_rank 0 should have 12 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_0_dp1], 8,
            "Worker 0 dp_rank 1 should have 8 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_1_dp0], 16,
            "Worker 1 dp_rank 0 should have 16 active tokens (from request_2 added by seq_manager_2)"
        );

        seq_manager_1.free(&"request_2".to_string(), Instant::now())?;

        seq_manager_2.free(&"request_0".to_string(), Instant::now())?;
        seq_manager_2.free(&"request_1".to_string(), Instant::now())?;

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        let blocks_phase2 = seq_manager_2.active_blocks();
        let tokens_phase2 = seq_manager_2.active_tokens(Instant::now());

        let all_workers = vec![
            WorkerWithDpRank::new(0, 0),
            WorkerWithDpRank::new(0, 1),
            WorkerWithDpRank::new(1, 0),
        ];

        for worker in all_workers {
            assert_eq!(
                blocks_phase2[&worker], 0,
                "Worker (id={}, dp_rank={}) should have 0 active blocks after all requests freed",
                worker.worker_id, worker.dp_rank
            );
            assert_eq!(
                tokens_phase2[&worker], 0,
                "Worker (id={}, dp_rank={}) should have 0 active tokens after all requests freed",
                worker.worker_id, worker.dp_rank
            );
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_multi_worker_no_token_sequence_sync() -> Result<()> {
        dynamo_runtime::logging::init();

        let block_size = 4;

        let runtime = Runtime::from_current()?;
        let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

        let namespace = distributed.namespace("test_no_token_seq_sync")?;
        let endpoint = namespace.component("sequences")?.endpoint("generate");

        let mut workers_with_configs = HashMap::new();
        workers_with_configs.insert(
            0,
            crate::local_model::runtime_config::ModelRuntimeConfig::new(),
        );
        workers_with_configs.insert(
            1,
            crate::local_model::runtime_config::ModelRuntimeConfig::new(),
        );
        workers_with_configs.insert(
            2,
            crate::local_model::runtime_config::ModelRuntimeConfig::new(),
        );

        let seq_manager_1 = create_multi_worker_sequences(
            endpoint.clone(),
            block_size,
            workers_with_configs.clone(),
            true,
            1,
            crate::discovery::WORKER_TYPE_DECODE,
            CancellationToken::new(),
        )
        .await?;
        let seq_manager_2 = create_multi_worker_sequences(
            endpoint,
            block_size,
            workers_with_configs,
            true,
            2,
            crate::discovery::WORKER_TYPE_DECODE,
            CancellationToken::new(),
        )
        .await?;

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        let decay_now = Instant::now();

        seq_manager_1.add_request(
            SequenceRequest {
                request_id: "request_0".to_string(),
                token_sequence: None,
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: tracking_hint(12),
                worker: WorkerWithDpRank::from_worker_id(0),
                lora_name: None,
            },
            decay_now,
        )?;

        seq_manager_1.add_request(
            SequenceRequest {
                request_id: "request_1".to_string(),
                token_sequence: None,
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: tracking_hint(8),
                worker: WorkerWithDpRank::from_worker_id(1),
                lora_name: None,
            },
            decay_now,
        )?;

        seq_manager_2.add_request(
            SequenceRequest {
                request_id: "request_2".to_string(),
                token_sequence: None,
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: tracking_hint(16),
                worker: WorkerWithDpRank::from_worker_id(2),
                lora_name: None,
            },
            decay_now,
        )?;

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        let tokens_phase1 = seq_manager_1.active_tokens(Instant::now());

        let worker_0 = WorkerWithDpRank::from_worker_id(0);
        let worker_1 = WorkerWithDpRank::from_worker_id(1);
        let worker_2 = WorkerWithDpRank::from_worker_id(2);

        assert_eq!(
            tokens_phase1[&worker_0], 12,
            "Worker 0 should have 12 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_1], 8,
            "Worker 1 should have 8 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_2], 16,
            "Worker 2 should have 16 active tokens (from request_2 added by seq_manager_2)"
        );

        seq_manager_1.mark_prefill_completed(&"request_2".to_string(), Instant::now())?;
        seq_manager_1.free(&"request_2".to_string(), Instant::now())?;

        seq_manager_2.mark_prefill_completed(&"request_0".to_string(), Instant::now())?;
        seq_manager_2.mark_prefill_completed(&"request_1".to_string(), Instant::now())?;
        seq_manager_2.free(&"request_0".to_string(), Instant::now())?;
        seq_manager_2.free(&"request_1".to_string(), Instant::now())?;

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        let tokens_phase2 = seq_manager_2.active_tokens(Instant::now());

        for worker_id in 0..=2 {
            let worker = WorkerWithDpRank::from_worker_id(worker_id);
            assert_eq!(
                tokens_phase2[&worker], 0,
                "Worker {} should have 0 active tokens after all requests freed",
                worker_id
            );
        }

        Ok(())
    }
}
