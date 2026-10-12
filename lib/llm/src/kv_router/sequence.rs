// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-specific glue for [`ActiveSequencesMultiWorker`].
//!
//! This module provides the concrete [`SequencePublisher`] and [`SequenceSubscriber`]
//! implementations that wire the runtime-agnostic business logic (in `dynamo_kv_router`)
//! to the configured event transport and Prometheus metrics.

mod direct_zmq;

pub use dynamo_kv_router::multi_worker_sequence::{
    ActiveSequencesMultiWorker, ReplicaRequestLeaseObserver, SchedulerLoadSnapshot, SequenceError,
    SequencePublishQueueError, SequencePublisher, SequenceRequest, SequenceSubscriber,
};
use dynamo_kv_router::protocols::{
    ActiveSequenceEvent, ActiveSequenceEventBatch, MAX_REPLICA_BATCH_DURATION,
    MAX_REPLICA_BATCH_EVENTS, SchedulerGroup, SchedulerLoad, WorkerWithDpRank,
};
pub use dynamo_kv_router::sequence::{ActiveSequences, RequestId};

use anyhow::Result;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::{
    EventPublisher, EventSubscriber, EventTransportKind, TypedEventSubscriber,
};
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::kv_router::{ACTIVE_SEQUENCES_SUBJECT, SCHEDULER_LOAD_SUBJECT};
use crate::utils::retry::{Backoff, FailureStreak};
#[cfg(test)]
use dynamo_runtime::transports::event_plane::MsgpackCodec;

// Match the existing standalone replica-sync queue. Lifecycle callers enqueue without awaiting;
// if the queue is full, the newest event is dropped without blocking the local mutation.
const REPLICA_EVENT_CHANNEL_CAPACITY: usize = 100_000;
/// Every interval, a router publishes the load of every registered worker rank on
/// [`SCHEDULER_LOAD_SUBJECT`]; nothing is published between ticks. Each message is also a
/// freshness heartbeat: the KV DC Relay derives its freshness window from this interval. Cost:
/// one small message per rank per router per interval.
pub(crate) const SCHEDULER_LOAD_PUBLISH_INTERVAL: Duration = Duration::from_secs(1);
const SCHEDULER_LOAD_CONNECT_RETRY_INITIAL: Duration = Duration::from_millis(100);
const SCHEDULER_LOAD_CONNECT_RETRY_MAX: Duration = Duration::from_secs(30);

/// How active-sequence events are framed on the wire for a transport.
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

/// Cloneable handle for bounded active-sequence event publication.
#[derive(Clone)]
pub struct ActiveSequenceEventPublisher {
    event_tx: mpsc::Sender<ActiveSequenceEvent>,
    cancellation_token: CancellationToken,
}

impl ActiveSequenceEventPublisher {
    pub(crate) fn channel(
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

    pub async fn for_endpoint(endpoint: &Endpoint, capacity: usize) -> Result<Self> {
        anyhow::ensure!(
            capacity > 0,
            "active-sequence queue capacity must be positive"
        );
        let cancellation_token = CancellationToken::new();
        let transport_kind = endpoint.drt().default_event_transport_kind();
        let event_publisher = EventPublisher::for_endpoint_with_transport(
            endpoint,
            ACTIVE_SEQUENCES_SUBJECT,
            transport_kind,
        )
        .await?;
        let (event_sender, event_rx) = Self::channel(capacity, cancellation_token.clone());
        match active_sequence_event_wire_format(transport_kind) {
            ActiveSequenceEventWireFormat::Singleton => {
                tokio::spawn(run_replica_singleton_publisher(
                    event_publisher,
                    event_rx,
                    cancellation_token,
                ));
            }
            ActiveSequenceEventWireFormat::Batch => {
                tokio::spawn(run_replica_batch_publisher(
                    event_publisher,
                    event_rx,
                    cancellation_token,
                ));
            }
        }
        Ok(event_sender)
    }

    /// Emit a worker-origin completion mark. `router_id` carries the worker's source DRT identity.
    pub fn mark_prefill_completed(
        &self,
        request_id: String,
        worker_id: u64,
        dp_rank: u32,
    ) -> anyhow::Result<()> {
        let worker = WorkerWithDpRank::new(worker_id, dp_rank);
        self.enqueue(ActiveSequenceEvent {
            request_id,
            worker,
            data: dynamo_kv_router::protocols::ActiveSequenceEventData::MarkPrefillCompleted,
            router_id: worker.worker_id,
            lora_name: None,
        })
    }
}

/// One event per message, for transports that carry singletons.
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

/// Replica-sync channels for an embedded selection partition over the runtime
/// event plane. Inbound events on `ACTIVE_SEQUENCES_SUBJECT` are always
/// forwarded; outbound events are published only when `publishes_outbound` is
/// set. The inbound leg runs when the returned [`ReplicaIngress`] is started,
/// so the caller can install every consumer of lifecycle events first.
pub(crate) async fn host_replica_channels(
    endpoint: &Endpoint,
    router_id: u64,
    publishes_outbound: bool,
    cancellation_token: CancellationToken,
) -> Result<(
    dynamo_kv_router::services::selection::HostReplicaChannels,
    ReplicaIngress,
)> {
    let transport_kind = endpoint.drt().default_event_transport_kind();
    let event_sender = if publishes_outbound {
        let (event_sender, event_rx) = mpsc::channel(REPLICA_EVENT_CHANNEL_CAPACITY);
        let publisher_cancellation_token = cancellation_token.clone();
        let event_publisher = EventPublisher::for_endpoint_with_transport(
            endpoint,
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

    let (inbound_tx, inbound_rx) = mpsc::channel(REPLICA_EVENT_CHANNEL_CAPACITY);
    let ingress = ReplicaIngress {
        endpoint: endpoint.clone(),
        inbound_tx: inbound_tx.clone(),
        cancellation_token,
    };
    Ok((
        dynamo_kv_router::services::selection::HostReplicaChannels {
            outbound: event_sender,
            inbound_tx,
            inbound_rx,
            process_id: router_id,
            ingress_observer: None,
        },
        ingress,
    ))
}

/// The inbound leg of [`host_replica_channels`]: peer replica events and
/// worker-origin completion marks feed `inbound_tx` from the direct-ZMQ
/// fan-in or the runtime subscriber, whichever the transport selects.
pub(crate) struct ReplicaIngress {
    endpoint: Endpoint,
    inbound_tx: mpsc::Sender<ActiveSequenceEvent>,
    cancellation_token: CancellationToken,
}

impl ReplicaIngress {
    pub(crate) async fn start(self) {
        let transport_kind = self.endpoint.drt().default_event_transport_kind();
        let direct = direct_zmq::DirectZmqSequenceConfig::from_env();
        let ingress_result = if direct.should_use_direct(transport_kind) {
            direct_zmq::start(
                self.endpoint,
                self.inbound_tx,
                direct.rcvhwm,
                self.cancellation_token,
            )
            .await
            .map(|_supervisor| ())
        } else {
            RuntimeSequenceSubscriber::for_endpoint(&self.endpoint)
                .await
                .map(|subscriber| {
                    tokio::spawn(forward_replica_events(
                        subscriber,
                        self.inbound_tx,
                        self.cancellation_token,
                    ));
                })
        };
        if let Err(error) = ingress_result {
            tracing::warn!(
                %error,
                "active-sequence event ingress unavailable; continuing with response-side cleanup"
            );
        }
    }
}

async fn forward_replica_events(
    mut subscriber: RuntimeSequenceSubscriber,
    forward_tx: mpsc::Sender<ActiveSequenceEvent>,
    cancellation_token: CancellationToken,
) {
    loop {
        let next = tokio::select! {
            _ = cancellation_token.cancelled() => break,
            next = subscriber.next_event() => next,
        };
        match next {
            Some(Ok(event)) => {
                if forward_tx.send(event).await.is_err() {
                    break;
                }
            }
            Some(Err(error)) => {
                tracing::warn!(%error, "replica-sync subscriber error; continuing");
            }
            None => break,
        }
    }
}

/// Latest scheduler load of every registered worker rank, shared by the request path, catalog
/// membership, and the task that publishes it on [`SCHEDULER_LOAD_SUBJECT`].
#[derive(Default)]
pub(crate) struct SchedulerRankLoads(Mutex<HashMap<WorkerWithDpRank, SchedulerLoadSnapshot>>);

impl SchedulerRankLoads {
    pub(crate) fn record(&self, snapshot: SchedulerLoadSnapshot) {
        self.0.lock().insert(snapshot.worker, snapshot);
    }

    pub(crate) fn record_batch(&self, snapshots: &[SchedulerLoadSnapshot]) {
        self.0.lock().extend(
            snapshots
                .iter()
                .map(|snapshot| (snapshot.worker, *snapshot)),
        );
    }

    /// Start publishing `worker` as idle. Membership is applied after the scheduler admits the
    /// rank, so a request routed in between may already have recorded real load; keep it.
    pub(crate) fn register(&self, worker: WorkerWithDpRank) {
        self.0
            .lock()
            .entry(worker)
            .or_insert(SchedulerLoadSnapshot {
                worker,
                active_decode_blocks: 0,
                active_prefill_tokens: 0,
            });
    }

    pub(crate) fn remove(&self, worker: &WorkerWithDpRank) {
        self.0.lock().remove(worker);
    }

    pub(crate) fn snapshots(&self) -> Vec<SchedulerLoadSnapshot> {
        self.0.lock().values().copied().collect()
    }
}

/// Publish every rank in the returned [`SchedulerRankLoads`] once per
/// [`SCHEDULER_LOAD_PUBLISH_INTERVAL`] until `cancellation_token` fires.
///
/// With replica sync, every router on `endpoint` reports the same requests, so they form one
/// replica group; otherwise this router's view is standalone and consumers sum it with others.
pub(crate) fn start_scheduler_load_publisher(
    endpoint: &Endpoint,
    replica_sync: bool,
    cancellation_token: CancellationToken,
) -> Arc<SchedulerRankLoads> {
    let group = if replica_sync {
        SchedulerGroup::ReplicaGroup {
            group_id: endpoint.id().to_string(),
        }
    } else {
        // Not the DRT connection id: one process may host several routers for the same
        // endpoint, and their independent views must be summed.
        SchedulerGroup::Standalone {
            scheduler_id: rand::random(),
        }
    };
    let loads = Arc::new(SchedulerRankLoads::default());
    let endpoint = endpoint.clone();
    tokio::spawn(run_scheduler_load_publisher(
        move || {
            let endpoint = endpoint.clone();
            async move { EventPublisher::for_endpoint(&endpoint, SCHEDULER_LOAD_SUBJECT).await }
        },
        Arc::clone(&loads),
        group,
        cancellation_token,
    ));
    loads
}

trait SchedulerLoadPublisher: Send + Sync {
    fn publish_load(&self, load: &SchedulerLoad) -> impl Future<Output = Result<()>> + Send;
}

impl SchedulerLoadPublisher for EventPublisher {
    async fn publish_load(&self, load: &SchedulerLoad) -> Result<()> {
        self.publish(load).await
    }
}

/// Router startup never waits on the event plane: `connect` is retried with capped exponential
/// backoff, and a publish failure discards the publisher and connects a new one.
async fn run_scheduler_load_publisher<P, C, F>(
    mut connect: C,
    loads: Arc<SchedulerRankLoads>,
    group: SchedulerGroup,
    cancellation_token: CancellationToken,
) where
    P: SchedulerLoadPublisher,
    C: FnMut() -> F,
    F: Future<Output = Result<P>>,
{
    let mut publish_tick = tokio::time::interval(SCHEDULER_LOAD_PUBLISH_INTERVAL);
    publish_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut retry = Backoff::new(
        SCHEDULER_LOAD_CONNECT_RETRY_INITIAL,
        SCHEDULER_LOAD_CONNECT_RETRY_MAX,
    );
    let mut failures = FailureStreak::default();

    'connect: loop {
        let connected = tokio::select! {
            _ = cancellation_token.cancelled() => return,
            connected = connect() => connected,
        };
        let publisher = match connected {
            Ok(publisher) => publisher,
            Err(error) => {
                report_scheduler_load_failure(&mut failures, "connect", &error);
                tokio::select! {
                    _ = cancellation_token.cancelled() => return,
                    _ = tokio::time::sleep(retry.next_delay()) => {}
                }
                continue;
            }
        };
        retry.reset();

        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => return,
                _ = publish_tick.tick() => {}
            }
            for snapshot in loads.snapshots() {
                let load = SchedulerLoad {
                    worker_id: snapshot.worker.worker_id,
                    dp_rank: snapshot.worker.dp_rank,
                    active_decode_blocks: snapshot.active_decode_blocks,
                    active_prefill_tokens: snapshot.active_prefill_tokens,
                    group: group.clone(),
                };
                let published = tokio::select! {
                    _ = cancellation_token.cancelled() => return,
                    published = publisher.publish_load(&load) => published,
                };
                if let Err(error) = published {
                    report_scheduler_load_failure(&mut failures, "publish", &error);
                    continue 'connect;
                }
            }
            if let Some(failures) = failures.recover() {
                tracing::info!(failures, "Scheduler load publishing recovered");
            }
        }
    }
}

fn report_scheduler_load_failure(
    failures: &mut FailureStreak,
    operation: &str,
    error: &anyhow::Error,
) {
    if failures.fail() {
        tracing::warn!(
            operation,
            %error,
            "Scheduler load publishing failed; retrying in the background"
        );
    } else {
        tracing::debug!(operation, %error, failures = failures.failures(), "Scheduler load publishing still failing");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::ActiveSequenceEventData;
    use tokio::time::Instant;

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
            ActiveSequenceEventPublisher::channel(3, CancellationToken::new());
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
            ActiveSequenceEventPublisher::channel(1, CancellationToken::new());
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
    fn active_sequence_publish_sender_classifies_closed_queue_by_cancellation() {
        let cancellation_token = CancellationToken::new();
        let (sender, event_rx) =
            ActiveSequenceEventPublisher::channel(1, cancellation_token.clone());
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
    #[serial_test::serial]
    async fn worker_completion_ingress_runs_without_router_replica_sync() -> Result<()> {
        let runtime = dynamo_runtime::Runtime::from_current()?;
        let distributed = dynamo_runtime::DistributedRuntime::new(
            runtime,
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await?;
        let endpoint = distributed
            .namespace(format!(
                "worker-completion-ingress-{}",
                uuid::Uuid::new_v4()
            ))?
            .component("workers")?
            .endpoint("generate");
        let cancel = CancellationToken::new();
        let (mut channels, ingress) =
            host_replica_channels(&endpoint, 99, false, cancel.child_token()).await?;
        assert!(channels.outbound.is_none());
        ingress.start().await;

        let publisher = ActiveSequenceEventPublisher::for_endpoint(&endpoint, 16).await?;
        let received = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                publisher.mark_prefill_completed("worker-origin-mark".to_string(), 42, 0)?;
                tokio::select! {
                    event = channels.inbound_rx.recv() => return Ok::<_, anyhow::Error>(event),
                    _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {}
                }
            }
        })
        .await??
        .expect("inbound channel stays open");
        assert_eq!(received.request_id, "worker-origin-mark");
        assert!(matches!(
            received.data,
            ActiveSequenceEventData::MarkPrefillCompleted
        ));

        drop(publisher);
        cancel.cancel();
        distributed.shutdown();
        Ok(())
    }

    /// Replica sync is scoped to the endpoint: an ingress on endpoint B never
    /// sees events published on endpoint A, even once both planes are live.
    #[tokio::test]
    #[serial_test::serial]
    async fn active_sequence_replica_sync_isolated_by_endpoint() -> Result<()> {
        let runtime = dynamo_runtime::Runtime::from_current()?;
        let distributed = dynamo_runtime::DistributedRuntime::new(
            runtime,
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await?;
        let component = distributed
            .namespace(format!(
                "active-sequence-endpoint-isolation-{}",
                uuid::Uuid::new_v4()
            ))?
            .component("workers")?;
        let endpoint_a = component.endpoint("generate-a");
        let endpoint_b = component.endpoint("generate-b");
        let cancel = CancellationToken::new();
        let (mut channels_a, ingress_a) =
            host_replica_channels(&endpoint_a, 1, false, cancel.child_token()).await?;
        let (mut channels_b, ingress_b) =
            host_replica_channels(&endpoint_b, 2, false, cancel.child_token()).await?;
        ingress_a.start().await;
        ingress_b.start().await;

        // Publish on both planes until each ingress has received something, so
        // B's silence about A cannot be blamed on B's subscription not being up.
        let publisher_a = ActiveSequenceEventPublisher::for_endpoint(&endpoint_a, 16).await?;
        let publisher_b = ActiveSequenceEventPublisher::for_endpoint(&endpoint_b, 16).await?;
        let mut received_a = Vec::new();
        let mut received_b = Vec::new();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while received_a.is_empty() || received_b.is_empty() {
                publisher_a.mark_prefill_completed("endpoint-a-mark".to_string(), 42, 0)?;
                publisher_b.mark_prefill_completed("endpoint-b-mark".to_string(), 42, 0)?;
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                while let Ok(event) = channels_a.inbound_rx.try_recv() {
                    received_a.push(event.request_id);
                }
                while let Ok(event) = channels_b.inbound_rx.try_recv() {
                    received_b.push(event.request_id);
                }
            }
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        // Give late A events 250ms of silence to leak into B before judging.
        while let Ok(Some(event)) = tokio::time::timeout(
            std::time::Duration::from_millis(250),
            channels_b.inbound_rx.recv(),
        )
        .await
        {
            received_b.push(event.request_id);
        }
        assert!(
            received_a.iter().all(|id| id == "endpoint-a-mark"),
            "endpoint A received endpoint B sequence state: {received_a:?}"
        );
        assert!(
            received_b.iter().all(|id| id == "endpoint-b-mark"),
            "endpoint B received endpoint A sequence state: {received_b:?}"
        );

        drop((publisher_a, publisher_b));
        cancel.cancel();
        distributed.shutdown();
        Ok(())
    }

    fn load_snapshot(worker_id: u64, blocks: u64) -> SchedulerLoadSnapshot {
        SchedulerLoadSnapshot {
            worker: WorkerWithDpRank::new(worker_id, 0),
            active_decode_blocks: blocks,
            active_prefill_tokens: 0,
        }
    }

    fn sorted_snapshots(loads: &SchedulerRankLoads) -> Vec<SchedulerLoadSnapshot> {
        let mut snapshots = loads.snapshots();
        snapshots.sort_by_key(|snapshot| snapshot.worker.worker_id);
        snapshots
    }

    #[test]
    fn registration_keeps_load_recorded_before_it_and_removal_stops_publishing() {
        let loads = SchedulerRankLoads::default();
        let busy = load_snapshot(1, 10);
        let idle = load_snapshot(2, 0);

        loads.record(busy);
        loads.register(busy.worker);
        loads.register(idle.worker);
        assert_eq!(sorted_snapshots(&loads), [busy, idle]);

        loads.record_batch(&[load_snapshot(1, 4), load_snapshot(2, 6)]);
        loads.remove(&busy.worker);
        assert_eq!(sorted_snapshots(&loads), [load_snapshot(2, 6)]);
    }

    /// Records published loads; fails the publish calls whose 0-based index is in `fail_at`.
    struct RecordingLoadPublisher {
        published_tx: mpsc::UnboundedSender<SchedulerLoad>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
        fail_at: &'static [usize],
    }

    impl SchedulerLoadPublisher for RecordingLoadPublisher {
        async fn publish_load(&self, load: &SchedulerLoad) -> Result<()> {
            let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            anyhow::ensure!(!self.fail_at.contains(&call), "synthetic publish failure");
            self.published_tx.send(load.clone()).unwrap();
            Ok(())
        }
    }

    async fn next_loads(
        published_rx: &mut mpsc::UnboundedReceiver<SchedulerLoad>,
        count: usize,
    ) -> Vec<SchedulerLoad> {
        let mut published = Vec::with_capacity(count);
        for _ in 0..count {
            let load =
                tokio::time::timeout(10 * SCHEDULER_LOAD_PUBLISH_INTERVAL, published_rx.recv())
                    .await
                    .expect("scheduler load was not published");
            published.push(load.unwrap());
        }
        published.sort_by_key(|load| load.worker_id);
        published
    }

    #[tokio::test(start_paused = true)]
    async fn scheduler_load_publisher_publishes_every_rank_each_tick_and_reconnects() {
        let group = SchedulerGroup::Standalone { scheduler_id: 9 };
        let expected = |loads: &[SchedulerLoadSnapshot]| -> Vec<SchedulerLoad> {
            loads
                .iter()
                .map(|snapshot| SchedulerLoad {
                    worker_id: snapshot.worker.worker_id,
                    dp_rank: snapshot.worker.dp_rank,
                    active_decode_blocks: snapshot.active_decode_blocks,
                    active_prefill_tokens: snapshot.active_prefill_tokens,
                    group: group.clone(),
                })
                .collect()
        };
        let loads = Arc::new(SchedulerRankLoads::default());
        loads.record_batch(&[load_snapshot(1, 10), load_snapshot(2, 20)]);
        let (published_tx, mut published_rx) = mpsc::unbounded_channel();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let connects = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cancel = CancellationToken::new();
        let start = Instant::now();
        let task = tokio::spawn(run_scheduler_load_publisher(
            {
                let connects = Arc::clone(&connects);
                let calls = Arc::clone(&calls);
                move || {
                    let attempt = connects.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let publisher = RecordingLoadPublisher {
                        published_tx: published_tx.clone(),
                        calls: Arc::clone(&calls),
                        // The second publish of the second tick fails.
                        fail_at: &[3],
                    };
                    // The first connection attempt fails before any publisher exists.
                    async move {
                        anyhow::ensure!(attempt > 0, "synthetic connect failure");
                        Ok(publisher)
                    }
                }
            },
            Arc::clone(&loads),
            group.clone(),
            cancel.clone(),
        ));

        // The first tick waits out the connect retry, then publishes every rank.
        assert_eq!(
            next_loads(&mut published_rx, 2).await,
            expected(&[load_snapshot(1, 10), load_snapshot(2, 20)])
        );
        assert_eq!(start.elapsed(), SCHEDULER_LOAD_CONNECT_RETRY_INITIAL);
        assert_eq!(connects.load(std::sync::atomic::Ordering::SeqCst), 2);

        // The next tick republishes the unchanged map, but its second publish fails.
        assert_eq!(next_loads(&mut published_rx, 1).await.len(), 1);

        // The rebuilt publisher resumes on the following tick with the latest load of every rank.
        loads.record(load_snapshot(2, 0));
        assert_eq!(
            next_loads(&mut published_rx, 2).await,
            expected(&[load_snapshot(1, 10), load_snapshot(2, 0)])
        );
        assert_eq!(start.elapsed(), 2 * SCHEDULER_LOAD_PUBLISH_INTERVAL);
        assert_eq!(connects.load(std::sync::atomic::Ordering::SeqCst), 3);

        cancel.cancel();
        task.await.unwrap();
    }
}
