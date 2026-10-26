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
use dynamo_kv_router::services::selection::{ModelLoadResponse, SelectionService};

use anyhow::Result;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::{
    EventPublisher, EventSubscriber, EventTransportKind, TypedEventSubscriber,
};
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
/// One complete rank batch per interval; the relay derives freshness from this cadence.
pub(crate) const SCHEDULER_LOAD_PUBLISH_INTERVAL: Duration = Duration::from_secs(1);
const SCHEDULER_LOAD_RETRY_INITIAL: Duration = Duration::from_millis(100);
const SCHEDULER_LOAD_RETRY_MAX: Duration = Duration::from_secs(5);

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
    let (event_sender, local_publisher_id) = if publishes_outbound {
        let (event_sender, event_rx) = mpsc::channel(REPLICA_EVENT_CHANNEL_CAPACITY);
        let publisher_cancellation_token = cancellation_token.clone();
        let event_publisher = EventPublisher::for_endpoint_with_transport(
            endpoint,
            ACTIVE_SEQUENCES_SUBJECT,
            transport_kind,
        )
        .await?;
        let local_publisher_id = event_publisher.publisher_id();
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
        (Some(event_sender), Some(local_publisher_id))
    } else {
        (None, None)
    };

    let (inbound_tx, inbound_rx) = mpsc::channel(REPLICA_EVENT_CHANNEL_CAPACITY);
    let ingress = ReplicaIngress {
        endpoint: endpoint.clone(),
        inbound_tx: inbound_tx.clone(),
        local_publisher_id,
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
    local_publisher_id: Option<u64>,
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
                self.local_publisher_id,
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

/// Publishes scheduler snapshots on the secondary runtime when serving-load reporting is enabled.
pub(crate) fn start_scheduler_load_publisher(
    endpoint: &Endpoint,
    service: Arc<SelectionService>,
    replica_sync: bool,
    cancellation_token: CancellationToken,
) {
    if !crate::kv_dc_relay::serving_load_publishing_enabled() {
        return;
    }
    let group = if replica_sync {
        SchedulerGroup::ReplicaGroup {
            group_id: endpoint.id().to_string(),
        }
    } else {
        // One process may host independent routers for the same endpoint; DRT identity
        // would merge their disjoint request sets.
        SchedulerGroup::Standalone {
            scheduler_id: rand::random(),
        }
    };
    endpoint
        .drt()
        .runtime()
        .secondary()
        .spawn(run_scheduler_load_publisher(
            endpoint.clone(),
            service,
            group,
            cancellation_token,
        ));
}

fn scheduler_load_batch(
    models: Vec<ModelLoadResponse>,
    group: &SchedulerGroup,
) -> Vec<SchedulerLoad> {
    let mut ranks = HashMap::<WorkerWithDpRank, SchedulerLoad>::new();
    for load in models.into_iter().flat_map(|model| model.loads) {
        let rank = WorkerWithDpRank::new(load.worker_id, load.dp_rank);
        ranks
            .entry(rank)
            .and_modify(|current| {
                // Each model/routing-group partition tracks disjoint requests.
                current.active_decode_blocks = current
                    .active_decode_blocks
                    .saturating_add(load.potential_decode_blocks as u64);
                current.active_prefill_tokens = current
                    .active_prefill_tokens
                    .saturating_add(load.potential_prefill_tokens as u64);
            })
            .or_insert_with(|| SchedulerLoad {
                worker_id: load.worker_id,
                dp_rank: load.dp_rank,
                // loads() supplies no hypothetical request, so these are active values.
                active_decode_blocks: load.potential_decode_blocks as u64,
                active_prefill_tokens: load.potential_prefill_tokens as u64,
                group: group.clone(),
            });
    }
    ranks.into_values().collect()
}

async fn run_scheduler_load_publisher(
    endpoint: Endpoint,
    service: Arc<SelectionService>,
    group: SchedulerGroup,
    cancellation_token: CancellationToken,
) {
    // Publish on the first tick; retry pacing matches the frontend load publisher.
    let mut publish_tick = tokio::time::interval(SCHEDULER_LOAD_PUBLISH_INTERVAL);
    publish_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut retry = Backoff::new(SCHEDULER_LOAD_RETRY_INITIAL, SCHEDULER_LOAD_RETRY_MAX);
    let mut failures = FailureStreak::default();

    loop {
        let connected = tokio::select! {
            _ = cancellation_token.cancelled() => return,
            connected = EventPublisher::for_endpoint(&endpoint, SCHEDULER_LOAD_SUBJECT) => connected,
        };
        let error = match connected {
            Ok(publisher) => loop {
                tokio::select! {
                    _ = cancellation_token.cancelled() => return,
                    _ = publish_tick.tick() => {}
                }
                let loads = scheduler_load_batch(service.loads(None, None), &group);
                let published = tokio::select! {
                    _ = cancellation_token.cancelled() => return,
                    published = publisher.publish(&loads) => published,
                };
                match published {
                    Ok(()) => {
                        retry.reset();
                        if let Some(failures) = failures.recover() {
                            tracing::info!(failures, "Scheduler load publishing recovered");
                        }
                    }
                    Err(error) => break error,
                }
            },
            Err(error) => error,
        };
        let delay = retry.next_delay();
        if failures.fail() {
            tracing::warn!(%error, retry_ms = delay.as_millis(), "Scheduler load publishing failed; reconnecting");
        } else {
            tracing::debug!(%error, failures = failures.failures(), retry_ms = delay.as_millis(), "Scheduler load publishing still failing");
        }
        tokio::select! {
            _ = cancellation_token.cancelled() => return,
            _ = tokio::time::sleep(delay) => {}
        }
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

    #[tokio::test]
    #[serial_test::serial]
    async fn replica_ingress_skips_own_outbound_publisher() -> Result<()> {
        let runtime = dynamo_runtime::Runtime::from_current()?;
        let distributed = dynamo_runtime::DistributedRuntime::new(
            runtime,
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await?;
        let endpoint = distributed
            .namespace(format!("replica-ingress-self-{}", uuid::Uuid::new_v4()))?
            .component("workers")?
            .endpoint("generate");
        let cancel = CancellationToken::new();
        let (mut channels, ingress) =
            host_replica_channels(&endpoint, 1, true, cancel.child_token()).await?;
        let outbound = channels
            .outbound
            .clone()
            .expect("publishing replica has an outbound sender");
        ingress.start().await;

        let peer = ActiveSequenceEventPublisher::for_endpoint(&endpoint, 16).await?;
        let mut received = Vec::new();
        // Keep sending own events after the peer is live, so a late own
        // subscription would still surface.
        let mut rounds_after_peer = 0;
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while rounds_after_peer < 5 {
                outbound.try_send(ActiveSequenceEvent {
                    request_id: "own-event".to_string(),
                    worker: WorkerWithDpRank::new(42, 0),
                    data: ActiveSequenceEventData::MarkPrefillCompleted,
                    router_id: 1,
                    lora_name: None,
                })?;
                peer.mark_prefill_completed("peer-mark".to_string(), 42, 0)?;
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                while let Ok(event) = channels.inbound_rx.try_recv() {
                    received.push(event.request_id);
                }
                if received.iter().any(|id| id == "peer-mark") {
                    rounds_after_peer += 1;
                }
            }
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        assert!(
            received.iter().all(|id| id == "peer-mark"),
            "replica received its own outbound events: {received:?}"
        );

        drop(peer);
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

    #[test]
    fn scheduler_batch_sums_disjoint_partition_loads_per_rank() {
        let model = |name: &str, worker_id, blocks, tokens| ModelLoadResponse {
            model_name: name.into(),
            routing_group: "default".into(),
            loads: vec![dynamo_kv_router::protocols::PotentialLoad {
                worker_id,
                dp_rank: 0,
                potential_decode_blocks: blocks,
                potential_prefill_tokens: tokens,
                active_requests: 0,
            }],
            pending_count: 0,
            pending_isl_tokens: 0,
        };
        let group = SchedulerGroup::Standalone { scheduler_id: 9 };
        let mut batch = scheduler_load_batch(
            vec![
                model("a", 1, 3, 7),
                model("b", 1, 5, 4),
                model("b", 2, 2, 1),
            ],
            &group,
        );
        batch.sort_by_key(|load| load.worker_id);
        assert_eq!(batch.len(), 2);
        assert_eq!(
            (
                batch[0].active_decode_blocks,
                batch[0].active_prefill_tokens
            ),
            (8, 11)
        );
        assert_eq!((batch[1].worker_id, batch[1].active_decode_blocks), (2, 2));
        assert!(batch.iter().all(|load| load.group == group));
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn scheduler_publisher_batches_live_loads_and_removes_departed_ranks() -> Result<()> {
        use dynamo_kv_router::services::selection::{SelectionServiceBuilder, WorkerRequest};

        let distributed = dynamo_runtime::DistributedRuntime::new(
            dynamo_runtime::Runtime::from_current()?,
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await?;
        let endpoint = distributed
            .namespace(format!("scheduler-load-{}", uuid::Uuid::new_v4()))?
            .component("workers")?
            .endpoint("generate");
        let service = Arc::new(
            SelectionServiceBuilder::new(
                dynamo_kv_router::config::KvRouterConfig {
                    use_kv_events: false,
                    router_queue_threshold: None,
                    ..Default::default()
                },
                dynamo_kv_router::WorkerType::Aggregated,
                dynamo_custom_policy_builtin::default_registry(),
            )
            .indexer_threads(1)
            .build()
            .await?,
        );
        for worker_id in [42, 43] {
            service
                .upsert_worker(WorkerRequest {
                    worker_id,
                    endpoint: Some("http://127.0.0.1:8000".into()),
                    block_size: Some(4),
                    data_parallel_start_rank: Some(0),
                    data_parallel_size: Some(2),
                    max_num_batched_tokens: Some(1024),
                    total_kv_blocks: Some(100),
                    is_eagle: Some(false),
                    ..Default::default()
                })
                .await?;
        }
        // Explicit reservations need a hint to account for prefill tokens.
        service
            .create_reservation(serde_json::from_value(serde_json::json!({
                "selection_id": "busy", "worker_id": 42, "dp_rank": 0,
                "token_ids": [1, 2, 3, 4], "track_prefill_tokens": true,
                "effective_prefill_tokens": 4
            }))?)
            .await?;
        let mut subscriber = EventSubscriber::for_endpoint(&endpoint, SCHEDULER_LOAD_SUBJECT)
            .await?
            .typed::<Vec<SchedulerLoad>>();
        let cancel = CancellationToken::new();
        let task = tokio::spawn(run_scheduler_load_publisher(
            endpoint,
            Arc::clone(&service),
            SchedulerGroup::Standalone { scheduler_id: 9 },
            cancel.clone(),
        ));
        let (_, first) = tokio::time::timeout(Duration::from_secs(5), subscriber.next())
            .await?
            .expect("scheduler publisher stopped")?;
        assert_eq!(first.len(), 4);
        let busy = first
            .iter()
            .find(|load| load.worker_id == 42 && load.dp_rank == 0)
            .unwrap();
        assert_eq!(busy.active_decode_blocks, 1);
        assert_eq!(busy.active_prefill_tokens, 4);
        service.free_reservation("busy").await?;
        service.delete_worker(42).await?;
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let (_, loads) = subscriber
                    .next()
                    .await
                    .expect("scheduler publisher stopped")?;
                if loads.iter().all(|load| load.worker_id == 43) {
                    assert_eq!(loads.len(), 2);
                    assert!(
                        loads.iter().all(|load| load.active_decode_blocks == 0
                            && load.active_prefill_tokens == 0)
                    );
                    return Ok::<_, anyhow::Error>(());
                }
            }
        })
        .await??;
        cancel.cancel();
        task.await?;
        distributed.shutdown();
        Ok(())
    }
}
