// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-specific glue for [`ActiveSequencesMultiWorker`].
//!
//! This module provides the concrete [`SequencePublisher`] and [`SequenceSubscriber`]
//! implementations that wire the runtime-agnostic business logic (in `dynamo_kv_router`)
//! to the configured event transport and Prometheus metrics.

pub use dynamo_kv_router::multi_worker_sequence::{
    ActiveSequencesMultiWorker, ReplicaRequestLeaseObserver, SchedulerLoadSnapshot, SequenceError,
    SequencePublishQueueError, SequencePublisher, SequenceRequest, SequenceSubscriber,
};
use dynamo_kv_router::protocols::{
    ActiveSequenceEvent, ActiveSequenceEventBatch, MAX_REPLICA_BATCH_DURATION,
    MAX_REPLICA_BATCH_EVENTS, WorkerWithDpRank,
};
pub use dynamo_kv_router::sequence::{ActiveSequences, RequestId};

use anyhow::Result;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::{
    EventPublisher, EventSubscriber, EventTransportKind, TypedEventSubscriber,
};
use std::collections::VecDeque;
use std::future::Future;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::kv_router::ACTIVE_SEQUENCES_SUBJECT;
#[cfg(test)]
#[cfg(test)]
use dynamo_runtime::transports::event_plane::MsgpackCodec;

// Match the existing standalone replica-sync queue. Lifecycle callers enqueue without awaiting;
// if the queue is full, the newest event is dropped without blocking the local mutation.
const REPLICA_EVENT_CHANNEL_CAPACITY: usize = 100_000;

/// How active-sequence events are framed on the wire for a transport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[cfg(test)]
fn active_sequence_event_channel(
    enabled: bool,
    capacity: usize,
    cancellation_token: &CancellationToken,
) -> Option<(
    ActiveSequenceEventPublisher,
    mpsc::Receiver<ActiveSequenceEvent>,
)> {
    enabled
        .then(|| ActiveSequenceEventPublisher::channel(capacity, cancellation_token.child_token()))
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

/// Replica-sync channels for an embedded selection partition, carried over the
/// runtime event plane exactly like the runtime scheduler's replica sync:
/// outbound events are published on `ACTIVE_SEQUENCES_SUBJECT`, peer events
/// are subscribed from it and forwarded to the partition.
pub(crate) async fn host_replica_channels(
    endpoint: &Endpoint,
    router_id: u64,
    cancellation_token: CancellationToken,
) -> Result<dynamo_kv_router::services::selection::HostReplicaChannels> {
    let transport_kind = endpoint.drt().default_event_transport_kind();
    let (outbound, outbound_rx) = mpsc::channel(REPLICA_EVENT_CHANNEL_CAPACITY);
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
                outbound_rx,
                cancellation_token.clone(),
            ));
        }
        ActiveSequenceEventWireFormat::Batch => {
            tokio::spawn(run_replica_batch_publisher(
                event_publisher,
                outbound_rx,
                cancellation_token.clone(),
            ));
        }
    }

    let (inbound_tx, inbound_rx) = mpsc::channel(REPLICA_EVENT_CHANNEL_CAPACITY);
    let mut subscriber = RuntimeSequenceSubscriber::for_endpoint(endpoint).await?;
    let forward_tx = inbound_tx.clone();
    tokio::spawn(async move {
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
    });
    Ok(dynamo_kv_router::services::selection::HostReplicaChannels {
        outbound,
        inbound_tx,
        inbound_rx,
        process_id: router_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::ActiveSequenceEventData;
    use std::sync::Arc;
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
    fn active_sequence_publish_channel_is_absent_when_replica_sync_disabled() {
        assert!(active_sequence_event_channel(false, 1, &CancellationToken::new()).is_none());
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
}
