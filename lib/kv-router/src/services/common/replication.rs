// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replica-sync plumbing the selection kernel and the slot-tracker service drive:
//! scoped sequence events, the per-partition publisher and subscriber, the inbox
//! that delivers peer events into a partition, the session-affinity sink, and the
//! host channels an embedding host supplies. Transport-free: the ZMQ mesh that
//! carries these events between routers is `replica_sync`.

use std::sync::Arc;
use std::task::{Context, Poll};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::identity::{RoutingPartitionId, RoutingPartitionRef};
use crate::protocols::{ActiveSequenceEvent, WorkerWithDpRank};
use crate::sequences::multi_worker::RateLimitedFailureLog;
use crate::sequences::{
    LocalWorkerLoad, SchedulerLoadSnapshot, SequencePublishQueueError, SequencePublisher,
    SequenceSubscriber,
};
#[cfg(feature = "standalone-selection")]
use dynamo_session_affinity::{AffinityReplicaSink, AffinityTarget, AffinityVersion};

pub(crate) const REPLICA_EVENT_CHANNEL_CAPACITY: usize = 100_000;

pub(crate) const AFFINITY_EVENT_CHANNEL_CAPACITY: usize = 4_096;

pub use dynamo_session_affinity::wire::AffinityBindingEvent;

#[cfg(feature = "standalone-selection")]
/// Publishes bindings from a [`super::super::selection::affinity::SessionAffinity`]
/// into a transport channel. Best effort: a full channel drops the update.
pub(crate) struct AffinityMeshSink {
    partition: RoutingPartitionId,
    tx: mpsc::Sender<AffinityBindingEvent>,
    cancel_token: CancellationToken,
    full_drops: parking_lot::Mutex<RateLimitedFailureLog>,
    unexpected_closed_drops: parking_lot::Mutex<RateLimitedFailureLog>,
}

#[cfg(feature = "standalone-selection")]
impl AffinityMeshSink {
    pub(crate) fn new(
        partition: RoutingPartitionId,
        tx: mpsc::Sender<AffinityBindingEvent>,
        cancel_token: CancellationToken,
    ) -> Self {
        Self {
            partition,
            tx,
            cancel_token,
            full_drops: parking_lot::Mutex::default(),
            unexpected_closed_drops: parking_lot::Mutex::default(),
        }
    }
}

#[cfg(feature = "standalone-selection")]
impl AffinityReplicaSink for AffinityMeshSink {
    fn publish(&self, session_id: &str, target: AffinityTarget, version: AffinityVersion) {
        let update = AffinityBindingEvent {
            partition: self.partition.clone(),
            session_id: session_id.to_string(),
            worker_id: target.worker_id,
            dp_rank: target.dp_rank,
            sequence: version.sequence,
            writer_id: version.writer_id,
        };
        match self.tx.try_send(update) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                let Some(dropped_bindings) = self.full_drops.lock().record(Instant::now()) else {
                    return;
                };
                tracing::warn!(
                    partition = %self.partition,
                    capacity = self.tx.max_capacity(),
                    dropped_bindings_since_last_log = dropped_bindings,
                    "Session affinity replication channel full; dropping binding"
                );
            }
            Err(mpsc::error::TrySendError::Closed(_)) if self.cancel_token.is_cancelled() => {
                tracing::trace!("dropping session affinity replica update; channel closed");
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                let Some(dropped_bindings) =
                    self.unexpected_closed_drops.lock().record(Instant::now())
                else {
                    return;
                };
                tracing::warn!(
                    partition = %self.partition,
                    dropped_bindings_since_last_log = dropped_bindings,
                    "Session affinity replication channel closed unexpectedly; dropping binding"
                );
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ScopedReplicaEvent {
    pub model_name: String,
    pub routing_group: String,
    pub block_size: u32,
    pub event: ActiveSequenceEvent,
}

impl ScopedReplicaEvent {
    pub(crate) fn partition_ref(&self) -> RoutingPartitionRef<'_> {
        RoutingPartitionRef::new(&self.model_name, &self.routing_group)
    }

    pub(crate) fn into_parts(self) -> (RoutingPartitionId, u32, ActiveSequenceEvent) {
        (
            RoutingPartitionId::new(self.model_name, self.routing_group),
            self.block_size,
            self.event,
        )
    }
}

pub(crate) type ReplicaEventSender = mpsc::Sender<ScopedReplicaEvent>;

#[derive(Debug, Clone)]
pub struct ReplicaSyncConfig {
    process_id: u64,
    outbound_tx: ReplicaEventSender,
    affinity_tx: Option<mpsc::Sender<AffinityBindingEvent>>,
    cancel_token: CancellationToken,
}

impl ReplicaSyncConfig {
    pub(crate) fn new(
        process_id: u64,
        outbound_tx: ReplicaEventSender,
        cancel_token: CancellationToken,
    ) -> Self {
        Self {
            process_id,
            outbound_tx,
            affinity_tx: None,
            cancel_token,
        }
    }

    pub(crate) fn with_affinity_publisher(
        mut self,
        tx: mpsc::Sender<AffinityBindingEvent>,
    ) -> Self {
        self.affinity_tx = Some(tx);
        self
    }

    pub(crate) fn process_id(&self) -> u64 {
        self.process_id
    }

    /// The token that stops every publisher and subscriber built from this config.
    pub(crate) fn cancel_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    #[cfg(feature = "standalone-selection")]
    /// Sink that publishes session bindings into the mesh, when this runtime
    /// carries them.
    pub(crate) fn affinity_sink(
        &self,
        partition: &RoutingPartitionId,
    ) -> Option<Arc<dyn AffinityReplicaSink>> {
        self.affinity_tx.clone().map(|tx| {
            Arc::new(AffinityMeshSink::new(
                partition.clone(),
                tx,
                self.cancel_token.clone(),
            )) as Arc<dyn AffinityReplicaSink>
        })
    }

    pub(crate) fn is_self_event(&self, event: &ActiveSequenceEvent) -> bool {
        event.router_id == self.process_id
    }
}

pub(crate) struct ScopedReplicaSync {
    pub publisher: ScopedSequencePublisher,
    pub enabled: bool,
    pub process_id: u64,
    pub channel: Option<(mpsc::Sender<ActiveSequenceEvent>, ChannelSequenceSubscriber)>,
}

/// Receives the scheduler-owned load snapshots a partition's active-sequence
/// tracker publishes (per worker: active decode blocks, active prefill
/// tokens). An embedding host uses it to drive overload detection from the
/// same numbers the scheduler books against.
pub trait SchedulerLoadSink: Send + Sync {
    fn publish(&self, snapshot: SchedulerLoadSnapshot);

    /// Per-worker load after any local mutation. The sink owns the metric
    /// label so it matches the host's cleanup path.
    fn observe_local_load(&self, _worker: &WorkerWithDpRank, _load: LocalWorkerLoad) {}

    fn publish_batch(&self, snapshots: Vec<SchedulerLoadSnapshot>) {
        for snapshot in snapshots {
            self.publish(snapshot);
        }
    }
}

/// Observes the partition's inbound replica funnel once per drain batch, after
/// the batch is applied and flushed. `queue_depth` is the number of events
/// still waiting in the inbound channel at that moment; `applied` is the batch
/// size. Called from the apply task, so implementations must be cheap.
pub trait ReplicaIngressObserver: Send + Sync {
    fn observe_drain(&self, queue_depth: usize, applied: usize);
}

/// Replica-sync plumbing an embedding host supplies for one partition when it
/// carries active-sequence events over its own transport (for example the
/// Dynamo runtime event plane) instead of the service's ZMQ peer mesh:
/// events the partition emits go to `outbound`; events from peer replicas
/// arrive on `inbound_rx` (the host also holds `inbound_tx`).
///
/// `outbound: None` is ingress-only: the partition publishes nothing and the
/// tracker keeps only worker-origin completion marks from `inbound_rx`.
pub struct HostReplicaChannels {
    pub outbound: Option<mpsc::Sender<ActiveSequenceEvent>>,
    pub inbound_tx: mpsc::Sender<ActiveSequenceEvent>,
    pub inbound_rx: mpsc::Receiver<ActiveSequenceEvent>,
    /// This replica's id; events carrying it are ignored on receipt.
    pub process_id: u64,
    /// Optional drain-batch observer for the inbound funnel.
    pub ingress_observer: Option<Arc<dyn ReplicaIngressObserver>>,
}

#[cfg(feature = "standalone-selection")]
/// Per-partition factory for [`HostReplicaChannels`]; `None` disables replica
/// sync for that partition.
pub type HostReplicaSyncFactory =
    Arc<dyn Fn(&RoutingPartitionId) -> Option<HostReplicaChannels> + Send + Sync>;

#[derive(Clone)]
pub struct ScopedSequencePublisher {
    replica: Option<ScopedReplicaPublisher>,
    host_tx: Option<mpsc::Sender<ActiveSequenceEvent>>,
    load_sink: Option<Arc<dyn SchedulerLoadSink>>,
}

#[derive(Clone)]
struct ScopedReplicaPublisher {
    partition: Arc<RoutingPartitionId>,
    block_size: u32,
    tx: ReplicaEventSender,
    cancel_token: CancellationToken,
}

impl ScopedSequencePublisher {
    pub(crate) fn disabled() -> Self {
        Self {
            replica: None,
            host_tx: None,
            load_sink: None,
        }
    }

    pub(crate) fn host(outbound: mpsc::Sender<ActiveSequenceEvent>) -> Self {
        Self {
            replica: None,
            host_tx: Some(outbound),
            load_sink: None,
        }
    }

    #[cfg(feature = "standalone-selection")]
    pub(crate) fn with_load_sink(mut self, sink: Option<Arc<dyn SchedulerLoadSink>>) -> Self {
        self.load_sink = sink;
        self
    }

    pub(crate) fn enabled(
        partition: Arc<RoutingPartitionId>,
        block_size: u32,
        tx: ReplicaEventSender,
        cancel_token: CancellationToken,
    ) -> Self {
        Self {
            replica: Some(ScopedReplicaPublisher {
                partition,
                block_size,
                tx,
                cancel_token,
            }),
            host_tx: None,
            load_sink: None,
        }
    }
}

impl SequencePublisher for ScopedSequencePublisher {
    fn enqueue_event(&self, event: ActiveSequenceEvent) -> Result<()> {
        if let Some(host_tx) = &self.host_tx {
            return match host_tx.try_send(event) {
                Ok(()) => Ok(()),
                Err(mpsc::error::TrySendError::Full(event)) => Err(anyhow::Error::new(
                    SequencePublishQueueError::full(event, host_tx.max_capacity()),
                )
                .context("host replica publisher")),
                Err(mpsc::error::TrySendError::Closed(event)) => Err(anyhow::Error::new(
                    SequencePublishQueueError::closed(event, host_tx.max_capacity(), true),
                )
                .context("host replica publisher")),
            };
        }
        let Some(replica) = &self.replica else {
            return Ok(());
        };
        let envelope = ScopedReplicaEvent {
            model_name: replica.partition.model_name.clone(),
            routing_group: replica.partition.routing_group.clone(),
            block_size: replica.block_size,
            event,
        };
        match replica.tx.try_send(envelope) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(event)) => {
                let error = SequencePublishQueueError::full(event.event, replica.tx.max_capacity());
                Err(anyhow::Error::new(error).context(format!(
                    "replica publisher for model_name={} routing_group={}",
                    event.model_name, event.routing_group
                )))
            }
            Err(mpsc::error::TrySendError::Closed(event)) => {
                let error = SequencePublishQueueError::closed(
                    event.event,
                    replica.tx.max_capacity(),
                    replica.cancel_token.is_cancelled(),
                );
                Err(anyhow::Error::new(error).context(format!(
                    "replica publisher for model_name={} routing_group={}",
                    event.model_name, event.routing_group
                )))
            }
        }
    }

    fn publish_scheduler_load(&self, load: SchedulerLoadSnapshot) {
        if let Some(sink) = &self.load_sink {
            sink.publish(load);
        }
    }

    fn publish_scheduler_load_batch(&self, loads: Vec<SchedulerLoadSnapshot>) {
        if let Some(sink) = &self.load_sink {
            sink.publish_batch(loads);
        }
    }

    fn observe_load(&self, worker: &WorkerWithDpRank, _worker_type: &str, load: LocalWorkerLoad) {
        if let Some(sink) = &self.load_sink {
            sink.observe_local_load(worker, load);
        }
    }
}

pub(crate) struct ChannelSequenceSubscriber {
    rx: mpsc::Receiver<ActiveSequenceEvent>,
    observer: Option<Arc<dyn ReplicaIngressObserver>>,
}

impl ChannelSequenceSubscriber {
    pub(crate) fn new(rx: mpsc::Receiver<ActiveSequenceEvent>) -> Self {
        Self { rx, observer: None }
    }

    pub(crate) fn with_observer(
        rx: mpsc::Receiver<ActiveSequenceEvent>,
        observer: Option<Arc<dyn ReplicaIngressObserver>>,
    ) -> Self {
        Self { rx, observer }
    }
}

impl SequenceSubscriber for ChannelSequenceSubscriber {
    async fn next_event(&mut self) -> Option<Result<ActiveSequenceEvent>> {
        self.rx.recv().await.map(Ok)
    }

    fn poll_next_event(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<ActiveSequenceEvent>>> {
        self.rx.poll_recv(cx).map(|event| event.map(Ok))
    }

    fn record_drain(&mut self, applied: usize) {
        if let Some(observer) = &self.observer {
            observer.observe_drain(self.rx.len(), applied);
        }
    }
}

pub(crate) fn setup_scoped_replica_sync(
    config: Option<&ReplicaSyncConfig>,
    partition: &RoutingPartitionId,
    block_size: u32,
    host: Option<HostReplicaChannels>,
) -> ScopedReplicaSync {
    let Some(config) = config else {
        // No peer mesh: an embedding host may still carry replica events.
        if let Some(host) = host {
            let enabled = host.outbound.is_some();
            return ScopedReplicaSync {
                publisher: host
                    .outbound
                    .map(ScopedSequencePublisher::host)
                    .unwrap_or_else(ScopedSequencePublisher::disabled),
                enabled,
                process_id: host.process_id,
                channel: Some((
                    host.inbound_tx,
                    ChannelSequenceSubscriber::with_observer(
                        host.inbound_rx,
                        host.ingress_observer,
                    ),
                )),
            };
        }
        return ScopedReplicaSync {
            publisher: ScopedSequencePublisher::disabled(),
            enabled: false,
            process_id: 0,
            channel: None,
        };
    };

    let (replica_tx, replica_rx) = mpsc::channel(REPLICA_EVENT_CHANNEL_CAPACITY);
    ScopedReplicaSync {
        publisher: ScopedSequencePublisher::enabled(
            Arc::new(partition.clone()),
            block_size,
            config.outbound_tx.clone(),
            config.cancel_token.clone(),
        ),
        enabled: true,
        process_id: config.process_id,
        channel: Some((replica_tx, ChannelSequenceSubscriber::new(replica_rx))),
    }
}

/// Inbound side of one partition's replica channel: forwards mesh events to
/// the partition's tracker. Best effort; a full channel drops the event.
pub(crate) struct ReplicaInbox {
    tx: mpsc::Sender<ActiveSequenceEvent>,
    full_drops: parking_lot::Mutex<RateLimitedFailureLog>,
}

impl ReplicaInbox {
    pub(crate) fn new(tx: mpsc::Sender<ActiveSequenceEvent>) -> Self {
        Self {
            tx,
            full_drops: parking_lot::Mutex::default(),
        }
    }

    pub(crate) fn deliver(
        &self,
        partition: &RoutingPartitionId,
        expected_block_size: u32,
        block_size: u32,
        event: ActiveSequenceEvent,
    ) {
        if block_size != expected_block_size {
            tracing::debug!(
                %partition,
                expected_block_size,
                received_block_size = block_size,
                "Dropping replica event with mismatched block size"
            );
            return;
        }
        match self.tx.try_send(event) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(event)) => {
                let Some(dropped_events) = self.full_drops.lock().record(Instant::now()) else {
                    return;
                };
                tracing::warn!(
                    %partition,
                    request_id = %event.request_id,
                    capacity = self.tx.max_capacity(),
                    dropped_events_since_last_log = dropped_events,
                    "Replica subscriber channel full; dropping event"
                );
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::debug!(
                    %partition,
                    "Replica subscriber channel closed; dropping event"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::protocols::{ActiveSequenceEventData, WorkerWithDpRank};

    fn event() -> ScopedReplicaEvent {
        ScopedReplicaEvent {
            model_name: "model".to_string(),
            routing_group: "group".to_string(),
            block_size: 16,
            event: ActiveSequenceEvent {
                request_id: "request".to_string(),
                worker: WorkerWithDpRank::new(1, 0),
                data: ActiveSequenceEventData::Free,
                router_id: 42,
                lora_name: None,
            },
        }
    }

    #[test]
    fn replica_event_wire_schema_uses_routing_group() {
        let payload = rmp_serde::to_vec_named(&event()).unwrap();
        let value: serde_json::Value = rmp_serde::from_slice(&payload).unwrap();
        let fields = value.as_object().unwrap();

        assert_eq!(fields.len(), 4);
        assert!(fields.contains_key("model_name"));
        assert_eq!(value["routing_group"], "group");
        assert!(fields.contains_key("block_size"));
        assert!(fields.contains_key("event"));
        assert!(value.get("partition").is_none());
        assert!(value.get("tenant_id").is_none());

        let decoded: ScopedReplicaEvent = rmp_serde::from_slice(&payload).unwrap();
        assert_eq!(
            decoded.partition_ref(),
            RoutingPartitionRef::new("model", "group")
        );
    }

    #[test]
    fn previous_flat_replica_event_wire_schema_is_accepted() {
        let previous = serde_json::json!({
            "model_name": "model",
            "routing_group": "group",
            "block_size": 16,
            "event": event().event,
        });
        let payload = rmp_serde::to_vec_named(&previous).unwrap();

        let decoded: ScopedReplicaEvent = rmp_serde::from_slice(&payload).unwrap();
        assert_eq!(
            decoded.partition_ref(),
            RoutingPartitionRef::new("model", "group")
        );
        assert_eq!(decoded.block_size, 16);
    }

    #[test]
    fn legacy_replica_event_wire_schema_is_rejected() {
        let legacy = serde_json::json!({
            "model_name": "model",
            "tenant_id": "tenant",
            "block_size": 16,
            "event": event().event,
        });
        let payload = rmp_serde::to_vec_named(&legacy).unwrap();

        assert!(rmp_serde::from_slice::<ScopedReplicaEvent>(&payload).is_err());
    }

    #[tokio::test]
    async fn scoped_publisher_reports_full_and_closed_channels() {
        let (tx, mut rx) = mpsc::channel(1);
        let cancel_token = CancellationToken::new();
        let publisher = ScopedSequencePublisher::enabled(
            Arc::new(RoutingPartitionId::new("model", "group")),
            16,
            tx,
            cancel_token.clone(),
        );
        let event = event().event;

        publisher.enqueue_event(event.clone()).unwrap();
        let full = publisher.enqueue_event(event.clone()).unwrap_err();

        assert_eq!(rx.len(), 1);
        assert_eq!(rx.recv().await.unwrap().event.request_id, "request");
        let full = format!("{full:#}");
        assert!(full.contains("queue full"));
        assert!(full.contains("model_name=model"));
        assert!(full.contains("routing_group=group"));
        assert!(full.contains("capacity=1"));

        drop(rx);
        let unexpected_closed = publisher.enqueue_event(event.clone()).unwrap_err();
        assert!(matches!(
            unexpected_closed.downcast_ref::<SequencePublishQueueError>(),
            Some(SequencePublishQueueError::Closed {
                during_shutdown: false,
                ..
            })
        ));

        cancel_token.cancel();
        let shutdown_closed = publisher.enqueue_event(event).unwrap_err();
        assert!(matches!(
            shutdown_closed.downcast_ref::<SequencePublishQueueError>(),
            Some(SequencePublishQueueError::Closed {
                during_shutdown: true,
                ..
            })
        ));
    }

    #[cfg(feature = "standalone-selection")]
    #[test]
    fn affinity_sink_warns_on_unexpected_close() {
        let publish_to_closed_channel = |cancel_token: CancellationToken| {
            let (tx, rx) = mpsc::channel(1);
            drop(rx);
            let sink =
                AffinityMeshSink::new(RoutingPartitionId::new("model", "group"), tx, cancel_token);
            sink.publish(
                "session",
                AffinityTarget::new(1, None),
                AffinityVersion {
                    sequence: 1,
                    writer_id: 7,
                },
            );
            sink
        };

        let unexpected = publish_to_closed_channel(CancellationToken::new());
        assert_eq!(
            unexpected
                .unexpected_closed_drops
                .lock()
                .record(Instant::now()),
            None
        );

        let cancel_token = CancellationToken::new();
        cancel_token.cancel();
        let shutdown = publish_to_closed_channel(cancel_token);
        assert_eq!(
            shutdown
                .unexpected_closed_drops
                .lock()
                .record(Instant::now()),
            Some(1)
        );
    }

    #[test]
    fn ingress_only_host_channels_keep_replica_sync_disabled() {
        let (inbound_tx, inbound_rx) = mpsc::channel(1);
        let scoped = setup_scoped_replica_sync(
            None,
            &RoutingPartitionId::new("model", "default"),
            16,
            Some(HostReplicaChannels {
                outbound: None,
                inbound_tx,
                inbound_rx,
                process_id: 7,
                ingress_observer: None,
            }),
        );
        assert!(!scoped.enabled);
        assert_eq!(scoped.process_id, 7);
        assert!(scoped.channel.is_some());
    }

    #[derive(Default)]
    struct RecordingIngressObserver {
        applied: std::sync::atomic::AtomicUsize,
        batches: std::sync::atomic::AtomicUsize,
        last_depth: std::sync::atomic::AtomicUsize,
        max_depth: std::sync::atomic::AtomicUsize,
    }

    impl ReplicaIngressObserver for RecordingIngressObserver {
        fn observe_drain(&self, queue_depth: usize, applied: usize) {
            use std::sync::atomic::Ordering::Relaxed;
            self.applied.fetch_add(applied, Relaxed);
            self.batches.fetch_add(1, Relaxed);
            self.last_depth.store(queue_depth, Relaxed);
            self.max_depth.fetch_max(queue_depth, Relaxed);
        }
    }

    #[tokio::test]
    async fn host_ingress_observer_counts_applied_events_and_drains_to_zero() {
        use crate::sequences::{ActiveSequencesMultiWorker, NoopSequencePublisher};
        use std::sync::atomic::Ordering::Relaxed;

        const N: usize = 1_000;
        let observer = Arc::new(RecordingIngressObserver::default());
        let (inbound_tx, inbound_rx) = mpsc::channel(REPLICA_EVENT_CHANNEL_CAPACITY);
        let scoped = setup_scoped_replica_sync(
            None,
            &RoutingPartitionId::new("model", "default"),
            16,
            Some(HostReplicaChannels {
                outbound: None,
                inbound_tx,
                inbound_rx,
                process_id: 7,
                ingress_observer: Some(Arc::clone(&observer) as Arc<dyn ReplicaIngressObserver>),
            }),
        );
        let (inbound_tx, subscriber) = scoped.channel.expect("host channel");
        let tracker = Arc::new(ActiveSequencesMultiWorker::new_without_expiry(
            NoopSequencePublisher,
            16,
            std::collections::HashMap::from([(1_u64, (0_u32, 1_u32))]),
            true,
            scoped.process_id,
            "test",
        ));
        let cancel_token = CancellationToken::new();
        tracker.start_replica_sync(subscriber, cancel_token.clone());

        for index in 0..N {
            inbound_tx
                .try_send(ActiveSequenceEvent {
                    request_id: format!("req-{index}"),
                    worker: WorkerWithDpRank::new(1, 0),
                    data: ActiveSequenceEventData::AddRequest {
                        token_sequence: Some(vec![index as u64]),
                        track_prefill_tokens: false,
                        expected_output_tokens: None,
                        prefill_load_hint: None,
                    },
                    router_id: 99,
                    lora_name: None,
                })
                .expect("inbound channel has capacity");
        }

        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while observer.applied.load(Relaxed) < N {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("all events applied");

        assert_eq!(observer.applied.load(Relaxed), N);
        assert!(observer.batches.load(Relaxed) >= 1);
        assert!(observer.batches.load(Relaxed) <= N);
        assert_eq!(observer.last_depth.load(Relaxed), 0);
        // All N events were queued before the first drain, whose batch is
        // capped, so the first sample must show the backlog.
        assert!(
            observer.max_depth.load(Relaxed) >= N - crate::protocols::MAX_REPLICA_BATCH_EVENTS,
            "depth sample reflects the queued backlog: {}",
            observer.max_depth.load(Relaxed)
        );
        cancel_token.cancel();
    }
}
