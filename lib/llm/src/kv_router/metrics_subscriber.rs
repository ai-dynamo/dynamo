// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-selecting subscription for worker KV load metrics.

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use anyhow::Result;
use dynamo_kv_router::protocols::{ActiveLoad, WorkerWithDpRank};
use dynamo_runtime::{
    component::{Component, Endpoint},
    protocols::EndpointId,
    traits::DistributedRuntimeProvider,
    transports::event_plane::{
        Codec, EventEnvelope, EventSubscriber, TypedEventSubscriber, uses_direct_zmq,
    },
};
use parking_lot::Mutex;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::KV_METRICS_SUBJECT;
use crate::{
    direct_zmq_fan_in::{
        ContinuityMode, FanInEvent, FanInObservation, start_direct_zmq_fan_in_for_endpoint_id,
    },
    direct_zmq_sub_pool::KV_ZMQ_RCVHWM,
};

const MAX_PENDING_ACTIVE_LOADS: usize = 100_000;

pub(crate) struct KvMetricsSubscriber {
    inner: KvMetricsSubscriberInner,
}

enum KvMetricsSubscriberInner {
    Standard(TypedEventSubscriber<ActiveLoad>),
    Direct(DirectKvMetricsSubscriber),
}

struct DirectKvMetricsSubscriber {
    receiver: ActiveLoadReceiver,
    cancellation_token: CancellationToken,
}

type PublisherRank = (u64, WorkerWithDpRank);

struct PendingLoad {
    order: u64,
    envelope: EventEnvelope,
    load: ActiveLoad,
}

struct PendingActiveLoads {
    capacity: usize,
    order: BTreeMap<u64, PublisherRank>,
    next_order: u64,
    values: HashMap<PublisherRank, PendingLoad>,
    fault: Option<DirectKvMetricsFault>,
}

#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq, Eq)]
enum DirectKvMetricsFault {
    #[error("direct-ZMQ KV metrics payload decode failed")]
    PayloadDecode,
    #[error("direct-ZMQ KV metrics envelope decode failed")]
    EnvelopeDecode,
    #[error("direct-ZMQ KV metrics publisher identity mismatch")]
    IdentityMismatch,
    #[error("direct-ZMQ KV metrics transport disconnected")]
    Disconnected,
    #[error("direct-ZMQ KV metrics discovery watch reset")]
    DiscoveryReset,
}

fn fault_for_event(event: FanInEvent) -> Option<DirectKvMetricsFault> {
    match event {
        FanInEvent::EnvelopeDecodeError => Some(DirectKvMetricsFault::EnvelopeDecode),
        FanInEvent::IdentityMismatch => Some(DirectKvMetricsFault::IdentityMismatch),
        FanInEvent::Disconnected => Some(DirectKvMetricsFault::Disconnected),
        FanInEvent::DiscoveryReset => Some(DirectKvMetricsFault::DiscoveryReset),
        _ => None,
    }
}

impl PendingActiveLoads {
    fn new(capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "active-load mailbox capacity must be positive"
        );
        Self {
            capacity,
            order: BTreeMap::new(),
            next_order: 0,
            values: HashMap::new(),
            fault: None,
        }
    }

    fn push(&mut self, envelope: EventEnvelope, load: ActiveLoad) -> PushOutcome {
        let worker = WorkerWithDpRank::new(load.worker_id, load.dp_rank);
        let key = (envelope.publisher_id, worker);
        if let Some(pending) = self.values.get_mut(&key) {
            if envelope.sequence <= pending.envelope.sequence {
                return PushOutcome::Accepted { should_wake: false };
            }
            let order = self.next_order;
            self.next_order = self
                .next_order
                .checked_add(1)
                .expect("load mailbox order exhausted");
            // A coalesced replacement must follow earlier sequences from this publisher.
            self.order.remove(&pending.order);
            self.order.insert(order, key);
            pending.order = order;
            pending.envelope = envelope;
            let pending = &mut pending.load;
            let ActiveLoad {
                worker_id: _,
                dp_rank: _,
                active_decode_blocks,
                active_prefill_tokens,
                scheduler_load_scope,
                kv_used_blocks,
            } = load;
            if scheduler_load_scope.is_some() {
                pending.active_decode_blocks = active_decode_blocks;
                pending.active_prefill_tokens = active_prefill_tokens;
                pending.scheduler_load_scope = scheduler_load_scope;
            } else if pending.scheduler_load_scope.is_none() {
                if active_decode_blocks.is_some() {
                    pending.active_decode_blocks = active_decode_blocks;
                }
                if active_prefill_tokens.is_some() {
                    pending.active_prefill_tokens = active_prefill_tokens;
                }
            }
            if kv_used_blocks.is_some() {
                pending.kv_used_blocks = kv_used_blocks;
            }
            return PushOutcome::Accepted { should_wake: false };
        }
        if self.values.len() >= self.capacity {
            return PushOutcome::Full;
        }

        let should_wake = self.values.is_empty() && self.fault.is_none();
        let order = self.next_order;
        self.next_order = self
            .next_order
            .checked_add(1)
            .expect("load mailbox order exhausted");
        self.order.insert(order, key);
        self.values.insert(
            key,
            PendingLoad {
                order,
                envelope,
                load,
            },
        );
        PushOutcome::Accepted { should_wake }
    }

    fn push_fault(&mut self, fault: DirectKvMetricsFault) {
        self.order.clear();
        self.values.clear();
        self.fault = Some(fault);
    }

    fn pop(&mut self) -> Option<Result<(EventEnvelope, ActiveLoad)>> {
        if let Some(fault) = self.fault.take() {
            return Some(Err(fault.into()));
        }
        let (_order, key) = self.order.pop_first()?;
        let pending = self
            .values
            .remove(&key)
            .expect("active-load order and values must stay synchronized");
        Some(Ok((pending.envelope, pending.load)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PushOutcome {
    Accepted { should_wake: bool },
    Full,
}

#[derive(Clone)]
struct ActiveLoadSender {
    wake_tx: mpsc::Sender<()>,
    pending: Arc<Mutex<PendingActiveLoads>>,
}

impl ActiveLoadSender {
    fn send_enveloped(&self, envelope: EventEnvelope, load: ActiveLoad) -> Result<()> {
        if self.wake_tx.is_closed() {
            anyhow::bail!("direct-ZMQ KV metrics consumer closed");
        }

        let worker_id = load.worker_id;
        let dp_rank = load.dp_rank;
        let outcome = self.pending.lock().push(envelope, load);
        match outcome {
            PushOutcome::Accepted { should_wake: false } => Ok(()),
            PushOutcome::Full => {
                tracing::trace!(
                    worker_id,
                    dp_rank,
                    "Direct-ZMQ KV metrics consumer is full; dropping newest update"
                );
                Ok(())
            }
            PushOutcome::Accepted { should_wake: true } => match self.wake_tx.try_send(()) {
                Ok(()) | Err(mpsc::error::TrySendError::Full(())) => Ok(()),
                Err(mpsc::error::TrySendError::Closed(())) => {
                    anyhow::bail!("direct-ZMQ KV metrics consumer closed")
                }
            },
        }
    }

    #[cfg(test)]
    fn send(&self, load: ActiveLoad) -> Result<()> {
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT_SEQUENCE: AtomicU64 = AtomicU64::new(1);
        let sequence = NEXT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        self.send_enveloped(
            EventEnvelope {
                publisher_id: 1,
                sequence,
                published_at: sequence,
                topic: KV_METRICS_SUBJECT.to_owned(),
                payload: Default::default(),
            },
            load,
        )
    }

    fn send_fault(&self, fault: DirectKvMetricsFault) -> Result<()> {
        if self.wake_tx.is_closed() {
            anyhow::bail!("direct-ZMQ KV metrics consumer closed");
        }
        self.pending.lock().push_fault(fault);
        match self.wake_tx.try_send(()) {
            Ok(()) | Err(mpsc::error::TrySendError::Full(())) => Ok(()),
            Err(mpsc::error::TrySendError::Closed(())) => {
                anyhow::bail!("direct-ZMQ KV metrics consumer closed")
            }
        }
    }
}

struct ActiveLoadReceiver {
    wake_rx: mpsc::Receiver<()>,
    pending: Arc<Mutex<PendingActiveLoads>>,
}

impl ActiveLoadReceiver {
    #[cfg(test)]
    async fn recv(&mut self) -> Option<Result<ActiveLoad>> {
        self.recv_enveloped()
            .await
            .map(|result| result.map(|(_envelope, load)| load))
    }

    async fn recv_enveloped(&mut self) -> Option<Result<(EventEnvelope, ActiveLoad)>> {
        loop {
            if let Some(event) = self.pending.lock().pop() {
                return Some(event);
            }
            self.wake_rx.recv().await?;
        }
    }
}

fn active_load_mailbox() -> (ActiveLoadSender, ActiveLoadReceiver) {
    active_load_mailbox_with_capacity(MAX_PENDING_ACTIVE_LOADS)
}

fn active_load_mailbox_with_capacity(capacity: usize) -> (ActiveLoadSender, ActiveLoadReceiver) {
    let (wake_tx, wake_rx) = mpsc::channel(1);
    let pending = Arc::new(Mutex::new(PendingActiveLoads::new(capacity)));
    (
        ActiveLoadSender {
            wake_tx,
            pending: pending.clone(),
        },
        ActiveLoadReceiver { wake_rx, pending },
    )
}

impl Drop for DirectKvMetricsSubscriber {
    fn drop(&mut self) {
        self.cancellation_token.cancel();
    }
}

impl KvMetricsSubscriber {
    pub(crate) async fn for_endpoint(endpoint: &Endpoint) -> Result<Self> {
        Self::new(endpoint.component(), endpoint.id()).await
    }

    pub(crate) async fn for_endpoint_id(
        component: &Component,
        endpoint: &EndpointId,
    ) -> Result<Self> {
        Self::new(component, endpoint.clone()).await
    }

    async fn new(component: &Component, endpoint_id: EndpointId) -> Result<Self> {
        let drt = component.drt();
        if uses_direct_zmq(drt.default_event_transport_kind()) {
            return Ok(Self {
                inner: KvMetricsSubscriberInner::Direct(
                    DirectKvMetricsSubscriber::start(component, endpoint_id).await?,
                ),
            });
        }

        let subscriber =
            EventSubscriber::for_endpoint_id(drt, &endpoint_id, KV_METRICS_SUBJECT).await?;
        Ok(Self {
            inner: KvMetricsSubscriberInner::Standard(subscriber.typed::<ActiveLoad>()),
        })
    }

    pub(crate) async fn next(&mut self) -> Option<Result<ActiveLoad>> {
        self.next_enveloped()
            .await
            .map(|result| result.map(|(_envelope, load)| load))
    }

    pub(crate) async fn next_enveloped(&mut self) -> Option<Result<(EventEnvelope, ActiveLoad)>> {
        match &mut self.inner {
            KvMetricsSubscriberInner::Standard(subscriber) => subscriber.next().await,
            KvMetricsSubscriberInner::Direct(subscriber) => {
                subscriber.receiver.recv_enveloped().await
            }
        }
    }
}

impl DirectKvMetricsSubscriber {
    async fn start(component: &Component, endpoint_id: EndpointId) -> Result<Self> {
        let cancellation_token = component.drt().primary_token().child_token();
        let (sender, receiver) = active_load_mailbox();
        let handler_cancel = cancellation_token.clone();
        let codec = Codec::default();
        let handler_sender = sender.clone();
        let handler =
            move |envelope: dynamo_runtime::transports::event_plane::ValidatedEnvelope| {
                let load = match codec.decode_payload::<ActiveLoad>(&envelope.payload) {
                    Ok(load) => load,
                    Err(error) => {
                        let _ = handler_sender.send_fault(DirectKvMetricsFault::PayloadDecode);
                        return Err(error);
                    }
                };
                // The load is decoded already; retain only attribution for Relay freshness.
                let metadata = EventEnvelope {
                    publisher_id: envelope.publisher_id,
                    sequence: envelope.sequence,
                    published_at: envelope.published_at,
                    topic: KV_METRICS_SUBJECT.to_owned(),
                    payload: Default::default(),
                };
                if let Err(error) = handler_sender.send_enveloped(metadata, load) {
                    handler_cancel.cancel();
                    return Err(error);
                }
                Ok(())
            };
        let observer = move |observation: FanInObservation| {
            if let Some(fault) = fault_for_event(observation.event) {
                let _ = sender.send_fault(fault);
            }
        };
        let supervisor = start_direct_zmq_fan_in_for_endpoint_id(
            component.clone(),
            endpoint_id,
            KV_METRICS_SUBJECT,
            KV_ZMQ_RCVHWM,
            None,
            ContinuityMode::Disabled,
            cancellation_token.clone(),
            handler,
            observer,
        )
        .await?;
        drop(supervisor);

        Ok(Self {
            receiver,
            cancellation_token,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, time::Duration};

    use dynamo_runtime::{
        DistributedRuntime, Runtime, discovery::EventTransportKind, distributed::DistributedConfig,
        transports::event_plane::EventPublisher,
    };

    use super::*;
    use crate::direct_zmq_sub_pool::ENDPOINTS_PER_SUB_ENV;

    fn load(
        worker_id: u64,
        dp_rank: u32,
        active_decode_blocks: Option<u64>,
        active_prefill_tokens: Option<u64>,
        kv_used_blocks: Option<u64>,
    ) -> ActiveLoad {
        ActiveLoad {
            worker_id,
            dp_rank,
            active_decode_blocks,
            active_prefill_tokens,
            scheduler_load_scope: None,
            kv_used_blocks,
        }
    }

    fn envelope(publisher_id: u64, sequence: u64) -> EventEnvelope {
        EventEnvelope {
            publisher_id,
            sequence,
            published_at: sequence,
            topic: KV_METRICS_SUBJECT.to_owned(),
            payload: Default::default(),
        }
    }

    #[tokio::test]
    async fn direct_metrics_mailbox_keeps_publishers_and_latest_sequence_separate() {
        let (sender, mut receiver) = active_load_mailbox_with_capacity(2);
        sender
            .send_enveloped(envelope(7, 2), load(1, 0, Some(7), None, None))
            .unwrap();
        sender
            .send_enveloped(envelope(8, 1), load(1, 0, Some(20), None, None))
            .unwrap();
        sender
            .send_enveloped(envelope(7, 3), load(1, 0, None, None, Some(10)))
            .unwrap();
        sender
            .send_enveloped(envelope(7, 1), load(1, 0, None, None, Some(99)))
            .unwrap();

        let (first, first_load) = receiver.recv_enveloped().await.unwrap().unwrap();
        assert_eq!((first.publisher_id, first.sequence), (8, 1));
        assert_eq!(first_load, load(1, 0, Some(20), None, None));
        let (second, second_load) = receiver.recv_enveloped().await.unwrap().unwrap();
        assert_eq!((second.publisher_id, second.sequence), (7, 3));
        assert_eq!(second_load, load(1, 0, Some(7), None, Some(10)));
    }

    #[tokio::test]
    async fn direct_metrics_mailbox_delivers_latest_events_in_publisher_order() {
        let (sender, mut receiver) = active_load_mailbox_with_capacity(2);
        sender
            .send_enveloped(envelope(7, 1), load(1, 0, None, None, Some(10)))
            .unwrap();
        sender
            .send_enveloped(envelope(7, 2), load(1, 1, None, None, Some(20)))
            .unwrap();
        sender
            .send_enveloped(envelope(7, 3), load(1, 0, None, None, Some(30)))
            .unwrap();

        let (first, first_load) = receiver.recv_enveloped().await.unwrap().unwrap();
        assert_eq!(first.sequence, 2);
        assert_eq!(first_load.dp_rank, 1);
        let (second, second_load) = receiver.recv_enveloped().await.unwrap().unwrap();
        assert_eq!(second.sequence, 3);
        assert_eq!(second_load.dp_rank, 0);
        assert_eq!(second_load.kv_used_blocks, Some(30));
    }

    #[tokio::test]
    async fn direct_metrics_mailbox_preserves_scoped_scheduler_load() {
        use dynamo_kv_router::protocols::SchedulerLoadScope;

        let (sender, mut receiver) = active_load_mailbox_with_capacity(1);
        let mut scheduler = load(1, 0, Some(7), Some(11), None);
        scheduler.scheduler_load_scope = Some(SchedulerLoadScope::Local);
        sender.send(scheduler.clone()).unwrap();
        sender.send(load(1, 0, Some(99), None, Some(5))).unwrap();

        scheduler.kv_used_blocks = Some(5);
        assert_eq!(receiver.recv().await.unwrap().unwrap(), scheduler);
    }

    #[tokio::test]
    async fn direct_metrics_mailbox_merges_partial_updates_in_key_order() {
        let (sender, mut receiver) = active_load_mailbox_with_capacity(4);

        sender.send(load(1, 0, Some(7), None, None)).unwrap();
        sender.send(load(2, 0, None, None, Some(9))).unwrap();
        sender.send(load(1, 0, None, Some(11), None)).unwrap();
        sender.send(load(1, 0, Some(0), None, Some(5))).unwrap();
        sender.send(load(1, 1, None, None, Some(13))).unwrap();

        assert_eq!(
            receiver.recv().await.unwrap().unwrap(),
            load(2, 0, None, None, Some(9))
        );
        sender.send(load(1, 0, None, None, Some(6))).unwrap();
        assert_eq!(
            receiver.recv().await.unwrap().unwrap(),
            load(1, 1, None, None, Some(13))
        );
        assert_eq!(
            receiver.recv().await.unwrap().unwrap(),
            load(1, 0, Some(0), Some(11), Some(6))
        );
    }

    #[tokio::test]
    async fn direct_metrics_mailbox_bounds_distinct_keys_and_drains_on_close() {
        let (default_sender, _receiver) = active_load_mailbox();
        assert_eq!(
            default_sender.pending.lock().capacity,
            MAX_PENDING_ACTIVE_LOADS
        );

        let (sender, mut receiver) = active_load_mailbox_with_capacity(1);

        sender.send(load(1, 0, Some(7), None, None)).unwrap();
        sender.send(load(1, 0, None, None, Some(8))).unwrap();
        sender.send(load(2, 0, None, None, Some(9))).unwrap();

        assert_eq!(
            receiver.recv().await.unwrap().unwrap(),
            load(1, 0, Some(7), None, Some(8))
        );
        sender.send(load(2, 0, None, None, Some(10))).unwrap();
        drop(sender);
        assert_eq!(
            receiver.recv().await.unwrap().unwrap(),
            load(2, 0, None, None, Some(10))
        );
        assert!(receiver.recv().await.is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn direct_metrics_mailbox_wakes_a_waiter_and_reports_receiver_close() {
        let (sender, mut receiver) = active_load_mailbox_with_capacity(1);
        let waiter = tokio::spawn(async move {
            let load = receiver.recv().await;
            (load, receiver)
        });
        tokio::task::yield_now().await;

        sender.send(load(1, 0, None, None, Some(3))).unwrap();
        let (received, receiver) = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("waiting receiver must wake")
            .unwrap();
        assert_eq!(received.unwrap().unwrap(), load(1, 0, None, None, Some(3)));

        drop(receiver);
        assert!(sender.send(load(1, 0, None, None, Some(4))).is_err());
    }

    #[tokio::test]
    async fn direct_metrics_fault_discards_stale_loads_before_later_updates() {
        let (sender, mut receiver) = active_load_mailbox_with_capacity(4);

        sender.send(load(1, 0, None, None, Some(3))).unwrap();
        sender
            .send_fault(DirectKvMetricsFault::Disconnected)
            .unwrap();
        sender.send(load(2, 0, None, None, Some(4))).unwrap();

        let error = receiver.recv().await.unwrap().unwrap_err();
        assert!(error.to_string().contains("transport disconnected"));
        assert_eq!(
            receiver.recv().await.unwrap().unwrap(),
            load(2, 0, None, None, Some(4))
        );

        assert_eq!(
            fault_for_event(FanInEvent::EnvelopeDecodeError),
            Some(DirectKvMetricsFault::EnvelopeDecode)
        );
        assert_eq!(
            fault_for_event(FanInEvent::IdentityMismatch),
            Some(DirectKvMetricsFault::IdentityMismatch)
        );
        assert_eq!(
            fault_for_event(FanInEvent::DiscoveryReset),
            Some(DirectKvMetricsFault::DiscoveryReset)
        );
        assert_eq!(fault_for_event(FanInEvent::SourceStarted), None);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn direct_metrics_fan_in_receives_several_publishers() {
        temp_env::async_with_vars(
            [
                (
                    dynamo_runtime::config::environment_names::zmq_broker::DYN_ZMQ_BROKER_URL,
                    None::<&str>,
                ),
                (
                    dynamo_runtime::config::environment_names::zmq_broker::DYN_ZMQ_BROKER_ENABLED,
                    None::<&str>,
                ),
                (ENDPOINTS_PER_SUB_ENV, Some("64")),
            ],
            async {
                let runtime = Runtime::from_current().expect("create runtime handle");
                let distributed =
                    DistributedRuntime::new(runtime, DistributedConfig::process_local())
                        .await
                        .expect("create distributed runtime");
                let endpoint = distributed
                    .namespace(format!("kv-metrics-fan-in-{}", uuid::Uuid::new_v4()))
                    .expect("create namespace")
                    .component("frontend")
                    .expect("create component")
                    .endpoint("generate");
                let mut subscriber = KvMetricsSubscriber::for_endpoint(&endpoint)
                    .await
                    .expect("create direct metrics subscriber");
                let publisher_a = EventPublisher::for_endpoint_with_transport(
                    &endpoint,
                    KV_METRICS_SUBJECT,
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create publisher A");
                let publisher_b = EventPublisher::for_endpoint_with_transport(
                    &endpoint,
                    KV_METRICS_SUBJECT,
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create publisher B");

                let mut observed = HashSet::new();
                tokio::time::timeout(Duration::from_secs(5), async {
                    while observed.len() != 2 {
                        publisher_a
                            .publish(&ActiveLoad {
                                worker_id: 1,
                                ..ActiveLoad::default()
                            })
                            .await
                            .expect("publish A");
                        publisher_b
                            .publish(&ActiveLoad {
                                worker_id: 2,
                                ..ActiveLoad::default()
                            })
                            .await
                            .expect("publish B");
                        if let Ok(Some(Ok(load))) =
                            tokio::time::timeout(Duration::from_millis(50), subscriber.next()).await
                        {
                            observed.insert(load.worker_id);
                        }
                        tokio::time::sleep(Duration::from_millis(20)).await;
                    }
                })
                .await
                .expect("receive metrics from both publishers");
                assert_eq!(observed, HashSet::from([1, 2]));

                distributed.shutdown();
            },
        )
        .await;
    }
}
