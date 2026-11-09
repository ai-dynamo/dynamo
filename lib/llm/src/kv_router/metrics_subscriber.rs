// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-selecting subscription for worker KV load metrics.

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use anyhow::Result;
use dynamo_kv_router::protocols::{ActiveLoad, WorkerWithDpRank};
use dynamo_runtime::{
    component::{Component, Endpoint},
    protocols::EndpointId,
    traits::DistributedRuntimeProvider,
    transports::event_plane::{Codec, EventSubscriber, TypedEventSubscriber, uses_direct_zmq},
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
/// Every direct-ZMQ source task pushes into this mailbox. Sharding by worker spreads those
/// producers over independent locks while keeping per-worker merging and ordering. The
/// consumer scans the shards for each load, so keep the count small.
const ACTIVE_LOAD_SHARDS: usize = 16;

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

struct PendingActiveLoads {
    capacity: usize,
    order: VecDeque<WorkerWithDpRank>,
    values: HashMap<WorkerWithDpRank, ActiveLoad>,
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
            order: VecDeque::new(),
            values: HashMap::new(),
        }
    }

    fn push(&mut self, load: ActiveLoad) -> PushOutcome {
        let worker = WorkerWithDpRank::new(load.worker_id, load.dp_rank);
        if let Some(pending) = self.values.get_mut(&worker) {
            let ActiveLoad {
                worker_id: _,
                dp_rank: _,
                active_decode_blocks,
                active_prefill_tokens,
                kv_used_blocks,
            } = load;
            if active_decode_blocks.is_some() {
                pending.active_decode_blocks = active_decode_blocks;
            }
            if active_prefill_tokens.is_some() {
                pending.active_prefill_tokens = active_prefill_tokens;
            }
            if kv_used_blocks.is_some() {
                pending.kv_used_blocks = kv_used_blocks;
            }
            return PushOutcome::Accepted { should_wake: false };
        }
        if self.values.len() >= self.capacity {
            return PushOutcome::Full;
        }

        let should_wake = self.values.is_empty();
        self.order.push_back(worker);
        self.values.insert(worker, load);
        PushOutcome::Accepted { should_wake }
    }

    fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    fn clear(&mut self) {
        self.order.clear();
        self.values.clear();
    }

    fn pop(&mut self) -> Option<ActiveLoad> {
        let worker = self.order.pop_front()?;
        Some(
            self.values
                .remove(&worker)
                .expect("active-load order and values must stay synchronized"),
        )
    }
}

/// One mailbox shard, aligned so neighboring shard locks never share a cache line.
#[repr(align(128))]
struct ActiveLoadShard(Mutex<PendingActiveLoads>);

struct ActiveLoadMailbox {
    shards: Box<[ActiveLoadShard]>,
    fault: Mutex<Option<DirectKvMetricsFault>>,
}

impl ActiveLoadMailbox {
    fn new(capacity: usize, shards: usize) -> Self {
        assert!(shards > 0, "active-load mailbox needs at least one shard");
        let shard_capacity = capacity.div_ceil(shards);
        Self {
            shards: (0..shards)
                .map(|_| ActiveLoadShard(Mutex::new(PendingActiveLoads::new(shard_capacity))))
                .collect(),
            fault: Mutex::new(None),
        }
    }

    /// Every update for one worker rank maps to the same shard.
    fn shard_index(&self, worker_id: u64, dp_rank: u32) -> usize {
        let hash = (worker_id ^ (u64::from(dp_rank) << 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (hash >> 32) as usize % self.shards.len()
    }

    fn shard(&self, worker_id: u64, dp_rank: u32) -> &Mutex<PendingActiveLoads> {
        &self.shards[self.shard_index(worker_id, dp_rank)].0
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
    mailbox: Arc<ActiveLoadMailbox>,
}

impl ActiveLoadSender {
    fn send(&self, load: ActiveLoad) -> Result<()> {
        if self.wake_tx.is_closed() {
            anyhow::bail!("direct-ZMQ KV metrics consumer closed");
        }

        let worker_id = load.worker_id;
        let dp_rank = load.dp_rank;
        let outcome = self.mailbox.shard(worker_id, dp_rank).lock().push(load);
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

    fn send_fault(&self, fault: DirectKvMetricsFault) -> Result<()> {
        if self.wake_tx.is_closed() {
            anyhow::bail!("direct-ZMQ KV metrics consumer closed");
        }
        {
            // Hold every shard until the fault is recorded, so no load lands between clearing
            // and recording, and `try_pop` sees the fault before any later load.
            let mut shards: Vec<_> = self
                .mailbox
                .shards
                .iter()
                .map(|shard| shard.0.lock())
                .collect();
            for pending in &mut shards {
                pending.clear();
            }
            *self.mailbox.fault.lock() = Some(fault);
        }
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
    mailbox: Arc<ActiveLoadMailbox>,
    next_shard: usize,
}

impl ActiveLoadReceiver {
    async fn recv(&mut self) -> Option<Result<ActiveLoad>> {
        loop {
            if let Some(load) = self.try_pop() {
                return Some(load);
            }
            // A shard that becomes non-empty after the scan above sends a wake.
            self.wake_rx.recv().await?;
        }
    }

    /// Return a pending fault first, then one load, visiting shards round-robin.
    fn try_pop(&mut self) -> Option<Result<ActiveLoad>> {
        let shards = self.mailbox.shards.len();
        for offset in 0..shards {
            let index = (self.next_shard + offset) % shards;
            let mut pending = self.mailbox.shards[index].0.lock();
            if pending.is_empty() {
                continue;
            }
            // `send_fault` records its fault while holding every shard, so a load found
            // here while a fault is pending was sent after that fault.
            if let Some(fault) = self.mailbox.fault.lock().take() {
                return Some(Err(fault.into()));
            }
            self.next_shard = (index + 1) % shards;
            return pending.pop().map(Ok);
        }
        self.mailbox
            .fault
            .lock()
            .take()
            .map(|fault| Err(fault.into()))
    }
}

fn active_load_mailbox() -> (ActiveLoadSender, ActiveLoadReceiver) {
    active_load_mailbox_with(MAX_PENDING_ACTIVE_LOADS, ACTIVE_LOAD_SHARDS)
}

fn active_load_mailbox_with(
    capacity: usize,
    shards: usize,
) -> (ActiveLoadSender, ActiveLoadReceiver) {
    let (wake_tx, wake_rx) = mpsc::channel(1);
    let mailbox = Arc::new(ActiveLoadMailbox::new(capacity, shards));
    (
        ActiveLoadSender {
            wake_tx,
            mailbox: mailbox.clone(),
        },
        ActiveLoadReceiver {
            wake_rx,
            mailbox,
            next_shard: 0,
        },
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
        match &mut self.inner {
            KvMetricsSubscriberInner::Standard(subscriber) => subscriber
                .next()
                .await
                .map(|result| result.map(|(_envelope, load)| load)),
            KvMetricsSubscriberInner::Direct(subscriber) => subscriber.receiver.recv().await,
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
                if let Err(error) = handler_sender.send(load) {
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
            kv_used_blocks,
        }
    }

    #[tokio::test]
    async fn direct_metrics_mailbox_merges_partial_updates_in_key_order() {
        let (sender, mut receiver) = active_load_mailbox_with(4, 1);

        sender.send(load(1, 0, Some(7), None, None)).unwrap();
        sender.send(load(2, 0, None, None, Some(9))).unwrap();
        sender.send(load(1, 0, None, Some(11), None)).unwrap();
        sender.send(load(1, 0, Some(0), None, Some(5))).unwrap();
        sender.send(load(1, 1, None, None, Some(13))).unwrap();

        assert_eq!(
            receiver.recv().await.unwrap().unwrap(),
            load(1, 0, Some(0), Some(11), Some(5))
        );
        sender.send(load(1, 0, None, None, Some(6))).unwrap();
        assert_eq!(
            receiver.recv().await.unwrap().unwrap(),
            load(2, 0, None, None, Some(9))
        );
        assert_eq!(
            receiver.recv().await.unwrap().unwrap(),
            load(1, 1, None, None, Some(13))
        );
        assert_eq!(
            receiver.recv().await.unwrap().unwrap(),
            load(1, 0, None, None, Some(6))
        );
    }

    #[tokio::test]
    async fn direct_metrics_mailbox_bounds_distinct_keys_and_drains_on_close() {
        let (default_sender, _receiver) = active_load_mailbox();
        assert_eq!(default_sender.mailbox.shards.len(), ACTIVE_LOAD_SHARDS);
        assert!(default_sender.mailbox.shards.iter().all(|shard| {
            shard.0.lock().capacity == MAX_PENDING_ACTIVE_LOADS.div_ceil(ACTIVE_LOAD_SHARDS)
        }));

        let (sender, mut receiver) = active_load_mailbox_with(1, 1);

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
        let (sender, mut receiver) = active_load_mailbox_with(1, 1);
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
        let (sender, mut receiver) = active_load_mailbox_with(4, 1);

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
    async fn direct_metrics_sharded_mailbox_merges_per_key_and_orders_faults() {
        let (sender, mut receiver) = active_load_mailbox_with(64, 8);
        let used_shards: HashSet<_> = (0..16)
            .map(|worker_id| sender.mailbox.shard_index(worker_id, 0))
            .collect();
        assert!(used_shards.len() > 1, "test keys must span several shards");

        for worker_id in 0..16 {
            sender
                .send(load(worker_id, 0, Some(worker_id), None, None))
                .unwrap();
            sender
                .send(load(worker_id, 0, None, None, Some(worker_id + 100)))
                .unwrap();
        }
        let mut merged = HashMap::new();
        for _ in 0..16 {
            let update = receiver.recv().await.unwrap().unwrap();
            assert!(merged.insert(update.worker_id, update).is_none());
        }
        for worker_id in 0..16 {
            assert_eq!(
                merged[&worker_id],
                load(worker_id, 0, Some(worker_id), None, Some(worker_id + 100))
            );
        }

        for worker_id in 0..16 {
            sender
                .send(load(worker_id, 1, None, None, Some(1)))
                .unwrap();
        }
        sender
            .send_fault(DirectKvMetricsFault::Disconnected)
            .unwrap();
        sender.send(load(7, 1, None, None, Some(2))).unwrap();
        let error = receiver.recv().await.unwrap().unwrap_err();
        assert!(error.to_string().contains("transport disconnected"));
        assert_eq!(
            receiver.recv().await.unwrap().unwrap(),
            load(7, 1, None, None, Some(2))
        );
        assert!(receiver.try_pop().is_none());
    }

    #[test]
    fn direct_metrics_fault_precedes_later_loads_under_concurrency() {
        const UPDATES: u64 = 100_000;

        let (sender, mut receiver) = active_load_mailbox();
        let producer = std::thread::spawn(move || {
            for value in 1..=UPDATES {
                sender
                    .send_fault(DirectKvMetricsFault::Disconnected)
                    .unwrap();
                sender.send(load(1, 0, None, None, Some(value))).unwrap();
            }
        });

        // Every load is sent after a fault, so a fault must arrive before each load.
        let mut fault_seen = false;
        let mut last_value = 0;
        while last_value < UPDATES {
            match receiver.try_pop() {
                Some(Err(_)) => fault_seen = true,
                Some(Ok(update)) => {
                    let value = update.kv_used_blocks.unwrap();
                    assert!(fault_seen, "load {value} arrived before its fault");
                    assert!(value > last_value);
                    fault_seen = false;
                    last_value = value;
                }
                None => std::hint::spin_loop(),
            }
        }
        producer.join().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn direct_metrics_sharded_mailbox_delivers_latest_values_under_concurrency() {
        const PRODUCERS: u64 = 32;
        const KEYS_PER_PRODUCER: u64 = 8;
        const UPDATES: u64 = 1000;

        let (sender, mut receiver) = active_load_mailbox();
        let consumer = tokio::spawn(async move {
            let mut latest = HashMap::new();
            while let Some(update) = receiver.recv().await {
                let update = update.unwrap();
                let value = update.kv_used_blocks.unwrap();
                let previous = latest.insert(update.worker_id, value);
                assert!(
                    previous.is_none_or(|previous| previous < value),
                    "worker {} went from {previous:?} to {value}",
                    update.worker_id
                );
            }
            latest
        });
        let producers: Vec<_> = (0..PRODUCERS)
            .map(|producer| {
                let sender = sender.clone();
                tokio::spawn(async move {
                    for value in 1..=UPDATES {
                        for key in 0..KEYS_PER_PRODUCER {
                            let worker_id = producer * KEYS_PER_PRODUCER + key;
                            sender
                                .send(load(worker_id, 0, None, None, Some(value)))
                                .unwrap();
                        }
                        if value % 64 == 0 {
                            tokio::task::yield_now().await;
                        }
                    }
                })
            })
            .collect();
        drop(sender);
        for producer in producers {
            producer.await.unwrap();
        }

        let latest = tokio::time::timeout(Duration::from_secs(30), consumer)
            .await
            .expect("consumer must drain after all senders close")
            .unwrap();
        assert_eq!(latest.len() as u64, PRODUCERS * KEYS_PER_PRODUCER);
        assert!(latest.values().all(|&value| value == UPDATES));
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
