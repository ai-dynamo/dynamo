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
    direct_zmq_fan_in::{ContinuityMode, start_direct_zmq_fan_in_for_endpoint_id},
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

struct PendingActiveLoads {
    capacity: usize,
    order: VecDeque<WorkerWithDpRank>,
    values: HashMap<WorkerWithDpRank, ActiveLoad>,
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

    fn pop(&mut self) -> Option<ActiveLoad> {
        let worker = self.order.pop_front()?;
        Some(
            self.values
                .remove(&worker)
                .expect("active-load order and values must stay synchronized"),
        )
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
    fn send(&self, load: ActiveLoad) -> Result<()> {
        if self.wake_tx.is_closed() {
            anyhow::bail!("direct-ZMQ KV metrics consumer closed");
        }

        let worker_id = load.worker_id;
        let dp_rank = load.dp_rank;
        let outcome = self.pending.lock().push(load);
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
}

struct ActiveLoadReceiver {
    wake_rx: mpsc::Receiver<()>,
    pending: Arc<Mutex<PendingActiveLoads>>,
}

impl ActiveLoadReceiver {
    async fn recv(&mut self) -> Option<ActiveLoad> {
        loop {
            if let Some(load) = self.pending.lock().pop() {
                return Some(load);
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
        match &mut self.inner {
            KvMetricsSubscriberInner::Standard(subscriber) => subscriber
                .next()
                .await
                .map(|result| result.map(|(_envelope, load)| load)),
            KvMetricsSubscriberInner::Direct(subscriber) => {
                subscriber.receiver.recv().await.map(Ok)
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
        let handler =
            move |envelope: dynamo_runtime::transports::event_plane::ValidatedEnvelope| {
                let load = codec.decode_payload::<ActiveLoad>(&envelope.payload)?;
                if let Err(error) = sender.send(load) {
                    handler_cancel.cancel();
                    return Err(error);
                }
                Ok(())
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
            |_| {},
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
        let (sender, mut receiver) = active_load_mailbox_with_capacity(4);

        sender.send(load(1, 0, Some(7), None, None)).unwrap();
        sender.send(load(2, 0, None, None, Some(9))).unwrap();
        sender.send(load(1, 0, None, Some(11), None)).unwrap();
        sender.send(load(1, 0, Some(0), None, Some(5))).unwrap();
        sender.send(load(1, 1, None, None, Some(13))).unwrap();

        assert_eq!(
            receiver.recv().await,
            Some(load(1, 0, Some(0), Some(11), Some(5)))
        );
        sender.send(load(1, 0, None, None, Some(6))).unwrap();
        assert_eq!(receiver.recv().await, Some(load(2, 0, None, None, Some(9))));
        assert_eq!(
            receiver.recv().await,
            Some(load(1, 1, None, None, Some(13)))
        );
        assert_eq!(receiver.recv().await, Some(load(1, 0, None, None, Some(6))));
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
            receiver.recv().await,
            Some(load(1, 0, Some(7), None, Some(8)))
        );
        sender.send(load(2, 0, None, None, Some(10))).unwrap();
        drop(sender);
        assert_eq!(
            receiver.recv().await,
            Some(load(2, 0, None, None, Some(10)))
        );
        assert_eq!(receiver.recv().await, None);
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
        assert_eq!(received, Some(load(1, 0, None, None, Some(3))));

        drop(receiver);
        assert!(sender.send(load(1, 0, None, None, Some(4))).is_err());
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
