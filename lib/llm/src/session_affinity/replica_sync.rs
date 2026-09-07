// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Weak;

use anyhow::{Context, Result};
use dynamo_runtime::{
    component::{Client, Instance},
    discovery::EventTransportKind,
    traits::DistributedRuntimeProvider,
    transports::event_plane::{
        Codec, EventPublisher, EventSubscriber, ValidatedEnvelope, uses_direct_zmq,
    },
};
use serde::{Deserialize, Serialize};
use tokio::{
    sync::{mpsc, watch},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

use super::coordinator::{AffinityCoordinatorInner, AffinityTarget, AffinityVersion};
use crate::direct_zmq_fan_in::{
    ContinuityMode, FanInEvent, FanInObservation, start_direct_zmq_fan_in,
};

pub(super) const SESSION_AFFINITY_SUBJECT: &str = "session_affinity_events";
const OUTBOUND_CHANNEL_CAPACITY: usize = 4_096;
const DIRECT_ZMQ_RCVHWM: i32 = 1_024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct SessionAffinityUpdate {
    pub session_id: String,
    pub worker_id: u64,
    pub dp_rank: Option<u32>,
    pub sequence: u64,
    pub writer_id: u64,
}

#[derive(Clone)]
struct ReplicaUpdateSender {
    tx: mpsc::Sender<SessionAffinityUpdate>,
}

impl ReplicaUpdateSender {
    fn publish(&self, session_id: &str, target: AffinityTarget, version: AffinityVersion) {
        let update = SessionAffinityUpdate {
            session_id: session_id.to_string(),
            worker_id: target.worker_id,
            dp_rank: target.dp_rank,
            sequence: version.sequence,
            writer_id: version.writer_id,
        };
        if let Err(error) = self.tx.try_send(update) {
            tracing::trace!(
                worker_id = target.worker_id,
                dp_rank = ?target.dp_rank,
                %error,
                "dropping best-effort session affinity update"
            );
        }
    }
}

#[derive(Clone)]
struct ReplicaUpdateApplier {
    local_publisher_id: u64,
    discovered_instances: watch::Receiver<Vec<Instance>>,
    coordinator: Weak<AffinityCoordinatorInner>,
}

impl ReplicaUpdateApplier {
    fn apply(&self, source_publisher_id: u64, update: SessionAffinityUpdate) -> bool {
        if source_publisher_id == self.local_publisher_id {
            return true;
        }

        let Some(coordinator) = self.coordinator.upgrade() else {
            return false;
        };
        coordinator.observe_replica_sequence(update.sequence);
        if !self
            .discovered_instances
            .borrow()
            .iter()
            .any(|instance| instance.id() == update.worker_id)
        {
            return true;
        }
        let target = AffinityTarget {
            worker_id: update.worker_id,
            dp_rank: update.dp_rank,
        };
        let outcome = coordinator.apply_replica_update(
            update.session_id,
            target,
            AffinityVersion {
                sequence: update.sequence,
                writer_id: update.writer_id,
            },
        );
        drop(coordinator);
        tracing::trace!(
            worker_id = target.worker_id,
            dp_rank = ?target.dp_rank,
            ?outcome,
            "processed best-effort session affinity update"
        );
        true
    }
}

pub(super) struct ReplicaSyncRuntime {
    sender: ReplicaUpdateSender,
    cancel: CancellationToken,
    publisher_task: Option<JoinHandle<()>>,
    subscriber_task: Option<JoinHandle<()>>,
}

impl ReplicaSyncRuntime {
    pub(super) async fn start(
        client: Client,
        coordinator: Weak<AffinityCoordinatorInner>,
        parent_cancel: &CancellationToken,
    ) -> Result<Self> {
        let endpoint = &client.endpoint;
        let router_id = endpoint.drt().discovery().instance_id();
        let transport_kind = endpoint.drt().default_event_transport_kind();
        let publisher = EventPublisher::for_endpoint_with_transport(
            endpoint,
            SESSION_AFFINITY_SUBJECT,
            transport_kind,
        )
        .await
        .context("create session affinity event publisher")?;
        let publisher_id = publisher.publisher_id();
        if let Some(inner) = coordinator.upgrade() {
            inner
                .writer_id
                .store(router_id, std::sync::atomic::Ordering::Relaxed);
        }
        let applier = ReplicaUpdateApplier {
            local_publisher_id: publisher_id,
            discovered_instances: client.instance_source.as_ref().clone(),
            coordinator,
        };

        let cancel = parent_cancel.child_token();
        let subscriber_task =
            if should_use_direct_sync(transport_kind, uses_direct_zmq(transport_kind)) {
                let codec = Codec::default();
                let handler_applier = applier.clone();
                let handler = move |envelope: ValidatedEnvelope| {
                    let source_publisher_id = envelope.publisher_id;
                    let update = codec
                        .decode_payload::<SessionAffinityUpdate>(&envelope.payload)
                        .context("decode session affinity update")?;
                    handler_applier.apply(source_publisher_id, update);
                    Ok(())
                };
                let observer = |observation: FanInObservation| match observation.event {
                    FanInEvent::SequenceGap { missing } => tracing::warn!(
                        publisher_id = observation.publisher_id,
                        generation = observation.generation,
                        missing,
                        "session affinity direct-ZMQ source skipped envelopes"
                    ),
                    FanInEvent::OutOfOrder => tracing::warn!(
                        publisher_id = observation.publisher_id,
                        generation = observation.generation,
                        "session affinity direct-ZMQ source received a non-increasing sequence"
                    ),
                    _ => {}
                };
                start_direct_zmq_fan_in(
                    endpoint.clone(),
                    SESSION_AFFINITY_SUBJECT,
                    DIRECT_ZMQ_RCVHWM,
                    Some(publisher_id),
                    ContinuityMode::TrackFromZero,
                    cancel.clone(),
                    handler,
                    observer,
                )
                .await
                .context("start direct-ZMQ session affinity subscriber")?
            } else {
                let mut subscriber = EventSubscriber::for_endpoint_with_transport(
                    endpoint,
                    SESSION_AFFINITY_SUBJECT,
                    transport_kind,
                )
                .await
                .context("create session affinity event subscriber")?
                .typed::<SessionAffinityUpdate>();
                let subscriber_cancel = cancel.clone();
                tokio::spawn(async move {
                    loop {
                        let event = tokio::select! {
                            _ = subscriber_cancel.cancelled() => return,
                            event = subscriber.next() => event,
                        };
                        let Some(event) = event else {
                            return;
                        };
                        let (source_publisher_id, update) = match event {
                            Ok((envelope, update)) => (envelope.publisher_id, update),
                            Err(error) => {
                                tracing::trace!(
                                    %error,
                                    "failed to receive best-effort session affinity update"
                                );
                                continue;
                            }
                        };
                        if !applier.apply(source_publisher_id, update) {
                            return;
                        }
                    }
                })
            };

        let (tx, mut rx) = mpsc::channel(OUTBOUND_CHANNEL_CAPACITY);
        let sender = ReplicaUpdateSender { tx };

        let publisher_cancel = cancel.clone();
        let publisher_task = tokio::spawn(async move {
            loop {
                let update = tokio::select! {
                    _ = publisher_cancel.cancelled() => return,
                    update = rx.recv() => update,
                };
                let Some(update) = update else {
                    return;
                };
                if let Err(error) = publisher.publish(&update).await {
                    tracing::trace!(
                        worker_id = update.worker_id,
                        dp_rank = ?update.dp_rank,
                        %error,
                        "failed to publish best-effort session affinity update"
                    );
                }
            }
        });

        Ok(Self {
            sender,
            cancel,
            publisher_task: Some(publisher_task),
            subscriber_task: Some(subscriber_task),
        })
    }

    pub(super) fn publish(
        &self,
        session_id: &str,
        target: AffinityTarget,
        version: AffinityVersion,
    ) {
        self.sender.publish(session_id, target, version);
    }

    pub(super) fn shutdown_now(&mut self) {
        self.cancel.cancel();
        if let Some(task) = self.publisher_task.take() {
            task.abort();
        }
        drop(self.subscriber_task.take());
    }

    #[cfg(test)]
    pub(super) fn for_test(capacity: usize) -> (Self, mpsc::Receiver<SessionAffinityUpdate>) {
        let (tx, rx) = mpsc::channel(capacity);
        (
            Self {
                sender: ReplicaUpdateSender { tx },
                cancel: CancellationToken::new(),
                publisher_task: None,
                subscriber_task: None,
            },
            rx,
        )
    }
}

impl Drop for ReplicaSyncRuntime {
    fn drop(&mut self) {
        self.shutdown_now();
    }
}

fn should_use_direct_sync(transport_kind: EventTransportKind, direct_zmq_topology: bool) -> bool {
    transport_kind == EventTransportKind::Zmq && direct_zmq_topology
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_affinity::AffinityCoordinator;
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        discovery::{DiscoveryQuery, EventChannelQuery},
        distributed::DistributedConfig,
    };
    use std::time::Duration;

    #[test]
    fn direct_sync_is_selected_only_for_unbrokered_zmq() {
        assert!(should_use_direct_sync(EventTransportKind::Zmq, true));
        assert!(!should_use_direct_sync(EventTransportKind::Zmq, false));
        assert!(!should_use_direct_sync(EventTransportKind::Nats, false));
    }

    #[tokio::test]
    async fn replica_update_backpressure_is_nonfatal() {
        let (runtime, mut rx) = ReplicaSyncRuntime::for_test(1);
        runtime.publish(
            "first",
            AffinityTarget {
                worker_id: 10,
                dp_rank: Some(0),
            },
            AffinityVersion {
                sequence: 1,
                writer_id: 7,
            },
        );
        runtime.publish(
            "second",
            AffinityTarget {
                worker_id: 11,
                dp_rank: Some(0),
            },
            AffinityVersion {
                sequence: 2,
                writer_id: 7,
            },
        );

        let update = rx.recv().await.unwrap();
        assert_eq!(update.session_id, "first");
        assert_eq!(update.writer_id, 7);
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn rejected_worker_update_still_advances_replica_clock() {
        let coordinator = AffinityCoordinator::new(Duration::from_secs(10)).unwrap();
        let baseline = coordinator.next_version_for_test();
        let sequence = baseline.sequence.saturating_add(10);
        let (_tx, discovered_instances) = watch::channel(Vec::new());
        let applier = ReplicaUpdateApplier {
            local_publisher_id: 7,
            discovered_instances,
            coordinator: coordinator.downgrade_for_test(),
        };

        assert!(applier.apply(
            8,
            SessionAffinityUpdate {
                session_id: "unknown-worker".to_string(),
                worker_id: 10,
                dp_rank: Some(0),
                sequence,
                writer_id: 9,
            }
        ));

        assert_eq!(
            coordinator.next_version_for_test().sequence,
            sequence.saturating_add(1)
        );
    }

    #[tokio::test]
    async fn replica_runtime_drop_unregisters_before_replacement() {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let namespace_name = format!("session-affinity-replica-{}", uuid::Uuid::new_v4());
        let component_name = "workers";
        let component = drt
            .namespace(namespace_name.clone())
            .unwrap()
            .component(component_name)
            .unwrap();
        let endpoint = component.endpoint("generate");
        let query = DiscoveryQuery::EventChannels(EventChannelQuery::endpoint_topic(
            endpoint.id(),
            SESSION_AFFINITY_SUBJECT,
        ));
        let client = endpoint.client().await.unwrap();

        let original = AffinityCoordinator::new(Duration::from_secs(10)).unwrap();
        original.enable_replica_sync(client.clone()).await.unwrap();
        wait_for_registration_count(&drt, &query, 1).await;

        drop(original);
        wait_for_registration_count(&drt, &query, 0).await;

        let replacement = AffinityCoordinator::new(Duration::from_secs(10)).unwrap();
        replacement.enable_replica_sync(client).await.unwrap();
        wait_for_registration_count(&drt, &query, 1).await;

        drop(replacement);
        wait_for_registration_count(&drt, &query, 0).await;
    }

    async fn wait_for_registration_count(
        drt: &DistributedRuntime,
        query: &DiscoveryQuery,
        expected: usize,
    ) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let registrations = drt.discovery().list(query.clone()).await.unwrap();
                if registrations.len() == expected {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("event registration count did not reach {expected}"));
    }
}
