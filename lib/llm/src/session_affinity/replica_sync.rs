// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

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
    time::Instant,
};
use tokio_util::sync::CancellationToken;

use crate::direct_zmq_fan_in::{
    ContinuityMode, FanInEvent, FanInObservation, start_direct_zmq_fan_in,
};
use dynamo_kv_router::services::selection::affinity::{
    AffinityReplicaSink, AffinityTarget, AffinityVersion, SessionAffinity, WeakSessionAffinity,
};

pub(super) const SESSION_AFFINITY_SUBJECT: &str = "session_affinity_events";
const OUTBOUND_CHANNEL_CAPACITY: usize = 4_096;
const DIRECT_ZMQ_RCVHWM: i32 = 1_024;
/// Peer bindings held for undiscovered workers, the same bound as the outbound queue.
const MAX_HELD_UPDATES: usize = OUTBOUND_CHANNEL_CAPACITY;

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

impl AffinityReplicaSink for ReplicaUpdateSender {
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

impl SessionAffinityUpdate {
    fn version(&self) -> AffinityVersion {
        AffinityVersion {
            sequence: self.sequence,
            writer_id: self.writer_id,
        }
    }
}

/// Peer bindings whose worker this replica has not discovered yet, newest per
/// session. They stay out of the table until discovery lists the worker.
#[derive(Default)]
struct HeldUpdates {
    sessions: HashMap<String, (SessionAffinityUpdate, Instant)>,
}

impl HeldUpdates {
    /// Keep `update` unless a newer version is already held for its session.
    /// When full, the oldest held update is dropped.
    fn hold(&mut self, update: SessionAffinityUpdate, now: Instant) {
        if let Some((held, received)) = self.sessions.get_mut(&update.session_id) {
            if update.version() >= held.version() {
                (*held, *received) = (update, now);
            }
            return;
        }
        if self.sessions.len() >= MAX_HELD_UPDATES
            && let Some(oldest) = self
                .sessions
                .iter()
                .min_by_key(|(_, (_, received))| *received)
                .map(|(session_id, _)| session_id.clone())
        {
            tracing::debug!("dropping oldest held session affinity update: hold limit reached");
            self.sessions.remove(&oldest);
        }
        self.sessions
            .insert(update.session_id.clone(), (update, now));
    }
}

#[derive(Clone)]
struct ReplicaUpdateApplier {
    local_publisher_id: u64,
    discovered_instances: watch::Receiver<Vec<Instance>>,
    table: WeakSessionAffinity,
    held: Arc<parking_lot::Mutex<HeldUpdates>>,
}

impl ReplicaUpdateApplier {
    fn apply(&self, source_publisher_id: u64, update: SessionAffinityUpdate) -> bool {
        if source_publisher_id == self.local_publisher_id {
            return true;
        }

        let Some(table) = self.table.upgrade() else {
            return false;
        };
        table.observe_replica_sequence(update.sequence);
        // `apply_held` reads discovery under this lock after each change, so an
        // update is either applied here or still held when that check runs.
        let mut held = self.held.lock();
        if !self
            .discovered_instances
            .borrow()
            .iter()
            .any(|instance| instance.id() == update.worker_id)
        {
            tracing::trace!(
                worker_id = update.worker_id,
                "holding session affinity update until its worker is discovered"
            );
            held.hold(update, Instant::now());
            return true;
        }
        if held
            .sessions
            .get(&update.session_id)
            .is_some_and(|(pending, _)| pending.version() <= update.version())
        {
            held.sessions.remove(&update.session_id);
        }
        apply_update(&table, update);
        true
    }

    /// Apply held updates whose worker is now discovered, dropping those held
    /// for the table's TTL or longer. Returns false once the table is gone.
    fn apply_held(&self) -> bool {
        let Some(table) = self.table.upgrade() else {
            return false;
        };
        let mut held = self.held.lock();
        if held.sessions.is_empty() {
            return true;
        }
        let discovered: HashSet<u64> = self
            .discovered_instances
            .borrow()
            .iter()
            .map(Instance::id)
            .collect();
        let (now, ttl) = (Instant::now(), table.ttl());
        for (_, (update, received)) in held.sessions.extract_if(|_, (update, received)| {
            now.saturating_duration_since(*received) >= ttl
                || discovered.contains(&update.worker_id)
        }) {
            if now.saturating_duration_since(received) < ttl {
                apply_update(&table, update);
            }
        }
        true
    }
}

fn apply_update(table: &SessionAffinity, update: SessionAffinityUpdate) {
    let target = AffinityTarget::new(update.worker_id, update.dp_rank);
    let version = update.version();
    let outcome = table.apply_replica_update(update.session_id, target, version);
    tracing::trace!(
        worker_id = target.worker_id,
        dp_rank = ?target.dp_rank,
        ?outcome,
        "processed best-effort session affinity update"
    );
}

pub(super) struct ReplicaSyncRuntime {
    sender: ReplicaUpdateSender,
    cancel: CancellationToken,
    publisher_task: Option<JoinHandle<()>>,
    subscriber_task: Option<JoinHandle<()>>,
    held_task: Option<JoinHandle<()>>,
}

impl ReplicaSyncRuntime {
    /// Returns the runtime and this replica's writer id (its discovery instance id).
    pub(super) async fn start(client: Client, table: WeakSessionAffinity) -> Result<(Self, u64)> {
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
        let applier = ReplicaUpdateApplier {
            local_publisher_id: publisher_id,
            discovered_instances: client.instance_source.as_ref().clone(),
            table,
            held: Arc::default(),
        };

        let cancel = CancellationToken::new();
        let held_applier = applier.clone();
        let mut discovered = applier.discovered_instances.clone();
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

        let held_cancel = cancel.clone();
        let held_task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = held_cancel.cancelled() => return,
                    changed = discovered.changed() => if changed.is_err() {
                        return;
                    },
                }
                if !held_applier.apply_held() {
                    return;
                }
            }
        });

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

        Ok((
            Self {
                sender,
                cancel,
                publisher_task: Some(publisher_task),
                subscriber_task: Some(subscriber_task),
                held_task: Some(held_task),
            },
            router_id,
        ))
    }

    /// The sink the table publishes into.
    pub(super) fn sink(&self) -> Arc<dyn AffinityReplicaSink> {
        Arc::new(self.sender.clone())
    }

    #[cfg(test)]
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
        if let Some(task) = self.held_task.take() {
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
                held_task: None,
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
        let coordinator = AffinityCoordinator::new(
            Duration::from_secs(10),
            crate::session_affinity::SessionAffinityMode::Hard,
        )
        .unwrap();
        let baseline = coordinator.next_version_for_test();
        let sequence = baseline.sequence.saturating_add(10);
        let (tx, discovered_instances) = watch::channel(Vec::new());
        let applier = ReplicaUpdateApplier {
            local_publisher_id: 7,
            discovered_instances,
            table: coordinator.table_for_test(),
            held: Arc::default(),
        };
        let update = |session_id: &str, worker_id, sequence| SessionAffinityUpdate {
            session_id: session_id.to_string(),
            worker_id,
            dp_rank: Some(0),
            sequence,
            writer_id: 9,
        };

        assert!(applier.apply(8, update("unknown-worker", 10, sequence)));

        assert_eq!(
            coordinator.next_version_for_test().sequence,
            sequence.saturating_add(1)
        );

        // The update waits for discovery instead of being lost, and a held
        // update still loses to a newer binding applied in the meantime.
        let table = coordinator.table_for_test().upgrade().unwrap();
        let bound = |session_id| {
            table
                .query_target(session_id, None)
                .unwrap()
                .map(|target| target.worker_id)
        };
        assert!(applier.apply(8, update("superseded", 11, sequence)));
        assert_eq!(bound("unknown-worker"), None);
        tx.send_modify(|instances| instances.push(instance(12)));
        assert!(applier.apply(8, update("superseded", 12, sequence + 1)));
        assert_eq!(bound("superseded"), Some(12));

        tx.send_modify(|instances| instances.extend([instance(10), instance(11)]));
        assert!(applier.apply_held());
        assert_eq!(bound("unknown-worker"), Some(10));
        assert_eq!(bound("superseded"), Some(12));
        assert!(applier.held.lock().sessions.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn held_update_expires_after_table_ttl() {
        let coordinator = AffinityCoordinator::new(
            Duration::from_secs(10),
            crate::session_affinity::SessionAffinityMode::Hard,
        )
        .unwrap();
        let (tx, discovered_instances) = watch::channel(Vec::new());
        let applier = ReplicaUpdateApplier {
            local_publisher_id: 7,
            discovered_instances,
            table: coordinator.table_for_test(),
            held: Arc::default(),
        };
        assert!(applier.apply(
            8,
            SessionAffinityUpdate {
                session_id: "s".to_string(),
                worker_id: 10,
                dp_rank: Some(0),
                sequence: 1,
                writer_id: 9,
            }
        ));
        tokio::time::advance(Duration::from_secs(10)).await;
        tx.send_modify(|instances| instances.push(instance(10)));
        assert!(applier.apply_held());
        let table = coordinator.table_for_test().upgrade().unwrap();
        assert_eq!(table.query_target("s", None).unwrap(), None);
        assert!(applier.held.lock().sessions.is_empty());
    }

    fn instance(instance_id: u64) -> Instance {
        Instance {
            component: "workers".to_string(),
            endpoint: "generate".to_string(),
            namespace: "test".to_string(),
            instance_id,
            transport: dynamo_runtime::component::TransportType::Tcp("127.0.0.1:1".to_string()),
            device_type: None,
            request_plane_codec: None,
        }
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

        let original = AffinityCoordinator::new(
            Duration::from_secs(10),
            crate::session_affinity::SessionAffinityMode::Hard,
        )
        .unwrap();
        let shared = original.clone();
        let (first, second) = tokio::join!(
            original.enable_replica_sync(client.clone()),
            shared.enable_replica_sync(client.clone()),
        );
        first.unwrap();
        second.unwrap();
        wait_for_registration_count(&drt, &query, 1).await;

        drop(original);
        shared.enable_replica_sync(client.clone()).await.unwrap();
        wait_for_registration_count(&drt, &query, 1).await;
        drop(shared);
        wait_for_registration_count(&drt, &query, 0).await;

        let replacement = AffinityCoordinator::new(
            Duration::from_secs(10),
            crate::session_affinity::SessionAffinityMode::Hard,
        )
        .unwrap();
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
