// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Weak;

use anyhow::{Context, Result};
use dynamo_runtime::{
    component::Client,
    traits::DistributedRuntimeProvider,
    transports::event_plane::{EventPublisher, EventSubscriber},
};
use serde::{Deserialize, Serialize};
use tokio::{sync::mpsc, task::JoinHandle};
use tokio_util::sync::CancellationToken;

use super::coordinator::{AffinityCoordinatorInner, AffinityTarget};

pub(super) const SESSION_AFFINITY_SUBJECT: &str = "session_affinity_events";
const OUTBOUND_CHANNEL_CAPACITY: usize = 4_096;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct SessionAffinityUpdate {
    pub session_id: String,
    pub worker_id: u64,
    pub dp_rank: Option<u32>,
    pub router_id: u64,
}

#[derive(Clone)]
struct ReplicaUpdateSender {
    router_id: u64,
    tx: mpsc::Sender<SessionAffinityUpdate>,
}

impl ReplicaUpdateSender {
    fn publish(&self, session_id: &str, target: AffinityTarget) {
        let update = SessionAffinityUpdate {
            session_id: session_id.to_string(),
            worker_id: target.worker_id,
            dp_rank: target.dp_rank,
            router_id: self.router_id,
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
        let component = client.endpoint.component();
        let router_id = component.drt().discovery().instance_id();
        let publisher = EventPublisher::for_component(component, SESSION_AFFINITY_SUBJECT)
            .await
            .context("create session affinity event publisher")?;
        let mut subscriber = EventSubscriber::for_component(component, SESSION_AFFINITY_SUBJECT)
            .await
            .context("create session affinity event subscriber")?
            .typed::<SessionAffinityUpdate>();
        let local_worker_ids = client.instance_avail_watcher();

        let cancel = parent_cancel.child_token();
        let (tx, mut rx) = mpsc::channel(OUTBOUND_CHANNEL_CAPACITY);
        let sender = ReplicaUpdateSender { router_id, tx };

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

        let subscriber_cancel = cancel.clone();
        let subscriber_task = tokio::spawn(async move {
            loop {
                let event = tokio::select! {
                    _ = subscriber_cancel.cancelled() => return,
                    event = subscriber.next() => event,
                };
                let Some(event) = event else {
                    return;
                };
                let update = match event {
                    Ok((_envelope, update)) => update,
                    Err(error) => {
                        tracing::trace!(
                            %error,
                            "failed to receive best-effort session affinity update"
                        );
                        continue;
                    }
                };
                if update.router_id == router_id {
                    continue;
                }
                let worker_ids = local_worker_ids.borrow();
                if !should_apply_update(router_id, worker_ids.as_slice(), &update) {
                    continue;
                }
                drop(worker_ids);
                let Some(coordinator) = coordinator.upgrade() else {
                    return;
                };
                let target = AffinityTarget {
                    worker_id: update.worker_id,
                    dp_rank: update.dp_rank,
                };
                let outcome = coordinator.apply_replica_update(update.session_id, target);
                drop(coordinator);
                tracing::trace!(
                    worker_id = target.worker_id,
                    dp_rank = ?target.dp_rank,
                    ?outcome,
                    "processed best-effort session affinity update"
                );
            }
        });

        Ok(Self {
            sender,
            cancel,
            publisher_task: Some(publisher_task),
            subscriber_task: Some(subscriber_task),
        })
    }

    pub(super) fn publish(&self, session_id: &str, target: AffinityTarget) {
        self.sender.publish(session_id, target);
    }

    pub(super) fn shutdown_now(&mut self) {
        self.cancel.cancel();
        if let Some(task) = self.publisher_task.take() {
            task.abort();
        }
        if let Some(task) = self.subscriber_task.take() {
            task.abort();
        }
    }

    #[cfg(test)]
    pub(super) fn for_test(
        router_id: u64,
        capacity: usize,
    ) -> (Self, mpsc::Receiver<SessionAffinityUpdate>) {
        let (tx, rx) = mpsc::channel(capacity);
        (
            Self {
                sender: ReplicaUpdateSender { router_id, tx },
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

fn should_apply_update(
    local_router_id: u64,
    local_worker_ids: &[u64],
    update: &SessionAffinityUpdate,
) -> bool {
    update.router_id != local_router_id && local_worker_ids.contains(&update.worker_id)
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

    fn update(router_id: u64, worker_id: u64) -> SessionAffinityUpdate {
        SessionAffinityUpdate {
            session_id: "session".to_string(),
            worker_id,
            dp_rank: Some(0),
            router_id,
        }
    }

    #[test]
    fn replica_update_filter_rejects_self_and_unknown_workers() {
        assert!(!should_apply_update(7, &[10, 11], &update(7, 10)));
        assert!(!should_apply_update(7, &[10, 11], &update(8, 12)));
        assert!(should_apply_update(7, &[10, 11], &update(8, 10)));
    }

    #[tokio::test]
    async fn replica_update_backpressure_is_nonfatal() {
        let (runtime, mut rx) = ReplicaSyncRuntime::for_test(7, 1);
        runtime.publish(
            "first",
            AffinityTarget {
                worker_id: 10,
                dp_rank: Some(0),
            },
        );
        runtime.publish(
            "second",
            AffinityTarget {
                worker_id: 11,
                dp_rank: Some(0),
            },
        );

        assert_eq!(rx.recv().await.unwrap().session_id, "first");
        assert!(rx.try_recv().is_err());
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
        let client = component.endpoint("generate").client().await.unwrap();
        let query = DiscoveryQuery::EventChannels(EventChannelQuery::topic(
            namespace_name,
            component_name,
            SESSION_AFFINITY_SUBJECT,
        ));

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
