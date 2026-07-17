// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    sync::Arc,
    time::Duration,
};

use dynamo_kv_router::protocols::{KV_EVENT_SUBJECT, WorkerWithDpRank};
use dynamo_runtime::{
    discovery::{
        Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
        EventSourceInstanceId, EventSourceQuery,
    },
    protocols::EndpointId,
    transports::event_plane::EventScope,
};
use futures::StreamExt;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use super::{
    KvEventSource, KvSourceId, KvSourceMembership, KvSourceMembershipView, KvSourceStatus,
    KvStateEndpointResolution, RuntimeConfigWatch, resolve_kv_state_endpoint,
};

struct BoundSourceStream {
    stream: dynamo_runtime::discovery::DiscoveryStream,
    cancel: CancellationToken,
}

impl Drop for BoundSourceStream {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// A subscription to the shared KV-source membership coordinator for one serving endpoint.
///
/// Clones receive independent watch cursors while retaining the same coordinator lifetime.
pub struct KvSourceMembershipWatch {
    coordinator: Arc<KvSourceMembershipCoordinator>,
    receiver: watch::Receiver<KvSourceMembershipView>,
}

impl Clone for KvSourceMembershipWatch {
    fn clone(&self) -> Self {
        self.coordinator.subscribe()
    }
}

impl Deref for KvSourceMembershipWatch {
    type Target = watch::Receiver<KvSourceMembershipView>;

    fn deref(&self) -> &Self::Target {
        &self.receiver
    }
}

impl DerefMut for KvSourceMembershipWatch {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.receiver
    }
}

impl KvSourceMembershipWatch {
    #[cfg(test)]
    pub(crate) fn shares_coordinator_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.coordinator, &other.coordinator)
    }
}

pub(crate) struct KvSourceMembershipCoordinator {
    sender: watch::Sender<KvSourceMembershipView>,
    anchor: watch::Receiver<KvSourceMembershipView>,
    cancel: CancellationToken,
}

impl KvSourceMembershipCoordinator {
    pub(crate) fn start(
        serving_endpoint: EndpointId,
        runtime_configs: RuntimeConfigWatch,
        discovery: Arc<dyn Discovery>,
    ) -> Arc<Self> {
        let initial_configs = runtime_configs.borrow().clone();
        let initial_view = KvSourceMembership::new().view(&serving_endpoint, &initial_configs);
        let (sender, anchor) = watch::channel(initial_view);

        let coordinator = Arc::new(Self {
            sender,
            anchor,
            cancel: CancellationToken::new(),
        });
        tokio::spawn(run_membership_coordinator(
            serving_endpoint,
            runtime_configs,
            discovery,
            coordinator.sender.clone(),
            coordinator.cancel.clone(),
        ));
        coordinator
    }

    pub(crate) fn subscribe(self: &Arc<Self>) -> KvSourceMembershipWatch {
        KvSourceMembershipWatch {
            coordinator: self.clone(),
            receiver: self.anchor.clone(),
        }
    }
}

impl Drop for KvSourceMembershipCoordinator {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[derive(Default)]
struct LifecycleTracker {
    entries: HashMap<WorkerWithDpRank, LogicalLifecycle>,
    endpoint_resolution: Option<KvStateEndpointResolution>,
}

#[derive(Default)]
struct LogicalLifecycle {
    generation: u64,
    ever_active: bool,
    previous: Option<KvSourceStatus>,
}

impl LifecycleTracker {
    fn apply(&mut self, mut view: KvSourceMembershipView) -> KvSourceMembershipView {
        let endpoint_remapped = self
            .endpoint_resolution
            .as_ref()
            .is_some_and(|previous| previous != &view.endpoint_resolution);
        self.endpoint_resolution = Some(view.endpoint_resolution.clone());

        let expected: Vec<_> = view.sources.keys().copied().collect();
        for worker in &expected {
            let status = view
                .sources
                .get(worker)
                .expect("worker came from source membership view");
            let lifecycle = self.entries.entry(*worker).or_default();
            if endpoint_remapped || transition_requires_reset(lifecycle, status) {
                lifecycle.generation = lifecycle.generation.saturating_add(1);
            }
            lifecycle.ever_active |= status.active_source().is_some();
            lifecycle.previous = Some(status.clone());
            view.lifecycle_generations
                .insert(*worker, lifecycle.generation);
        }

        for (worker, lifecycle) in &mut self.entries {
            if expected.contains(worker) {
                continue;
            }
            if lifecycle.ever_active && lifecycle.previous.is_some() {
                lifecycle.generation = lifecycle.generation.saturating_add(1);
            }
            lifecycle.previous = None;
        }

        view
    }
}

fn transition_requires_reset(lifecycle: &LogicalLifecycle, current: &KvSourceStatus) -> bool {
    let Some(previous) = lifecycle.previous.as_ref() else {
        return lifecycle.ever_active;
    };
    if previous == current {
        return false;
    }

    match (previous, current) {
        (KvSourceStatus::Missing, KvSourceStatus::Missing) => false,
        (
            KvSourceStatus::Missing,
            KvSourceStatus::ActiveLiveOnly(_) | KvSourceStatus::ActiveRecoverable(_),
        ) => lifecycle.ever_active,
        _ => true,
    }
}

async fn run_membership_coordinator(
    serving_endpoint: EndpointId,
    mut runtime_configs: RuntimeConfigWatch,
    discovery: Arc<dyn Discovery>,
    sender: watch::Sender<KvSourceMembershipView>,
    cancel: CancellationToken,
) {
    let mut configs = runtime_configs.borrow_and_update().clone();
    let mut resolution = resolve_kv_state_endpoint(&serving_endpoint, configs.values());
    let mut membership = KvSourceMembership::new();
    let mut source_ids = HashMap::new();
    let mut lifecycle = LifecycleTracker::default();
    let mut source_stream = bind_source_stream(&discovery, &resolution, &cancel).await;
    let mut retry_delay = Duration::from_millis(100);
    publish_view(
        &sender,
        &mut lifecycle,
        membership.view(&serving_endpoint, &configs),
    );

    loop {
        enum CoordinatorInput {
            RuntimeConfig(bool),
            Source(Option<anyhow::Result<DiscoveryEvent>>),
            Retry,
            Cancelled,
        }

        let input = if let Some(binding) = source_stream.as_mut() {
            tokio::select! {
                _ = cancel.cancelled() => CoordinatorInput::Cancelled,
                changed = runtime_configs.changed() => CoordinatorInput::RuntimeConfig(changed.is_ok()),
                event = binding.stream.next() => CoordinatorInput::Source(event),
            }
        } else {
            tokio::select! {
                _ = cancel.cancelled() => CoordinatorInput::Cancelled,
                changed = runtime_configs.changed() => CoordinatorInput::RuntimeConfig(changed.is_ok()),
                _ = tokio::time::sleep(retry_delay), if matches!(resolution, KvStateEndpointResolution::Resolved(_)) => CoordinatorInput::Retry,
            }
        };

        match input {
            CoordinatorInput::Cancelled | CoordinatorInput::RuntimeConfig(false) => break,
            CoordinatorInput::RuntimeConfig(true) => {
                configs = runtime_configs.borrow_and_update().clone();
                let next_resolution =
                    resolve_kv_state_endpoint(&serving_endpoint, configs.values());
                if next_resolution != resolution {
                    resolution = next_resolution;
                    membership = KvSourceMembership::new();
                    source_ids.clear();
                    drop(source_stream.take());
                    publish_view(
                        &sender,
                        &mut lifecycle,
                        membership.view(&serving_endpoint, &configs),
                    );
                    source_stream = bind_source_stream(&discovery, &resolution, &cancel).await;
                    if source_stream.is_some() {
                        retry_delay = Duration::from_millis(100);
                    }
                } else {
                    publish_view(
                        &sender,
                        &mut lifecycle,
                        membership.view(&serving_endpoint, &configs),
                    );
                }
            }
            CoordinatorInput::Source(Some(event)) => {
                reconcile_discovery_event(event, &resolution, &mut membership, &mut source_ids);
                publish_view(
                    &sender,
                    &mut lifecycle,
                    membership.view(&serving_endpoint, &configs),
                );
            }
            CoordinatorInput::Source(None) => {
                membership = KvSourceMembership::new();
                source_ids.clear();
                publish_view(
                    &sender,
                    &mut lifecycle,
                    membership.view(&serving_endpoint, &configs),
                );
                source_stream = None;
            }
            CoordinatorInput::Retry => {
                source_stream = bind_source_stream(&discovery, &resolution, &cancel).await;
                if source_stream.is_some() {
                    retry_delay = Duration::from_millis(100);
                } else {
                    retry_delay = (retry_delay * 2).min(Duration::from_secs(5));
                }
            }
        }
    }
}

async fn bind_source_stream(
    discovery: &Arc<dyn Discovery>,
    resolution: &KvStateEndpointResolution,
    cancel: &CancellationToken,
) -> Option<BoundSourceStream> {
    let KvStateEndpointResolution::Resolved(endpoint) = resolution else {
        return None;
    };
    let binding_cancel = cancel.child_token();
    match discovery
        .list_and_watch(
            DiscoveryQuery::EventSources(EventSourceQuery::endpoint_topic(
                endpoint.clone(),
                KV_EVENT_SUBJECT,
            )),
            Some(binding_cancel.clone()),
        )
        .await
    {
        Ok(stream) => Some(BoundSourceStream {
            stream,
            cancel: binding_cancel,
        }),
        Err(error) => {
            tracing::error!(%endpoint, %error, "Failed to watch exact KV event sources");
            None
        }
    }
}

fn reconcile_discovery_event(
    event: anyhow::Result<DiscoveryEvent>,
    resolution: &KvStateEndpointResolution,
    membership: &mut KvSourceMembership,
    source_ids: &mut HashMap<u64, KvSourceId>,
) {
    let KvStateEndpointResolution::Resolved(expected_endpoint) = resolution else {
        return;
    };
    let expected_scope = EventScope::Endpoint {
        endpoint: expected_endpoint.clone(),
    };

    match event {
        Ok(DiscoveryEvent::Added(DiscoveryInstance::EventSource {
            scope,
            topic,
            publisher_id,
            metadata,
        })) => {
            if scope != expected_scope || topic != KV_EVENT_SUBJECT {
                tracing::warn!(publisher_id, "Ignoring incorrectly scoped KV event source");
                return;
            }
            let source = match serde_json::from_value::<KvEventSource>(metadata) {
                Ok(source) => source,
                Err(error) => {
                    if let Some(existing) = source_ids.get(&publisher_id) {
                        membership.invalidate_descriptor(existing.clone());
                    }
                    tracing::warn!(publisher_id, %error, "Ignoring malformed KV event source");
                    return;
                }
            };
            if source.kv_state_endpoint != *expected_endpoint || source.publisher_id != publisher_id
            {
                if let Some(existing) = source_ids.get(&publisher_id) {
                    membership.invalidate_descriptor(existing.clone());
                }
                tracing::warn!(
                    publisher_id,
                    "Ignoring KV event source with inconsistent attribution"
                );
                return;
            }

            let source_id = source.source_id();
            if let Some(existing) = source_ids.get(&publisher_id) {
                if existing != &source_id {
                    membership.invalidate_descriptor(existing.clone());
                    tracing::warn!(
                        publisher_id,
                        "KV publisher changed its immutable logical source attribution"
                    );
                    return;
                }
            } else {
                source_ids.insert(publisher_id, source_id);
            }
            if let Err(error) = membership.add(source) {
                tracing::warn!(%error, "KV publisher changed its immutable source descriptor");
            }
        }
        Ok(DiscoveryEvent::Removed(DiscoveryInstanceId::EventSource(EventSourceInstanceId {
            scope,
            topic,
            publisher_id,
        }))) => {
            if scope != expected_scope || topic != KV_EVENT_SUBJECT {
                return;
            }
            let Some(source_id) = source_ids.remove(&publisher_id) else {
                return;
            };
            membership.remove(&source_id);
        }
        Ok(DiscoveryEvent::Added(_)) | Ok(DiscoveryEvent::Removed(_)) => {}
        Err(error) => tracing::error!(%error, "KV event-source discovery stream failed"),
    }
}

fn publish_view(
    sender: &watch::Sender<KvSourceMembershipView>,
    lifecycle: &mut LifecycleTracker,
    view: KvSourceMembershipView,
) {
    let view = lifecycle.apply(view);
    if *sender.borrow() != view {
        sender.send_replace(view);
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc, time::Duration};

    use dynamo_kv_router::protocols::WorkerWithDpRank;
    use dynamo_runtime::{
        component::{Instance, TransportType},
        discovery::{Discovery, DiscoverySpec, MockDiscovery, SharedMockRegistry},
    };

    use super::*;
    use crate::discovery::{KvSourceAmbiguity, KvSourceKey};
    use crate::local_model::runtime_config::ModelRuntimeConfig;

    fn endpoint(name: &str) -> EndpointId {
        EndpointId {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            name: name.to_string(),
        }
    }

    fn runtime_config(kv_state_endpoint: Option<EndpointId>) -> ModelRuntimeConfig {
        ModelRuntimeConfig {
            data_parallel_start_rank: 4,
            data_parallel_size: 1,
            kv_state_endpoint,
            ..Default::default()
        }
    }

    fn source(endpoint: &EndpointId, worker_id: u64, publisher_id: u64) -> KvEventSource {
        KvEventSource {
            kv_state_endpoint: endpoint.clone(),
            worker: WorkerWithDpRank::new(worker_id, 4),
            publisher_id,
            recovery_target: None,
        }
    }

    fn recoverable_source(
        endpoint: &EndpointId,
        worker_id: u64,
        publisher_id: u64,
    ) -> KvEventSource {
        KvEventSource {
            recovery_target: Some(Instance {
                namespace: "ns".to_string(),
                component: "query".to_string(),
                endpoint: "rank-4".to_string(),
                instance_id: publisher_id,
                transport: TransportType::Tcp("tcp://127.0.0.1:1234".to_string()),
                device_type: None,
            }),
            ..source(endpoint, worker_id, publisher_id)
        }
    }

    async fn register_source(
        discovery: &dyn Discovery,
        source: &KvEventSource,
    ) -> DiscoveryInstance {
        discovery
            .register(DiscoverySpec::EventSource {
                scope: EventScope::Endpoint {
                    endpoint: source.kv_state_endpoint.clone(),
                },
                topic: KV_EVENT_SUBJECT.to_string(),
                publisher_id: source.publisher_id,
                metadata: serde_json::to_value(source).unwrap(),
            })
            .await
            .unwrap()
    }

    async fn wait_for(
        watch: &mut KvSourceMembershipWatch,
        predicate: impl Fn(&KvSourceMembershipView) -> bool,
    ) {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if predicate(&watch.borrow()) {
                    return;
                }
                watch.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn exact_watch_observes_late_overlap_and_filters_unrelated_sources() {
        let serving = endpoint("generate");
        let kv = endpoint("kv");
        let other = endpoint("other");
        let configs = HashMap::from([(42, runtime_config(Some(kv.clone())))]);
        let (_configs_tx, configs_rx) = watch::channel(configs);
        let discovery: Arc<dyn Discovery> =
            Arc::new(MockDiscovery::new(Some(1), SharedMockRegistry::new()));
        let coordinator =
            KvSourceMembershipCoordinator::start(serving, configs_rx, discovery.clone());
        let mut watch = coordinator.subscribe();
        let worker = WorkerWithDpRank::new(42, 4);
        assert_eq!(
            watch.borrow().status(&worker),
            Some(&KvSourceStatus::Missing)
        );

        register_source(discovery.as_ref(), &source(&other, 42, 900)).await;
        register_source(discovery.as_ref(), &source(&kv, 99, 901)).await;
        let source_a = source(&kv, 42, 100);
        register_source(discovery.as_ref(), &source_a).await;
        wait_for(&mut watch, |view| {
            view.status(&worker) == Some(&KvSourceStatus::ActiveLiveOnly(source_a.clone()))
                && view.sources.len() == 1
        })
        .await;

        let generation_a = watch.borrow().lifecycle_generation(&worker).unwrap();
        let source_b = recoverable_source(&kv, 42, 205);
        register_source(discovery.as_ref(), &source_b).await;
        wait_for(&mut watch, |view| {
            matches!(
                view.status(&worker),
                Some(KvSourceStatus::Ambiguous(KvSourceAmbiguity::Incarnations {
                    publisher_ids
                })) if publisher_ids == &vec![100, 205]
            )
        })
        .await;
        assert!(watch.borrow().lifecycle_generation(&worker).unwrap() > generation_a);
    }

    #[tokio::test]
    async fn omitted_mapping_watches_the_serving_endpoint() {
        let serving = endpoint("generate");
        let configs = HashMap::from([(42, runtime_config(None))]);
        let (_configs_tx, configs_rx) = watch::channel(configs);
        let discovery: Arc<dyn Discovery> =
            Arc::new(MockDiscovery::new(Some(1), SharedMockRegistry::new()));
        let coordinator =
            KvSourceMembershipCoordinator::start(serving.clone(), configs_rx, discovery.clone());
        let mut watch = coordinator.subscribe();
        let worker = WorkerWithDpRank::new(42, 4);
        let fallback_source = source(&serving, 42, 100);

        register_source(discovery.as_ref(), &fallback_source).await;
        wait_for(&mut watch, |view| {
            view.resolved_kv_state_endpoint() == Some(&serving)
                && view.status(&worker)
                    == Some(&KvSourceStatus::ActiveLiveOnly(fallback_source.clone()))
        })
        .await;
    }

    #[tokio::test]
    async fn remap_rebinds_exact_watch_and_preserves_cumulative_reset_generation() {
        let serving = endpoint("generate");
        let kv_a = endpoint("kv-a");
        let kv_b = endpoint("kv-b");
        let (configs_tx, configs_rx) =
            watch::channel(HashMap::from([(42, runtime_config(Some(kv_a.clone())))]));
        let discovery: Arc<dyn Discovery> =
            Arc::new(MockDiscovery::new(Some(1), SharedMockRegistry::new()));
        let coordinator =
            KvSourceMembershipCoordinator::start(serving.clone(), configs_rx, discovery.clone());
        let mut slow_consumer = coordinator.subscribe();
        let worker = WorkerWithDpRank::new(42, 4);

        let source_a = source(&kv_a, 42, 100);
        let source_b = recoverable_source(&kv_b, 42, 205);
        let instance_a = register_source(discovery.as_ref(), &source_a).await;
        wait_for(&mut slow_consumer, |view| {
            view.status(&worker) == Some(&KvSourceStatus::ActiveLiveOnly(source_a.clone()))
        })
        .await;
        let generation_a = slow_consumer
            .borrow()
            .lifecycle_generation(&worker)
            .unwrap();

        configs_tx
            .send(HashMap::from([(42, runtime_config(Some(kv_b.clone())))]))
            .unwrap();
        register_source(discovery.as_ref(), &source_b).await;
        discovery.unregister(instance_a).await.unwrap();
        wait_for(&mut slow_consumer, |view| {
            view.resolved_kv_state_endpoint() == Some(&kv_b)
                && view.status(&worker)
                    == Some(&KvSourceStatus::ActiveRecoverable(source_b.clone()))
        })
        .await;

        assert!(
            slow_consumer
                .borrow()
                .lifecycle_generation(&worker)
                .unwrap()
                > generation_a
        );
        assert_eq!(slow_consumer.borrow().serving_endpoint, serving);
    }

    #[test]
    fn coalesced_ambiguity_still_advances_the_published_reset_fence() {
        let serving = endpoint("generate");
        let kv = endpoint("kv");
        let configs = HashMap::from([(42, runtime_config(Some(kv.clone())))]);
        let worker = WorkerWithDpRank::new(42, 4);
        let mut membership = KvSourceMembership::new();
        let mut lifecycle = LifecycleTracker::default();

        membership.add(source(&kv, 42, 100)).unwrap();
        let active_a = lifecycle.apply(membership.view(&serving, &configs));
        membership.add(source(&kv, 42, 205)).unwrap();
        let _unobserved_ambiguity = lifecycle.apply(membership.view(&serving, &configs));
        membership
            .remove(&KvSourceId {
                key: KvSourceKey::new(kv.clone(), worker),
                publisher_id: 100,
            })
            .unwrap();
        let active_b = lifecycle.apply(membership.view(&serving, &configs));

        assert!(
            active_b.lifecycle_generation(&worker).unwrap()
                > active_a.lifecycle_generation(&worker).unwrap()
        );
    }

    #[test]
    fn immutable_recovery_target_violation_fails_closed_and_advances_generation() {
        let serving = endpoint("generate");
        let kv = endpoint("kv");
        let configs = HashMap::from([(42, runtime_config(Some(kv.clone())))]);
        let worker = WorkerWithDpRank::new(42, 4);
        let mut membership = KvSourceMembership::new();
        let mut lifecycle = LifecycleTracker::default();

        membership.add(source(&kv, 42, 100)).unwrap();
        let active = lifecycle.apply(membership.view(&serving, &configs));
        assert!(membership.add(recoverable_source(&kv, 42, 100)).is_err());
        let invalid = lifecycle.apply(membership.view(&serving, &configs));

        assert!(matches!(
            invalid.status(&worker),
            Some(KvSourceStatus::Ambiguous(
                KvSourceAmbiguity::ConflictingDescriptor { publisher_id: 100 }
            ))
        ));
        assert!(
            invalid.lifecycle_generation(&worker).unwrap()
                > active.lifecycle_generation(&worker).unwrap()
        );
    }
}
