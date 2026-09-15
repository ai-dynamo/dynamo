// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::discovery::{DcDiscoveryFilter, DcMembershipView, MembershipState};
use super::namespace_source::{KvDcRelaySourcesStatus, discovery::DiscoveryNamespaces};
use super::namespace_source::{
    NamespaceScope, NamespaceSelection, NamespaceSource, NamespaceUpdates,
};
use dynamo_runtime::discovery::{
    Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
};
use futures::{Stream, StreamExt, future::try_join_all};
use std::{
    collections::{HashMap, HashSet},
    pin::Pin,
    sync::Arc,
    time::Duration,
};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tokio_util::sync::DropGuard;

const RECONCILE_INTERVAL: Duration = Duration::from_secs(30);

pub(crate) struct DcMembershipWatch {
    receiver: watch::Receiver<DcMembershipView>,
    cancel: CancellationToken,
    task: JoinHandle<()>,
    sources_status: watch::Receiver<KvDcRelaySourcesStatus>,
}

impl DcMembershipWatch {
    pub(crate) async fn start_sources(
        discovery: Arc<dyn Discovery>,
        sources: super::host::KvDcRelaySources,
        parent_cancel: CancellationToken,
    ) -> anyhow::Result<Self> {
        let (source, filter): (Box<dyn NamespaceSource>, _) = match sources {
            super::host::KvDcRelaySources::File(file) => {
                (Box::new(file), DcDiscoveryFilter::default())
            }
            super::host::KvDcRelaySources::Discovery(config) => {
                config.validate()?;
                let filter = config.filter();
                (Box::new(DiscoveryNamespaces { config }), filter)
            }
        };
        Self::start_namespace_source(discovery, source, filter, parent_cancel).await
    }

    pub(crate) fn sources_status(&self) -> KvDcRelaySourcesStatus {
        self.sources_status.borrow().clone()
    }

    pub(crate) fn subscribe(&self) -> watch::Receiver<DcMembershipView> {
        self.receiver.clone()
    }

    pub(crate) async fn shutdown(self) {
        self.cancel.cancel();
        if let Err(error) = self.task.await
            && !error.is_cancelled()
        {
            tracing::warn!(%error, "KV DC Relay model-card watch failed during shutdown");
        }
    }
}

pub(super) fn publish_membership_if_changed(
    sender: &watch::Sender<DcMembershipView>,
    next: DcMembershipView,
) {
    sender.send_if_modified(move |current| {
        if current == &next {
            return false;
        }
        *current = next;
        true
    });
}

async fn list_queries(
    discovery: &Arc<dyn Discovery>,
    queries: &[DiscoveryQuery],
) -> anyhow::Result<Vec<DiscoveryInstance>> {
    let results = try_join_all(queries.iter().cloned().map(|query| discovery.list(query))).await?;
    Ok(results.into_iter().flatten().collect())
}

const SETUP_TIMEOUT: Duration = Duration::from_secs(10);

type SourceWatchStream =
    Pin<Box<dyn Stream<Item = (DiscoveryQuery, u64, Option<DiscoveryEvent>)> + Send>>;

struct NamespaceWatch {
    epoch: u64,
    _guard: DropGuard,
}

struct NamespaceMembership {
    discovery: Arc<dyn Discovery>,
    filter: DcDiscoveryFilter,
    state: MembershipState,
    watches: HashMap<DiscoveryQuery, NamespaceWatch>,
    streams: futures::stream::SelectAll<SourceWatchStream>,
    epoch: u64,
    selection: Option<NamespaceSelection>,
    sender: watch::Sender<DcMembershipView>,
}

impl NamespaceMembership {
    async fn apply(
        &mut self,
        selection: NamespaceSelection,
        cancel: &CancellationToken,
    ) -> anyhow::Result<usize> {
        // Stage additions before publishing a new source set. Dropped guards cancel any
        // partially opened watches if a snapshot or subsequent watch setup fails.
        let mut additions = HashMap::new();
        let mut streams = Vec::<SourceWatchStream>::new();
        let queries = selection.scope.queries();
        for key in &queries {
            if self.watches.contains_key(key) {
                continue;
            }
            let token = cancel.child_token();
            let guard = token.clone().drop_guard();
            let stream = self
                .discovery
                .list_and_watch(key.clone(), Some(token))
                .await?;
            self.epoch = self
                .epoch
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("sources watch epoch exhausted"))?;
            let epoch = self.epoch;
            let query = key.clone();
            // Notifications trigger a fresh snapshot rather than replaying old
            // Added records over a newer list result.
            let notifications = stream.map(move |event| (query.clone(), epoch, event.ok()));
            let query = key.clone();
            streams.push(Box::pin(
                notifications.chain(futures::stream::once(async move { (query, epoch, None) })),
            ));
            additions.insert(
                key.clone(),
                NamespaceWatch {
                    epoch,
                    _guard: guard,
                },
            );
        }
        let instances = list_queries(&self.discovery, &queries).await?;

        let count = match &selection.scope {
            NamespaceScope::Namespaces(namespaces) => namespaces.len(),
            NamespaceScope::All => instances
                .iter()
                .filter_map(|instance| match instance.id() {
                    DiscoveryInstanceId::Model(id) => Some(id.namespace),
                    _ => None,
                })
                .collect::<HashSet<_>>()
                .len(),
        };

        // Namespace identity is independent of the lifetime of any DGD or worker.
        self.watches.retain(|query, _| queries.contains(query));
        self.watches.extend(additions);
        for stream in streams {
            self.streams.push(stream);
        }
        if self.state.replace_all(instances, &self.filter) {
            publish_membership_if_changed(&self.sender, self.state.view(&self.filter));
        }
        self.selection = Some(selection);
        Ok(count)
    }

    async fn bounded_apply(
        &mut self,
        selection: NamespaceSelection,
        cancel: &CancellationToken,
    ) -> anyhow::Result<usize> {
        tokio::select! {
            _ = cancel.cancelled() => anyhow::bail!("sources update cancelled"),
            result = tokio::time::timeout(SETUP_TIMEOUT, self.apply(selection, cancel)) => {
                result.map_err(|_| anyhow::anyhow!("sources discovery setup timed out"))?
            }
        }
    }

    async fn run(
        mut self,
        mut updates: NamespaceUpdates,
        status: watch::Sender<KvDcRelaySourcesStatus>,
        cancel: CancellationToken,
    ) {
        let mut source_invalid = false;
        let mut discovery_dirty = false;
        let mut refresh = tokio::time::interval(Duration::from_millis(50));
        refresh.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut last_reconcile_error = None;
        let mut desired = self.selection.clone();
        let mut retry = tokio::time::interval(Duration::from_secs(1));
        retry.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut reconcile = tokio::time::interval(RECONCILE_INTERVAL);
        reconcile.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // Startup already installed a snapshot; wait for the first refresh window.
        reconcile.tick().await;
        refresh.tick().await;
        loop {
            let update = tokio::select! {
                _ = cancel.cancelled() => return,
                next = updates.next() => {
                    match next {
                        Some(Ok(selection)) => {
                            source_invalid = false;
                            desired = Some(selection.clone());
                            status.send_modify(|s| s.desired_revision = selection.revision.clone());
                            if self.selection.as_ref() == Some(&selection) && self.watches.len() == selection.scope.watch_count() {
                                if status.borrow().last_error.is_some() { Some(selection) } else { None }
                            } else { Some(selection) }
                        }
                        Some(Err(error)) => {
                            source_invalid = true;
                            status.send_modify(|s| {
                                s.desired_revision = None;
                                s.last_error = Some(error.to_string());
                            });
                            None
                        }
                        None => {
                            source_invalid = true;
                            status.send_modify(|s| s.last_error = Some("Namespace source closed; retaining last applied sources".into()));
                            updates = Box::pin(futures::stream::pending());
                            None
                        },
                    }
                },
                _ = refresh.tick(), if discovery_dirty => {
                    discovery_dirty = false;
                    desired.clone()
                },
                _ = reconcile.tick() => desired.clone(),
                _ = retry.tick(), if desired != self.selection || desired.as_ref().is_some_and(|selection| self.watches.len() != selection.scope.watch_count()) => desired.clone(),
                event = self.streams.next(), if !self.streams.is_empty() => {
                    match event {
                        Some((namespace, epoch, event)) if self.watches.get(&namespace).is_some_and(|w| w.epoch == epoch) => {
                            if event.is_none() {
                                self.watches.remove(&namespace);
                                status.send_modify(|s| s.last_error = Some("Sources discovery watch closed; retrying".into()));
                                None
                            } else if let Some(event @ DiscoveryEvent::ModelTaintsUpdated(_)) = event {
                                // Taints are runtime updates, not namespace selection changes.
                                if self.state.apply(event, &self.filter) {
                                    publish_membership_if_changed(&self.sender, self.state.view(&self.filter));
                                }
                                None
                            } else {
                                // Coalesce replay and update bursts before reading a fresh snapshot.
                                discovery_dirty = true;
                                None
                            }
                        }
                        _ => None,
                    }
                },
            };
            if let Some(selection) = update {
                let revision = selection.revision.clone();
                let query_count = selection.scope.watch_count();
                match self.bounded_apply(selection, &cancel).await {
                    Ok(count) => {
                        last_reconcile_error = None;
                        status.send_modify(|s| {
                            s.applied_revision = revision;
                            s.count = count;
                            if !source_invalid {
                                s.last_error = None;
                            }
                        });
                    }
                    Err(error) => {
                        let detail = format!("{error:#}");
                        if last_reconcile_error.as_ref() != Some(&detail) {
                            tracing::warn!(
                                error = %detail,
                                query_count,
                                "KV DC Relay sources reconciliation failed; retaining last applied sources"
                            );
                            last_reconcile_error = Some(detail);
                        }
                        status.send_modify(|s| {
                            s.last_error = Some(
                                "Sources discovery reconciliation failed; retaining last applied sources"
                                    .into(),
                            );
                        });
                    }
                }
            }
        }
    }
}

impl DcMembershipWatch {
    pub(super) async fn start_namespace_source(
        discovery: Arc<dyn Discovery>,
        source: Box<dyn NamespaceSource>,
        filter: DcDiscoveryFilter,
        parent_cancel: CancellationToken,
    ) -> anyhow::Result<Self> {
        let cancel = parent_cancel.child_token();
        let guard = cancel.clone().drop_guard();
        let mut updates = source.updates(cancel.clone());
        let selection = tokio::select! {
            _ = cancel.cancelled() => anyhow::bail!("sources startup cancelled"),
            result = tokio::time::timeout(SETUP_TIMEOUT, updates.next()) => {
                result.map_err(|_| anyhow::anyhow!("sources startup timed out"))?
                    .ok_or_else(|| anyhow::anyhow!("namespace source closed during startup"))??
            }
        };
        let revision = selection.revision.clone();
        let mut state = MembershipState::default();
        let (sender, receiver) = watch::channel(state.view(&filter));
        let mut membership = NamespaceMembership {
            discovery,
            filter,
            state,
            watches: HashMap::new(),
            streams: Default::default(),
            epoch: 0,
            selection: None,
            sender,
        };
        let count = membership.bounded_apply(selection, &cancel).await?;
        let (status, sources_status) = watch::channel(KvDcRelaySourcesStatus {
            desired_revision: revision.clone(),
            applied_revision: revision,
            count,
            last_error: None,
        });
        let task_cancel = cancel.clone();
        let task = tokio::spawn(async move {
            membership.run(updates, status, task_cancel).await;
        });
        guard.disarm();
        Ok(Self {
            receiver,
            cancel,
            task,
            sources_status,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_dc_relay::namespace_source::discovery::KvDcRelayDiscoveryConfig;
    use crate::kv_dc_relay::namespace_source::file::{
        KvDcRelaySourcesFile, RelaySource, SourcesDocument,
    };
    use crate::{model_card::ModelDeploymentCard, worker_type::WorkerType};
    use dynamo_runtime::discovery::{DiscoverySpec, MockDiscovery, SharedMockRegistry};
    use dynamo_runtime::protocols::EndpointId;

    struct CountingDiscovery {
        inner: MockDiscovery,
        lists: std::sync::atomic::AtomicUsize,
        watches: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl Discovery for CountingDiscovery {
        fn instance_id(&self) -> u64 {
            self.inner.instance_id()
        }

        async fn register_internal(
            &self,
            spec: DiscoverySpec,
        ) -> anyhow::Result<DiscoveryInstance> {
            self.inner.register_internal(spec).await
        }

        async fn unregister(&self, instance: DiscoveryInstance) -> anyhow::Result<()> {
            self.inner.unregister(instance).await
        }

        async fn list(&self, query: DiscoveryQuery) -> anyhow::Result<Vec<DiscoveryInstance>> {
            self.lists
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.list(query).await
        }

        async fn list_and_watch(
            &self,
            query: DiscoveryQuery,
            cancel: Option<CancellationToken>,
        ) -> anyhow::Result<dynamo_runtime::discovery::DiscoveryStream> {
            self.watches
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.list_and_watch(query, cancel).await
        }
    }

    #[tokio::test(start_paused = true)]
    async fn discovery_replay_coalesces_snapshot_reads() {
        for watch_all in [false, true] {
            let discovery = Arc::new(CountingDiscovery {
                inner: MockDiscovery::new(Some(1), SharedMockRegistry::new()),
                lists: Default::default(),
                watches: Default::default(),
            });
            let namespaces = (0..20).map(|i| format!("ns{i}")).collect::<Vec<_>>();
            for namespace in &namespaces {
                for index in 0..10 {
                    let mut card =
                        ModelDeploymentCard::with_name_only(&format!("{namespace}-{index}"));
                    card.source_path = Some(format!("test/{namespace}/{index}"));
                    card.kv_cache_block_size = 64;
                    card.worker_type = Some(WorkerType::Aggregated);
                    discovery
                        .register(DiscoverySpec::Model {
                            namespace: namespace.clone(),
                            component: format!("worker{index}"),
                            endpoint: "generate".into(),
                            card_json: serde_json::to_value(card).unwrap(),
                            model_suffix: None,
                        })
                        .await
                        .unwrap();
                }
            }
            discovery
                .lists
                .store(0, std::sync::atomic::Ordering::Relaxed);
            let relay = DcMembershipWatch::start_sources(
                discovery.clone(),
                super::super::host::KvDcRelaySources::Discovery(KvDcRelayDiscoveryConfig {
                    namespaces: if watch_all { vec![] } else { namespaces },
                    watch_all,
                    ..Default::default()
                }),
                CancellationToken::new(),
            )
            .await
            .unwrap();
            // Virtual time bounds the observation before periodic reconciliation.
            tokio::time::sleep(Duration::from_millis(100)).await;
            assert_eq!(relay.subscribe().borrow().endpoints.len(), 200);
            let lists = discovery.lists.load(std::sync::atomic::Ordering::Relaxed);
            assert!(
                lists <= if watch_all { 3 } else { 60 },
                "watch_all={watch_all}: {lists} snapshot reads for one replay"
            );
            assert_eq!(relay.sources_status().count, 20);
            assert_eq!(
                discovery.watches.load(std::sync::atomic::Ordering::Relaxed),
                if watch_all { 1 } else { 20 }
            );
            relay.shutdown().await;
        }
    }

    fn document(names: &[&str]) -> SourcesDocument {
        let mut doc = SourcesDocument {
            version: 1,
            revision: String::new(),
            connection_revision: Some("connection".into()),
            sources: names
                .iter()
                .map(|name| RelaySource {
                    namespace: (*name).into(),
                })
                .collect(),
        };
        doc.canonicalize(Some("connection")).unwrap();
        doc
    }

    #[tokio::test]
    async fn source_updates_preserve_unchanged_membership_and_watch_epoch() {
        let discovery: Arc<dyn Discovery> =
            Arc::new(MockDiscovery::new(Some(1), SharedMockRegistry::new()));
        let mut registrations = Vec::new();
        for namespace in ["a", "b"] {
            let mut card = ModelDeploymentCard::with_name_only(namespace);
            card.source_path = Some(format!("test/{namespace}"));
            card.kv_cache_block_size = 64;
            card.worker_type = Some(WorkerType::Aggregated);
            registrations.push(
                discovery
                    .register(DiscoverySpec::Model {
                        namespace: namespace.into(),
                        component: "worker".into(),
                        endpoint: "generate".into(),
                        card_json: serde_json::to_value(card).unwrap(),
                        model_suffix: None,
                    })
                    .await
                    .unwrap(),
            );
        }
        let mut state = MembershipState::default();
        let (sender, receiver) = watch::channel(state.view(&DcDiscoveryFilter::default()));
        let mut membership = NamespaceMembership {
            discovery,
            filter: DcDiscoveryFilter::default(),
            state,
            watches: HashMap::new(),
            streams: Default::default(),
            epoch: 0,
            selection: None,
            sender,
        };
        let cancel = CancellationToken::new();
        membership
            .bounded_apply(document(&["a"]).into(), &cancel)
            .await
            .unwrap();
        let endpoint = EndpointId::from("a.worker.generate");
        let generation = receiver.borrow().endpoints[&endpoint].generation;
        let epoch = membership.watches[&DiscoveryQuery::NamespacedModels {
            namespace: "a".into(),
        }]
            .epoch;
        membership
            .bounded_apply(document(&["a", "b"]).into(), &cancel)
            .await
            .unwrap();
        assert_eq!(
            membership.watches[&DiscoveryQuery::NamespacedModels {
                namespace: "a".into()
            }]
                .epoch,
            epoch
        );
        assert_eq!(
            receiver.borrow().endpoints[&endpoint].generation,
            generation
        );
        assert_eq!(receiver.borrow().endpoints.len(), 2);
        membership
            .bounded_apply(document(&["a"]).into(), &cancel)
            .await
            .unwrap();
        assert_eq!(receiver.borrow().endpoints.len(), 1);
        assert!(
            !membership
                .watches
                .contains_key(&DiscoveryQuery::NamespacedModels {
                    namespace: "b".into()
                })
        );
        membership
            .bounded_apply(document(&[]).into(), &cancel)
            .await
            .unwrap();
        assert!(receiver.borrow().endpoints.is_empty());
        assert!(membership.watches.is_empty());
        cancel.cancel();
        drop(registrations);
    }

    #[tokio::test]
    async fn file_reload_accepts_atomic_replacement_and_keeps_last_valid_sources() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("sources.json");
        let initial = document(&[]);
        tokio::fs::write(&path, serde_json::to_vec(&initial).unwrap())
            .await
            .unwrap();
        let discovery: Arc<dyn Discovery> =
            Arc::new(MockDiscovery::new(Some(1), SharedMockRegistry::new()));
        let membership = DcMembershipWatch::start_namespace_source(
            discovery,
            Box::new(KvDcRelaySourcesFile {
                path: path.clone(),
                connection_revision: Some("connection".into()),
            }),
            DcDiscoveryFilter::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        let mut status = membership.sources_status.clone();
        let next = document(&["a"]);
        let staged = directory.path().join("next.json");
        tokio::fs::write(&staged, serde_json::to_vec(&next).unwrap())
            .await
            .unwrap();
        tokio::fs::rename(&staged, &path).await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if status.borrow().applied_revision.as_ref() == Some(&next.revision) {
                    break;
                }
                status.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        tokio::fs::write(&path, b"{private-invalid-json")
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if status.borrow().last_error.is_some() {
                    break;
                }
                status.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        assert_eq!(
            status.borrow().applied_revision.as_ref(),
            Some(&next.revision)
        );
        tokio::time::timeout(Duration::from_secs(2), membership.shutdown())
            .await
            .unwrap();
    }
    async fn register_model(
        discovery: &Arc<dyn Discovery>,
        namespace: &str,
        component: &str,
    ) -> DiscoveryInstance {
        let mut card = ModelDeploymentCard::with_name_only(namespace);
        card.source_path = Some(format!("test/{namespace}"));
        card.kv_cache_block_size = 64;
        card.worker_type = Some(WorkerType::Aggregated);
        discovery
            .register(DiscoverySpec::Model {
                namespace: namespace.into(),
                component: component.into(),
                endpoint: "generate".into(),
                card_json: serde_json::to_value(card).unwrap(),
                model_suffix: None,
            })
            .await
            .unwrap()
    }

    async fn wait_for_count(membership: &DcMembershipWatch, count: usize) {
        let mut status = membership.sources_status.clone();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if status.borrow().count == count && status.borrow().last_error.is_none() {
                    break;
                }
                status.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn discovery_tracks_added_and_removed_namespaces_with_endpoint_filters() {
        let discovery: Arc<dyn Discovery> =
            Arc::new(MockDiscovery::new(Some(1), SharedMockRegistry::new()));
        let membership = DcMembershipWatch::start_sources(
            discovery.clone(),
            crate::kv_dc_relay::host::KvDcRelaySources::Discovery(KvDcRelayDiscoveryConfig {
                watch_all: true,
                endpoint_prefixes: vec!["a.worker".into()],
                ..Default::default()
            }),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        let a = register_model(&discovery, "a", "worker").await;
        let b = register_model(&discovery, "b", "worker").await;
        wait_for_count(&membership, 2).await;
        assert_eq!(membership.receiver.borrow().endpoints.len(), 1);
        assert!(
            membership
                .receiver
                .borrow()
                .endpoints
                .contains_key(&EndpointId::from("a.worker.generate"))
        );
        discovery.unregister(a).await.unwrap();
        wait_for_count(&membership, 1).await;
        assert!(membership.receiver.borrow().endpoints.is_empty());
        discovery.unregister(b).await.unwrap();
        wait_for_count(&membership, 0).await;
        membership.shutdown().await;
    }

    struct TestSource(tokio::sync::mpsc::UnboundedReceiver<anyhow::Result<NamespaceSelection>>);

    impl NamespaceSource for TestSource {
        fn updates(mut self: Box<Self>, cancel: CancellationToken) -> NamespaceUpdates {
            Box::pin(async_stream::stream! {
                loop {
                    tokio::select! {
                        _ = cancel.cancelled() => break,
                        update = self.0.recv() => match update {
                            Some(update) => yield update,
                            None => break,
                        }
                    }
                }
            })
        }
    }

    #[tokio::test]
    async fn source_error_retains_membership_until_a_successful_empty_snapshot() {
        let discovery: Arc<dyn Discovery> =
            Arc::new(MockDiscovery::new(Some(1), SharedMockRegistry::new()));
        let _registration = register_model(&discovery, "a", "worker").await;
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        sender.send(Ok(document(&["a"]).into())).unwrap();
        let membership = DcMembershipWatch::start_namespace_source(
            discovery,
            Box::new(TestSource(receiver)),
            DcDiscoveryFilter::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        let mut status = membership.sources_status.clone();
        sender
            .send(Err(anyhow::anyhow!("source temporarily unavailable")))
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if status.borrow().last_error.is_some() {
                    break;
                }
                status.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        assert_eq!(membership.receiver.borrow().endpoints.len(), 1);
        assert_eq!(status.borrow().count, 1);
        sender.send(Ok(document(&[]).into())).unwrap();
        wait_for_count(&membership, 0).await;
        assert!(membership.receiver.borrow().endpoints.is_empty());
        membership.shutdown().await;
    }
}
