// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::namespace_source::{NamespaceSelection, NamespaceSource, NamespaceUpdates};
use super::*;
use crate::kv_dc_relay::sources::KvDcRelaySourcesStatus;
use tokio_util::sync::DropGuard;

const SETUP_TIMEOUT: Duration = Duration::from_secs(10);

type SourceWatchStream = Pin<Box<dyn Stream<Item = (String, u64, Option<DiscoveryEvent>)> + Send>>;

struct NamespaceWatch {
    epoch: u64,
    _guard: DropGuard,
}

struct NamespaceMembership {
    discovery: Arc<dyn Discovery>,
    filter: DcDiscoveryFilter,
    state: MembershipState,
    watches: HashMap<String, NamespaceWatch>,
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
    ) -> anyhow::Result<()> {
        // Stage additions before publishing a new source set. Dropped guards cancel any
        // partially opened watches if a snapshot or subsequent watch setup fails.
        let mut additions = HashMap::new();
        let mut streams = Vec::<SourceWatchStream>::new();
        for source in &selection.namespaces {
            if self.watches.contains_key(source) {
                continue;
            }
            let token = cancel.child_token();
            let guard = token.clone().drop_guard();
            let query = DiscoveryQuery::NamespacedModels {
                namespace: source.clone(),
            };
            let stream = self.discovery.list_and_watch(query, Some(token)).await?;
            self.epoch = self
                .epoch
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("sources watch epoch exhausted"))?;
            let epoch = self.epoch;
            let namespace = source.clone();
            // Notifications trigger a fresh snapshot rather than replaying old
            // Added records over a newer list result.
            let notifications = stream.map(move |event| (namespace.clone(), epoch, event.ok()));
            let namespace = source.clone();
            streams.push(Box::pin(notifications.chain(futures::stream::once(
                async move { (namespace, epoch, None) },
            ))));
            additions.insert(
                source.clone(),
                NamespaceWatch {
                    epoch,
                    _guard: guard,
                },
            );
        }
        let queries = selection
            .namespaces
            .iter()
            .map(|s| DiscoveryQuery::NamespacedModels {
                namespace: s.clone(),
            })
            .collect::<Vec<_>>();
        let instances = list_queries(&self.discovery, &queries).await?;

        // Namespace identity is independent of the lifetime of any DGD or worker.
        self.watches
            .retain(|namespace, _| selection.namespaces.contains(namespace));
        self.watches.extend(additions);
        for stream in streams {
            self.streams.push(stream);
        }
        if self.state.replace_all(instances, &self.filter) {
            publish_membership_if_changed(&self.sender, self.state.view(&self.filter));
        }
        self.selection = Some(selection);
        Ok(())
    }

    async fn bounded_apply(
        &mut self,
        selection: NamespaceSelection,
        cancel: &CancellationToken,
    ) -> anyhow::Result<()> {
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
        let mut desired = self.selection.clone();
        let mut retry = tokio::time::interval(Duration::from_secs(1));
        retry.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut reconcile = tokio::time::interval(RECONCILE_INTERVAL);
        reconcile.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            let update = tokio::select! {
                _ = cancel.cancelled() => return,
                next = updates.next() => {
                    match next {
                        Some(Ok(selection)) => {
                            source_invalid = false;
                            desired = Some(selection.clone());
                            status.send_modify(|s| s.desired_revision = selection.revision.clone());
                            if self.selection.as_ref() == Some(&selection) && self.watches.len() == selection.namespaces.len() {
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
                _ = reconcile.tick() => desired.clone(),
                _ = retry.tick(), if desired != self.selection || desired.as_ref().is_some_and(|selection| self.watches.len() != selection.namespaces.len()) => desired.clone(),
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
                            } else {desired.clone()}
                        }
                        _ => None,
                    }
                },
            };
            if let Some(selection) = update {
                let revision = selection.revision.clone();
                let count = selection.namespaces.len();
                match self.bounded_apply(selection, &cancel).await {
                    Ok(()) => status.send_modify(|s| {
                        s.applied_revision = revision;
                        s.count = count;
                        if !source_invalid {
                            s.last_error = None;
                        }
                    }),
                    Err(_) => status.send_modify(|s| {
                        s.last_error = Some(
                            "Sources discovery reconciliation failed; retaining last applied sources"
                                .into(),
                        )
                    }),
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
        let count = selection.namespaces.len();
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
        membership.bounded_apply(selection, &cancel).await?;
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
    use crate::kv_dc_relay::sources::{KvDcRelaySourcesFile, RelaySource, SourcesDocument};
    use dynamo_runtime::discovery::{DiscoverySpec, MockDiscovery, SharedMockRegistry};

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
        let epoch = membership.watches["a"].epoch;
        membership
            .bounded_apply(document(&["a", "b"]).into(), &cancel)
            .await
            .unwrap();
        assert_eq!(membership.watches["a"].epoch, epoch);
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
        assert!(!membership.watches.contains_key("b"));
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
