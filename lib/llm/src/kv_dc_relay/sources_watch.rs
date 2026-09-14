// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::kv_dc_relay::sources::{KvDcRelaySourcesFile, KvDcRelaySourcesStatus, SourcesDocument};
use tokio_util::sync::DropGuard;

const SETUP_TIMEOUT: Duration = Duration::from_secs(10);

type SourceWatchStream = Pin<Box<dyn Stream<Item = (String, u64, bool)> + Send>>;

struct NamespaceWatch {
    epoch: u64,
    _guard: DropGuard,
}

struct FileMembership {
    discovery: Arc<dyn Discovery>,
    state: MembershipState,
    watches: HashMap<String, NamespaceWatch>,
    streams: futures::stream::SelectAll<SourceWatchStream>,
    epoch: u64,
    document: Option<SourcesDocument>,
    sender: watch::Sender<DcMembershipView>,
}

impl FileMembership {
    async fn apply(
        &mut self,
        document: SourcesDocument,
        cancel: &CancellationToken,
    ) -> anyhow::Result<()> {
        // Stage additions before publishing a new source set. Dropped guards cancel any
        // partially opened watches if a snapshot or subsequent watch setup fails.
        let mut additions = HashMap::new();
        let mut streams = Vec::<SourceWatchStream>::new();
        for source in &document.sources {
            if self.watches.contains_key(&source.namespace) {
                continue;
            }
            let token = cancel.child_token();
            let guard = token.clone().drop_guard();
            let query = DiscoveryQuery::NamespacedModels {
                namespace: source.namespace.clone(),
            };
            let stream = self.discovery.list_and_watch(query, Some(token)).await?;
            self.epoch = self
                .epoch
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("sources watch epoch exhausted"))?;
            let epoch = self.epoch;
            let namespace = source.namespace.clone();
            // Notifications trigger a fresh snapshot rather than replaying old
            // Added records over a newer list result.
            let notifications = stream.map(move |event| (namespace.clone(), epoch, event.is_ok()));
            let namespace = source.namespace.clone();
            streams.push(Box::pin(notifications.chain(futures::stream::once(
                async move { (namespace, epoch, false) },
            ))));
            additions.insert(
                source.namespace.clone(),
                NamespaceWatch {
                    epoch,
                    _guard: guard,
                },
            );
        }
        let queries = document
            .sources
            .iter()
            .map(|s| DiscoveryQuery::NamespacedModels {
                namespace: s.namespace.clone(),
            })
            .collect::<Vec<_>>();
        let instances = list_queries(&self.discovery, &queries).await?;

        // Namespace identity is independent of the lifetime of any DGD or worker.
        self.watches.retain(|namespace, _| {
            document
                .sources
                .iter()
                .any(|source| &source.namespace == namespace)
        });
        self.watches.extend(additions);
        for stream in streams {
            self.streams.push(stream);
        }
        let filter = DcDiscoveryFilter::default();
        self.state.replace_all(instances, &filter);
        publish_membership_if_changed(&self.sender, self.state.view(&filter));
        self.document = Some(document);
        Ok(())
    }

    async fn bounded_apply(
        &mut self,
        document: SourcesDocument,
        cancel: &CancellationToken,
    ) -> anyhow::Result<()> {
        tokio::select! {
            _ = cancel.cancelled() => anyhow::bail!("sources update cancelled"),
            result = tokio::time::timeout(SETUP_TIMEOUT, self.apply(document, cancel)) => {
                result.map_err(|_| anyhow::anyhow!("sources discovery setup timed out"))?
            }
        }
    }

    async fn run(
        mut self,
        file: KvDcRelaySourcesFile,
        status: watch::Sender<KvDcRelaySourcesStatus>,
        cancel: CancellationToken,
    ) {
        let mut file_invalid = false;
        let mut poll = tokio::time::interval(Duration::from_secs(1));
        poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut reconcile = tokio::time::interval(RECONCILE_INTERVAL);
        reconcile.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            let update = tokio::select! {
                _ = cancel.cancelled() => return,
                _ = poll.tick() => {
                    let read = tokio::select! {
                        _ = cancel.cancelled() => return,
                        result = tokio::time::timeout(SETUP_TIMEOUT, file.read()) => result,
                    };
                    match read {
                        Ok(Ok(document)) => {
                            file_invalid = false;
                            status.send_modify(|s| s.desired_revision = Some(document.revision.clone()));
                            if self.document.as_ref() == Some(&document) && self.watches.len() == document.sources.len() {
                                status.send_modify(|s| s.last_error = None);
                                None
                            } else {Some(document)}
                        }
                        _ => {
                            file_invalid = true;
                            status.send_modify(|s| { s.desired_revision = None; s.last_error = Some("Sources file unavailable, invalid, or incompatible with this connection".into()); });
                            None
                        }
                    }
                },
                _ = reconcile.tick() => self.document.clone(),
                event = self.streams.next(), if !self.streams.is_empty() => {
                    match event {
                        Some((namespace, epoch, healthy)) if self.watches.get(&namespace).is_some_and(|w| w.epoch == epoch) => {
                            if !healthy {
                                self.watches.remove(&namespace);
                                status.send_modify(|s| s.last_error = Some("Sources discovery watch closed; retrying".into()));
                                None
                            } else {self.document.clone()}
                        }
                        _ => None,
                    }
                },
            };
            if let Some(document) = update {
                let revision = document.revision.clone();
                let count = document.sources.len();
                match self.bounded_apply(document, &cancel).await {
                    Ok(()) => status.send_modify(|s| {
                        s.applied_revision = Some(revision);
                        s.count = count;
                        if !file_invalid {
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
    pub(super) async fn start_file(
        discovery: Arc<dyn Discovery>,
        file: KvDcRelaySourcesFile,
        parent_cancel: CancellationToken,
    ) -> anyhow::Result<Self> {
        let cancel = parent_cancel.child_token();
        let guard = cancel.clone().drop_guard();
        let document = tokio::select! {
            _ = cancel.cancelled() => anyhow::bail!("sources startup cancelled"),
            result = tokio::time::timeout(SETUP_TIMEOUT, file.read()) => result.map_err(|_| anyhow::anyhow!("sources file startup timed out"))??,
        };
        let revision = document.revision.clone();
        let count = document.sources.len();
        let mut state = MembershipState::default();
        let (sender, receiver) = watch::channel(state.view(&DcDiscoveryFilter::default()));
        let mut membership = FileMembership {
            discovery,
            state,
            watches: HashMap::new(),
            streams: Default::default(),
            epoch: 0,
            document: None,
            sender,
        };
        membership.bounded_apply(document, &cancel).await?;
        let (status, sources_status) = watch::channel(KvDcRelaySourcesStatus {
            desired_revision: Some(revision.clone()),
            applied_revision: Some(revision),
            count,
            last_error: None,
        });
        let task_cancel = cancel.clone();
        let task = tokio::spawn(async move {
            membership.run(file, status, task_cancel).await;
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
    use crate::kv_dc_relay::sources::RelaySource;
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
        let mut membership = FileMembership {
            discovery,
            state,
            watches: HashMap::new(),
            streams: Default::default(),
            epoch: 0,
            document: None,
            sender,
        };
        let cancel = CancellationToken::new();
        membership
            .bounded_apply(document(&["a"]), &cancel)
            .await
            .unwrap();
        let endpoint = EndpointId::from("a.worker.generate");
        let generation = receiver.borrow().endpoints[&endpoint].generation;
        let epoch = membership.watches["a"].epoch;
        membership
            .bounded_apply(document(&["a", "b"]), &cancel)
            .await
            .unwrap();
        assert_eq!(membership.watches["a"].epoch, epoch);
        assert_eq!(
            receiver.borrow().endpoints[&endpoint].generation,
            generation
        );
        assert_eq!(receiver.borrow().endpoints.len(), 2);
        membership
            .bounded_apply(document(&["a"]), &cancel)
            .await
            .unwrap();
        assert_eq!(receiver.borrow().endpoints.len(), 1);
        assert!(!membership.watches.contains_key("b"));
        membership
            .bounded_apply(document(&[]), &cancel)
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
        let membership = DcMembershipWatch::start_file(
            discovery,
            KvDcRelaySourcesFile {
                path: path.clone(),
                connection_revision: Some("connection".into()),
            },
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
}
