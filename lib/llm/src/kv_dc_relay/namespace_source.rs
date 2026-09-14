// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::kv_dc_relay::sources::{KvDcRelaySourcesFile, SourcesDocument};

/// Source selection is independent of endpoint discovery and connection configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct NamespaceSelection {
    pub namespaces: Vec<String>,
    pub revision: Option<String>,
}

impl From<SourcesDocument> for NamespaceSelection {
    fn from(document: SourcesDocument) -> Self {
        Self {
            namespaces: document
                .sources
                .into_iter()
                .map(|source| source.namespace)
                .collect(),
            revision: Some(document.revision),
        }
    }
}

pub(super) type NamespaceUpdates =
    Pin<Box<dyn Stream<Item = anyhow::Result<NamespaceSelection>> + Send>>;

/// Emits complete snapshots. Errors retain the last applied selection; an empty
/// successful snapshot explicitly removes all namespace watches.
pub(super) trait NamespaceSource: Send {
    fn updates(self: Box<Self>, cancel: CancellationToken) -> NamespaceUpdates;
}

impl NamespaceSource for KvDcRelaySourcesFile {
    fn updates(self: Box<Self>, cancel: CancellationToken) -> NamespaceUpdates {
        Box::pin(async_stream::stream! {
            let mut poll = tokio::time::interval(Duration::from_secs(1));
            poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = poll.tick() => {}
                }
                let result = tokio::select! {
                    _ = cancel.cancelled() => break,
                    result = tokio::time::timeout(Duration::from_secs(10), self.read()) => result,
                };
                // File parsing errors omit the document contents.
                yield match result {
                    Ok(Ok(document)) => Ok(document.into()),
                    Ok(Err(error)) => Err(error),
                    Err(_) => Err(anyhow::anyhow!("sources file read timed out")),
                };
            }
        })
    }
}

pub(super) struct DiscoveryNamespaces {
    pub discovery: Arc<dyn Discovery>,
    pub config: KvDcRelayDiscoveryConfig,
}

impl NamespaceSource for DiscoveryNamespaces {
    fn updates(self: Box<Self>, cancel: CancellationToken) -> NamespaceUpdates {
        Box::pin(async_stream::stream! {
            if !self.config.watch_all {
                let mut namespaces = self.config.namespaces.clone();
                namespaces.sort();
                yield Ok(NamespaceSelection { namespaces, revision: None });
                cancel.cancelled().await;
                return;
            }
            let queries = self.config.queries();
            loop {
                let token = cancel.child_token();
                let _guard = token.clone().drop_guard();
                let opened = tokio::select! {
                    _ = cancel.cancelled() => break,
                    result = tokio::time::timeout(Duration::from_secs(10), self.discovery.list_and_watch(DiscoveryQuery::AllModels, Some(token))) => result,
                };
                if let Ok(Ok(mut events)) = opened {
                    let mut reconcile = tokio::time::interval(RECONCILE_INTERVAL);
                    reconcile.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                    loop {
                        let result = tokio::select! {
                            _ = cancel.cancelled() => return,
                            result = tokio::time::timeout(Duration::from_secs(10), list_queries(&self.discovery, &queries)) => result,
                        };
                        yield match result {
                            Ok(Ok(instances)) => {
                                let mut namespaces = instances.into_iter().filter_map(|instance| {
                                    match instance.id() {
                                        DiscoveryInstanceId::Model(id) => Some(id.namespace),
                                        _ => None,
                                    }
                                }).collect::<Vec<_>>();
                                namespaces.sort();
                                namespaces.dedup();
                                Ok(NamespaceSelection { namespaces, revision: None })
                            }
                            _ => Err(anyhow::anyhow!("namespace discovery snapshot failed")),
                        };
                        tokio::select! {
                            _ = cancel.cancelled() => return,
                            _ = reconcile.tick() => {},
                            event = events.next() => {
                                if !matches!(event, Some(Ok(_))) { break; }
                            }
                        }
                    }
                }
                yield Err(anyhow::anyhow!("namespace discovery watch unavailable; retrying"));
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {}
                }
            }
        })
    }
}
