// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{NamespaceSelection, NamespaceSource, NamespaceUpdates};
use crate::kv_dc_relay::discovery::DcDiscoveryFilter;
use dynamo_runtime::discovery::{Discovery, DiscoveryInstanceId, DiscoveryQuery};
use futures::StreamExt;
use std::{collections::HashSet, sync::Arc, time::Duration};
use tokio_util::sync::CancellationToken;

const RECONCILE_INTERVAL: Duration = Duration::from_secs(30);

/// Selects which Dynamo endpoints one Relay supervises.
///
/// The watch scope also fixes a naming invariant: request-facing model and
/// adapter names must be unique across every namespace one Relay watches. A
/// local ModelManager may resolve a name collision by its own first-wins
/// order, but a Relay federates independently owned endpoints and has no safe
/// canonical owner to choose, so a name claimed by conflicting targets is
/// omitted from every endpoint (fail-closed, recorded as a serving conflict)
/// rather than arbitrated per namespace.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvDcRelayDiscoveryConfig {
    pub namespaces: Vec<String>,
    pub endpoint_prefixes: Vec<String>,
    pub watch_all: bool,
}

impl KvDcRelayDiscoveryConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.watch_all || !self.namespaces.is_empty(),
            "KV DC Relay requires at least one discovery namespace or explicit watch_all"
        );
        anyhow::ensure!(
            !self.watch_all || self.namespaces.is_empty(),
            "KV DC Relay watch_all cannot be combined with explicit discovery namespaces"
        );

        let mut unique_namespaces = HashSet::new();
        for namespace in &self.namespaces {
            anyhow::ensure!(
                !namespace.trim().is_empty(),
                "KV DC Relay discovery namespaces must not be empty"
            );
            anyhow::ensure!(
                namespace.trim() == namespace,
                "KV DC Relay discovery namespaces must not contain surrounding whitespace"
            );
            anyhow::ensure!(
                unique_namespaces.insert(namespace),
                "duplicate KV DC Relay discovery namespace: {namespace}"
            );
        }

        let mut unique_prefixes = HashSet::new();
        for prefix in &self.endpoint_prefixes {
            anyhow::ensure!(
                !prefix.trim().is_empty(),
                "KV DC Relay endpoint prefixes must not be empty"
            );
            anyhow::ensure!(
                prefix.trim() == prefix,
                "KV DC Relay endpoint prefixes must not contain surrounding whitespace"
            );
            anyhow::ensure!(
                unique_prefixes.insert(prefix),
                "duplicate KV DC Relay endpoint prefix: {prefix}"
            );
            anyhow::ensure!(
                self.watch_all
                    || self.namespaces.iter().any(|namespace| {
                        prefix == namespace
                            || prefix
                                .strip_prefix(namespace)
                                .is_some_and(|suffix| suffix.starts_with('.'))
                    }),
                "KV DC Relay endpoint prefix {prefix} is outside the configured namespaces"
            );
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn queries(&self) -> Vec<DiscoveryQuery> {
        if self.watch_all {
            vec![DiscoveryQuery::AllModels]
        } else {
            self.namespaces
                .iter()
                .map(|namespace| DiscoveryQuery::NamespacedModels {
                    namespace: namespace.clone(),
                })
                .collect()
        }
    }

    pub(crate) fn filter(&self) -> DcDiscoveryFilter {
        DcDiscoveryFilter {
            endpoint_prefixes: self.endpoint_prefixes.clone(),
        }
    }
}

pub(crate) struct DiscoveryNamespaces {
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
                            result = tokio::time::timeout(Duration::from_secs(10), self.discovery.list(DiscoveryQuery::AllModels)) => result,
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
