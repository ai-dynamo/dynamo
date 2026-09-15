// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{NamespaceScope, NamespaceSelection, NamespaceSource, NamespaceUpdates};
use crate::kv_dc_relay::discovery::DcDiscoveryFilter;
#[cfg(test)]
use dynamo_runtime::discovery::DiscoveryQuery;
use std::collections::HashSet;
use tokio_util::sync::CancellationToken;

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
    pub config: KvDcRelayDiscoveryConfig,
}

impl NamespaceSource for DiscoveryNamespaces {
    fn updates(self: Box<Self>, cancel: CancellationToken) -> NamespaceUpdates {
        Box::pin(async_stream::stream! {
            let scope = if self.config.watch_all {
                NamespaceScope::All
            } else {
                let mut namespaces = self.config.namespaces.clone();
                namespaces.sort();
                NamespaceScope::Namespaces(namespaces)
            };
            yield Ok(NamespaceSelection { scope, revision: None });
            cancel.cancelled().await;
        })
    }
}
