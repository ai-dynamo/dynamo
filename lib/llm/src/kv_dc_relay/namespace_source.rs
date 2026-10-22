// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::discovery::DcMembershipView;
use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;

use dynamo_runtime::discovery::DiscoveryQuery;
use futures::Stream;
use serde::Serialize;
use tokio_util::sync::CancellationToken;

pub(super) mod discovery;
pub(super) mod file;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NamespaceSelection {
    pub scope: NamespaceScope,
    pub logical_namespaces: BTreeMap<String, String>,
    pub revision: Option<String>,
}

impl NamespaceSelection {
    /// Rewrite only the outward identity, after physical discovery/domain resolution.
    /// Two physical endpoints cannot own one public endpoint, even during a rollout.
    pub(super) fn project(&self, mut view: DcMembershipView) -> (DcMembershipView, usize) {
        if self.logical_namespaces.is_empty() {
            return (view, 0);
        }
        let endpoints = Arc::make_mut(&mut view.endpoints);
        let mut owners = std::collections::HashMap::new();
        for membership in endpoints.values_mut() {
            if let Some(logical) = self.logical_namespaces.get(&membership.endpoint.namespace) {
                membership.namespace.clone_from(logical);
            }
            *owners.entry(membership.public_endpoint()).or_insert(0usize) += 1;
        }
        let conflicts = owners.values().filter(|count| **count > 1).count();
        endpoints.retain(|_, membership| owners[&membership.public_endpoint()] == 1);
        (view, conflicts)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum NamespaceScope {
    All,
    Namespaces(Vec<String>),
}

impl NamespaceScope {
    pub(super) fn queries(&self) -> Vec<DiscoveryQuery> {
        match self {
            Self::All => vec![DiscoveryQuery::AllModels],
            Self::Namespaces(namespaces) => namespaces
                .iter()
                .map(|namespace| DiscoveryQuery::NamespacedModels {
                    namespace: namespace.clone(),
                })
                .collect(),
        }
    }

    pub(super) fn watch_count(&self) -> usize {
        match self {
            Self::All => 1,
            Self::Namespaces(namespaces) => namespaces.len(),
        }
    }
}

pub(crate) type NamespaceUpdates =
    Pin<Box<dyn Stream<Item = anyhow::Result<NamespaceSelection>> + Send>>;

/// Emits complete snapshots. Errors retain the last applied selection; an empty
/// explicit namespace selection removes all namespace watches.
pub(crate) trait NamespaceSource: Send {
    fn updates(self: Box<Self>, cancel: CancellationToken) -> NamespaceUpdates;
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KvDcRelaySourcesStatus {
    pub desired_revision: Option<String>,
    pub applied_revision: Option<String>,
    pub count: usize,
    /// Number of public endpoints withheld because multiple physical sources claim them.
    pub logical_endpoint_conflicts: usize,
    pub last_error: Option<String>,
}
