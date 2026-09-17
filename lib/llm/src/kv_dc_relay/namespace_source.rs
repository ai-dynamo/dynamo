// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;

use dynamo_runtime::discovery::DiscoveryQuery;
use futures::Stream;
use serde::Serialize;
use tokio_util::sync::CancellationToken;

pub(super) mod discovery;
pub(super) mod file;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NamespaceSelection {
    pub scope: NamespaceScope,
    pub revision: Option<String>,
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
    pub last_error: Option<String>,
}
