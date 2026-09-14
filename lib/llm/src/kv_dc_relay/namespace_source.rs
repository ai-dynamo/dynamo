// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;

use futures::Stream;
use serde::Serialize;
use tokio_util::sync::CancellationToken;

pub(super) mod discovery;
pub(super) mod file;

/// Source selection is independent of endpoint discovery and connection configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NamespaceSelection {
    pub namespaces: Vec<String>,
    pub revision: Option<String>,
}

pub(crate) type NamespaceUpdates =
    Pin<Box<dyn Stream<Item = anyhow::Result<NamespaceSelection>> + Send>>;

/// Emits complete snapshots. Errors retain the last applied selection; an empty
/// successful snapshot explicitly removes all namespace watches.
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
