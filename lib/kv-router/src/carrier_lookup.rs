// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_tokens::PositionalLineageHash;

use crate::carrier_feed::{CarrierFeedReplica, FeedHolder, FeedKind, ManifestKey};

pub trait CarrierLookup: Send + Sync {
    fn kind(&self, manifest: &ManifestKey) -> Option<FeedKind>;
    /// `plhs` must be ordered by ascending `position()`.
    fn deepest_by_holder(
        &self,
        manifest: &ManifestKey,
        plhs: &[PositionalLineageHash],
    ) -> Vec<(FeedHolder, PositionalLineageHash)>;
}

impl CarrierLookup for CarrierFeedReplica {
    fn kind(&self, manifest: &ManifestKey) -> Option<FeedKind> {
        CarrierFeedReplica::kind(self, manifest)
    }

    fn deepest_by_holder(
        &self,
        manifest: &ManifestKey,
        plhs: &[PositionalLineageHash],
    ) -> Vec<(FeedHolder, PositionalLineageHash)> {
        CarrierFeedReplica::deepest_by_holder(self, manifest, plhs)
    }
}

/// Resolves a worker-advertised `hub_url` to an in-process index. `None` = not local.
pub trait CarrierLookupSource: Send + Sync {
    fn lookup(&self, hub_url: &str) -> Option<Arc<dyn CarrierLookup>>;
}
