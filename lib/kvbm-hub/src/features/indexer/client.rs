// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client-side velo lookup wrapper for the KV indexer feature.
//!
//! Mirrors [`ConditionalDisaggClient`](crate::features::disagg::client::ConditionalDisaggClient):
//! a thin wrapper over an [`Arc<Messenger>`] that knows the indexer
//! [`QUERY_HANDLER`] name and the hub's velo [`InstanceId`], exposing a single
//! per-request [`find_blocks`](IndexerLookupClient::find_blocks) call. Construct
//! it via [`HubClient::indexer_lookup_client`](crate::HubClient::indexer_lookup_client),
//! which gates on the indexer being enabled and supplies the hub's `InstanceId`.

use std::sync::Arc;

use anyhow::Result;
use kvbm_logical::SequenceHash;
use velo::Messenger;
use velo_ext::InstanceId;

use super::protocol::{FindBlocksHit, QUERY_HANDLER, QueryRequest};

/// Velo-plane lookup client for the hub's KV block index.
pub struct IndexerLookupClient {
    messenger: Arc<Messenger>,
    /// Hub's velo `InstanceId` — the target of the lookup unary RPC.
    hub_velo_id: InstanceId,
}

impl std::fmt::Debug for IndexerLookupClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexerLookupClient")
            .field("hub_velo_id", &self.hub_velo_id)
            .finish()
    }
}

impl IndexerLookupClient {
    /// Wrap a [`Messenger`] targeting the hub at `hub_velo_id`.
    pub fn new(messenger: Arc<Messenger>, hub_velo_id: InstanceId) -> Arc<Self> {
        Arc::new(Self {
            messenger,
            hub_velo_id,
        })
    }

    /// The hub's velo `InstanceId` this client targets.
    pub fn hub_velo_id(&self) -> InstanceId {
        self.hub_velo_id
    }

    /// Resolve a candidate block sequence to the deepest indexed block and its
    /// holders, over velo.
    ///
    /// `hashes` are the block-sequence PLHs in position order (low → high). The
    /// hub walks them and returns the deepest one present — so `[x, y, z]` with
    /// `z` missing but `y` indexed yields `Some(hit)` where `hit.matched == y`
    /// and `hit.candidates` are the instances holding `y`. A full miss returns
    /// `Ok(None)`.
    pub async fn find_blocks(&self, hashes: Vec<SequenceHash>) -> Result<Option<FindBlocksHit>> {
        let req = QueryRequest { hashes };
        let hit = self
            .messenger
            .typed_unary::<Option<FindBlocksHit>>(QUERY_HANDLER)?
            .payload(&req)?
            .instance(self.hub_velo_id)
            .send()
            .await?;
        Ok(hit)
    }
}
