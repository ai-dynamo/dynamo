// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connector-side implementation of the engine's [`RemoteBlockDiscovery`] seam.
//!
//! The engine's remote-search driver (`kvbm_engine::leader::remote_search`) is
//! transport-only — it knows how to pin + pull from a holder but not *which*
//! holder. This wires that gap to the hub's KV indexer: [`HubRemoteDiscovery`]
//! queries the indexer for the deepest-placed block + its holders, then
//! peer-resolves each holder (so velo can reach it for the control RPC and the
//! RDMA pull) before handing the candidate set back to the engine.

use std::sync::Arc;

use anyhow::Result;
use futures::future::BoxFuture;
use kvbm_common::SequenceHash;
use kvbm_engine::leader::{RemoteBlockDiscovery, RemoteCandidates};
use kvbm_hub::IndexerLookupClient;

use crate::connector::leader::p2p::peer_resolver::PeerResolver;

/// Hub-indexer-backed [`RemoteBlockDiscovery`].
pub struct HubRemoteDiscovery {
    indexer: Arc<IndexerLookupClient>,
    peer_resolver: Arc<dyn PeerResolver>,
}

impl HubRemoteDiscovery {
    pub fn new(indexer: Arc<IndexerLookupClient>, peer_resolver: Arc<dyn PeerResolver>) -> Self {
        Self {
            indexer,
            peer_resolver,
        }
    }
}

impl RemoteBlockDiscovery for HubRemoteDiscovery {
    fn discover(
        &self,
        hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
        let indexer = Arc::clone(&self.indexer);
        let peer_resolver = Arc::clone(&self.peer_resolver);
        Box::pin(async move {
            let Some(hit) = indexer.find_blocks(hashes).await? else {
                return Ok(None);
            };

            // Resolve + register each holder so velo can reach it for both the
            // control RPC and the RDMA pull. Drop candidates that fail to
            // resolve rather than failing the whole search.
            let mut reachable = Vec::with_capacity(hit.candidates.len());
            for candidate in hit.candidates {
                match peer_resolver.resolve_and_register(candidate).await {
                    Ok(()) => reachable.push(candidate),
                    Err(e) => tracing::debug!(
                        error = %e, %candidate,
                        "remote-search candidate failed to resolve; skipping"
                    ),
                }
            }

            if reachable.is_empty() {
                return Ok(None);
            }

            Ok(Some(RemoteCandidates {
                deepest: hit.matched,
                instances: reachable,
            }))
        })
    }
}
