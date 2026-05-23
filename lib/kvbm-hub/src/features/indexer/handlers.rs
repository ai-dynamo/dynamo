// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side velo active-message handler for the KV indexer lookup.
//!
//! Installed on the hub's own velo in
//! [`IndexerManager::attach`](super::manager::IndexerManager) (when a transport
//! is configured). This is the **client → hub** direction — the reverse of the
//! hub → client heartbeat handler in [`crate::handlers`] — and the velo-plane
//! equivalent of `POST /v1/features/indexer/query`, keeping the result typed
//! ([`SequenceHash`] + [`InstanceId`]) instead of stringifying it.

use std::sync::Arc;

use uuid::Uuid;
use velo::Handler;
use velo_ext::InstanceId;

use super::index::PositionalIndex;
use super::protocol::{FindBlocksHit, QUERY_HANDLER, QueryRequest};

/// Build the indexer-lookup velo handler over a shared [`PositionalIndex`].
///
/// Resolves the candidate hashes to the deepest indexed block and its holders
/// via [`PositionalIndex::query_holders`], reconstructing each holder's
/// [`InstanceId`] from the raw `u128` the index stores (publishers stamp
/// `velo_id.as_u128()`). Returns `Ok(None)` on a full miss.
pub fn create_query_handler(index: Arc<PositionalIndex>) -> Handler {
    Handler::typed_unary_async::<QueryRequest, Option<FindBlocksHit>, _, _>(
        QUERY_HANDLER,
        move |ctx| {
            let index = Arc::clone(&index);
            async move {
                Ok(index
                    .query_holders(&ctx.input.hashes)
                    .map(|(matched, ids)| FindBlocksHit {
                        matched,
                        candidates: ids
                            .into_iter()
                            .map(|u| InstanceId::from(Uuid::from_u128(u)))
                            .collect(),
                    }))
            }
        },
    )
    .build()
}
