// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kernel-side hook for an external shared KV cache (Mooncake, HiCache). The scheduler owns
//! this seam; the indexer stage never calls it.

use async_trait::async_trait;

use std::sync::Arc;

use crate::indexer::KvRouterError;
use crate::protocols::SharedCacheHits;
use crate::session_prefix_index::SessionPrefixIndexer;

/// Trait for querying an external shared KV cache pool.
///
/// Implementations check which blocks/pages from a request's token sequence
/// exist in the shared cache. The returned `SharedCacheHits` describes which
/// block positions are available externally (and thus cheaper to prefill).
#[async_trait]
pub trait SharedKvCache: Send + Sync {
    /// Query which blocks exist in the shared cache for the given token sequence.
    ///
    /// `cache_namespace` must be honored by implementations to avoid scoring
    /// shared-cache hits across isolated cache namespaces.
    async fn check_blocks(
        &self,
        tokens: &[u32],
        block_size: u32,
        cache_namespace: Option<&str>,
    ) -> Result<SharedCacheHits, KvRouterError>;

    /// Hands the router's session prefix index to the cache so residency events that name
    /// a session can update it. The default ignores the index.
    fn attach_session_prefix_index(&self, _index: Arc<SessionPrefixIndexer>) {}
}
