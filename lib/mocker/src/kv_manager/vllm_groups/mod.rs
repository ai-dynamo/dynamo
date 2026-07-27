// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV cache groups over one shared physical block pool.
//!
//! vLLM splits a model's layers into KV cache groups — one per attention type —
//! each keeping its own per-request block table while drawing blocks from a
//! single pool with a uniform page size. Cache visibility is namespaced per
//! group (`BlockHashWithGroupId`), which is what keeps two groups that hash the
//! same tokens from satisfying each other's lookups.
//!
//! This module holds the group implementations and the vocabulary they exchange
//! with the coordinator in [`super::vllm_backend`]. A group hands its newly
//! cache-visible blocks back to the coordinator and never touches the
//! router-facing event path: router events carry no group identity, so only the
//! coordinator can decide which group's blocks may be reported.

mod full_attention;

use dynamo_tokens::blocks::UniqueBlock;
use dynamo_tokens::{BlockHash, SequenceHash};
use uuid::Uuid;

use crate::cache::vllm_block_pool::{GroupedHash, VllmBlockPool};

pub(crate) use full_attention::FullAttentionGroup;

/// Event metadata a group retains until its block becomes cache-visible.
pub(crate) struct PendingStore {
    pub(crate) parent_hash: Option<SequenceHash>,
    pub(crate) local_hash: Option<BlockHash>,
    pub(crate) token_ids: Option<Vec<u32>>,
}

/// A block that just became cache-visible, for the coordinator to report.
pub(crate) struct StoredBlock {
    pub(crate) hash: SequenceHash,
    pub(crate) metadata: PendingStore,
}

/// One group's prefix-cache lookup result, in tokens.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CacheHit {
    /// Longest cached token prefix this group can serve.
    pub(crate) tokens: usize,
    /// Portion of `tokens` whose copies are already active, and so are not
    /// re-consumed from physical capacity when the request claims them.
    pub(crate) active_tokens: usize,
}

/// Everything a group needs to size and commit one allocation step.
pub(crate) struct GroupAllocation<'a> {
    pub(crate) request_id: Uuid,
    /// Attention blocks added by this step, in request order.
    pub(crate) blocks: &'a [UniqueBlock],
    pub(crate) local_hashes: &'a [BlockHash],
    pub(crate) token_ids: Option<&'a [Vec<u32>]>,
    pub(crate) parent: Option<&'a UniqueBlock>,
    /// Leading entries of `blocks` that are already resident and reusable.
    pub(crate) reusable_prefix_blocks: usize,
    /// Destination transfers arrive already computed, so their fresh blocks
    /// enter the prefix cache immediately instead of waiting for a compute
    /// watermark.
    pub(crate) cache_fresh: bool,
}

/// A KV cache group sharing the coordinator's physical pool.
///
/// vLLM's `SingleTypeKVCacheManager` hierarchy, narrowed to the group kinds the
/// mocker models. Dispatch is static so the single-group hot path keeps its
/// shape.
pub(crate) enum GroupManager {
    FullAttention(FullAttentionGroup),
}

impl GroupManager {
    /// This group if it is the main attention group.
    ///
    /// The coordinator resolves the attention group by kind, the way vLLM's
    /// coordinator resolves `full_attention_group_id`, so no caller depends on
    /// where the group sits among the others.
    pub(crate) fn as_full_attention(&self) -> Option<&FullAttentionGroup> {
        match self {
            Self::FullAttention(group) => Some(group),
        }
    }

    pub(crate) fn as_full_attention_mut(&mut self) -> Option<&mut FullAttentionGroup> {
        match self {
            Self::FullAttention(group) => Some(group),
        }
    }

    pub(crate) fn is_full_attention(&self) -> bool {
        self.as_full_attention().is_some()
    }

    /// Release everything this group holds for `request_id`.
    ///
    /// `blocks` is the attention table's release order, leaf first, so that the
    /// pool ages a leaf ahead of the parent it descends from. A group whose own
    /// table is not parallel to the attention table ignores the list and drops
    /// the request's whole state, which the coordinator's `deref_for_request`
    /// guarantees is what a release means.
    pub(crate) fn release(
        &mut self,
        pool: &mut VllmBlockPool,
        request_id: Uuid,
        blocks: &[UniqueBlock],
    ) {
        match self {
            Self::FullAttention(group) => group.deref(pool, request_id, blocks),
        }
    }

    /// Cached-prefix keys this group may pin for `alloc`.
    pub(crate) fn prefix_keys(&self, alloc: &GroupAllocation) -> Vec<GroupedHash> {
        match self {
            Self::FullAttention(group) => group.prefix_keys(alloc),
        }
    }

    /// Pool blocks `alloc` needs beyond the prefix this group can reuse.
    pub(crate) fn fresh_blocks(&self, alloc: &GroupAllocation) -> usize {
        match self {
            Self::FullAttention(group) => group.fresh_blocks(alloc),
        }
    }

    /// This group's prefix hit, or `None` when the group takes no part in
    /// prefix caching and so constrains no boundary.
    pub(crate) fn longest_cache_hit(
        &self,
        pool: &VllmBlockPool,
        blocks: &[UniqueBlock],
    ) -> Option<CacheHit> {
        match self {
            Self::FullAttention(group) => Some(group.longest_cache_hit(pool, blocks)),
        }
    }
}
