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

use crate::cache::vllm_block_pool::{BlockCopyId, BlockReservation, GroupedHash, VllmBlockPool};

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

impl GroupAllocation<'_> {
    /// Check that the step's metadata describes its own blocks.
    pub(crate) fn validate(&self) {
        let full_blocks = self
            .blocks
            .iter()
            .filter(|block| matches!(block, UniqueBlock::FullBlock(_)))
            .count();
        assert!(
            self.local_hashes.is_empty() || self.local_hashes.len() == full_blocks,
            "local hashes must be empty or align with full blocks"
        );
        assert!(
            self.token_ids.is_none_or(|ids| ids.len() == full_blocks),
            "token IDs must align with full blocks"
        );
        assert!(!matches!(self.parent, Some(UniqueBlock::PartialBlock(_))));
    }
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
            Self::FullAttention(group) => group.release(pool, request_id, blocks),
        }
    }

    /// Take this group's share of `alloc` out of the shared reservation.
    ///
    /// `prefix_copies` yields the activated pins in the order the groups asked
    /// for them, so each group drains exactly the ones it requested through
    /// [`Self::prefix_keys`] and leaves the rest for the groups behind it.
    ///
    /// Returns the blocks that became cache-visible, for the coordinator to
    /// report. Only the main attention group returns them: the router indexes a
    /// single block namespace with no group identity.
    pub(crate) fn commit(
        &mut self,
        pool: &mut VllmBlockPool,
        reservation: &mut BlockReservation,
        alloc: &GroupAllocation,
        prefix_copies: &mut impl Iterator<Item = BlockCopyId>,
        materialize_store_events: bool,
    ) -> Option<Vec<Option<StoredBlock>>> {
        match self {
            Self::FullAttention(group) => group.commit(
                pool,
                reservation,
                alloc,
                prefix_copies,
                materialize_store_events,
            ),
        }
    }

    /// Make everything this group completed between the two token watermarks
    /// cache-visible, returning the blocks the coordinator may report.
    pub(crate) fn finalize_computed_prefix(
        &mut self,
        pool: &mut VllmBlockPool,
        request_id: Uuid,
        computed_before: usize,
        computed_after: usize,
        materialize_store_events: bool,
    ) -> Option<Vec<Option<StoredBlock>>> {
        match self {
            Self::FullAttention(group) => group.finalize_computed_prefix(
                pool,
                request_id,
                computed_before,
                computed_after,
                materialize_store_events,
            ),
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
