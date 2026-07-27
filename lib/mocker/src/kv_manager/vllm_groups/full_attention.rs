// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The main attention KV cache group.

use dynamo_tokens::SequenceHash;
use dynamo_tokens::blocks::UniqueBlock;
use rustc_hash::{FxHashMap, FxHashSet};
use uuid::Uuid;

use crate::cache::vllm_block_pool::{
    BlockCopyId, BlockReservation, GroupedHash, KvCacheGroupId, VllmBlockPool,
};

use super::{CacheHit, GroupAllocation, PendingStore, StoredBlock};

struct FullBlock {
    copy: BlockCopyId,
    hash: SequenceHash,
    /// Whether a freshly allocated full block still needs to become cache-visible.
    pending_cache: bool,
    /// Event metadata retained until `pending_cache` is finalized.
    pending_store: Option<PendingStore>,
}

enum OwnedBlock {
    Partial { copy: BlockCopyId, uuid: Uuid },
    Full(FullBlock),
}

/// Per-request block table for the main attention group.
///
/// This is vLLM's `FullAttentionManager`: a dense block table whose cache hits
/// are a contiguous prefix scanned left to right.
pub(crate) struct FullAttentionGroup {
    group: KvCacheGroupId,
    request_blocks: FxHashMap<Uuid, Vec<OwnedBlock>>,
    partial_uuids: FxHashSet<Uuid>,
    block_size: usize,
    enable_prefix_caching: bool,
}

impl FullAttentionGroup {
    pub(crate) fn new(
        group: KvCacheGroupId,
        block_size: usize,
        enable_prefix_caching: bool,
    ) -> Self {
        Self {
            group,
            request_blocks: FxHashMap::default(),
            partial_uuids: FxHashSet::default(),
            block_size,
            enable_prefix_caching,
        }
    }

    fn key(&self, hash: SequenceHash) -> GroupedHash {
        GroupedHash::new(self.group, hash)
    }

    pub(crate) fn id(&self) -> KvCacheGroupId {
        self.group
    }

    pub(crate) fn holds_request(&self, request_id: Uuid) -> bool {
        self.request_blocks.contains_key(&request_id)
    }

    pub(crate) fn block_count(&self, request_id: Uuid) -> usize {
        self.request_blocks.get(&request_id).map_or(0, Vec::len)
    }

    /// Cached-prefix keys this step may pin, taken from the caller-authorized
    /// reusable prefix.
    pub(crate) fn prefix_keys(&self, alloc: &GroupAllocation) -> Vec<GroupedHash> {
        alloc.blocks[..alloc.reusable_prefix_blocks]
            .iter()
            .map(|block| match block {
                UniqueBlock::FullBlock(hash) => self.key(*hash),
                UniqueBlock::PartialBlock(_) => {
                    panic!("a reusable prefix can contain only full blocks")
                }
            })
            .collect()
    }

    pub(crate) fn fresh_blocks(&self, alloc: &GroupAllocation) -> usize {
        alloc.blocks.len() - alloc.reusable_prefix_blocks
    }

    /// Longest resident prefix of `blocks`, for a destination reservation whose
    /// reusable prefix has not been authorized by a prior lookup.
    pub(crate) fn resident_prefix(
        &self,
        pool: &VllmBlockPool,
        blocks: &[UniqueBlock],
    ) -> Vec<GroupedHash> {
        if !self.enable_prefix_caching {
            return Vec::new();
        }
        blocks
            .iter()
            .map_while(|block| match block {
                UniqueBlock::FullBlock(hash) if pool.prefix_hit(self.key(*hash)).is_some() => {
                    Some(self.key(*hash))
                }
                _ => None,
            })
            .collect()
    }

    pub(crate) fn longest_cache_hit(
        &self,
        pool: &VllmBlockPool,
        blocks: &[UniqueBlock],
    ) -> CacheHit {
        if !self.enable_prefix_caching {
            return CacheHit::default();
        }
        let mut overlap = 0;
        let mut active = 0;
        for block in blocks {
            let UniqueBlock::FullBlock(hash) = block else {
                break;
            };
            let Some(hit) = pool.prefix_hit(self.key(*hash)) else {
                break;
            };
            overlap += 1;
            active += usize::from(hit.is_active);
        }
        CacheHit {
            tokens: overlap * self.block_size,
            active_tokens: active * self.block_size,
        }
    }

    pub(crate) fn commit(
        &mut self,
        pool: &mut VllmBlockPool,
        reservation: &mut BlockReservation,
        alloc: &GroupAllocation,
        prefix_copies: &[BlockCopyId],
        materialize_store_events: bool,
    ) -> Option<Vec<Option<StoredBlock>>> {
        let prefix_len = alloc.reusable_prefix_blocks;
        assert_eq!(prefix_copies.len(), prefix_len);
        let mut prefix_copies = prefix_copies.iter().copied();
        let mut cursor = match alloc.parent {
            None => None,
            Some(UniqueBlock::FullBlock(hash)) => Some(*hash),
            Some(UniqueBlock::PartialBlock(_)) => unreachable!("validated above"),
        };
        let mut full_idx = 0;
        let owned = self.request_blocks.entry(alloc.request_id).or_default();
        owned.reserve(alloc.blocks.len());
        let mut stores = (alloc.cache_fresh && materialize_store_events)
            .then(|| Vec::with_capacity(alloc.blocks.len()));

        for (block_idx, block) in alloc.blocks.iter().enumerate() {
            match block {
                UniqueBlock::FullBlock(hash) => {
                    let local_hash = alloc.local_hashes.get(full_idx).copied();
                    full_idx += 1;

                    if block_idx < prefix_len {
                        let Some(copy) = prefix_copies.next() else {
                            panic!("prefix reservation returned too few copies")
                        };
                        owned.push(OwnedBlock::Full(FullBlock {
                            copy,
                            hash: *hash,
                            pending_cache: false,
                            pending_store: None,
                        }));
                        cursor = Some(*hash);
                        continue;
                    }

                    if alloc.cache_fresh {
                        let (copy, became_visible) =
                            pool.allocate_cached(reservation, GroupedHash::new(self.group, *hash));
                        owned.push(OwnedBlock::Full(FullBlock {
                            copy,
                            hash: *hash,
                            pending_cache: false,
                            pending_store: None,
                        }));
                        if let Some(stores) = &mut stores {
                            let metadata = PendingStore {
                                parent_hash: cursor,
                                local_hash,
                                token_ids: alloc
                                    .token_ids
                                    .and_then(|ids| ids.get(full_idx - 1).cloned()),
                            };
                            stores.push(became_visible.then_some(StoredBlock {
                                hash: *hash,
                                metadata,
                            }));
                        }
                    } else {
                        let copy = pool.allocate_private(reservation);
                        let pending_cache = self.enable_prefix_caching;
                        let pending_store =
                            (pending_cache && materialize_store_events).then(|| PendingStore {
                                parent_hash: cursor,
                                local_hash,
                                token_ids: alloc
                                    .token_ids
                                    .and_then(|ids| ids.get(full_idx - 1).cloned()),
                            });
                        owned.push(OwnedBlock::Full(FullBlock {
                            copy,
                            hash: *hash,
                            pending_cache,
                            pending_store,
                        }));
                    }
                    cursor = Some(*hash);
                }
                UniqueBlock::PartialBlock(uuid) => {
                    let copy = pool.allocate_private(reservation);
                    assert!(
                        self.partial_uuids.insert(*uuid),
                        "partial block {uuid} is already allocated"
                    );
                    owned.push(OwnedBlock::Partial { copy, uuid: *uuid });
                    if let Some(stores) = &mut stores {
                        stores.push(None);
                    }
                }
            }
        }
        assert!(prefix_copies.next().is_none());
        stores
    }

    /// Make blocks completed by this scheduling decision cache-visible.
    pub(crate) fn finalize_computed_prefix(
        &mut self,
        pool: &mut VllmBlockPool,
        request_id: Uuid,
        first_new_block: usize,
        completed_blocks: usize,
        materialize_store_events: bool,
    ) -> Option<Vec<Option<StoredBlock>>> {
        let Some(blocks) = self.request_blocks.get_mut(&request_id) else {
            panic!("request {request_id} owns no block table")
        };
        let completed_blocks = completed_blocks.min(blocks.len());
        if first_new_block >= completed_blocks {
            return None;
        }
        let mut stores = materialize_store_events
            .then(|| Vec::with_capacity(completed_blocks - first_new_block));
        for block in &mut blocks[first_new_block..completed_blocks] {
            match block {
                OwnedBlock::Full(full) => {
                    if !full.pending_cache {
                        if let Some(stores) = &mut stores {
                            stores.push(None);
                        }
                        continue;
                    }
                    full.pending_cache = false;
                    let metadata = full.pending_store.take();
                    let became_visible =
                        pool.cache_private(full.copy, GroupedHash::new(self.group, full.hash));
                    if let Some(stores) = &mut stores {
                        stores.push(became_visible.then(|| {
                            StoredBlock {
                                hash: full.hash,
                                metadata: metadata.expect(
                                    "materialized pending store must retain event metadata",
                                ),
                            }
                        }));
                    } else {
                        debug_assert!(metadata.is_none());
                    }
                }
                OwnedBlock::Partial { .. } => break,
            }
        }
        stores
    }

    /// Release request blocks in caller-provided eviction-priority order.
    ///
    /// Like vLLM's `BlockPool::free_blocks`, the physical pool is lineage-agnostic:
    /// reversing the request-owned table here makes suffix/leaf blocks older LRU
    /// candidates than their parents, so capacity pressure evicts the leaf first.
    pub(crate) fn deref(
        &mut self,
        pool: &mut VllmBlockPool,
        request_id: Uuid,
        blocks: &[UniqueBlock],
    ) {
        let released = {
            let Some(owned) = self.request_blocks.get_mut(&request_id) else {
                panic!("request {request_id} owns no block table")
            };
            assert!(
                blocks.len() <= owned.len(),
                "request releases too many blocks"
            );
            let start = owned.len() - blocks.len();
            for (expected, actual) in blocks.iter().zip(owned[start..].iter().rev()) {
                match (expected, actual) {
                    (UniqueBlock::FullBlock(expected), OwnedBlock::Full(full)) => {
                        assert_eq!(*expected, full.hash, "full-block Deref mismatch");
                    }
                    (UniqueBlock::PartialBlock(expected), OwnedBlock::Partial { uuid, .. }) => {
                        assert_eq!(expected, uuid, "partial Deref mismatch")
                    }
                    _ => panic!("Deref block kind disagrees with request table"),
                }
            }
            owned.split_off(start)
        };
        if self.request_blocks[&request_id].is_empty() {
            self.request_blocks.remove(&request_id);
        }

        for block in released.into_iter().rev() {
            match block {
                OwnedBlock::Partial { copy, uuid } => {
                    assert!(self.partial_uuids.remove(&uuid));
                    pool.release(copy);
                }
                OwnedBlock::Full(full) => pool.release(full.copy),
            }
        }
    }

    pub(crate) fn promote(
        &mut self,
        pool: &mut VllmBlockPool,
        request_id: Uuid,
        uuid: Uuid,
        hash: SequenceHash,
        metadata: PendingStore,
        materialize_store_events: bool,
    ) -> Option<StoredBlock> {
        let Some(blocks) = self.request_blocks.get_mut(&request_id) else {
            panic!("request {request_id} owns no block table")
        };
        let Some(last) = blocks.last_mut() else {
            panic!("Promote requires a request-owned partial tail")
        };
        let copy = match last {
            OwnedBlock::Partial { copy, uuid: actual } => {
                assert_eq!(*actual, uuid, "Promote partial UUID mismatch");
                *copy
            }
            OwnedBlock::Full(_) => panic!("Promote requires a partial tail"),
        };
        assert!(self.partial_uuids.remove(&uuid));

        let became_visible = self.enable_prefix_caching
            && pool.cache_private(copy, GroupedHash::new(self.group, hash));
        *last = OwnedBlock::Full(FullBlock {
            copy,
            hash,
            pending_cache: false,
            pending_store: None,
        });
        (became_visible && materialize_store_events).then_some(StoredBlock { hash, metadata })
    }

    pub(crate) fn validate_use_metadata(alloc: &GroupAllocation) {
        let full_blocks = alloc
            .blocks
            .iter()
            .filter(|block| matches!(block, UniqueBlock::FullBlock(_)))
            .count();
        assert!(
            alloc.local_hashes.is_empty() || alloc.local_hashes.len() == full_blocks,
            "local hashes must be empty or align with full blocks"
        );
        assert!(
            alloc.token_ids.is_none_or(|ids| ids.len() == full_blocks),
            "token IDs must align with full blocks"
        );
        assert!(!matches!(alloc.parent, Some(UniqueBlock::PartialBlock(_))));
    }

    pub(crate) fn validate_fresh_partials(&self, blocks: &[UniqueBlock]) {
        let mut first_partial = None;
        for (index, uuid) in blocks
            .iter()
            .enumerate()
            .filter_map(|(index, block)| match block {
                UniqueBlock::PartialBlock(uuid) => Some((index, uuid)),
                UniqueBlock::FullBlock(_) => None,
            })
        {
            let repeated_in_layout = first_partial.is_some_and(|first| {
                first == *uuid
                    || blocks[..index].iter().any(
                        |block| matches!(block, UniqueBlock::PartialBlock(seen) if seen == uuid),
                    )
            });
            assert!(
                !self.partial_uuids.contains(uuid) && !repeated_in_layout,
                "partial block {uuid} is already allocated"
            );
            first_partial.get_or_insert(*uuid);
        }
    }
}
