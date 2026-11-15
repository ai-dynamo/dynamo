// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The lower tier's previous storage: one map entry per edge, plus a
//! per-owner reverse map on each event lane.
//!
//! It is the reference the run storage is tested against: the store, remove
//! and clear that `LowerTierIndexer` ran before runs, without the lifecycle
//! callbacks.

use std::hash::BuildHasher;

use dashmap::DashMap;
use rustc_hash::{FxBuildHasher, FxHashMap};

use super::{
    EdgeCursor, EdgeLookup, EdgeRef, IndexedResidencyOwner, LowerTierContinuation,
    LowerTierMatchDetails, OwnerId, OwnerSet, OwnerTable, Removal, TransitionKey,
};
use crate::prelude::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, KvCacheStoreData,
    LocalBlockHash, ResidencyOwner, ResidencyProjection, ResidencyRoutingSnapshot, RouterEvent,
    WorkerWithDpRank,
};

type OwnerBlockIndex = FxHashMap<ExternalSequenceBlockHash, TransitionKey>;

pub(super) struct OwnerBlockState {
    owner: ResidencyOwner,
    id: OwnerId,
    blocks: OwnerBlockIndex,
}

pub(super) type FlatLane = FxHashMap<IndexedResidencyOwner, OwnerBlockState>;

/// One entry of the flat edge table.
struct FlatEdge {
    child_hash: ExternalSequenceBlockHash,
    owners: OwnerSet,
}

type FlatEdges = DashMap<TransitionKey, FlatEdge, FxBuildHasher>;

impl EdgeLookup for FlatEdges {
    type Cursor<'a> = FlatCursor<'a>;

    fn cursor(&self) -> FlatCursor<'_> {
        FlatCursor {
            edges: self,
            current: None,
        }
    }

    fn root_owners<R>(
        &self,
        local_hash: LocalBlockHash,
        read: impl FnOnce(&OwnerSet) -> R,
    ) -> Option<R> {
        self.get(&TransitionKey::new(None, local_hash))
            .map(|edge| read(&edge.owners))
    }
}

pub(super) struct FlatCursor<'a> {
    edges: &'a FlatEdges,
    current: Option<dashmap::mapref::one::Ref<'a, TransitionKey, FlatEdge>>,
}

impl EdgeCursor for FlatCursor<'_> {
    fn edge(
        &mut self,
        parent: Option<ExternalSequenceBlockHash>,
        local_hash: LocalBlockHash,
    ) -> Option<EdgeRef<'_>> {
        // Release the previous shard before taking the next one.
        self.current = None;
        self.current = self.edges.get(&TransitionKey::new(parent, local_hash));
        self.current.as_ref().map(|edge| EdgeRef {
            child_hash: edge.child_hash,
            owners: &edge.owners,
        })
    }
}

pub(super) struct FlatIndex {
    edges: FlatEdges,
    owners: OwnerTable,
}

impl FlatIndex {
    pub(super) fn new() -> Self {
        Self {
            edges: DashMap::with_hasher(FxBuildHasher),
            owners: OwnerTable::default(),
        }
    }

    /// Applies a store, a remove, or a clear of the event's own owner.
    pub(super) fn apply(&self, worker_blocks: &mut FlatLane, event: RouterEvent) {
        let owner = event.residency_owner().unwrap();
        match event.event.data {
            KvCacheEventData::Stored(store_data) => self
                .store_blocks_impl(worker_blocks, owner, store_data)
                .unwrap(),
            KvCacheEventData::Removed(remove_data) => self
                .remove_blocks_impl(worker_blocks, owner, &remove_data.block_hashes)
                .unwrap(),
            KvCacheEventData::Cleared => self.clear(worker_blocks, owner),
        }
    }

    fn store_blocks_impl(
        &self,
        worker_blocks: &mut FlatLane,
        owner: ResidencyOwner,
        store_data: KvCacheStoreData,
    ) -> Result<(), KvCacheEventError> {
        let mut parent_hash = store_data.parent_hash;
        let indexed_owner = IndexedResidencyOwner::from_exact(owner);
        let owner_state = worker_blocks
            .entry(indexed_owner)
            .or_insert_with(|| OwnerBlockState {
                owner,
                id: self.owners.intern(indexed_owner),
                blocks: OwnerBlockIndex::default(),
            });
        if owner_state.owner != owner {
            return Err(KvCacheEventError::UnsupportedResidencyDomain);
        }
        let owner_id = owner_state.id;
        let worker_map = &mut owner_state.blocks;

        for block in store_data.blocks {
            let key = TransitionKey::new(parent_hash, block.tokens_hash);

            // If this worker already has a different parent/local for the same
            // block_hash, or if the shared edge is owned by a conflicting
            // child_hash, stop the walk: any further blocks in this chain would
            // hang off an edge this index never accepted for the worker.
            if worker_map
                .get(&block.block_hash)
                .is_some_and(|existing_key| *existing_key != key)
            {
                break;
            }

            let inserted = match self.edges.entry(key) {
                dashmap::mapref::entry::Entry::Occupied(mut edge) => {
                    let edge = edge.get_mut();
                    if edge.child_hash == block.block_hash {
                        edge.owners.insert(owner_id, indexed_owner, &self.owners);
                        true
                    } else {
                        false
                    }
                }
                dashmap::mapref::entry::Entry::Vacant(edge) => {
                    edge.insert(FlatEdge {
                        child_hash: block.block_hash,
                        owners: OwnerSet::Single(owner_id),
                    });
                    true
                }
            };

            if !inserted {
                break;
            }

            worker_map.insert(block.block_hash, key);
            parent_hash = Some(block.block_hash);
        }
        Ok(())
    }

    fn remove_blocks_impl(
        &self,
        worker_blocks: &mut FlatLane,
        owner: ResidencyOwner,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> Result<(), KvCacheEventError> {
        let indexed_owner = IndexedResidencyOwner::from_exact(owner);
        let remove_worker_entry = {
            let Some(owner_state) = worker_blocks.get_mut(&indexed_owner) else {
                return Ok(());
            };
            if owner_state.owner != owner {
                return Err(KvCacheEventError::UnsupportedResidencyDomain);
            }
            let owner_id = owner_state.id;
            let worker_map = &mut owner_state.blocks;

            for block_hash in block_hashes {
                let Some(key) = worker_map.remove(block_hash) else {
                    continue;
                };

                self.remove_owner_from_edge(key, owner_id, indexed_owner);
            }

            worker_map.is_empty()
        };

        if remove_worker_entry {
            worker_blocks.remove(&indexed_owner);
        }

        Ok(())
    }

    pub(super) fn clear(&self, worker_blocks: &mut FlatLane, owner: ResidencyOwner) {
        let indexed_owner = IndexedResidencyOwner::from_exact(owner);
        let Some(owner_state) = worker_blocks.remove(&indexed_owner) else {
            return;
        };
        if owner_state.owner != owner {
            worker_blocks.insert(indexed_owner, owner_state);
            return;
        }

        for key in owner_state.blocks.into_values() {
            self.remove_owner_from_edge(key, owner_state.id, indexed_owner);
        }
    }

    fn remove_owner_from_edge(
        &self,
        key: TransitionKey,
        id: OwnerId,
        owner: IndexedResidencyOwner,
    ) {
        if let dashmap::mapref::entry::Entry::Occupied(mut edge) = self.edges.entry(key)
            && edge.get_mut().owners.remove(id, owner, &self.owners) == Removal::Emptied
        {
            edge.remove();
        }
    }

    pub(super) fn root_workers(
        &self,
        local_hash: LocalBlockHash,
        projection: &ResidencyProjection,
    ) -> Vec<WorkerWithDpRank> {
        super::root_workers(&self.edges, &self.owners, local_hash, projection)
    }

    pub(super) fn query<S: BuildHasher>(
        &self,
        local_hashes: &[LocalBlockHash],
        continuations: &std::collections::HashMap<WorkerWithDpRank, LowerTierContinuation, S>,
        retain_kv_transfer_extensions: bool,
        snapshot: &ResidencyRoutingSnapshot,
    ) -> LowerTierMatchDetails {
        super::query_edges(
            &self.edges,
            &self.owners,
            local_hashes,
            continuations,
            retain_kv_transfer_extensions,
            snapshot,
        )
    }

    pub(super) fn owners(&self) -> &OwnerTable {
        &self.owners
    }

    pub(super) fn cursor(&self) -> FlatCursor<'_> {
        self.edges.cursor()
    }

    /// Every edge: its key, child hash and owners.
    pub(super) fn held_edges(
        &self,
    ) -> FxHashMap<TransitionKey, (ExternalSequenceBlockHash, Vec<IndexedResidencyOwner>)> {
        self.edges
            .iter()
            .map(|edge| {
                let mut owners = Vec::new();
                edge.owners
                    .for_each_owner(&self.owners, |owner| owners.push(owner));
                (*edge.key(), (edge.child_hash, owners))
            })
            .collect()
    }
}
