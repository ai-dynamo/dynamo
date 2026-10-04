// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use super::children::NodeChildren;

impl ConcurrentRadixTreeCompressed {
    #[cfg(test)]
    pub(crate) fn run_cleanup_for_test(&self) {
        self.sweep_stale_children();
    }

    pub(super) fn sweep_stale_children(&self) {
        // Free expired retired snapshots first so their child `Arc`s are gone.
        NodeChildren::drain_graveyard(usize::MAX);
        let mut queue = VecDeque::from([self.root.clone()]);
        let mut edges = Vec::new();

        let mut guard = crossbeam_epoch::pin();
        let mut visited = 0usize;
        while let Some(parent) = queue.pop_front() {
            let children = parent.child_edges_snapshot();
            for (key, child) in children {
                queue.push_back(child.clone());
                edges.push(CleanupEdge {
                    parent: Arc::downgrade(&parent),
                    key,
                    child: Arc::downgrade(&child),
                });
            }
            // Let the epoch advance during long walks; child loads above nest in `guard`.
            visited += 1;
            if visited.is_multiple_of(64) {
                guard.repin();
            }
        }
        drop(guard);

        for edge in edges.into_iter().rev() {
            let Some(parent) = edge.parent.upgrade() else {
                continue;
            };
            let Some(child) = edge.child.upgrade() else {
                continue;
            };
            parent.remove_child_if_stale_leaf(edge.key, &child);
        }
    }

    /// Apply a remove operation (eviction).
    ///
    /// For each evicted block hash, finds its position in the node through its edge index.
    /// Updates the worker's match index without splitting the tree:
    /// - `pos >= current_cutoff`: no-op (already beyond coverage)
    /// - `pos < current_cutoff`: `new_cutoff = pos`; records the slot's cutoff in `cutoffs`
    ///   or removes it entirely if `new_cutoff == 0`.
    ///
    /// Lookup entries for the newly uncovered suffix are removed eagerly so
    /// later duplicate remove events fast-path through the missing-hash case.
    pub(super) fn apply_removed(
        &self,
        lookup: &mut LaneLookup,
        worker: WorkerWithDpRank,
        op: KvCacheRemoveData,
        id: u64,
        guard: &Guard,
    ) -> Result<(), KvCacheEventError> {
        if !lookup.contains_worker(worker) {
            return Err(KvCacheEventError::BlockNotFound);
        }

        let block_hashes = op.block_hashes;
        let table = self.slots.table(guard);
        let Some(slot) = table.slot_of(worker) else {
            // The rank is being removed: its slot is unmapped, and the sweep that releases
            // the slot drops its coverage. Only the lookup entries are left to scrub.
            self.remove_lookup_hashes(lookup, worker, &block_hashes);
            return Ok(());
        };
        let worker = EventWorker {
            rank: worker,
            slot,
            table,
        };
        let mut index = 0;

        while let Some(&block_hash) = block_hashes.get(index) {
            let Some(mut node) = lookup.node(worker.rank, block_hash) else {
                tracing::debug!(
                    worker_id = worker.rank.worker_id.to_string(),
                    dp_rank = worker.rank.dp_rank,
                    id,
                    block_hash = ?block_hash,
                    "Block not found during remove; skipping"
                );
                self.remove_lookup_hashes(lookup, worker.rank, &[block_hash]);
                index += 1;
                continue;
            };

            // The grouped removal validates `block_hash` against the edge under its own
            // lock, so a stale lookup entry costs no separate probe on the common path.
            loop {
                // TODO(CORRECTNESS): Invalidate this worker throughout the descendant
                // subtree when a mid-edge removal leaves the node alive for another
                // worker. Otherwise stale descendants can be reused as store parents,
                // reactivated by restoring only the removed block, or emitted by dumps
                // without a valid worker-specific parent. Preserve CRTC's locking and
                // snapshot guarantees when implementing the traversal.
                if let Some((consumed, stale_hashes)) =
                    node.remove_worker_for_leading_hashes(worker.slot, &block_hashes[index..])
                {
                    self.remove_lookup_hashes(lookup, worker.rank, &stale_hashes);
                    index += consumed;
                    break;
                }

                // A cross-thread split moved the hash below `node`; retry the run there.
                if let Some(resolved) = self.repair_stale(
                    lookup,
                    worker.table,
                    &node,
                    block_hash,
                    LookupRepairDirection::TowardHead,
                ) {
                    node = resolved;
                    continue;
                }

                tracing::debug!(
                    worker_id = worker.rank.worker_id.to_string(),
                    dp_rank = worker.rank.dp_rank,
                    id,
                    block_hash = ?block_hash,
                    "Block not found in subtree during remove; skipping"
                );
                // The remove event says this worker evicted the block, so its lookup
                // entry must not outlive the event. A repair miss with a live entry
                // happens when the hash's node was split off and the split child was
                // later dropped by clear_children_if_unreachable; without this scrub
                // the entry (and the per-worker tracked-block count) leaks permanently.
                self.remove_lookup_hashes(lookup, worker.rank, &[block_hash]);
                index += 1;
                break;
            }
        }

        Ok(())
    }

    fn remove_lookup_hashes(
        &self,
        lookup: &mut LaneLookup,
        worker: WorkerWithDpRank,
        hashes: &[ExternalSequenceBlockHash],
    ) {
        // Check the lifecycle once per run, not once per hash: `release_hash` is not
        // inlined, and a call per evicted block is a measurable share of writer time.
        if self.lifecycle.is_enabled() {
            lookup.remove_all(worker, hashes, |hash| self.release_hash(worker, hash));
        } else {
            lookup.remove_all(worker, hashes, |_| {});
        }
    }

    /// Drops `target`'s lookups on this lane, releasing their hashes.
    fn erase_lane_lookups(&self, lookup: &mut LaneLookup, target: WorkerRemovalTarget) {
        lookup.remove_workers(
            |worker| target.matches(worker),
            |worker, hash| self.release_hash(worker, hash),
        );
    }

    /// Applies a `Cleared` event: drops the rank's lookups on this lane and its coverage
    /// everywhere. The rank keeps its slot.
    pub(super) fn clear_worker_coverage(&self, lookup: &mut LaneLookup, worker: WorkerWithDpRank) {
        self.erase_lane_lookups(lookup, WorkerRemovalTarget::DpRank(worker));
        let slot = self.slots.table(&crossbeam_epoch::pin()).slot_of(worker);
        if let Some(slot) = slot {
            self.clear_rank_slot(worker, slot);
        }
        // A removal on another lane may have unmapped this slot or, if the rank has stored
        // since, an earlier one. Its sweep drops that coverage; wait for it.
        self.slots
            .wait_for_release(WorkerRemovalTarget::DpRank(worker));
    }

    /// Clears `slot` from the tree for as long as it is still `worker`'s. Returns false
    /// once a removal has unmapped it; that removal sweeps the rest before releasing it.
    pub(super) fn clear_rank_slot(&self, worker: WorkerWithDpRank, slot: Slot) -> bool {
        self.sweep_slots(&SlotSet::from_iter([slot]), |_, table| {
            table.slot_of(worker) == Some(slot)
        })
    }

    /// Removes `target`'s ranks: drops their lookups on this lane and, with `sweep_tree`,
    /// releases their slots. A released slot is unmapped first, so later events for the
    /// rank start over with a new slot, then swept out of the tree and freed.
    ///
    /// `ThreadPoolIndexer` removes a whole worker by sending this to every lane without
    /// `sweep_tree` and then to one lane with it; the rank keeps its slot until that sweep.
    pub(super) fn remove_worker_coverage(
        &self,
        lookup: &mut LaneLookup,
        target: WorkerRemovalTarget,
        sweep_tree: bool,
    ) {
        self.erase_lane_lookups(lookup, target);
        if !sweep_tree {
            return;
        }

        let slots = self.slots.unmap(target);
        if !slots.is_empty() {
            // Events on other lanes that resolved a slot before the unmap may still be
            // writing its bits; let them finish so the sweep below sees every bit.
            wait_for_pinned_threads();
            self.sweep_slots(&slots.iter().copied().collect(), |_, _| true);
            self.slots.release(slots);
        }
        // A removal on another lane may have unmapped some of these ranks first; return
        // only once its sweep has dropped their coverage too.
        self.slots.wait_for_release(target);
    }

    /// Clears `slots` from every node reachable from the root or an anchor, stopping
    /// with `false` as soon as `proceed` rejects a node. `proceed` gets the slot table
    /// current under the pin that stays held while the node is cleared, so a check
    /// against it holds back the release of any slot it sees mapped.
    pub(super) fn sweep_slots(
        &self,
        slots: &SlotSet,
        mut proceed: impl FnMut(&SharedNode, &SlotTable) -> bool,
    ) -> bool {
        let mut queue = VecDeque::new();
        self.root.push_children_into(&mut queue);
        let anchor_roots: Vec<_> = self
            .anchor_nodes
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        queue.extend(anchor_roots);

        // No visited set: the tree has no cycles and clearing a node twice is harmless.
        // Deduplicating by address would be unsound, because a node cleared and dropped
        // here can be freed and its address reused by a split suffix carrying the slots.
        let mut guard = crossbeam_epoch::pin();
        let mut visited = 0usize;
        while let Some(node) = queue.pop_front() {
            // Let the epoch advance during long sweeps.
            if visited.is_multiple_of(64) {
                guard.repin();
            }
            visited += 1;
            if !proceed(&node, self.slots.table(&guard)) {
                return false;
            }
            queue.extend(node.remove_slots_and_snapshot_children(slots));
        }
        true
    }
}
