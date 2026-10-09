// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use super::children::NodeChildren;

impl ConcurrentRadixTreeCompressed {
    #[cfg(test)]
    pub(crate) fn run_cleanup_for_test(&self) {
        self.sweep_stale_children();
    }

    /// Unlinks holder-less, childless leaves reachable from the root or a branch anchor.
    ///
    /// One BFS from the root and every anchor visits children by reference under the
    /// walk's pin. It records an edge only for a holder-less child, and enqueues (clones)
    /// only children that have children of their own. The unlink phase goes deepest-first:
    /// an unlocked pre-check skips candidates that are covered or have children again,
    /// and `remove_child_if_stale_leaf` re-validates under the parent's exclusive gate, a
    /// child `try_write`, and the exact live strong count. A holder-less internal node
    /// whose children were all unlinked earlier in the pass passes the pre-check when its
    /// turn comes, so a cascade completes in one pass. Anchors are never unlinked.
    ///
    /// The walk recounts linked and holder-less blocks exactly; the reclamation estimates
    /// are overwritten with those counts minus what was unlinked.
    pub(super) fn sweep_stale_children(&self) -> SweepOutcome {
        let started = std::time::Instant::now();
        // Free some expired garbage first so its child `Arc`s are gone. Leftovers keep
        // their children's `retired_snapshot_refs` counted, which the strong-count check
        // subtracts, so a bounded drain only delays unlinks it would block.
        NodeChildren::drain_graveyard(GRAVEYARD_NODES_PER_TASK);

        let mut queue = VecDeque::from([self.root.clone()]);
        queue.extend(self.anchor_nodes.iter().map(|entry| entry.value().clone()));
        let mut candidates = Vec::new();
        let mut outcome = SweepOutcome::default();

        let mut guard = crossbeam_epoch::pin();
        let mut visited = 0usize;
        while let Some(parent) = queue.pop_front() {
            let mut weak_parent = None;
            parent.for_each_child(&guard, |key, child| {
                let (len, holder_less) = child.sweep_probe();
                outcome.nodes += 1;
                outcome.linked += len as u64;
                if holder_less {
                    outcome.dead += len as u64;
                    candidates.push(CleanupEdge {
                        parent: weak_parent
                            .get_or_insert_with(|| Arc::downgrade(&parent))
                            .clone(),
                        key,
                        child: Arc::downgrade(child),
                    });
                }
                if child.has_children_in(&guard) {
                    queue.push_back(child.clone());
                }
            });
            // Let the epoch advance during long walks. The queue owns its `Arc`s, so no
            // borrow outlives the pin.
            visited += 1;
            if visited.is_multiple_of(64) {
                guard.repin();
            }
        }
        drop(guard);

        outcome.candidates = candidates.len() as u64;
        for edge in candidates.into_iter().rev() {
            let (Some(parent), Some(child)) = (edge.parent.upgrade(), edge.child.upgrade()) else {
                continue;
            };
            if !child.looks_reclaimable() {
                outcome.skipped_busy += 1;
                continue;
            }
            match parent.remove_child_if_stale_leaf(edge.key, &child) {
                StaleLeafOutcome::Unlinked { edge_len } => {
                    outcome.reclaimed_nodes += 1;
                    outcome.reclaimed_blocks += edge_len as u64;
                }
                StaleLeafOutcome::Held => outcome.skipped_held += 1,
                StaleLeafOutcome::Busy => outcome.skipped_busy += 1,
                StaleLeafOutcome::Detached => {}
            }
        }

        self.reclaim.finish_sweep(started, &outcome);
        outcome
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
            let Some(origin) = lookup.node(worker.rank, block_hash) else {
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
            // The node lookup repair moved the run to, if any. Entries that still name
            // `origin` are stale entries for hashes that moved with it.
            let mut resolved: Option<SharedNode> = None;

            // The grouped removal validates `block_hash` against the edge under its own
            // lock, so a stale lookup entry costs no separate probe on the common path.
            loop {
                let node = resolved.as_ref().unwrap_or(&origin);
                let stale_origin = resolved.is_some().then_some(&origin);
                // TODO(CORRECTNESS): Invalidate this worker throughout the descendant
                // subtree when a mid-edge removal leaves the node alive for another
                // worker. Otherwise stale descendants can be reused as store parents,
                // reactivated by restoring only the removed block, or emitted by dumps
                // without a valid worker-specific parent. Preserve CRTC's locking and
                // snapshot guarantees when implementing the traversal.
                //
                // The run takes only the following hashes whose entries name this node
                // (or the stale node repair resolved it from). An entry naming another
                // node marks where the worker's coverage of that hash lives: once cleanup
                // unlinks a subtree the lane still names, a partial restore can store some
                // of its hashes on a new live node, and consuming them here would leave
                // that coverage behind.
                let run =
                    lookup.run_naming(worker.rank, &block_hashes[index..], node, stale_origin);
                if let Some(removal) = node.remove_worker_for_leading_hashes(
                    worker.slot,
                    &block_hashes[index..index + run],
                ) {
                    lookup.tally.dead += removal.dead_blocks as i64;
                    self.remove_lookup_hashes_naming(
                        lookup,
                        worker.rank,
                        &removal.stale_hashes,
                        node,
                        stale_origin,
                    );
                    index += removal.consumed;
                    break;
                }

                // A cross-thread split moved the hash below `node`; retry the run there.
                if let Some(next) = self.repair_stale(
                    lookup,
                    worker.table,
                    node,
                    block_hash,
                    LookupRepairDirection::TowardHead,
                ) {
                    resolved = Some(next);
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

    /// Scrubs `worker`'s entries for `hashes`, which a removal just uncovered on `node`,
    /// that name `node` or `origin`, the stale node lookup repair resolved `node` from.
    /// An entry naming any other node stays: the worker still holds that hash there, and
    /// only that hash's own removal may drop it. Releases only the hashes whose entries
    /// were removed, plus hashes with no entry at all.
    fn remove_lookup_hashes_naming(
        &self,
        lookup: &mut LaneLookup,
        worker: WorkerWithDpRank,
        hashes: &[ExternalSequenceBlockHash],
        node: &SharedNode,
        origin: Option<&SharedNode>,
    ) {
        if self.lifecycle.is_enabled() {
            lookup.remove_all_naming(worker, hashes, node, origin, |hash| {
                self.release_hash(worker, hash)
            });
        } else {
            lookup.remove_all_naming(worker, hashes, node, origin, |_| {});
        }
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
            self.clear_rank_slot(worker, slot, &mut lookup.tally.dead);
        }
        // A removal on another lane may have unmapped this slot or, if the rank has stored
        // since, an earlier one. Its sweep drops that coverage; wait for it.
        self.slots
            .wait_for_release(WorkerRemovalTarget::DpRank(worker));
    }

    /// Clears `slot` from the tree for as long as it is still `worker`'s. Returns false
    /// once a removal has unmapped it; that removal sweeps the rest before releasing it.
    /// Adds the blocks of nodes it leaves holder-less to `dead`.
    pub(super) fn clear_rank_slot(
        &self,
        worker: WorkerWithDpRank,
        slot: Slot,
        dead: &mut i64,
    ) -> bool {
        self.sweep_slots(&SlotSet::from_iter([slot]), dead, |_, table| {
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
            self.sweep_slots(
                &slots.iter().copied().collect(),
                &mut lookup.tally.dead,
                |_, _| true,
            );
            self.slots.release(slots);
        }
        // A removal on another lane may have unmapped some of these ranks first; return
        // only once its sweep has dropped their coverage too.
        self.slots.wait_for_release(target);
    }

    /// Clears `slots` from every node reachable from the root or an anchor, stopping
    /// with `false` as soon as `proceed` rejects a node. `proceed` gets the slot table
    /// current under the pin that stays held while the node is cleared, so a check
    /// against it holds back the release of any slot it sees mapped. Adds the blocks of
    /// nodes it leaves holder-less to `dead`.
    pub(super) fn sweep_slots(
        &self,
        slots: &SlotSet,
        dead: &mut i64,
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
            let (children, dead_blocks) = node.remove_slots_and_snapshot_children(slots);
            *dead += dead_blocks as i64;
            queue.extend(children);
        }
        true
    }
}
