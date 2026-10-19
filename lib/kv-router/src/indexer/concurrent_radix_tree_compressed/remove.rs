// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

impl ConcurrentRadixTreeCompressed {
    #[cfg(test)]
    pub(crate) fn run_cleanup_for_test(&self) -> usize {
        self.sweep_stale_children()
    }

    /// Unlinks holder-less, childless leaves that nothing else references, under the root
    /// and every anchor, and returns how many it unlinked.
    ///
    /// The walk takes no gate: it records only holder-less children and descends only
    /// into nodes with children, visiting children by reference. Each parent's candidates
    /// are then re-checked and unlinked under one try-locked exclusive gate, deepest
    /// parents first, so a dead chain goes in one pass. A parent whose gate stays taken
    /// keeps its dead children until the next sweep.
    pub(super) fn sweep_stale_children(&self) -> usize {
        const GATE_ATTEMPTS: usize = 4;
        let mut queue = VecDeque::from([self.root.clone()]);
        queue.extend(self.anchor_nodes.iter().map(|entry| entry.value().clone()));
        let mut candidates = Vec::new();

        while let Some(parent) = queue.pop_front() {
            parent.for_each_child(|key, child| {
                if child.is_holderless() {
                    candidates.push(CleanupEdge {
                        parent: Arc::downgrade(&parent),
                        key,
                        child: Arc::downgrade(child),
                    });
                }
                if child.has_children() {
                    queue.push_back(child.clone());
                }
            });
        }

        // A parent's candidates are contiguous, since the walk lists one parent's children
        // at a time, and later parents are deeper, so walking groups backwards is
        // deepest-first.
        let mut removed = 0;
        let mut end = candidates.len();
        while end > 0 {
            let parent = candidates[end - 1].parent.as_ptr();
            let mut start = end - 1;
            while start > 0 && std::ptr::eq(candidates[start - 1].parent.as_ptr(), parent) {
                start -= 1;
            }
            let group = &candidates[start..end];
            end = start;
            let batch: Vec<_> = group
                .iter()
                .filter(|edge| {
                    edge.child
                        .upgrade()
                        .is_some_and(|child| child.is_holderless() && !child.has_children())
                })
                .map(|edge| (edge.key, edge.child.clone()))
                .collect();
            if batch.is_empty() {
                continue;
            }
            let Some(parent) = group[0].parent.upgrade() else {
                continue;
            };
            for attempt in 1..=GATE_ATTEMPTS {
                if let Some(count) = parent.try_remove_stale_children(&batch) {
                    removed += count;
                    break;
                }
                if attempt < GATE_ATTEMPTS {
                    std::thread::yield_now();
                }
            }
        }
        removed
    }

    /// Queues a leaf this lane's removal just killed for the lane's next idle drain. A
    /// leaf whose parent cannot be resolved is left to the sweep.
    fn queue_dead_leaf(
        &self,
        lookup: &mut LaneLookup,
        worker: WorkerWithDpRank,
        node: &SharedNode,
    ) {
        let Some((key, parent_tail)) = node.dead_leaf_key_and_tail() else {
            return;
        };
        let Some(parent) = self.dead_leaf_parent(lookup, worker, node, key, parent_tail) else {
            return;
        };
        self.push_pending_unlink(
            lookup,
            CleanupEdge {
                parent: Arc::downgrade(&parent),
                key,
                child: Arc::downgrade(node),
            },
        );
    }

    /// Queues a dead leaf on this lane, draining the oldest quarter of the queue at once
    /// if it is full.
    fn push_pending_unlink(&self, lookup: &mut LaneLookup, edge: CleanupEdge) {
        lookup.pending_unlinks.push(edge);
        if lookup.pending_unlinks.len() >= PENDING_UNLINK_LIMIT {
            self.drain_pending_unlinks(lookup, Some(PENDING_UNLINK_LIMIT / 4));
        }
    }

    /// The parent of a dead leaf: the root, or the node whose edge ends with the leaf's
    /// parent tail, which is an anchor or wherever the evicting worker's lookup places
    /// that block. Verified by identity. A stale lookup entry is not repaired on the event
    /// path; it leaves the leaf to the sweep.
    fn dead_leaf_parent(
        &self,
        lookup: &mut LaneLookup,
        worker: WorkerWithDpRank,
        node: &SharedNode,
        key: LocalBlockHash,
        parent_tail: ExternalSequenceBlockHash,
    ) -> Option<SharedNode> {
        if self.root.has_child(key, node) {
            return Some(self.root.clone());
        }
        let parent = match self.anchor_nodes.get(&parent_tail) {
            Some(anchor) => anchor.value().clone(),
            None => lookup
                .node(worker, parent_tail)
                .filter(|parent| parent.contains_edge_hash(parent_tail))?,
        };
        parent.has_child(key, node).then_some(parent)
    }

    /// Queues a split suffix that was born holder-less and childless, which no removal
    /// will kill, and counts it toward the next sweep.
    pub(super) fn queue_split_born_dead_leaf(
        &self,
        lookup: &mut LaneLookup,
        prefix: &SharedNode,
        suffix: &SharedNode,
    ) {
        let Some((key, _)) = suffix.dead_leaf_key_and_tail() else {
            return;
        };
        self.note_death();
        self.push_pending_unlink(
            lookup,
            CleanupEdge {
                parent: Arc::downgrade(prefix),
                key,
                child: Arc::downgrade(suffix),
            },
        );
    }

    /// Unlinks up to `limit` of this lane's oldest queued dead leaves, or all of them for
    /// `None`, under one try-locked gate and one version bump per parent. A leaf whose
    /// parent is busy, that regained a holder or a child, or that something still
    /// references leaves the queue and is left to the sweep.
    pub(super) fn drain_pending_unlinks(&self, lookup: &mut LaneLookup, limit: Option<usize>) {
        let count = limit.map_or(lookup.pending_unlinks.len(), |limit| {
            limit.min(lookup.pending_unlinks.len())
        });
        // Oldest first: the newest entry may be the leaf a removal is still holding.
        let mut pending: Vec<_> = lookup.pending_unlinks.drain(..count).collect();
        pending.sort_unstable_by_key(|edge| edge.parent.as_ptr() as usize);
        for group in pending.chunk_by(|a, b| std::ptr::eq(a.parent.as_ptr(), b.parent.as_ptr())) {
            let Some(parent) = group[0].parent.upgrade() else {
                continue;
            };
            let batch: Vec<_> = group
                .iter()
                .map(|edge| (edge.key, edge.child.clone()))
                .collect();
            let _ = parent.try_remove_stale_children(&batch);
        }
    }

    /// Apply a remove operation (eviction).
    ///
    /// For each evicted block hash, finds its position in the node through its edge index.
    /// Updates the worker's match index without splitting the tree:
    /// - `pos >= current_cutoff`: no-op (already beyond coverage)
    /// - `pos < current_cutoff`: `new_cutoff = pos`; moves worker to `worker_cutoffs`
    ///   or removes entirely if `new_cutoff == 0`.
    ///
    /// Lookup entries for the newly uncovered suffix are removed eagerly so
    /// later duplicate remove events fast-path through the missing-hash case.
    pub(super) fn apply_removed(
        &self,
        lookup: &mut LaneLookup,
        worker: WorkerWithDpRank,
        op: KvCacheRemoveData,
        id: u64,
    ) -> Result<(), KvCacheEventError> {
        if !lookup.contains_worker(worker) {
            return Err(KvCacheEventError::BlockNotFound);
        }

        let block_hashes = op.block_hashes;
        let mut index = 0;

        while let Some(&block_hash) = block_hashes.get(index) {
            let Some(mut node) = lookup.node(worker, block_hash) else {
                tracing::debug!(
                    worker_id = worker.worker_id.to_string(),
                    dp_rank = worker.dp_rank,
                    id,
                    block_hash = ?block_hash,
                    "Block not found during remove; skipping"
                );
                self.remove_lookup_hashes(lookup, worker, &[block_hash]);
                index += 1;
                continue;
            };

            let mut died = false;
            // The grouped removal validates `block_hash` against the edge under its own
            // lock, so a stale lookup entry costs no separate probe on the common path.
            loop {
                // TODO(CORRECTNESS): Invalidate this worker throughout the descendant
                // subtree when a mid-edge removal leaves the node alive for another
                // worker. Otherwise stale descendants can be reused as store parents,
                // reactivated by restoring only the removed block, or emitted by dumps
                // without a valid worker-specific parent. Preserve CRTC's locking and
                // snapshot guarantees when implementing the traversal.
                //
                // Group only the following hashes whose entries name `node`. A hash whose
                // entry names another node is left to its own removal: that node may hold
                // the worker's coverage, since `node` can still contain the hash after
                // being unlinked when the worker later re-stored it elsewhere.
                let run_end =
                    index + 1 + lookup.leading_naming(worker, &node, &block_hashes[index + 1..]);
                if let Some((consumed, stale_hashes, became_dead)) =
                    node.remove_worker_for_leading_hashes(worker, &block_hashes[index..run_end])
                {
                    self.remove_lookup_hashes(lookup, worker, &stale_hashes);
                    if became_dead {
                        died = true;
                        self.queue_dead_leaf(lookup, worker, &node);
                    }
                    index += consumed;
                    break;
                }

                // A cross-thread split moved the hash below `node`; retry the run there.
                if let Some(resolved) =
                    self.repair_stale(lookup, &node, block_hash, LookupRepairDirection::TowardHead)
                {
                    node = resolved;
                    continue;
                }

                tracing::debug!(
                    worker_id = worker.worker_id.to_string(),
                    dp_rank = worker.dp_rank,
                    id,
                    block_hash = ?block_hash,
                    "Block not found in subtree during remove; skipping"
                );
                // The remove event says this worker evicted the block, so its lookup
                // entry must not outlive the event. A repair miss with a live entry
                // happens when the hash's node was split off and the split child was
                // later dropped by clear_children_if_unreachable; without this scrub
                // the entry (and the per-worker tracked-block count) leaks permanently.
                self.remove_lookup_hashes(lookup, worker, &[block_hash]);
                index += 1;
                break;
            }
            // Counted only once `node` is dropped, so a sweep that sees the count finds the
            // dead leaf referenced by its parent alone.
            drop(node);
            if died {
                self.note_death();
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

    pub(super) fn erase_worker_coverage(
        &self,
        lookup: &mut LaneLookup,
        target: WorkerRemovalTarget,
        sweep_tree: bool,
    ) {
        lookup.remove_workers(
            |worker| target.matches(worker),
            |worker, hash| self.release_hash(worker, hash),
        );
        if !sweep_tree {
            return;
        }

        let mut queue = VecDeque::new();
        self.root.push_children_into(&mut queue);
        let anchor_roots: Vec<_> = self
            .anchor_nodes
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        queue.extend(anchor_roots);

        let mut seen = FxHashSet::<usize>::default();
        while let Some(node) = queue.pop_front() {
            let ptr = Arc::as_ptr(&node) as usize;
            if !seen.insert(ptr) {
                continue;
            }

            let children = node.remove_target_and_snapshot_children(target);
            queue.extend(children);
        }
    }
}
