// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

impl ConcurrentRadixTreeCompressed {
    #[cfg(test)]
    pub(crate) fn run_cleanup_for_test(&self) {
        self.sweep_stale_children();
    }

    pub(super) fn sweep_stale_children(&self) {
        let mut queue = VecDeque::from([self.root.clone()]);
        let mut edges = Vec::new();

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
        }

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
                if let Some((consumed, stale_hashes)) =
                    node.remove_worker_for_leading_hashes(worker, &block_hashes[index..run_end])
                {
                    self.remove_lookup_hashes(lookup, worker, &stale_hashes);
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
