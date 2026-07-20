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
    /// For each evicted block hash, finds its position in the node via `edge_index` (O(1)).
    /// Updates the worker's match index without splitting the tree:
    /// - `pos >= current_cutoff`: no-op (already beyond coverage)
    /// - `pos < current_cutoff`: `new_cutoff = pos`; moves worker to `worker_cutoffs`
    ///   or removes entirely if `new_cutoff == 0`.
    ///
    /// Lookup entries for the newly uncovered suffix are removed eagerly so
    /// later duplicate remove events fast-path through the missing-hash case.
    pub(super) fn apply_removed(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker: WorkerWithDpRank,
        op: KvCacheRemoveData,
        id: u64,
    ) -> Result<(), KvCacheEventError> {
        if !lookup.contains_key(&worker) {
            return Err(KvCacheEventError::BlockNotFound);
        }

        let mut group_node: Option<SharedNode> = None;
        let mut group_hashes: Vec<ExternalSequenceBlockHash> = Vec::new();

        for block_hash in op.block_hashes {
            if group_node
                .as_ref()
                .is_some_and(|node| node.contains_edge_hash(block_hash))
            {
                group_hashes.push(block_hash);
                continue;
            }

            self.apply_removed_group(lookup, worker, group_node.take(), &group_hashes, id);
            group_hashes.clear();

            match self.resolve_lookup(
                lookup,
                worker,
                block_hash,
                LookupRepairDirection::TowardHead,
            ) {
                Some(node) => {
                    group_node = Some(node);
                    group_hashes.push(block_hash);
                }
                None => {
                    tracing::debug!(
                        worker_id = worker.worker_id.to_string(),
                        dp_rank = worker.dp_rank,
                        id,
                        block_hash = ?block_hash,
                        "Block not found during remove; skipping"
                    );
                    // The remove event says this worker evicted the block, so
                    // any lookup entry for it must not outlive the event. A
                    // resolve miss with a live entry happens when the hash's
                    // node was split off and the split child was later dropped
                    // by clear_children_if_unreachable — without this scrub the
                    // entry (and the per-worker tracked-block count) leaks
                    // permanently. Mirrors the scrubs in apply_removed_hash's
                    // miss branches.
                    Self::remove_lookup_hashes(lookup, worker, [block_hash]);
                }
            }
        }

        self.apply_removed_group(lookup, worker, group_node, &group_hashes, id);

        Ok(())
    }

    pub(super) fn apply_removed_group(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker: WorkerWithDpRank,
        node: Option<SharedNode>,
        block_hashes: &[ExternalSequenceBlockHash],
        id: u64,
    ) {
        let Some(cur_node) = node else {
            return;
        };
        if block_hashes.is_empty() {
            return;
        }

        match cur_node.remove_worker_for_hashes(worker, block_hashes) {
            Some(outcome) => {
                Self::remove_lookup_hashes(lookup, worker, outcome.stale_hashes);
                for block_hash in outcome.unmatched_hashes {
                    self.apply_removed_hash(lookup, worker, block_hash, id);
                }
            }
            None => {
                for &block_hash in block_hashes {
                    self.apply_removed_hash(lookup, worker, block_hash, id);
                }
            }
        }
    }

    fn apply_removed_hash(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker: WorkerWithDpRank,
        block_hash: ExternalSequenceBlockHash,
        id: u64,
    ) {
        let Some(mut cur_node) = self.resolve_lookup(
            lookup,
            worker,
            block_hash,
            LookupRepairDirection::TowardHead,
        ) else {
            tracing::debug!(
                worker_id = worker.worker_id.to_string(),
                dp_rank = worker.dp_rank,
                id,
                block_hash = ?block_hash,
                "Block not found during batched remove fallback; skipping"
            );
            Self::remove_lookup_hashes(lookup, worker, [block_hash]);
            return;
        };

        loop {
            // TODO(CORRECTNESS): Invalidate this worker throughout the descendant
            // subtree when a mid-edge removal leaves the node alive for another
            // worker. Otherwise stale descendants can be reused as store parents,
            // reactivated by restoring only the removed block, or emitted by dumps
            // without a valid worker-specific parent. Preserve CRTC's locking and
            // snapshot guarantees when implementing the traversal.
            match cur_node.remove_worker_for_hashes(worker, std::slice::from_ref(&block_hash)) {
                Some(outcome) => {
                    debug_assert!(outcome.unmatched_hashes.is_empty());
                    Self::remove_lookup_hashes(lookup, worker, outcome.stale_hashes);
                    return;
                }
                None => {
                    // Hash was moved to a descendant by a concurrent split.
                    match Self::find_in_subtree(&cur_node, block_hash) {
                        Some(resolved) => {
                            self.repair_lookup_for_resolved_node(
                                lookup,
                                block_hash,
                                &resolved,
                                LookupRepairDirection::TowardHead,
                            );
                            #[cfg(feature = "bench")]
                            self.bench_metrics
                                .lookup_repair_scans
                                .fetch_add(1, Ordering::Relaxed);
                            cur_node = resolved;
                            // Retry the loop with the resolved node.
                        }
                        None => {
                            // Hash not found anywhere -- evicted by a concurrent clear.
                            tracing::debug!(
                                worker_id = worker.worker_id.to_string(),
                                dp_rank = worker.dp_rank,
                                id,
                                block_hash = ?block_hash,
                                "Block not found in subtree during batched remove; skipping"
                            );
                            Self::remove_lookup_hashes(lookup, worker, [block_hash]);
                            return;
                        }
                    }
                }
            }
        }
    }

    fn remove_lookup_hashes(
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker: WorkerWithDpRank,
        hashes: impl IntoIterator<Item = ExternalSequenceBlockHash>,
    ) {
        if let Some(wl) = lookup.get_mut(&worker) {
            for hash in hashes {
                wl.remove(&hash);
            }
        }
    }

    pub(super) fn erase_worker_coverage(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        target: WorkerRemovalTarget,
        sweep_tree: bool,
    ) {
        lookup.retain(|worker, _| !target.matches(*worker));
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
