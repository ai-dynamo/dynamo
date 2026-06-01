// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

impl ConcurrentRadixTreeCompressed {
    /// Search a node's subtree for the node whose edge contains `hash`.
    /// Used to resolve stale lookup entries caused by cross-thread splits.
    pub(super) fn find_in_subtree(
        start: &SharedNode,
        hash: ExternalSequenceBlockHash,
    ) -> Option<SharedNode> {
        let mut queue = VecDeque::from(start.children_snapshot());
        while let Some(node) = queue.pop_front() {
            if node.contains_edge_hash(hash) {
                return Some(node);
            }
            queue.extend(node.children_snapshot());
        }
        None
    }

    /// Look up `hash` in a worker's lookup, resolving stale entries caused by
    /// cross-thread splits. Returns the `SharedNode` whose edge contains `hash`.
    pub(super) fn resolve_lookup(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker: WorkerWithDpRank,
        hash: ExternalSequenceBlockHash,
        direction: LookupRepairDirection,
    ) -> Option<SharedNode> {
        let node = lookup.get(&worker)?.get(&hash)?.clone();

        // Fast path: hash is still in this node's edge_index.
        if node.contains_edge_hash(hash) {
            return Some(node);
        }

        // Slow path: hash was moved to a descendant by a cross-thread split.
        let resolved = Self::find_in_subtree(&node, hash)?;
        #[cfg(feature = "bench")]
        self.bench_metrics
            .lookup_repair_scans
            .fetch_add(1, Ordering::Relaxed);
        self.repair_lookup_for_resolved_node(lookup, hash, &resolved, direction);
        Some(resolved)
    }

    pub(super) fn repair_lookup_for_resolved_node(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        hash: ExternalSequenceBlockHash,
        resolved: &SharedNode,
        direction: LookupRepairDirection,
    ) {
        #[cfg(feature = "bench")]
        let mut changed_entries = 0u64;

        for (&worker, worker_lookup) in lookup.iter_mut() {
            for covered_hash in resolved.lookup_hashes_for_worker_repair(worker, hash, direction) {
                let previous = worker_lookup.insert(covered_hash, resolved.clone());
                let changed = match previous {
                    Some(previous) => !Arc::ptr_eq(&previous, resolved),
                    None => true,
                };
                if changed {
                    #[cfg(feature = "bench")]
                    {
                        changed_entries += 1;
                    }
                }
            }
        }

        #[cfg(feature = "bench")]
        self.bench_metrics
            .lookup_repair_entries
            .fetch_add(changed_entries, Ordering::Relaxed);
    }
}
