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
        // Reuse one BFS queue. `push_children_into` preserves the same per-node
        // snapshot semantics as `children_snapshot`, but avoids allocating a
        // fresh child Vec for each scanned node during lookup repair.
        let mut queue = VecDeque::new();
        start.push_children_into(&mut queue);
        while let Some(node) = queue.pop_front() {
            if node.contains_edge_hash(hash) {
                return Some(node);
            }
            node.push_children_into(&mut queue);
        }
        None
    }

    /// Resolve `hash`, which a locked operation found missing from `stale`, to the
    /// descendant a cross-thread split moved it to, and repair the lookup range
    /// that `direction` will use.
    pub(super) fn repair_stale(
        &self,
        lookup: &mut LaneLookup,
        table: &SlotTable,
        stale: &SharedNode,
        hash: ExternalSequenceBlockHash,
        direction: LookupRepairDirection,
    ) -> Option<SharedNode> {
        let resolved = Self::find_in_subtree(stale, hash)?;
        #[cfg(any(test, feature = "bench"))]
        self.bench_metrics
            .lookup_repair_scans
            .fetch_add(1, Ordering::Relaxed);
        self.repair_lookup_for_resolved_node(lookup, table, hash, stale, &resolved, direction);
        Some(resolved)
    }

    /// Look up `hash` in a worker's lookup, resolving stale entries caused by
    /// cross-thread splits. Returns the `SharedNode` whose edge contains `hash`.
    pub(super) fn resolve_lookup(
        &self,
        lookup: &mut LaneLookup,
        worker: EventWorker<'_>,
        hash: ExternalSequenceBlockHash,
        direction: LookupRepairDirection,
    ) -> Option<SharedNode> {
        let node = lookup.node(worker.rank, hash)?;
        if node.contains_edge_hash(hash) {
            return Some(node);
        }
        self.repair_stale(lookup, worker.table, &node, hash, direction)
    }

    /// Repoints this lane's entries that still name `stale` at `resolved`, over the range
    /// of `resolved`'s edge each rank covers on the side of `hash` that `direction` picks.
    /// Entries naming any other node are left to their own lazy repair, so an entry never
    /// moves onto a node only because a recycled slot inherited stale bits there.
    pub(super) fn repair_lookup_for_resolved_node(
        &self,
        lookup: &mut LaneLookup,
        table: &SlotTable,
        hash: ExternalSequenceBlockHash,
        stale: &SharedNode,
        resolved: &SharedNode,
        direction: LookupRepairDirection,
    ) {
        let _changed_entries = lookup.redirect(stale, resolved, |worker| {
            table
                .slot_of(worker)
                .map(|slot| resolved.lookup_hashes_for_slot_repair(slot, hash, direction))
                .unwrap_or_default()
        });

        #[cfg(any(test, feature = "bench"))]
        self.bench_metrics
            .lookup_repair_entries
            .fetch_add(_changed_entries as u64, Ordering::Relaxed);
    }
}
