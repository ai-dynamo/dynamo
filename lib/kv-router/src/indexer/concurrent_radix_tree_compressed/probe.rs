// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Probe hooks for tests and benchmarks: reclamation settings, quiescence, structural
//! and memory reports, and invariant checks.

use super::children::NodeChildren;
use super::*;

/// Structural counters and the state of stale-leaf reclamation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CrtcShapeReport {
    /// Nodes reachable from the root or an anchor, anchors excluded.
    pub nodes: u64,
    /// Blocks on those nodes.
    pub linked_blocks: u64,
    /// Blocks on those nodes that no rank covers.
    pub dead_blocks: u64,
    /// Holder-less childless leaves among them.
    pub dead_leaves: u64,
    /// Of those, leaves something besides their parent references.
    pub dead_leaves_held: u64,
    pub anchors: u64,
    pub splits: u64,
    pub repair_scans: u64,
    pub repair_entries: u64,
    /// The volume trigger's shared estimates.
    pub dead_estimate: u64,
    pub linked_estimate: u64,
    pub sweeps_volume: u64,
    /// Sweeps scheduled by the timer or run directly.
    pub sweeps_other: u64,
    pub sweep_total_us: u64,
    pub sweep_max_us: u64,
    pub reclaimed_nodes: u64,
    pub reclaimed_blocks: u64,
    pub skipped_held: u64,
    pub skipped_busy: u64,
}

/// Heap bytes the tree's nodes hold, by category. Approximate for `DashMap`s; exact for
/// the rest up to allocator rounding.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CrtcMemoryReport {
    pub nodes: u64,
    /// `Node` allocations, including the `Arc` counts.
    pub node_bytes: u64,
    /// Bytes of edge entries in use.
    pub edge_len_bytes: u64,
    /// Bytes of edge allocations, in use or not.
    pub edge_capacity_bytes: u64,
    pub edge_index_bytes: u64,
    pub cutoff_bytes: u64,
    pub compact_child_bytes: u64,
    pub sharded_child_bytes: u64,
    pub coverage_overflow_bytes: u64,
    /// Graves waiting in the shared graveyard.
    pub shared_graves: u64,
}

impl CrtcMemoryReport {
    /// Edge capacity over edge length.
    pub fn edge_slack(&self) -> f64 {
        if self.edge_len_bytes == 0 {
            return 1.0;
        }
        self.edge_capacity_bytes as f64 / self.edge_len_bytes as f64
    }

    pub fn total_bytes(&self) -> u64 {
        self.node_bytes
            + self.edge_capacity_bytes
            + self.edge_index_bytes
            + self.cutoff_bytes
            + self.compact_child_bytes
            + self.sharded_child_bytes
            + self.coverage_overflow_bytes
    }
}

/// One node as the probes see it.
pub(super) struct NodeProbe {
    pub(super) edge_len: usize,
    pub(super) edge_capacity: usize,
    pub(super) edge_index_slots: usize,
    pub(super) cutoffs_capacity: usize,
    pub(super) holder_less: bool,
    pub(super) childless: bool,
    #[allow(dead_code, reason = "reported for debugging")]
    pub(super) anchor: bool,
    pub(super) compact_child_bytes: usize,
    pub(super) sharded_child_bytes: usize,
    pub(super) coverage_overflow_bytes: usize,
}

impl ConcurrentRadixTreeCompressed {
    /// Replaces the reclamation settings.
    pub fn probe_set_reclaim(&self, config: ReclaimConfig) {
        self.reclaim.configure(config);
    }

    /// Finishes deferred work: a full sweep, then epoch flushes and graveyard drains until
    /// nothing more is freed. Garbage still in other threads' local epoch bags stays.
    pub fn probe_quiesce(&self) {
        self.sweep_stale_children();
        for _ in 0..64 {
            NodeChildren::flush_retired();
            if NodeChildren::drain_graveyard(usize::MAX) {
                break;
            }
        }
        // Whatever the drains released may have held candidates; sweep once more.
        self.sweep_stale_children();
        NodeChildren::flush_retired();
        NodeChildren::drain_graveyard(usize::MAX);
    }

    /// Visits every node reachable from the root or an anchor once per parent edge, as
    /// `sweep_slots` does, with the node and whether it is an anchor seed.
    fn probe_walk(&self, mut visit: impl FnMut(&SharedNode, bool)) {
        let mut queue = VecDeque::new();
        self.root.push_children_into(&mut queue);
        let anchors: Vec<_> = self
            .anchor_nodes
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        for anchor in &anchors {
            visit(anchor, true);
            anchor.push_children_into(&mut queue);
        }
        while let Some(node) = queue.pop_front() {
            visit(&node, false);
            node.push_children_into(&mut queue);
        }
    }

    pub fn probe_shape(&self) -> CrtcShapeReport {
        let mut report = CrtcShapeReport::default();
        self.probe_walk(|node, anchor| {
            if anchor {
                report.anchors += 1;
                return;
            }
            let probe = node.probe();
            report.nodes += 1;
            report.linked_blocks += probe.edge_len as u64;
            if probe.holder_less {
                report.dead_blocks += probe.edge_len as u64;
                if probe.childless {
                    report.dead_leaves += 1;
                    // The walk's queue and the parent's map hold two references.
                    if Node::live_strong_count_for_probe(node).is_none_or(|count| count > 2) {
                        report.dead_leaves_held += 1;
                    }
                }
            }
        });
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        let stats = &self.reclaim.stats;
        (report.dead_estimate, report.linked_estimate) = self.reclaim.estimates();
        report.splits = load(&self.bench_metrics.node_splits);
        report.repair_scans = load(&self.bench_metrics.lookup_repair_scans);
        report.repair_entries = load(&self.bench_metrics.lookup_repair_entries);
        report.sweeps_volume = load(&stats.sweeps_volume);
        report.sweeps_other = load(&stats.sweeps_other);
        report.sweep_total_us = load(&stats.total_us);
        report.sweep_max_us = load(&stats.max_us);
        report.reclaimed_nodes = load(&stats.reclaimed_nodes);
        report.reclaimed_blocks = load(&stats.reclaimed_blocks);
        report.skipped_held = load(&stats.skipped_held);
        report.skipped_busy = load(&stats.skipped_busy);
        report
    }

    pub fn probe_memory(&self) -> CrtcMemoryReport {
        const ENTRY: usize = size_of::<(LocalBlockHash, ExternalSequenceBlockHash)>();
        // An `Arc` allocation holds both counts before the node.
        const NODE: usize = size_of::<Node>() + 2 * size_of::<usize>();
        let mut report = CrtcMemoryReport::default();
        let mut add = |probe: NodeProbe| {
            report.nodes += 1;
            report.node_bytes += NODE as u64;
            report.edge_len_bytes += (probe.edge_len * ENTRY) as u64;
            report.edge_capacity_bytes += (probe.edge_capacity * ENTRY) as u64;
            report.edge_index_bytes += (probe.edge_index_slots * size_of::<u32>()) as u64;
            report.cutoff_bytes += (probe.cutoffs_capacity * 8) as u64;
            report.compact_child_bytes += probe.compact_child_bytes as u64;
            report.sharded_child_bytes += probe.sharded_child_bytes as u64;
            report.coverage_overflow_bytes += probe.coverage_overflow_bytes as u64;
        };
        add(self.root.probe());
        self.probe_walk(|node, _| add(node.probe()));
        report.shared_graves = NodeChildren::shared_graves() as u64;
        report
    }

    /// Checks the tree and `lanes` after quiescence:
    /// - every lane holds one `Arc` per node its entries name (`assert_invariants`);
    /// - every reachable node's edge index matches its edge, every cutoff lies in
    ///   `(0, len)`, cutoffs are sorted by slot, and no slot is both full and cut off;
    /// - no holder-less childless leaf is reachable unless something holds it.
    ///
    /// Returns the number of held dead leaves left.
    #[cfg(test)]
    pub(super) fn probe_check(&self, lanes: &[&LaneLookup]) -> Result<u64, String> {
        for lane in lanes {
            lane.assert_invariants();
        }
        let mut errors = Vec::new();
        let mut held = 0;
        self.probe_walk(|node, anchor| {
            if let Err(error) = node.check_invariants() {
                errors.push(error);
            }
            if anchor {
                return;
            }
            let probe = node.probe();
            if !(probe.holder_less && probe.childless) {
                return;
            }
            match Node::live_strong_count_for_probe(node) {
                Some(count) if count > 2 => held += 1,
                count => errors.push(format!(
                    "unreclaimed dead leaf of {} blocks, live strong count {count:?}",
                    probe.edge_len
                )),
            }
        });
        if errors.is_empty() {
            Ok(held)
        } else {
            Err(errors.join("; "))
        }
    }
}
