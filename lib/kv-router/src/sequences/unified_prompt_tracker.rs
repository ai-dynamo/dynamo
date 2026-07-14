// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_tokens::SequenceHash;
use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};

use super::compressed_path_arena::{CompressedNodeId, CompressedPathArena};
use crate::protocols::WorkerWithDpRank;

#[derive(Debug, Clone, Default, PartialEq)]
struct UnifiedNodeMetadata {
    coverage: FxHashMap<WorkerWithDpRank, u32>,
    terminals: FxHashMap<WorkerWithDpRank, u32>,
    fractions: FxHashMap<WorkerWithDpRank, f64>,
}

#[derive(Debug, Default)]
struct RootShard {
    arena: RwLock<CompressedPathArena<UnifiedNodeMetadata>>,
}

#[derive(Debug, Default)]
pub(super) struct UnifiedRequestHandle {
    shard: Option<Arc<RootShard>>,
    tail: Option<CompressedNodeId>,
    prompt_depth: usize,
}

/// Authoritative request-lifecycle and cross-worker prompt-membership trie.
#[derive(Debug, Default)]
pub(super) struct UnifiedPromptTracker {
    roots: RwLock<FxHashMap<SequenceHash, Arc<RootShard>>>,
    worker_totals: RwLock<FxHashMap<WorkerWithDpRank, f64>>,
}

impl UnifiedPromptTracker {
    pub(super) fn ensure_worker(&self, worker: WorkerWithDpRank) {
        self.worker_totals.write().entry(worker).or_insert(0.0);
    }

    pub(super) fn acquire(
        &self,
        worker: WorkerWithDpRank,
        sequence: &[SequenceHash],
    ) -> UnifiedRequestHandle {
        let Some(&first_hash) = sequence.first() else {
            return UnifiedRequestHandle::default();
        };
        let shard = self.root_shard(first_hash);
        let mut arena = shard.arena.write();

        let tail = match arena.root(first_hash) {
            None => arena.insert_root(sequence.to_vec(), UnifiedNodeMetadata::default()),
            Some(root) => Self::acquire_terminal(&mut arena, root, sequence),
        };

        let path = arena.path_from_root(tail);
        let mut total_delta = 0.0;
        for node_id in path {
            let node = &mut arena.nodes[node_id];
            let coverage = node.metadata.coverage.entry(worker).or_default();
            if *coverage == 0 {
                total_delta += node.edge.len() as f64
                    * node.metadata.fractions.get(&worker).copied().unwrap_or(1.0);
            }
            *coverage = coverage
                .checked_add(1)
                .expect("unified prompt coverage overflowed");
        }
        let terminal = arena.nodes[tail]
            .metadata
            .terminals
            .entry(worker)
            .or_default();
        *terminal = terminal
            .checked_add(1)
            .expect("unified prompt terminal ownership overflowed");
        self.apply_total_delta(worker, total_delta);

        UnifiedRequestHandle {
            shard: Some(shard.clone()),
            tail: Some(tail),
            prompt_depth: sequence.len(),
        }
    }

    fn acquire_terminal(
        arena: &mut CompressedPathArena<UnifiedNodeMetadata>,
        mut node_id: CompressedNodeId,
        sequence: &[SequenceHash],
    ) -> CompressedNodeId {
        let mut path_pos = 0;
        loop {
            let (edge_len, match_len) = {
                let node = &arena.nodes[node_id];
                let remaining = &sequence[path_pos..];
                let matched = node
                    .edge
                    .iter()
                    .zip(remaining)
                    .take_while(|(left, right)| left == right)
                    .count();
                (node.edge.len(), matched)
            };
            assert!(match_len > 0, "unified path selected a mismatched edge");
            if match_len < edge_len {
                let split_depth = path_pos + match_len;
                let existing = &arena.nodes[node_id].metadata;
                let prefix_metadata = UnifiedNodeMetadata {
                    coverage: existing.coverage.clone(),
                    terminals: FxHashMap::default(),
                    fractions: existing.fractions.clone(),
                };
                let prefix = arena.split_keep_suffix(node_id, match_len, prefix_metadata);
                if split_depth == sequence.len() {
                    return prefix;
                }
                return arena.insert_child(
                    prefix,
                    sequence[split_depth..].to_vec(),
                    UnifiedNodeMetadata::default(),
                );
            }
            path_pos += edge_len;
            if path_pos == sequence.len() {
                return node_id;
            }
            let next_hash = sequence[path_pos];
            match arena.nodes[node_id].children.get(&next_hash).copied() {
                Some(next) => node_id = next,
                None => {
                    return arena.insert_child(
                        node_id,
                        sequence[path_pos..].to_vec(),
                        UnifiedNodeMetadata::default(),
                    );
                }
            }
        }
    }

    pub(super) fn release(&self, worker: WorkerWithDpRank, handle: UnifiedRequestHandle) {
        let Some(shard) = handle.shard else {
            assert!(handle.tail.is_none() && handle.prompt_depth == 0);
            return;
        };
        let tail = handle.tail.expect("non-empty unified handle has no tail");
        let mut arena = shard.arena.write();
        let path = arena.path_from_root(tail);
        let depth: usize = path.iter().map(|id| arena.nodes[*id].edge.len()).sum();
        assert_eq!(depth, handle.prompt_depth, "unified request depth mismatch");

        Self::decrement(
            &mut arena.nodes[tail].metadata.terminals,
            worker,
            "terminal",
        );
        let mut total_delta = 0.0;
        for node_id in path.iter().copied() {
            let node = &mut arena.nodes[node_id];
            let old_fraction = node.metadata.fractions.get(&worker).copied().unwrap_or(1.0);
            if Self::decrement(&mut node.metadata.coverage, worker, "coverage") {
                total_delta -= node.edge.len() as f64 * old_fraction;
                node.metadata.fractions.remove(&worker);
            }
        }

        let mut current = Some(tail);
        let mut retained = None;
        while let Some(node_id) = current {
            let node = &arena.nodes[node_id];
            if !node.metadata.coverage.is_empty() || !node.children.is_empty() {
                retained = Some(node_id);
                break;
            }
            assert!(node.metadata.terminals.is_empty());
            let parent = node.parent;
            arena.remove_leaf(node_id);
            current = parent;
        }
        if let Some(node_id) = retained {
            Self::recompress_from(&mut arena, node_id);
        }
        self.apply_total_delta(worker, total_delta);
    }

    /// Returns true when the entry became absent.
    fn decrement(
        entries: &mut FxHashMap<WorkerWithDpRank, u32>,
        worker: WorkerWithDpRank,
        kind: &str,
    ) -> bool {
        let count = entries
            .get_mut(&worker)
            .unwrap_or_else(|| panic!("unified prompt {kind} is missing"));
        *count = count
            .checked_sub(1)
            .unwrap_or_else(|| panic!("unified prompt {kind} underflowed"));
        if *count == 0 {
            entries.remove(&worker);
            true
        } else {
            false
        }
    }

    pub(super) fn set_unique_suffix_fractional(
        &self,
        worker: WorkerWithDpRank,
        handle: &UnifiedRequestHandle,
        fraction: f64,
    ) {
        let (Some(shard), Some(mut node_id)) = (&handle.shard, handle.tail) else {
            return;
        };
        let mut arena = shard.arena.write();
        let mut total_delta = 0.0;
        loop {
            let incoming = {
                let node = &arena.nodes[node_id];
                node.metadata.terminals.get(&worker).copied().unwrap_or(0) as usize
                    + node
                        .children
                        .values()
                        .filter(|child| {
                            arena.nodes[**child].metadata.coverage.contains_key(&worker)
                        })
                        .count()
            };
            if incoming != 1 {
                break;
            }
            let node = &mut arena.nodes[node_id];
            let old = node
                .metadata
                .fractions
                .insert(worker, fraction)
                .unwrap_or(1.0);
            total_delta += node.edge.len() as f64 * (fraction - old);
            match node.parent {
                Some(parent) => node_id = parent,
                None => break,
            }
        }
        self.apply_total_delta(worker, total_delta);
    }

    pub(super) fn compute_overlap_depths(
        &self,
        query: Option<&[SequenceHash]>,
    ) -> FxHashMap<WorkerWithDpRank, usize> {
        let Some(query) = query.filter(|query| !query.is_empty()) else {
            return FxHashMap::default();
        };
        let Some(shard) = self.roots.read().get(&query[0]).cloned() else {
            return FxHashMap::default();
        };
        let arena = shard.arena.read();
        let Some(mut node_id) = arena.root(query[0]) else {
            return FxHashMap::default();
        };

        let mut scores = FxHashMap::default();
        let mut active = FxHashSet::default();
        let mut depth = 0;
        let mut query_pos = 0;
        let mut first = true;
        loop {
            let node = &arena.nodes[node_id];
            if first {
                active.extend(node.metadata.coverage.keys().copied());
                first = false;
            } else {
                active.retain(|worker| {
                    if node.metadata.coverage.contains_key(worker) {
                        true
                    } else {
                        scores.insert(*worker, depth);
                        false
                    }
                });
            }
            if active.is_empty() {
                break;
            }
            let matched = node
                .edge
                .iter()
                .zip(&query[query_pos..])
                .take_while(|(left, right)| left == right)
                .count();
            depth += matched;
            query_pos += matched;
            if matched < node.edge.len() || query_pos == query.len() {
                for worker in active {
                    scores.insert(worker, depth);
                }
                break;
            }
            let Some(next) = node.children.get(&query[query_pos]).copied() else {
                for worker in active {
                    scores.insert(worker, depth);
                }
                break;
            };
            node_id = next;
        }
        scores
    }

    pub(super) fn active_block_weight(&self, worker: WorkerWithDpRank) -> f64 {
        self.worker_totals
            .read()
            .get(&worker)
            .copied()
            .unwrap_or(0.0)
    }

    #[cfg(test)]
    pub(super) fn active_blocks(&self, worker: WorkerWithDpRank) -> usize {
        self.active_block_weight(worker).round() as usize
    }

    #[cfg(test)]
    pub(super) fn prompt_hashes(&self, handle: &UnifiedRequestHandle) -> Vec<SequenceHash> {
        let (Some(shard), Some(tail)) = (&handle.shard, handle.tail) else {
            return Vec::new();
        };
        let arena = shard.arena.read();
        let mut hashes = Vec::with_capacity(handle.prompt_depth);
        for node_id in arena.path_from_root(tail) {
            hashes.extend_from_slice(&arena.nodes[node_id].edge);
        }
        assert_eq!(hashes.len(), handle.prompt_depth);
        hashes
    }

    pub(super) fn remove_worker(&self, worker: WorkerWithDpRank) {
        let shards: Vec<_> = self.roots.read().values().cloned().collect();
        for shard in shards {
            let mut arena = shard.arena.write();
            for node in arena.nodes.values_mut() {
                node.metadata.coverage.remove(&worker);
                node.metadata.terminals.remove(&worker);
                node.metadata.fractions.remove(&worker);
            }
            Self::compact_after_worker_removal(&mut arena);
        }
        self.worker_totals.write().remove(&worker);
    }

    fn compact_after_worker_removal(arena: &mut CompressedPathArena<UnifiedNodeMetadata>) {
        loop {
            let dead = arena.nodes.iter().find_map(|(node_id, node)| {
                (node.metadata.coverage.is_empty() && node.children.is_empty()).then_some(node_id)
            });
            let Some(dead) = dead else { break };
            assert!(arena.nodes[dead].metadata.terminals.is_empty());
            arena.remove_leaf(dead);
        }
        loop {
            let merge = arena.nodes.iter().find_map(|(parent_id, parent)| {
                if parent.metadata.terminals.is_empty() && parent.children.len() == 1 {
                    let child_id = *parent.children.values().next().unwrap();
                    let child = &arena.nodes[child_id];
                    (parent.metadata.coverage == child.metadata.coverage
                        && parent.metadata.fractions == child.metadata.fractions)
                        .then_some((parent_id, child_id))
                } else {
                    None
                }
            });
            let Some((parent, child)) = merge else {
                break;
            };
            arena.merge_parent_into_child(parent, child);
        }
    }

    fn recompress_from(
        arena: &mut CompressedPathArena<UnifiedNodeMetadata>,
        mut parent_id: CompressedNodeId,
    ) {
        loop {
            let child_id = {
                let parent = &arena.nodes[parent_id];
                if !parent.metadata.terminals.is_empty() || parent.children.len() != 1 {
                    return;
                }
                *parent.children.values().next().unwrap()
            };
            if arena.nodes[parent_id].metadata.coverage != arena.nodes[child_id].metadata.coverage
                || arena.nodes[parent_id].metadata.fractions
                    != arena.nodes[child_id].metadata.fractions
            {
                return;
            }
            arena.merge_parent_into_child(parent_id, child_id);
            parent_id = child_id;
        }
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn is_empty(&self) -> bool {
        self.roots
            .read()
            .values()
            .all(|shard| shard.arena.read().nodes.is_empty())
            && self
                .worker_totals
                .read()
                .values()
                .all(|total| *total == 0.0)
    }

    #[cfg(test)]
    pub(super) fn worker_hashes(&self, worker: WorkerWithDpRank) -> FxHashSet<SequenceHash> {
        let mut hashes = FxHashSet::default();
        for shard in self.roots.read().values() {
            let arena = shard.arena.read();
            for node in arena.nodes.values() {
                if node.metadata.coverage.contains_key(&worker) {
                    hashes.extend(node.edge.iter().copied());
                }
            }
        }
        hashes
    }

    fn root_shard(&self, first_hash: SequenceHash) -> Arc<RootShard> {
        if let Some(shard) = self.roots.read().get(&first_hash).cloned() {
            return shard;
        }
        self.roots.write().entry(first_hash).or_default().clone()
    }

    fn apply_total_delta(&self, worker: WorkerWithDpRank, delta: f64) {
        if delta == 0.0 {
            return;
        }
        let mut totals = self.worker_totals.write();
        let total = totals.entry(worker).or_insert(0.0);
        *total += delta;
        assert!(
            *total >= -f64::EPSILON,
            "unified prompt total became negative"
        );
        if total.abs() < f64::EPSILON {
            *total = 0.0;
        }
    }

    #[cfg(any(test, debug_assertions))]
    pub(super) fn assert_consistent(&self) {
        let mut recomputed = FxHashMap::<WorkerWithDpRank, f64>::default();
        for shard in self.roots.read().values() {
            let arena = shard.arena.read();
            arena.assert_topology();
            for node in arena.nodes.values() {
                assert!(!node.metadata.coverage.is_empty() || !node.children.is_empty());
                for (&worker, &count) in &node.metadata.coverage {
                    assert!(count > 0);
                    *recomputed.entry(worker).or_default() += node.edge.len() as f64
                        * node.metadata.fractions.get(&worker).copied().unwrap_or(1.0);
                }
                for (&worker, &terminals) in &node.metadata.terminals {
                    assert!(terminals > 0);
                    assert!(node.metadata.coverage.contains_key(&worker));
                }
            }
        }
        for (&worker, &total) in self.worker_totals.read().iter() {
            assert!((recomputed.get(&worker).copied().unwrap_or(0.0) - total).abs() < 1e-9);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::thread;

    use super::*;

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    #[test]
    fn duplicate_and_shared_prefix_lifecycles_are_exact() {
        let tracker = UnifiedPromptTracker::default();
        let w1 = worker(1);
        let w2 = worker(2);
        let one = tracker.acquire(w1, &[1, 2, 3]);
        let duplicate = tracker.acquire(w1, &[1, 2, 3]);
        let other = tracker.acquire(w2, &[1, 2, 4]);

        assert_eq!(tracker.active_blocks(w1), 3);
        assert_eq!(tracker.active_blocks(w2), 3);
        assert_eq!(
            tracker.compute_overlap_depths(Some(&[1, 2, 3, 9])),
            FxHashMap::from_iter([(w1, 3), (w2, 2)])
        );
        tracker.release(w1, one);
        assert_eq!(tracker.active_blocks(w1), 3);
        tracker.release(w1, duplicate);
        assert_eq!(tracker.active_blocks(w1), 0);
        tracker.release(w2, other);
        assert!(tracker.is_empty());
    }

    #[test]
    fn shorter_prompt_split_keeps_longer_handle_valid() {
        let tracker = UnifiedPromptTracker::default();
        let w = worker(1);
        let longer = tracker.acquire(w, &[1, 2, 3, 4]);
        let longer_tail = longer.tail;
        let shorter = tracker.acquire(w, &[1, 2]);
        assert_eq!(longer.tail, longer_tail);
        tracker.release(w, shorter);
        tracker.release(w, longer);
        assert!(tracker.is_empty());
    }

    #[test]
    fn branch_release_recompression_keeps_survivor_handle_valid() {
        let tracker = UnifiedPromptTracker::default();
        let w = worker(1);
        let survivor = tracker.acquire(w, &[1, 2, 3, 4]);
        let survivor_tail = survivor.tail;
        let branch = tracker.acquire(w, &[1, 2, 8, 9]);

        tracker.release(w, branch);
        assert_eq!(survivor.tail, survivor_tail);
        assert_eq!(tracker.prompt_hashes(&survivor), vec![1, 2, 3, 4]);
        tracker.release(w, survivor);
        assert!(tracker.is_empty());
    }

    #[test]
    fn worker_removal_preserves_other_workers() {
        let tracker = UnifiedPromptTracker::default();
        let w1 = worker(1);
        let w2 = worker(2);
        let _one = tracker.acquire(w1, &[1, 2, 3]);
        let two = tracker.acquire(w2, &[1, 2, 4]);
        tracker.remove_worker(w1);

        assert_eq!(tracker.worker_hashes(w1), FxHashSet::default());
        assert_eq!(tracker.worker_hashes(w2), FxHashSet::from_iter([1, 2, 4]));
        assert_eq!(
            tracker.compute_overlap_depths(Some(&[1, 2, 3])),
            FxHashMap::from_iter([(w2, 2)])
        );
        tracker.release(w2, two);
        assert!(tracker.is_empty());
    }

    #[test]
    fn concurrent_same_root_mutations_drain_exactly() {
        const THREADS: usize = 8;
        const ITERATIONS: usize = 200;
        let tracker = Arc::new(UnifiedPromptTracker::default());
        let start = Arc::new(Barrier::new(THREADS));
        let threads = (0..THREADS)
            .map(|thread_id| {
                let tracker = tracker.clone();
                let start = start.clone();
                thread::spawn(move || {
                    let worker = worker(thread_id as u64);
                    start.wait();
                    for iteration in 0..ITERATIONS {
                        let suffix = 10 + ((thread_id + iteration) % 4) as u64;
                        let handle = tracker.acquire(worker, &[1, 2, suffix, 99]);
                        assert_eq!(tracker.prompt_hashes(&handle), vec![1, 2, suffix, 99]);
                        tracker.release(worker, handle);
                    }
                })
            })
            .collect::<Vec<_>>();

        for thread in threads {
            thread.join().unwrap();
        }
        tracker.assert_consistent();
        assert!(tracker.is_empty());
    }

    #[test]
    fn concurrent_independent_roots_retain_empty_shards() {
        const THREADS: usize = 8;
        let tracker = Arc::new(UnifiedPromptTracker::default());
        let start = Arc::new(Barrier::new(THREADS));
        let threads = (0..THREADS)
            .map(|thread_id| {
                let tracker = tracker.clone();
                let start = start.clone();
                thread::spawn(move || {
                    let worker = worker(thread_id as u64);
                    let root = 1_000 + thread_id as u64;
                    start.wait();
                    let handle = tracker.acquire(worker, &[root, root + 100, root + 200]);
                    tracker.release(worker, handle);
                })
            })
            .collect::<Vec<_>>();

        for thread in threads {
            thread.join().unwrap();
        }
        let roots = tracker.roots.read();
        assert_eq!(roots.len(), THREADS);
        assert!(
            roots
                .values()
                .all(|shard| shard.arena.read().nodes.is_empty())
        );
        drop(roots);
        assert!(tracker.is_empty());
    }
}
