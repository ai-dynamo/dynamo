// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::Arc;

use dynamo_tokens::SequenceHash;
use parking_lot::RwLock;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::cleanup::{self, CleanableNode, CleanupGuard, CleanupState};
use crate::protocols::WorkerWithDpRank;

type SharedNode = Arc<RwLock<PromptTrieNode>>;
pub(super) type WorkerLookup = FxHashMap<SequenceHash, SharedNode>;

#[derive(Debug)]
pub(super) struct PromptTrieNode {
    edge: Vec<SequenceHash>,
    edge_index: FxHashMap<SequenceHash, usize>,
    worker_cutoffs: FxHashMap<WorkerWithDpRank, usize>,
    full_edge_workers: FxHashSet<WorkerWithDpRank>,
    children: FxHashMap<SequenceHash, SharedNode>,
}

impl PromptTrieNode {
    fn new() -> Self {
        Self {
            edge: Vec::new(),
            edge_index: FxHashMap::default(),
            worker_cutoffs: FxHashMap::default(),
            full_edge_workers: FxHashSet::default(),
            children: FxHashMap::default(),
        }
    }

    fn edge_index_for(edge: &[SequenceHash]) -> FxHashMap<SequenceHash, usize> {
        let mut edge_index = FxHashMap::with_capacity_and_hasher(edge.len(), FxBuildHasher);
        for (i, &hash) in edge.iter().enumerate() {
            edge_index.insert(hash, i);
        }
        edge_index
    }

    fn full_edge_workers_for(worker: WorkerWithDpRank) -> FxHashSet<WorkerWithDpRank> {
        let mut full_edge_workers = FxHashSet::with_capacity_and_hasher(1, FxBuildHasher);
        full_edge_workers.insert(worker);
        full_edge_workers
    }

    fn current_cutoff(&self, worker: WorkerWithDpRank) -> usize {
        if self.full_edge_workers.contains(&worker) {
            self.edge.len()
        } else {
            self.worker_cutoffs.get(&worker).copied().unwrap_or(0)
        }
    }

    fn covers_pos(&self, worker: WorkerWithDpRank, pos: usize) -> bool {
        self.full_edge_workers.contains(&worker)
            || matches!(self.worker_cutoffs.get(&worker), Some(&cutoff) if pos < cutoff)
    }

    fn clear_children_if_unreachable(&mut self) {
        if self.full_edge_workers.is_empty() {
            self.children.clear();
        }
    }

    fn uncovered_suffix_hashes(&self, cutoff: usize) -> Vec<SequenceHash> {
        debug_assert!(cutoff <= self.edge.len());
        self.edge[cutoff..].to_vec()
    }

    fn lookup_hashes_for_worker_repair(
        &self,
        worker: WorkerWithDpRank,
        hash: SequenceHash,
        direction: LookupRepairDirection,
    ) -> Vec<SequenceHash> {
        let Some(&pos) = self.edge_index.get(&hash) else {
            return Vec::new();
        };

        let cutoff = self.current_cutoff(worker).min(self.edge.len());
        let range = match direction {
            LookupRepairDirection::TowardTail => {
                if pos < cutoff {
                    pos..cutoff
                } else {
                    0..0
                }
            }
            LookupRepairDirection::TowardHead => 0..pos.min(cutoff),
        };

        self.edge[range].to_vec()
    }

    fn drop_worker(&mut self, worker: WorkerWithDpRank) {
        self.full_edge_workers.remove(&worker);
        self.worker_cutoffs.remove(&worker);
        self.clear_children_if_unreachable();
    }

    fn promote_to_full(&mut self, worker: WorkerWithDpRank) {
        if !self.full_edge_workers.contains(&worker) {
            self.worker_cutoffs.remove(&worker);
            self.full_edge_workers.insert(worker);
        }
    }

    fn remove_worker_at_pos(
        &mut self,
        worker: WorkerWithDpRank,
        pos: usize,
        removed_hash: SequenceHash,
    ) -> RemoveOutcome {
        let current_cutoff = self.current_cutoff(worker);
        if pos >= current_cutoff {
            return RemoveOutcome {
                stale_hashes: vec![removed_hash],
            };
        }

        let new_cutoff = pos;
        let stale_hashes = self.uncovered_suffix_hashes(new_cutoff);

        if new_cutoff == 0 {
            self.drop_worker(worker);
        } else {
            self.full_edge_workers.remove(&worker);
            self.worker_cutoffs.insert(worker, new_cutoff);
            self.clear_children_if_unreachable();
        }

        RemoveOutcome { stale_hashes }
    }

    #[cfg(any(test, feature = "bench"))]
    fn live_children(&self) -> Vec<SharedNode> {
        self.children
            .values()
            .filter(|child| {
                let guard = child.read();
                guard.has_any_workers() || !guard.children.is_empty()
            })
            .cloned()
            .collect()
    }
}

impl CleanableNode for PromptTrieNode {
    type ChildKey = SequenceHash;

    fn has_any_workers(&self) -> bool {
        !self.full_edge_workers.is_empty() || !self.worker_cutoffs.is_empty()
    }

    fn children(&self) -> &FxHashMap<SequenceHash, SharedNode> {
        &self.children
    }

    fn remove_child(&mut self, key: &SequenceHash) {
        self.children.remove(key);
    }
}

struct RemoveOutcome {
    stale_hashes: Vec<SequenceHash>,
}

#[derive(Clone, Copy)]
enum LookupRepairDirection {
    TowardTail,
    TowardHead,
}

pub(super) struct PromptMembershipTrie {
    root: SharedNode,
    cleanup: CleanupState,
}

impl Default for PromptMembershipTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PromptMembershipTrie {
    fn drop(&mut self) {
        let mut stack: Vec<SharedNode> = Vec::new();
        {
            let mut root = self.root.write();
            stack.extend(root.children.drain().map(|(_, child)| child));
        }

        while let Some(node) = stack.pop() {
            if let Ok(rwlock) = Arc::try_unwrap(node) {
                let mut inner = rwlock.into_inner();
                stack.extend(inner.children.drain().map(|(_, child)| child));
            }
        }
    }
}

impl PromptMembershipTrie {
    pub(super) fn new() -> Self {
        Self {
            root: Arc::new(RwLock::new(PromptTrieNode::new())),
            cleanup: CleanupState::new(),
        }
    }

    /// Run the stale-child sweep if the throttle interval has elapsed.
    ///
    /// Safe to call from any write path; the sweep is a no-op until
    /// [`CLEANUP_INTERVAL_MS`](crate::cleanup::CLEANUP_INTERVAL_MS) has passed
    /// since the last completion, and only one sweep runs at a time.
    pub(super) fn maybe_cleanup(&self) {
        if !self.cleanup.try_schedule() {
            return;
        }
        let mut guard = CleanupGuard::new(&self.cleanup);
        cleanup::sweep_stale_children(&self.root);
        guard.mark_completed();
    }

    fn children_snapshot(node: &SharedNode) -> Vec<SharedNode> {
        let guard = node.read();
        guard.children.values().cloned().collect()
    }

    fn find_in_subtree(start: &SharedNode, hash: SequenceHash) -> Option<SharedNode> {
        let mut queue = VecDeque::from(Self::children_snapshot(start));
        while let Some(node) = queue.pop_front() {
            let guard = node.read();
            if guard.edge_index.contains_key(&hash) {
                drop(guard);
                return Some(node);
            }
            queue.extend(guard.children.values().cloned());
        }

        None
    }

    fn repair_lookup_for_resolved_node(
        worker_lookup: &mut WorkerLookup,
        worker: WorkerWithDpRank,
        hash: SequenceHash,
        resolved: &SharedNode,
        direction: LookupRepairDirection,
    ) {
        let repair_hashes = {
            let guard = resolved.read();
            guard.lookup_hashes_for_worker_repair(worker, hash, direction)
        };
        for repair_hash in repair_hashes {
            worker_lookup.insert(repair_hash, resolved.clone());
        }
    }

    fn repair_stale_lookup_from_node(
        worker_lookup: &mut WorkerLookup,
        worker: WorkerWithDpRank,
        node: &SharedNode,
        hash: SequenceHash,
        direction: LookupRepairDirection,
    ) -> Option<SharedNode> {
        let resolved = Self::find_in_subtree(node, hash)?;
        Self::repair_lookup_for_resolved_node(worker_lookup, worker, hash, &resolved, direction);
        Some(resolved)
    }

    fn resolve_lookup(
        worker_lookup: &mut WorkerLookup,
        worker: WorkerWithDpRank,
        hash: SequenceHash,
        direction: LookupRepairDirection,
    ) -> Option<SharedNode> {
        let node = worker_lookup.get(&hash)?.clone();
        let found = {
            let guard = node.read();
            guard.edge_index.contains_key(&hash)
        };
        if found {
            return Some(node);
        }

        Self::repair_stale_lookup_from_node(worker_lookup, worker, &node, hash, direction)
    }

    fn split_node(node: &mut PromptTrieNode, pos: usize) -> SharedNode {
        debug_assert!(pos > 0 && pos < node.edge.len());

        let suffix_edge = node.edge.split_off(pos);
        let suffix_first_hash = suffix_edge[0];

        let suffix_edge_index = PromptTrieNode::edge_index_for(&suffix_edge);
        for &hash in &suffix_edge {
            node.edge_index.remove(&hash);
        }

        let mut suffix_full =
            FxHashSet::with_capacity_and_hasher(node.full_edge_workers.len(), FxBuildHasher);
        let mut suffix_cutoffs =
            FxHashMap::with_capacity_and_hasher(node.worker_cutoffs.len(), FxBuildHasher);
        let mut to_promote = Vec::with_capacity(node.worker_cutoffs.len());

        for &worker in &node.full_edge_workers {
            suffix_full.insert(worker);
        }

        for (&worker, &cutoff) in &node.worker_cutoffs {
            if cutoff >= pos {
                to_promote.push(worker);
                let suffix_cutoff = cutoff - pos;
                if suffix_cutoff > 0 {
                    suffix_cutoffs.insert(worker, suffix_cutoff);
                }
            }
        }

        for worker in to_promote {
            node.worker_cutoffs.remove(&worker);
            node.full_edge_workers.insert(worker);
        }

        let suffix_children = std::mem::take(&mut node.children);
        let suffix = Arc::new(RwLock::new(PromptTrieNode {
            edge: suffix_edge,
            edge_index: suffix_edge_index,
            worker_cutoffs: suffix_cutoffs,
            full_edge_workers: suffix_full,
            children: suffix_children,
        }));
        node.children.insert(suffix_first_hash, suffix.clone());
        suffix
    }

    pub(super) fn store_chain(
        &self,
        worker: WorkerWithDpRank,
        lookup: &Arc<RwLock<WorkerLookup>>,
        parent: Option<SequenceHash>,
        hashes: &[SequenceHash],
    ) {
        if hashes.is_empty() {
            return;
        }

        let mut worker_lookup = lookup.write();
        let parent = match parent {
            Some(parent_hash) => loop {
                let Some(node) = Self::resolve_lookup(
                    &mut worker_lookup,
                    worker,
                    parent_hash,
                    LookupRepairDirection::TowardTail,
                ) else {
                    tracing::warn!(?worker, ?parent_hash, "prompt parent hash not found");
                    return;
                };

                {
                    let guard = node.read();
                    let Some(&pos) = guard.edge_index.get(&parent_hash) else {
                        continue;
                    };
                    if !guard.covers_pos(worker, pos) {
                        worker_lookup.remove(&parent_hash);
                        tracing::warn!(
                            ?worker,
                            ?parent_hash,
                            pos,
                            "worker no longer covers prompt parent"
                        );
                        return;
                    }
                }

                let split_suffix = {
                    let mut guard = node.write();
                    if !guard.edge_index.contains_key(&parent_hash) {
                        continue;
                    }
                    if !guard.edge.is_empty() && *guard.edge.last().unwrap() != parent_hash {
                        let split_pos = guard
                            .edge
                            .iter()
                            .position(|hash| *hash == parent_hash)
                            .expect("parent hash presence was checked above");
                        Some(Self::split_node(&mut guard, split_pos + 1))
                    } else {
                        None
                    }
                };

                if split_suffix.is_some() {
                    continue;
                }

                break node;
            },
            None => self.root.clone(),
        };

        self.insert_hashes_from(worker, &mut worker_lookup, &parent, hashes);
    }

    fn insert_hashes_from(
        &self,
        worker: WorkerWithDpRank,
        worker_lookup: &mut WorkerLookup,
        parent: &SharedNode,
        hashes: &[SequenceHash],
    ) {
        let mut current_parent = parent.clone();
        let mut remaining = hashes;
        let mut last_hash = None;

        while !remaining.is_empty() {
            let first_hash = remaining[0];

            let child = {
                let mut parent_guard = current_parent.write();
                if let Some(last_hash) = last_hash
                    && !parent_guard.edge_index.contains_key(&last_hash)
                {
                    drop(parent_guard);
                    if let Some(resolved) = Self::resolve_lookup(
                        worker_lookup,
                        worker,
                        last_hash,
                        LookupRepairDirection::TowardTail,
                    ) {
                        current_parent = resolved;
                    }
                    continue;
                }

                match parent_guard.children.get(&first_hash).cloned() {
                    Some(existing) => existing,
                    None => {
                        let edge = remaining.to_vec();
                        let edge_index = PromptTrieNode::edge_index_for(&edge);
                        let full_edge_workers = PromptTrieNode::full_edge_workers_for(worker);

                        let new_node = Arc::new(RwLock::new(PromptTrieNode {
                            edge,
                            edge_index,
                            worker_cutoffs: FxHashMap::default(),
                            full_edge_workers,
                            children: FxHashMap::default(),
                        }));
                        parent_guard.children.insert(first_hash, new_node.clone());
                        drop(parent_guard);

                        for &hash in remaining {
                            worker_lookup.insert(hash, new_node.clone());
                        }
                        return;
                    }
                }
            };

            {
                let mut child_guard = child.write();
                let edge_len = child_guard.edge.len();

                let mut match_len = 0;
                for (&edge_hash, &query_hash) in child_guard.edge.iter().zip(remaining.iter()) {
                    if edge_hash != query_hash {
                        break;
                    }
                    match_len += 1;
                }

                debug_assert!(match_len >= 1);

                if match_len < edge_len {
                    let _suffix = Self::split_node(&mut child_guard, match_len);
                    child_guard.promote_to_full(worker);

                    let tail = &remaining[match_len..];
                    if !tail.is_empty() {
                        let edge = tail.to_vec();
                        let edge_index = PromptTrieNode::edge_index_for(&edge);
                        let full_edge_workers = PromptTrieNode::full_edge_workers_for(worker);
                        let tail_first_hash = tail[0];

                        let new_node = Arc::new(RwLock::new(PromptTrieNode {
                            edge,
                            edge_index,
                            worker_cutoffs: FxHashMap::default(),
                            full_edge_workers,
                            children: FxHashMap::default(),
                        }));
                        child_guard
                            .children
                            .insert(tail_first_hash, new_node.clone());
                        drop(child_guard);

                        for &hash in &remaining[..match_len] {
                            worker_lookup.insert(hash, child.clone());
                        }
                        for &hash in tail {
                            worker_lookup.insert(hash, new_node.clone());
                        }
                    } else {
                        drop(child_guard);
                        for &hash in &remaining[..match_len] {
                            worker_lookup.insert(hash, child.clone());
                        }
                    }
                    return;
                }

                child_guard.promote_to_full(worker);
                drop(child_guard);

                for &hash in &remaining[..edge_len] {
                    worker_lookup.insert(hash, child.clone());
                }

                last_hash = Some(remaining[edge_len - 1]);
                remaining = &remaining[edge_len..];
                current_parent = child;
            }
        }
    }

    pub(super) fn remove_chain(
        &self,
        worker: WorkerWithDpRank,
        lookup: &Arc<RwLock<WorkerLookup>>,
        hashes: &[SequenceHash],
    ) {
        let mut worker_lookup = lookup.write();
        if worker_lookup.is_empty() {
            return;
        }

        'outer: for &hash in hashes {
            let mut current_node = match Self::resolve_lookup(
                &mut worker_lookup,
                worker,
                hash,
                LookupRepairDirection::TowardHead,
            ) {
                Some(node) => node,
                None => continue,
            };

            loop {
                let update = {
                    let mut guard = current_node.write();
                    guard
                        .edge_index
                        .get(&hash)
                        .copied()
                        .map(|pos| guard.remove_worker_at_pos(worker, pos, hash))
                };

                match update {
                    Some(outcome) => {
                        for stale_hash in outcome.stale_hashes {
                            worker_lookup.remove(&stale_hash);
                        }
                        continue 'outer;
                    }
                    None => match Self::repair_stale_lookup_from_node(
                        &mut worker_lookup,
                        worker,
                        &current_node,
                        hash,
                        LookupRepairDirection::TowardHead,
                    ) {
                        Some(resolved) => {
                            current_node = resolved;
                        }
                        None => {
                            worker_lookup.remove(&hash);
                            continue 'outer;
                        }
                    },
                }
            }
        }
    }

    pub(super) fn remove_worker(
        &self,
        worker: WorkerWithDpRank,
        lookup: &Arc<RwLock<WorkerLookup>>,
    ) {
        let mut worker_lookup = lookup.write();
        if worker_lookup.is_empty() {
            return;
        }

        let hashes: Vec<_> = worker_lookup.keys().copied().collect();
        let mut nodes = Vec::new();
        let mut seen = FxHashSet::<usize>::with_capacity_and_hasher(hashes.len(), FxBuildHasher);
        for hash in hashes {
            let Some(node) = Self::resolve_lookup(
                &mut worker_lookup,
                worker,
                hash,
                LookupRepairDirection::TowardTail,
            ) else {
                worker_lookup.remove(&hash);
                continue;
            };
            let ptr = Arc::as_ptr(&node) as usize;
            if seen.insert(ptr) {
                nodes.push(node);
            }
        }
        worker_lookup.clear();
        drop(worker_lookup);

        for node in nodes {
            let mut guard = node.write();
            guard.drop_worker(worker);
        }
    }

    pub(super) fn compute_overlap_depths(
        &self,
        query: Option<&[SequenceHash]>,
    ) -> FxHashMap<WorkerWithDpRank, usize> {
        let Some(query) = query else {
            return FxHashMap::default();
        };
        if query.is_empty() {
            return FxHashMap::default();
        }

        let mut matched_depth = FxHashMap::default();
        let mut active = FxHashSet::default();
        let mut active_count = 0usize;
        let mut query_pos = 0usize;
        let mut depth = 0usize;
        let mut first_node = true;

        let mut next_child = {
            let root = self.root.read();
            root.children.get(&query[0]).cloned()
        };

        loop {
            if query_pos >= query.len() {
                break;
            }

            let Some(child) = next_child.take() else {
                break;
            };

            let edge_len;
            let edge_match_len;
            {
                let guard = child.read();
                edge_len = guard.edge.len();
                let walk_len = edge_len.min(query.len() - query_pos);

                let mut match_len = 1usize;
                for i in 1..walk_len {
                    if guard.edge[i] != query[query_pos + i] {
                        break;
                    }
                    match_len += 1;
                }
                edge_match_len = match_len;

                let prev_depth = depth;
                if first_node {
                    active = guard.full_edge_workers.clone();
                    active_count = active.len();
                    for (&worker, &cutoff) in &guard.worker_cutoffs {
                        let contribution = cutoff.min(edge_match_len);
                        if contribution > 0 {
                            matched_depth.insert(worker, contribution);
                        }
                    }
                    first_node = false;
                } else if !guard.worker_cutoffs.is_empty() {
                    active.retain(|worker| {
                        if guard.full_edge_workers.contains(worker) {
                            true
                        } else if let Some(&cutoff) = guard.worker_cutoffs.get(worker) {
                            matched_depth.insert(*worker, prev_depth + cutoff.min(edge_match_len));
                            false
                        } else {
                            matched_depth.insert(*worker, prev_depth);
                            false
                        }
                    });
                    active_count = active.len();
                } else {
                    let full_count = guard.full_edge_workers.len();
                    if full_count != active_count {
                        active.retain(|worker| {
                            if guard.full_edge_workers.contains(worker) {
                                true
                            } else {
                                matched_depth.insert(*worker, prev_depth);
                                false
                            }
                        });
                        active_count = active.len();
                    }
                }

                next_child = if edge_match_len == edge_len
                    && active_count > 0
                    && query_pos + edge_match_len < query.len()
                {
                    guard
                        .children
                        .get(&query[query_pos + edge_match_len])
                        .cloned()
                } else {
                    None
                };
            }

            if active_count == 0 {
                break;
            }

            depth += edge_match_len;
            if edge_match_len < edge_len {
                break;
            }
            query_pos += edge_match_len;
        }

        for worker in active {
            matched_depth.insert(worker, depth);
        }

        matched_depth
    }

    #[cfg(test)]
    pub(super) fn worker_hashes(&self) -> FxHashMap<WorkerWithDpRank, FxHashSet<SequenceHash>> {
        let mut worker_hashes = FxHashMap::default();
        let mut stack = vec![self.root.clone()];

        while let Some(node) = stack.pop() {
            let guard = node.read();
            for &worker in &guard.full_edge_workers {
                worker_hashes
                    .entry(worker)
                    .or_insert_with(FxHashSet::default)
                    .extend(guard.edge.iter().copied());
            }
            for (&worker, &cutoff) in &guard.worker_cutoffs {
                worker_hashes
                    .entry(worker)
                    .or_insert_with(FxHashSet::default)
                    .extend(guard.edge[..cutoff].iter().copied());
            }
            stack.extend(guard.children.values().cloned());
        }

        worker_hashes
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn is_empty(&self) -> bool {
        let root = self.root.read();
        root.live_children().is_empty()
    }
}

#[cfg(any(test, feature = "bench"))]
pub(super) fn lookup_live_hashes(lookup: &Arc<RwLock<WorkerLookup>>) -> Vec<SequenceHash> {
    let worker_lookup = lookup.read();
    worker_lookup
        .iter()
        .filter_map(|(&hash, node)| node.read().has_any_workers().then_some(hash))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(worker_id: u64, dp_rank: u32) -> WorkerWithDpRank {
        WorkerWithDpRank::new(worker_id, dp_rank)
    }

    fn lookup() -> Arc<RwLock<WorkerLookup>> {
        Arc::new(RwLock::new(WorkerLookup::default()))
    }

    fn lookup_node(lookup: &Arc<RwLock<WorkerLookup>>, hash: SequenceHash) -> SharedNode {
        lookup
            .read()
            .get(&hash)
            .cloned()
            .expect("hash should exist in worker lookup")
    }

    fn node_contains_hash(node: &SharedNode, hash: SequenceHash) -> bool {
        node.read().edge_index.contains_key(&hash)
    }

    #[test]
    fn parent_continuation_chains_extend_and_trim() {
        let trie = PromptMembershipTrie::new();
        let worker = worker(1, 0);
        let lookup = lookup();

        trie.store_chain(worker, &lookup, None, &[1, 2, 3]);
        trie.store_chain(worker, &lookup, Some(3), &[4, 5]);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4, 5])),
            FxHashMap::from_iter([(worker, 5)]),
        );

        trie.remove_chain(worker, &lookup, &[4, 5]);
        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4, 5])),
            FxHashMap::from_iter([(worker, 3)]),
        );
    }

    #[test]
    fn branching_continuations_across_workers_match_expected_depths() {
        let trie = PromptMembershipTrie::new();
        let worker_a = worker(1, 0);
        let worker_b = worker(2, 0);
        let lookup_a = lookup();
        let lookup_b = lookup();

        trie.store_chain(worker_a, &lookup_a, None, &[1, 2, 3, 4]);
        trie.store_chain(worker_b, &lookup_b, None, &[1, 2, 5]);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4])),
            FxHashMap::from_iter([(worker_a, 4), (worker_b, 2)]),
        );
        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 5])),
            FxHashMap::from_iter([(worker_a, 2), (worker_b, 3)]),
        );
    }

    #[test]
    fn partial_suffix_removal_keeps_prefix() {
        let trie = PromptMembershipTrie::new();
        let worker = worker(1, 0);
        let lookup = lookup();

        trie.store_chain(worker, &lookup, None, &[1, 2, 3, 4, 5]);
        trie.remove_chain(worker, &lookup, &[3, 4, 5]);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4, 5])),
            FxHashMap::from_iter([(worker, 2)]),
        );
    }

    #[test]
    fn remove_worker_preserves_other_workers() {
        let trie = PromptMembershipTrie::new();
        let worker_a = worker(1, 0);
        let worker_b = worker(2, 0);
        let lookup_a = lookup();
        let lookup_b = lookup();

        trie.store_chain(worker_a, &lookup_a, None, &[1, 2, 3]);
        trie.store_chain(worker_b, &lookup_b, None, &[1, 2, 3]);

        trie.remove_worker(worker_a, &lookup_a);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3])),
            FxHashMap::from_iter([(worker_b, 3)]),
        );
    }

    #[test]
    fn multiple_dp_ranks_with_same_worker_id_remain_isolated() {
        let trie = PromptMembershipTrie::new();
        let worker_a = worker(1, 0);
        let worker_b = worker(1, 1);
        let lookup_a = lookup();
        let lookup_b = lookup();

        trie.store_chain(worker_a, &lookup_a, None, &[1, 2, 3]);
        trie.store_chain(worker_b, &lookup_b, None, &[1, 2]);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3])),
            FxHashMap::from_iter([(worker_a, 3), (worker_b, 2)]),
        );
    }

    #[test]
    fn clear_worker_state_then_reuse_starts_empty() {
        let trie = PromptMembershipTrie::new();
        let worker = worker(1, 0);
        let lookup = lookup();

        trie.store_chain(worker, &lookup, None, &[1, 2, 3]);
        trie.remove_worker(worker, &lookup);
        assert!(trie.compute_overlap_depths(Some(&[1, 2, 3])).is_empty());

        trie.store_chain(worker, &lookup, None, &[1, 2]);
        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3])),
            FxHashMap::from_iter([(worker, 2)]),
        );
    }

    #[test]
    fn redundant_batched_remove_is_idempotent() {
        let trie = PromptMembershipTrie::new();
        let worker = worker(1, 0);
        let lookup = lookup();

        trie.store_chain(worker, &lookup, None, &[1, 2, 3, 4]);
        trie.remove_chain(worker, &lookup, &[2, 3, 4]);
        trie.remove_chain(worker, &lookup, &[2, 3, 4]);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4])),
            FxHashMap::from_iter([(worker, 1)]),
        );
    }

    #[test]
    fn stale_lookup_repair_batches_toward_tail() {
        let trie = PromptMembershipTrie::new();
        let worker_a = worker(1, 0);
        let worker_b = worker(2, 0);
        let lookup_a = lookup();
        let lookup_b = lookup();

        trie.store_chain(worker_a, &lookup_a, None, &[1, 2, 3, 4]);
        trie.store_chain(worker_b, &lookup_b, None, &[1, 2, 5]);

        let stale_node = lookup_node(&lookup_a, 3);
        assert!(!node_contains_hash(&stale_node, 3));
        assert!(!node_contains_hash(&stale_node, 4));

        let mut worker_lookup = lookup_a.write();
        let resolved = PromptMembershipTrie::resolve_lookup(
            &mut worker_lookup,
            worker_a,
            3,
            LookupRepairDirection::TowardTail,
        )
        .expect("stale suffix should resolve");

        assert!(Arc::ptr_eq(
            worker_lookup.get(&3).expect("hash 3 should be repaired"),
            &resolved,
        ));
        assert!(Arc::ptr_eq(
            worker_lookup
                .get(&4)
                .expect("hash 4 should be batch repaired"),
            &resolved,
        ));
    }

    #[test]
    fn stale_lookup_repair_batches_toward_head() {
        let trie = PromptMembershipTrie::new();
        let worker_a = worker(1, 0);
        let worker_b = worker(2, 0);
        let lookup_a = lookup();
        let lookup_b = lookup();

        trie.store_chain(worker_a, &lookup_a, None, &[1, 2, 3, 4]);
        trie.store_chain(worker_b, &lookup_b, None, &[1, 2, 5]);

        let stale_node = lookup_node(&lookup_a, 4);
        assert!(!node_contains_hash(&stale_node, 3));
        assert!(!node_contains_hash(&stale_node, 4));

        let mut worker_lookup = lookup_a.write();
        let resolved = PromptMembershipTrie::resolve_lookup(
            &mut worker_lookup,
            worker_a,
            4,
            LookupRepairDirection::TowardHead,
        )
        .expect("stale suffix should resolve");

        assert!(Arc::ptr_eq(
            worker_lookup
                .get(&3)
                .expect("hash 3 should be batch repaired"),
            &resolved,
        ));
        assert!(!Arc::ptr_eq(
            worker_lookup
                .get(&4)
                .expect("current remove hash is not pre-repaired"),
            &resolved,
        ));
    }
}
