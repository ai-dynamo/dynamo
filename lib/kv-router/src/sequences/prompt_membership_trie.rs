// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dynamo_tokens::SequenceHash;
use parking_lot::RwLock;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::cleanup::{self, CleanableNode, CleanupGuard, CleanupState};
use crate::protocols::WorkerWithDpRank;

type SharedNode = Arc<RwLock<PromptTrieNode>>;

/// Source of [`PromptTrieNode::full_version`] values: unique across every node, so a
/// (node, version) pair names one exact full-edge worker set even if a node's memory is reused.
static NEXT_FULL_VERSION: AtomicU64 = AtomicU64::new(1);

fn next_full_version() -> u64 {
    NEXT_FULL_VERSION.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Default)]
pub(super) struct PromptTrieNode {
    edge: Vec<SequenceHash>,
    worker_cutoffs: FxHashMap<WorkerWithDpRank, usize>,
    full_edge_workers: FxHashSet<WorkerWithDpRank>,
    /// Changes whenever `full_edge_workers` does, from [`next_full_version`].
    full_version: u64,
    children: FxHashMap<SequenceHash, SharedNode>,
}

impl PromptTrieNode {
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

    fn drop_worker(&mut self, worker: WorkerWithDpRank) {
        if self.full_edge_workers.remove(&worker) {
            self.full_version = next_full_version();
        }
        self.worker_cutoffs.remove(&worker);
        self.clear_children_if_unreachable();
    }

    fn promote_to_full(&mut self, worker: WorkerWithDpRank) {
        if !self.full_edge_workers.contains(&worker) {
            self.worker_cutoffs.remove(&worker);
            self.full_edge_workers.insert(worker);
            self.full_version = next_full_version();
        }
    }

    fn extend_worker_to(&mut self, worker: WorkerWithDpRank, end: usize) {
        debug_assert!(end > 0 && end <= self.edge.len());
        if end == self.edge.len() {
            self.promote_to_full(worker);
            return;
        }
        if self.full_edge_workers.contains(&worker) {
            return;
        }
        self.worker_cutoffs
            .entry(worker)
            .and_modify(|cutoff| *cutoff = (*cutoff).max(end))
            .or_insert(end);
    }

    fn truncate_worker_at(&mut self, worker: WorkerWithDpRank, pos: usize) {
        let current_cutoff = self.current_cutoff(worker);
        if pos >= current_cutoff {
            return;
        }

        if pos == 0 {
            self.drop_worker(worker);
        } else {
            if self.full_edge_workers.remove(&worker) {
                self.full_version = next_full_version();
            }
            self.worker_cutoffs.insert(worker, pos);
            self.clear_children_if_unreachable();
        }
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

/// Per-worker overlap depths for one query, from
/// [`PromptMembershipTrie::compute_overlap_depths_into`]. Reused across queries so its
/// allocations are kept.
#[derive(Debug, Default)]
pub(super) struct OverlapDepths {
    /// Workers covering the query's shared leading edges; each matches `base_depth`
    /// unless `overrides` says otherwise. Kept across queries while `base_key` matches.
    base: FxHashSet<WorkerWithDpRank>,
    /// The first node (address) and its [`PromptTrieNode::full_version`] that `base`
    /// copies.
    base_key: Option<(usize, u64)>,
    /// Whether this query has a base set.
    has_base: bool,
    base_depth: usize,
    overrides: FxHashMap<WorkerWithDpRank, usize>,
    /// Scratch for the workers still matching past the base set.
    active: FxHashSet<WorkerWithDpRank>,
}

impl OverlapDepths {
    fn clear(&mut self) {
        self.has_base = false;
        self.base_depth = 0;
        self.overrides.clear();
        self.active.clear();
    }

    /// This query's base set, the key identifying its exact contents, and its depth.
    pub(super) fn base(&self) -> Option<(&FxHashSet<WorkerWithDpRank>, (usize, u64), usize)> {
        let key = self.base_key?;
        self.has_base.then_some((&self.base, key, self.base_depth))
    }

    /// Workers whose depth differs from what the base set gives them.
    pub(super) fn overrides(&self) -> impl Iterator<Item = (WorkerWithDpRank, usize)> + '_ {
        self.overrides
            .iter()
            .map(|(&worker, &depth)| (worker, depth))
    }

    /// Blocks of the query `worker`'s active prompts share; 0 when it shares none.
    #[cfg(test)]
    pub(super) fn get(&self, worker: &WorkerWithDpRank) -> usize {
        if !self.overrides.is_empty()
            && let Some(&depth) = self.overrides.get(worker)
        {
            return depth;
        }
        if self.has_base && self.base.contains(worker) {
            self.base_depth
        } else {
            0
        }
    }
}

#[derive(Default)]
pub(super) struct PromptMembershipTrie {
    root: SharedNode,
    cleanup: CleanupState,
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

    fn split_node(node: &mut PromptTrieNode, pos: usize) {
        debug_assert!(pos > 0 && pos < node.edge.len());

        let suffix_edge = node.edge.split_off(pos);
        let suffix_first_hash = suffix_edge[0];

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

        if !to_promote.is_empty() {
            node.full_version = next_full_version();
        }
        for worker in to_promote {
            node.worker_cutoffs.remove(&worker);
            node.full_edge_workers.insert(worker);
        }

        let suffix_children = std::mem::take(&mut node.children);
        let suffix = Arc::new(RwLock::new(PromptTrieNode {
            edge: suffix_edge,
            worker_cutoffs: suffix_cutoffs,
            full_edge_workers: suffix_full,
            full_version: next_full_version(),
            children: suffix_children,
        }));
        node.children.insert(suffix_first_hash, suffix);
    }

    fn matching_prefix_len(edge: &[SequenceHash], path: &[SequenceHash]) -> usize {
        edge.iter()
            .zip(path)
            .take_while(|(edge_hash, path_hash)| edge_hash == path_hash)
            .count()
    }

    fn new_path_node(worker: WorkerWithDpRank, edge: Vec<SequenceHash>) -> SharedNode {
        debug_assert!(!edge.is_empty());
        Arc::new(RwLock::new(PromptTrieNode {
            edge,
            worker_cutoffs: FxHashMap::default(),
            full_edge_workers: PromptTrieNode::full_edge_workers_for(worker),
            full_version: next_full_version(),
            children: FxHashMap::default(),
        }))
    }

    pub(super) fn store_path(
        &self,
        worker: WorkerWithDpRank,
        path: &[SequenceHash],
        new_suffix_start: usize,
    ) {
        assert!(
            new_suffix_start < path.len(),
            "prompt store suffix must start inside a non-empty path"
        );

        let mut current = self.root.read().children.get(&path[0]).cloned();
        if current.is_none() {
            let mut root = self.root.write();
            current = root.children.get(&path[0]).cloned();
            if current.is_none() {
                if new_suffix_start != 0 {
                    tracing::warn!(
                        ?worker,
                        new_suffix_start,
                        "prompt store prefix is missing from the trie"
                    );
                    return;
                }
                root.children
                    .insert(path[0], Self::new_path_node(worker, path.to_vec()));
                return;
            }
        }
        let mut current = current.expect("root child was inserted or observed");

        let mut path_pos = 0;
        loop {
            let mut node = current.write();
            let remaining = &path[path_pos..];
            let edge_len = node.edge.len();
            let match_len = Self::matching_prefix_len(&node.edge, remaining);
            if match_len == 0 {
                tracing::warn!(?worker, path_pos, "prompt trie path changed during store");
                return;
            }

            let existing_prefix_len = new_suffix_start.saturating_sub(path_pos).min(match_len);
            if existing_prefix_len > 0 && !node.covers_pos(worker, existing_prefix_len - 1) {
                tracing::warn!(
                    ?worker,
                    path_pos,
                    existing_prefix_len,
                    "worker no longer covers the prompt store prefix"
                );
                return;
            }

            let path_ends = match_len == remaining.len();
            let edge_ends = match_len == edge_len;
            if !edge_ends {
                if path_ends {
                    if path_pos + match_len > new_suffix_start {
                        node.extend_worker_to(worker, match_len);
                    }
                    return;
                }

                if path_pos + match_len < new_suffix_start {
                    tracing::warn!(
                        ?worker,
                        path_pos,
                        match_len,
                        new_suffix_start,
                        "prompt store diverged inside its existing prefix"
                    );
                    return;
                }

                Self::split_node(&mut node, match_len);
                if path_pos + match_len > new_suffix_start {
                    node.promote_to_full(worker);
                }
                let tail = &remaining[match_len..];
                debug_assert!(!tail.is_empty());
                node.children
                    .insert(tail[0], Self::new_path_node(worker, tail.to_vec()));
                return;
            }

            if path_pos + edge_len > new_suffix_start {
                node.promote_to_full(worker);
            }
            path_pos += edge_len;
            if path_pos == path.len() {
                return;
            }

            let next_hash = path[path_pos];
            let next = match node.children.get(&next_hash).cloned() {
                Some(next) => next,
                None => {
                    if path_pos < new_suffix_start {
                        tracing::warn!(
                            ?worker,
                            path_pos,
                            new_suffix_start,
                            "prompt store prefix is missing from the trie"
                        );
                        return;
                    }
                    node.children.insert(
                        next_hash,
                        Self::new_path_node(worker, path[path_pos..].to_vec()),
                    );
                    return;
                }
            };
            drop(node);
            current = next;
        }
    }

    pub(super) fn remove_path(
        &self,
        worker: WorkerWithDpRank,
        path: &[SequenceHash],
        remove_from: usize,
    ) {
        assert!(
            remove_from < path.len(),
            "prompt removal must remove a non-empty suffix"
        );

        let mut current = {
            let root = self.root.read();
            let Some(node) = root.children.get(&path[0]).cloned() else {
                return;
            };
            node
        };
        let mut path_pos = 0;

        loop {
            let mut node = current.write();
            let remaining = &path[path_pos..];
            let edge_len = node.edge.len();
            let match_len = Self::matching_prefix_len(&node.edge, remaining);
            if match_len == 0 || (match_len < edge_len && match_len < remaining.len()) {
                tracing::warn!(?worker, path_pos, "prompt removal path is missing");
                return;
            }

            if path_pos + edge_len <= remove_from {
                if match_len != edge_len {
                    tracing::warn!(?worker, path_pos, "prompt removal prefix is missing");
                    return;
                }
                path_pos += edge_len;
                let Some(next) = node.children.get(&path[path_pos]).cloned() else {
                    return;
                };
                drop(node);
                current = next;
                continue;
            }

            let local_remove_from = remove_from.saturating_sub(path_pos);
            debug_assert!(local_remove_from < match_len);
            node.truncate_worker_at(worker, local_remove_from);

            if match_len == remaining.len() {
                return;
            }
            debug_assert_eq!(match_len, node.edge.len());
            path_pos += match_len;
            let Some(next) = node.children.get(&path[path_pos]).cloned() else {
                return;
            };
            drop(node);
            current = next;
        }
    }

    pub(super) fn remove_worker(&self, worker: WorkerWithDpRank) {
        let mut nodes = {
            let root = self.root.read();
            VecDeque::from_iter(root.children.values().cloned())
        };
        while let Some(node) = nodes.pop_front() {
            let mut guard = node.write();
            guard.drop_worker(worker);
            nodes.extend(guard.children.values().cloned());
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
                    matched_depth
                        .reserve(guard.full_edge_workers.len() + guard.worker_cutoffs.len());
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

    /// [`Self::compute_overlap_depths`] into reusable scratch, without materializing a
    /// depth for every worker that shares the query's leading edges.
    ///
    /// A shared prefix, such as a common system prompt, is covered by nearly every worker
    /// with an active request. Those workers form the *base set*: the first node's
    /// full-edge workers, carried through every following node that keeps all of them.
    /// Each base worker that is not overridden has the depth at which the base set
    /// splits; the rest of the walk only touches the workers that match deeper.
    pub(super) fn compute_overlap_depths_into(
        &self,
        query: Option<&[SequenceHash]>,
        out: &mut OverlapDepths,
    ) {
        out.clear();
        let Some(query) = query else {
            return;
        };
        if query.is_empty() {
            return;
        }

        let Some(first) = self.root.read().children.get(&query[0]).cloned() else {
            return;
        };
        let mut query_pos = 0usize;
        let mut depth = 0usize;

        // The first node seeds the base set; its partial workers only reach their cutoff.
        let mut next_child = {
            let guard = first.read();
            let edge_len = guard.edge.len();
            let match_len = Self::edge_match_len(&guard.edge, &query[query_pos..]);
            let key = (Arc::as_ptr(&first) as usize, guard.full_version);
            if out.base_key != Some(key) {
                out.base.clone_from(&guard.full_edge_workers);
                out.base_key = Some(key);
            }
            for (&worker, &cutoff) in &guard.worker_cutoffs {
                let contribution = cutoff.min(match_len);
                if contribution > 0 {
                    out.overrides.insert(worker, contribution);
                }
            }
            if out.base.is_empty() {
                return;
            }
            out.has_base = true;
            depth += match_len;
            out.base_depth = depth;
            if match_len < edge_len {
                return;
            }
            query_pos += match_len;
            if query_pos < query.len() {
                guard.children.get(&query[query_pos]).cloned()
            } else {
                None
            }
        };

        // Nodes that keep every base worker and have no partial workers change nothing.
        let base_count = out.base.len();
        let mut active = std::mem::take(&mut out.active);
        let narrowed = loop {
            if query_pos >= query.len() {
                break false;
            }
            let Some(child) = next_child.take() else {
                break false;
            };
            let guard = child.read();
            let edge_len = guard.edge.len();
            let match_len = Self::edge_match_len(&guard.edge, &query[query_pos..]);
            let prev_depth = depth;

            if guard.worker_cutoffs.is_empty() && guard.full_edge_workers.len() == base_count {
                next_child = if match_len == edge_len && query_pos + match_len < query.len() {
                    guard.children.get(&query[query_pos + match_len]).cloned()
                } else {
                    None
                };
                depth += match_len;
                out.base_depth = depth;
                if match_len < edge_len {
                    break false;
                }
                query_pos += match_len;
                continue;
            }

            // This node splits the base set: workers that stop here keep `prev_depth`.
            out.base_depth = prev_depth;
            for &worker in &guard.full_edge_workers {
                if out.base.contains(&worker) {
                    active.insert(worker);
                }
            }
            for (&worker, &cutoff) in &guard.worker_cutoffs {
                if out.base.contains(&worker) {
                    out.overrides
                        .insert(worker, prev_depth + cutoff.min(match_len));
                }
            }
            next_child = if match_len == edge_len
                && !active.is_empty()
                && query_pos + match_len < query.len()
            {
                guard.children.get(&query[query_pos + match_len]).cloned()
            } else {
                None
            };
            if active.is_empty() {
                break true;
            }
            depth += match_len;
            if match_len < edge_len {
                for &worker in &active {
                    out.overrides.insert(worker, depth);
                }
                active.clear();
                break true;
            }
            query_pos += match_len;
            break true;
        };

        if narrowed {
            // The remaining walk follows only the workers that matched past the split, as
            // `compute_overlap_depths` does for every worker.
            let mut active_count = active.len();
            while active_count > 0 && query_pos < query.len() {
                let Some(child) = next_child.take() else {
                    break;
                };
                let guard = child.read();
                let edge_len = guard.edge.len();
                let edge_match_len = Self::edge_match_len(&guard.edge, &query[query_pos..]);
                let prev_depth = depth;
                if !guard.worker_cutoffs.is_empty() {
                    active.retain(|worker| {
                        if guard.full_edge_workers.contains(worker) {
                            true
                        } else if let Some(&cutoff) = guard.worker_cutoffs.get(worker) {
                            out.overrides
                                .insert(*worker, prev_depth + cutoff.min(edge_match_len));
                            false
                        } else {
                            out.overrides.insert(*worker, prev_depth);
                            false
                        }
                    });
                    active_count = active.len();
                } else if guard.full_edge_workers.len() != active_count {
                    active.retain(|worker| {
                        if guard.full_edge_workers.contains(worker) {
                            true
                        } else {
                            out.overrides.insert(*worker, prev_depth);
                            false
                        }
                    });
                    active_count = active.len();
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
                drop(guard);
                if active_count == 0 {
                    break;
                }
                depth += edge_match_len;
                if edge_match_len < edge_len {
                    break;
                }
                query_pos += edge_match_len;
            }
            for &worker in &active {
                out.overrides.insert(worker, depth);
            }
        }
        active.clear();
        out.active = active;
    }

    /// Length of the common prefix of `edge` and `query`, counting the first hash as a
    /// match: a child is only reached through its first hash.
    fn edge_match_len(edge: &[SequenceHash], query: &[SequenceHash]) -> usize {
        let walk_len = edge.len().min(query.len());
        let mut match_len = 1usize;
        for i in 1..walk_len {
            if edge[i] != query[i] {
                break;
            }
            match_len += 1;
        }
        match_len
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

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_depths(
        prompts_by_worker: &FxHashMap<WorkerWithDpRank, Vec<Vec<SequenceHash>>>,
        query: &[SequenceHash],
    ) -> FxHashMap<WorkerWithDpRank, usize> {
        prompts_by_worker
            .iter()
            .filter_map(|(&worker, prompts)| {
                let depth = prompts
                    .iter()
                    .map(|prompt| PromptMembershipTrie::matching_prefix_len(prompt, query))
                    .max()
                    .unwrap_or(0);
                (depth > 0).then_some((worker, depth))
            })
            .collect()
    }

    fn worker(worker_id: u64, dp_rank: u32) -> WorkerWithDpRank {
        WorkerWithDpRank::new(worker_id, dp_rank)
    }

    #[test]
    fn full_path_continuations_extend_and_trim() {
        let trie = PromptMembershipTrie::default();
        let worker = worker(1, 0);

        trie.store_path(worker, &[1, 2, 3], 0);
        trie.store_path(worker, &[1, 2, 3, 4, 5], 3);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4, 5])),
            FxHashMap::from_iter([(worker, 5)]),
        );

        trie.remove_path(worker, &[1, 2, 3, 4, 5], 3);
        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4, 5])),
            FxHashMap::from_iter([(worker, 3)]),
        );
    }

    #[test]
    fn branching_continuations_across_workers_match_expected_depths() {
        let trie = PromptMembershipTrie::default();
        let worker_a = worker(1, 0);
        let worker_b = worker(2, 0);

        trie.store_path(worker_a, &[1, 2, 3, 4], 0);
        trie.store_path(worker_b, &[1, 2, 5], 0);

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
        let trie = PromptMembershipTrie::default();
        let worker = worker(1, 0);

        trie.store_path(worker, &[1, 2, 3, 4, 5], 0);
        trie.remove_path(worker, &[1, 2, 3, 4, 5], 2);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4, 5])),
            FxHashMap::from_iter([(worker, 2)]),
        );
    }

    #[test]
    fn remove_worker_preserves_other_workers() {
        let trie = PromptMembershipTrie::default();
        let worker_a = worker(1, 0);
        let worker_b = worker(2, 0);

        trie.store_path(worker_a, &[1, 2, 3], 0);
        trie.store_path(worker_b, &[1, 2, 3], 0);

        trie.remove_worker(worker_a);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3])),
            FxHashMap::from_iter([(worker_b, 3)]),
        );
    }

    #[test]
    fn multiple_dp_ranks_with_same_worker_id_remain_isolated() {
        let trie = PromptMembershipTrie::default();
        let worker_a = worker(1, 0);
        let worker_b = worker(1, 1);

        trie.store_path(worker_a, &[1, 2, 3], 0);
        trie.store_path(worker_b, &[1, 2], 0);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3])),
            FxHashMap::from_iter([(worker_a, 3), (worker_b, 2)]),
        );
    }

    #[test]
    fn clear_worker_state_then_reuse_starts_empty() {
        let trie = PromptMembershipTrie::default();
        let worker = worker(1, 0);

        trie.store_path(worker, &[1, 2, 3], 0);
        trie.remove_worker(worker);
        assert!(trie.compute_overlap_depths(Some(&[1, 2, 3])).is_empty());

        trie.store_path(worker, &[1, 2], 0);
        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3])),
            FxHashMap::from_iter([(worker, 2)]),
        );
    }

    #[test]
    fn redundant_batched_remove_is_idempotent() {
        let trie = PromptMembershipTrie::default();
        let worker = worker(1, 0);

        trie.store_path(worker, &[1, 2, 3, 4], 0);
        trie.remove_path(worker, &[1, 2, 3, 4], 1);
        trie.remove_path(worker, &[1, 2, 3, 4], 1);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4])),
            FxHashMap::from_iter([(worker, 1)]),
        );
    }

    #[test]
    fn path_ending_inside_another_workers_edge_uses_a_cutoff() {
        let trie = PromptMembershipTrie::default();
        let worker_a = worker(1, 0);
        let worker_b = worker(2, 0);

        trie.store_path(worker_a, &[1, 2, 3, 4], 0);
        trie.store_path(worker_b, &[1, 2], 0);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4])),
            FxHashMap::from_iter([(worker_a, 4), (worker_b, 2)]),
        );
    }

    #[test]
    fn full_path_mutation_follows_suffix_after_a_split() {
        let trie = PromptMembershipTrie::default();
        let worker_a = worker(1, 0);
        let worker_b = worker(2, 0);

        trie.store_path(worker_a, &[1, 2, 3, 4], 0);
        trie.store_path(worker_b, &[1, 2, 5], 0);
        trie.remove_path(worker_a, &[1, 2, 3, 4], 2);
        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4])),
            FxHashMap::from_iter([(worker_a, 2), (worker_b, 2)]),
        );

        trie.store_path(worker_a, &[1, 2, 3, 4], 2);
        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4])),
            FxHashMap::from_iter([(worker_a, 4), (worker_b, 2)]),
        );
    }

    #[test]
    fn split_translates_worker_cutoffs_before_at_and_after_boundary() {
        let before = worker(1, 0);
        let at = worker(2, 0);
        let after = worker(3, 0);
        let full = worker(4, 0);
        let mut node = PromptTrieNode {
            edge: vec![1, 2, 3, 4],
            worker_cutoffs: FxHashMap::from_iter([(before, 1), (at, 2), (after, 3)]),
            full_edge_workers: PromptTrieNode::full_edge_workers_for(full),
            full_version: 0,
            children: FxHashMap::default(),
        };

        PromptMembershipTrie::split_node(&mut node, 2);
        assert_eq!(node.current_cutoff(before), 1);
        assert_eq!(node.current_cutoff(at), 2);
        assert_eq!(node.current_cutoff(after), 2);
        assert_eq!(node.current_cutoff(full), 2);

        let suffix = node.children.get(&3).expect("split suffix").read();
        assert_eq!(suffix.current_cutoff(before), 0);
        assert_eq!(suffix.current_cutoff(at), 0);
        assert_eq!(suffix.current_cutoff(after), 1);
        assert_eq!(suffix.current_cutoff(full), 2);
    }

    /// Random stores and removals against a naive per-worker prompt model, checking both
    /// overlap-depth computations after every step.
    fn randomized_against_naive_model(
        workers: &[WorkerWithDpRank],
        prompts: &[Vec<SequenceHash>],
        steps: usize,
        mut seed: u64,
    ) {
        let trie = PromptMembershipTrie::default();
        let mut active: FxHashMap<WorkerWithDpRank, Vec<Vec<SequenceHash>>> = FxHashMap::default();
        let mut counts: FxHashMap<WorkerWithDpRank, FxHashMap<SequenceHash, usize>> =
            FxHashMap::default();
        let mut depths = OverlapDepths::default();

        for step in 0..steps {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let worker = workers[(seed as usize) % workers.len()];
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);

            if step % 211 == 0 {
                trie.remove_worker(worker);
                active.remove(&worker);
                counts.remove(&worker);
            } else if seed & 1 == 0
                || active
                    .get(&worker)
                    .is_none_or(|requests| requests.is_empty())
            {
                let prompt = prompts[((seed >> 8) as usize) % prompts.len()].clone();
                let worker_counts = counts.entry(worker).or_default();
                let new_suffix_start = prompt
                    .iter()
                    .position(|hash| !worker_counts.contains_key(hash));
                for &hash in &prompt {
                    *worker_counts.entry(hash).or_default() += 1;
                }
                active.entry(worker).or_default().push(prompt.clone());
                if let Some(new_suffix_start) = new_suffix_start {
                    trie.store_path(worker, &prompt, new_suffix_start);
                }
            } else {
                let requests = active.get_mut(&worker).expect("worker has active prompts");
                let request_idx = ((seed >> 8) as usize) % requests.len();
                let prompt = requests.swap_remove(request_idx);
                let worker_counts = counts.get_mut(&worker).expect("worker has counts");
                let remove_from = prompt
                    .iter()
                    .position(|hash| worker_counts.get(hash) == Some(&1));
                for &hash in &prompt {
                    let count = worker_counts
                        .get_mut(&hash)
                        .expect("active hash has a count");
                    *count -= 1;
                    if *count == 0 {
                        worker_counts.remove(&hash);
                    }
                }
                if let Some(remove_from) = remove_from {
                    trie.remove_path(worker, &prompt, remove_from);
                }
            }

            let expected_hashes: FxHashMap<_, _> = counts
                .iter()
                .filter_map(|(&worker, counts)| {
                    (!counts.is_empty()).then_some((worker, counts.keys().copied().collect()))
                })
                .collect();
            assert_eq!(trie.worker_hashes(), expected_hashes, "step {step}");
            for query in prompts {
                let expected = expected_depths(&active, query);
                assert_eq!(
                    trie.compute_overlap_depths(Some(query)),
                    expected,
                    "step {step}, query {query:?}"
                );
                trie.compute_overlap_depths_into(Some(query), &mut depths);
                for worker in workers {
                    assert_eq!(
                        depths.get(worker),
                        expected.get(worker).copied().unwrap_or(0),
                        "step {step}, query {query:?}, {worker:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn randomized_prefix_union_matches_naive_model() {
        let workers = [worker(1, 0), worker(2, 0), worker(2, 1)];
        // Every hash denotes exactly one lineage position. Shared prefixes
        // reuse hashes; branches and unrelated roots use distinct hashes.
        let prompts = [
            vec![1, 2, 3, 4],
            vec![1, 2, 3, 5],
            vec![1, 2, 6],
            vec![1, 7],
            vec![8, 9, 10],
            vec![8, 9, 11],
            vec![12],
        ];
        randomized_against_naive_model(&workers, &prompts, 5_000, 42);
    }

    /// Many workers behind one shared prefix (a common system prompt), as the base set of
    /// `compute_overlap_depths_into` sees it: pass-through nodes, splits with and without
    /// partial workers, and queries that stop inside the shared edges.
    #[test]
    fn randomized_shared_prefix_matches_naive_model() {
        let workers: Vec<_> = (0..16).map(|id| worker(id, id as u32 % 2)).collect();
        let shared = [100, 101, 102, 103];
        let with_shared = |tail: &[SequenceHash]| [&shared[..], tail].concat();
        let prompts = [
            with_shared(&[1, 2, 3]),
            with_shared(&[1, 2, 4]),
            with_shared(&[1, 5]),
            with_shared(&[6]),
            with_shared(&[6, 7, 8]),
            shared.to_vec(),
            shared[..2].to_vec(),
            vec![100],
            vec![9, 10],
        ];
        randomized_against_naive_model(&workers, &prompts, 6_000, 7);
    }

    #[test]
    fn concurrent_cross_worker_splits_preserve_both_paths() {
        for _ in 0..128 {
            let trie = Arc::new(PromptMembershipTrie::default());
            let worker_a = worker(1, 0);
            let worker_b = worker(2, 0);
            let worker_c = worker(3, 0);
            trie.store_path(worker_a, &[1, 2, 3, 4, 5, 6], 0);

            let barrier = Arc::new(std::sync::Barrier::new(3));
            let left = {
                let trie = Arc::clone(&trie);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    trie.store_path(worker_b, &[1, 2, 7, 8], 0);
                })
            };
            let right = {
                let trie = Arc::clone(&trie);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    trie.store_path(worker_c, &[1, 2, 3, 9], 0);
                })
            };
            barrier.wait();
            left.join().unwrap();
            right.join().unwrap();

            assert_eq!(
                trie.compute_overlap_depths(Some(&[1, 2, 7, 8])),
                FxHashMap::from_iter([(worker_a, 2), (worker_b, 4), (worker_c, 2)])
            );
            assert_eq!(
                trie.compute_overlap_depths(Some(&[1, 2, 3, 9])),
                FxHashMap::from_iter([(worker_a, 3), (worker_b, 2), (worker_c, 4)])
            );
        }
    }

    #[test]
    fn concurrent_first_root_insertion_preserves_both_workers() {
        for _ in 0..128 {
            let trie = Arc::new(PromptMembershipTrie::default());
            let worker_a = worker(1, 0);
            let worker_b = worker(2, 0);
            let barrier = Arc::new(std::sync::Barrier::new(3));

            let left = {
                let trie = Arc::clone(&trie);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    trie.store_path(worker_a, &[1, 2, 3], 0);
                })
            };
            let right = {
                let trie = Arc::clone(&trie);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    trie.store_path(worker_b, &[1, 2, 3], 0);
                })
            };
            barrier.wait();
            left.join().unwrap();
            right.join().unwrap();

            assert_eq!(
                trie.compute_overlap_depths(Some(&[1, 2, 3])),
                FxHashMap::from_iter([(worker_a, 3), (worker_b, 3)])
            );
        }
    }

    #[test]
    fn cleanup_does_not_unlink_a_pinned_child() {
        let trie = PromptMembershipTrie::default();
        let worker = worker(1, 0);
        trie.store_path(worker, &[1, 2, 3], 0);
        let pinned = trie
            .root
            .read()
            .children
            .get(&1)
            .cloned()
            .expect("root child");

        trie.remove_worker(worker);
        cleanup::sweep_stale_children(&trie.root);
        assert!(trie.root.read().children.contains_key(&1));

        drop(pinned);
        cleanup::sweep_stale_children(&trie.root);
        assert!(!trie.root.read().children.contains_key(&1));
    }
}
