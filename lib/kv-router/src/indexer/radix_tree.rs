// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-threaded compressed radix tree for KV cache routing.

use std::{cell::RefCell, collections::VecDeque, rc::Rc};

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use super::compressed_radix::{NodeState, append_dump_events};
use super::{EventWarningKind, MatchDetails, PreBoundEventCounters};
use crate::protocols::*;

pub(crate) type SharedRadixBlock = Rc<RefCell<RadixBlock>>;
type WorkerLookup = FxHashMap<ExternalSequenceBlockHash, SharedRadixBlock>;

#[derive(Debug)]
pub(crate) struct RadixBlock {
    state: NodeState,
    children: FxHashMap<LocalBlockHash, SharedRadixBlock>,
    /// Once a node has children it is never eligible for leaf extension again.
    internal: bool,
}

impl RadixBlock {
    fn root() -> Self {
        Self {
            state: NodeState::empty(),
            children: FxHashMap::default(),
            internal: true,
        }
    }

    fn for_blocks(blocks: &[KvCacheStoredBlockData], worker: WorkerWithDpRank) -> Self {
        Self {
            state: NodeState::for_blocks(blocks, worker),
            children: FxHashMap::default(),
            internal: false,
        }
    }

    fn clear_children_if_unreachable(&mut self) {
        if self.state.full_edge_workers.is_empty() {
            self.children.clear();
        }
    }
}

pub struct RadixTree {
    root: SharedRadixBlock,
    lookup: FxHashMap<WorkerWithDpRank, WorkerLookup>,
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RadixTree {
    fn drop(&mut self) {
        let mut stack = self
            .root
            .borrow_mut()
            .children
            .drain()
            .map(|(_, child)| child)
            .collect::<Vec<_>>();
        for (_, worker_lookup) in self.lookup.drain() {
            stack.extend(worker_lookup.into_values());
        }

        while let Some(block) = stack.pop() {
            if let Ok(cell) = Rc::try_unwrap(block) {
                stack.extend(cell.into_inner().children.into_values());
            }
        }
    }
}

impl RadixTree {
    pub fn new() -> Self {
        Self {
            root: Rc::new(RefCell::new(RadixBlock::root())),
            lookup: FxHashMap::default(),
        }
    }

    pub fn find_match_details(
        &self,
        sequence: Vec<LocalBlockHash>,
        early_exit: bool,
    ) -> MatchDetails {
        let mut details = MatchDetails::new();
        if sequence.is_empty() {
            return details;
        }

        let mut next = self.root.borrow().children.get(&sequence[0]).cloned();
        let mut active = FxHashSet::default();
        let mut matched_depth = 0usize;
        let mut seq_pos = 0usize;
        let mut first_node = true;
        let mut last_matched_hash = None;

        while seq_pos < sequence.len() {
            let Some(node) = next.take() else {
                break;
            };
            let node = node.borrow();
            let walk_len = node.state.edge.len().min(sequence.len() - seq_pos);
            let edge_match_len = node
                .state
                .edge
                .iter()
                .take(walk_len)
                .zip(&sequence[seq_pos..])
                .take_while(|((local_hash, _), query_hash)| local_hash == *query_hash)
                .count();

            if edge_match_len == 0 {
                break;
            }

            if first_node {
                active.clone_from(&node.state.full_edge_workers);
                for (&worker, &cutoff) in &node.state.worker_cutoffs {
                    let contribution = cutoff.min(edge_match_len);
                    if contribution == 0 {
                        continue;
                    }
                    details
                        .overlap_scores
                        .scores
                        .insert(worker, contribution as u32);
                    details
                        .last_matched_hashes
                        .insert(worker, node.state.edge[contribution - 1].1);
                }
                first_node = false;
            } else {
                active.retain(|worker| {
                    if node.state.full_edge_workers.contains(worker) {
                        return true;
                    }

                    if let Some(&cutoff) = node.state.worker_cutoffs.get(worker) {
                        let contribution = cutoff.min(edge_match_len);
                        details
                            .overlap_scores
                            .scores
                            .insert(*worker, (matched_depth + contribution) as u32);
                        let hash = if contribution > 0 {
                            node.state.edge[contribution - 1].1
                        } else {
                            last_matched_hash.expect("non-root match must have a prior hash")
                        };
                        details.last_matched_hashes.insert(*worker, hash);
                    } else {
                        details
                            .overlap_scores
                            .scores
                            .insert(*worker, matched_depth as u32);
                        if let Some(hash) = last_matched_hash {
                            details.last_matched_hashes.insert(*worker, hash);
                        }
                    }
                    false
                });
            }

            matched_depth += edge_match_len;
            last_matched_hash = Some(node.state.edge[edge_match_len - 1].1);

            if active.is_empty()
                || edge_match_len < node.state.edge.len()
                || matched_depth == sequence.len()
                || (early_exit && active.len() == 1)
            {
                break;
            }

            seq_pos += edge_match_len;
            next = node.children.get(&sequence[seq_pos]).cloned();
        }

        for worker in active {
            details
                .overlap_scores
                .scores
                .insert(worker, matched_depth as u32);
            if let Some(hash) = last_matched_hash {
                details.last_matched_hashes.insert(worker, hash);
            }
        }

        details
    }

    pub fn find_matches(&self, sequence: Vec<LocalBlockHash>, early_exit: bool) -> OverlapScores {
        self.find_match_details(sequence, early_exit).overlap_scores
    }

    pub fn apply_event(&mut self, event: RouterEvent) -> Result<(), KvCacheEventError> {
        self.apply_event_with_counters(event, None)
    }

    pub(crate) fn apply_event_with_counters(
        &mut self,
        event: RouterEvent,
        counters: Option<&PreBoundEventCounters>,
    ) -> Result<(), KvCacheEventError> {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let event_id = event.event.event_id;
        self.lookup.entry(worker).or_default();

        match event.event.data {
            KvCacheEventData::Stored(store) => self.apply_stored(worker, store, event_id, counters),
            KvCacheEventData::Removed(remove) => self.apply_removed(worker, remove, event_id),
            KvCacheEventData::Cleared => {
                self.remove_worker_dp_rank(worker.worker_id, worker.dp_rank);
                Ok(())
            }
        }
    }

    fn apply_stored(
        &mut self,
        worker: WorkerWithDpRank,
        store: KvCacheStoreData,
        event_id: u64,
        counters: Option<&PreBoundEventCounters>,
    ) -> Result<(), KvCacheEventError> {
        if let Some(parent_hash) = store.parent_hash
            && store
                .blocks
                .iter()
                .any(|block| block.block_hash == parent_hash)
        {
            tracing::warn!(
                worker_id = worker.worker_id,
                dp_rank = worker.dp_rank,
                event_id,
                ?parent_hash,
                "Detected self-referencing store event"
            );
            return Err(KvCacheEventError::InvalidBlockSequence);
        }

        let mut duplicate_store = !store.blocks.is_empty();
        let mut parent = match store.parent_hash {
            Some(parent_hash) => {
                let Some(node) = self.lookup[&worker].get(&parent_hash).cloned() else {
                    self.log_missing_parent(worker, event_id, &store);
                    return Err(KvCacheEventError::ParentBlockNotFound);
                };
                let parent_pos = {
                    let node_ref = node.borrow();
                    let Some(&pos) = node_ref.state.edge_index.get(&parent_hash) else {
                        self.log_missing_parent(worker, event_id, &store);
                        return Err(KvCacheEventError::ParentBlockNotFound);
                    };
                    if !node_ref.state.covers_pos(worker, pos) {
                        self.lookup.get_mut(&worker).unwrap().remove(&parent_hash);
                        self.log_missing_parent(worker, event_id, &store);
                        return Err(KvCacheEventError::ParentBlockNotFound);
                    }
                    pos
                };

                if let Some(done_duplicate) =
                    self.try_reuse_parent_edge(worker, &node, parent_pos, &store.blocks)
                {
                    duplicate_store &= done_duplicate;
                    if duplicate_store && let Some(counters) = counters {
                        counters.inc_warning(EventWarningKind::DuplicateStore);
                    }
                    return Ok(());
                }

                if !node.borrow().state.tail_hash_is(parent_hash) {
                    self.split_node(&node, parent_pos + 1);
                }
                node
            }
            None => self.root.clone(),
        };

        let mut remaining = store.blocks.as_slice();
        while !remaining.is_empty() {
            let first_local = remaining[0].tokens_hash;
            let child = parent.borrow().children.get(&first_local).cloned();

            let Some(child) = child else {
                let can_extend = {
                    let parent_ref = parent.borrow();
                    !parent_ref.state.edge.is_empty()
                        && !parent_ref.internal
                        && parent_ref
                            .state
                            .covers_pos(worker, parent_ref.state.edge.len() - 1)
                };
                if can_extend {
                    parent
                        .borrow_mut()
                        .state
                        .append_blocks_to_leaf(worker, remaining);
                    self.update_lookup_for_blocks(worker, remaining, &parent);
                    duplicate_store = false;
                    break;
                }

                let child = Rc::new(RefCell::new(RadixBlock::for_blocks(remaining, worker)));
                {
                    let mut parent_ref = parent.borrow_mut();
                    parent_ref.internal = true;
                    parent_ref.children.insert(first_local, child.clone());
                }
                self.update_lookup_for_blocks(worker, remaining, &child);
                duplicate_store = false;
                break;
            };

            let (edge_len, match_len, mismatch) = {
                let child_ref = child.borrow();
                let edge_len = child_ref.state.edge.len();
                let mut mismatch = None;
                let match_len = child_ref
                    .state
                    .edge
                    .iter()
                    .zip(remaining)
                    .take_while(|((local_hash, existing_hash), block)| {
                        if local_hash != &block.tokens_hash {
                            return false;
                        }
                        if existing_hash != &block.block_hash && mismatch.is_none() {
                            mismatch = Some((block.block_hash, *existing_hash));
                        }
                        true
                    })
                    .count();
                (edge_len, match_len, mismatch)
            };

            if let Some((expected, actual)) = mismatch {
                duplicate_store = false;
                tracing::warn!(
                    ?expected,
                    ?actual,
                    "block_hash mismatch: sequence hashes should be uniform across workers"
                );
            }

            if match_len < edge_len {
                if match_len == remaining.len() {
                    let changed = child
                        .borrow_mut()
                        .state
                        .cover_prefix_for_worker(worker, match_len);
                    duplicate_store &= !changed;
                    duplicate_store &= !self.update_lookup_for_blocks(worker, remaining, &child);
                    break;
                }

                self.split_node(&child, match_len);
                child.borrow_mut().state.promote_to_full(worker);
                self.update_lookup_for_blocks(worker, &remaining[..match_len], &child);

                let tail = &remaining[match_len..];
                let tail_node = Rc::new(RefCell::new(RadixBlock::for_blocks(tail, worker)));
                {
                    let mut child_ref = child.borrow_mut();
                    child_ref.internal = true;
                    child_ref
                        .children
                        .insert(tail[0].tokens_hash, tail_node.clone());
                }
                self.update_lookup_for_blocks(worker, tail, &tail_node);
                duplicate_store = false;
                break;
            }

            if child.borrow_mut().state.promote_to_full(worker) {
                duplicate_store = false;
            }
            if self.update_lookup_for_blocks(worker, &remaining[..edge_len], &child) {
                duplicate_store = false;
            }
            remaining = &remaining[edge_len..];
            parent = child;
        }

        if duplicate_store && let Some(counters) = counters {
            counters.inc_warning(EventWarningKind::DuplicateStore);
        }
        Ok(())
    }

    fn try_reuse_parent_edge(
        &mut self,
        worker: WorkerWithDpRank,
        node: &SharedRadixBlock,
        parent_pos: usize,
        blocks: &[KvCacheStoredBlockData],
    ) -> Option<bool> {
        let (reuse_cutoff, append_start) = {
            let node_ref = node.borrow();
            if node_ref.state.suffix_matches_store(parent_pos, blocks) {
                (Some(parent_pos + 1 + blocks.len()), None)
            } else if !node_ref.internal {
                (
                    None,
                    node_ref.state.store_starts_with_suffix(parent_pos, blocks),
                )
            } else {
                (None, None)
            }
        };

        if let Some(cutoff) = reuse_cutoff {
            let changed = node
                .borrow_mut()
                .state
                .cover_prefix_for_worker(worker, cutoff);
            let lookup_changed = self.update_lookup_for_blocks(worker, blocks, node);
            return Some(!changed && !lookup_changed);
        }

        if let Some(append_start) = append_start {
            node.borrow_mut()
                .state
                .append_blocks_to_leaf(worker, &blocks[append_start..]);
            self.update_lookup_for_blocks(worker, blocks, node);
            return Some(false);
        }

        None
    }

    fn split_node(&mut self, node: &SharedRadixBlock, pos: usize) {
        let suffix = {
            let mut node_ref = node.borrow_mut();
            debug_assert!(pos > 0 && pos < node_ref.state.edge.len());

            let suffix_edge = node_ref.state.edge.split_off(pos);
            let suffix_first_local = suffix_edge[0].0;
            for &(_, hash) in &suffix_edge {
                node_ref.state.edge_index.remove(&hash);
            }

            let mut suffix_full = FxHashSet::with_capacity_and_hasher(
                node_ref.state.full_edge_workers.len(),
                FxBuildHasher,
            );
            suffix_full.extend(node_ref.state.full_edge_workers.iter().copied());
            let mut suffix_cutoffs = FxHashMap::with_capacity_and_hasher(
                node_ref.state.worker_cutoffs.len(),
                FxBuildHasher,
            );
            let mut promote_prefix = Vec::new();
            for (&worker, &cutoff) in &node_ref.state.worker_cutoffs {
                if cutoff < pos {
                    continue;
                }
                promote_prefix.push(worker);
                let suffix_cutoff = cutoff - pos;
                if suffix_cutoff > 0 {
                    suffix_cutoffs.insert(worker, suffix_cutoff);
                }
            }
            for worker in promote_prefix {
                node_ref.state.worker_cutoffs.remove(&worker);
                node_ref.state.full_edge_workers.insert(worker);
            }

            let suffix = Rc::new(RefCell::new(RadixBlock {
                state: NodeState {
                    edge_index: NodeState::edge_index_for(&suffix_edge),
                    edge: suffix_edge,
                    worker_cutoffs: suffix_cutoffs,
                    full_edge_workers: suffix_full,
                },
                children: std::mem::take(&mut node_ref.children),
                internal: node_ref.internal,
            }));
            node_ref.children.insert(suffix_first_local, suffix.clone());
            node_ref.internal = true;
            suffix
        };

        let suffix_ref = suffix.borrow();
        for &worker in &suffix_ref.state.full_edge_workers {
            let blocks = suffix_ref.state.edge.iter().map(|&(_, hash)| hash);
            let lookup = self.lookup.get_mut(&worker).unwrap();
            for hash in blocks {
                lookup.insert(hash, suffix.clone());
            }
        }
        for (&worker, &cutoff) in &suffix_ref.state.worker_cutoffs {
            let lookup = self.lookup.get_mut(&worker).unwrap();
            for &(_, hash) in &suffix_ref.state.edge[..cutoff] {
                lookup.insert(hash, suffix.clone());
            }
        }
    }

    fn update_lookup_for_blocks(
        &mut self,
        worker: WorkerWithDpRank,
        blocks: &[KvCacheStoredBlockData],
        node: &SharedRadixBlock,
    ) -> bool {
        let lookup = self.lookup.get_mut(&worker).unwrap();
        let mut changed = false;
        for block in blocks {
            match lookup.insert(block.block_hash, node.clone()) {
                Some(existing) if Rc::ptr_eq(&existing, node) => {}
                _ => changed = true,
            }
        }
        changed
    }

    fn log_missing_parent(
        &self,
        worker: WorkerWithDpRank,
        event_id: u64,
        store: &KvCacheStoreData,
    ) {
        tracing::warn!(
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            event_id,
            parent_hash = ?store.parent_hash,
            num_blocks = store.blocks.len(),
            "Failed to find parent block; skipping store operation"
        );
    }

    fn apply_removed(
        &mut self,
        worker: WorkerWithDpRank,
        remove: KvCacheRemoveData,
        event_id: u64,
    ) -> Result<(), KvCacheEventError> {
        if !self.lookup.contains_key(&worker) {
            return Err(KvCacheEventError::BlockNotFound);
        }
        let mut first_error = None;
        let mut eagerly_removed = FxHashSet::default();

        for block_hash in remove.block_hashes {
            let Some(node) = self
                .lookup
                .get(&worker)
                .and_then(|lookup| lookup.get(&block_hash))
                .cloned()
            else {
                if eagerly_removed.contains(&block_hash) {
                    continue;
                }
                tracing::warn!(
                    worker_id = worker.worker_id,
                    dp_rank = worker.dp_rank,
                    event_id,
                    ?block_hash,
                    "Failed to find block to remove; skipping remove operation"
                );
                first_error.get_or_insert(KvCacheEventError::BlockNotFound);
                continue;
            };

            let outcome = {
                let mut node_ref = node.borrow_mut();
                let Some(&pos) = node_ref.state.edge_index.get(&block_hash) else {
                    first_error.get_or_insert(KvCacheEventError::BlockNotFound);
                    continue;
                };
                // TODO(CORRECTNESS): Invalidate this worker throughout the descendant
                // subtree when a mid-edge removal leaves the node alive for another
                // worker. Otherwise stale descendants can be reused as store parents,
                // reactivated by restoring only the removed block, or emitted by dumps
                // without a valid worker-specific parent.
                let outcome = node_ref.state.remove_worker_at_pos(worker, pos, block_hash);
                node_ref.clear_children_if_unreachable();
                outcome
            };
            let lookup = self.lookup.get_mut(&worker).unwrap();
            for stale_hash in outcome.stale_hashes {
                lookup.remove(&stale_hash);
                eagerly_removed.insert(stale_hash);
            }
        }

        first_error.map_or(Ok(()), Err)
    }

    fn remove_or_clear_worker_blocks(&mut self, worker_id: WorkerId, keep_worker: bool) {
        let workers = self
            .lookup
            .keys()
            .filter(|worker| worker.worker_id == worker_id)
            .copied()
            .collect::<Vec<_>>();

        for worker in workers {
            let Some((worker_key, blocks)) = self.lookup.remove_entry(&worker) else {
                continue;
            };
            let mut seen = FxHashSet::default();
            for node in blocks.into_values() {
                if !seen.insert(Rc::as_ptr(&node)) {
                    continue;
                }
                let mut node_ref = node.borrow_mut();
                node_ref.state.drop_worker(worker);
                node_ref.clear_children_if_unreachable();
            }
            if keep_worker {
                self.lookup.insert(worker_key, FxHashMap::default());
            }
        }
    }

    pub fn remove_worker(&mut self, worker_id: WorkerId) {
        self.remove_or_clear_worker_blocks(worker_id, false);
    }

    pub fn remove_worker_dp_rank(&mut self, worker_id: WorkerId, dp_rank: DpRank) {
        let worker = WorkerWithDpRank { worker_id, dp_rank };
        let Some(blocks) = self.lookup.remove(&worker) else {
            return;
        };
        let mut seen = FxHashSet::default();
        for node in blocks.into_values() {
            if !seen.insert(Rc::as_ptr(&node)) {
                continue;
            }
            let mut node_ref = node.borrow_mut();
            node_ref.state.drop_worker(worker);
            node_ref.clear_children_if_unreachable();
        }
    }

    pub fn clear_all_blocks(&mut self, worker_id: WorkerId) {
        self.remove_or_clear_worker_blocks(worker_id, true);
    }

    pub fn get_workers(&self) -> Vec<WorkerId> {
        let mut workers = self
            .lookup
            .keys()
            .map(|worker| worker.worker_id)
            .collect::<Vec<_>>();
        workers.sort_unstable();
        workers.dedup();
        workers
    }

    pub fn dump_tree_as_events(&self) -> Vec<RouterEvent> {
        let mut events = Vec::new();
        let mut event_id = 0u64;
        let mut queue = self
            .root
            .borrow()
            .children
            .values()
            .cloned()
            .map(|node| (node, None))
            .collect::<VecDeque<_>>();

        while let Some((start_node, parent_hash)) = queue.pop_front() {
            let mut merged_edge = Vec::new();
            let mut current = start_node;

            loop {
                let (full_workers, cutoffs, live_children, can_merge) = {
                    let node = current.borrow();
                    if !node.state.has_any_workers() && node.children.is_empty() {
                        break;
                    }
                    merged_edge.extend_from_slice(&node.state.edge);
                    let live_children = node
                        .children
                        .values()
                        .filter(|child| {
                            let child = child.borrow();
                            child.state.has_any_workers() || !child.children.is_empty()
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    let can_merge =
                        node.state.worker_cutoffs.is_empty() && live_children.len() == 1 && {
                            let child = live_children[0].borrow();
                            child.state.worker_cutoffs.is_empty()
                                && child.state.full_edge_workers == node.state.full_edge_workers
                                && child.state.has_any_workers()
                        };
                    (
                        node.state
                            .full_edge_workers
                            .iter()
                            .copied()
                            .collect::<Vec<_>>(),
                        node.state
                            .worker_cutoffs
                            .iter()
                            .map(|(&worker, &cutoff)| (worker, cutoff))
                            .collect::<Vec<_>>(),
                        live_children,
                        can_merge,
                    )
                };

                if can_merge {
                    current = live_children[0].clone();
                    continue;
                }
                if merged_edge.is_empty() {
                    break;
                }

                let last_hash = merged_edge.last().unwrap().1;

                append_dump_events(
                    &mut events,
                    &mut event_id,
                    parent_hash,
                    &merged_edge,
                    &full_workers,
                    &cutoffs,
                );
                for child in live_children {
                    queue.push_back((child, Some(last_hash)));
                }
                break;
            }
        }

        events
    }

    pub fn current_size(&self) -> usize {
        self.lookup.values().map(FxHashMap::len).sum()
    }

    #[cfg(test)]
    pub(crate) fn tree_size_for_worker(&self, worker: WorkerWithDpRank) -> Option<usize> {
        self.lookup.get(&worker).map(FxHashMap::len)
    }

    #[cfg(test)]
    pub(crate) fn edge_lengths_for_test(&self) -> Vec<usize> {
        let mut queue = VecDeque::from([self.root.clone()]);
        let mut lengths = Vec::new();
        while let Some(node) = queue.pop_front() {
            let node = node.borrow();
            for child in node.children.values() {
                lengths.push(child.borrow().state.edge.len());
                queue.push_back(child.clone());
            }
        }
        lengths.sort_unstable();
        lengths
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::indexer::WorkerKvQueryResponse;
    use crate::test_utils::{create_store_event, make_store_event, snapshot_events};

    #[test]
    fn rejects_self_referencing_store() {
        let mut tree = RadixTree::new();
        tree.apply_event(create_store_event(0, 0, vec![1], None))
            .unwrap();
        let result = tree.apply_event(create_store_event(
            0,
            1,
            vec![1, 2],
            Some(ExternalSequenceBlockHash(100)),
        ));
        assert!(matches!(
            result,
            Err(KvCacheEventError::InvalidBlockSequence)
        ));
    }

    #[test]
    fn linear_tree_dumps_as_one_event() {
        let mut tree = RadixTree::new();
        tree.apply_event(make_store_event(0, &[1, 2, 3, 4]))
            .unwrap();

        assert_eq!(tree.edge_lengths_for_test(), vec![4]);
        assert_eq!(
            snapshot_events(tree.dump_tree_as_events()),
            vec![make_store_event(0, &[1, 2, 3, 4])]
        );
    }

    #[test]
    fn compact_dump_preserves_order_and_replays() {
        let mut tree = RadixTree::new();
        tree.apply_event(make_store_event(1, &[1, 2, 3, 4, 5, 6]))
            .unwrap();
        tree.apply_event(make_store_event(2, &[1, 2, 3])).unwrap();
        tree.apply_event(crate::test_utils::make_store_event_with_parent(
            2,
            &[1, 2, 3],
            &[7, 8],
        ))
        .unwrap();

        let events = tree.dump_tree_as_events();
        let represented_blocks = events
            .iter()
            .map(|event| match &event.event.data {
                KvCacheEventData::Stored(store) => store.blocks.len(),
                _ => 0,
            })
            .sum::<usize>();
        assert_eq!(events.len(), 4);
        assert_eq!(represented_blocks, 11);
        assert_eq!(
            events
                .iter()
                .map(|event| event.event.event_id)
                .collect::<Vec<_>>(),
            (0..events.len() as u64).collect::<Vec<_>>()
        );

        let mut seen = HashSet::new();
        for event in &events {
            let KvCacheEventData::Stored(store) = &event.event.data else {
                panic!("tree dumps must contain only stored events");
            };
            if let Some(parent_hash) = store.parent_hash {
                assert!(
                    seen.contains(&(event.worker_id, event.event.dp_rank, parent_hash)),
                    "dump child appeared before its worker-specific parent"
                );
            }
            for block in &store.blocks {
                seen.insert((event.worker_id, event.event.dp_rank, block.block_hash));
            }
        }

        let mut restored = RadixTree::new();
        for event in events.clone() {
            restored.apply_event(event).unwrap();
        }
        assert_eq!(
            snapshot_events(restored.dump_tree_as_events()),
            snapshot_events(events.clone())
        );

        let mut uncompressed_events = Vec::new();
        let mut uncompressed_event_id = 0;
        for event in &events {
            let KvCacheEventData::Stored(store) = &event.event.data else {
                unreachable!();
            };
            let mut parent_hash = store.parent_hash;
            for block in &store.blocks {
                append_dump_events(
                    &mut uncompressed_events,
                    &mut uncompressed_event_id,
                    parent_hash,
                    &[(block.tokens_hash, block.block_hash)],
                    &[WorkerWithDpRank::new(event.worker_id, event.event.dp_rank)],
                    &[],
                );
                parent_hash = Some(block.block_hash);
            }
        }

        let compact = WorkerKvQueryResponse::TreeDump {
            events,
            last_event_id: 42,
        };
        let uncompressed = WorkerKvQueryResponse::TreeDump {
            events: uncompressed_events,
            last_event_id: 42,
        };
        assert!(
            serde_json::to_vec(&compact).unwrap().len()
                < serde_json::to_vec(&uncompressed).unwrap().len()
        );
    }
}
