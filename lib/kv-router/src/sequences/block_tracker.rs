// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;

use super::compressed_path_arena::{CompressedNodeId, CompressedPathArena};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct BlockNodeMetadata {
    terminal_owners: u32,
    fraction: Option<f64>,
}

impl BlockNodeMetadata {
    fn terminal() -> Self {
        Self {
            terminal_owners: 1,
            fraction: None,
        }
    }

    fn increment_terminal(&mut self) {
        self.terminal_owners = self
            .terminal_owners
            .checked_add(1)
            .expect("block ownership count overflowed");
    }

    fn edge_weight(self, edge_len: usize) -> f64 {
        edge_len as f64 * self.fraction.unwrap_or(1.0)
    }
}

/// The compressed prompt tail plus request-local output hashes.
///
/// This type intentionally does not implement `Clone`. New request owners must
/// be acquired through [`BlockTracker::acquire_prompt`].
#[derive(Debug, Default)]
#[must_use = "request block chains must be retained by a request and released through BlockTracker"]
pub(super) struct RequestBlockChain {
    prompt_tail: Option<CompressedNodeId>,
    prompt_depth: usize,
    output_hashes: Vec<SequenceHash>,
}

impl RequestBlockChain {
    fn new(prompt_tail: Option<CompressedNodeId>, prompt_depth: usize) -> Self {
        Self {
            prompt_tail,
            prompt_depth,
            output_hashes: Vec::new(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct ReleasedPromptPath {
    pub(super) path: Vec<SequenceHash>,
    pub(super) remove_from: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct BlockLoad {
    pub(super) prompt_units: f64,
    pub(super) output_blocks: f64,
}

/// Exact per-worker prompt liveness backed by compressed arena edges.
#[derive(Debug, Default)]
pub(super) struct BlockTracker {
    arena: CompressedPathArena<BlockNodeMetadata>,
    prompt_total: f64,
    // Output hashes remain outside the prompt topology. They are unique per
    // request, but retaining the map preserves active-hash diagnostics and
    // fractional load accounting.
    output_blocks: FxHashMap<SequenceHash, f64>,
    output_total: f64,
}

impl BlockTracker {
    /// Acquire a request terminal and return the first block that became newly
    /// present for this worker.
    ///
    /// Each [`SequenceHash`] must identify one prefix ancestry and depth. This
    /// tracker trusts callers to preserve that lineage and does not validate
    /// the hash recurrence in production. It also assumes distinct live
    /// lineages have distinct hashes; collision detection and recovery are out
    /// of scope. That collision scope is local to one worker and DP-rank
    /// tracker because each owns an independent [`BlockTracker`].
    pub(super) fn acquire_prompt(
        &mut self,
        sequence: &[SequenceHash],
    ) -> (RequestBlockChain, Option<usize>) {
        let Some(&first_hash) = sequence.first() else {
            return (RequestBlockChain::default(), None);
        };

        let Some(mut node_id) = self.arena.root(first_hash) else {
            let tail = self
                .arena
                .insert_root(sequence.to_vec(), BlockNodeMetadata::terminal());
            self.prompt_total += sequence.len() as f64;
            return (RequestBlockChain::new(Some(tail), sequence.len()), Some(0));
        };

        let mut path_pos = 0;
        loop {
            let (edge_len, match_len) = {
                let node = &self.arena.nodes[node_id];
                let remaining = &sequence[path_pos..];
                let probe_len = node.edge.len().min(remaining.len());
                debug_assert!(probe_len > 0, "compressed path probe cannot be empty");
                // A SequenceHash commits to its full prefix, so equality at
                // the probe boundary implies equality before it under the
                // trusted lineage contract documented above.
                let match_len = if node.edge[probe_len - 1] == remaining[probe_len - 1] {
                    probe_len
                } else {
                    node.edge
                        .iter()
                        .zip(remaining)
                        .position(|(left, right)| left != right)
                        .unwrap_or(probe_len)
                };
                (node.edge.len(), match_len)
            };
            assert!(match_len > 0, "compressed path selected a mismatched edge");

            if match_len < edge_len {
                let fraction = self.arena.nodes[node_id].metadata.fraction;
                let split_depth = path_pos + match_len;
                let prompt_ends_at_split = split_depth == sequence.len();
                let prefix_id = self.arena.split_keep_suffix(
                    node_id,
                    match_len,
                    BlockNodeMetadata {
                        terminal_owners: u32::from(prompt_ends_at_split),
                        fraction,
                    },
                );

                if prompt_ends_at_split {
                    return (
                        RequestBlockChain::new(Some(prefix_id), sequence.len()),
                        None,
                    );
                }

                let tail = self.arena.insert_child(
                    prefix_id,
                    sequence[split_depth..].to_vec(),
                    BlockNodeMetadata::terminal(),
                );
                self.prompt_total += (sequence.len() - split_depth) as f64;
                return (
                    RequestBlockChain::new(Some(tail), sequence.len()),
                    Some(split_depth),
                );
            }

            path_pos += edge_len;
            if path_pos == sequence.len() {
                self.arena.nodes[node_id].metadata.increment_terminal();
                return (RequestBlockChain::new(Some(node_id), sequence.len()), None);
            }

            let next_hash = sequence[path_pos];
            if let Some(next) = self.arena.nodes[node_id].children.get(&next_hash).copied() {
                node_id = next;
                continue;
            }

            let tail = self.arena.insert_child(
                node_id,
                sequence[path_pos..].to_vec(),
                BlockNodeMetadata::terminal(),
            );
            self.prompt_total += (sequence.len() - path_pos) as f64;
            return (
                RequestBlockChain::new(Some(tail), sequence.len()),
                Some(path_pos),
            );
        }
    }

    /// Append a unique output block without adding it to the prompt trie.
    pub(super) fn append_output(&mut self, chain: &mut RequestBlockChain, hash: SequenceHash) {
        #[cfg(any(test, debug_assertions))]
        debug_assert!(
            !self.contains_block(&hash),
            "random output hash unexpectedly collided with a live block"
        );
        assert!(
            self.output_blocks.insert(hash, 1.0).is_none(),
            "output block hash unexpectedly became live twice"
        );
        self.output_total += 1.0;
        chain.output_hashes.push(hash);
    }

    /// Release outputs and the request terminal, returning the prompt suffix
    /// that became absent from the worker.
    pub(super) fn release(&mut self, chain: RequestBlockChain) -> Option<ReleasedPromptPath> {
        for hash in chain.output_hashes {
            let weight = self
                .output_blocks
                .remove(&hash)
                .expect("request output hash is missing from active bookkeeping");
            self.output_total -= weight;
        }
        if self.output_blocks.is_empty() {
            self.output_total = 0.0;
        }

        let Some(tail) = chain.prompt_tail else {
            assert_eq!(chain.prompt_depth, 0, "empty prompt retained a depth");
            return None;
        };

        let terminal = &mut self.arena.nodes[tail].metadata.terminal_owners;
        *terminal = terminal
            .checked_sub(1)
            .expect("request terminal ownership underflowed");

        let tail_remains_live = {
            let tail_node = &self.arena.nodes[tail];
            tail_node.metadata.terminal_owners != 0 || !tail_node.children.is_empty()
        };
        if tail_remains_live {
            return None;
        }

        let path_ids = self.arena.path_from_root(tail);
        let mut path = Vec::with_capacity(chain.prompt_depth);
        for node_id in &path_ids {
            path.extend_from_slice(&self.arena.nodes[*node_id].edge);
        }
        assert_eq!(
            path.len(),
            chain.prompt_depth,
            "request terminal depth differs from its compressed path"
        );

        let mut removed_hashes = 0;
        let mut current = Some(tail);
        while let Some(node_id) = current {
            let node = &self.arena.nodes[node_id];
            if node.metadata.terminal_owners != 0 || !node.children.is_empty() {
                break;
            }
            let parent = node.parent;
            let removed = self.arena.remove_leaf(node_id);
            let edge_len = removed.edge.len();
            removed_hashes += edge_len;
            self.prompt_total -= removed.metadata.edge_weight(edge_len);
            current = parent;
        }

        if self.arena.nodes.is_empty() {
            self.prompt_total = 0.0;
        }

        debug_assert!(removed_hashes > 0, "dead prompt tail removed no hashes");
        let remove_from = chain
            .prompt_depth
            .checked_sub(removed_hashes)
            .expect("removed prompt suffix exceeds the request prompt");
        Some(ReleasedPromptPath { path, remove_from })
    }

    /// Mark the request's structurally exclusive suffix as fractional.
    pub(super) fn set_unique_suffix_fractional(
        &mut self,
        chain: &RequestBlockChain,
        fraction: f64,
    ) {
        for hash in &chain.output_hashes {
            let weight = self
                .output_blocks
                .get_mut(hash)
                .expect("request output hash is missing from active bookkeeping");
            let old_weight = std::mem::replace(weight, fraction);
            self.output_total += fraction - old_weight;
        }

        let mut current = chain.prompt_tail;
        while let Some(node_id) = current {
            let (parent, incoming, edge_len, old_fraction) = {
                let node = &self.arena.nodes[node_id];
                (
                    node.parent,
                    node.metadata.terminal_owners as usize + node.children.len(),
                    node.edge.len(),
                    node.metadata.fraction.unwrap_or(1.0),
                )
            };
            if incoming != 1 {
                break;
            }
            self.prompt_total += edge_len as f64 * (fraction - old_fraction);
            self.arena.nodes[node_id].metadata.fraction = Some(fraction);
            current = parent;
        }
    }

    pub(super) fn active_blocks(&self) -> usize {
        let load = self.load();
        super::estimate_physical_active_blocks(load.prompt_units, load.output_blocks, 1)
    }

    pub(super) fn load(&self) -> BlockLoad {
        BlockLoad {
            prompt_units: self.prompt_total,
            output_blocks: self.output_total,
        }
    }

    #[cfg(any(test, debug_assertions))]
    pub(super) fn contains_block(&self, hash: &SequenceHash) -> bool {
        self.output_blocks.contains_key(hash)
            || self
                .arena
                .nodes
                .values()
                .any(|node| node.edge.contains(hash))
    }

    #[cfg(any(test, debug_assertions))]
    pub(super) fn assert_consistent<'a>(
        &self,
        chains: impl IntoIterator<Item = &'a RequestBlockChain>,
    ) {
        use rustc_hash::FxHashSet;

        self.arena.assert_topology();
        let mut expected_terminals = FxHashMap::<CompressedNodeId, u32>::default();
        let mut expected_outputs = FxHashSet::default();
        for chain in chains {
            match chain.prompt_tail {
                Some(tail) => {
                    let path = self.arena.path_from_root(tail);
                    let depth: usize = path
                        .iter()
                        .map(|node_id| self.arena.nodes[*node_id].edge.len())
                        .sum();
                    assert_eq!(depth, chain.prompt_depth, "request prompt depth mismatch");
                    *expected_terminals.entry(tail).or_default() += 1;
                }
                None => assert_eq!(chain.prompt_depth, 0, "empty prompt retained a depth"),
            }
            for &hash in &chain.output_hashes {
                assert!(
                    expected_outputs.insert(hash),
                    "output hash is owned by multiple requests"
                );
            }
        }

        let mut prompt_hashes = FxHashSet::default();
        for (node_id, node) in &self.arena.nodes {
            assert_eq!(
                node.metadata.terminal_owners,
                expected_terminals.get(&node_id).copied().unwrap_or(0),
                "compressed terminal ownership mismatch"
            );
            assert!(
                node.metadata.terminal_owners != 0 || !node.children.is_empty(),
                "unowned compressed leaf remained live"
            );
            for &hash in &node.edge {
                assert!(prompt_hashes.insert(hash), "prompt hash appears twice");
            }
        }
        assert_eq!(
            expected_outputs,
            self.output_blocks.keys().copied().collect(),
            "output bookkeeping differs from request ownership"
        );

        let expected_prompt_total = self
            .arena
            .nodes
            .values()
            .map(|node| node.metadata.edge_weight(node.edge.len()))
            .sum::<f64>();
        let expected_output_total = self.output_blocks.values().sum::<f64>();
        for (label, actual, expected) in [
            ("prompt", self.prompt_total, expected_prompt_total),
            ("output", self.output_total, expected_output_total),
        ] {
            let scale = actual.abs().max(expected.abs()).max(1.0);
            assert!(
                (actual - expected).abs() <= 1e-9 * scale,
                "{label} running total differs from reconstructed weight: {actual} != {expected}"
            );
        }
        if self.arena.nodes.is_empty() {
            assert_eq!(self.prompt_total, 0.0, "drained prompt total is not zero");
        }
        if self.output_blocks.is_empty() {
            assert_eq!(self.output_total, 0.0, "drained output total is not zero");
        }
        assert_eq!(
            self.active_blocks(),
            (expected_prompt_total + expected_output_total).round() as usize,
            "constant-time active block count differs from reconstructed weight"
        );
    }

    #[cfg(test)]
    pub(super) fn active_hashes(&self) -> impl Iterator<Item = SequenceHash> + '_ {
        let mut hashes = Vec::with_capacity(self.arena.hash_count() + self.output_blocks.len());
        for node in self.arena.nodes.values() {
            hashes.extend_from_slice(&node.edge);
        }
        hashes.extend(self.output_blocks.keys().copied());
        hashes.into_iter()
    }

    #[cfg(test)]
    pub(super) fn prompt_hashes(
        &self,
        chain: &RequestBlockChain,
    ) -> impl Iterator<Item = SequenceHash> {
        let mut hashes = Vec::with_capacity(chain.prompt_depth);
        if let Some(tail) = chain.prompt_tail {
            for node_id in self.arena.path_from_root(tail) {
                hashes.extend_from_slice(&self.arena.nodes[node_id].edge);
            }
        }
        hashes.into_iter()
    }

    #[cfg(test)]
    fn node_id_for(&self, hash: SequenceHash) -> CompressedNodeId {
        self.arena
            .nodes
            .iter()
            .find_map(|(node_id, node)| node.edge.contains(&hash).then_some(node_id))
            .expect("expected live block hash")
    }

    #[cfg(test)]
    fn incoming_for(&self, hash: SequenceHash) -> u32 {
        let node = &self.arena.nodes[self.node_id_for(hash)];
        node.metadata.terminal_owners + node.children.len() as u32
    }

    #[cfg(test)]
    fn fraction_for(&self, hash: SequenceHash) -> Option<f64> {
        let node = &self.arena.nodes[self.node_id_for(hash)];
        node.metadata.fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashSet;

    fn released(path: &[SequenceHash], remove_from: usize) -> Option<ReleasedPromptPath> {
        Some(ReleasedPromptPath {
            path: path.to_vec(),
            remove_from,
        })
    }

    fn expected_release(
        blocks: &[SequenceHash],
        prompt_depth: usize,
        counts: &FxHashMap<SequenceHash, usize>,
    ) -> Option<ReleasedPromptPath> {
        let remove_from = blocks[..prompt_depth]
            .iter()
            .position(|hash| counts[hash] == 1)?;
        released(&blocks[..prompt_depth], remove_from)
    }

    #[test]
    fn duplicate_prompt_tracks_terminal_owners() {
        let mut tracker = BlockTracker::default();
        let (first, first_new) = tracker.acquire_prompt(&[1, 2, 3]);
        let (second, second_new) = tracker.acquire_prompt(&[1, 2, 3]);

        assert_eq!(first_new, Some(0));
        assert_eq!(second_new, None);
        assert_eq!(tracker.incoming_for(3), 2);
        tracker.assert_consistent([&first, &second]);
        assert_eq!(tracker.release(first), None);
        assert_eq!(tracker.release(second), released(&[1, 2, 3], 0));
        tracker.assert_consistent(std::iter::empty());
    }

    #[test]
    fn shorter_prompt_splits_without_invalidating_longer_handle() {
        let mut tracker = BlockTracker::default();
        let (longer, _) = tracker.acquire_prompt(&[1, 2, 3, 4]);
        let old_tail = longer.prompt_tail.unwrap();
        let (shorter, first_new) = tracker.acquire_prompt(&[1, 2]);

        assert_eq!(first_new, None);
        assert_eq!(longer.prompt_tail, Some(old_tail));
        assert!(tracker.arena.nodes.contains_key(old_tail));
        tracker.assert_consistent([&longer, &shorter]);
        assert_eq!(tracker.release(shorter), None);
        assert!(tracker.arena.nodes.contains_key(old_tail));
        assert_eq!(tracker.release(longer), released(&[1, 2, 3, 4], 0));
    }

    #[test]
    fn branch_release_preserves_split_and_survivor_handle() {
        let mut tracker = BlockTracker::default();
        let (left, _) = tracker.acquire_prompt(&[1, 2, 3]);
        let (right, first_new) = tracker.acquire_prompt(&[1, 2, 4]);
        let right_tail = right.prompt_tail.unwrap();

        assert_eq!(first_new, Some(2));
        assert_eq!(tracker.release(left), released(&[1, 2, 3], 2));
        assert_eq!(tracker.arena.nodes.len(), 2);
        assert!(tracker.arena.nodes.contains_key(right_tail));
        tracker.assert_consistent([&right]);
        assert_eq!(tracker.release(right), released(&[1, 2, 4], 0));
    }

    #[test]
    fn longer_prompt_reports_only_the_new_suffix() {
        let mut tracker = BlockTracker::default();
        let (shorter, _) = tracker.acquire_prompt(&[1, 2]);
        let (longer, first_new) = tracker.acquire_prompt(&[1, 2, 3, 4]);

        assert_eq!(first_new, Some(2));
        assert_eq!(tracker.active_blocks(), 4);
        assert_eq!(tracker.release(longer), released(&[1, 2, 3, 4], 2));
        assert_eq!(tracker.release(shorter), released(&[1, 2], 0));
    }

    #[test]
    fn output_blocks_stay_outside_prompt_arena() {
        let mut tracker = BlockTracker::default();
        let (mut chain, _) = tracker.acquire_prompt(&[1, 2]);
        let prompt_nodes = tracker.arena.nodes.len();

        tracker.append_output(&mut chain, 42);
        assert_eq!(tracker.arena.nodes.len(), prompt_nodes);
        assert_eq!(tracker.active_blocks(), 3);
        tracker.assert_consistent([&chain]);
        assert_eq!(tracker.release(chain), released(&[1, 2], 0));
        assert!(tracker.output_blocks.is_empty());
    }

    #[test]
    fn fractional_unique_suffix_preserves_shared_prefix() {
        let mut tracker = BlockTracker::default();
        let (longer, _) = tracker.acquire_prompt(&[1, 2, 3]);
        let (shared, _) = tracker.acquire_prompt(&[1, 2]);

        tracker.set_unique_suffix_fractional(&longer, 0.5);
        assert_eq!(tracker.fraction_for(3), Some(0.5));
        assert_eq!(tracker.fraction_for(1), None);
        assert_eq!(tracker.active_blocks(), 3);
        assert_eq!(tracker.release(longer), released(&[1, 2, 3], 2));
        assert_eq!(tracker.active_blocks(), 2);
        assert_eq!(tracker.release(shared), released(&[1, 2], 0));
    }

    #[test]
    fn fraction_mismatch_preserves_split_and_totals() {
        let mut tracker = BlockTracker::default();
        let (longer, _) = tracker.acquire_prompt(&[1, 2, 3, 4]);
        let longer_tail = longer.prompt_tail.unwrap();
        let (shorter, _) = tracker.acquire_prompt(&[1, 2]);

        tracker.set_unique_suffix_fractional(&longer, 0.5);
        assert_eq!(tracker.prompt_total, 3.0);
        assert_eq!(tracker.active_blocks(), 3);

        assert_eq!(tracker.release(shorter), None);
        assert_eq!(tracker.arena.nodes.len(), 2);
        assert!(tracker.arena.nodes.contains_key(longer_tail));
        assert_eq!(tracker.prompt_total, 3.0);
        tracker.assert_consistent([&longer]);

        assert_eq!(tracker.release(longer), released(&[1, 2, 3, 4], 0));
        assert_eq!(tracker.prompt_total, 0.0);
        assert_eq!(tracker.active_blocks(), 0);
    }

    #[test]
    fn equal_fractional_edges_remain_split_and_preserve_child_handle() {
        let mut tracker = BlockTracker::default();
        let (longer, _) = tracker.acquire_prompt(&[1, 2, 3, 4]);
        let longer_tail = longer.prompt_tail.unwrap();
        tracker.set_unique_suffix_fractional(&longer, 0.5);
        let (shorter, _) = tracker.acquire_prompt(&[1, 2]);

        assert_eq!(tracker.arena.nodes.len(), 2);
        assert_eq!(tracker.prompt_total, 2.0);
        assert_eq!(tracker.release(shorter), None);
        assert_eq!(tracker.arena.nodes.len(), 2);
        assert!(tracker.arena.nodes.contains_key(longer_tail));
        assert_eq!(tracker.prompt_total, 2.0);
        tracker.assert_consistent([&longer]);

        assert_eq!(tracker.release(longer), released(&[1, 2, 3, 4], 0));
        assert_eq!(tracker.prompt_total, 0.0);
    }

    #[test]
    fn running_totals_follow_prompt_output_and_fraction_lifecycle() {
        let mut tracker = BlockTracker::default();
        let (mut left, _) = tracker.acquire_prompt(&[1, 2, 3]);
        let (right, _) = tracker.acquire_prompt(&[1, 2, 4]);
        let (duplicate_left, duplicate_new) = tracker.acquire_prompt(&[1, 2, 3]);

        assert_eq!(duplicate_new, None);
        assert_eq!(tracker.prompt_total, 4.0);
        assert_eq!(tracker.output_total, 0.0);
        assert_eq!(tracker.release(duplicate_left), None);
        assert_eq!(tracker.prompt_total, 4.0);
        tracker.append_output(&mut left, 42);
        assert_eq!(tracker.output_total, 1.0);
        assert_eq!(tracker.active_blocks(), 5);

        tracker.set_unique_suffix_fractional(&left, 0.5);
        assert_eq!(tracker.prompt_total, 3.5);
        assert_eq!(tracker.output_total, 0.5);
        assert_eq!(tracker.active_blocks(), 4);

        tracker.set_unique_suffix_fractional(&left, 0.25);
        assert_eq!(tracker.prompt_total, 3.25);
        assert_eq!(tracker.output_total, 0.25);
        assert_eq!(tracker.active_blocks(), 4);
        tracker.assert_consistent([&left, &right]);

        assert_eq!(tracker.release(left), released(&[1, 2, 3], 2));
        assert_eq!(tracker.prompt_total, 3.0);
        assert_eq!(tracker.output_total, 0.0);
        assert_eq!(tracker.active_blocks(), 3);
        tracker.assert_consistent([&right]);

        assert_eq!(tracker.release(right), released(&[1, 2, 4], 0));
        assert_eq!(tracker.prompt_total, 0.0);
        assert_eq!(tracker.output_total, 0.0);
        assert_eq!(tracker.active_blocks(), 0);
    }

    #[test]
    fn output_only_chain_releases_without_membership_delta() {
        let mut tracker = BlockTracker::default();
        let (mut chain, first_new) = tracker.acquire_prompt(&[]);
        tracker.append_output(&mut chain, 42);
        assert_eq!(first_new, None);
        assert_eq!(tracker.release(chain), None);
        assert_eq!(tracker.active_blocks(), 0);
    }

    #[test]
    fn randomized_lifecycle_matches_reference_model() {
        const SLOTS: usize = 32;
        const STEPS: usize = 10_000;
        let prompts = [
            vec![1, 2, 3, 4],
            vec![1, 2, 3, 5],
            vec![1, 2, 6],
            vec![1, 7],
            vec![8, 9, 10],
            vec![8, 9, 11],
            vec![12],
        ];
        let mut rng = fastrand::Rng::with_seed(0x5eed_51a7);
        let mut tracker = BlockTracker::default();
        let mut chains = (0..SLOTS).map(|_| None).collect::<Vec<_>>();
        let mut reference = (0..SLOTS)
            .map(|_| None::<(Vec<SequenceHash>, usize)>)
            .collect::<Vec<_>>();
        let mut counts = FxHashMap::<SequenceHash, usize>::default();
        let mut next_output = 1_000_000_u64;

        for _ in 0..STEPS {
            let slot = rng.usize(0..SLOTS);
            match (&mut chains[slot], &mut reference[slot]) {
                (chain @ None, reference @ None) => {
                    let prompt = &prompts[rng.usize(0..prompts.len())];
                    let expected_first_new =
                        prompt.iter().position(|hash| !counts.contains_key(hash));
                    let (new_chain, first_new) = tracker.acquire_prompt(prompt);
                    assert_eq!(first_new, expected_first_new);
                    for &hash in prompt {
                        *counts.entry(hash).or_default() += 1;
                    }
                    *chain = Some(new_chain);
                    *reference = Some((prompt.clone(), prompt.len()));
                }
                (Some(chain), Some((blocks, _))) if rng.f64() < 0.35 => {
                    let hash = next_output;
                    next_output += 1;
                    tracker.append_output(chain, hash);
                    blocks.push(hash);
                    counts.insert(hash, 1);
                }
                (chain @ Some(_), reference @ Some(_)) => {
                    let chain = chain.take().unwrap();
                    let (blocks, prompt_depth) = reference.take().unwrap();
                    assert_eq!(
                        tracker.release(chain),
                        expected_release(&blocks, prompt_depth, &counts)
                    );
                    for hash in blocks {
                        let count = counts.get_mut(&hash).unwrap();
                        *count -= 1;
                        if *count == 0 {
                            counts.remove(&hash);
                        }
                    }
                }
                _ => unreachable!(),
            }

            tracker.assert_consistent(chains.iter().filter_map(Option::as_ref));
            assert_eq!(
                tracker.active_hashes().collect::<FxHashSet<_>>(),
                counts.keys().copied().collect()
            );
        }

        for slot in 0..SLOTS {
            if let Some(chain) = chains[slot].take() {
                let (blocks, prompt_depth) = reference[slot].take().unwrap();
                assert_eq!(
                    tracker.release(chain),
                    expected_release(&blocks, prompt_depth, &counts)
                );
                for hash in blocks {
                    let count = counts.get_mut(&hash).unwrap();
                    *count -= 1;
                    if *count == 0 {
                        counts.remove(&hash);
                    }
                }
            }
        }
        tracker.assert_consistent(std::iter::empty());
        assert!(counts.is_empty());
    }

    #[test]
    #[should_panic(expected = "block ownership count overflowed")]
    fn ownership_count_overflow_panics() {
        let mut tracker = BlockTracker::default();
        let (chain, _) = tracker.acquire_prompt(&[1]);
        let node_id = tracker.node_id_for(1);
        tracker.arena.nodes[node_id].metadata.terminal_owners = u32::MAX;
        let _ = tracker.acquire_prompt(&[1]);
        drop(chain);
    }

    #[test]
    fn long_chain_release_is_iterative() {
        const DEPTH: usize = 65_536;
        let sequence = (1..=DEPTH as u64).collect::<Vec<_>>();
        let mut tracker = BlockTracker::default();
        let (chain, first_new) = tracker.acquire_prompt(&sequence);

        assert_eq!(first_new, Some(0));
        assert_eq!(tracker.arena.nodes.len(), 1);
        assert_eq!(tracker.active_blocks(), DEPTH);
        assert_eq!(tracker.release(chain), released(&sequence, 0));
        assert_eq!(tracker.active_blocks(), 0);
    }
}
