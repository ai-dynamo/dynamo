// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV Cache Sequence Management for LLM Inference
//!
//! This module provides efficient management of token sequences and their associated KV cache blocks
//! for distributed LLM inference. It implements a shared block system where multiple requests can
//! reuse the same KV cache blocks for common token prefixes, significantly reducing memory usage.
//!
//! # Key Components
//!
//! - [`ActiveSequences`]: Per-worker sequence manager that tracks active requests and their
//!   token sequences, managing shared KV cache blocks efficiently.
//!
//! # Architecture
//!
//! The system uses a block-based approach where token sequences are divided into fixed-size blocks.
//! Each block is identified by a hash of its contents, allowing for deduplication when multiple
//! requests share common prefixes (e.g., system prompts, few-shot examples).

use dynamo_tokens::SequenceHash;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;
use uuid::Uuid;

#[cfg(test)]
use rustc_hash::FxHashSet;

use super::block_tracker::BlockTracker;
use super::prefill_tracker::{PrefillLoadState, PrefillLoadTracker};
use super::prompt_registry::WorkerLoadSnapshot;
use crate::protocols::PrefillLoadHint;

/// Duration after which stale requests may be expired (5 minutes).
const EXPIRY_DURATION: Duration = Duration::from_secs(300);

/// How often we *check* for stale requests (30 seconds). This is not
/// the expiration time, that is EXPIRY_DURATION.
const CHECK_EXPIRY_FREQUENCY: Duration = Duration::from_secs(30);

// TODO: use the common request_id if it exists in the repo
pub type RequestId = String;

#[derive(Debug)]
pub(super) struct RequestState {
    prompt_blocks: Vec<(SequenceHash, Arc<()>)>,
    selector_prompt_blocks: Vec<(SequenceHash, Arc<()>)>,
    output_blocks: Vec<(SequenceHash, Arc<()>)>,
    selector_output_blocks: Vec<(SequenceHash, Arc<()>)>,
    staged_selector_tracking: bool,
    started_at: Instant,
    expected_output_tokens: Option<u32>,
}

impl RequestState {
    fn all_blocks(&self) -> impl Iterator<Item = &(SequenceHash, Arc<()>)> {
        self.prompt_blocks.iter().chain(self.output_blocks.iter())
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct PromptMembershipUpdates {
    pub active: PromptMembershipDelta,
    pub selector: PromptMembershipDelta,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct PromptMembershipStore {
    pub parent: Option<SequenceHash>,
    pub hashes: Vec<SequenceHash>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct PromptMembershipRemove {
    pub hashes: Vec<SequenceHash>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct PromptMembershipDelta {
    pub stores: Vec<PromptMembershipStore>,
    pub removes: Vec<PromptMembershipRemove>,
}

impl PromptMembershipDelta {
    fn extend(&mut self, other: Self) {
        self.stores.extend(other.stores);
        self.removes.extend(other.removes);
    }

    fn push_store(&mut self, parent: Option<SequenceHash>, hashes: Vec<SequenceHash>) {
        if hashes.is_empty() {
            return;
        }
        self.stores.push(PromptMembershipStore { parent, hashes });
    }

    fn push_remove(&mut self, hashes: Vec<SequenceHash>) {
        if hashes.is_empty() {
            return;
        }
        self.removes.push(PromptMembershipRemove { hashes });
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct SequenceMutationOutcome {
    pub membership_delta: PromptMembershipDelta,
    pub selector_membership_delta: PromptMembershipDelta,
    pub expired_request_ids: HashSet<RequestId>,
}

/// A multi-request sequence manager that handles multiple active sequences with shared KV cache
#[derive(Debug)]
pub struct ActiveSequences {
    requests: HashMap<RequestId, RequestState>,
    prefill: PrefillLoadTracker,
    blocks: BlockTracker,
    selector_blocks: BlockTracker,
    block_size: usize,
    last_expiry_check_time: Instant,
}

impl ActiveSequences {
    /// Create a new SharedSequenceManager instance
    pub(super) fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be greater than 0");

        Self {
            requests: HashMap::new(),
            prefill: PrefillLoadTracker::default(),
            blocks: BlockTracker::default(),
            selector_blocks: BlockTracker::default(),
            block_size,
            last_expiry_check_time: Instant::now(),
        }
    }

    #[cfg(any(test, debug_assertions))]
    fn assert_consistent(&self) {
        self.prefill.assert_consistent();
        let active_prefills: HashSet<RequestId> = self.prefill.prefills.keys().cloned().collect();
        let active_requests: HashSet<RequestId> = self.requests.keys().cloned().collect();
        assert!(
            active_prefills.is_subset(&active_requests),
            "prefill tracker cannot reference missing request state",
        );
        assert!(
            self.blocks
                .fractional_blocks
                .keys()
                .all(|hash| self.blocks.unique_blocks.contains_key(hash)),
            "fractional_blocks cannot reference non-active blocks",
        );
        assert!(
            self.selector_blocks
                .fractional_blocks
                .keys()
                .all(|hash| self.selector_blocks.unique_blocks.contains_key(hash)),
            "selector fractional_blocks cannot reference non-active blocks",
        );
        for request in self.requests.values() {
            if request.staged_selector_tracking {
                assert!(
                    request.selector_prompt_blocks.len() <= request.prompt_blocks.len(),
                    "staged selector prompt cannot be longer than the full prompt",
                );
            } else {
                assert!(request.selector_prompt_blocks.is_empty());
                assert!(request.selector_output_blocks.is_empty());
            }
        }
    }

    #[inline]
    fn validate_state(&self) {
        #[cfg(any(test, debug_assertions))]
        self.assert_consistent();
    }

    pub(super) fn active_blocks(&self) -> usize {
        self.blocks.active_blocks()
    }

    pub(super) fn selector_decode_blocks(&self) -> usize {
        self.selector_blocks.active_blocks()
    }

    #[cfg(test)]
    pub(super) fn active_tokens(&self, decay_now: Instant) -> usize {
        self.prefill.snapshot().active_tokens_at(decay_now)
    }

    /// Add a new request with optional prompt-token load accounting.
    /// Returns block membership transitions plus any expired request IDs removed during cleanup.
    pub(super) fn add_request_with_prefill_tracking(
        &mut self,
        request_id: RequestId,
        token_sequence: Option<Vec<SequenceHash>>,
        expected_output_tokens: Option<u32>,
        track_prefill_tokens: bool,
        prefill_load_hint: Option<PrefillLoadHint>,
        decay_now: Instant,
    ) -> SequenceMutationOutcome {
        if self.requests.contains_key(&request_id) {
            tracing::error!("Request {request_id} is already active. Ignoring duplicate add.");
            return SequenceMutationOutcome::default();
        }

        let mut outcome = self.force_expiry();
        let started_at = Instant::now();

        let sequence = token_sequence.unwrap_or_default();
        let (prompt_blocks, membership_delta) =
            Self::touch_prompt_chain(&mut self.blocks, &sequence, None);
        outcome.membership_delta.extend(membership_delta);

        // Aggregated routing tracks both prompt-side prefill load and active blocks. While
        // prefill is in progress, only its already-cached prefix belongs in the selector's
        // decode term; the uncached suffix is represented by the prefill term and is promoted
        // when prefill completes. Other configurations bypass this tracker and retain the
        // legacy full active-block projection.
        let staged_selector_tracking = track_prefill_tokens && !sequence.is_empty();
        let effective_prefill_tokens = prefill_load_hint
            .map(|hint| hint.initial_effective_prefill_tokens)
            .unwrap_or(0);
        let selector_prompt_len = if staged_selector_tracking {
            // `sequence` contains only complete prompt blocks. Effective prefill tokens are
            // the uncached complete blocks plus any partial final block, so integer division
            // recovers the uncached complete-block count without adding replica wire fields.
            sequence
                .len()
                .saturating_sub(effective_prefill_tokens / self.block_size)
        } else {
            0
        };
        let (selector_prompt_blocks, selector_membership_delta) = Self::touch_prompt_chain(
            &mut self.selector_blocks,
            &sequence[..selector_prompt_len],
            None,
        );
        outcome
            .selector_membership_delta
            .extend(selector_membership_delta);

        let prefill = if track_prefill_tokens {
            prefill_load_hint.and_then(|hint| {
                (hint.initial_effective_prefill_tokens > 0).then_some(PrefillLoadState {
                    initial_effective_prefill_tokens: hint.initial_effective_prefill_tokens,
                    expected_prefill_duration: hint.expected_prefill_duration,
                })
            })
        } else {
            None
        };

        self.requests.insert(
            request_id.clone(),
            RequestState {
                prompt_blocks,
                selector_prompt_blocks,
                output_blocks: Vec::new(),
                selector_output_blocks: Vec::new(),
                staged_selector_tracking,
                started_at,
                expected_output_tokens,
            },
        );

        if let Some(prefill) = prefill {
            self.prefill.insert(&request_id, prefill, decay_now);
        }

        self.validate_state();
        outcome
    }

    /// Mark prefill as completed for a request, removing it from prompt-load tracking.
    pub(super) fn mark_prefill_completed(
        &mut self,
        request_id: &RequestId,
        decay_now: Instant,
    ) -> PromptMembershipDelta {
        let _ = self.prefill.remove(request_id, decay_now);
        let Some(request_state) = self.requests.get_mut(request_id) else {
            self.validate_state();
            return PromptMembershipDelta::default();
        };
        if !request_state.staged_selector_tracking {
            self.validate_state();
            return PromptMembershipDelta::default();
        }

        let promoted_from = request_state.selector_prompt_blocks.len();
        let parent = promoted_from
            .checked_sub(1)
            .map(|idx| request_state.prompt_blocks[idx].0);
        let promoted_hashes: Vec<_> = request_state.prompt_blocks[promoted_from..]
            .iter()
            .map(|(hash, _)| *hash)
            .collect();
        let (promoted_blocks, selector_delta) =
            Self::touch_prompt_chain(&mut self.selector_blocks, &promoted_hashes, parent);
        request_state.selector_prompt_blocks.extend(promoted_blocks);
        self.validate_state();
        selector_delta
    }

    /// Free all blocks associated with a request.
    ///
    /// This removes any remaining prefill load without promoting its uncached suffix, since all
    /// selector and active-block references are removed as part of the same operation.
    pub(super) fn free(
        &mut self,
        request_id: &RequestId,
        decay_now: Instant,
    ) -> PromptMembershipUpdates {
        let _ = self.prefill.remove(request_id, decay_now);

        let Some(request_state) = self.requests.remove(request_id) else {
            tracing::warn!("Trying to free non-existent request {request_id}");
            return PromptMembershipUpdates::default();
        };

        let _ = request_state.expected_output_tokens;
        let active_delta =
            Self::release_prompt_chain(&mut self.blocks, request_state.prompt_blocks);
        let selector_delta = Self::release_prompt_chain(
            &mut self.selector_blocks,
            request_state.selector_prompt_blocks,
        );

        for (block_hash, rc) in request_state.output_blocks {
            drop(rc);
            self.blocks.try_remove_block(&block_hash);
        }
        for (block_hash, rc) in request_state.selector_output_blocks {
            drop(rc);
            self.selector_blocks.try_remove_block(&block_hash);
        }

        self.validate_state();
        PromptMembershipUpdates {
            active: active_delta,
            selector: selector_delta,
        }
    }

    /// Add an output block with a random hash and optional fractional decay weight.
    ///
    /// This is used during generation to track output blocks as they are created.
    pub(super) fn add_output_block(
        &mut self,
        request_id: &RequestId,
        decay_fraction: Option<f64>,
    ) -> Option<SequenceHash> {
        if !self.requests.contains_key(request_id) {
            tracing::warn!("Request {request_id} not found for add_output_block");
            return None;
        }

        // TODO: Output blocks still use random hashes, so indexing them mainly simplifies
        // generic block bookkeeping and usually adds little real reuse signal.
        let random_hash: SequenceHash = Uuid::new_v4().as_u64_pair().0;
        let acquire = self.blocks.touch_block(&random_hash);
        let request_state = self
            .requests
            .get_mut(request_id)
            .expect("request existence was checked above");
        request_state.output_blocks.push((random_hash, acquire.rc));
        if request_state.staged_selector_tracking {
            let selector_acquire = self.selector_blocks.touch_block(&random_hash);
            request_state
                .selector_output_blocks
                .push((random_hash, selector_acquire.rc));
        }

        if let Some(frac) = decay_fraction {
            self.set_single_ref_blocks_as_fractional(request_id, frac);
        }

        self.validate_state();
        acquire.became_present_on_worker.then_some(random_hash)
    }

    /// Force expiry of stale requests if the timer has elapsed.
    /// Returns block membership transitions plus the set of expired request IDs that were removed.
    pub(super) fn force_expiry(&mut self) -> SequenceMutationOutcome {
        let now = Instant::now();

        if now < self.last_expiry_check_time + CHECK_EXPIRY_FREQUENCY {
            return SequenceMutationOutcome::default();
        }

        self.last_expiry_check_time = now;
        let expired_requests_time = now - EXPIRY_DURATION;
        let expired_request_ids: HashSet<RequestId> = self
            .requests
            .iter()
            .filter(|(_, state)| state.started_at < expired_requests_time)
            .map(|(request_id, _)| request_id.clone())
            .collect();

        let mut outcome = SequenceMutationOutcome {
            expired_request_ids,
            ..Default::default()
        };

        for request_id in &outcome.expired_request_ids {
            tracing::warn!("Expiring stale request: {}", request_id);
            let updates = self.free(request_id, now);
            outcome.membership_delta.extend(updates.active);
            outcome.selector_membership_delta.extend(updates.selector);
        }

        self.validate_state();
        outcome
    }

    /// Find all blocks in a request that have only a single strong reference (only used by this request)
    /// and insert them into fractional_blocks with the given fraction value.
    fn set_single_ref_blocks_as_fractional(&mut self, request_id: &RequestId, fraction: f64) {
        let Some(request_state) = self.requests.get(request_id) else {
            tracing::warn!(
                "Request {request_id} not found for set_single_ref_blocks_as_fractional"
            );
            return;
        };

        for (hash, rc) in request_state.all_blocks() {
            if Arc::strong_count(rc) == 1 {
                self.blocks.fractional_blocks.insert(*hash, fraction);
            }
        }
        for (hash, rc) in request_state
            .selector_prompt_blocks
            .iter()
            .chain(request_state.selector_output_blocks.iter())
        {
            if Arc::strong_count(rc) == 1 {
                self.selector_blocks
                    .fractional_blocks
                    .insert(*hash, fraction);
            }
        }
    }

    pub(super) fn worker_load_snapshot(&self) -> WorkerLoadSnapshot {
        WorkerLoadSnapshot {
            active_blocks: self.active_blocks(),
            selector_decode_blocks: self.selector_decode_blocks(),
            active_requests: self.requests.len(),
            prefill: self.prefill.snapshot(),
        }
    }

    fn touch_prompt_chain(
        tracker: &mut BlockTracker,
        hashes: &[SequenceHash],
        parent: Option<SequenceHash>,
    ) -> (Vec<(SequenceHash, Arc<()>)>, PromptMembershipDelta) {
        let mut first_new_idx = None;
        let blocks: Vec<_> = hashes
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, hash)| {
                let acquire = tracker.touch_block(&hash);
                if acquire.became_present_on_worker && first_new_idx.is_none() {
                    first_new_idx = Some(idx);
                }
                (hash, acquire.rc)
            })
            .collect();

        let mut delta = PromptMembershipDelta::default();
        if let Some(first_new_idx) = first_new_idx {
            let store_parent = first_new_idx
                .checked_sub(1)
                .map(|idx| blocks[idx].0)
                .or(parent);
            delta.push_store(
                store_parent,
                blocks[first_new_idx..]
                    .iter()
                    .map(|(hash, _)| *hash)
                    .collect(),
            );
        }
        (blocks, delta)
    }

    fn release_prompt_chain(
        tracker: &mut BlockTracker,
        blocks: Vec<(SequenceHash, Arc<()>)>,
    ) -> PromptMembershipDelta {
        let hashes: Vec<_> = blocks.iter().map(|(hash, _)| *hash).collect();
        let mut first_absent_idx = None;
        for (idx, (hash, rc)) in blocks.into_iter().enumerate() {
            drop(rc);
            if tracker.try_remove_block(&hash) && first_absent_idx.is_none() {
                first_absent_idx = Some(idx);
            }
        }

        let mut delta = PromptMembershipDelta::default();
        if let Some(first_absent_idx) = first_absent_idx {
            delta.push_remove(hashes[first_absent_idx..].to_vec());
        }
        delta
    }

    #[cfg(test)]
    pub(super) fn active_block_hashes(&self) -> FxHashSet<SequenceHash> {
        self.blocks.unique_blocks.keys().copied().collect()
    }

    #[cfg(test)]
    pub(super) fn active_prompt_hashes(&self) -> FxHashSet<SequenceHash> {
        self.requests
            .values()
            .flat_map(|state| state.prompt_blocks.iter().map(|(hash, _)| *hash))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn prefill_hint(tokens: usize, duration_secs: u64) -> PrefillLoadHint {
        PrefillLoadHint {
            initial_effective_prefill_tokens: tokens,
            expected_prefill_duration: Some(Duration::from_secs(duration_secs)),
        }
    }

    fn tracking_hint(tokens: usize) -> Option<PrefillLoadHint> {
        (tokens > 0).then_some(PrefillLoadHint {
            initial_effective_prefill_tokens: tokens,
            expected_prefill_duration: None,
        })
    }

    #[test]
    fn test_prompt_membership_delta_only_reports_first_add_and_last_remove() {
        let mut seq_manager = ActiveSequences::new(4);
        let decay_now = Instant::now();

        let first = seq_manager.add_request_with_prefill_tracking(
            "r1".to_string(),
            Some(vec![1, 2]),
            None,
            true,
            tracking_hint(8),
            decay_now,
        );
        assert_eq!(
            first.membership_delta,
            PromptMembershipDelta {
                stores: vec![PromptMembershipStore {
                    parent: None,
                    hashes: vec![1, 2],
                }],
                removes: Vec::new(),
            }
        );
        assert!(first.expired_request_ids.is_empty());

        let second = seq_manager.add_request_with_prefill_tracking(
            "r2".to_string(),
            Some(vec![1, 2, 3]),
            None,
            true,
            tracking_hint(12),
            decay_now,
        );
        assert_eq!(
            second.membership_delta,
            PromptMembershipDelta {
                stores: vec![PromptMembershipStore {
                    parent: Some(2),
                    hashes: vec![3],
                }],
                removes: Vec::new(),
            }
        );

        let first_free = seq_manager.free(&"r1".to_string(), decay_now);
        assert!(first_free.active.removes.is_empty());
        assert!(first_free.active.stores.is_empty());

        let second_free = seq_manager.free(&"r2".to_string(), decay_now);
        assert!(second_free.active.stores.is_empty());
        assert_eq!(
            second_free.active.removes,
            vec![PromptMembershipRemove {
                hashes: vec![1, 2, 3],
            }]
        );
    }

    #[test]
    fn test_generic_block_membership_includes_output_blocks() {
        let mut seq_manager = ActiveSequences::new(4);
        let decay_now = Instant::now();

        let outcome = seq_manager.add_request_with_prefill_tracking(
            "r1".to_string(),
            Some(vec![1, 2, 3]),
            None,
            true,
            tracking_hint(12),
            decay_now,
        );
        assert_eq!(
            outcome.membership_delta.stores,
            vec![PromptMembershipStore {
                parent: None,
                hashes: vec![1, 2, 3],
            }]
        );
        assert_eq!(
            seq_manager.active_block_hashes(),
            [1, 2, 3].into_iter().collect()
        );

        let output_hash = seq_manager
            .add_output_block(&"r1".to_string(), Some(0.5))
            .expect("request exists");
        assert_eq!(
            seq_manager.active_block_hashes(),
            [1, 2, 3, output_hash].into_iter().collect()
        );

        seq_manager.mark_prefill_completed(&"r1".to_string(), decay_now);
        assert_eq!(seq_manager.active_tokens(decay_now), 0);
        assert_eq!(
            seq_manager.active_block_hashes(),
            [1, 2, 3, output_hash].into_iter().collect()
        );

        let free_delta = seq_manager.free(&"r1".to_string(), decay_now);
        assert_eq!(
            free_delta.active.removes,
            vec![PromptMembershipRemove {
                hashes: vec![1, 2, 3],
            }]
        );
    }

    #[test]
    fn test_active_sequences_shared_blocks() {
        let block_size = 4;
        let mut seq_manager = ActiveSequences::new(block_size);
        let decay_now = Instant::now();

        seq_manager.add_request_with_prefill_tracking(
            "request_1".to_string(),
            Some(vec![1, 2, 3]),
            None,
            true,
            tracking_hint(12),
            decay_now,
        );
        assert_eq!(seq_manager.active_blocks(), 3);
        assert_eq!(seq_manager.active_tokens(decay_now), 12);

        seq_manager.add_request_with_prefill_tracking(
            "request_2".to_string(),
            Some(vec![4]),
            None,
            true,
            tracking_hint(4),
            decay_now,
        );
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.active_tokens(decay_now), 16);

        seq_manager.add_request_with_prefill_tracking(
            "request_3".to_string(),
            Some(vec![1, 2, 3, 4]),
            None,
            true,
            tracking_hint(0),
            decay_now,
        );
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.active_tokens(decay_now), 16);

        seq_manager.free(&"request_2".to_string(), decay_now);
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.active_tokens(decay_now), 12);

        seq_manager.free(&"request_3".to_string(), decay_now);
        assert_eq!(seq_manager.active_blocks(), 3);
        assert_eq!(seq_manager.active_tokens(decay_now), 12);

        seq_manager.free(&"request_1".to_string(), decay_now);
        assert_eq!(seq_manager.active_blocks(), 0);
        assert_eq!(seq_manager.active_tokens(decay_now), 0);
    }

    #[test]
    fn staged_decode_cost_uses_cached_complete_prefix_for_partial_prompt_blocks() {
        let decay_now = Instant::now();

        for (request_id, effective_prefill_tokens, expected_selector_blocks) in
            [("zero", 18, 0), ("partial", 10, 2), ("full", 2, 4)]
        {
            let mut sequences = ActiveSequences::new(4);
            sequences.add_request_with_prefill_tracking(
                request_id.to_string(),
                Some(vec![1, 2, 3, 4]),
                None,
                true,
                tracking_hint(effective_prefill_tokens),
                decay_now,
            );

            assert_eq!(sequences.active_blocks(), 4);
            assert_eq!(sequences.selector_decode_blocks(), expected_selector_blocks);
        }
    }

    #[test]
    fn staged_decode_cost_promotes_uncached_suffix_and_frees_both_trackers() {
        let mut sequences = ActiveSequences::new(4);
        let decay_now = Instant::now();
        let added = sequences.add_request_with_prefill_tracking(
            "r1".to_string(),
            Some(vec![1, 2, 3, 4]),
            None,
            true,
            tracking_hint(8),
            decay_now,
        );

        assert_eq!(sequences.active_blocks(), 4);
        assert_eq!(sequences.selector_decode_blocks(), 2);
        assert_eq!(
            added.selector_membership_delta.stores,
            vec![PromptMembershipStore {
                parent: None,
                hashes: vec![1, 2],
            }]
        );

        let promoted = sequences.mark_prefill_completed(&"r1".to_string(), decay_now);
        assert_eq!(sequences.selector_decode_blocks(), 4);
        assert_eq!(
            promoted.stores,
            vec![PromptMembershipStore {
                parent: Some(2),
                hashes: vec![3, 4],
            }]
        );
        assert!(
            sequences
                .mark_prefill_completed(&"r1".to_string(), decay_now)
                .stores
                .is_empty()
        );

        let removed = sequences.free(&"r1".to_string(), decay_now);
        assert_eq!(sequences.active_blocks(), 0);
        assert_eq!(sequences.selector_decode_blocks(), 0);
        assert_eq!(
            removed.selector.removes,
            vec![PromptMembershipRemove {
                hashes: vec![1, 2, 3, 4],
            }]
        );
    }

    #[test]
    fn staged_decode_cost_deduplicates_shared_prefix_and_tracks_fractional_output() {
        let mut sequences = ActiveSequences::new(4);
        let decay_now = Instant::now();

        for request_id in ["r1", "r2"] {
            sequences.add_request_with_prefill_tracking(
                request_id.to_string(),
                Some(vec![1, 2, 3]),
                None,
                true,
                tracking_hint(4),
                decay_now,
            );
        }
        assert_eq!(sequences.selector_decode_blocks(), 2);

        sequences.mark_prefill_completed(&"r1".to_string(), decay_now);
        assert_eq!(sequences.selector_decode_blocks(), 3);
        assert!(
            sequences
                .add_output_block(&"r1".to_string(), Some(0.5))
                .is_some()
        );
        assert_eq!(sequences.selector_decode_blocks(), 3);

        sequences.free(&"r1".to_string(), decay_now);
        assert_eq!(sequences.selector_decode_blocks(), 2);
        sequences.free(&"r2".to_string(), decay_now);
        assert_eq!(sequences.selector_decode_blocks(), 0);
    }

    #[test]
    fn staged_decode_cost_is_bypassed_without_prefill_tracking() {
        let mut sequences = ActiveSequences::new(4);
        let decay_now = Instant::now();
        sequences.add_request_with_prefill_tracking(
            "r1".to_string(),
            Some(vec![1, 2, 3]),
            None,
            false,
            None,
            decay_now,
        );

        assert_eq!(sequences.active_blocks(), 3);
        assert_eq!(sequences.selector_decode_blocks(), 0);
        assert!(
            sequences
                .mark_prefill_completed(&"r1".to_string(), decay_now)
                .stores
                .is_empty()
        );
        assert_eq!(sequences.selector_decode_blocks(), 0);
    }

    #[test]
    fn test_output_blocks_with_fractional_decay() {
        let block_size = 4;
        let mut seq_manager = ActiveSequences::new(block_size);
        let decay_now = Instant::now();

        seq_manager.add_request_with_prefill_tracking(
            "r1".to_string(),
            Some(vec![1, 2, 3]),
            None,
            true,
            tracking_hint(12),
            decay_now,
        );
        assert_eq!(seq_manager.active_blocks(), 3);

        assert!(
            seq_manager
                .add_output_block(&"r1".to_string(), Some(0.5))
                .is_some()
        );
        assert_eq!(seq_manager.active_blocks(), 2);

        seq_manager.add_request_with_prefill_tracking(
            "r2".to_string(),
            Some(vec![1, 2]),
            None,
            true,
            tracking_hint(8),
            decay_now,
        );
        assert_eq!(seq_manager.active_blocks(), 2);

        assert!(
            seq_manager
                .add_output_block(&"r1".to_string(), Some(0.0))
                .is_some()
        );
        assert_eq!(seq_manager.active_blocks(), 1);

        seq_manager.free(&"r2".to_string(), decay_now);
        seq_manager.free(&"r1".to_string(), decay_now);
        assert_eq!(seq_manager.active_blocks(), 0);
        assert_eq!(seq_manager.active_tokens(decay_now), 0);
    }

    #[test]
    fn test_mark_prefill_completed() {
        let block_size = 4;
        let mut seq_manager = ActiveSequences::new(block_size);
        let decay_now = Instant::now();

        seq_manager.add_request_with_prefill_tracking(
            "r1".to_string(),
            Some(vec![1, 2, 3]),
            None,
            true,
            tracking_hint(12),
            decay_now,
        );
        assert_eq!(seq_manager.active_tokens(decay_now), 12);

        seq_manager.mark_prefill_completed(&"r1".to_string(), decay_now);
        assert_eq!(seq_manager.active_tokens(decay_now), 0);

        seq_manager.mark_prefill_completed(&"r1".to_string(), decay_now);
        assert_eq!(seq_manager.active_tokens(decay_now), 0);

        seq_manager.add_request_with_prefill_tracking(
            "r2".to_string(),
            Some(vec![4, 5]),
            None,
            true,
            tracking_hint(8),
            decay_now,
        );
        assert_eq!(seq_manager.active_tokens(decay_now), 8);

        seq_manager.free(&"r2".to_string(), decay_now);
        assert_eq!(seq_manager.active_tokens(decay_now), 0);
    }

    #[test]
    fn test_add_request_without_prefill_tracking_keeps_active_tokens_zero() {
        let mut seq_manager = ActiveSequences::new(4);
        let decay_now = Instant::now();

        seq_manager.add_request_with_prefill_tracking(
            "r1".to_string(),
            Some(vec![1, 2, 3]),
            None,
            false,
            None,
            decay_now,
        );

        assert_eq!(seq_manager.active_tokens(decay_now), 0);
        assert!(seq_manager.prefill.prefill_order.is_empty());
        assert_eq!(seq_manager.prefill.prefill_full_tokens_sum, 0);

        seq_manager.mark_prefill_completed(&"r1".to_string(), decay_now);
        assert_eq!(seq_manager.active_tokens(decay_now), 0);
        seq_manager.free(&"r1".to_string(), decay_now);
        assert_eq!(seq_manager.active_blocks(), 0);
    }

    #[test]
    fn test_prefill_queue_and_sum_invariants_survive_idempotent_cleanup() {
        let mut seq_manager = ActiveSequences::new(4);
        let decay_now = Instant::now();

        seq_manager.add_request_with_prefill_tracking(
            "r1".to_string(),
            Some(vec![1]),
            None,
            true,
            Some(prefill_hint(50, 10)),
            decay_now,
        );
        seq_manager.add_request_with_prefill_tracking(
            "r2".to_string(),
            Some(vec![2]),
            None,
            true,
            Some(prefill_hint(30, 10)),
            decay_now,
        );

        assert_eq!(seq_manager.prefill.prefill_full_tokens_sum, 80);
        assert_eq!(
            seq_manager.prefill.prefill_order,
            VecDeque::from(vec!["r1".to_string(), "r2".to_string()])
        );

        seq_manager.mark_prefill_completed(&"r1".to_string(), decay_now);
        seq_manager.mark_prefill_completed(&"r1".to_string(), decay_now);
        assert_eq!(seq_manager.prefill.prefill_full_tokens_sum, 30);
        assert_eq!(
            seq_manager.prefill.prefill_order,
            VecDeque::from(vec!["r2".to_string()])
        );

        seq_manager.free(&"r1".to_string(), decay_now);
        seq_manager.free(&"r1".to_string(), decay_now);
        assert_eq!(seq_manager.prefill.prefill_full_tokens_sum, 30);
        assert_eq!(
            seq_manager.prefill.prefill_order,
            VecDeque::from(vec!["r2".to_string()])
        );

        seq_manager.free(&"r2".to_string(), decay_now);
        assert_eq!(seq_manager.prefill.prefill_full_tokens_sum, 0);
        assert!(seq_manager.prefill.prefill_order.is_empty());
        assert!(seq_manager.requests.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn test_force_expiry() {
        let block_size = 4;
        let mut seq_manager = ActiveSequences::new(block_size);

        seq_manager.add_request_with_prefill_tracking(
            "r1".to_string(),
            Some(vec![1, 2]),
            None,
            true,
            tracking_hint(0),
            Instant::now(),
        );
        seq_manager.add_request_with_prefill_tracking(
            "r2".to_string(),
            Some(vec![3, 4]),
            None,
            true,
            tracking_hint(4),
            Instant::now(),
        );
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.selector_decode_blocks(), 3);

        tokio::time::advance(Duration::from_secs(20)).await;
        let expired = seq_manager.force_expiry();
        assert!(
            expired.expired_request_ids.is_empty(),
            "no check before CHECK_EXPIRY_FREQUENCY"
        );
        assert_eq!(seq_manager.active_blocks(), 4);

        tokio::time::advance(Duration::from_secs(11)).await;
        let expired = seq_manager.force_expiry();
        assert!(
            expired.expired_request_ids.is_empty(),
            "requests not old enough to expire"
        );
        assert_eq!(seq_manager.active_blocks(), 4);
        seq_manager.assert_consistent();

        tokio::time::advance(Duration::from_secs(270)).await;
        let expired = seq_manager.force_expiry();
        assert_eq!(
            expired.expired_request_ids,
            HashSet::from(["r1".to_string(), "r2".to_string()])
        );
        assert_eq!(seq_manager.active_blocks(), 0);
        assert_eq!(seq_manager.selector_decode_blocks(), 0);
        assert_eq!(seq_manager.active_tokens(Instant::now()), 0);
        seq_manager.assert_consistent();

        tokio::time::advance(Duration::from_secs(31)).await;
        let expired = seq_manager.add_request_with_prefill_tracking(
            "r3".to_string(),
            Some(vec![5]),
            None,
            true,
            tracking_hint(4),
            Instant::now(),
        );
        assert!(expired.expired_request_ids.is_empty());
        assert_eq!(seq_manager.active_blocks(), 1);
        assert_eq!(seq_manager.active_tokens(Instant::now()), 4);
        seq_manager.assert_consistent();
    }

    #[tokio::test(start_paused = true)]
    async fn test_force_expiry_reanchors_new_oldest_request() {
        let mut seq_manager = ActiveSequences::new(4);
        let first_decay_now = Instant::now();

        seq_manager.add_request_with_prefill_tracking(
            "r1".to_string(),
            Some(vec![1]),
            None,
            true,
            Some(prefill_hint(40, 100)),
            first_decay_now,
        );
        tokio::time::advance(Duration::from_secs(250)).await;
        seq_manager.add_request_with_prefill_tracking(
            "r2".to_string(),
            Some(vec![2]),
            None,
            true,
            Some(prefill_hint(30, 100)),
            Instant::now(),
        );

        tokio::time::advance(Duration::from_secs(60)).await;
        let expired = seq_manager.force_expiry();
        assert_eq!(
            expired.expired_request_ids,
            HashSet::from(["r1".to_string()])
        );
        assert_eq!(seq_manager.active_tokens(Instant::now()), 30);
        assert!(
            seq_manager
                .prefill
                .anchored_prefill
                .as_ref()
                .is_some_and(|(request_id, _)| request_id == "r2")
        );

        tokio::time::advance(Duration::from_secs(20)).await;
        assert_eq!(seq_manager.active_tokens(Instant::now()), 24);
    }
}
