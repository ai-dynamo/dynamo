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

use derive_getters::Getters;
use dynamo_tokens::SequenceHash;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::time::Instant;
use uuid::Uuid;

use crate::protocols::PrefillLoadHint;

/// Duration after which stale requests may be expired (5 minutes).
const EXPIRY_DURATION: Duration = Duration::from_secs(300);

/// How often we *check* for stale requests (30 seconds). This is not
/// the expiration time, that is EXPIRY_DURATION.
const CHECK_EXPIRY_FREQUENCY: Duration = Duration::from_secs(30);

// TODO: use the common request_id if it exists in the repo
pub type RequestId = String;

#[derive(Debug, Clone, Copy)]
struct PrefillLoadState {
    initial_effective_prefill_tokens: usize,
    expected_prefill_duration: Option<Duration>,
}

#[derive(Debug)]
struct RequestState {
    blocks: Vec<(SequenceHash, Arc<()>)>,
    started_at: Instant,
    prefill: Option<PrefillLoadState>,
    expected_output_tokens: Option<u32>,
}

/// A multi-request sequence manager that handles multiple active sequences with shared KV cache
#[derive(Debug, Getters)]
pub struct ActiveSequences {
    requests: HashMap<RequestId, RequestState>,

    /// Admission order for requests still contributing prompt-side load.
    /// The front entry is the only request whose prompt load decays with time.
    prefill_order: VecDeque<RequestId>,

    /// Cached sum of each active prefill request's full prompt-side load.
    prefill_full_tokens_sum: usize,

    unique_blocks: HashMap<SequenceHash, Weak<()>>,

    /// Fractional block counts for blocks that are partially cached.
    /// When a block is in both unique_blocks and fractional_blocks,
    /// it contributes the fractional value instead of 1 to active_blocks().
    fractional_blocks: HashMap<SequenceHash, f64>,

    #[getter(copy)]
    block_size: usize,

    last_expiry_check_time: Instant,
}

impl ActiveSequences {
    /// Create a new SharedSequenceManager instance
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 1, "block_size must be greater than 1");

        Self {
            requests: HashMap::new(),
            prefill_order: VecDeque::new(),
            prefill_full_tokens_sum: 0,
            unique_blocks: HashMap::new(),
            fractional_blocks: HashMap::new(),
            block_size,
            last_expiry_check_time: Instant::now(),
        }
    }

    fn touch_block(&mut self, block: &SequenceHash) -> Arc<()> {
        if let Some(weak) = self.unique_blocks.get(block)
            && let Some(rc) = weak.upgrade()
        {
            return rc;
        }

        let rc = Arc::new(());
        self.unique_blocks.insert(*block, Arc::downgrade(&rc));
        rc
    }

    fn try_remove_block(&mut self, block: &SequenceHash) {
        if let Some(weak) = self.unique_blocks.get(block)
            && weak.strong_count() == 0
        {
            self.unique_blocks.remove(block);
            self.fractional_blocks.remove(block);
        }
    }

    #[cfg(any(test, debug_assertions))]
    fn assert_consistent(&self) {
        let active_prefills: HashSet<RequestId> = self
            .requests
            .iter()
            .filter(|(_, state)| state.prefill.is_some())
            .map(|(request_id, _)| request_id.clone())
            .collect();
        let ordered_prefills: HashSet<RequestId> = self.prefill_order.iter().cloned().collect();
        let recomputed_prefill_sum: usize = self
            .requests
            .values()
            .filter_map(|state| state.prefill)
            .map(|prefill| prefill.initial_effective_prefill_tokens)
            .sum();

        assert_eq!(
            ordered_prefills.len(),
            self.prefill_order.len(),
            "prefill_order contains duplicate request ids",
        );
        assert_eq!(
            ordered_prefills, active_prefills,
            "prefill_order must match requests with active prefill load",
        );
        assert_eq!(
            self.prefill_full_tokens_sum, recomputed_prefill_sum,
            "prefill_full_tokens_sum drifted from request state",
        );
        if let Some(oldest_request_id) = self.prefill_order.front() {
            assert!(
                self.requests
                    .get(oldest_request_id)
                    .is_some_and(|state| state.prefill.is_some()),
                "prefill_order front must point to an active prefill request",
            );
        }
        assert!(
            self.fractional_blocks
                .keys()
                .all(|hash| self.unique_blocks.contains_key(hash)),
            "fractional_blocks cannot reference non-active blocks",
        );
    }

    #[inline]
    fn validate_state(&self) {
        #[cfg(any(test, debug_assertions))]
        self.assert_consistent();
    }

    pub fn active_blocks(&self) -> usize {
        let mut count = self.unique_blocks.len() as f64;
        for (hash, frac) in &self.fractional_blocks {
            if self.unique_blocks.contains_key(hash) {
                count = count - 1.0 + frac;
            }
        }
        count.round() as usize
    }

    fn insert_prefill_load(&mut self, request_id: &RequestId, prefill: PrefillLoadState) {
        self.prefill_full_tokens_sum += prefill.initial_effective_prefill_tokens;
        self.prefill_order.push_back(request_id.clone());
    }

    fn remove_prefill_load(&mut self, request_id: &RequestId) -> Option<PrefillLoadState> {
        let state = self.requests.get_mut(request_id)?;
        let prefill = state.prefill.take()?;
        self.prefill_full_tokens_sum = self
            .prefill_full_tokens_sum
            .checked_sub(prefill.initial_effective_prefill_tokens)
            .expect("prefill_full_tokens_sum underflow");
        self.prefill_order
            .retain(|queued_request_id| queued_request_id != request_id);
        Some(prefill)
    }

    fn oldest_prefill_request_id(&self) -> Option<&RequestId> {
        self.prefill_order.front()
    }

    fn remaining_prefill_tokens_at(&self, request_id: &RequestId, now: Instant) -> usize {
        let Some(prefill) = self
            .requests
            .get(request_id)
            .and_then(|state| state.prefill)
        else {
            return 0;
        };

        let Some(expected_prefill_duration) = prefill.expected_prefill_duration else {
            return prefill.initial_effective_prefill_tokens;
        };

        if expected_prefill_duration.is_zero() {
            return 0;
        }

        let started_at = self
            .requests
            .get(request_id)
            .map(|state| state.started_at)
            .expect("prefill-active request missing started_at timestamp");
        let elapsed = now.saturating_duration_since(started_at);
        let remaining_fraction = (1.0
            - (elapsed.as_secs_f64() / expected_prefill_duration.as_secs_f64()))
        .clamp(0.0, 1.0);

        ((prefill.initial_effective_prefill_tokens as f64) * remaining_fraction).ceil() as usize
    }

    fn active_prefill_tokens_at(&self, now: Instant) -> usize {
        let Some(oldest_request_id) = self.oldest_prefill_request_id() else {
            return 0;
        };
        let oldest_full = self
            .requests
            .get(oldest_request_id)
            .and_then(|state| state.prefill)
            .map(|prefill| prefill.initial_effective_prefill_tokens)
            .expect("prefill_order front missing prefill load");
        let oldest_remaining = self.remaining_prefill_tokens_at(oldest_request_id, now);

        self.prefill_full_tokens_sum
            .checked_sub(oldest_full)
            .expect("prefill_full_tokens_sum smaller than oldest load")
            + oldest_remaining
    }

    pub fn active_tokens(&self) -> usize {
        self.active_prefill_tokens_at(Instant::now())
    }

    /// Find all blocks in a request that have only a single strong reference (only used by this request)
    /// and insert them into fractional_blocks with the given fraction value.
    pub fn set_single_ref_blocks_as_fractional(&mut self, request_id: &RequestId, fraction: f64) {
        let Some(request_state) = self.requests.get(request_id) else {
            tracing::warn!(
                "Request {request_id} not found for set_single_ref_blocks_as_fractional"
            );
            return;
        };

        for (hash, rc) in &request_state.blocks {
            if Arc::strong_count(rc) == 1 {
                self.fractional_blocks.insert(*hash, fraction);
            }
        }
    }

    /// Add a new request with its initial tokens.
    /// Returns the set of expired request IDs that were removed during cleanup.
    pub fn add_request(
        &mut self,
        request_id: RequestId,
        token_sequence: Option<Vec<SequenceHash>>,
        isl: usize,
        overlap: u32,
        expected_output_tokens: Option<u32>,
    ) -> HashSet<RequestId> {
        self.add_request_with_prefill_tracking(
            request_id,
            token_sequence,
            isl,
            overlap,
            expected_output_tokens,
            true,
            None,
        )
    }

    /// Add a new request with optional prompt-token load accounting.
    /// Returns the set of expired request IDs that were removed during cleanup.
    #[allow(clippy::too_many_arguments)]
    pub fn add_request_with_prefill_tracking(
        &mut self,
        request_id: RequestId,
        token_sequence: Option<Vec<SequenceHash>>,
        isl: usize,
        overlap: u32,
        expected_output_tokens: Option<u32>,
        track_prefill_tokens: bool,
        prefill_load_hint: Option<PrefillLoadHint>,
    ) -> HashSet<RequestId> {
        if self.requests.contains_key(&request_id) {
            tracing::error!("Request {request_id} is already active. Ignoring duplicate add.");
            return HashSet::new();
        }

        let removed_requests = self.force_expiry();
        let started_at = Instant::now();

        let blocks = match token_sequence {
            Some(sequence) => sequence
                .into_iter()
                .map(|block| {
                    let rc = self.touch_block(&block);
                    (block, rc)
                })
                .collect(),
            None => Vec::new(),
        };

        let prefill = if track_prefill_tokens {
            let default_tokens = self.new_tokens(isl, overlap);
            let hint = prefill_load_hint.unwrap_or(PrefillLoadHint {
                initial_effective_prefill_tokens: default_tokens,
                expected_prefill_duration: None,
            });

            (hint.initial_effective_prefill_tokens > 0).then_some(PrefillLoadState {
                initial_effective_prefill_tokens: hint.initial_effective_prefill_tokens,
                expected_prefill_duration: hint.expected_prefill_duration,
            })
        } else {
            None
        };

        self.requests.insert(
            request_id.clone(),
            RequestState {
                blocks,
                started_at,
                prefill,
                expected_output_tokens,
            },
        );

        if let Some(prefill) = self
            .requests
            .get(&request_id)
            .and_then(|state| state.prefill)
        {
            self.insert_prefill_load(&request_id, prefill);
        }

        self.validate_state();
        removed_requests
    }

    /// Mark prefill as completed for a request, removing it from prompt-load tracking.
    pub fn mark_prefill_completed(&mut self, request_id: &RequestId) {
        let _ = self.remove_prefill_load(request_id);
        self.validate_state();
    }

    pub fn new_tokens(&self, isl: usize, overlap: u32) -> usize {
        let cached_tokens = (overlap as usize) * self.block_size;
        isl.checked_sub(cached_tokens).unwrap_or_else(|| {
            tracing::error!(
                "prefill_tokens < 0 with ISL {isl} < cached_tokens {cached_tokens} (overlap {overlap} * block_size {}), returning 0",
                self.block_size
            );
            0
        })
    }

    pub fn potential_blocks_and_tokens(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        isl: usize,
        overlap: u32,
    ) -> (usize, usize) {
        self.potential_blocks_and_tokens_with_prefill_tracking(token_sequence, isl, overlap, true)
    }

    pub fn potential_blocks_and_tokens_with_prefill_tracking(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        isl: usize,
        overlap: u32,
        track_prefill_tokens: bool,
    ) -> (usize, usize) {
        let potential_blocks = if let Some(token_seq) = token_sequence {
            self.new_blocks(token_seq) + self.active_blocks()
        } else {
            self.active_blocks()
        };
        let active_tokens = self.active_tokens();
        let potential_tokens = if track_prefill_tokens {
            self.new_tokens(isl, overlap) + active_tokens
        } else {
            active_tokens
        };

        (potential_blocks, potential_tokens)
    }

    /// Match a request against existing blocks and return the number of new blocks that would be added
    pub fn new_blocks(&self, token_sequence: &[SequenceHash]) -> usize {
        token_sequence
            .iter()
            .filter(|block| !self.unique_blocks.contains_key(block))
            .count()
    }

    /// Return the total number of blocks that would be used if the token sequence was added.
    pub fn potential_blocks(&self, token_sequence: &[SequenceHash]) -> usize {
        self.new_blocks(token_sequence) + self.active_blocks()
    }

    /// Free all blocks associated with a request.
    ///
    /// This implicitly calls [`Self::mark_prefill_completed`] first, so callers do not need
    /// to invoke both when the request is finishing.
    pub fn free(&mut self, request_id: &RequestId) -> usize {
        self.mark_prefill_completed(request_id);

        let Some(request_state) = self.requests.remove(request_id) else {
            tracing::warn!("Trying to free non-existent request {request_id}");
            return self.active_blocks();
        };

        let _ = request_state.expected_output_tokens;
        for (block_hash, rc) in request_state.blocks {
            drop(rc);
            self.try_remove_block(&block_hash);
        }

        self.validate_state();
        self.active_blocks()
    }

    /// Add an output block with a random hash and optional fractional decay weight.
    ///
    /// This is used during generation to track output blocks as they are created.
    pub fn add_output_block(
        &mut self,
        request_id: &RequestId,
        decay_fraction: Option<f64>,
    ) -> bool {
        if !self.requests.contains_key(request_id) {
            tracing::warn!("Request {request_id} not found for add_output_block");
            return false;
        }

        let random_hash: SequenceHash = Uuid::new_v4().as_u64_pair().0;
        let rc = self.touch_block(&random_hash);
        self.requests
            .get_mut(request_id)
            .expect("request existence was checked above")
            .blocks
            .push((random_hash, rc));

        if let Some(frac) = decay_fraction {
            self.set_single_ref_blocks_as_fractional(request_id, frac);
        }

        self.validate_state();
        true
    }

    /// Force expiry of stale requests if the timer has elapsed.
    /// Returns the set of expired request IDs that were removed.
    pub fn force_expiry(&mut self) -> HashSet<RequestId> {
        let now = Instant::now();

        if now < self.last_expiry_check_time + CHECK_EXPIRY_FREQUENCY {
            return HashSet::new();
        }

        self.last_expiry_check_time = now;
        let expired_requests_time = now - EXPIRY_DURATION;
        let expired_requests: HashSet<RequestId> = self
            .requests
            .iter()
            .filter(|(_, state)| state.started_at < expired_requests_time)
            .map(|(request_id, _)| request_id.clone())
            .collect();

        for request_id in &expired_requests {
            tracing::warn!("Expiring stale request: {}", request_id);
            self.free(request_id);
        }

        self.validate_state();
        expired_requests
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefill_hint(tokens: usize, duration_secs: u64) -> PrefillLoadHint {
        PrefillLoadHint {
            initial_effective_prefill_tokens: tokens,
            expected_prefill_duration: Some(Duration::from_secs(duration_secs)),
        }
    }

    #[test]
    fn test_active_sequences_shared_blocks() {
        let block_size = 4;
        let mut seq_manager = ActiveSequences::new(block_size);

        seq_manager.add_request("request_1".to_string(), Some(vec![1, 2, 3]), 12, 0, None);
        assert_eq!(seq_manager.active_blocks(), 3);
        assert_eq!(seq_manager.active_tokens(), 12);

        seq_manager.add_request("request_2".to_string(), Some(vec![4]), 4, 0, None);
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.active_tokens(), 16);

        seq_manager.add_request("request_3".to_string(), Some(vec![1, 2, 3, 4]), 16, 4, None);
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.active_tokens(), 16);

        seq_manager.free(&"request_2".to_string());
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.active_tokens(), 12);

        seq_manager.free(&"request_3".to_string());
        assert_eq!(seq_manager.active_blocks(), 3);
        assert_eq!(seq_manager.active_tokens(), 12);

        seq_manager.free(&"request_1".to_string());
        assert_eq!(seq_manager.active_blocks(), 0);
        assert_eq!(seq_manager.active_tokens(), 0);
    }

    #[test]
    fn test_output_blocks_with_fractional_decay() {
        let block_size = 4;
        let mut seq_manager = ActiveSequences::new(block_size);

        seq_manager.add_request("r1".to_string(), Some(vec![1, 2, 3]), 12, 0, None);
        assert_eq!(seq_manager.active_blocks(), 3);

        assert!(seq_manager.add_output_block(&"r1".to_string(), Some(0.5)));
        assert_eq!(seq_manager.active_blocks(), 2);

        seq_manager.add_request("r2".to_string(), Some(vec![1, 2]), 8, 0, None);
        assert_eq!(seq_manager.active_blocks(), 2);

        assert!(seq_manager.add_output_block(&"r1".to_string(), Some(0.0)));
        assert_eq!(seq_manager.active_blocks(), 1);

        seq_manager.free(&"r2".to_string());
        seq_manager.free(&"r1".to_string());
        assert_eq!(seq_manager.active_blocks(), 0);
        assert_eq!(seq_manager.active_tokens(), 0);
    }

    #[test]
    fn test_mark_prefill_completed() {
        let block_size = 4;
        let mut seq_manager = ActiveSequences::new(block_size);

        seq_manager.add_request("r1".to_string(), Some(vec![1, 2, 3]), 12, 0, None);
        assert_eq!(seq_manager.active_tokens(), 12);

        seq_manager.mark_prefill_completed(&"r1".to_string());
        assert_eq!(seq_manager.active_tokens(), 0);

        seq_manager.mark_prefill_completed(&"r1".to_string());
        assert_eq!(seq_manager.active_tokens(), 0);

        seq_manager.add_request("r2".to_string(), Some(vec![4, 5]), 8, 0, None);
        assert_eq!(seq_manager.active_tokens(), 8);

        seq_manager.free(&"r2".to_string());
        assert_eq!(seq_manager.active_tokens(), 0);
    }

    #[test]
    fn test_add_request_without_prefill_tracking_keeps_active_tokens_zero() {
        let mut seq_manager = ActiveSequences::new(4);

        seq_manager.add_request_with_prefill_tracking(
            "r1".to_string(),
            Some(vec![1, 2, 3]),
            12,
            0,
            None,
            false,
            None,
        );

        assert_eq!(seq_manager.active_tokens(), 0);
        assert!(seq_manager.prefill_order.is_empty());
        assert_eq!(seq_manager.prefill_full_tokens_sum, 0);

        seq_manager.mark_prefill_completed(&"r1".to_string());
        assert_eq!(seq_manager.active_tokens(), 0);
        seq_manager.free(&"r1".to_string());
        assert_eq!(seq_manager.active_blocks(), 0);
    }

    #[test]
    fn test_potential_blocks_and_tokens_without_prefill_tracking_ignores_prompt_load() {
        let mut seq_manager = ActiveSequences::new(4);
        seq_manager.add_request_with_prefill_tracking(
            "r1".to_string(),
            Some(vec![1, 2, 3]),
            12,
            0,
            None,
            false,
            None,
        );

        let (blocks, tokens) = seq_manager.potential_blocks_and_tokens_with_prefill_tracking(
            Some(&[1, 2, 3, 4]),
            16,
            0,
            false,
        );
        assert_eq!(blocks, 4);
        assert_eq!(tokens, 0);
    }

    #[tokio::test(start_paused = true)]
    async fn test_prefill_decay_only_applies_to_oldest_request() {
        let mut seq_manager = ActiveSequences::new(4);

        seq_manager.add_request_with_prefill_tracking(
            "r1".to_string(),
            Some(vec![1]),
            100,
            0,
            None,
            true,
            Some(prefill_hint(100, 10)),
        );
        tokio::time::advance(Duration::from_secs(2)).await;
        seq_manager.add_request_with_prefill_tracking(
            "r2".to_string(),
            Some(vec![2]),
            60,
            0,
            None,
            true,
            Some(prefill_hint(60, 6)),
        );

        assert_eq!(seq_manager.active_tokens(), 140);

        tokio::time::advance(Duration::from_secs(3)).await;
        assert_eq!(seq_manager.active_tokens(), 110);
        assert!(seq_manager.active_tokens() <= 160);
        assert!(seq_manager.active_tokens() >= 60);
    }

    #[tokio::test(start_paused = true)]
    async fn test_prefill_decay_hands_off_to_next_oldest_request() {
        let mut seq_manager = ActiveSequences::new(4);

        seq_manager.add_request_with_prefill_tracking(
            "r1".to_string(),
            Some(vec![1]),
            100,
            0,
            None,
            true,
            Some(prefill_hint(100, 10)),
        );
        seq_manager.add_request_with_prefill_tracking(
            "r2".to_string(),
            Some(vec![2]),
            40,
            0,
            None,
            true,
            Some(prefill_hint(40, 8)),
        );

        tokio::time::advance(Duration::from_secs(3)).await;
        assert_eq!(seq_manager.active_tokens(), 110);

        seq_manager.mark_prefill_completed(&"r1".to_string());
        assert_eq!(seq_manager.active_tokens(), 25);
        assert_eq!(
            seq_manager.prefill_order,
            VecDeque::from(vec!["r2".to_string()])
        );
    }

    #[test]
    fn test_prefill_queue_and_sum_invariants_survive_idempotent_cleanup() {
        let mut seq_manager = ActiveSequences::new(4);

        seq_manager.add_request_with_prefill_tracking(
            "r1".to_string(),
            Some(vec![1]),
            50,
            0,
            None,
            true,
            Some(prefill_hint(50, 10)),
        );
        seq_manager.add_request_with_prefill_tracking(
            "r2".to_string(),
            Some(vec![2]),
            30,
            0,
            None,
            true,
            Some(prefill_hint(30, 10)),
        );

        assert_eq!(seq_manager.prefill_full_tokens_sum, 80);
        assert_eq!(
            seq_manager.prefill_order,
            VecDeque::from(vec!["r1".to_string(), "r2".to_string()])
        );

        seq_manager.mark_prefill_completed(&"r1".to_string());
        seq_manager.mark_prefill_completed(&"r1".to_string());
        assert_eq!(seq_manager.prefill_full_tokens_sum, 30);
        assert_eq!(
            seq_manager.prefill_order,
            VecDeque::from(vec!["r2".to_string()])
        );

        seq_manager.free(&"r1".to_string());
        seq_manager.free(&"r1".to_string());
        assert_eq!(seq_manager.prefill_full_tokens_sum, 30);
        assert_eq!(
            seq_manager.prefill_order,
            VecDeque::from(vec!["r2".to_string()])
        );

        seq_manager.free(&"r2".to_string());
        assert_eq!(seq_manager.prefill_full_tokens_sum, 0);
        assert!(seq_manager.prefill_order.is_empty());
        assert!(seq_manager.requests.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn test_force_expiry() {
        let block_size = 4;
        let mut seq_manager = ActiveSequences::new(block_size);

        seq_manager.add_request("r1".to_string(), Some(vec![1, 2]), 8, 0, None);
        seq_manager.add_request("r2".to_string(), Some(vec![3, 4]), 8, 0, None);
        assert_eq!(seq_manager.active_blocks(), 4);

        tokio::time::advance(Duration::from_secs(20)).await;
        let expired = seq_manager.force_expiry();
        assert!(expired.is_empty(), "no check before CHECK_EXPIRY_FREQUENCY");
        assert_eq!(seq_manager.active_blocks(), 4);

        tokio::time::advance(Duration::from_secs(11)).await;
        let expired = seq_manager.force_expiry();
        assert!(expired.is_empty(), "requests not old enough to expire");
        assert_eq!(seq_manager.active_blocks(), 4);
        seq_manager.assert_consistent();

        tokio::time::advance(Duration::from_secs(270)).await;
        let expired = seq_manager.force_expiry();
        assert_eq!(expired, HashSet::from(["r1".to_string(), "r2".to_string()]));
        assert_eq!(seq_manager.active_blocks(), 0);
        assert_eq!(seq_manager.active_tokens(), 0);
        seq_manager.assert_consistent();

        tokio::time::advance(Duration::from_secs(31)).await;
        let expired = seq_manager.add_request("r3".to_string(), Some(vec![5]), 4, 0, None);
        assert!(expired.is_empty());
        assert_eq!(seq_manager.active_blocks(), 1);
        assert_eq!(seq_manager.active_tokens(), 4);
        seq_manager.assert_consistent();
    }
}
