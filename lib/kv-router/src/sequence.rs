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
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;
use uuid::Uuid;

/// Duration after which stale requests are forcibly expired (5 minutes)
const EXPIRY_DURATION: Duration = Duration::from_secs(300);

// TODO: use the common request_id if it exists in the repo
pub type RequestId = String;

/// A multi-request sequence manager that handles multiple active sequences with shared KV cache
#[derive(Debug, Getters)]
pub struct ActiveSequences {
    active_seqs: HashMap<RequestId, Vec<(SequenceHash, Arc<()>)>>,

    prefill_tokens: HashMap<RequestId, usize>,

    /// Expected output tokens per request (used for resource estimation)
    expected_output_tokens: HashMap<RequestId, u32>,

    unique_blocks: HashMap<SequenceHash, std::sync::Weak<()>>,

    /// Fractional block counts for blocks that are partially cached
    /// When a block is in both unique_blocks and fractional_blocks,
    /// it contributes the fractional value instead of 1 to active_blocks()
    fractional_blocks: HashMap<SequenceHash, f64>,

    #[getter(copy)]
    block_size: usize,

    #[getter(copy)]
    active_tokens: usize,

    /// Timer for when to force expiry of stale requests
    expiry_timer: Instant,

    /// Set of request IDs to check for expiry
    expiry_requests: HashSet<RequestId>,
}

impl ActiveSequences {
    /// Create a new SharedSequenceManager instance
    pub fn new(block_size: usize) -> Self {
        // TODO: make this not a hard req
        assert!(block_size > 1, "block_size must be greater than 1");

        Self {
            active_seqs: HashMap::new(),
            prefill_tokens: HashMap::new(),
            expected_output_tokens: HashMap::new(),
            unique_blocks: HashMap::new(),
            fractional_blocks: HashMap::new(),
            block_size,
            active_tokens: 0,
            expiry_timer: Instant::now() + EXPIRY_DURATION,
            expiry_requests: HashSet::new(),
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

    pub fn active_blocks(&self) -> usize {
        let mut count = self.unique_blocks.len() as f64;
        for (hash, frac) in &self.fractional_blocks {
            if self.unique_blocks.contains_key(hash) {
                // Subtract 1 (the full block) and add the fractional value
                count = count - 1.0 + frac;
            }
        }
        count.round() as usize
    }

    /// Find all blocks in a request that have only a single strong reference (only used by this request)
    /// and insert them into fractional_blocks with the given fraction value.
    pub fn set_single_ref_blocks_as_fractional(&mut self, request_id: &RequestId, fraction: f64) {
        let Some(blocks) = self.active_seqs.get(request_id) else {
            tracing::warn!(
                "Request {request_id} not found for set_single_ref_blocks_as_fractional"
            );
            return;
        };

        for (hash, rc) in blocks {
            // A block with strong_count == 1 means only this request holds a reference
            if Arc::strong_count(rc) == 1 {
                self.fractional_blocks.insert(*hash, fraction);
            }
        }
    }

    /// Add a new request with its initial tokens
    /// Returns the set of expired request IDs that were removed during cleanup
    pub fn add_request(
        &mut self,
        request_id: RequestId,
        token_sequence: Option<Vec<SequenceHash>>,
        isl: usize,
        overlap: u32,
        expected_output_tokens: Option<u32>,
    ) -> HashSet<RequestId> {
        // Check for double-add and log error, returning early
        if self.active_seqs.contains_key(&request_id) {
            tracing::error!("Request {request_id} is already active. Ignoring duplicate add.");
            return HashSet::new();
        }

        // Lazily check and clean up expired requests, capturing removed IDs
        let removed_requests = self.force_expiry();

        let prefill_tokens = self.new_tokens(isl, overlap);
        self.prefill_tokens
            .insert(request_id.clone(), prefill_tokens);
        self.active_tokens += prefill_tokens;

        // Store expected output tokens if provided
        if let Some(tokens) = expected_output_tokens {
            self.expected_output_tokens
                .insert(request_id.clone(), tokens);
        }

        if let Some(sequence) = token_sequence {
            let sequence_with_refs: Vec<(SequenceHash, Arc<()>)> = sequence
                .iter()
                .map(|block| (*block, self.touch_block(block)))
                .collect();
            self.active_seqs
                .insert(request_id.clone(), sequence_with_refs);
        } else {
            // dummy empty sequence
            self.active_seqs.insert(request_id.clone(), Vec::new());
        }

        removed_requests
    }

    /// Mark prefill as completed for a request, removing it from prefill_tokens tracking
    pub fn mark_prefill_completed(&mut self, request_id: &RequestId) {
        if let Some(tokens) = self.prefill_tokens.remove(request_id) {
            self.active_tokens = self
                .active_tokens
                .checked_sub(tokens)
                .expect("active_tokens underflow");
        }
    }

    pub fn new_tokens(&self, isl: usize, overlap: u32) -> usize {
        let cached_tokens = (overlap as usize) * self.block_size;
        isl.checked_sub(cached_tokens)
            .unwrap_or_else(|| {
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
        let potential_blocks = if let Some(token_seq) = token_sequence {
            self.new_blocks(token_seq) + self.active_blocks()
        } else {
            self.active_blocks()
        };
        let potential_tokens = self.new_tokens(isl, overlap) + self.active_tokens;
        (potential_blocks, potential_tokens)
    }

    /// Match a request against existing blocks and return the number of new blocks that would be added
    pub fn new_blocks(&self, token_sequence: &[SequenceHash]) -> usize {
        token_sequence
            .iter()
            .filter(|block| !self.unique_blocks.contains_key(block))
            .count()
    }

    /// Return the total number of blocks that would be used if the token sequence was added
    /// This is the sum of new blocks that would be added plus the current active blocks
    pub fn potential_blocks(&self, token_sequence: &[SequenceHash]) -> usize {
        self.new_blocks(token_sequence) + self.active_blocks()
    }

    /// Free all blocks associated with a request
    pub fn free(&mut self, request_id: &RequestId) -> usize {
        self.mark_prefill_completed(request_id);

        self.expiry_requests.remove(request_id);

        // Remove expected output tokens tracking
        self.expected_output_tokens.remove(request_id);

        // Remove from active_seqs and get the token sequence
        let token_seq = match self.active_seqs.remove(request_id) {
            Some(seq) => seq,
            None => {
                tracing::warn!("Trying to free non-existent request {request_id}");
                return self.active_blocks();
            }
        };

        // Drop each Rc reference, then clean up the corresponding weak reference
        for (block_hash, rc) in token_seq {
            drop(rc);
            self.try_remove_block(&block_hash);
        }

        self.active_blocks()
    }

    /// Add an output block with a random hash and optional fractional decay weight.
    ///
    /// This is used during generation to track output blocks as they are created.
    /// The decay_fraction (if provided) represents how "temporary" the block is:
    /// - 1.0 means fully counted (early in generation)
    /// - 0.0 means not counted (near end of expected output)
    /// - Computed as: 1 - (current_osl / expected_output_tokens)
    ///
    /// Returns true if the block was added, false if the request was not found.
    pub fn add_output_block(
        &mut self,
        request_id: &RequestId,
        decay_fraction: Option<f64>,
    ) -> bool {
        // Check if request exists first (immutable borrow)
        if !self.active_seqs.contains_key(request_id) {
            tracing::warn!("Request {request_id} not found for add_output_block");
            return false;
        }

        // Generate a random block hash using UUID
        let random_hash: SequenceHash = Uuid::new_v4().as_u64_pair().0;

        // Touch the block (adds to unique_blocks)
        let rc = self.touch_block(&random_hash);

        // Now we can safely get_mut and push
        self.active_seqs
            .get_mut(request_id)
            .unwrap()
            .push((random_hash, rc));

        // Apply fractional decay to all single-ref blocks in this request if provided
        if let Some(frac) = decay_fraction {
            self.set_single_ref_blocks_as_fractional(request_id, frac);
        }

        true
    }

    /// Force expiry of stale requests if the timer has elapsed
    /// Returns the set of expired request IDs that were removed
    pub fn force_expiry(&mut self) -> HashSet<RequestId> {
        let now = Instant::now();

        // Early return if timer hasn't expired yet
        if now < self.expiry_timer {
            return HashSet::new();
        }

        // Process expired requests - drain to avoid clone
        let expired_requests: HashSet<RequestId> = self.expiry_requests.drain().collect();
        for request_id in &expired_requests {
            tracing::warn!("Force expiring stale request: {}", request_id);
            self.free(request_id);
        }

        self.expiry_timer = now + EXPIRY_DURATION;
        self.expiry_requests = self.active_seqs.keys().cloned().collect();

        expired_requests
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        // Add request with 3 prefill blocks
        seq_manager.add_request("r1".to_string(), Some(vec![1, 2, 3]), 12, 0, None);
        assert_eq!(seq_manager.active_blocks(), 3);

        // Add output block with 0.5 decay fraction.
        // This adds a random block and sets all single-ref blocks to 0.5.
        assert!(seq_manager.add_output_block(&"r1".to_string(), Some(0.5)));
        // 4 unique blocks, all single-ref → all fractional at 0.5
        // active_blocks = 4 - 4 + 4*0.5 = 2
        assert_eq!(seq_manager.active_blocks(), 2);

        // Add second request sharing prefix [1, 2]
        seq_manager.add_request("r2".to_string(), Some(vec![1, 2]), 8, 0, None);
        // Blocks 1,2 now have strong_count=2 but still have fractional 0.5 from before
        // No new unique blocks → active_blocks = 4 - 4 + 2.0 = 2
        assert_eq!(seq_manager.active_blocks(), 2);

        // Add another output block with 0.0 decay for r1.
        // set_single_ref_blocks_as_fractional updates only single-ref blocks:
        //   blocks 1,2: strong_count=2, NOT updated (remain 0.5)
        //   block 3, old output, new output: strong_count=1, set to 0.0
        // active_blocks = 5 - 5 + (0.5+0.5+0.0+0.0+0.0) = 1
        assert!(seq_manager.add_output_block(&"r1".to_string(), Some(0.0)));
        assert_eq!(seq_manager.active_blocks(), 1);

        // Free both requests, verify clean state
        seq_manager.free(&"r2".to_string());
        seq_manager.free(&"r1".to_string());
        assert_eq!(seq_manager.active_blocks(), 0);
        assert_eq!(seq_manager.active_tokens(), 0);
    }

    #[test]
    fn test_mark_prefill_completed() {
        let block_size = 4;
        let mut seq_manager = ActiveSequences::new(block_size);

        // Add request with isl=12, overlap=0 → active_tokens=12
        seq_manager.add_request("r1".to_string(), Some(vec![1, 2, 3]), 12, 0, None);
        assert_eq!(seq_manager.active_tokens(), 12);

        // Mark prefill completed → active_tokens drops to 0
        seq_manager.mark_prefill_completed(&"r1".to_string());
        assert_eq!(seq_manager.active_tokens(), 0);

        // Double-mark: no panic, still 0
        seq_manager.mark_prefill_completed(&"r1".to_string());
        assert_eq!(seq_manager.active_tokens(), 0);

        // Add second request with isl=8
        seq_manager.add_request("r2".to_string(), Some(vec![4, 5]), 8, 0, None);
        assert_eq!(seq_manager.active_tokens(), 8);

        // Free it (internally calls mark_prefill_completed) → active_tokens=0
        seq_manager.free(&"r2".to_string());
        assert_eq!(seq_manager.active_tokens(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn test_force_expiry() {
        let block_size = 4;
        let mut seq_manager = ActiveSequences::new(block_size);

        // Add two requests
        seq_manager.add_request("r1".to_string(), Some(vec![1, 2]), 8, 0, None);
        seq_manager.add_request("r2".to_string(), Some(vec![3, 4]), 8, 0, None);
        assert_eq!(seq_manager.active_blocks(), 4);

        // First expiry cycle: advance past EXPIRY_DURATION.
        // This populates expiry_requests with {r1, r2} but doesn't expire anything
        // since expiry_requests started empty.
        tokio::time::advance(Duration::from_secs(301)).await;
        let expired = seq_manager.force_expiry();
        assert!(expired.is_empty());

        // Second expiry cycle: advance again so the timer expires.
        // Adding r3 triggers force_expiry which drains {r1, r2}.
        tokio::time::advance(Duration::from_secs(301)).await;
        let expired = seq_manager.add_request("r3".to_string(), Some(vec![5]), 4, 0, None);
        assert_eq!(expired, HashSet::from(["r1".to_string(), "r2".to_string()]));

        // Only r3's block remains
        assert_eq!(seq_manager.active_blocks(), 1);
        assert_eq!(seq_manager.active_tokens(), 4);
    }
}
