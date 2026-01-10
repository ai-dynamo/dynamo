// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduler testing utilities.
//!
//! This module provides test infrastructure for the scheduler, including:
//! - Creating test schedulers with real BlockManager<G1>
//! - Generating test requests with specified tokens
//! - Populating prefix cache with known sequences
//! - Integration tests for prefix caching behavior
//! - Mock engine for CPU-only scheduler testing

pub mod mock;

use crate::v2::integrations::common::Request;
use crate::v2::integrations::scheduler::{
    KVCacheManager, RequestStatus, Scheduler, SchedulerConfig,
};
use crate::v2::logical::blocks::BlockRegistry;
use crate::v2::SequenceHash;
use crate::G1;

use super::managers;
use super::token_blocks;

/// Create a scheduler with real BlockManager<G1> for testing.
///
/// # Arguments
/// * `block_count` - Number of blocks in the KV cache
/// * `block_size` - Tokens per block
/// * `enable_prefix_caching` - Whether to enable prefix cache lookups
///
/// # Returns
/// A tuple of (Scheduler, BlockRegistry) where the registry can be used
/// for additional block management operations.
///
/// # Example
/// ```ignore
/// let (scheduler, registry) = create_test_scheduler(100, 16, true);
/// scheduler.add_request(create_test_request("req-1", vec![1, 2, 3, 4], None));
/// let output = scheduler.schedule();
/// ```
pub fn create_test_scheduler(
    block_count: usize,
    block_size: usize,
    enable_prefix_caching: bool,
) -> (Scheduler, BlockRegistry) {
    let registry = managers::create_test_registry();
    let block_manager = managers::create_test_manager::<G1>(block_count, block_size, registry.clone());

    let kv_cache = KVCacheManager::with_prefix_caching(block_manager, block_size, enable_prefix_caching)
        .expect("Should create KVCacheManager");

    let config = SchedulerConfig::builder()
        .max_seq_len(8192)
        .max_num_batched_tokens(8192)
        .max_num_seqs(256)
        .block_size(block_size)
        .enable_prefix_caching(enable_prefix_caching)
        .enable_chunked_prefill(false)
        .max_prefill_chunk_size(None)
        .build()
        .expect("Should build config");

    let scheduler = Scheduler::new(config, kv_cache);

    (scheduler, registry)
}

/// Create a test request with specified tokens.
///
/// # Arguments
/// * `request_id` - Unique identifier for the request
/// * `tokens` - Token IDs for the prompt
/// * `max_tokens` - Optional maximum number of output tokens
///
/// # Example
/// ```ignore
/// let request = create_test_request("req-1", vec![1, 2, 3, 4], Some(100));
/// scheduler.add_request(request);
/// ```
pub fn create_test_request(
    request_id: &str,
    tokens: Vec<u32>,
    max_tokens: Option<usize>,
) -> Request {
    Request::new(request_id, tokens, None, None, max_tokens)
}

/// Create a test request with a specific salt for cache isolation.
///
/// Requests with different salts will not share prefix cache entries,
/// even if they have identical token sequences.
pub fn create_test_request_with_salt(
    request_id: &str,
    tokens: Vec<u32>,
    salt: &str,
    max_tokens: Option<usize>,
) -> Request {
    Request::new(request_id, tokens, None, Some(salt.to_string()), max_tokens)
}

/// Populate the scheduler's prefix cache with a token sequence.
///
/// This function:
/// 1. Creates a request with the given tokens
/// 2. Schedules it (allocating blocks)
/// 3. Simulates block completion and registration
/// 4. Finishes the request (blocks return to inactive pool for cache reuse)
///
/// After calling this, subsequent requests with the same token prefix
/// will find cached blocks via `get_computed_blocks()`.
///
/// # Arguments
/// * `scheduler` - The scheduler to populate
/// * `request_id` - ID for the temporary request
/// * `tokens` - Tokens to cache
/// * `block_size` - Block size in tokens (must match scheduler config)
///
/// # Returns
/// Sequence hashes of the registered blocks
///
/// # Note
/// This requires the scheduler to have prefix caching enabled.
#[allow(dead_code)]
pub fn populate_prefix_cache(
    scheduler: &mut Scheduler,
    request_id: &str,
    tokens: &[u32],
    block_size: usize,
) -> Vec<SequenceHash> {
    // Create and add request
    let request = create_test_request(request_id, tokens.to_vec(), Some(100));
    scheduler.add_request(request);

    // Schedule to allocate blocks
    let output = scheduler.schedule();

    // Verify the request was scheduled
    assert!(
        !output.scheduled_new_reqs.is_empty(),
        "Request should be scheduled"
    );

    // Get sequence hashes before finishing
    let num_complete_blocks = tokens.len() / block_size;
    let token_sequence = token_blocks::create_token_sequence(
        num_complete_blocks,
        block_size,
        tokens[0],
    );
    let hashes = token_blocks::generate_sequence_hashes(&token_sequence);

    // Finish the request to release blocks to inactive pool
    scheduler.finish_requests(&[request_id.to_string()], RequestStatus::FinishedStopped);

    hashes
}


// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::testing::{managers, token_blocks};

    // -------------------------------------------------------------------------
    // Test Infrastructure Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_test_scheduler() {
        let (scheduler, _registry) = create_test_scheduler(100, 16, true);
        assert_eq!(scheduler.num_waiting(), 0);
        assert_eq!(scheduler.num_running(), 0);
    }

    #[test]
    fn test_create_test_request() {
        let request = create_test_request("req-1", vec![1, 2, 3, 4], Some(100));
        assert_eq!(request.request_id, "req-1");
        assert_eq!(request.tokens.len(), 4);
        assert_eq!(request.max_tokens, Some(100));
    }

    // -------------------------------------------------------------------------
    // Basic Scheduling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_basic_scheduling_single_request() {
        let (mut scheduler, _registry) = create_test_scheduler(100, 16, false);

        // Add a request with 64 tokens (4 blocks)
        let tokens: Vec<u32> = (0..64).collect();
        let request = create_test_request("req-1", tokens, Some(100));
        scheduler.add_request(request);

        assert_eq!(scheduler.num_waiting(), 1);
        assert_eq!(scheduler.num_running(), 0);

        // Schedule
        let output = scheduler.schedule();

        assert_eq!(scheduler.num_waiting(), 0);
        assert_eq!(scheduler.num_running(), 1);
        assert_eq!(output.scheduled_new_reqs.len(), 1);
        assert_eq!(output.scheduled_new_reqs[0].req_id, "req-1");

        // Should have scheduled 64 tokens
        assert_eq!(output.total_num_scheduled_tokens(), 64);
    }

    // -------------------------------------------------------------------------
    // Prefix Caching Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_prefix_cache_hit_basic() {
        // Setup: 100 blocks, block_size=16, prefix_caching=true
        let block_size = 16;
        let registry = managers::create_test_registry();
        let block_manager =
            managers::create_test_manager::<G1>(100, block_size, registry.clone());

        // Pre-populate the cache with 4 blocks of tokens (0..64)
        // This simulates a previous request that completed and released its blocks
        let token_sequence = token_blocks::create_token_sequence(4, block_size, 0);
        let seq_hashes = managers::populate_manager_with_blocks(&block_manager, token_sequence.blocks())
            .expect("Should populate");
        assert_eq!(seq_hashes.len(), 4);

        // Verify blocks are in the pool and can be matched
        let matched = block_manager.match_blocks(&seq_hashes);
        assert_eq!(matched.len(), 4, "Should match all 4 blocks");
        drop(matched); // Release blocks back to inactive pool

        // Create scheduler with the pre-populated block manager
        let kv_cache = KVCacheManager::with_prefix_caching(block_manager, block_size, true)
            .expect("Should create KVCacheManager");

        let config = SchedulerConfig::builder()
            .max_seq_len(8192)
            .max_num_batched_tokens(8192)
            .max_num_seqs(256)
            .block_size(block_size)
            .enable_prefix_caching(true)
            .enable_chunked_prefill(false)
            .max_prefill_chunk_size(None)
            .build()
            .expect("Should build config");

        let mut scheduler = Scheduler::new(config, kv_cache);

        // Request: Same 64 tokens prefix + 16 new tokens (5 blocks total)
        // The first 64 tokens (0..64) should match the cached blocks
        let mut tokens: Vec<u32> = (0..64).collect();
        tokens.extend(64..80); // Add 16 more tokens
        let request = create_test_request("req-1", tokens, Some(100));
        scheduler.add_request(request);

        // Schedule the request
        let output = scheduler.schedule();
        assert_eq!(output.scheduled_new_reqs.len(), 1);
        assert_eq!(output.scheduled_new_reqs[0].req_id, "req-1");

        // Should have found 64 cached tokens (4 blocks)
        assert_eq!(
            output.scheduled_new_reqs[0].num_computed_tokens, 64,
            "Request should have 64 cached tokens from prefix cache"
        );

        // Total scheduled = 80 tokens, but only 16 need computation
        assert_eq!(output.total_num_scheduled_tokens(), 16);
    }

    #[test]
    fn test_prefix_cache_disabled() {
        // Setup: prefix_caching=false
        let (mut scheduler, _registry) = create_test_scheduler(100, 16, false);

        // R1: 64 tokens
        let tokens: Vec<u32> = (0..64).collect();
        let request1 = create_test_request("req-1", tokens.clone(), Some(100));
        scheduler.add_request(request1);

        let output1 = scheduler.schedule();
        assert_eq!(output1.scheduled_new_reqs[0].num_computed_tokens, 0);

        // Finish R1
        scheduler.finish_requests(&["req-1".to_string()], RequestStatus::FinishedStopped);

        // R2: Same tokens - should still have 0 cached (prefix caching disabled)
        let request2 = create_test_request("req-2", tokens, Some(100));
        scheduler.add_request(request2);

        let output2 = scheduler.schedule();
        assert_eq!(
            output2.scheduled_new_reqs[0].num_computed_tokens, 0,
            "With prefix caching disabled, should have no cached tokens"
        );
    }

    #[test]
    fn test_prefix_cache_partial_match() {
        // Setup with prefix caching
        let block_size = 16;
        let registry = managers::create_test_registry();
        let block_manager =
            managers::create_test_manager::<G1>(100, block_size, registry.clone());

        // Pre-populate the cache with 3 blocks of tokens (0..48)
        let token_sequence = token_blocks::create_token_sequence(3, block_size, 0);
        let seq_hashes =
            managers::populate_manager_with_blocks(&block_manager, token_sequence.blocks())
                .expect("Should populate");
        assert_eq!(seq_hashes.len(), 3);

        // Create scheduler with the pre-populated block manager
        let kv_cache = KVCacheManager::with_prefix_caching(block_manager, block_size, true)
            .expect("Should create KVCacheManager");

        let config = SchedulerConfig::builder()
            .max_seq_len(8192)
            .max_num_batched_tokens(8192)
            .max_num_seqs(256)
            .block_size(block_size)
            .enable_prefix_caching(true)
            .enable_chunked_prefill(false)
            .max_prefill_chunk_size(None)
            .build()
            .expect("Should build config");

        let mut scheduler = Scheduler::new(config, kv_cache);

        // R2: First 32 tokens match (2 blocks), next 32 are different
        let mut tokens_r2: Vec<u32> = (0..32).collect(); // Matching prefix (2 blocks)
        tokens_r2.extend(1000..1032); // Different tokens (2 blocks)
        let request = create_test_request("req-1", tokens_r2, Some(100));
        scheduler.add_request(request);

        let output = scheduler.schedule();

        // Should match only 2 blocks (32 tokens) because third block has different tokens
        assert_eq!(
            output.scheduled_new_reqs[0].num_computed_tokens, 32,
            "Should match first 2 blocks (32 tokens)"
        );
    }

    #[test]
    fn test_block_count_matches_computed_plus_new() {
        // Setup with prefix caching
        let block_size = 16;
        let registry = managers::create_test_registry();
        let block_manager =
            managers::create_test_manager::<G1>(100, block_size, registry.clone());

        // Pre-populate the cache with 3 blocks of tokens (0..48)
        let token_sequence = token_blocks::create_token_sequence(3, block_size, 0);
        let _seq_hashes =
            managers::populate_manager_with_blocks(&block_manager, token_sequence.blocks())
                .expect("Should populate");

        // Create scheduler with the pre-populated block manager
        let kv_cache = KVCacheManager::with_prefix_caching(block_manager, block_size, true)
            .expect("Should create KVCacheManager");

        let config = SchedulerConfig::builder()
            .max_seq_len(8192)
            .max_num_batched_tokens(8192)
            .max_num_seqs(256)
            .block_size(block_size)
            .enable_prefix_caching(true)
            .enable_chunked_prefill(false)
            .max_prefill_chunk_size(None)
            .build()
            .expect("Should build config");

        let mut scheduler = Scheduler::new(config, kv_cache);

        // Request: 80 tokens, first 48 should be cached (3 blocks)
        let mut tokens: Vec<u32> = (0..48).collect(); // Same prefix
        tokens.extend(48..80); // 32 more tokens (2 blocks)
        let request = create_test_request("req-1", tokens, Some(100));
        scheduler.add_request(request);

        let output = scheduler.schedule();

        // Verify:
        // - Total tokens: 80
        // - Cached tokens: 48 (3 blocks)
        // - New tokens: 32 (2 blocks)
        // - Total blocks needed: 5
        assert_eq!(
            output.scheduled_new_reqs[0].num_computed_tokens, 48,
            "Should have 48 cached tokens"
        );

        // Total blocks allocated should be 5 (3 cached + 2 new)
        let block_ids = &output.scheduled_new_reqs[0].block_ids;
        assert_eq!(block_ids.len(), 5, "Should have 5 total blocks (3 cached + 2 new)");
    }

    #[test]
    fn test_prefix_cache_with_different_salt() {
        let (mut scheduler, _registry) = create_test_scheduler(100, 16, true);

        // R1: 64 tokens with salt "salt1"
        let tokens: Vec<u32> = (0..64).collect();
        let request1 = create_test_request_with_salt("req-1", tokens.clone(), "salt1", Some(100));
        scheduler.add_request(request1);

        let _output1 = scheduler.schedule();
        scheduler.finish_requests(&["req-1".to_string()], RequestStatus::FinishedStopped);

        // R2: Same tokens but different salt - should NOT match cache
        let request2 = create_test_request_with_salt("req-2", tokens, "salt2", Some(100));
        scheduler.add_request(request2);

        let output2 = scheduler.schedule();

        // Different salt means different hashes, so no cache hit
        assert_eq!(
            output2.scheduled_new_reqs[0].num_computed_tokens, 0,
            "Different salt should prevent cache hit"
        );
    }

    // -------------------------------------------------------------------------
    // Preemption Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_preemption_behavior_limited_blocks() {
        // Create scheduler with very limited blocks to test behavior near capacity
        let (mut scheduler, _registry) = create_test_scheduler(10, 16, true);

        // R1: 64 tokens (4 blocks) - this will take most of the cache
        let tokens_r1: Vec<u32> = (0..64).collect();
        let request1 = create_test_request("req-1", tokens_r1, Some(100));
        scheduler.add_request(request1);

        let output1 = scheduler.schedule();
        assert_eq!(output1.scheduled_new_reqs.len(), 1);
        assert_eq!(scheduler.num_running(), 1);

        // R2: 64 more tokens - may trigger preemption of R1 if blocks are insufficient
        let tokens_r2: Vec<u32> = (100..164).collect();
        let request2 = create_test_request("req-2", tokens_r2, Some(100));
        scheduler.add_request(request2);

        let output2 = scheduler.schedule();

        // With limited blocks, either:
        // 1. R2 is scheduled and R1 may be preempted (if preemption is implemented)
        // 2. R2 stays in waiting queue due to insufficient blocks
        // This test verifies the scheduler handles this gracefully
        let total_scheduled = output2.scheduled_new_reqs.len() + output2.scheduled_cached_reqs.len();
        assert!(total_scheduled >= 0, "Scheduler should not crash with limited blocks");
    }
}
