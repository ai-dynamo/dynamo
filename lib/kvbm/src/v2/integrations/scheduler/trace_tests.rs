// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trace-based unit tests for the scheduler.
//!
//! These tests are derived from real execution traces captured by the `RecordingScheduler`.
//! Each test is a "micro-snapshot" that verifies a specific state transition:
//!
//! ```text
//! Setup (initial state) → Action → Assert (expected state)
//! ```
//!
//! # Trace Source
//!
//! Traces are captured using `.sandbox/capture_scheduler_trace.sh` which runs the
//! RecordingScheduler wrapper around vLLM's scheduler. The traces record:
//!
//! - `schedule_output`: Scheduler decisions (scheduled requests, tokens, blocks)
//! - `model_runner_output`: Model outputs (sampled tokens per request)
//! - `engine_core_outputs`: Final outputs with finish reasons and stats
//!
//! # Test Categories
//!
//! 1. **Fresh request scheduling**: New request enters the scheduler
//! 2. **Decode iteration**: Running request continues generating tokens
//! 3. **Forward pass completion**: State update after model output received
//! 4. **Request completion**: Request finishes (stop token or max tokens)

#[cfg(test)]
mod tests {
    use crate::v2::integrations::common::Request;
    use crate::v2::integrations::scheduler::request::{RequestStatus, SchedulerRequest};

    const BLOCK_SIZE: usize = 16;

    // =========================================================================
    // Trace: scheduler_trace_20260103_175657.json
    // Model: gpt2
    // Description: Single request completing 17 iterations
    // =========================================================================

    /// Test derived from iteration 0: Fresh request scheduling
    ///
    /// Trace state:
    /// - prompt_token_ids: [151644, 872, 198, 45764, 23811, 304, 6896, 2326, 4244, 13, 151645, 198, 151644, 77091, 198]
    /// - block_ids: [[1]]
    /// - num_computed_tokens: 0
    /// - num_scheduled_tokens: 15
    #[test]
    fn test_fresh_request_iteration_0() {
        // Setup: Create request with exact prompt from trace
        let prompt_tokens: Vec<u32> = vec![
            151644, 872, 198, 45764, 23811, 304, 6896, 2326, 4244, 13, 151645, 198, 151644, 77091,
            198,
        ];
        let request = Request::new(
            "chatcmpl-02f123d0e222426ebc6bbc8b3ed0f04a",
            prompt_tokens.clone(),
            None,
            None,
            Some(100),
        );
        let sched_req = SchedulerRequest::new(request, BLOCK_SIZE);

        // Assert initial state matches trace expectations
        assert_eq!(sched_req.total_known_tokens(), 15);
        assert_eq!(sched_req.num_computed_tokens, 0);
        assert!(sched_req.is_prefilling());
        assert_eq!(sched_req.tokens_to_compute(), 15);
        assert_eq!(sched_req.status, RequestStatus::Waiting);

        // Verify block calculation (ceil(15/16) = 1 block needed)
        assert_eq!(sched_req.num_blocks_required(BLOCK_SIZE), 1);
    }

    /// Test derived from iteration 0→1: First decode after prefill
    ///
    /// Trace state at iteration 1:
    /// - scheduled_cached_reqs.num_computed_tokens: [15]
    /// - num_scheduled_tokens: 1
    /// - sampled_token_ids: [[151667]]
    #[test]
    fn test_decode_iteration_1() {
        // Setup: Request after prefill complete (15 tokens computed)
        let prompt_tokens: Vec<u32> = vec![
            151644, 872, 198, 45764, 23811, 304, 6896, 2326, 4244, 13, 151645, 198, 151644, 77091,
            198,
        ];
        let request = Request::new("req-decode-1", prompt_tokens.clone(), None, None, Some(100));
        let mut sched_req = SchedulerRequest::new(request, BLOCK_SIZE);

        // Simulate state after prefill: all prompt tokens computed
        sched_req.apply_cache_matches(15, 0);
        sched_req.status = RequestStatus::Running;

        // Verify pre-decode state
        assert_eq!(sched_req.num_computed_tokens, 15);
        assert!(!sched_req.is_prefilling()); // Past prefill
        assert_eq!(sched_req.tokens_to_compute(), 0); // All computed

        // Action: Model outputs 1 token (from trace: 151667)
        let output_token: Vec<u32> = vec![151667];
        let tokens_before_output = sched_req.total_known_tokens();
        assert_eq!(tokens_before_output, 15);

        sched_req.extend_tokens(&output_token).unwrap();
        sched_req.add_output_tokens(1);
        sched_req.apply_forward_pass_completion(tokens_before_output);

        // Assert post-decode state
        assert_eq!(sched_req.total_known_tokens(), 16); // 15 + 1
        assert_eq!(sched_req.num_output_tokens, 1);
        assert_eq!(sched_req.num_computed_tokens, 15); // Was 15 BEFORE output
        assert_eq!(sched_req.tokens_to_compute(), 1); // Need to compute the new token
    }

    /// Test derived from iteration 2: Continued decode
    ///
    /// Trace state at iteration 2:
    /// - num_computed_tokens: [16] (15 prompt + 1 output from iter 1)
    /// - num_scheduled_tokens: 1
    /// - sampled_token_ids: [[198]]
    #[test]
    fn test_decode_iteration_2() {
        // Setup: Request after first decode (16 tokens, 1 output)
        let prompt_tokens: Vec<u32> = vec![
            151644, 872, 198, 45764, 23811, 304, 6896, 2326, 4244, 13, 151645, 198, 151644, 77091,
            198,
        ];
        let request = Request::new("req-decode-2", prompt_tokens.clone(), None, None, Some(100));
        let mut sched_req = SchedulerRequest::new(request, BLOCK_SIZE);

        // Setup state after iteration 1
        let first_output: Vec<u32> = vec![151667];
        sched_req.extend_tokens(&first_output).unwrap();
        sched_req.add_output_tokens(1);
        sched_req.apply_cache_matches(16, 0); // Now 16 tokens computed
        sched_req.status = RequestStatus::Running;

        // Verify state matches trace iteration 2 input
        assert_eq!(sched_req.total_known_tokens(), 16);
        assert_eq!(sched_req.num_computed_tokens, 16);
        assert!(!sched_req.is_prefilling());
        assert_eq!(sched_req.tokens_to_compute(), 0);

        // Action: Model outputs second token (from trace: 198)
        let output_token: Vec<u32> = vec![198];
        let tokens_before = sched_req.total_known_tokens();
        sched_req.extend_tokens(&output_token).unwrap();
        sched_req.add_output_tokens(1);
        sched_req.apply_forward_pass_completion(tokens_before);

        // Assert: Now at 17 tokens
        assert_eq!(sched_req.total_known_tokens(), 17);
        assert_eq!(sched_req.num_output_tokens, 2);
        assert_eq!(sched_req.num_computed_tokens, 16);
        assert_eq!(sched_req.tokens_to_compute(), 1);

        // Note: Still only 1 block allocated (16 tokens per block)
        // Will need 2nd block after 16th token
        assert_eq!(sched_req.num_blocks_required(BLOCK_SIZE), 2);
    }

    /// Test block boundary crossing
    ///
    /// Derived from iteration 1→2 where token count crosses from 16 to 17,
    /// requiring a second block.
    #[test]
    fn test_block_boundary_crossing() {
        let prompt_tokens: Vec<u32> = (0..15).collect();
        let request = Request::new("req-boundary", prompt_tokens.clone(), None, None, Some(100));
        let mut sched_req = SchedulerRequest::new(request, BLOCK_SIZE);

        // At 15 tokens: 1 block needed
        assert_eq!(sched_req.num_blocks_required(BLOCK_SIZE), 1);

        // Add output to reach 16 tokens (still 1 block)
        let output1: Vec<u32> = vec![100];
        sched_req.extend_tokens(&output1).unwrap();
        sched_req.add_output_tokens(1);
        assert_eq!(sched_req.total_known_tokens(), 16);
        assert_eq!(sched_req.num_blocks_required(BLOCK_SIZE), 1);

        // Add another token to reach 17 (needs 2 blocks)
        let output2: Vec<u32> = vec![101];
        sched_req.extend_tokens(&output2).unwrap();
        sched_req.add_output_tokens(1);
        assert_eq!(sched_req.total_known_tokens(), 17);
        assert_eq!(sched_req.num_blocks_required(BLOCK_SIZE), 2);
    }

    // =========================================================================
    // apply_cache_matches() and apply_forward_pass_completion() tests
    // =========================================================================

    /// Test apply_cache_matches with local cache only
    #[test]
    fn test_apply_cache_matches_local_only() {
        let prompt_tokens: Vec<u32> = (0..32).collect();
        let request = Request::new("req-cache-local", prompt_tokens, None, None, Some(100));
        let mut sched_req = SchedulerRequest::new(request, BLOCK_SIZE);

        // Simulate 16 tokens found in local prefix cache
        sched_req.apply_cache_matches(16, 0);

        assert_eq!(sched_req.num_computed_tokens, 16);
        assert!(sched_req.is_prefilling()); // Still 16 more tokens to compute
        assert_eq!(sched_req.tokens_to_compute(), 16);
    }

    /// Test apply_cache_matches with external (G2) cache
    #[test]
    fn test_apply_cache_matches_with_external() {
        let prompt_tokens: Vec<u32> = (0..64).collect();
        let request = Request::new("req-cache-ext", prompt_tokens, None, None, Some(100));
        let mut sched_req = SchedulerRequest::new(request, BLOCK_SIZE);

        // Simulate 16 local + 32 external cached tokens
        sched_req.apply_cache_matches(16, 32);

        assert_eq!(sched_req.num_computed_tokens, 48);
        assert!(sched_req.is_prefilling()); // Still 16 more to compute
        assert_eq!(sched_req.tokens_to_compute(), 16);
    }

    /// Test apply_forward_pass_completion after decode
    #[test]
    fn test_apply_forward_pass_completion() {
        let prompt_tokens: Vec<u32> = (0..16).collect();
        let request = Request::new("req-fwd-pass", prompt_tokens, None, None, Some(100));
        let mut sched_req = SchedulerRequest::new(request, BLOCK_SIZE);

        // Setup: prefill complete
        sched_req.apply_cache_matches(16, 0);
        sched_req.status = RequestStatus::Running;

        // Simulate decode: output 3 tokens
        let output: Vec<u32> = vec![100, 101, 102];
        let tokens_before = sched_req.total_known_tokens();
        assert_eq!(tokens_before, 16);

        sched_req.extend_tokens(&output).unwrap();
        sched_req.add_output_tokens(3);
        sched_req.apply_forward_pass_completion(tokens_before);

        // After forward pass: computed = tokens_before (16)
        assert_eq!(sched_req.num_computed_tokens, 16);
        assert_eq!(sched_req.total_known_tokens(), 19);
        assert_eq!(sched_req.tokens_to_compute(), 3); // 3 new tokens need compute
    }

    // =========================================================================
    // Preemption and resume tests
    // =========================================================================

    /// Test preemption resets computed tokens
    #[test]
    fn test_preemption_resets_state() {
        let prompt_tokens: Vec<u32> = (0..32).collect();
        let request = Request::new("req-preempt", prompt_tokens, None, None, Some(100));
        let mut sched_req = SchedulerRequest::new(request, BLOCK_SIZE);

        // Setup: request running with 20 computed tokens
        sched_req.apply_cache_matches(20, 0);
        sched_req.status = RequestStatus::Running;

        // Add some output
        let output: Vec<u32> = vec![100, 101];
        sched_req.extend_tokens(&output).unwrap();
        sched_req.add_output_tokens(2);
        sched_req.apply_forward_pass_completion(20);

        // Verify pre-preemption state
        assert_eq!(sched_req.total_known_tokens(), 34); // 32 + 2
        assert_eq!(sched_req.num_computed_tokens, 20);

        // Action: Preempt
        sched_req.preempt();

        // Assert: State after preemption
        assert_eq!(sched_req.status, RequestStatus::Preempted);
        assert_eq!(sched_req.num_computed_tokens, 0); // Reset!
        assert_eq!(sched_req.total_known_tokens(), 34); // Token sequence preserved
        assert!(sched_req.is_prefilling()); // Must re-prefill all 34 tokens
        assert_eq!(sched_req.tokens_to_compute(), 34);
    }

    /// Test resumed request needs full recompute
    ///
    /// After preemption and resume, the request must recompute ALL tokens,
    /// not just the original prompt. This is a key insight from the API redesign.
    #[test]
    fn test_resumed_request_full_recompute() {
        let prompt_tokens: Vec<u32> = (0..16).collect();
        let request = Request::new("req-resume", prompt_tokens, None, None, Some(100));
        let mut sched_req = SchedulerRequest::new(request, BLOCK_SIZE);

        // Setup: Generate 48 output tokens (total 64 tokens)
        sched_req.apply_cache_matches(16, 0);
        sched_req.status = RequestStatus::Running;

        let output: Vec<u32> = (100..148).collect(); // 48 tokens
        sched_req.extend_tokens(&output).unwrap();
        sched_req.add_output_tokens(48);
        sched_req.apply_forward_pass_completion(16);

        // Verify: 64 total tokens
        assert_eq!(sched_req.total_known_tokens(), 64);
        assert_eq!(sched_req.original_prompt_len(), 16); // Original prompt unchanged

        // Preempt and resume
        sched_req.preempt();
        sched_req.resume();

        // Assert: After resume, must recompute ALL 64 tokens
        assert_eq!(sched_req.status, RequestStatus::Waiting);
        assert_eq!(sched_req.num_computed_tokens, 0);
        assert_eq!(sched_req.tokens_to_compute(), 64); // Full recompute!
        assert!(sched_req.is_prefilling());
        assert!(sched_req.resumed_from_preemption);

        // Key insight: original_prompt_len() vs total_known_tokens()
        // - original_prompt_len() = 16 (for prefix cache lookup)
        // - total_known_tokens() = 64 (for scheduling)
        assert_eq!(sched_req.original_prompt_len(), 16);
        assert_eq!(sched_req.total_known_tokens(), 64);
    }

    // =========================================================================
    // remaining_prefill() vs is_prefilling() tests
    // =========================================================================

    /// Test remaining_prefill returns None when not prefilling
    #[test]
    fn test_remaining_prefill_in_decode_phase() {
        let prompt_tokens: Vec<u32> = (0..16).collect();
        let request = Request::new("req-decode-phase", prompt_tokens, None, None, Some(100));
        let mut sched_req = SchedulerRequest::new(request, BLOCK_SIZE);

        // Complete prefill
        sched_req.apply_cache_matches(16, 0);

        // In decode phase: remaining_prefill returns None
        assert!(!sched_req.is_prefilling());
        assert_eq!(sched_req.remaining_prefill(), None);
    }

    /// Test remaining_prefill returns Some during prefill
    #[test]
    fn test_remaining_prefill_during_prefill() {
        let prompt_tokens: Vec<u32> = (0..64).collect();
        let request = Request::new("req-in-prefill", prompt_tokens, None, None, Some(100));
        let mut sched_req = SchedulerRequest::new(request, BLOCK_SIZE);

        // Partial prefill: 32 of 64 tokens computed
        sched_req.apply_cache_matches(32, 0);

        assert!(sched_req.is_prefilling());
        assert_eq!(sched_req.remaining_prefill(), Some(32));
    }

    // =========================================================================
    // max_total_tokens and remaining_output_capacity tests
    // =========================================================================

    /// Test max_total_tokens calculation
    #[test]
    fn test_max_total_tokens() {
        let prompt_tokens: Vec<u32> = (0..100).collect();
        let request = Request::new("req-max", prompt_tokens, None, None, Some(50)); // max 50 outputs
        let sched_req = SchedulerRequest::new(request, BLOCK_SIZE);

        // max_total_tokens = prompt (100) + max_output (50) = 150
        let max_seq_len = 2048;
        assert_eq!(sched_req.max_total_tokens(max_seq_len), 150);
    }

    /// Test remaining_output_capacity
    #[test]
    fn test_remaining_output_capacity() {
        let prompt_tokens: Vec<u32> = (0..16).collect();
        let request = Request::new("req-capacity", prompt_tokens, None, None, Some(100));
        let mut sched_req = SchedulerRequest::new(request, BLOCK_SIZE);

        // Initially: 100 remaining
        assert_eq!(sched_req.remaining_output_capacity(), 100);

        // After 30 outputs: 70 remaining
        sched_req.add_output_tokens(30);
        assert_eq!(sched_req.remaining_output_capacity(), 70);

        // After 100 outputs: 0 remaining
        sched_req.add_output_tokens(70);
        assert_eq!(sched_req.remaining_output_capacity(), 0);
    }
}
