// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for abort request behavior.
//!
//! These tests verify the abort request functionality across different request states:
//! - Queued (waiting) requests
//! - Running requests
//! - Preempted requests (back in waiting queue after eviction)

use super::engine::{MockEngineCore, MockEngineCoreConfig, TestRequest};

fn default_config() -> MockEngineCoreConfig {
    MockEngineCoreConfig {
        max_seq_len: 8192,
        max_num_batched_tokens: 4096,
        max_num_seqs: 128,
        block_size: 16,
        total_blocks: 512,
        seed: 42,
        vocab_size: 50257,
        enable_projection: true,
    }
}

// =============================================================================
// Scenario 1a: Abort Queued (Waiting) Request
// =============================================================================

#[test]
fn test_abort_queued_request_removes_from_waiting() {
    let mut engine = MockEngineCore::new(default_config()).expect("Should create engine");

    // Add request but don't schedule
    engine.add_request(TestRequest {
        request_id: "req-1".into(),
        prompt_tokens: (0..100).collect(),
        max_tokens: 50,
    });

    assert_eq!(engine.num_waiting(), 1);
    assert_eq!(engine.num_running(), 0);

    // Abort before scheduling
    engine.abort_request("req-1");

    // Request should be removed from waiting queue
    assert_eq!(engine.num_waiting(), 0);
    assert_eq!(engine.num_running(), 0);

    // Internal tracking should also be cleared
    assert!(!engine.requests.contains_key("req-1"));
    assert!(!engine.output_tokens.contains_key("req-1"));
}

#[test]
fn test_abort_queued_request_no_effect_on_others() {
    let mut engine = MockEngineCore::new(default_config()).expect("Should create engine");

    // Add multiple requests
    for i in 0..3 {
        engine.add_request(TestRequest {
            request_id: format!("req-{i}"),
            prompt_tokens: (0..100).collect(),
            max_tokens: 50,
        });
    }

    assert_eq!(engine.num_waiting(), 3);

    // Abort middle request
    engine.abort_request("req-1");

    // Only aborted request should be removed
    assert_eq!(engine.num_waiting(), 2);
    assert!(engine.requests.contains_key("req-0"));
    assert!(!engine.requests.contains_key("req-1"));
    assert!(engine.requests.contains_key("req-2"));
}

#[test]
fn test_abort_nonexistent_request_is_no_op() {
    let mut engine = MockEngineCore::new(default_config()).expect("Should create engine");

    engine.add_request(TestRequest {
        request_id: "req-1".into(),
        prompt_tokens: (0..100).collect(),
        max_tokens: 50,
    });

    // Abort nonexistent request
    engine.abort_request("req-does-not-exist");

    // Original request should still be there
    assert_eq!(engine.num_waiting(), 1);
    assert!(engine.requests.contains_key("req-1"));
}

// =============================================================================
// Scenario 1b: Abort Running Request
// =============================================================================

#[test]
fn test_abort_running_request() {
    let mut engine = MockEngineCore::new(default_config()).expect("Should create engine");

    engine.add_request(TestRequest {
        request_id: "req-1".into(),
        prompt_tokens: (0..100).collect(),
        max_tokens: 50,
    });

    // Schedule to start running
    engine.step();
    assert_eq!(engine.num_waiting(), 0);
    assert_eq!(engine.num_running(), 1);

    // Record cache usage before abort
    let usage_before = engine.cache_usage();
    assert!(usage_before > 0.0, "Running request should have allocated blocks");

    // Abort while running
    engine.abort_request("req-1");

    // Request should be removed
    assert_eq!(engine.num_running(), 0);
    assert!(!engine.requests.contains_key("req-1"));

    // Blocks should be freed (cache usage should drop)
    let usage_after = engine.cache_usage();
    assert!(
        usage_after < usage_before,
        "Cache usage should decrease after abort: before={usage_before}, after={usage_after}"
    );
}

#[test]
fn test_abort_running_request_with_output_tokens() {
    let mut engine = MockEngineCore::new(default_config()).expect("Should create engine");

    engine.add_request(TestRequest {
        request_id: "req-1".into(),
        prompt_tokens: (0..100).collect(),
        max_tokens: 50,
    });

    // Run a few iterations to generate some tokens
    for _ in 0..5 {
        engine.step();
    }

    assert_eq!(engine.num_running(), 1);

    // Verify we have generated some output tokens
    let output_len = engine.output_tokens.get("req-1").map(|t| t.len()).unwrap_or(0);
    assert!(output_len > 0, "Should have generated output tokens");

    // Abort mid-generation
    engine.abort_request("req-1");

    // Request and output tracking should be cleaned up
    assert_eq!(engine.num_running(), 0);
    assert!(!engine.requests.contains_key("req-1"));
    assert!(!engine.output_tokens.contains_key("req-1"));
}

#[test]
fn test_abort_one_of_multiple_running_requests() {
    let mut engine = MockEngineCore::new(default_config()).expect("Should create engine");

    // Add multiple requests
    for i in 0..3 {
        engine.add_request(TestRequest {
            request_id: format!("req-{i}"),
            prompt_tokens: (0..64).collect(), // Small enough to all fit
            max_tokens: 50,
        });
    }

    // Schedule all
    engine.step();
    assert_eq!(engine.num_running(), 3);

    // Abort one
    engine.abort_request("req-1");

    // Only aborted request should be removed
    assert_eq!(engine.num_running(), 2);
    assert!(engine.requests.contains_key("req-0"));
    assert!(!engine.requests.contains_key("req-1"));
    assert!(engine.requests.contains_key("req-2"));

    // Remaining requests should still be able to complete
    let outputs = engine.run_to_completion(1000);
    assert!(!outputs.is_empty());
    assert!(engine.finished.contains("req-0"));
    assert!(engine.finished.contains("req-2"));
}

// =============================================================================
// Scenario 1c: Abort Preempted Request
// =============================================================================

#[test]
fn test_abort_request_after_limited_blocks_pressure() {
    let mut config = default_config();
    config.total_blocks = 20; // Very limited blocks
    config.max_num_seqs = 2; // Limit concurrent sequences

    let mut engine = MockEngineCore::new(config).expect("Should create engine");

    // Add request that will use most blocks
    engine.add_request(TestRequest {
        request_id: "req-big".into(),
        prompt_tokens: (0..200).collect(), // Large prompt
        max_tokens: 50,
    });

    // Add smaller request
    engine.add_request(TestRequest {
        request_id: "req-small".into(),
        prompt_tokens: (0..32).collect(),
        max_tokens: 10,
    });

    // Run a few steps
    for _ in 0..3 {
        if engine.step().is_none() {
            break;
        }
    }

    // Abort the big request
    engine.abort_request("req-big");

    // Small request should still be able to proceed
    let outputs = engine.run_to_completion(100);
    assert!(!outputs.is_empty() || engine.finished.contains("req-small"));
}

#[test]
fn test_abort_then_add_new_request() {
    let mut engine = MockEngineCore::new(default_config()).expect("Should create engine");

    // Add and schedule request
    engine.add_request(TestRequest {
        request_id: "req-1".into(),
        prompt_tokens: (0..100).collect(),
        max_tokens: 50,
    });
    engine.step();

    // Abort it
    engine.abort_request("req-1");

    // Add new request with same ID
    engine.add_request(TestRequest {
        request_id: "req-1".into(),
        prompt_tokens: (0..50).collect(),
        max_tokens: 20,
    });

    // Should be able to schedule and complete
    let outputs = engine.run_to_completion(100);
    assert!(!outputs.is_empty());
    assert!(engine.finished.contains("req-1"));
    assert_eq!(engine.output_tokens["req-1"].len(), 20);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_abort_already_finished_request() {
    let mut engine = MockEngineCore::new(default_config()).expect("Should create engine");

    engine.add_request(TestRequest {
        request_id: "req-1".into(),
        prompt_tokens: (0..50).collect(),
        max_tokens: 5, // Very short - will finish quickly
    });

    // Run to completion
    engine.run_to_completion(100);
    assert!(engine.finished.contains("req-1"));

    // Try to abort finished request (should be no-op)
    let running_before = engine.num_running();
    let waiting_before = engine.num_waiting();

    engine.abort_request("req-1");

    // State should be unchanged
    assert_eq!(engine.num_running(), running_before);
    assert_eq!(engine.num_waiting(), waiting_before);
}

#[test]
fn test_abort_all_requests() {
    let mut engine = MockEngineCore::new(default_config()).expect("Should create engine");

    // Add several requests
    for i in 0..5 {
        engine.add_request(TestRequest {
            request_id: format!("req-{i}"),
            prompt_tokens: (0..64).collect(),
            max_tokens: 30,
        });
    }

    // Schedule them
    engine.step();

    // Abort all
    for i in 0..5 {
        engine.abort_request(&format!("req-{i}"));
    }

    // Everything should be empty
    assert_eq!(engine.num_waiting(), 0);
    assert_eq!(engine.num_running(), 0);
    assert!(engine.requests.is_empty());
    assert!(engine.output_tokens.is_empty());
}
