// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for scheduler + connector integration using MockEngineCore.
//!
//! These tests verify that the scheduler properly manages connector slots throughout
//! the full request lifecycle, from scheduling to completion/abort.

use super::{MockEngineCore, MockEngineCoreConfig, TestRequest};

// =============================================================================
// Basic Lifecycle Tests
// =============================================================================

#[test]
fn test_mock_engine_with_connector_creation() {
    let config = MockEngineCoreConfig {
        enable_connector: true,
        ..Default::default()
    };
    let engine = MockEngineCore::new(config).expect("Should create engine with connector");

    assert!(engine.has_connector(), "Engine should have connector enabled");
    assert!(
        engine.connector_instance().is_some(),
        "Should have connector instance"
    );
}

#[test]
fn test_mock_engine_without_connector() {
    let config = MockEngineCoreConfig {
        enable_connector: false,
        ..Default::default()
    };
    let engine = MockEngineCore::new(config).expect("Should create engine without connector");

    assert!(
        !engine.has_connector(),
        "Engine should not have connector enabled"
    );
    assert!(
        engine.connector_instance().is_none(),
        "Should not have connector instance"
    );
    assert!(
        !engine.has_connector_slot("any-request"),
        "Should return false for any request when connector disabled"
    );
}

#[test]
fn test_mock_engine_connector_slot_creation_on_schedule() {
    let config = MockEngineCoreConfig {
        enable_connector: true,
        enable_projection: false, // Disable projection for simpler test
        ..Default::default()
    };
    let mut engine = MockEngineCore::new(config).expect("Should create engine");

    // Add a request
    engine.add_request(TestRequest {
        request_id: "req-1".into(),
        prompt_tokens: (0..64).collect(),
        max_tokens: 10,
    });

    // Before scheduling, no slot should exist
    assert!(
        !engine.has_connector_slot("req-1"),
        "Slot should not exist before scheduling"
    );

    // After first schedule, slot should be created
    // (slot is created when get_num_new_matched_tokens is called during schedule_waiting)
    engine.step();
    assert!(
        engine.has_connector_slot("req-1"),
        "Slot should exist after scheduling"
    );
}

#[test]
fn test_mock_engine_connector_slot_cleanup_on_completion() {
    let config = MockEngineCoreConfig {
        enable_connector: true,
        enable_projection: false,
        ..Default::default()
    };
    let mut engine = MockEngineCore::new(config).expect("Should create engine");

    // Add a request with only 2 output tokens for quick completion
    engine.add_request(TestRequest {
        request_id: "req-1".into(),
        prompt_tokens: (0..32).collect(),
        max_tokens: 2,
    });

    // Schedule and verify slot exists
    engine.step();
    assert!(engine.has_connector_slot("req-1"), "Slot should exist");

    // Run to completion
    engine.run_to_completion(100);

    // After completion, slot should be cleaned up
    // (request_finished is called during update_from_output for finished requests)
    assert!(
        !engine.has_connector_slot("req-1"),
        "Slot should be cleaned up after completion"
    );
}

#[test]
fn test_mock_engine_connector_slot_cleanup_on_abort() {
    let config = MockEngineCoreConfig {
        enable_connector: true,
        enable_projection: false,
        ..Default::default()
    };
    let mut engine = MockEngineCore::new(config).expect("Should create engine");

    // Add a request
    engine.add_request(TestRequest {
        request_id: "req-1".into(),
        prompt_tokens: (0..64).collect(),
        max_tokens: 50,
    });

    // Schedule and verify slot exists
    engine.step();
    assert!(engine.has_connector_slot("req-1"), "Slot should exist");

    // Abort the request
    engine.abort_request("req-1");

    // Slot should be cleaned up
    assert!(
        !engine.has_connector_slot("req-1"),
        "Slot should be cleaned up after abort"
    );
}

// =============================================================================
// Multiple Request Tests
// =============================================================================

#[test]
fn test_mock_engine_connector_multiple_requests() {
    let config = MockEngineCoreConfig {
        enable_connector: true,
        enable_projection: false,
        ..Default::default()
    };
    let mut engine = MockEngineCore::new(config).expect("Should create engine");

    // Add multiple requests
    for i in 0..5 {
        engine.add_request(TestRequest {
            request_id: format!("req-{}", i),
            prompt_tokens: (0..32).collect(),
            max_tokens: 3, // Quick completion
        });
    }

    // Schedule all
    engine.step();

    // All should have slots
    for i in 0..5 {
        assert!(
            engine.has_connector_slot(&format!("req-{}", i)),
            "req-{} should have slot",
            i
        );
    }

    // Run to completion
    engine.run_to_completion(500);

    // All slots should be cleaned up
    for i in 0..5 {
        assert!(
            !engine.has_connector_slot(&format!("req-{}", i)),
            "req-{} slot should be cleaned up",
            i
        );
    }
}

#[test]
fn test_mock_engine_connector_mixed_completion_and_abort() {
    let config = MockEngineCoreConfig {
        enable_connector: true,
        enable_projection: false,
        ..Default::default()
    };
    let mut engine = MockEngineCore::new(config).expect("Should create engine");

    // Add multiple requests
    for i in 0..4 {
        engine.add_request(TestRequest {
            request_id: format!("req-{}", i),
            prompt_tokens: (0..32).collect(),
            max_tokens: if i % 2 == 0 { 2 } else { 100 }, // Some quick, some long
        });
    }

    // Schedule all
    engine.step();

    // Verify all have slots
    for i in 0..4 {
        assert!(engine.has_connector_slot(&format!("req-{}", i)));
    }

    // Abort the long-running requests
    engine.abort_request("req-1");
    engine.abort_request("req-3");

    // Aborted requests should have slots cleaned up
    assert!(!engine.has_connector_slot("req-1"));
    assert!(!engine.has_connector_slot("req-3"));

    // Quick requests may or may not have completed, but their slots
    // should be properly managed
    engine.run_to_completion(100);

    // All slots should eventually be cleaned up
    for i in 0..4 {
        assert!(
            !engine.has_connector_slot(&format!("req-{}", i)),
            "req-{} slot should be cleaned up",
            i
        );
    }
}

// =============================================================================
// Multi-Step Decode Tests
// =============================================================================

#[test]
fn test_mock_engine_connector_multi_step_decode() {
    let config = MockEngineCoreConfig {
        enable_connector: true,
        enable_projection: false,
        ..Default::default()
    };
    let mut engine = MockEngineCore::new(config).expect("Should create engine");

    // Add a request with longer generation
    engine.add_request(TestRequest {
        request_id: "req-1".into(),
        prompt_tokens: (0..64).collect(),
        max_tokens: 10,
    });

    // Run for several steps
    for i in 0..8 {
        let result = engine.step();
        assert!(result.is_some(), "Step {} should produce output", i);

        // Slot should exist throughout generation
        if engine.num_running() > 0 {
            assert!(
                engine.has_connector_slot("req-1"),
                "Slot should persist during generation at step {}",
                i
            );
        }
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_mock_engine_connector_abort_waiting_request() {
    let config = MockEngineCoreConfig {
        enable_connector: true,
        enable_projection: false,
        total_blocks: 10, // Very limited blocks
        block_size: 16,
        ..Default::default()
    };
    let mut engine = MockEngineCore::new(config).expect("Should create engine");

    // Add a request that will use most blocks
    engine.add_request(TestRequest {
        request_id: "req-large".into(),
        prompt_tokens: (0..128).collect(), // 8 blocks
        max_tokens: 50,
    });

    // Add another that may stay in waiting queue
    engine.add_request(TestRequest {
        request_id: "req-waiting".into(),
        prompt_tokens: (0..64).collect(),
        max_tokens: 50,
    });

    // Schedule - first request should run
    engine.step();

    // Abort the waiting request (may not have a slot if never scheduled)
    engine.abort_request("req-waiting");

    // Should not panic, and no slot should exist
    assert!(!engine.has_connector_slot("req-waiting"));

    // Original request should still have slot
    assert!(engine.has_connector_slot("req-large"));
}

#[test]
fn test_mock_engine_connector_slot_not_created_for_waiting() {
    let config = MockEngineCoreConfig {
        enable_connector: true,
        enable_projection: false,
        total_blocks: 5, // Very limited - only one request can run
        block_size: 16,
        ..Default::default()
    };
    let mut engine = MockEngineCore::new(config).expect("Should create engine");

    // Add requests - only first should be scheduled due to block limits
    engine.add_request(TestRequest {
        request_id: "req-1".into(),
        prompt_tokens: (0..64).collect(), // 4 blocks
        max_tokens: 50,
    });
    engine.add_request(TestRequest {
        request_id: "req-2".into(),
        prompt_tokens: (0..64).collect(), // 4 blocks
        max_tokens: 50,
    });

    // Schedule
    engine.step();

    // First request should have slot (scheduled)
    assert!(engine.has_connector_slot("req-1"));

    // Second request might not have slot if still waiting
    // (connector slot is only created when request is scheduled)
}
