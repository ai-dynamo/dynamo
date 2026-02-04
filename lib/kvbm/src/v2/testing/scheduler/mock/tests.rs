// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the mock engine core.

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
        enable_connector: false,
    }
}

#[test]
fn test_single_request_lifecycle() {
    let mut engine = MockEngineCore::new(default_config()).expect("Should create engine");

    engine.add_request(TestRequest {
        request_id: "test-1".into(),
        prompt_tokens: (0..100).collect(),
        max_tokens: 50,
    });

    let outputs = engine.run_to_completion(1000);

    // Should complete in roughly max_tokens iterations (plus prefill)
    assert!(!outputs.is_empty());
    assert!(engine.finished.contains("test-1"));
    assert_eq!(engine.output_tokens["test-1"].len(), 50);
}

#[test]
fn test_deterministic_output() {
    let request = TestRequest {
        request_id: "test-1".into(),
        prompt_tokens: (0..50).collect(),
        max_tokens: 20,
    };

    // Run 1
    let mut engine1 = MockEngineCore::new(default_config()).expect("Should create engine");
    engine1.add_request(request.clone());
    engine1.run_to_completion(1000);
    let tokens1 = engine1.output_tokens["test-1"].clone();

    // Run 2 with same seed
    let mut engine2 = MockEngineCore::new(default_config()).expect("Should create engine");
    engine2.add_request(request);
    engine2.run_to_completion(1000);
    let tokens2 = engine2.output_tokens["test-1"].clone();

    assert_eq!(tokens1, tokens2, "Same seed should produce same tokens");
}

#[test]
fn test_different_seeds_different_output() {
    let request = TestRequest {
        request_id: "test-1".into(),
        prompt_tokens: (0..50).collect(),
        max_tokens: 20,
    };

    // Run with seed 42
    let mut config1 = default_config();
    config1.seed = 42;
    let mut engine1 = MockEngineCore::new(config1).expect("Should create engine");
    engine1.add_request(request.clone());
    engine1.run_to_completion(1000);
    let tokens1 = engine1.output_tokens["test-1"].clone();

    // Run with different seed
    let mut config2 = default_config();
    config2.seed = 99;
    let mut engine2 = MockEngineCore::new(config2).expect("Should create engine");
    engine2.add_request(request);
    engine2.run_to_completion(1000);
    let tokens2 = engine2.output_tokens["test-1"].clone();

    // Very unlikely to be the same
    assert_ne!(
        tokens1, tokens2,
        "Different seeds should produce different tokens"
    );
}

#[test]
fn test_concurrent_requests() {
    let mut engine = MockEngineCore::new(default_config()).expect("Should create engine");

    // Add multiple concurrent requests
    for i in 0..5 {
        engine.add_request(TestRequest {
            request_id: format!("req-{i}"),
            prompt_tokens: (0..(100 + i * 10) as u32).collect(),
            max_tokens: 30,
        });
    }

    let outputs = engine.run_to_completion(1000);

    // All should complete
    assert!(!outputs.is_empty());
    for i in 0..5 {
        assert!(
            engine.finished.contains(&format!("req-{i}")),
            "Request req-{i} should be finished"
        );
        assert_eq!(
            engine.output_tokens[&format!("req-{i}")].len(),
            30,
            "Request req-{i} should have 30 output tokens"
        );
    }
}

#[test]
fn test_projection_enabled() {
    let mut config = default_config();
    config.enable_projection = true;

    let mut engine = MockEngineCore::new(config).expect("Should create engine");

    engine.add_request(TestRequest {
        request_id: "test-1".into(),
        prompt_tokens: (0..100).collect(),
        max_tokens: 50,
    });

    // Run a few iterations
    for _ in 0..5 {
        engine.step();
    }

    // Projection state should be available
    assert!(
        engine.projection_state().is_some(),
        "Projection state should be available when enabled"
    );
}

#[test]
fn test_projection_disabled() {
    let mut config = default_config();
    config.enable_projection = false;

    let mut engine = MockEngineCore::new(config).expect("Should create engine");

    engine.add_request(TestRequest {
        request_id: "test-1".into(),
        prompt_tokens: (0..100).collect(),
        max_tokens: 50,
    });

    // Run a few iterations
    for _ in 0..5 {
        engine.step();
    }

    // Projection state should not be available
    assert!(
        engine.projection_state().is_none(),
        "Projection state should be None when disabled"
    );
}

#[test]
fn test_projection_choke_point_detection() {
    let mut config = default_config();
    config.total_blocks = 20; // Limited blocks to force memory pressure
    config.enable_projection = true;

    let mut engine = MockEngineCore::new(config).expect("Should create engine");

    // Add multiple requests that will exhaust blocks
    for i in 0..5 {
        engine.add_request(TestRequest {
            request_id: format!("req-{i}"),
            prompt_tokens: (0..100).collect(),
            max_tokens: 50,
        });
    }

    // Run enough iterations to trigger memory pressure
    for _ in 0..20 {
        if engine.step().is_none() {
            break;
        }
    }

    // With limited blocks and multiple requests, we should see high cache usage
    let usage = engine.cache_usage();
    assert!(
        usage > 0.5,
        "Cache usage should be high with limited blocks, got {usage}"
    );
}

#[test]
fn test_step_returns_none_when_empty() {
    let mut engine = MockEngineCore::new(default_config()).expect("Should create engine");

    // No requests added
    assert!(
        engine.step().is_none(),
        "Step should return None when no requests"
    );
}

#[test]
fn test_step_output_contains_scheduled_requests() {
    let mut engine = MockEngineCore::new(default_config()).expect("Should create engine");

    engine.add_request(TestRequest {
        request_id: "test-1".into(),
        prompt_tokens: (0..100).collect(),
        max_tokens: 50,
    });

    let output = engine.step().expect("Should have output");

    // First step schedules the new request
    assert_eq!(output.schedule_output.scheduled_new_reqs.len(), 1);
    assert_eq!(
        output.schedule_output.scheduled_new_reqs[0].req_id,
        "test-1"
    );
}

#[test]
fn test_finished_requests_removed_from_model_output() {
    let mut engine = MockEngineCore::new(default_config()).expect("Should create engine");

    engine.add_request(TestRequest {
        request_id: "test-1".into(),
        prompt_tokens: (0..50).collect(),
        max_tokens: 5, // Very short to finish quickly
    });

    let outputs = engine.run_to_completion(100);

    // Should finish after max_tokens decode steps (plus prefill)
    assert!(engine.finished.contains("test-1"));
    assert_eq!(engine.output_tokens["test-1"].len(), 5);

    // After completion, no more steps should be executed
    let last_output = outputs.last().unwrap();
    assert!(
        !last_output.finished.is_empty() || outputs.len() > 5,
        "Request should finish or we should have enough iterations"
    );
}

#[test]
fn test_iteration_counter() {
    let mut engine = MockEngineCore::new(default_config()).expect("Should create engine");

    engine.add_request(TestRequest {
        request_id: "test-1".into(),
        prompt_tokens: (0..100).collect(),
        max_tokens: 10,
    });

    assert_eq!(engine.iteration(), 0, "Initial iteration should be 0");

    engine.step();
    assert_eq!(
        engine.iteration(),
        1,
        "After first step, iteration should be 1"
    );

    engine.step();
    assert_eq!(
        engine.iteration(),
        2,
        "After second step, iteration should be 2"
    );
}

#[test]
fn test_accessors() {
    let config = default_config();
    let mut engine = MockEngineCore::new(config.clone()).expect("Should create engine");

    engine.add_request(TestRequest {
        request_id: "test-1".into(),
        prompt_tokens: (0..100).collect(),
        max_tokens: 50,
    });

    // Before scheduling
    assert_eq!(engine.num_waiting(), 1);
    assert_eq!(engine.num_running(), 0);

    // After scheduling
    engine.step();
    assert_eq!(engine.num_waiting(), 0);
    assert_eq!(engine.num_running(), 1);

    // Config accessor
    assert_eq!(engine.config().block_size, config.block_size);
    assert_eq!(engine.config().seed, config.seed);
}
