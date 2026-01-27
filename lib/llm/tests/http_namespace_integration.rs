// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for HTTP service namespace discovery functionality.
//! These tests verify that the HTTP service correctly filters models based on namespace configuration.

use dynamo_llm::namespace::{GLOBAL_NAMESPACE, is_global_namespace};
use dynamo_runtime::protocols::EndpointId;

// Helper function to create a test ModelDeploymentCard
fn create_test_endpoint(namespace: &str, component: &str, endpoint_name: &str) -> EndpointId {
    EndpointId {
        namespace: namespace.to_string(),
        component: component.to_string(),
        name: endpoint_name.to_string(),
    }
}

#[test]
fn test_endpoint_id_namespace_extraction() {
    // Test endpoint ID parsing for different namespace formats
    let test_cases = vec![
        ("vllm-agg.frontend.http", "vllm-agg", "frontend", "http"),
        (
            "sglang-prod.backend.generate",
            "sglang-prod",
            "backend",
            "generate",
        ),
        ("dynamo.frontend.http", "dynamo", "frontend", "http"),
        (
            "tensorrt-llm.backend.inference",
            "tensorrt-llm",
            "backend",
            "inference",
        ),
        (
            "test-namespace.component.endpoint",
            "test-namespace",
            "component",
            "endpoint",
        ),
    ];

    for (endpoint_str, expected_namespace, expected_component, expected_name) in test_cases {
        let endpoint: EndpointId = endpoint_str.parse().expect("Failed to parse endpoint");

        assert_eq!(endpoint.namespace, expected_namespace);
        assert_eq!(endpoint.component, expected_component);
        assert_eq!(endpoint.name, expected_name);

        // Test namespace classification
        let is_global = is_global_namespace(&endpoint.namespace);
        if expected_namespace == GLOBAL_NAMESPACE {
            assert!(
                is_global,
                "Namespace '{}' should be classified as global",
                expected_namespace
            );
        } else {
            assert!(
                !is_global,
                "Namespace '{}' should not be classified as global",
                expected_namespace
            );
        }
    }
}

#[test]
fn test_model_discovery_scoping_scenarios() {
    // Test various scenarios for model discovery scoping

    // Scenario 1: Frontend configured for specific namespace should only see models from that namespace
    let frontend_namespace = "vllm-agg";
    let available_models = vec![
        create_test_endpoint("vllm-agg", "backend", "generate"),
        create_test_endpoint("vllm-agg", "backend", "generate"),
        create_test_endpoint("sglang-prod", "backend", "generate"),
        create_test_endpoint("dynamo", "backend", "generate"),
    ];

    let visible_models: Vec<&EndpointId> = available_models
        .iter()
        .filter(|endpoint| {
            let is_global = is_global_namespace(frontend_namespace);
            is_global || endpoint.namespace == frontend_namespace
        })
        .collect();

    assert_eq!(visible_models.len(), 2);
    assert!(visible_models.iter().all(|m| m.namespace == "vllm-agg"));

    // Scenario 2: Frontend configured for global namespace should see all models
    let frontend_namespace = GLOBAL_NAMESPACE;
    let visible_models_global: Vec<&EndpointId> = available_models
        .iter()
        .filter(|endpoint| {
            let is_global = is_global_namespace(frontend_namespace);
            is_global || endpoint.namespace == frontend_namespace
        })
        .collect();

    assert_eq!(visible_models_global.len(), 4); // Should see all models

    // Scenario 3: Frontend configured for non-existent namespace should see no models
    let frontend_namespace = "non-existent-namespace";
    let visible_models_none: Vec<&EndpointId> = available_models
        .iter()
        .filter(|endpoint| {
            let is_global = is_global_namespace(frontend_namespace);
            is_global || endpoint.namespace == frontend_namespace
        })
        .collect();

    assert_eq!(visible_models_none.len(), 0); // Should see no models
}

#[test]
fn test_namespace_boundary_conditions() {
    // Test edge cases and boundary conditions for namespace handling

    let test_models = vec![
        create_test_endpoint("", "backend", "generate"), // Empty namespace
        create_test_endpoint("dynamo", "backend", "generate"), // Global namespace
        create_test_endpoint("ns-with-special-chars_123", "backend", "generate"),
    ];

    // Test filtering with empty target namespace (should be treated as global)
    let target_namespace = "";
    let is_global = is_global_namespace(target_namespace);
    assert!(is_global); // Empty namespace should be treated as global

    let filtered_empty: Vec<&EndpointId> = test_models
        .iter()
        .filter(|model| is_global || model.namespace == target_namespace)
        .collect();

    assert_eq!(filtered_empty.len(), 3); // All models should be visible

    // Test filtering with exact "dynamo" namespace
    let target_namespace = "dynamo";
    let is_global = is_global_namespace(target_namespace);
    assert!(is_global);

    let filtered_global: Vec<&EndpointId> = test_models
        .iter()
        .filter(|model| is_global || model.namespace == target_namespace)
        .collect();

    assert_eq!(filtered_global.len(), 3); // All models should be visible

    // Test case sensitivity - "GLOBAL" should not be treated as global
    let target_namespace = "DYNAMO";
    let is_global = is_global_namespace(target_namespace);
    assert!(!is_global); // Should be case-sensitive

    let filtered_uppercase: Vec<&EndpointId> = test_models
        .iter()
        .filter(|model| is_global || model.namespace == target_namespace)
        .collect();

    assert_eq!(filtered_uppercase.len(), 0); // No models should be visible
}
