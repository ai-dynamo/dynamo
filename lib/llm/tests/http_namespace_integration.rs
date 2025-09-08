// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for HTTP service namespace discovery functionality.
//! These tests verify that the HTTP service correctly filters models based on namespace configuration.

use dynamo_llm::{
    discovery::ModelEntry,
    model_type::{ModelInput, ModelType},
    namespace::{GLOBAL_NAMESPACE, is_global_namespace},
};
use dynamo_runtime::protocols::EndpointId;

// Helper function to create a test ModelEntry
fn create_test_model_entry(
    name: &str,
    namespace: &str,
    component: &str,
    endpoint_name: &str,
    model_type: ModelType,
    model_input: ModelInput,
) -> ModelEntry {
    ModelEntry {
        name: name.to_string(),
        endpoint_id: EndpointId {
            namespace: namespace.to_string(),
            component: component.to_string(),
            name: endpoint_name.to_string(),
        },
        model_type,
        model_input,
        runtime_config: None,
    }
}

#[test]
fn test_namespace_filtering_behavior() {
    // Test the core namespace filtering logic used in HTTP service
    let test_models = vec![
        create_test_model_entry(
            "model-1",
            "vllm-agg",
            "backend",
            "generate",
            ModelType::Chat,
            ModelInput::Tokens,
        ),
        create_test_model_entry(
            "model-2",
            "sglang-prod",
            "backend",
            "generate",
            ModelType::Chat,
            ModelInput::Tokens,
        ),
        create_test_model_entry(
            "model-3",
            "dynamo",
            "backend",
            "generate",
            ModelType::Completions,
            ModelInput::Tokens,
        ),
        create_test_model_entry(
            "model-4",
            "tensorrt-llm",
            "backend",
            "generate",
            ModelType::Embedding,
            ModelInput::Tokens,
        ),
    ];

    // Test filtering for specific namespace "vllm-agg"
    let target_namespace = "vllm-agg";
    let is_global = is_global_namespace(target_namespace);

    let filtered_models: Vec<&ModelEntry> = test_models
        .iter()
        .filter(|model| is_global || model.endpoint_id.namespace == target_namespace)
        .collect();

    assert_eq!(filtered_models.len(), 1);
    assert_eq!(filtered_models[0].name, "model-1");
    assert_eq!(filtered_models[0].endpoint_id.namespace, "vllm-agg");

    // Test filtering for global namespace (should include all models)
    let target_namespace = GLOBAL_NAMESPACE;
    let is_global = is_global_namespace(target_namespace);

    let filtered_models_global: Vec<&ModelEntry> = test_models
        .iter()
        .filter(|model| is_global || model.endpoint_id.namespace == target_namespace)
        .collect();

    assert_eq!(filtered_models_global.len(), 4); // All models should be included

    // Test filtering for empty namespace (treated as global)
    let target_namespace = "";
    let is_global = is_global_namespace(target_namespace);

    let filtered_models_empty: Vec<&ModelEntry> = test_models
        .iter()
        .filter(|model| is_global || model.endpoint_id.namespace == target_namespace)
        .collect();

    assert_eq!(filtered_models_empty.len(), 4); // All models should be included
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
        create_test_model_entry(
            "llama-7b",
            "vllm-agg",
            "backend",
            "generate",
            ModelType::Chat,
            ModelInput::Tokens,
        ),
        create_test_model_entry(
            "mistral-7b",
            "vllm-agg",
            "backend",
            "generate",
            ModelType::Chat,
            ModelInput::Tokens,
        ),
        create_test_model_entry(
            "gpt-3.5",
            "sglang-prod",
            "backend",
            "generate",
            ModelType::Chat,
            ModelInput::Tokens,
        ),
        create_test_model_entry(
            "claude-3",
            "dynamo",
            "backend",
            "generate",
            ModelType::Chat,
            ModelInput::Tokens,
        ),
    ];

    let visible_models: Vec<&ModelEntry> = available_models
        .iter()
        .filter(|model| {
            let is_global = is_global_namespace(frontend_namespace);
            is_global || model.endpoint_id.namespace == frontend_namespace
        })
        .collect();

    assert_eq!(visible_models.len(), 2);
    assert!(
        visible_models
            .iter()
            .all(|m| m.endpoint_id.namespace == "vllm-agg")
    );

    // Scenario 2: Frontend configured for global namespace should see all models
    let frontend_namespace = GLOBAL_NAMESPACE;
    let visible_models_global: Vec<&ModelEntry> = available_models
        .iter()
        .filter(|model| {
            let is_global = is_global_namespace(frontend_namespace);
            is_global || model.endpoint_id.namespace == frontend_namespace
        })
        .collect();

    assert_eq!(visible_models_global.len(), 4); // Should see all models

    // Scenario 3: Frontend configured for non-existent namespace should see no models
    let frontend_namespace = "non-existent-namespace";
    let visible_models_none: Vec<&ModelEntry> = available_models
        .iter()
        .filter(|model| {
            let is_global = is_global_namespace(frontend_namespace);
            is_global || model.endpoint_id.namespace == frontend_namespace
        })
        .collect();

    assert_eq!(visible_models_none.len(), 0); // Should see no models
}

#[test]
fn test_namespace_boundary_conditions() {
    // Test edge cases and boundary conditions for namespace handling

    let test_models = vec![
        create_test_model_entry(
            "model-1",
            "",
            "backend",
            "generate",
            ModelType::Chat,
            ModelInput::Tokens,
        ), // Empty namespace
        create_test_model_entry(
            "model-2",
            "dynamo",
            "backend",
            "generate",
            ModelType::Chat,
            ModelInput::Tokens,
        ), // Global namespace
        create_test_model_entry(
            "model-3",
            "ns-with-special-chars_123",
            "backend",
            "generate",
            ModelType::Chat,
            ModelInput::Tokens,
        ),
    ];

    // Test filtering with empty target namespace (should be treated as global)
    let target_namespace = "";
    let is_global = is_global_namespace(target_namespace);
    assert!(is_global); // Empty namespace should be treated as global

    let filtered_empty: Vec<&ModelEntry> = test_models
        .iter()
        .filter(|model| is_global || model.endpoint_id.namespace == target_namespace)
        .collect();

    assert_eq!(filtered_empty.len(), 3); // All models should be visible

    // Test filtering with exact "dynamo" namespace
    let target_namespace = "dynamo";
    let is_global = is_global_namespace(target_namespace);
    assert!(is_global);

    let filtered_global: Vec<&ModelEntry> = test_models
        .iter()
        .filter(|model| is_global || model.endpoint_id.namespace == target_namespace)
        .collect();

    assert_eq!(filtered_global.len(), 3); // All models should be visible

    // Test case sensitivity - "GLOBAL" should not be treated as global
    let target_namespace = "DYNAMO";
    let is_global = is_global_namespace(target_namespace);
    assert!(!is_global); // Should be case-sensitive

    let filtered_uppercase: Vec<&ModelEntry> = test_models
        .iter()
        .filter(|model| is_global || model.endpoint_id.namespace == target_namespace)
        .collect();

    assert_eq!(filtered_uppercase.len(), 0); // No models should be visible
}
