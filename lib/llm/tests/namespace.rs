// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::{
    discovery::ModelEntry,
    model_type::{ModelInput, ModelType},
    namespace::{GLOBAL_NAMESPACE, is_global_namespace},
};
use dynamo_runtime::protocols::EndpointId;

#[test]
fn test_is_global_namespace_with_global_string() {
    assert!(is_global_namespace(GLOBAL_NAMESPACE));
    assert!(is_global_namespace("dynamo"));
}

#[test]
fn test_is_global_namespace_with_empty_string() {
    assert!(is_global_namespace(""));
}

#[test]
fn test_is_global_namespace_with_specific_namespace() {
    assert!(!is_global_namespace("test-namespace"));
    assert!(!is_global_namespace("my-custom-namespace"));
}

#[test]
fn test_is_global_namespace_with_whitespace() {
    // Whitespace should not be considered global
    assert!(!is_global_namespace(" "));
    assert!(!is_global_namespace("  "));
    assert!(!is_global_namespace("\t"));
    assert!(!is_global_namespace("\n"));
}

#[test]
fn test_is_global_namespace_case_sensitivity() {
    // Should be case sensitive
    assert!(!is_global_namespace("Dynamo"));
    assert!(!is_global_namespace("DYNAMO"));
}

#[test]
fn test_global_namespace_constant() {
    assert_eq!(GLOBAL_NAMESPACE, "dynamo");
}

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
fn test_model_entry_creation_with_different_namespaces() {
    // Test creating ModelEntry with specific namespace
    let model_vllm = create_test_model_entry(
        "test-model-1",
        "vllm-agg",
        "backend",
        "generate",
        ModelType::Chat,
        ModelInput::Tokens,
    );

    assert_eq!(model_vllm.name, "test-model-1");
    assert_eq!(model_vllm.endpoint_id.namespace, "vllm-agg");
    assert_eq!(model_vllm.endpoint_id.component, "backend");
    assert_eq!(model_vllm.endpoint_id.name, "generate");
    assert_eq!(model_vllm.model_type, ModelType::Chat);
    assert_eq!(model_vllm.model_input, ModelInput::Tokens);

    // Test creating ModelEntry with global namespace
    let model_global = create_test_model_entry(
        "test-model-2",
        "dynamo",
        "frontend",
        "http",
        ModelType::Completions,
        ModelInput::Text,
    );

    assert_eq!(model_global.name, "test-model-2");
    assert_eq!(model_global.endpoint_id.namespace, "dynamo");
    assert_eq!(model_global.endpoint_id.component, "frontend");
    assert_eq!(model_global.endpoint_id.name, "http");
    assert_eq!(model_global.model_type, ModelType::Completions);
    assert_eq!(model_global.model_input, ModelInput::Text);
}

#[test]
fn test_namespace_filtering_logic() {
    // Test the core logic that would be used in namespace filtering
    let models = vec![
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
            ModelType::Chat,
            ModelInput::Tokens,
        ),
        create_test_model_entry(
            "model-4",
            "",
            "backend",
            "generate",
            ModelType::Chat,
            ModelInput::Tokens,
        ),
    ];

    // Test filtering for specific namespace "vllm-agg"
    let target_namespace = "vllm-agg";
    let global_namespace = is_global_namespace(target_namespace);
    let filtered_vllm: Vec<&ModelEntry> = models
        .iter()
        .filter(|model| global_namespace || model.endpoint_id.namespace == target_namespace)
        .collect();

    assert_eq!(filtered_vllm.len(), 1);
    assert_eq!(filtered_vllm[0].name, "model-1");
    assert_eq!(filtered_vllm[0].endpoint_id.namespace, "vllm-agg");

    // Test filtering for global namespace (should include all)
    let target_namespace = "dynamo";
    let global_namespace = is_global_namespace(target_namespace);
    let filtered_global: Vec<&ModelEntry> = models
        .iter()
        .filter(|model| global_namespace || model.endpoint_id.namespace == target_namespace)
        .collect();

    assert_eq!(filtered_global.len(), 4); // All models should be included

    // Test filtering for empty namespace (should include all, treated as global)
    let target_namespace = "";
    let global_namespace = is_global_namespace(target_namespace);
    let filtered_empty: Vec<&ModelEntry> = models
        .iter()
        .filter(|model| global_namespace || model.endpoint_id.namespace == target_namespace)
        .collect();

    assert_eq!(filtered_empty.len(), 4); // All models should be included

    // Test filtering for non-existent namespace
    let target_namespace = "non-existent";
    let global_namespace = is_global_namespace(target_namespace);
    let filtered_none: Vec<&ModelEntry> = models
        .iter()
        .filter(|model| global_namespace || model.endpoint_id.namespace == target_namespace)
        .collect();

    assert_eq!(filtered_none.len(), 0); // No models should match
}

#[test]
fn test_model_entry_serialization() {
    // Test that ModelEntry can be serialized and deserialized (important for etcd storage)
    let model = create_test_model_entry(
        "test-model",
        "vllm-agg",
        "backend",
        "generate",
        ModelType::Chat,
        ModelInput::Tokens,
    );

    // Serialize to JSON
    let json = serde_json::to_string(&model).expect("Failed to serialize ModelEntry");
    assert!(json.contains("test-model"));
    assert!(json.contains("vllm-agg"));
    assert!(json.contains("backend"));
    assert!(json.contains("generate"));

    // Deserialize from JSON
    let deserialized: ModelEntry =
        serde_json::from_str(&json).expect("Failed to deserialize ModelEntry");
    assert_eq!(deserialized.name, model.name);
    assert_eq!(
        deserialized.endpoint_id.namespace,
        model.endpoint_id.namespace
    );
    assert_eq!(
        deserialized.endpoint_id.component,
        model.endpoint_id.component
    );
    assert_eq!(deserialized.endpoint_id.name, model.endpoint_id.name);
    assert_eq!(deserialized.model_type, model.model_type);
    assert_eq!(deserialized.model_input, model.model_input);
}

#[test]
fn test_endpoint_namespace_parsing() {
    // Test Endpoint creation from string with namespace
    let endpoint1 = EndpointId::from("vllm-agg.backend.generate");
    assert_eq!(endpoint1.namespace, "vllm-agg");
    assert_eq!(endpoint1.component, "backend");
    assert_eq!(endpoint1.name, "generate");

    let endpoint2 = EndpointId::from("global.frontend.http");
    assert_eq!(endpoint2.namespace, "global");
    assert_eq!(endpoint2.component, "frontend");
    assert_eq!(endpoint2.name, "http");

    // Test with forward slash separator
    let endpoint3 = EndpointId::from("sglang-prod/backend/generate");
    assert_eq!(endpoint3.namespace, "sglang-prod");
    assert_eq!(endpoint3.component, "backend");
    assert_eq!(endpoint3.name, "generate");
}
