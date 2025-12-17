// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for DynamoWorkerMetadata CRD operations
//!
//! These tests verify create, update, get, and watch operations against a Kubernetes cluster.
//! Run with: `cargo test --lib discovery::kube::crd_tests --features integration`
//!
//! Prerequisites:
//! - A running Kubernetes cluster (minikube, kind, or real cluster)
//! - The DynamoWorkerMetadata CRD must be installed
//! - Valid kubeconfig with permissions to create/update/delete CRs

// #![cfg(all(test, feature = "integration"))]

use super::crd::{DynamoWorkerMetadata, DynamoWorkerMetadataSpec};
use anyhow::Result;
use kube::{
    Api, Client,
    api::{DeleteParams, Patch, PatchParams, PostParams},
};
use serde::{Deserialize, Serialize};

/// Mock metadata structure for testing serialization/deserialization
/// This simulates what DiscoveryMetadata would look like
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct MockDiscoveryMetadata {
    endpoints: std::collections::HashMap<String, MockEndpoint>,
    model_cards: std::collections::HashMap<String, MockModelCard>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct MockEndpoint {
    namespace: String,
    component: String,
    endpoint: String,
    instance_id: u64,
    transport: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct MockModelCard {
    namespace: String,
    component: String,
    endpoint: String,
    instance_id: u64,
    model_name: String,
}

impl MockDiscoveryMetadata {
    fn new() -> Self {
        Self {
            endpoints: std::collections::HashMap::new(),
            model_cards: std::collections::HashMap::new(),
        }
    }

    fn with_endpoint(mut self, key: &str, endpoint: MockEndpoint) -> Self {
        self.endpoints.insert(key.to_string(), endpoint);
        self
    }

    fn with_model_card(mut self, key: &str, card: MockModelCard) -> Self {
        self.model_cards.insert(key.to_string(), card);
        self
    }
}

/// Generate a unique CR name for testing to avoid conflicts
fn test_cr_name(suffix: &str) -> String {
    format!(
        "test-dwm-{}-{}",
        std::process::id(),
        suffix
    )
}

/// Cleanup helper to delete test CRs
async fn cleanup_cr(api: &Api<DynamoWorkerMetadata>, name: &str) {
    let _ = api.delete(name, &DeleteParams::default()).await;
}

/// Test creating a DynamoWorkerMetadata CR
#[tokio::test]
async fn test_create_cr() -> Result<()> {
    let client = Client::try_default().await?;
    let api: Api<DynamoWorkerMetadata> = Api::default_namespaced(client);

    let cr_name = test_cr_name("create");

    // Cleanup any leftover from previous runs
    cleanup_cr(&api, &cr_name).await;

    // Create mock metadata
    let mock_metadata = MockDiscoveryMetadata::new().with_endpoint(
        "test-ns/test-comp/test-ep",
        MockEndpoint {
            namespace: "test-ns".to_string(),
            component: "test-comp".to_string(),
            endpoint: "test-ep".to_string(),
            instance_id: 12345,
            transport: "nats://localhost:4222".to_string(),
        },
    );

    // Serialize to JSON Value
    let data = serde_json::to_value(&mock_metadata)?;

    // Create the spec
    let spec = DynamoWorkerMetadataSpec::new(data);

    // Create the CR
    let cr = DynamoWorkerMetadata::new(&cr_name, spec);

    // Submit to Kubernetes
    let created = api.create(&PostParams::default(), &cr).await?;

    // Verify
    assert_eq!(created.metadata.name, Some(cr_name.clone()));

    // Cleanup
    cleanup_cr(&api, &cr_name).await;

    Ok(())
}

/// Test updating a DynamoWorkerMetadata CR using server-side apply
#[tokio::test]
async fn test_update_cr_server_side_apply() -> Result<()> {
    let client = Client::try_default().await?;
    let api: Api<DynamoWorkerMetadata> = Api::default_namespaced(client);

    let cr_name = test_cr_name("update-ssa");

    // Cleanup any leftover
    cleanup_cr(&api, &cr_name).await;

    // Create initial metadata
    let initial_metadata = MockDiscoveryMetadata::new().with_endpoint(
        "ns/comp/ep1",
        MockEndpoint {
            namespace: "ns".to_string(),
            component: "comp".to_string(),
            endpoint: "ep1".to_string(),
            instance_id: 111,
            transport: "nats://localhost:4222".to_string(),
        },
    );

    let spec = DynamoWorkerMetadataSpec::new(serde_json::to_value(&initial_metadata)?);

    let cr = DynamoWorkerMetadata::new(&cr_name, spec);

    // Create using server-side apply (works for both create and update)
    let params = PatchParams::apply("dynamo-worker-test").force();
    api.patch(&cr_name, &params, &Patch::Apply(&cr)).await?;

    // Now update with additional endpoint
    let updated_metadata = MockDiscoveryMetadata::new()
        .with_endpoint(
            "ns/comp/ep1",
            MockEndpoint {
                namespace: "ns".to_string(),
                component: "comp".to_string(),
                endpoint: "ep1".to_string(),
                instance_id: 111,
                transport: "nats://localhost:4222".to_string(),
            },
        )
        .with_endpoint(
            "ns/comp/ep2",
            MockEndpoint {
                namespace: "ns".to_string(),
                component: "comp".to_string(),
                endpoint: "ep2".to_string(),
                instance_id: 111,
                transport: "nats://localhost:4222".to_string(),
            },
        );

    let updated_spec = DynamoWorkerMetadataSpec::new(serde_json::to_value(&updated_metadata)?);

    let updated_cr = DynamoWorkerMetadata::new(&cr_name, updated_spec);

    // Apply update
    let updated = api
        .patch(&cr_name, &params, &Patch::Apply(&updated_cr))
        .await?;

    // Verify update - deserialize the data back to MockDiscoveryMetadata
    let retrieved_metadata: MockDiscoveryMetadata =
        serde_json::from_value(updated.spec.data)?;

    assert_eq!(retrieved_metadata.endpoints.len(), 2);
    assert!(retrieved_metadata.endpoints.contains_key("ns/comp/ep1"));
    assert!(retrieved_metadata.endpoints.contains_key("ns/comp/ep2"));

    // Cleanup
    cleanup_cr(&api, &cr_name).await;

    Ok(())
}

/// Test getting a CR and deserializing its data
#[tokio::test]
async fn test_get_and_deserialize_cr() -> Result<()> {
    let client = Client::try_default().await?;
    let api: Api<DynamoWorkerMetadata> = Api::default_namespaced(client);

    let cr_name = test_cr_name("get-deser");

    // Cleanup any leftover
    cleanup_cr(&api, &cr_name).await;

    // Create metadata with both endpoints and model cards
    let metadata = MockDiscoveryMetadata::new()
        .with_endpoint(
            "myns/mycomp/generate",
            MockEndpoint {
                namespace: "myns".to_string(),
                component: "mycomp".to_string(),
                endpoint: "generate".to_string(),
                instance_id: 0xdeadbeef,
                transport: "nats://nats.default:4222".to_string(),
            },
        )
        .with_model_card(
            "myns/mycomp/generate",
            MockModelCard {
                namespace: "myns".to_string(),
                component: "mycomp".to_string(),
                endpoint: "generate".to_string(),
                instance_id: 0xdeadbeef,
                model_name: "llama-3-8b".to_string(),
            },
        );

    let spec = DynamoWorkerMetadataSpec::new(serde_json::to_value(&metadata)?);

    let cr = DynamoWorkerMetadata::new(&cr_name, spec);

    // Create the CR
    let params = PatchParams::apply("dynamo-worker-test").force();
    api.patch(&cr_name, &params, &Patch::Apply(&cr)).await?;

    // Get the CR back
    let fetched = api.get(&cr_name).await?;

    // Deserialize the data blob back to MockDiscoveryMetadata
    let deserialized: MockDiscoveryMetadata =
        serde_json::from_value(fetched.spec.data)?;

    // Verify the deserialized content
    assert_eq!(deserialized.endpoints.len(), 1);
    assert_eq!(deserialized.model_cards.len(), 1);

    let endpoint = deserialized
        .endpoints
        .get("myns/mycomp/generate")
        .expect("Endpoint not found");
    assert_eq!(endpoint.namespace, "myns");
    assert_eq!(endpoint.component, "mycomp");
    assert_eq!(endpoint.endpoint, "generate");
    assert_eq!(endpoint.instance_id, 0xdeadbeef);

    let model_card = deserialized
        .model_cards
        .get("myns/mycomp/generate")
        .expect("Model card not found");
    assert_eq!(model_card.model_name, "llama-3-8b");

    // Cleanup
    cleanup_cr(&api, &cr_name).await;

    Ok(())
}

/// Test that get_opt returns None for non-existent CR
#[tokio::test]
async fn test_get_nonexistent_cr() -> Result<()> {
    let client = Client::try_default().await?;
    let api: Api<DynamoWorkerMetadata> = Api::default_namespaced(client);

    let cr_name = test_cr_name("nonexistent");

    // Ensure it doesn't exist
    cleanup_cr(&api, &cr_name).await;

    // get_opt should return None instead of error
    let result = api.get_opt(&cr_name).await?;
    assert!(result.is_none());

    Ok(())
}

/// Test the full create-update-get lifecycle
#[tokio::test]
async fn test_full_lifecycle() -> Result<()> {
    let client = Client::try_default().await?;
    let api: Api<DynamoWorkerMetadata> = Api::default_namespaced(client);

    let cr_name = test_cr_name("lifecycle");
    let instance_id = 0x123456789abcdef0u64;

    // Cleanup
    cleanup_cr(&api, &cr_name).await;

    let params = PatchParams::apply("dynamo-worker-test").force();

    // Step 1: Create with empty metadata
    let empty_metadata = MockDiscoveryMetadata::new();
    let spec = DynamoWorkerMetadataSpec::new(serde_json::to_value(&empty_metadata)?);
    let cr = DynamoWorkerMetadata::new(&cr_name, spec);
    api.patch(&cr_name, &params, &Patch::Apply(&cr)).await?;

    // Verify empty
    let fetched = api.get(&cr_name).await?;
    let meta: MockDiscoveryMetadata = serde_json::from_value(fetched.spec.data)?;
    assert!(meta.endpoints.is_empty());
    assert!(meta.model_cards.is_empty());

    // Step 2: Add first endpoint
    let meta_v2 = MockDiscoveryMetadata::new().with_endpoint(
        "ns/comp/ep1",
        MockEndpoint {
            namespace: "ns".to_string(),
            component: "comp".to_string(),
            endpoint: "ep1".to_string(),
            instance_id,
            transport: "nats://localhost:4222".to_string(),
        },
    );
    let spec_v2 = DynamoWorkerMetadataSpec::new(serde_json::to_value(&meta_v2)?);
    let cr_v2 = DynamoWorkerMetadata::new(&cr_name, spec_v2);
    api.patch(&cr_name, &params, &Patch::Apply(&cr_v2)).await?;

    // Verify one endpoint
    let fetched = api.get(&cr_name).await?;
    let meta: MockDiscoveryMetadata = serde_json::from_value(fetched.spec.data)?;
    assert_eq!(meta.endpoints.len(), 1);

    // Step 3: Add second endpoint and a model card
    let meta_v3 = MockDiscoveryMetadata::new()
        .with_endpoint(
            "ns/comp/ep1",
            MockEndpoint {
                namespace: "ns".to_string(),
                component: "comp".to_string(),
                endpoint: "ep1".to_string(),
                instance_id,
                transport: "nats://localhost:4222".to_string(),
            },
        )
        .with_endpoint(
            "ns/comp/ep2",
            MockEndpoint {
                namespace: "ns".to_string(),
                component: "comp".to_string(),
                endpoint: "ep2".to_string(),
                instance_id,
                transport: "nats://localhost:4222".to_string(),
            },
        )
        .with_model_card(
            "ns/comp/ep1",
            MockModelCard {
                namespace: "ns".to_string(),
                component: "comp".to_string(),
                endpoint: "ep1".to_string(),
                instance_id,
                model_name: "test-model".to_string(),
            },
        );
    let spec_v3 = DynamoWorkerMetadataSpec::new(serde_json::to_value(&meta_v3)?);
    let cr_v3 = DynamoWorkerMetadata::new(&cr_name, spec_v3);
    api.patch(&cr_name, &params, &Patch::Apply(&cr_v3)).await?;

    // Verify final state
    let fetched = api.get(&cr_name).await?;
    let meta: MockDiscoveryMetadata = serde_json::from_value(fetched.spec.data)?;
    assert_eq!(meta.endpoints.len(), 2);
    assert_eq!(meta.model_cards.len(), 1);

    // Cleanup
    cleanup_cr(&api, &cr_name).await;

    Ok(())
}

