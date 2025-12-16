// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Custom Resource Definition for DynamoWorkerMetadata
//!
//! This module defines the Rust types for the DynamoWorkerMetadata CRD,
//! which stores discovery metadata for Dynamo worker pods in Kubernetes.
//!
//! The CRD schema is defined externally in YAML at:
//! `deploy/cloud/operator/config/crd/bases/nvidia.com_dynamoworkermetadatas.yaml`

use anyhow::Result;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::OwnerReference;
use kube::{Api, Client as KubeClient, CustomResource, api::{Patch, PatchParams}};
use serde::{Deserialize, Serialize};

use crate::discovery::DiscoveryMetadata;

/// Field manager name for server-side apply operations
const FIELD_MANAGER: &str = "dynamo-worker";

/// Spec for DynamoWorkerMetadata custom resource
///
/// This struct represents the `.spec` field of the DynamoWorkerMetadata CR.
/// The `data` field stores the serialized `DiscoveryMetadata` as a JSON blob.
#[derive(CustomResource, Clone, Debug, Deserialize, Serialize)]
#[kube(
    group = "nvidia.com",
    version = "v1alpha1",
    kind = "DynamoWorkerMetadata",
    namespaced,
    schema = "disabled"
)]
#[serde(rename_all = "camelCase")]
pub struct DynamoWorkerMetadataSpec {
    /// UID of the pod that owns this metadata
    /// Used for owner reference and correlation with EndpointSlice
    pub pod_uid: String,

    /// Instance ID derived from hashing the pod name
    /// Matches the instance_id used in DiscoveryInstance
    pub instance_id: u64,

    /// Raw JSON blob containing the DiscoveryMetadata
    /// This allows storing arbitrary discovery data without tight coupling
    pub data: serde_json::Value,
}

impl DynamoWorkerMetadataSpec {
    /// Create a new spec with the given pod UID, instance ID, and data
    pub fn new(pod_uid: String, instance_id: u64, data: serde_json::Value) -> Self {
        Self {
            pod_uid,
            instance_id,
            data,
        }
    }
}

/// Build a DynamoWorkerMetadata CR with owner reference set to the pod
///
/// # Arguments
/// * `pod_name` - Name of the pod (used as CR name and in owner reference)
/// * `pod_uid` - UID of the pod (for owner reference - enables garbage collection)
/// * `instance_id` - Instance ID derived from hashing the pod name
/// * `metadata` - The DiscoveryMetadata to serialize into the CR's data field
///
/// # Returns
/// A `DynamoWorkerMetadata` CR ready to be applied to the cluster
pub fn build_cr(
    pod_name: &str,
    pod_uid: &str,
    instance_id: u64,
    metadata: &DiscoveryMetadata,
) -> Result<DynamoWorkerMetadata> {
    // Serialize DiscoveryMetadata to JSON
    let data = serde_json::to_value(metadata)?;

    // Create the spec
    let spec = DynamoWorkerMetadataSpec::new(pod_uid.to_string(), instance_id, data);

    // Create the CR
    let mut cr = DynamoWorkerMetadata::new(pod_name, spec);

    // Set owner reference to the pod for automatic garbage collection
    cr.metadata.owner_references = Some(vec![OwnerReference {
        api_version: "v1".to_string(),
        kind: "Pod".to_string(),
        name: pod_name.to_string(),
        uid: pod_uid.to_string(),
        controller: Some(true),
        block_owner_deletion: Some(false),
    }]);

    Ok(cr)
}

/// Apply (create or update) a DynamoWorkerMetadata CR using server-side apply
///
/// This function uses Kubernetes server-side apply which:
/// - Creates the CR if it doesn't exist
/// - Updates the CR if it does exist
/// - Is idempotent and safe to call multiple times
///
/// # Arguments
/// * `kube_client` - Kubernetes client
/// * `namespace` - Namespace to create/update the CR in
/// * `cr` - The DynamoWorkerMetadata CR to apply
pub async fn apply_cr(
    kube_client: &KubeClient,
    namespace: &str,
    cr: &DynamoWorkerMetadata,
) -> Result<()> {
    let api: Api<DynamoWorkerMetadata> = Api::namespaced(kube_client.clone(), namespace);

    let cr_name = cr
        .metadata
        .name
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("CR must have a name"))?;

    let params = PatchParams::apply(FIELD_MANAGER).force();

    api.patch(cr_name, &params, &Patch::Apply(cr))
        .await
        .map_err(|e| anyhow::anyhow!("Failed to apply DynamoWorkerMetadata CR: {}", e))?;

    tracing::debug!(
        "Applied DynamoWorkerMetadata CR: name={}, namespace={}",
        cr_name,
        namespace
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kube::Resource;

    #[test]
    fn test_crd_metadata() {
        // Verify the CRD metadata is correct
        assert_eq!(DynamoWorkerMetadata::group(&()), "nvidia.com");
        assert_eq!(DynamoWorkerMetadata::version(&()), "v1alpha1");
        assert_eq!(DynamoWorkerMetadata::kind(&()), "DynamoWorkerMetadata");
        assert_eq!(DynamoWorkerMetadata::plural(&()), "dynamoworkermetadatas");
    }

    #[test]
    fn test_spec_creation() {
        let data = serde_json::json!({
            "endpoints": {},
            "model_cards": {}
        });

        let spec = DynamoWorkerMetadataSpec::new(
            "abc-123-def".to_string(),
            0x1234567890abcdef,
            data.clone(),
        );

        assert_eq!(spec.pod_uid, "abc-123-def");
        assert_eq!(spec.instance_id, 0x1234567890abcdef);
        assert_eq!(spec.data, data);
    }

    #[test]
    fn test_cr_creation() {
        let data = serde_json::json!({
            "endpoints": {},
            "model_cards": {}
        });

        let spec = DynamoWorkerMetadataSpec::new(
            "abc-123-def".to_string(),
            0x1234567890abcdef,
            data,
        );

        let cr = DynamoWorkerMetadata::new("my-worker-pod", spec);

        assert_eq!(cr.metadata.name, Some("my-worker-pod".to_string()));
        assert_eq!(cr.spec.pod_uid, "abc-123-def");
        assert_eq!(cr.spec.instance_id, 0x1234567890abcdef);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let data = serde_json::json!({
            "endpoints": {
                "ns/comp/ep": {
                    "type": "Endpoint",
                    "namespace": "ns",
                    "component": "comp",
                    "endpoint": "ep",
                    "instance_id": 12345,
                    "transport": { "Nats": "nats://localhost:4222" }
                }
            },
            "model_cards": {}
        });

        let spec = DynamoWorkerMetadataSpec::new(
            "pod-uid-123".to_string(),
            12345,
            data.clone(),
        );

        let cr = DynamoWorkerMetadata::new("test-pod", spec);

        // Serialize to JSON
        let json = serde_json::to_string(&cr).expect("Failed to serialize CR");

        // Deserialize back
        let deserialized: DynamoWorkerMetadata =
            serde_json::from_str(&json).expect("Failed to deserialize CR");

        assert_eq!(deserialized.spec.pod_uid, "pod-uid-123");
        assert_eq!(deserialized.spec.instance_id, 12345);
        assert_eq!(deserialized.spec.data, data);
    }

    #[test]
    fn test_camel_case_serialization() {
        let spec = DynamoWorkerMetadataSpec::new(
            "pod-uid".to_string(),
            42,
            serde_json::json!({}),
        );

        let json = serde_json::to_string(&spec).expect("Failed to serialize spec");

        // Verify camelCase field names in JSON
        assert!(json.contains("podUid"));
        assert!(json.contains("instanceId"));
        assert!(!json.contains("pod_uid"));
        assert!(!json.contains("instance_id"));
    }
}

