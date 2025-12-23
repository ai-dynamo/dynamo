// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Custom Resource Definition for DynamoWorkerMetadata
//!
//! This module defines the Rust types for the DynamoWorkerMetadata CRD,
//! which stores discovery metadata for Dynamo worker pods in Kubernetes.
//!
//! The CRD schema is defined in the Helm chart at:
//! `deploy/helm/charts/crds/templates/nvidia.com_dynamoworkermetadatas.yaml`

use anyhow::Result;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::OwnerReference;
use kube::{
    Api, Client as KubeClient, CustomResource,
    api::{Patch, PatchParams},
};
use serde::{Deserialize, Serialize};

use crate::discovery::DiscoveryMetadata;

/// Field manager name for server-side apply - identifies this client as the owner of fields it sets
const FIELD_MANAGER: &str = "dynamo-worker";

/// Spec for DynamoWorkerMetadata custom resource
/// The `data` field stores the serialized `DiscoveryMetadata` as a JSON blob.
#[derive(CustomResource, Clone, Debug, Deserialize, Serialize)]
#[kube(
    group = "nvidia.com",
    version = "v1alpha1",
    kind = "DynamoWorkerMetadata",
    namespaced,
    schema = "disabled"
)]
pub struct DynamoWorkerMetadataSpec {
    /// Raw JSON blob containing the DiscoveryMetadata
    pub data: serde_json::Value,
}

impl DynamoWorkerMetadataSpec {
    pub fn new(data: serde_json::Value) -> Self {
        Self { data }
    }
}

/// Build a DynamoWorkerMetadata CR with owner reference set to the pod
/// # Arguments
/// * `pod_name` - Name of the pod (used as CR name and in owner reference)
/// * `pod_uid` - UID of the pod (for owner reference - enables garbage collection)
/// * `metadata` - The DiscoveryMetadata to serialize into the CR's data field
///
/// # Returns
/// A `DynamoWorkerMetadata` CR ready to be applied to the cluster
pub fn build_cr(
    pod_name: &str,
    pod_uid: &str,
    metadata: &DiscoveryMetadata,
) -> Result<DynamoWorkerMetadata> {
    let data = serde_json::to_value(metadata)?;
    let spec = DynamoWorkerMetadataSpec::new(data);
    let mut cr = DynamoWorkerMetadata::new(pod_name, spec);

    // Set owner reference to the pod for automatic garbage collection
    cr.metadata.owner_references = Some(vec![OwnerReference {
        api_version: "v1".to_string(),
        kind: "Pod".to_string(),
        name: pod_name.to_string(),
        uid: pod_uid.to_string(),
        // Mark pod as the controlling owner - CR will be garbage collected when pod is deleted
        controller: Some(true),
        // Don't block pod deletion - allow CR cleanup to happen asynchronously
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

    // force() allows us to take ownership of this field even if another controller owns it
    // in practice the CR will only have one writer (the pod owner)
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

        let spec = DynamoWorkerMetadataSpec::new(data.clone());

        let cr = DynamoWorkerMetadata::new("test-pod", spec);

        let json = serde_json::to_string(&cr).expect("Failed to serialize CR");

        let deserialized: DynamoWorkerMetadata =
            serde_json::from_str(&json).expect("Failed to deserialize CR");

        assert_eq!(deserialized.spec.data, data);
    }
}
