// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Custom Resource Definition for DynamoWorkerMetadata
//!
//! This module defines the Rust types for the DynamoWorkerMetadata CRD,
//! which stores discovery metadata for Dynamo worker pods in Kubernetes.
//!
//! The CRD schema is defined at:
//! `deploy/operator/config/crd/bases/nvidia.com_dynamoworkermetadatas.yaml`

use anyhow::Result;
use async_trait::async_trait;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::OwnerReference;
use kube::{
    Api, Client as KubeClient, CustomResource,
    api::{Patch, PatchParams},
};
use serde::{Deserialize, Serialize};

use crate::discovery::{DiscoveryMetadata, EventScope};

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
/// * `cr_name` - Name of the CR (from KubeDiscoveryTarget::cr_name)
/// * `pod_name` - Name of the pod (used in owner reference)
/// * `pod_uid` - UID of the pod (for owner reference - enables garbage collection)
/// * `metadata` - The DiscoveryMetadata to serialize into the CR's data field
///
/// # Returns
/// A `DynamoWorkerMetadata` CR ready to be applied to the cluster
pub fn build_cr(
    cr_name: &str,
    pod_name: &str,
    pod_uid: &str,
    metadata: &DiscoveryMetadata,
) -> Result<DynamoWorkerMetadata> {
    let mut data = serde_json::to_value(metadata)?;
    add_legacy_event_channel_fields(&mut data)?;
    let spec = DynamoWorkerMetadataSpec::new(data);
    let mut cr = DynamoWorkerMetadata::new(cr_name, spec);

    // Set owner reference to the pod for automatic garbage collection
    cr.metadata.owner_references = Some(vec![OwnerReference {
        api_version: "v1".to_string(),
        kind: "Pod".to_string(),
        name: pod_name.to_string(),
        uid: pod_uid.to_string(),
        // Mark pod as the controlling owner - CR will be garbage collected when pod is deleted.
        // In container mode multiple CRs may share one pod; only one can be controller.
        controller: Some(cr_name == pod_name),
        // Don't block pod deletion - allow CR cleanup to happen asynchronously
        block_owner_deletion: Some(false),
    }]);

    Ok(cr)
}

/// Accept the pre-scope event-channel shape at the Kubernetes DWM boundary.
pub(super) fn deserialize_metadata(mut data: serde_json::Value) -> Result<DiscoveryMetadata> {
    add_current_event_channel_scopes(&mut data)?;
    Ok(serde_json::from_value(data)?)
}

fn add_legacy_event_channel_fields(data: &mut serde_json::Value) -> Result<()> {
    let Some(channels) = data
        .get_mut("event_channels")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return Ok(());
    };

    for channel in channels.values_mut() {
        let Some(channel) = channel.as_object_mut() else {
            continue;
        };
        let Some(scope) = channel.get("scope").cloned() else {
            continue;
        };
        let scope = serde_json::from_value::<EventScope>(scope)?;

        channel.insert(
            "namespace".to_string(),
            serde_json::Value::String(scope.namespace().to_string()),
        );
        channel.insert(
            "component".to_string(),
            // v1.2 represented namespace-scoped publishers with an empty component.
            serde_json::Value::String(scope.component().unwrap_or("").to_string()),
        );
    }

    Ok(())
}

fn add_current_event_channel_scopes(data: &mut serde_json::Value) -> Result<()> {
    let Some(channels) = data
        .get_mut("event_channels")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return Ok(());
    };

    for channel in channels.values_mut() {
        let Some(channel) = channel.as_object_mut() else {
            continue;
        };
        if channel.contains_key("scope") {
            continue;
        }
        let (Some(namespace), Some(component)) = (
            channel.get("namespace").and_then(serde_json::Value::as_str),
            channel.get("component").and_then(serde_json::Value::as_str),
        ) else {
            continue;
        };
        let scope = if component.is_empty() {
            EventScope::Namespace {
                name: namespace.to_string(),
            }
        } else {
            EventScope::Component {
                namespace: namespace.to_string(),
                component: component.to_string(),
            }
        };

        channel.insert("scope".to_string(), serde_json::to_value(scope)?);
    }

    Ok(())
}

/// The two CR operations a fenced write needs; a seam so the fencing logic is testable.
#[async_trait]
pub(super) trait CrApi: Send + Sync {
    /// Current CR, or `None` if it does not exist.
    async fn get_cr(&self, name: &str) -> Result<Option<DynamoWorkerMetadata>>;
    /// Server-side apply of `cr`; fails with 409 if `cr.metadata.resource_version` is stale.
    async fn apply(&self, name: &str, cr: &DynamoWorkerMetadata) -> Result<()>;
}

#[async_trait]
impl CrApi for Api<DynamoWorkerMetadata> {
    async fn get_cr(&self, name: &str) -> Result<Option<DynamoWorkerMetadata>> {
        Ok(self.get_opt(name).await?)
    }

    async fn apply(&self, name: &str, cr: &DynamoWorkerMetadata) -> Result<()> {
        // force() resolves field-manager conflicts only; staleness is fenced by the
        // resourceVersion carried in `cr`.
        let params = PatchParams::apply(FIELD_MANAGER).force();
        let applied = self.patch(name, &params, &Patch::Apply(cr)).await?;
        tracing::info!(
            cr_name = name,
            resource_version = ?applied.metadata.resource_version,
            "Applied DynamoWorkerMetadata CR"
        );
        Ok(())
    }
}

/// Annotation set to a fresh value on every write, so each apply changes the object.
const WRITE_ID_ANNOTATION: &str = "nvidia.com/dynamo-discovery-write-id";

/// Apply a DynamoWorkerMetadata CR, fenced by the CR's current resourceVersion.
///
/// The apply carries the resourceVersion just read, so a request still in flight from an older
/// attempt is rejected with 409 instead of landing after a newer write. The API server skips
/// the etcd write (and the resourceVersion bump) for an apply that changes nothing, so every
/// write also stamps a fresh `WRITE_ID_ANNOTATION`. A missing CR is created without a
/// resourceVersion.
pub(super) async fn apply_cr_fenced<A: CrApi>(api: &A, cr: &DynamoWorkerMetadata) -> Result<()> {
    let cr_name = cr
        .metadata
        .name
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("CR must have a name"))?;
    let wrap =
        |e: anyhow::Error| anyhow::anyhow!("Failed to apply DynamoWorkerMetadata {cr_name}: {e}");

    let mut cr = cr.clone();
    cr.metadata.annotations.get_or_insert_default().insert(
        WRITE_ID_ANNOTATION.to_string(),
        uuid::Uuid::new_v4().to_string(),
    );
    if let Some(current) = api.get_cr(cr_name).await.map_err(wrap)? {
        cr.metadata.resource_version = current.metadata.resource_version;
    }
    api.apply(cr_name, &cr).await.map_err(wrap)
}

/// Apply (create or update) a DynamoWorkerMetadata CR using resourceVersion-fenced
/// server-side apply.
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
    apply_cr_fenced(&api, cr).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{
        DiscoveryInstance, DiscoveryQuery, EventChannelQuery, EventScope, EventSourceQuery,
        EventTransport, MAX_JSON_SAFE_PUBLISHER_ID,
    };
    use crate::protocols::EndpointId;
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

    #[test]
    fn event_source_round_trips_through_kubernetes_metadata() {
        let mut metadata = DiscoveryMetadata::new();
        let endpoint = EndpointId {
            namespace: "workers".to_string(),
            component: "backend".to_string(),
            name: "kv-state".to_string(),
        };
        let source = DiscoveryInstance::EventSource {
            scope: EventScope::Endpoint {
                endpoint: endpoint.clone(),
            },
            topic: "kv-events".to_string(),
            publisher_id: MAX_JSON_SAFE_PUBLISHER_ID,
            metadata: serde_json::json!({"worker_id": 7, "dp_rank": 0}),
        };
        metadata.register_event_source(source.clone()).unwrap();

        let cr = build_cr("test-pod", "test-pod", "pod-uid", &metadata).unwrap();
        let publisher_id = cr.spec.data["event_sources"]
            .as_object()
            .and_then(|sources| sources.values().next())
            .and_then(|source| source.get("publisher_id"))
            .expect("serialized event source publisher ID");
        assert_eq!(publisher_id.as_u64(), Some(MAX_JSON_SAFE_PUBLISHER_ID));

        let round_trip = deserialize_metadata(cr.spec.data).unwrap();

        assert_eq!(
            round_trip.filter(&DiscoveryQuery::EventSources(
                EventSourceQuery::endpoint_topic(endpoint, "kv-events")
            )),
            vec![source]
        );
    }

    #[test]
    fn event_channel_metadata_supports_v1_2_wire_shape() {
        #[derive(serde::Deserialize)]
        #[serde(tag = "type")]
        enum LegacyDiscoveryInstance {
            EventChannel {
                namespace: String,
                component: String,
                topic: String,
                instance_id: u64,
                transport: EventTransport,
            },
        }

        let endpoint = EndpointId {
            namespace: "workers".to_string(),
            component: "backend".to_string(),
            name: "generate".to_string(),
        };
        let transport = EventTransport::zmq("tcp://worker:5555");
        let channel = DiscoveryInstance::EventChannel {
            scope: EventScope::Endpoint {
                endpoint: endpoint.clone(),
            },
            topic: "kv-events".to_string(),
            instance_id: 42,
            transport: transport.clone(),
        };
        let mut metadata = DiscoveryMetadata::new();
        metadata.register_event_channel(channel.clone()).unwrap();

        let cr = build_cr("test-pod", "test-pod", "pod-uid", &metadata).unwrap();
        let encoded_channel = cr.spec.data["event_channels"]
            .as_object()
            .and_then(|channels| channels.values().next())
            .cloned()
            .expect("serialized event channel");
        let LegacyDiscoveryInstance::EventChannel {
            namespace,
            component,
            topic,
            instance_id,
            transport: legacy_transport,
        } = serde_json::from_value(encoded_channel).expect("v1.2 event channel shape");
        assert_eq!(
            (namespace, component, topic, instance_id, legacy_transport),
            (
                "workers".to_string(),
                "backend".to_string(),
                "kv-events".to_string(),
                42,
                transport.clone(),
            )
        );
        let current_round_trip = deserialize_metadata(cr.spec.data).unwrap();
        assert_eq!(
            current_round_trip.filter(&DiscoveryQuery::EventChannels(
                EventChannelQuery::endpoint_topic(endpoint, "kv-events")
            )),
            vec![channel]
        );

        let namespace_transport = EventTransport::nats("namespace.workers");
        let legacy_metadata = serde_json::json!({
            "endpoints": {},
            "model_cards": {},
            "event_channels": {
                "workers/backend/kv-events/2a": {
                    "type": "EventChannel",
                    "namespace": "workers",
                    "component": "backend",
                    "topic": "kv-events",
                    "instance_id": 42,
                    "transport": transport,
                },
                "workers//namespace-events/2b": {
                    "type": "EventChannel",
                    "namespace": "workers",
                    "component": "",
                    "topic": "namespace-events",
                    "instance_id": 43,
                    "transport": namespace_transport,
                }
            }
        });
        let upgraded = deserialize_metadata(legacy_metadata).unwrap();
        assert_eq!(
            upgraded.filter(&DiscoveryQuery::EventChannels(EventChannelQuery::topic(
                "workers",
                "backend",
                "kv-events",
            ))),
            vec![DiscoveryInstance::EventChannel {
                scope: EventScope::Component {
                    namespace: "workers".to_string(),
                    component: "backend".to_string(),
                },
                topic: "kv-events".to_string(),
                instance_id: 42,
                transport,
            }]
        );
        assert_eq!(
            upgraded.filter(&DiscoveryQuery::EventChannels(
                EventChannelQuery::namespace_topic("workers", "namespace-events")
            )),
            vec![DiscoveryInstance::EventChannel {
                scope: EventScope::Namespace {
                    name: "workers".to_string(),
                },
                topic: "namespace-events".to_string(),
                instance_id: 43,
                transport: namespace_transport,
            }]
        );
    }
}
