// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Custom Resource Definition for DynamoWorkerMetadata
//!
//! This module defines the Rust types for the DynamoWorkerMetadata CRD,
//! which stores discovery metadata for Dynamo worker pods in Kubernetes.
//!
//! The CRD schema is defined at:
//! `deploy/operator/config/crd/bases/nvidia.com_dynamoworkermetadatas.yaml`

use std::time::Duration;

use anyhow::{Context, Result};
use k8s_openapi::apimachinery::pkg::apis::meta::v1::OwnerReference;
use kube::{Api, Client as KubeClient, CustomResource, api::PostParams};
use serde::{Deserialize, Serialize};

use crate::discovery::{DiscoveryMetadata, EventScope, ModelCardInstanceId};

/// Field manager name for discovery snapshot writes.
const FIELD_MANAGER: &str = "dynamo-worker";
const WRITE_ID_ANNOTATION: &str = "nvidia.com/discovery-write-id";
const MODEL_CARD_BASE_KEY_FIELD: &str = "card_base_key";
const MODEL_CARD_UNSET_FIELD: &str = "card_unset";
const APPLY_ATTEMPTS: u32 = 3;
const APPLY_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(10);
const APPLY_RETRY_DELAY: Duration = Duration::from_millis(500);

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
    compact_model_cards(&mut data);
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

/// Accept legacy and compacted wire shapes at the Kubernetes DWM boundary.
pub(super) fn deserialize_metadata(mut data: serde_json::Value) -> Result<DiscoveryMetadata> {
    add_current_event_channel_scopes(&mut data)?;
    expand_model_cards(&mut data)?;
    Ok(serde_json::from_value(data)?)
}

fn is_base_model_entry<'a>(
    entry: &'a serde_json::Value,
    expected_base_id: &ModelCardInstanceId,
) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
    let object = entry.as_object()?;
    if object.get("type").and_then(serde_json::Value::as_str) != Some("Model") {
        return None;
    }
    if object
        .get("model_suffix")
        .is_some_and(|value| !value.is_null())
    {
        return None;
    }
    if object.contains_key(MODEL_CARD_BASE_KEY_FIELD) {
        return None;
    }
    if object.get("namespace").and_then(serde_json::Value::as_str)
        != Some(expected_base_id.namespace.as_str())
        || object.get("component").and_then(serde_json::Value::as_str)
            != Some(expected_base_id.component.as_str())
        || object.get("endpoint").and_then(serde_json::Value::as_str)
            != Some(expected_base_id.endpoint.as_str())
        || object
            .get("instance_id")
            .and_then(serde_json::Value::as_u64)
            != Some(expected_base_id.instance_id)
    {
        return None;
    }
    object
        .get("card_json")
        .and_then(serde_json::Value::as_object)
}

fn expected_base_model_id(
    object: &serde_json::Map<String, serde_json::Value>,
) -> Option<ModelCardInstanceId> {
    if object.get("type").and_then(serde_json::Value::as_str) != Some("Model") {
        return None;
    }
    object
        .get("model_suffix")
        .and_then(serde_json::Value::as_str)?;
    Some(ModelCardInstanceId {
        namespace: object.get("namespace")?.as_str()?.to_string(),
        component: object.get("component")?.as_str()?.to_string(),
        endpoint: object.get("endpoint")?.as_str()?.to_string(),
        instance_id: object.get("instance_id")?.as_u64()?,
        model_suffix: None,
    })
}

fn compacted_entry_is_smaller(
    derived_card: &serde_json::Map<String, serde_json::Value>,
    base_key: &str,
    overlay: &serde_json::Map<String, serde_json::Value>,
    unset: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    let Ok(original_len) = serde_json::to_vec(derived_card).map(|bytes| bytes.len()) else {
        return false;
    };
    let Ok(overlay_len) = serde_json::to_vec(overlay).map(|bytes| bytes.len()) else {
        return false;
    };
    let Ok(base_key_len) = serde_json::to_vec(base_key).map(|bytes| bytes.len()) else {
        return false;
    };
    // Account for `,"card_base_key":<base_key>` plus optional `,"card_unset":<unset>`.
    let mut compacted_len = overlay_len + MODEL_CARD_BASE_KEY_FIELD.len() + 4 + base_key_len;
    if !unset.is_empty() {
        let Ok(unset_len) = serde_json::to_vec(unset).map(|bytes| bytes.len()) else {
            return false;
        };
        compacted_len += MODEL_CARD_UNSET_FIELD.len() + 4 + unset_len;
    }
    compacted_len < original_len
}

struct CompactedModelCard {
    entry_key: String,
    base_key: String,
    overlay: serde_json::Map<String, serde_json::Value>,
    unset: serde_json::Map<String, serde_json::Value>,
}

/// Compact suffixed model cards (such as runtime LoRA adapters) against their matching base card
/// so the base `ModelDeploymentCard` payload is stored once per worker endpoint in etcd.
fn compact_model_cards(data: &mut serde_json::Value) {
    let Some(model_cards) = data
        .get_mut("model_cards")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return;
    };

    let compacted: Vec<CompactedModelCard> = model_cards
        .iter()
        .filter_map(|(entry_key, entry)| {
            let object = entry.as_object()?;
            let expected_base_id = expected_base_model_id(object)?;
            let base_key = expected_base_id.to_path();
            let derived_card = object
                .get("card_json")
                .and_then(serde_json::Value::as_object)?;
            if base_key == *entry_key {
                return None;
            }
            let base_card = model_cards
                .get(&base_key)
                .and_then(|base_entry| is_base_model_entry(base_entry, &expected_base_id))?;
            let (overlay, unset) = diff_json_objects(base_card, derived_card);
            if !compacted_entry_is_smaller(derived_card, &base_key, &overlay, &unset) {
                return None;
            }
            Some(CompactedModelCard {
                entry_key: entry_key.clone(),
                base_key,
                overlay,
                unset,
            })
        })
        .collect();

    for CompactedModelCard {
        entry_key,
        base_key,
        overlay,
        unset,
    } in compacted
    {
        if let Some(object) = model_cards
            .get_mut(&entry_key)
            .and_then(serde_json::Value::as_object_mut)
        {
            object.insert(
                MODEL_CARD_BASE_KEY_FIELD.to_string(),
                serde_json::Value::String(base_key),
            );
            object.insert("card_json".to_string(), serde_json::Value::Object(overlay));
            if unset.is_empty() {
                object.remove(MODEL_CARD_UNSET_FIELD);
            } else {
                object.insert(
                    MODEL_CARD_UNSET_FIELD.to_string(),
                    serde_json::Value::Object(unset),
                );
            }
        }
    }
}

/// Reconstruct compacted suffixed model cards from their referenced base card.
fn expand_model_cards(data: &mut serde_json::Value) -> Result<()> {
    let Some(model_cards) = data
        .get_mut("model_cards")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return Ok(());
    };

    let mut expanded = Vec::new();
    for (entry_key, entry) in model_cards.iter() {
        let Some(object) = entry.as_object() else {
            continue;
        };
        let base_key = match object.get(MODEL_CARD_BASE_KEY_FIELD) {
            None => {
                if object
                    .get(MODEL_CARD_UNSET_FIELD)
                    .is_some_and(|value| !value.is_null())
                {
                    anyhow::bail!(
                        "model card '{entry_key}' specifies {MODEL_CARD_UNSET_FIELD} without {MODEL_CARD_BASE_KEY_FIELD}"
                    );
                }
                continue;
            }
            Some(serde_json::Value::String(base_key)) if !base_key.is_empty() => base_key,
            Some(other) => {
                anyhow::bail!(
                    "model card '{entry_key}' has invalid {MODEL_CARD_BASE_KEY_FIELD}: {other}"
                );
            }
        };

        let Some(expected_base_id) = expected_base_model_id(object) else {
            anyhow::bail!(
                "entry '{entry_key}' cannot specify {MODEL_CARD_BASE_KEY_FIELD} without a suffixed Model identity"
            );
        };
        let expected_base_key = expected_base_id.to_path();
        if base_key != &expected_base_key {
            anyhow::bail!(
                "model card '{entry_key}' references unexpected base card '{base_key}' (expected '{expected_base_key}')"
            );
        }
        if base_key == entry_key {
            anyhow::bail!("model card '{entry_key}' cannot reference itself as base");
        }
        let base_entry = model_cards.get(base_key).ok_or_else(|| {
            anyhow::anyhow!("model card '{entry_key}' references missing base card '{base_key}'")
        })?;
        let base_card = is_base_model_entry(base_entry, &expected_base_id).ok_or_else(|| {
            anyhow::anyhow!("model card '{entry_key}' references invalid base card '{base_key}'")
        })?;

        let mut reconstructed = base_card.clone();
        match object.get("card_json") {
            Some(serde_json::Value::Object(overlay)) => {
                apply_json_overlay(&mut reconstructed, overlay);
            }
            // Kubernetes SSA can rewrite empty objects `{}` to `null`.
            Some(serde_json::Value::Null) | None => {}
            Some(other) => {
                anyhow::bail!(
                    "compacted model card '{entry_key}' has non-object card_json overlay: {other}"
                );
            }
        }

        match object.get(MODEL_CARD_UNSET_FIELD) {
            Some(serde_json::Value::Object(unset)) => {
                apply_json_unset(&mut reconstructed, unset).with_context(|| {
                    format!("failed to apply {MODEL_CARD_UNSET_FIELD} for model card '{entry_key}'")
                })?;
            }
            Some(serde_json::Value::Null) | None => {}
            Some(other) => {
                anyhow::bail!(
                    "compacted model card '{entry_key}' has non-object {MODEL_CARD_UNSET_FIELD}: {other}"
                );
            }
        }

        expanded.push((entry_key.clone(), serde_json::Value::Object(reconstructed)));
    }

    for (entry_key, card_json) in expanded {
        if let Some(object) = model_cards
            .get_mut(&entry_key)
            .and_then(serde_json::Value::as_object_mut)
        {
            object.insert("card_json".to_string(), card_json);
            object.remove(MODEL_CARD_BASE_KEY_FIELD);
            object.remove(MODEL_CARD_UNSET_FIELD);
        }
    }

    Ok(())
}

/// Compute a lossless recursive object diff `(overlay, unset)` such that applying `overlay` and
/// then `unset` onto `base` reproduces `derived` identically, including explicit `null` values
/// and nested field removals.
fn diff_json_objects(
    base: &serde_json::Map<String, serde_json::Value>,
    derived: &serde_json::Map<String, serde_json::Value>,
) -> (
    serde_json::Map<String, serde_json::Value>,
    serde_json::Map<String, serde_json::Value>,
) {
    let mut overlay = serde_json::Map::new();
    let mut unset = serde_json::Map::new();

    for (key, derived_value) in derived {
        match (base.get(key), derived_value) {
            (Some(base_value), _) if base_value == derived_value => {}
            (Some(serde_json::Value::Object(base_obj)), serde_json::Value::Object(derived_obj)) => {
                let (sub_overlay, sub_unset) = diff_json_objects(base_obj, derived_obj);
                if !sub_overlay.is_empty() {
                    overlay.insert(key.clone(), serde_json::Value::Object(sub_overlay));
                }
                if !sub_unset.is_empty() {
                    unset.insert(key.clone(), serde_json::Value::Object(sub_unset));
                }
            }
            _ => {
                overlay.insert(key.clone(), derived_value.clone());
            }
        }
    }

    for key in base.keys() {
        if !derived.contains_key(key) {
            unset.insert(key.clone(), serde_json::Value::Bool(true));
        }
    }

    (overlay, unset)
}

fn apply_json_overlay(
    target: &mut serde_json::Map<String, serde_json::Value>,
    overlay: &serde_json::Map<String, serde_json::Value>,
) {
    for (key, value) in overlay {
        match (target.get_mut(key), value) {
            (
                Some(serde_json::Value::Object(target_obj)),
                serde_json::Value::Object(overlay_obj),
            ) => {
                apply_json_overlay(target_obj, overlay_obj);
            }
            _ => {
                target.insert(key.clone(), value.clone());
            }
        }
    }
}

fn apply_json_unset(
    target: &mut serde_json::Map<String, serde_json::Value>,
    unset: &serde_json::Map<String, serde_json::Value>,
) -> Result<()> {
    for (key, marker) in unset {
        match marker {
            serde_json::Value::Bool(true) => {
                target.remove(key);
            }
            serde_json::Value::Object(nested_unset) => {
                let Some(serde_json::Value::Object(nested_target)) = target.get_mut(key) else {
                    anyhow::bail!(
                        "cannot apply nested unset for key '{key}': target is missing or not an object"
                    );
                };
                apply_json_unset(nested_target, nested_unset)?;
            }
            // Defensive against Kubernetes SSA rewriting empty objects `{}` to `null`.
            serde_json::Value::Null => {}
            other => {
                anyhow::bail!(
                    "invalid unset marker for key '{key}': expected true or object, got {other}"
                );
            }
        }
    }
    Ok(())
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

/// Persist a discovery snapshot with bounded retries.
///
/// Fresh resourceVersion preconditions fence requests that outlive a timeout.
/// Each logical write changes an annotation so even a no-op snapshot advances
/// the version and fences earlier writes. An error can still mean the server
/// committed the snapshot without acknowledging it.
pub async fn apply_cr(
    kube_client: &KubeClient,
    namespace: &str,
    cr: &DynamoWorkerMetadata,
) -> Result<()> {
    let api: Api<DynamoWorkerMetadata> = Api::namespaced(kube_client.clone(), namespace);
    let cr_name = cr
        .metadata
        .name
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("CR must have a name"))?;
    let write_id = uuid::Uuid::new_v4().to_string();
    let mut candidate = cr.clone();
    candidate
        .metadata
        .annotations
        .get_or_insert_default()
        .insert(WRITE_ID_ANNOTATION.into(), write_id.clone());

    let mut attempt = 0;
    loop {
        attempt += 1;
        let result = tokio::time::timeout(APPLY_ATTEMPT_TIMEOUT, async {
            let params = PostParams {
                field_manager: Some(FIELD_MANAGER.into()),
                ..Default::default()
            };
            if let Some(mut current) = api.get_opt(cr_name).await? {
                current.spec = candidate.spec.clone();
                current.metadata.owner_references = candidate.metadata.owner_references.clone();
                current
                    .metadata
                    .annotations
                    .get_or_insert_default()
                    .insert(WRITE_ID_ANNOTATION.into(), write_id.clone());
                api.replace(cr_name, &params, &current).await
            } else {
                // A delayed create cannot overwrite a newer, already-created snapshot.
                api.create(&params, &candidate).await
            }
        })
        .await;

        let error = match result {
            Ok(Ok(applied)) => {
                tracing::info!(
                    namespace,
                    cr_name,
                    resource_version = ?applied.metadata.resource_version,
                    "Applied DynamoWorkerMetadata CR"
                );
                return Ok(());
            }
            Ok(Err(error)) => {
                let retryable = match &error {
                    kube::Error::Api(response) => {
                        matches!(response.code, 408 | 409 | 429 | 500..=599)
                    }
                    kube::Error::HyperError(_) | kube::Error::Service(_) => true,
                    _ => false,
                };
                if !retryable {
                    return Err(error).with_context(|| {
                        format!("Failed to apply DynamoWorkerMetadata {namespace}/{cr_name}")
                    });
                }
                anyhow::Error::from(error)
            }
            Err(error) => {
                anyhow::Error::from(error).context("discovery persistence attempt timed out")
            }
        };
        if attempt == APPLY_ATTEMPTS {
            return Err(error).with_context(|| {
                format!(
                    "Failed to apply DynamoWorkerMetadata {namespace}/{cr_name} after {attempt} attempts"
                )
            });
        }
        tracing::warn!(namespace, cr_name, attempt, error = %error, "Retrying discovery persistence");
        tokio::time::sleep(APPLY_RETRY_DELAY).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{
        DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery, EventChannelQuery, EventScope,
        EventSourceQuery, EventTransport, MAX_JSON_SAFE_PUBLISHER_ID,
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

    fn sample_base_card_json(padding_bytes: usize) -> serde_json::Value {
        let extra_blob = "x".repeat(padding_bytes);
        serde_json::json!({
            "display_name": "Qwen/Qwen2.5-7B-Instruct",
            "slug": "qwen-qwen2.5-7b-instruct",
            "aliases": ["qwen-default", "qwen-7b"],
            "model_info": {
                "HFConfigJson": {
                    "path": "/models/qwen/config.json",
                    "checksum": "abc123",
                    "size": 1024
                }
            },
            "tokenizer": {
                "HfTokenizerJson": {
                    "path": "/models/qwen/tokenizer.json",
                    "checksum": "def456",
                    "size": 2048
                }
            },
            "prompt_formatter": {
                "HfTokenizerConfigJson": {
                    "path": "/models/qwen/tokenizer_config.json",
                    "checksum": "ghi789",
                    "size": 512
                }
            },
            "kv_cache_block_size": 16,
            "migration_limit": 0,
            "model_type": "Chat",
            "model_input": "Tokens",
            "worker_type": "aggregated",
            "needs": [],
            "media_decoder": null,
            "media_fetcher": { "allow_direct_ip": false, "nested_keep": 1 },
            "extra_nested": { "only_in_base": "drop-me" },
            "runtime_config": {
                "total_kv_blocks": 8192,
                "max_num_seqs": 256,
                "max_gpu_lora_count": 64,
                "taints": ["gpu-pool-a"],
                "runtime_data": {
                    "capabilities": ["chat", "tools"],
                    "payload": extra_blob
                }
            }
        })
    }

    fn model_instance(
        namespace: &str,
        component: &str,
        endpoint: &str,
        instance_id: u64,
        model_suffix: Option<&str>,
        card_json: serde_json::Value,
    ) -> DiscoveryInstance {
        DiscoveryInstance::Model {
            namespace: namespace.to_string(),
            component: component.to_string(),
            endpoint: endpoint.to_string(),
            instance_id,
            card_json,
            model_suffix: model_suffix.map(str::to_string),
        }
    }

    fn model_cards_by_id(
        metadata: &DiscoveryMetadata,
    ) -> std::collections::HashMap<crate::discovery::DiscoveryInstanceId, DiscoveryInstance> {
        metadata
            .get_all_model_cards()
            .into_iter()
            .map(|instance| (instance.id(), instance))
            .collect()
    }

    #[test]
    fn lora_model_cards_compact_against_base_card_and_round_trip() {
        let mut metadata = DiscoveryMetadata::new();
        let base_card = sample_base_card_json(256);
        let base = model_instance("ns", "worker", "generate", 42, None, base_card.clone());
        let base_key = ModelCardInstanceId {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            endpoint: "generate".to_string(),
            instance_id: 42,
            model_suffix: None,
        }
        .to_path();

        // Sidecar-style LoRA: removes aliases, adds source_path/lora/user_data, keeps explicit null.
        let mut sidecar_lora_card = base_card.clone();
        let obj = sidecar_lora_card.as_object_mut().unwrap();
        obj.insert(
            "display_name".to_string(),
            serde_json::json!("math-adapter-r8"),
        );
        obj.insert("slug".to_string(), serde_json::json!("math-adapter-r8"));
        obj.insert(
            "source_path".to_string(),
            serde_json::json!("Qwen/Qwen2.5-7B-Instruct"),
        );
        obj.remove("aliases");
        obj.insert(
            "lora".to_string(),
            serde_json::json!({"name": "math-adapter-r8", "max_gpu_lora_count": 64}),
        );
        obj.insert(
            "user_data".to_string(),
            serde_json::json!({"lora_adapter": true, "lora_id": 7}),
        );
        let sidecar_lora = model_instance(
            "ns",
            "worker",
            "generate",
            42,
            Some("math-adapter-r8"),
            sidecar_lora_card,
        );

        // Python-style LoRA + edge cases: nested removal (`runtime_config.max_gpu_lora_count`),
        // object -> null, null -> object, and non-empty object -> empty object.
        let mut py_lora_card = base_card;
        let obj = py_lora_card.as_object_mut().unwrap();
        obj.insert("display_name".to_string(), serde_json::json!("code-lora"));
        obj.insert("slug".to_string(), serde_json::json!("code-lora"));
        obj.remove("aliases");
        obj.insert(
            "lora".to_string(),
            serde_json::json!({"name": "code-lora", "max_gpu_lora_count": 64}),
        );
        obj.insert("media_decoder".to_string(), serde_json::json!({"fps": 2}));
        obj.insert("media_fetcher".to_string(), serde_json::Value::Null);
        obj.insert("extra_nested".to_string(), serde_json::json!({}));
        obj["runtime_config"]
            .as_object_mut()
            .unwrap()
            .remove("max_gpu_lora_count");
        let py_lora = model_instance(
            "ns",
            "worker",
            "generate",
            42,
            Some("code-lora"),
            py_lora_card,
        );

        metadata.register_model_card(base).unwrap();
        metadata.register_model_card(sidecar_lora).unwrap();
        metadata.register_model_card(py_lora).unwrap();

        let cr = build_cr("worker-0", "worker-0", "pod-uid", &metadata).unwrap();
        let wire_cards = cr.spec.data["model_cards"].as_object().unwrap();

        // Base card stays full and untouched on the wire.
        let wire_base = wire_cards[&base_key].as_object().unwrap();
        assert!(!wire_base.contains_key(MODEL_CARD_BASE_KEY_FIELD));
        assert!(!wire_base.contains_key(MODEL_CARD_UNSET_FIELD));
        assert!(wire_base["card_json"].get("runtime_config").is_some());

        // Sidecar LoRA stores only changed fields and top-level unset.
        let sidecar_key = format!("{base_key}/math-adapter-r8");
        let wire_sidecar = wire_cards[&sidecar_key].as_object().unwrap();
        assert_eq!(
            wire_sidecar[MODEL_CARD_BASE_KEY_FIELD],
            serde_json::json!(base_key)
        );
        let sidecar_overlay = wire_sidecar["card_json"].as_object().unwrap();
        assert!(!sidecar_overlay.contains_key("model_info"));
        assert!(!sidecar_overlay.contains_key("tokenizer"));
        assert!(!sidecar_overlay.contains_key("runtime_config"));
        assert_eq!(sidecar_overlay["display_name"], "math-adapter-r8");
        assert_eq!(
            wire_sidecar[MODEL_CARD_UNSET_FIELD],
            serde_json::json!({"aliases": true})
        );

        // Python LoRA stores nested unset without duplicating runtime_config.runtime_data.
        let py_key = format!("{base_key}/code-lora");
        let wire_py = wire_cards[&py_key].as_object().unwrap();
        assert_eq!(
            wire_py[MODEL_CARD_BASE_KEY_FIELD],
            serde_json::json!(base_key)
        );
        assert!(wire_py["card_json"].get("runtime_config").is_none());
        assert_eq!(
            wire_py[MODEL_CARD_UNSET_FIELD],
            serde_json::json!({
                "aliases": true,
                "extra_nested": { "only_in_base": true },
                "runtime_config": { "max_gpu_lora_count": true }
            })
        );

        // Full CR JSON round-trip reproduces every card bit-for-bit.
        let wire_json = serde_json::to_string(&cr).unwrap();
        let parsed_cr: DynamoWorkerMetadata = serde_json::from_str(&wire_json).unwrap();
        // A pre-compaction reader deserializing DiscoveryMetadata directly still succeeds and
        // retains the complete untouched base model card.
        let legacy_parsed: DiscoveryMetadata =
            serde_json::from_value(parsed_cr.spec.data.clone()).unwrap();
        let legacy_cards = model_cards_by_id(&legacy_parsed);
        assert_eq!(
            legacy_cards[&DiscoveryInstanceId::Model(ModelCardInstanceId {
                namespace: "ns".to_string(),
                component: "worker".to_string(),
                endpoint: "generate".to_string(),
                instance_id: 42,
                model_suffix: None,
            })],
            model_cards_by_id(&metadata)[&DiscoveryInstanceId::Model(ModelCardInstanceId {
                namespace: "ns".to_string(),
                component: "worker".to_string(),
                endpoint: "generate".to_string(),
                instance_id: 42,
                model_suffix: None,
            })]
        );

        let restored = deserialize_metadata(parsed_cr.spec.data).unwrap();
        assert_eq!(model_cards_by_id(&restored), model_cards_by_id(&metadata));
    }

    #[test]
    fn compaction_is_skipped_when_delta_would_not_shrink_card() {
        let mut metadata = DiscoveryMetadata::new();
        let base = model_instance(
            "ns",
            "worker",
            "generate",
            7,
            None,
            serde_json::json!({ "display_name": "base", "a": 1 }),
        );
        // Disjoint tiny derived card: storing card_base_key + card_unset would be larger than raw.
        let disjoint = model_instance(
            "ns",
            "worker",
            "generate",
            7,
            Some("disjoint"),
            serde_json::json!({ "display_name": "disjoint", "b": 2 }),
        );
        metadata.register_model_card(base).unwrap();
        metadata.register_model_card(disjoint).unwrap();

        let cr = build_cr("worker-0", "worker-0", "pod-uid", &metadata).unwrap();
        let wire_disjoint = cr.spec.data["model_cards"]["ns/worker/generate/7/disjoint"]
            .as_object()
            .unwrap();
        assert!(!wire_disjoint.contains_key(MODEL_CARD_BASE_KEY_FIELD));
        assert!(!wire_disjoint.contains_key(MODEL_CARD_UNSET_FIELD));

        let restored = deserialize_metadata(cr.spec.data).unwrap();
        assert_eq!(model_cards_by_id(&restored), model_cards_by_id(&metadata));
    }

    #[test]
    fn unregistering_base_card_falls_back_to_uncompacted_lora_without_dangling_ref() {
        let mut metadata = DiscoveryMetadata::new();
        let base_card = sample_base_card_json(256);
        let base = model_instance("ns", "worker", "generate", 5, None, base_card.clone());
        let mut lora_card = base_card;
        lora_card["display_name"] = serde_json::json!("surviving-lora");
        lora_card["lora"] = serde_json::json!({"name": "surviving-lora"});
        let lora = model_instance(
            "ns",
            "worker",
            "generate",
            5,
            Some("surviving-lora"),
            lora_card,
        );

        metadata.register_model_card(base.clone()).unwrap();
        metadata.register_model_card(lora).unwrap();

        let compacted_cr = build_cr("worker-0", "worker-0", "pod-uid", &metadata).unwrap();
        assert!(
            compacted_cr.spec.data["model_cards"]["ns/worker/generate/5/surviving-lora"]
                .get(MODEL_CARD_BASE_KEY_FIELD)
                .is_some()
        );

        metadata.unregister_model_card(&base).unwrap();
        let after_base_removed = build_cr("worker-0", "worker-0", "pod-uid", &metadata).unwrap();
        let wire_lora =
            after_base_removed.spec.data["model_cards"]["ns/worker/generate/5/surviving-lora"]
                .as_object()
                .unwrap();
        assert!(!wire_lora.contains_key(MODEL_CARD_BASE_KEY_FIELD));
        assert_eq!(
            model_cards_by_id(&deserialize_metadata(after_base_removed.spec.data).unwrap()),
            model_cards_by_id(&metadata)
        );
    }

    #[test]
    fn many_lora_model_cards_stay_well_under_etcd_limit() {
        const ETCD_DEFAULT_MAX_REQUEST_BYTES: usize = 1_572_864; // 1.5 MiB
        const LORA_COUNT: usize = 250;

        let mut metadata = DiscoveryMetadata::new();
        let base_card = sample_base_card_json(8 * 1024);
        metadata
            .register_model_card(model_instance(
                "ns",
                "worker",
                "generate",
                99,
                None,
                base_card.clone(),
            ))
            .unwrap();

        for i in 0..LORA_COUNT {
            let name = format!("tenant-adapter-{i:04}");
            let mut lora_card = base_card.clone();
            let obj = lora_card.as_object_mut().unwrap();
            obj.insert("display_name".to_string(), serde_json::json!(name));
            obj.insert("slug".to_string(), serde_json::json!(name));
            obj.insert(
                "source_path".to_string(),
                serde_json::json!("Qwen/Qwen2.5-7B-Instruct"),
            );
            obj.remove("aliases");
            obj.insert(
                "lora".to_string(),
                serde_json::json!({"name": name, "max_gpu_lora_count": 64}),
            );
            obj.insert(
                "user_data".to_string(),
                serde_json::json!({"lora_adapter": true, "lora_id": i + 1}),
            );
            metadata
                .register_model_card(model_instance(
                    "ns",
                    "worker",
                    "generate",
                    99,
                    Some(&name),
                    lora_card,
                ))
                .unwrap();
        }

        let uncompacted_bytes = serde_json::to_vec(&metadata).unwrap().len();
        assert!(
            uncompacted_bytes > ETCD_DEFAULT_MAX_REQUEST_BYTES,
            "expected uncompacted payload ({uncompacted_bytes} B) to exceed 1.5 MiB"
        );

        let cr = build_cr("worker-0", "worker-0", "pod-uid", &metadata).unwrap();
        let compacted_bytes = serde_json::to_vec(&cr).unwrap().len();
        assert!(
            compacted_bytes < 256 * 1024,
            "expected compacted CR ({compacted_bytes} B) to stay well below 256 KiB (uncompacted was {uncompacted_bytes} B)"
        );

        let restored = deserialize_metadata(cr.spec.data).unwrap();
        assert_eq!(restored.get_all_model_cards().len(), LORA_COUNT + 1);
        assert_eq!(model_cards_by_id(&restored), model_cards_by_id(&metadata));
    }

    #[test]
    fn base_model_taint_update_preserves_lora_cards_with_nested_diff() {
        let mut metadata = DiscoveryMetadata::new();
        let base_card = sample_base_card_json(512);
        let base = model_instance("ns", "worker", "generate", 7, None, base_card.clone());
        let DiscoveryInstanceId::Model(base_id) = base.id() else {
            unreachable!()
        };

        let mut lora_card = base_card;
        lora_card["display_name"] = serde_json::json!("adapter-a");
        lora_card["slug"] = serde_json::json!("adapter-a");
        lora_card["lora"] = serde_json::json!({"name": "adapter-a"});
        let lora = model_instance("ns", "worker", "generate", 7, Some("adapter-a"), lora_card);

        metadata.register_model_card(base).unwrap();
        metadata.register_model_card(lora).unwrap();
        metadata
            .update_model_taints(
                &base_id,
                std::collections::HashSet::from(["draining".into()]),
            )
            .unwrap();

        let cr = build_cr("worker-0", "worker-0", "pod-uid", &metadata).unwrap();
        let lora_key = format!("{}/adapter-a", base_id.to_path());
        let wire_lora_overlay = &cr.spec.data["model_cards"][&lora_key]["card_json"];
        // Only `runtime_config.taints` differs; `runtime_config.runtime_data` is not duplicated.
        assert_eq!(
            wire_lora_overlay["runtime_config"],
            serde_json::json!({"taints": ["gpu-pool-a"]})
        );

        let restored = deserialize_metadata(cr.spec.data).unwrap();
        assert_eq!(model_cards_by_id(&restored), model_cards_by_id(&metadata));
    }

    #[test]
    fn legacy_uncompacted_orphan_and_multi_endpoint_cards_round_trip() {
        // 1. Legacy uncompacted CR (no card_base_key) still deserializes identically.
        let base_card = sample_base_card_json(64);
        let mut lora_card = base_card.clone();
        lora_card["display_name"] = serde_json::json!("legacy-lora");
        lora_card["lora"] = serde_json::json!({"name": "legacy-lora"});
        let mut legacy_metadata = DiscoveryMetadata::new();
        legacy_metadata
            .register_model_card(model_instance(
                "ns",
                "worker",
                "generate",
                11,
                None,
                base_card.clone(),
            ))
            .unwrap();
        legacy_metadata
            .register_model_card(model_instance(
                "ns",
                "worker",
                "generate",
                11,
                Some("legacy-lora"),
                lora_card.clone(),
            ))
            .unwrap();
        let raw_legacy_json = serde_json::to_value(&legacy_metadata).unwrap();
        let restored_legacy = deserialize_metadata(raw_legacy_json).unwrap();
        assert_eq!(
            model_cards_by_id(&restored_legacy),
            model_cards_by_id(&legacy_metadata)
        );

        // 2. Standalone suffixed card without a base card is left uncompacted (no dangling ref).
        let mut orphan_metadata = DiscoveryMetadata::new();
        orphan_metadata
            .register_model_card(model_instance(
                "ns",
                "worker",
                "generate",
                12,
                Some("orphan-lora"),
                lora_card,
            ))
            .unwrap();
        let orphan_cr = build_cr("worker-0", "worker-0", "pod-uid", &orphan_metadata).unwrap();
        let orphan_wire = orphan_cr.spec.data["model_cards"]
            .as_object()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .as_object()
            .unwrap();
        assert!(!orphan_wire.contains_key(MODEL_CARD_BASE_KEY_FIELD));
        assert_eq!(
            model_cards_by_id(&deserialize_metadata(orphan_cr.spec.data).unwrap()),
            model_cards_by_id(&orphan_metadata)
        );

        // 3. Multiple endpoints compact independently against their own base card.
        let mut multi_ep = DiscoveryMetadata::new();
        let mut base_a = base_card.clone();
        base_a["display_name"] = serde_json::json!("model-a");
        let mut lora_a = base_a.clone();
        lora_a["display_name"] = serde_json::json!("lora-a");
        lora_a["lora"] = serde_json::json!({"name": "lora-a"});

        let mut base_b = base_card.clone();
        base_b["display_name"] = serde_json::json!("model-b");
        base_b["kv_cache_block_size"] = serde_json::json!(32);
        let mut lora_b = base_b.clone();
        lora_b["display_name"] = serde_json::json!("lora-b");
        lora_b["lora"] = serde_json::json!({"name": "lora-b"});

        multi_ep
            .register_model_card(model_instance("ns", "worker", "ep_a", 1, None, base_a))
            .unwrap();
        multi_ep
            .register_model_card(model_instance(
                "ns",
                "worker",
                "ep_a",
                1,
                Some("lora-a"),
                lora_a,
            ))
            .unwrap();
        multi_ep
            .register_model_card(model_instance("ns", "worker", "ep_b", 1, None, base_b))
            .unwrap();
        multi_ep
            .register_model_card(model_instance(
                "ns",
                "worker",
                "ep_b",
                1,
                Some("lora-b"),
                lora_b,
            ))
            .unwrap();

        let multi_cr = build_cr("worker-0", "worker-0", "pod-uid", &multi_ep).unwrap();
        assert_eq!(
            multi_cr.spec.data["model_cards"]["ns/worker/ep_a/1/lora-a"][MODEL_CARD_BASE_KEY_FIELD],
            "ns/worker/ep_a/1"
        );
        assert_eq!(
            multi_cr.spec.data["model_cards"]["ns/worker/ep_b/1/lora-b"][MODEL_CARD_BASE_KEY_FIELD],
            "ns/worker/ep_b/1"
        );
        assert_eq!(
            model_cards_by_id(&deserialize_metadata(multi_cr.spec.data).unwrap()),
            model_cards_by_id(&multi_ep)
        );

        // 4. SSA rewriting an empty card_json overlay or card_unset to null still expands cleanly.
        let ssa_null_overlay = serde_json::json!({
            "endpoints": {},
            "model_cards": {
                "ns/worker/generate/1": {
                    "type": "Model",
                    "namespace": "ns",
                    "component": "worker",
                    "endpoint": "generate",
                    "instance_id": 1,
                    "card_json": base_card,
                },
                "ns/worker/generate/1/identical": {
                    "type": "Model",
                    "namespace": "ns",
                    "component": "worker",
                    "endpoint": "generate",
                    "instance_id": 1,
                    "model_suffix": "identical",
                    "card_base_key": "ns/worker/generate/1",
                    "card_json": null,
                    "card_unset": null,
                }
            }
        });
        let expanded = deserialize_metadata(ssa_null_overlay).unwrap();
        assert_eq!(expanded.get_all_model_cards().len(), 2);
        let cards = model_cards_by_id(&expanded);
        let identical_id = DiscoveryInstanceId::Model(ModelCardInstanceId {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            endpoint: "generate".to_string(),
            instance_id: 1,
            model_suffix: Some("identical".to_string()),
        });
        let DiscoveryInstance::Model { card_json, .. } = &cards[&identical_id] else {
            unreachable!()
        };
        assert_eq!(*card_json, base_card);
    }

    #[test]
    fn malformed_compacted_model_cards_are_rejected() {
        let base_entry = serde_json::json!({
            "type": "Model",
            "namespace": "ns",
            "component": "worker",
            "endpoint": "generate",
            "instance_id": 1,
            "card_json": { "display_name": "base", "scalar": 1 }
        });

        let invalid_cases = [
            // Missing referenced base key.
            serde_json::json!({
                "model_cards": {
                    "ns/worker/generate/1/lora": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "model_suffix": "lora",
                        "card_base_key": "ns/worker/generate/1",
                        "card_json": { "display_name": "lora" }
                    }
                }
            }),
            // Self-referential base key.
            serde_json::json!({
                "model_cards": {
                    "ns/worker/generate/1/lora": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "model_suffix": "lora",
                        "card_base_key": "ns/worker/generate/1/lora",
                        "card_json": {}
                    }
                }
            }),
            // Chained base key (pointing to a suffixed / compacted card).
            serde_json::json!({
                "model_cards": {
                    "ns/worker/generate/1": base_entry.clone(),
                    "ns/worker/generate/1/a": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "model_suffix": "a",
                        "card_base_key": "ns/worker/generate/1",
                        "card_json": { "display_name": "a" }
                    },
                    "ns/worker/generate/1/b": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "model_suffix": "b",
                        "card_base_key": "ns/worker/generate/1/a",
                        "card_json": { "display_name": "b" }
                    }
                }
            }),
            // Non-object card_json overlay.
            serde_json::json!({
                "model_cards": {
                    "ns/worker/generate/1": base_entry.clone(),
                    "ns/worker/generate/1/lora": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "model_suffix": "lora",
                        "card_base_key": "ns/worker/generate/1",
                        "card_json": ["not-an-object"]
                    }
                }
            }),
            // Invalid unset marker (false instead of true/object).
            serde_json::json!({
                "model_cards": {
                    "ns/worker/generate/1": base_entry.clone(),
                    "ns/worker/generate/1/lora": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "model_suffix": "lora",
                        "card_base_key": "ns/worker/generate/1",
                        "card_json": {},
                        "card_unset": { "scalar": false }
                    }
                }
            }),
            // Nested unset targeting a non-object scalar field.
            serde_json::json!({
                "model_cards": {
                    "ns/worker/generate/1": base_entry.clone(),
                    "ns/worker/generate/1/lora": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "model_suffix": "lora",
                        "card_base_key": "ns/worker/generate/1",
                        "card_json": {},
                        "card_unset": { "scalar": { "nested": true } }
                    }
                }
            }),
            // card_unset without card_base_key.
            serde_json::json!({
                "model_cards": {
                    "ns/worker/generate/1": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "card_json": { "display_name": "base" },
                        "card_unset": { "aliases": true }
                    }
                }
            }),
            // Cross-endpoint base reference.
            serde_json::json!({
                "model_cards": {
                    "ns/worker/other_ep/1": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "other_ep",
                        "instance_id": 1,
                        "card_json": { "display_name": "other-base" }
                    },
                    "ns/worker/generate/1/lora": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "model_suffix": "lora",
                        "card_base_key": "ns/worker/other_ep/1",
                        "card_json": { "display_name": "lora" }
                    }
                }
            }),
            // Cross-instance base reference.
            serde_json::json!({
                "model_cards": {
                    "ns/worker/generate/2": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 2,
                        "card_json": { "display_name": "other-instance" }
                    },
                    "ns/worker/generate/1/lora": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "model_suffix": "lora",
                        "card_base_key": "ns/worker/generate/2",
                        "card_json": { "display_name": "lora" }
                    }
                }
            }),
            // Unsuffixed base model card cannot specify card_base_key.
            serde_json::json!({
                "model_cards": {
                    "ns/worker/generate/1": base_entry.clone(),
                    "ns/worker/generate/2": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 2,
                        "card_base_key": "ns/worker/generate/1",
                        "card_json": { "display_name": "not-suffixed" }
                    }
                }
            }),
            // Base entry stored under the expected key with mismatched embedded namespace.
            serde_json::json!({
                "model_cards": {
                    "ns/worker/generate/1": {
                        "type": "Model",
                        "namespace": "other-ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "card_json": { "display_name": "other-ns-base" }
                    },
                    "ns/worker/generate/1/lora": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "model_suffix": "lora",
                        "card_base_key": "ns/worker/generate/1",
                        "card_json": { "display_name": "lora" }
                    }
                }
            }),
            // Base entry stored under the expected key with mismatched embedded component.
            serde_json::json!({
                "model_cards": {
                    "ns/worker/generate/1": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "other-worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "card_json": { "display_name": "other-component-base" }
                    },
                    "ns/worker/generate/1/lora": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "model_suffix": "lora",
                        "card_base_key": "ns/worker/generate/1",
                        "card_json": { "display_name": "lora" }
                    }
                }
            }),
            // Base entry stored under the expected key with mismatched embedded endpoint.
            serde_json::json!({
                "model_cards": {
                    "ns/worker/generate/1": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "other_ep",
                        "instance_id": 1,
                        "card_json": { "display_name": "other-ep-base" }
                    },
                    "ns/worker/generate/1/lora": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "model_suffix": "lora",
                        "card_base_key": "ns/worker/generate/1",
                        "card_json": { "display_name": "lora" }
                    }
                }
            }),
            // Base entry stored under the expected key with mismatched embedded instance_id.
            serde_json::json!({
                "model_cards": {
                    "ns/worker/generate/1": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 2,
                        "card_json": { "display_name": "other-instance-base" }
                    },
                    "ns/worker/generate/1/lora": {
                        "type": "Model",
                        "namespace": "ns",
                        "component": "worker",
                        "endpoint": "generate",
                        "instance_id": 1,
                        "model_suffix": "lora",
                        "card_base_key": "ns/worker/generate/1",
                        "card_json": { "display_name": "lora" }
                    }
                }
            }),
        ];

        for (index, case) in invalid_cases.into_iter().enumerate() {
            assert!(
                deserialize_metadata(case).is_err(),
                "invalid compacted CR case {index} should fail deserialization"
            );
        }
    }
}
