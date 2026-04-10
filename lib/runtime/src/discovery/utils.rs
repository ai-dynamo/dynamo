// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Utility functions for working with discovery streams

use serde::Deserialize;

use super::{DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryStream};

/// Collapse composite-keyed state into a flat HashMap<u64, V>.
/// When multiple entries share the same instance_id (e.g., base model +
/// LoRA adapters on the same worker), the base model (suffix=None) is
/// preferred. If no base model exists, an arbitrary LoRA entry is used.
fn collapse_by_instance_id<V: Clone>(
    state: &std::collections::HashMap<(u64, Option<String>), V>,
) -> std::collections::HashMap<u64, V> {
    let mut result = std::collections::HashMap::new();
    for ((id, suffix), val) in state {
        if suffix.is_none() || !result.contains_key(id) {
            result.insert(*id, val.clone());
        }
    }
    result
}

/// Helper to watch a discovery stream and extract a specific field into a HashMap
///
/// This helper spawns a background task that:
/// - Deserializes ModelCards from discovery events
/// - Extracts a specific field using the provided extractor function
/// - Maintains a HashMap<instance_id, Field> that auto-updates on Add/Remove events
/// - Returns a watch::Receiver that consumers can use to read the current state
///
/// # Type Parameters
/// - `T`: The type to deserialize from DiscoveryInstance (e.g., ModelDeploymentCard)
/// - `V`: The extracted field type (e.g., ModelRuntimeConfig)
/// - `F`: The extractor function type
///
/// # Arguments
/// - `stream`: The discovery event stream to watch
/// - `extractor`: Function that extracts the desired field from the deserialized type
///
/// # Example
/// ```ignore
/// let stream = discovery.list_and_watch(DiscoveryQuery::ComponentModels { ... }, None).await?;
/// let runtime_configs_rx = watch_and_extract_field(
///     stream,
///     |card: ModelDeploymentCard| card.runtime_config,
/// );
///
/// // Use it:
/// let configs = runtime_configs_rx.borrow();
/// if let Some(config) = configs.get(&worker_id) {
///     // Use config...
/// }
/// ```
pub fn watch_and_extract_field<T, V, F>(
    stream: DiscoveryStream,
    extractor: F,
) -> tokio::sync::watch::Receiver<std::collections::HashMap<u64, V>>
where
    T: for<'de> Deserialize<'de> + 'static,
    V: Clone + Send + Sync + 'static,
    F: Fn(T) -> V + Send + 'static,
{
    use futures::StreamExt;
    use std::collections::HashMap;

    let (tx, rx) = tokio::sync::watch::channel(HashMap::new());

    tokio::spawn(async move {
        // Internal state keyed by (instance_id, model_suffix) to correctly
        // track base models and LoRA adapters sharing the same instance_id.
        // Collapsed to HashMap<u64, V> for consumers, preferring suffix=None
        // (base model) when multiple entries exist for the same instance_id.
        let mut state: HashMap<(u64, Option<String>), V> = HashMap::new();
        let mut stream = stream;

        while let Some(result) = stream.next().await {
            match result {
                Ok(DiscoveryEvent::Added(instance)) => {
                    let instance_id = instance.instance_id();
                    let model_suffix = match &instance {
                        DiscoveryInstance::Model { model_suffix, .. } => model_suffix.clone(),
                        _ => None,
                    };

                    // Deserialize the full instance into type T
                    let deserialized: T = match instance.deserialize_model() {
                        Ok(d) => d,
                        Err(e) => {
                            tracing::warn!(
                                instance_id,
                                error = %e,
                                "Failed to deserialize discovery instance, skipping"
                            );
                            continue;
                        }
                    };

                    // Extract the field we care about
                    let value = extractor(deserialized);

                    tracing::debug!(
                        instance_id,
                        ?model_suffix,
                        state_len = state.len(),
                        "watch_and_extract_field: inserting instance"
                    );

                    let key = (instance_id, model_suffix);
                    state.insert(key, value);

                    let collapsed = collapse_by_instance_id(&state);
                    if tx.send(collapsed).is_err() {
                        tracing::debug!("watch_and_extract_field receiver dropped, stopping");
                        break;
                    }
                }
                Ok(DiscoveryEvent::Removed(id)) => {
                    let instance_id = id.instance_id();
                    let model_suffix = match &id {
                        DiscoveryInstanceId::Model(mid) => mid.model_suffix.clone(),
                        _ => None,
                    };

                    let key = (instance_id, model_suffix.clone());
                    let had_entry = state.contains_key(&key);

                    tracing::debug!(
                        instance_id,
                        ?model_suffix,
                        had_entry,
                        state_len = state.len(),
                        "watch_and_extract_field: removing instance"
                    );

                    state.remove(&key);

                    let collapsed = collapse_by_instance_id(&state);
                    if tx.send(collapsed).is_err() {
                        tracing::debug!("watch_and_extract_field receiver dropped, stopping");
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "Discovery event stream error in watch_and_extract_field");
                    // Continue processing other events
                }
            }
        }

        tracing::debug!("watch_and_extract_field task stopped");
    });

    rx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::mock::{MockDiscovery, SharedMockRegistry};
    use crate::discovery::{Discovery, DiscoveryQuery, DiscoverySpec};

    /// Minimal struct that mirrors the fields watch_and_extract_field deserializes.
    #[derive(serde::Deserialize, Clone, Debug)]
    struct FakeCard {
        display_name: String,
    }

    fn model_spec(name: &str) -> DiscoverySpec {
        DiscoverySpec::Model {
            namespace: "ns".to_string(),
            component: "comp".to_string(),
            endpoint: "generate".to_string(),
            card_json: serde_json::json!({ "display_name": name }),
            model_suffix: None,
        }
    }

    /// Poll a watch receiver until the predicate is satisfied, or timeout after 1s.
    async fn poll_until(
        rx: &tokio::sync::watch::Receiver<std::collections::HashMap<u64, String>>,
        pred: impl Fn(&std::collections::HashMap<u64, String>) -> bool,
        msg: &str,
    ) {
        for _ in 0..100 {
            if pred(&rx.borrow()) {
                return;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
        panic!("{}: state={:?}", msg, *rx.borrow());
    }

    fn lora_spec(lora_name: &str) -> DiscoverySpec {
        DiscoverySpec::Model {
            namespace: "ns".to_string(),
            component: "comp".to_string(),
            endpoint: "generate".to_string(),
            card_json: serde_json::json!({
                "display_name": lora_name,
                "source_path": "base-model",
                "lora": { "name": lora_name },
            }),
            model_suffix: Some(lora_name.to_string()),
        }
    }

    /// Unregistering a single LoRA adapter must not remove the worker's
    /// runtime config. Base model and other LoRA adapters on the same worker
    /// share the same instance_id; removing one must leave the others intact.
    #[tokio::test]
    async fn test_lora_unregister_preserves_worker_runtime_config() {
        // All registrations use the same instance_id (same worker)
        let discovery = MockDiscovery::new(Some(42), SharedMockRegistry::new());

        let query = DiscoveryQuery::EndpointModels {
            namespace: "ns".to_string(),
            component: "comp".to_string(),
            endpoint: "generate".to_string(),
        };

        let stream = discovery.list_and_watch(query, None).await.unwrap();

        // Watch the stream, extracting display_name as a stand-in for runtime_config
        let rx = watch_and_extract_field(stream, |card: FakeCard| card.display_name);

        // Register base model + LoRA-A + LoRA-B on the same worker (instance_id=42)
        let base = discovery.register(model_spec("base-model")).await.unwrap();
        let lora_a = discovery.register(lora_spec("lora-a")).await.unwrap();
        discovery.register(lora_spec("lora-b")).await.unwrap();

        poll_until(
            &rx,
            |s| s.contains_key(&42),
            "Worker 42 should be present after registrations",
        )
        .await;

        // Unregister LoRA-A only — base model and LoRA-B remain.
        discovery.unregister(lora_a).await.unwrap();

        // Base model is preferred in the collapsed view.
        poll_until(
            &rx,
            |s| s.get(&42).map(|v| v.as_str()) == Some("base-model"),
            "Worker 42 should have base-model after removing lora-a",
        )
        .await;

        {
            let state = rx.borrow();
            assert_eq!(state.get(&42).map(|s| s.as_str()), Some("base-model"));
        }

        // Unregister the base model — lora-b should be the fallback.
        discovery.unregister(base).await.unwrap();

        poll_until(
            &rx,
            |s| s.get(&42).map(|v| v.as_str()) == Some("lora-b"),
            "Worker 42 should fall back to lora-b after removing base model",
        )
        .await;

        {
            let state = rx.borrow();
            assert_eq!(state.get(&42).map(|s| s.as_str()), Some("lora-b"));
        }
    }
}
