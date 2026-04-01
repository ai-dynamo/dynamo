// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Utility functions for working with discovery streams

use serde::Deserialize;

use super::{DiscoveryEvent, DiscoveryInstance, DiscoveryStream};

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
///
/// This helper is only appropriate when the watched objects are keyed solely by
/// raw `instance_id`. For model-card streams, use
/// `watch_and_extract_base_model_field` when reducing to worker-level state.
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
        let mut state: HashMap<u64, V> = HashMap::new();
        let mut stream = stream;

        while let Some(result) = stream.next().await {
            match result {
                Ok(DiscoveryEvent::Added(instance)) => {
                    let instance_id = instance.instance_id();

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

                    // Update state and send
                    state.insert(instance_id, value);
                    if tx.send(state.clone()).is_err() {
                        tracing::debug!("watch_and_extract_field receiver dropped, stopping");
                        break;
                    }
                }
                Ok(DiscoveryEvent::Removed(id)) => {
                    // Remove from state and send update
                    state.remove(&id.instance_id());
                    if tx.send(state.clone()).is_err() {
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

/// Helper to watch a model-card discovery stream and extract a field from base model cards only.
///
/// This is intended for worker-level fields such as runtime config where LoRA model-card churn
/// must not add, replace, or delete the worker's state.
pub fn watch_and_extract_base_model_field<T, V, F>(
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
        let mut state: HashMap<u64, V> = HashMap::new();
        let mut stream = stream;

        while let Some(result) = stream.next().await {
            match result {
                Ok(DiscoveryEvent::Added(DiscoveryInstance::Model {
                    instance_id,
                    card_json,
                    model_suffix,
                    ..
                })) => {
                    // Runtime config and similar worker-level state is sourced from the
                    // canonical base model card, not LoRA overlays on the same worker.
                    if model_suffix.is_some() {
                        continue;
                    }

                    let deserialized: T = match serde_json::from_value(card_json) {
                        Ok(d) => d,
                        Err(e) => {
                            tracing::warn!(
                                instance_id,
                                error = %e,
                                "Failed to deserialize base model discovery instance, skipping"
                            );
                            continue;
                        }
                    };

                    let value = extractor(deserialized);
                    state.insert(instance_id, value);
                    if tx.send(state.clone()).is_err() {
                        tracing::debug!(
                            "watch_and_extract_base_model_field receiver dropped, stopping"
                        );
                        break;
                    }
                }
                Ok(DiscoveryEvent::Added(_)) => {
                    tracing::debug!(
                        "watch_and_extract_base_model_field received non-model discovery instance"
                    );
                }
                Ok(DiscoveryEvent::Removed(id)) => {
                    let model_id = match id.extract_model_id() {
                        Ok(model_id) => model_id,
                        Err(_) => {
                            tracing::debug!(
                                "watch_and_extract_base_model_field received non-model removal"
                            );
                            continue;
                        }
                    };

                    if model_id.model_suffix.is_some() {
                        continue;
                    }

                    state.remove(&model_id.instance_id);
                    if tx.send(state.clone()).is_err() {
                        tracing::debug!(
                            "watch_and_extract_base_model_field receiver dropped, stopping"
                        );
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "Discovery event stream error in watch_and_extract_base_model_field"
                    );
                }
            }
        }

        tracing::debug!("watch_and_extract_base_model_field task stopped");
    });

    rx
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use serde::{Deserialize, Serialize};

    use super::watch_and_extract_base_model_field;
    use crate::discovery::{
        Discovery, DiscoveryQuery, DiscoverySpec, MockDiscovery, SharedMockRegistry,
    };

    #[derive(Debug, Clone, Deserialize, Serialize)]
    struct TestCard {
        runtime_config: u32,
    }

    fn endpoint_model_query() -> DiscoveryQuery {
        DiscoveryQuery::EndpointModels {
            namespace: "ns".to_string(),
            component: "comp".to_string(),
            endpoint: "generate".to_string(),
        }
    }

    fn base_model_spec(runtime_config: u32) -> DiscoverySpec {
        DiscoverySpec::Model {
            namespace: "ns".to_string(),
            component: "comp".to_string(),
            endpoint: "generate".to_string(),
            card_json: serde_json::json!({
                "display_name": "base-model",
                "source_path": "base-repo",
                "runtime_config": runtime_config,
            }),
            model_suffix: None,
        }
    }

    fn lora_model_spec(name: &str, runtime_config: u32) -> DiscoverySpec {
        DiscoverySpec::Model {
            namespace: "ns".to_string(),
            component: "comp".to_string(),
            endpoint: "generate".to_string(),
            card_json: serde_json::json!({
                "display_name": name,
                "source_path": "base-repo",
                "runtime_config": runtime_config,
                "lora": {
                    "name": name,
                },
            }),
            model_suffix: Some(name.to_string()),
        }
    }

    async fn wait_for_state<F>(
        rx: &mut tokio::sync::watch::Receiver<HashMap<u64, u32>>,
        predicate: F,
    ) where
        F: Fn(&HashMap<u64, u32>) -> bool,
    {
        if predicate(&rx.borrow()) {
            return;
        }

        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                rx.changed()
                    .await
                    .expect("base model field watch should remain open");
                if predicate(&rx.borrow()) {
                    break;
                }
            }
        })
        .await
        .expect("timed out waiting for watch state");
    }

    #[tokio::test]
    async fn watch_and_extract_base_model_field_ignores_lora_add_remove() {
        let registry = SharedMockRegistry::new();
        let discovery = MockDiscovery::new(Some(0xabc), registry);

        let stream = discovery
            .list_and_watch(endpoint_model_query(), None)
            .await
            .unwrap();
        let mut rx =
            watch_and_extract_base_model_field(stream, |card: TestCard| card.runtime_config);

        let base = discovery.register(base_model_spec(42)).await.unwrap();
        let lora_a = discovery
            .register(lora_model_spec("adapter-a", 100))
            .await
            .unwrap();
        let _lora_b = discovery
            .register(lora_model_spec("adapter-b", 101))
            .await
            .unwrap();

        wait_for_state(&mut rx, |state| state.get(&0xabc) == Some(&42)).await;

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(rx.borrow().get(&0xabc), Some(&42));

        discovery.unregister(lora_a).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(rx.borrow().get(&0xabc), Some(&42));

        discovery.unregister(base).await.unwrap();
        wait_for_state(&mut rx, |state| !state.contains_key(&0xabc)).await;
    }

    #[tokio::test]
    async fn watch_and_extract_base_model_field_handles_lora_before_base_and_base_removal() {
        let registry = SharedMockRegistry::new();
        let discovery = MockDiscovery::new(Some(0xabc), registry);

        let stream = discovery
            .list_and_watch(endpoint_model_query(), None)
            .await
            .unwrap();
        let mut rx =
            watch_and_extract_base_model_field(stream, |card: TestCard| card.runtime_config);

        let lora_a = discovery
            .register(lora_model_spec("adapter-a", 100))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(rx.borrow().is_empty());

        let base = discovery.register(base_model_spec(42)).await.unwrap();
        let _lora_b = discovery
            .register(lora_model_spec("adapter-b", 101))
            .await
            .unwrap();

        wait_for_state(&mut rx, |state| state.get(&0xabc) == Some(&42)).await;

        discovery.unregister(lora_a).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(rx.borrow().get(&0xabc), Some(&42));

        discovery.unregister(base).await.unwrap();
        wait_for_state(&mut rx, |state| state.is_empty()).await;
    }
}
