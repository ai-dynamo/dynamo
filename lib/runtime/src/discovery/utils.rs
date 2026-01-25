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
