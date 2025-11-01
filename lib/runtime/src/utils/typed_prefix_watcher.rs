// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic etcd watcher utilities for maintaining collated state from etcd prefixes.
//!
//! This module provides reusable patterns for watching etcd prefixes and maintaining
//! HashMap-based state that automatically updates based on etcd events.

use crate::Result;
use crate::transports::etcd::{Client as EtcdClient, WatchEvent};
use etcd_client::KeyValue;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::fmt::Debug;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

/// A generic etcd prefix watcher that maintains a HashMap of deserialized values.
///
/// This struct watches an etcd prefix and maintains a HashMap where:
/// - Keys are extracted from the etcd KeyValue (e.g., lease_id, key string, etc.)
/// - Values are extracted from the deserialized type using a value extractor
///
/// # Type Parameters
/// - `K`: The key type for the HashMap (must be hashable)
/// - `V`: The value type stored in the HashMap
pub struct TypedPrefixWatcher<K, V>
where
    K: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    rx: watch::Receiver<HashMap<K, V>>,
}

impl<K, V> TypedPrefixWatcher<K, V>
where
    K: Clone + Eq + std::hash::Hash + Send + Sync + Debug + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Get a receiver for the current state
    pub fn receiver(&self) -> watch::Receiver<HashMap<K, V>> {
        self.rx.clone()
    }

    /// Get the current state
    pub fn current(&self) -> HashMap<K, V> {
        self.rx.borrow().clone()
    }
}

/// Watch an etcd prefix and maintain a HashMap of values with field extraction
///
/// This function watches an etcd prefix and maintains a HashMap where values are
/// extracted from a deserialized type using a value extractor function.
///
/// # Type Parameters
/// - `K`: The key type for the HashMap
/// - `V`: The value type stored in the HashMap
/// - `T`: The type to deserialize from etcd
///
/// # Arguments
/// - `client`: The etcd client to use
/// - `prefix`: The prefix to watch in etcd
/// - `key_extractor`: Function to extract the key from a KeyValue
/// - `value_extractor`: Function to extract the value from the deserialized type
/// - `cancellation_token`: Token to stop the watcher
///
/// # Example
/// ```ignore
/// // Watch for ModelDeploymentCard objects and extract runtime_config field
/// let watcher = watch_prefix_with_extraction(
///     etcd_client,
///     "v1/mdc/",
///     |kv| Some(kv.lease()),  // Use lease_id as key
///     |card: ModelDeploymentCard| card.runtime_config,  // Extract runtime_config field
///     cancellation_token,
/// ).await?;
/// ```
pub async fn watch_prefix_with_extraction<K, V, T>(
    client: EtcdClient,
    prefix: impl Into<String>,
    key_extractor: impl Fn(&KeyValue) -> Option<K> + Send + 'static,
    value_extractor: impl Fn(T) -> Option<V> + Send + 'static,
    cancellation_token: CancellationToken,
) -> Result<TypedPrefixWatcher<K, V>>
where
    K: Clone + Eq + std::hash::Hash + Send + Sync + Debug + 'static,
    V: Clone + Send + Sync + 'static,
    T: DeserializeOwned + Send + 'static,
{
    let (watch_tx, watch_rx) = watch::channel(HashMap::new());
    let prefix = prefix.into();

    let prefix_watcher = client.kv_get_and_watch_prefix(&prefix).await?;
    let (prefix_str, mut events_rx) = prefix_watcher.dissolve();

    tokio::spawn(async move {
        let mut state: HashMap<K, V> = HashMap::new();

        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    tracing::debug!("TypedPrefixWatcher for prefix '{}' cancelled", prefix_str);
                    break;
                }
                event = events_rx.recv() => {
                    let Some(event) = event else {
                        tracing::debug!("TypedPrefixWatcher watch stream closed for prefix '{}'", prefix_str);
                        break;
                    };

                    match event {
                        WatchEvent::Put(kv) => {
                            // Extract the key
                            let Some(key) = key_extractor(&kv) else {
                                tracing::trace!("Skipping entry - key extractor returned None");
                                continue;
                            };

                            // Deserialize the value
                            let deserialized = match serde_json::from_slice::<T>(kv.value()) {
                                Ok(val) => val,
                                Err(e) => {
                                    tracing::warn!(
                                        "Failed to deserialize value from etcd. Key: {}, Error: {}",
                                        kv.key_str().unwrap_or("<invalid>"),
                                        e
                                    );
                                    continue;
                                }
                            };

                            // Extract the value
                            match value_extractor(deserialized) {
                                Some(v) => {
                                    state.insert(key.clone(), v);
                                    tracing::trace!("Updated entry for key {:?}", key);
                                }
                                None => {
                                    state.remove(&key);
                                    tracing::trace!("Removed entry for key {:?} (extractor returned None)", key);
                                }
                            }

                            if watch_tx.send(state.clone()).is_err() {
                                tracing::error!("Failed to send update; receiver dropped");
                                break;
                            }
                        }
                        WatchEvent::Delete(kv) => {
                            if let Some(key) = key_extractor(&kv) {
                                state.remove(&key);
                                tracing::trace!("Removed entry for deleted key {:?}", key);

                                if watch_tx.send(state.clone()).is_err() {
                                    tracing::error!("Failed to send update; receiver dropped");
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        tracing::debug!("TypedPrefixWatcher for prefix '{}' stopped", prefix_str);
    });

    Ok(TypedPrefixWatcher { rx: watch_rx })
}

/// Watch an etcd prefix and maintain a HashMap of values without field extraction
///
/// This is a simpler version when you want to store the entire deserialized value.
///
/// # Example
/// ```ignore
/// // Watch for TestConfig objects directly
/// let watcher = watch_prefix(
///     etcd_client,
///     "configs/",
///     |kv| Some(kv.lease()),  // Use lease_id as key
///     cancellation_token,
/// ).await?;
/// ```
pub async fn watch_prefix<K, V>(
    client: EtcdClient,
    prefix: impl Into<String>,
    key_extractor: impl Fn(&KeyValue) -> Option<K> + Send + 'static,
    cancellation_token: CancellationToken,
) -> Result<TypedPrefixWatcher<K, V>>
where
    K: Clone + Eq + std::hash::Hash + Send + Sync + Debug + 'static,
    V: Clone + DeserializeOwned + Send + Sync + 'static,
{
    watch_prefix_with_extraction(
        client,
        prefix,
        key_extractor,
        |v: V| Some(v), // Identity function - just return the value
        cancellation_token,
    )
    .await
}

/// Common key extractors for convenience
pub mod key_extractors {
    use etcd_client::KeyValue;

    /// Extract the lease ID as the key
    pub fn lease_id(kv: &KeyValue) -> Option<u64> {
        Some(kv.lease() as u64)
    }

    /// Extract the key as a string (without prefix)
    pub fn key_string(prefix: &str) -> impl Fn(&KeyValue) -> Option<String> {
        let prefix = prefix.to_string();
        move |kv: &KeyValue| {
            kv.key_str()
                .ok()
                .map(|k| k.strip_prefix(&prefix).unwrap_or(k).to_string())
        }
    }

    /// Extract the full key as a string
    pub fn full_key_string(kv: &KeyValue) -> Option<String> {
        kv.key_str().ok().map(|s| s.to_string())
    }
}
