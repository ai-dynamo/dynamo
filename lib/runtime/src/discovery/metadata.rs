// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::{DiscoveryInstance, DiscoveryQuery};

/// Key for organizing metadata internally
/// Format: "namespace/component/endpoint"
fn make_endpoint_key(namespace: &str, component: &str, endpoint: &str) -> String {
    format!("{namespace}/{component}/{endpoint}")
}

/// Metadata stored on each pod and exposed via HTTP endpoint
/// This struct holds all discovery registrations for this pod instance
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiscoveryMetadata {
    /// Registered endpoint instances (key: "namespace/component/endpoint")
    endpoints: HashMap<String, DiscoveryInstance>,
    /// Registered model card instances (key: "namespace/component/endpoint")
    model_cards: HashMap<String, DiscoveryInstance>,
}

impl DiscoveryMetadata {
    /// Create a new empty metadata store
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
            model_cards: HashMap::new(),
        }
    }

    /// Register an endpoint instance
    pub fn register_endpoint(&mut self, instance: DiscoveryInstance) -> Result<()> {
        if let DiscoveryInstance::Endpoint(ref inst) = instance {
            let key = make_endpoint_key(&inst.namespace, &inst.component, &inst.endpoint);
            self.endpoints.insert(key, instance);
            Ok(())
        } else {
            anyhow::bail!("Cannot register non-endpoint instance as endpoint")
        }
    }

    /// Register a model card instance
    pub fn register_model_card(&mut self, instance: DiscoveryInstance) -> Result<()> {
        if let DiscoveryInstance::Model {
            ref namespace,
            ref component,
            ref endpoint,
            ..
        } = instance
        {
            let key = make_endpoint_key(namespace, component, endpoint);
            self.model_cards.insert(key, instance);
            Ok(())
        } else {
            anyhow::bail!("Cannot register non-model-card instance as model card")
        }
    }

    /// Unregister an endpoint instance
    pub fn unregister_endpoint(&mut self, instance: &DiscoveryInstance) -> Result<()> {
        if let DiscoveryInstance::Endpoint(inst) = instance {
            let key = make_endpoint_key(&inst.namespace, &inst.component, &inst.endpoint);
            self.endpoints.remove(&key);
            Ok(())
        } else {
            anyhow::bail!("Cannot unregister non-endpoint instance as endpoint")
        }
    }

    /// Unregister a model card instance
    pub fn unregister_model_card(&mut self, instance: &DiscoveryInstance) -> Result<()> {
        if let DiscoveryInstance::Model {
            namespace,
            component,
            endpoint,
            ..
        } = instance
        {
            let key = make_endpoint_key(namespace, component, endpoint);
            self.model_cards.remove(&key);
            Ok(())
        } else {
            anyhow::bail!("Cannot unregister non-model-card instance as model card")
        }
    }

    /// Get all registered endpoints
    pub fn get_all_endpoints(&self) -> Vec<DiscoveryInstance> {
        self.endpoints.values().cloned().collect()
    }

    /// Get all registered model cards
    pub fn get_all_model_cards(&self) -> Vec<DiscoveryInstance> {
        self.model_cards.values().cloned().collect()
    }

    /// Get all registered instances (endpoints and model cards)
    pub fn get_all(&self) -> Vec<DiscoveryInstance> {
        self.endpoints
            .values()
            .chain(self.model_cards.values())
            .cloned()
            .collect()
    }

    /// Filter this metadata by query
    pub fn filter(&self, query: &DiscoveryQuery) -> Vec<DiscoveryInstance> {
        let all_instances = match query {
            DiscoveryQuery::AllEndpoints
            | DiscoveryQuery::NamespacedEndpoints { .. }
            | DiscoveryQuery::ComponentEndpoints { .. }
            | DiscoveryQuery::Endpoint { .. } => self.get_all_endpoints(),

            DiscoveryQuery::AllModels
            | DiscoveryQuery::NamespacedModels { .. }
            | DiscoveryQuery::ComponentModels { .. }
            | DiscoveryQuery::EndpointModels { .. } => self.get_all_model_cards(),
        };

        filter_instances(all_instances, query)
    }
}

impl Default for DiscoveryMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Filter instances by query predicate
fn filter_instances(
    instances: Vec<DiscoveryInstance>,
    query: &DiscoveryQuery,
) -> Vec<DiscoveryInstance> {
    match query {
        DiscoveryQuery::AllEndpoints | DiscoveryQuery::AllModels => instances,

        DiscoveryQuery::NamespacedEndpoints { namespace } => instances
            .into_iter()
            .filter(|inst| match inst {
                DiscoveryInstance::Endpoint(i) => &i.namespace == namespace,
                _ => false,
            })
            .collect(),

        DiscoveryQuery::ComponentEndpoints {
            namespace,
            component,
        } => instances
            .into_iter()
            .filter(|inst| match inst {
                DiscoveryInstance::Endpoint(i) => {
                    &i.namespace == namespace && &i.component == component
                }
                _ => false,
            })
            .collect(),

        DiscoveryQuery::Endpoint {
            namespace,
            component,
            endpoint,
        } => instances
            .into_iter()
            .filter(|inst| match inst {
                DiscoveryInstance::Endpoint(i) => {
                    &i.namespace == namespace
                        && &i.component == component
                        && &i.endpoint == endpoint
                }
                _ => false,
            })
            .collect(),

        DiscoveryQuery::NamespacedModels { namespace } => instances
            .into_iter()
            .filter(|inst| match inst {
                DiscoveryInstance::Model { namespace: ns, .. } => ns == namespace,
                _ => false,
            })
            .collect(),

        DiscoveryQuery::ComponentModels {
            namespace,
            component,
        } => instances
            .into_iter()
            .filter(|inst| match inst {
                DiscoveryInstance::Model {
                    namespace: ns,
                    component: comp,
                    ..
                } => ns == namespace && comp == component,
                _ => false,
            })
            .collect(),

        DiscoveryQuery::EndpointModels {
            namespace,
            component,
            endpoint,
        } => instances
            .into_iter()
            .filter(|inst| match inst {
                DiscoveryInstance::Model {
                    namespace: ns,
                    component: comp,
                    endpoint: ep,
                    ..
                } => ns == namespace && comp == component && ep == endpoint,
                _ => false,
            })
            .collect(),
    }
}

/// Snapshot of all discovered instances and their metadata
#[derive(Clone, Debug)]
pub struct MetadataSnapshot {
    /// Map of instance_id -> metadata
    pub instances: HashMap<u64, Arc<DiscoveryMetadata>>,
    /// CR name -> generation for change detection
    pub generations: HashMap<String, i64>,
    /// Sequence number for debugging
    pub sequence: u64,
    /// Timestamp for observability
    pub timestamp: std::time::Instant,
}

impl MetadataSnapshot {
    pub fn empty() -> Self {
        Self {
            instances: HashMap::new(),
            generations: HashMap::new(),
            sequence: 0,
            timestamp: std::time::Instant::now(),
        }
    }

    /// Compare with previous snapshot and return true if changed.
    /// Logs diagnostic info about what changed.
    ///
    /// We check both:
    /// - `instances` keys: Which pods are ready AND have metadata
    /// - `generations`: CR metadata versions
    ///
    /// This is important because when a pod goes into Terminating, it's removed
    /// from the EndpointSlice (so not in `instances`) but the CR may still exist
    /// with the same generation until garbage-collected.
    pub fn has_changes_from(&self, prev: &MetadataSnapshot) -> bool {
        // Check if instance IDs changed (pods added/removed from ready state)
        let curr_instance_ids: HashSet<u64> = self.instances.keys().copied().collect();
        let prev_instance_ids: HashSet<u64> = prev.instances.keys().copied().collect();
        let instances_changed = curr_instance_ids != prev_instance_ids;

        // Check if CR generations changed (metadata updated)
        let generations_changed = self.generations != prev.generations;

        if !instances_changed && !generations_changed {
            tracing::trace!(
                "Snapshot (seq={}): no changes, {} instances",
                self.sequence,
                self.instances.len()
            );
            return false;
        }

        // Compute diff for logging
        let instances_added: Vec<_> = curr_instance_ids
            .difference(&prev_instance_ids)
            .map(|id| format!("{:x}", id))
            .collect();
        let instances_removed: Vec<_> = prev_instance_ids
            .difference(&curr_instance_ids)
            .map(|id| format!("{:x}", id))
            .collect();

        let crs_added: Vec<_> = self
            .generations
            .keys()
            .filter(|k| !prev.generations.contains_key(*k))
            .collect();
        let crs_removed: Vec<_> = prev
            .generations
            .keys()
            .filter(|k| !self.generations.contains_key(*k))
            .collect();
        let crs_updated: Vec<_> = self
            .generations
            .iter()
            .filter(|(k, v)| prev.generations.get(*k).is_some_and(|pv| pv != *v))
            .map(|(k, _)| k)
            .collect();

        tracing::info!(
            "Snapshot (seq={}): {} instances, instances_added={:?}, instances_removed={:?}, crs_added={:?}, crs_removed={:?}, crs_updated={:?}",
            self.sequence,
            self.instances.len(),
            instances_added,
            instances_removed,
            crs_added,
            crs_removed,
            crs_updated
        );

        true
    }

    /// Filter all instances in the snapshot by query
    pub fn filter(&self, query: &DiscoveryQuery) -> Vec<DiscoveryInstance> {
        self.instances
            .values()
            .flat_map(|metadata| metadata.filter(query))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::{Instance, TransportType};

    #[test]
    fn test_metadata_serde() {
        let mut metadata = DiscoveryMetadata::new();

        // Add an endpoint
        let instance = DiscoveryInstance::Endpoint(Instance {
            namespace: "test".to_string(),
            component: "comp1".to_string(),
            endpoint: "ep1".to_string(),
            instance_id: 123,
            transport: TransportType::Nats("nats://localhost:4222".to_string()),
        });

        metadata.register_endpoint(instance).unwrap();

        // Serialize
        let json = serde_json::to_string(&metadata).unwrap();

        // Deserialize
        let deserialized: DiscoveryMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.endpoints.len(), 1);
        assert_eq!(deserialized.model_cards.len(), 0);
    }

    #[tokio::test]
    async fn test_concurrent_registration() {
        use tokio::sync::RwLock;

        let metadata = Arc::new(RwLock::new(DiscoveryMetadata::new()));

        // Spawn multiple tasks registering concurrently
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let metadata = metadata.clone();
                tokio::spawn(async move {
                    let mut meta = metadata.write().await;
                    let instance = DiscoveryInstance::Endpoint(Instance {
                        namespace: "test".to_string(),
                        component: "comp1".to_string(),
                        endpoint: format!("ep{}", i),
                        instance_id: i,
                        transport: TransportType::Nats("nats://localhost:4222".to_string()),
                    });
                    meta.register_endpoint(instance).unwrap();
                })
            })
            .collect();

        // Wait for all to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify all registrations succeeded
        let meta = metadata.read().await;
        assert_eq!(meta.endpoints.len(), 10);
    }

    #[tokio::test]
    async fn test_metadata_accessors() {
        let mut metadata = DiscoveryMetadata::new();

        // Register endpoints
        for i in 0..3 {
            let instance = DiscoveryInstance::Endpoint(Instance {
                namespace: "test".to_string(),
                component: "comp1".to_string(),
                endpoint: format!("ep{}", i),
                instance_id: i,
                transport: TransportType::Nats("nats://localhost:4222".to_string()),
            });
            metadata.register_endpoint(instance).unwrap();
        }

        // Register model cards
        for i in 0..2 {
            let instance = DiscoveryInstance::Model {
                namespace: "test".to_string(),
                component: "comp1".to_string(),
                endpoint: format!("ep{}", i),
                instance_id: i,
                card_json: serde_json::json!({"model": "test"}),
                model_suffix: None,
            };
            metadata.register_model_card(instance).unwrap();
        }

        assert_eq!(metadata.get_all_endpoints().len(), 3);
        assert_eq!(metadata.get_all_model_cards().len(), 2);
        assert_eq!(metadata.get_all().len(), 5);
    }
}
