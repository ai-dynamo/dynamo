// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use std::collections::HashMap;
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
    /// Registered metrics endpoint instances (key: "namespace/instance_id")
    metrics_endpoints: HashMap<String, DiscoveryInstance>,
}

impl DiscoveryMetadata {
    /// Create a new empty metadata store
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
            model_cards: HashMap::new(),
            metrics_endpoints: HashMap::new(),
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
    /// Register a metrics endpoint instance
    pub fn register_metrics_endpoint(&mut self, instance: DiscoveryInstance) -> Result<()> {
        if let DiscoveryInstance::MetricsEndpoint {
            ref namespace,
            instance_id,
            ..
        } = instance
        {
            let key = format!("{}/{:x}", namespace, instance_id);
            self.metrics_endpoints.insert(key, instance);
            Ok(())
        } else {
            anyhow::bail!("Cannot register non-metrics-endpoint instance as metrics endpoint")
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

    /// Get all registered metrics endpoints
    pub fn get_all_metrics_endpoints(&self) -> Vec<DiscoveryInstance> {
        self.metrics_endpoints.values().cloned().collect()
    }

    /// Get all registered instances (endpoints, model cards, and metrics endpoints)
    pub fn get_all(&self) -> Vec<DiscoveryInstance> {
        self.endpoints
            .values()
            .chain(self.model_cards.values())
            .chain(self.metrics_endpoints.values())
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

            DiscoveryQuery::AllMetricsEndpoints
            | DiscoveryQuery::NamespacedMetricsEndpoints { .. } => self.get_all_metrics_endpoints(),
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
        DiscoveryQuery::AllEndpoints | DiscoveryQuery::AllModels | DiscoveryQuery::AllMetricsEndpoints => instances,

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

        DiscoveryQuery::NamespacedMetricsEndpoints { namespace } => instances
            .into_iter()
            .filter(|inst| match inst {
                DiscoveryInstance::MetricsEndpoint { namespace: ns, .. } => ns == namespace,
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
    /// Sequence number for debugging
    pub sequence: u64,
    /// Timestamp for observability
    pub timestamp: std::time::Instant,
}

impl MetadataSnapshot {
    pub fn empty() -> Self {
        Self {
            instances: HashMap::new(),
            sequence: 0,
            timestamp: std::time::Instant::now(),
        }
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

    #[test]
    fn test_metadata_register_metrics_endpoint() {
        let mut metadata = DiscoveryMetadata::new();

        // Add a metrics endpoint
        let instance = DiscoveryInstance::MetricsEndpoint {
            namespace: "test-ns".to_string(),
            instance_id: 123,
            url: "http://localhost:8080/metrics".to_string(),
            gpu_uuids: vec![],
        };

        metadata.register_metrics_endpoint(instance).unwrap();

        assert_eq!(metadata.get_all_metrics_endpoints().len(), 1);
        assert_eq!(metadata.get_all().len(), 1);
    }

    #[test]
    fn test_metadata_filter_metrics_endpoints() {
        let mut metadata = DiscoveryMetadata::new();

        // Register metrics endpoints
        for i in 0..3 {
            let instance = DiscoveryInstance::MetricsEndpoint {
                namespace: "ns1".to_string(),
                instance_id: i,
                url: format!("http://localhost:808{}/metrics", i),
                gpu_uuids: vec![],
            };
            metadata.register_metrics_endpoint(instance).unwrap();
        }

        for i in 0..2 {
            let instance = DiscoveryInstance::MetricsEndpoint {
                namespace: "ns2".to_string(),
                instance_id: i + 100,
                url: format!("http://localhost:808{}/metrics", i + 100),
                gpu_uuids: vec![],
            };
            metadata.register_metrics_endpoint(instance).unwrap();
        }

        // Filter all metrics endpoints
        let all = metadata.filter(&DiscoveryQuery::AllMetricsEndpoints);
        assert_eq!(all.len(), 5);

        // Filter by namespace
        let ns1 = metadata.filter(&DiscoveryQuery::NamespacedMetricsEndpoints {
            namespace: "ns1".to_string(),
        });
        assert_eq!(ns1.len(), 3);

        let ns2 = metadata.filter(&DiscoveryQuery::NamespacedMetricsEndpoints {
            namespace: "ns2".to_string(),
        });
        assert_eq!(ns2.len(), 2);
    }

    #[test]
    fn test_metadata_serde_with_metrics_endpoints() {
        let mut metadata = DiscoveryMetadata::new();

        // Add a metrics endpoint
        let instance = DiscoveryInstance::MetricsEndpoint {
            namespace: "test-ns".to_string(),
            instance_id: 456,
            url: "http://localhost:8080/metrics".to_string(),
            gpu_uuids: vec![],
        };

        metadata.register_metrics_endpoint(instance).unwrap();

        // Serialize
        let json = serde_json::to_string(&metadata).unwrap();

        // Deserialize
        let deserialized: DiscoveryMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.metrics_endpoints.len(), 1);
        assert_eq!(deserialized.endpoints.len(), 0);
        assert_eq!(deserialized.model_cards.len(), 0);
    }
}
