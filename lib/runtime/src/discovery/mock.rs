// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
    DiscoverySpec, DiscoveryStream,
};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

/// Shared in-memory registry for mock discovery
#[derive(Clone, Default)]
pub struct SharedMockRegistry {
    instances: Arc<Mutex<Vec<DiscoveryInstance>>>,
}

impl SharedMockRegistry {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Mock implementation of Discovery for testing
/// We can potentially remove this once we have KVStoreDiscovery fully tested
pub struct MockDiscovery {
    instance_id: u64,
    registry: SharedMockRegistry,
}

impl MockDiscovery {
    pub fn new(instance_id: Option<u64>, registry: SharedMockRegistry) -> Self {
        let instance_id = instance_id.unwrap_or_else(|| {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(1);
            COUNTER.fetch_add(1, Ordering::SeqCst)
        });

        Self {
            instance_id,
            registry,
        }
    }
}

/// Helper function to check if an instance matches a discovery query
fn matches_query(instance: &DiscoveryInstance, query: &DiscoveryQuery) -> bool {
    match (instance, query) {
        // Endpoint matching
        (DiscoveryInstance::Endpoint(_), DiscoveryQuery::AllEndpoints) => true,
        (DiscoveryInstance::Endpoint(inst), DiscoveryQuery::NamespacedEndpoints { namespace }) => {
            &inst.namespace == namespace
        }
        (
            DiscoveryInstance::Endpoint(inst),
            DiscoveryQuery::ComponentEndpoints {
                namespace,
                component,
            },
        ) => &inst.namespace == namespace && &inst.component == component,
        (
            DiscoveryInstance::Endpoint(inst),
            DiscoveryQuery::Endpoint {
                namespace,
                component,
                endpoint,
            },
        ) => {
            &inst.namespace == namespace
                && &inst.component == component
                && &inst.endpoint == endpoint
        }

        // Model matching
        (DiscoveryInstance::Model { .. }, DiscoveryQuery::AllModels) => true,
        (
            DiscoveryInstance::Model {
                namespace: inst_ns, ..
            },
            DiscoveryQuery::NamespacedModels { namespace },
        ) => inst_ns == namespace,
        (
            DiscoveryInstance::Model {
                namespace: inst_ns,
                component: inst_comp,
                ..
            },
            DiscoveryQuery::ComponentModels {
                namespace,
                component,
            },
        ) => inst_ns == namespace && inst_comp == component,
        (
            DiscoveryInstance::Model {
                namespace: inst_ns,
                component: inst_comp,
                endpoint: inst_ep,
                ..
            },
            DiscoveryQuery::EndpointModels {
                namespace,
                component,
                endpoint,
            },
        ) => inst_ns == namespace && inst_comp == component && inst_ep == endpoint,

        // EventChannel matching - unified query
        (
            DiscoveryInstance::EventChannel {
                namespace: inst_ns,
                component: inst_comp,
                topic: inst_topic,
                ..
            },
            DiscoveryQuery::EventChannels(query),
        ) => {
            query.namespace.as_ref().is_none_or(|ns| ns == inst_ns)
                && query.component.as_ref().is_none_or(|c| c == inst_comp)
                && query.topic.as_ref().is_none_or(|t| t == inst_topic)
        }

        // Cross-type matches return false
        (
            DiscoveryInstance::Endpoint(_),
            DiscoveryQuery::AllModels
            | DiscoveryQuery::NamespacedModels { .. }
            | DiscoveryQuery::ComponentModels { .. }
            | DiscoveryQuery::EndpointModels { .. }
            | DiscoveryQuery::EventChannels(_),
        ) => false,
        (
            DiscoveryInstance::Model { .. },
            DiscoveryQuery::AllEndpoints
            | DiscoveryQuery::NamespacedEndpoints { .. }
            | DiscoveryQuery::ComponentEndpoints { .. }
            | DiscoveryQuery::Endpoint { .. }
            | DiscoveryQuery::EventChannels(_),
        ) => false,
        (
            DiscoveryInstance::EventChannel { .. },
            DiscoveryQuery::AllEndpoints
            | DiscoveryQuery::NamespacedEndpoints { .. }
            | DiscoveryQuery::ComponentEndpoints { .. }
            | DiscoveryQuery::Endpoint { .. }
            | DiscoveryQuery::AllModels
            | DiscoveryQuery::NamespacedModels { .. }
            | DiscoveryQuery::ComponentModels { .. }
            | DiscoveryQuery::EndpointModels { .. },
        ) => false,
    }
}

#[async_trait]
impl Discovery for MockDiscovery {
    fn instance_id(&self) -> u64 {
        self.instance_id
    }

    async fn register_internal(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
        let instance = spec.with_instance_id(self.instance_id);

        self.registry
            .instances
            .lock()
            .unwrap()
            .push(instance.clone());

        Ok(instance)
    }

    async fn unregister(&self, instance: DiscoveryInstance) -> Result<()> {
        let target_id = instance.id();

        self.registry
            .instances
            .lock()
            .unwrap()
            .retain(|i| i.id() != target_id);

        Ok(())
    }

    async fn list(&self, query: DiscoveryQuery) -> Result<Vec<DiscoveryInstance>> {
        let instances = self.registry.instances.lock().unwrap();
        Ok(instances
            .iter()
            .filter(|instance| matches_query(instance, &query))
            .cloned()
            .collect())
    }

    async fn list_and_watch(
        &self,
        query: DiscoveryQuery,
        _cancel_token: Option<CancellationToken>,
    ) -> Result<DiscoveryStream> {
        use std::collections::HashSet;

        let registry = self.registry.clone();

        let stream = async_stream::stream! {
            let mut known_instances: HashSet<DiscoveryInstanceId> = HashSet::new();

            loop {
                let current: Vec<_> = {
                    let instances = registry.instances.lock().unwrap();
                    instances
                        .iter()
                        .filter(|instance| matches_query(instance, &query))
                        .cloned()
                        .collect()
                };

                let current_ids: HashSet<DiscoveryInstanceId> = current.iter().map(|i| i.id()).collect();

                // Emit Added events for new instances
                for instance in current {
                    let id = instance.id();
                    if known_instances.insert(id) {
                        yield Ok(DiscoveryEvent::Added(instance));
                    }
                }

                // Emit Removed events for instances that are gone
                for id in known_instances.difference(&current_ids).cloned().collect::<Vec<_>>() {
                    known_instances.remove(&id);
                    yield Ok(DiscoveryEvent::Removed(id));
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        };

        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn model_spec(
        namespace: &str,
        component: &str,
        endpoint: &str,
        model_name: &str,
    ) -> DiscoverySpec {
        DiscoverySpec::Model {
            namespace: namespace.to_string(),
            component: component.to_string(),
            endpoint: endpoint.to_string(),
            card_json: serde_json::json!({
                "display_name": model_name,
            }),
            model_suffix: None,
        }
    }

    fn lora_model_spec(
        namespace: &str,
        component: &str,
        endpoint: &str,
        model_name: &str,
        source_path: &str,
        lora_name: &str,
    ) -> DiscoverySpec {
        DiscoverySpec::Model {
            namespace: namespace.to_string(),
            component: component.to_string(),
            endpoint: endpoint.to_string(),
            card_json: serde_json::json!({
                "display_name": model_name,
                "source_path": source_path,
                "lora": {
                    "name": lora_name,
                },
            }),
            model_suffix: Some(lora_name.to_string()),
        }
    }

    #[tokio::test]
    async fn test_mock_discovery_add_and_remove() {
        let registry = SharedMockRegistry::new();
        let client1 = MockDiscovery::new(Some(1), registry.clone());
        let client2 = MockDiscovery::new(Some(2), registry.clone());

        let spec = DiscoverySpec::Endpoint {
            namespace: "test-ns".to_string(),
            component: "test-comp".to_string(),
            endpoint: "test-ep".to_string(),
            transport: crate::component::TransportType::Nats("test-subject".to_string()),
        };

        let query = DiscoveryQuery::Endpoint {
            namespace: "test-ns".to_string(),
            component: "test-comp".to_string(),
            endpoint: "test-ep".to_string(),
        };

        // Start watching
        let mut stream = client1.list_and_watch(query.clone(), None).await.unwrap();

        // Add first instance
        let instance1 = client1.register(spec.clone()).await.unwrap();

        let event = stream.next().await.unwrap().unwrap();
        match event {
            DiscoveryEvent::Added(DiscoveryInstance::Endpoint(inst)) => {
                assert_eq!(inst.instance_id, 1);
            }
            _ => panic!("Expected Added event for instance-1"),
        }

        // Add second instance
        client2.register(spec.clone()).await.unwrap();

        let event = stream.next().await.unwrap().unwrap();
        match event {
            DiscoveryEvent::Added(DiscoveryInstance::Endpoint(inst)) => {
                assert_eq!(inst.instance_id, 2);
            }
            _ => panic!("Expected Added event for instance-2"),
        }

        // Remove first instance
        client1.unregister(instance1).await.unwrap();

        let event = stream.next().await.unwrap().unwrap();
        match event {
            DiscoveryEvent::Removed(id) => {
                let endpoint_id = id.extract_endpoint_id().expect("Expected endpoint removal");
                assert_eq!(endpoint_id.instance_id, 1);
            }
            _ => panic!("Expected Removed event for instance-1"),
        }
    }

    #[tokio::test]
    async fn register_allows_same_model_name_on_same_endpoint() {
        let registry = SharedMockRegistry::new();
        let discovery1 = MockDiscovery::new(Some(1), registry.clone());
        let discovery2 = MockDiscovery::new(Some(2), registry);
        let spec = model_spec("ns", "comp", "generate", "model-a");

        discovery1.register(spec.clone()).await.unwrap();
        discovery2.register(spec).await.unwrap();

        let instances = discovery1
            .list(DiscoveryQuery::EndpointModels {
                namespace: "ns".to_string(),
                component: "comp".to_string(),
                endpoint: "generate".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(instances.len(), 2);
    }

    #[tokio::test]
    async fn register_rejects_different_model_name_on_same_endpoint() {
        let registry = SharedMockRegistry::new();
        let discovery1 = MockDiscovery::new(Some(1), registry.clone());
        let discovery2 = MockDiscovery::new(Some(2), registry);

        discovery1
            .register(model_spec("ns", "comp", "generate", "model-a"))
            .await
            .unwrap();

        let err = discovery2
            .register(model_spec("ns", "comp", "generate", "model-b"))
            .await
            .unwrap_err();

        assert!(err.to_string().contains(
            "Cannot register model 'model-b' on endpoint 'ns/comp/generate': a different model 'model-a' is already registered there"
        ));

        let instances = discovery1
            .list(DiscoveryQuery::EndpointModels {
                namespace: "ns".to_string(),
                component: "comp".to_string(),
                endpoint: "generate".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(instances.len(), 1);
    }

    #[tokio::test]
    async fn register_allows_different_model_names_on_different_endpoints() {
        let registry = SharedMockRegistry::new();
        let discovery1 = MockDiscovery::new(Some(1), registry.clone());
        let discovery2 = MockDiscovery::new(Some(2), registry);

        discovery1
            .register(model_spec("ns", "comp", "generate-a", "model-a"))
            .await
            .unwrap();
        discovery2
            .register(model_spec("ns", "comp", "generate-b", "model-b"))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn register_allows_lora_adapter_on_same_endpoint() {
        let registry = SharedMockRegistry::new();
        let discovery1 = MockDiscovery::new(Some(1), registry.clone());
        let discovery2 = MockDiscovery::new(Some(2), registry);

        discovery1
            .register(DiscoverySpec::Model {
                namespace: "ns".to_string(),
                component: "comp".to_string(),
                endpoint: "generate".to_string(),
                card_json: serde_json::json!({
                    "display_name": "base-model",
                    "source_path": "base-repo",
                }),
                model_suffix: None,
            })
            .await
            .unwrap();

        discovery2
            .register(lora_model_spec(
                "ns",
                "comp",
                "generate",
                "adapter-a",
                "base-repo",
                "adapter-a",
            ))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn register_rejects_lora_adapter_for_different_base_model() {
        let registry = SharedMockRegistry::new();
        let discovery1 = MockDiscovery::new(Some(1), registry.clone());
        let discovery2 = MockDiscovery::new(Some(2), registry);

        discovery1
            .register(DiscoverySpec::Model {
                namespace: "ns".to_string(),
                component: "comp".to_string(),
                endpoint: "generate".to_string(),
                card_json: serde_json::json!({
                    "display_name": "base-model",
                    "source_path": "base-repo",
                }),
                model_suffix: None,
            })
            .await
            .unwrap();

        let err = discovery2
            .register(lora_model_spec(
                "ns",
                "comp",
                "generate",
                "adapter-a",
                "other-base-repo",
                "adapter-a",
            ))
            .await
            .unwrap_err();

        assert!(err.to_string().contains(
            "Cannot register model 'adapter-a' on endpoint 'ns/comp/generate': a different model 'base-model' is already registered there"
        ));
    }
}
