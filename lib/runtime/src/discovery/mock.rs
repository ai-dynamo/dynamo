// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
    DiscoverySpec, DiscoveryStream, ModelCardInstanceId, model_with_updated_taints,
    reconcile_discovery_snapshot, validate_event_source_reregistration,
    validate_model_reregistration,
};
use anyhow::Result;
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
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
                scope: inst_scope,
                topic: inst_topic,
                ..
            },
            DiscoveryQuery::EventChannels(query),
        ) => {
            query.scope.as_ref().is_none_or(|scope| scope == inst_scope)
                && query.topic.as_ref().is_none_or(|t| t == inst_topic)
        }

        (
            DiscoveryInstance::EventSource {
                scope: inst_scope,
                topic: inst_topic,
                ..
            },
            DiscoveryQuery::EventSources(query),
        ) => {
            query.scope.as_ref().is_none_or(|scope| scope == inst_scope)
                && query.topic.as_ref().is_none_or(|t| t == inst_topic)
        }

        // Cross-type matches return false
        (
            DiscoveryInstance::Endpoint(_),
            DiscoveryQuery::AllModels
            | DiscoveryQuery::NamespacedModels { .. }
            | DiscoveryQuery::ComponentModels { .. }
            | DiscoveryQuery::EndpointModels { .. }
            | DiscoveryQuery::EventChannels(_)
            | DiscoveryQuery::EventSources(_),
        ) => false,
        (
            DiscoveryInstance::Model { .. },
            DiscoveryQuery::AllEndpoints
            | DiscoveryQuery::NamespacedEndpoints { .. }
            | DiscoveryQuery::ComponentEndpoints { .. }
            | DiscoveryQuery::Endpoint { .. }
            | DiscoveryQuery::EventChannels(_)
            | DiscoveryQuery::EventSources(_),
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
        (DiscoveryInstance::EventChannel { .. }, DiscoveryQuery::EventSources(_)) => false,
        (
            DiscoveryInstance::EventSource { .. },
            DiscoveryQuery::AllEndpoints
            | DiscoveryQuery::NamespacedEndpoints { .. }
            | DiscoveryQuery::ComponentEndpoints { .. }
            | DiscoveryQuery::Endpoint { .. }
            | DiscoveryQuery::AllModels
            | DiscoveryQuery::NamespacedModels { .. }
            | DiscoveryQuery::ComponentModels { .. }
            | DiscoveryQuery::EndpointModels { .. }
            | DiscoveryQuery::EventChannels(_),
        ) => false,
    }
}

#[async_trait]
impl Discovery for MockDiscovery {
    fn instance_id(&self) -> u64 {
        self.instance_id
    }

    async fn register_internal(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
        let instance = spec.into_instance(self.instance_id);
        let instance_id = instance.id();
        let mut instances = self.registry.instances.lock().unwrap();
        if let Some(existing) = instances
            .iter_mut()
            .find(|existing| existing.id() == instance_id)
        {
            match &instance {
                DiscoveryInstance::Endpoint(_) => {
                    *existing = instance.clone();
                    return Ok(instance);
                }
                DiscoveryInstance::EventSource { .. } => {
                    validate_event_source_reregistration(existing, &instance)?;
                    return Ok(existing.clone());
                }
                DiscoveryInstance::Model { .. } => {
                    validate_model_reregistration(existing, &instance)?;
                    return Ok(existing.clone());
                }
                DiscoveryInstance::EventChannel { .. } => {}
            }
        }
        instances.push(instance.clone());

        Ok(instance)
    }

    async fn update_model_taints_internal(
        &self,
        id: ModelCardInstanceId,
        taints: HashSet<String>,
    ) -> Result<()> {
        let target_id = DiscoveryInstanceId::Model(id);
        let mut instances = self.registry.instances.lock().unwrap();
        let existing = instances
            .iter_mut()
            .find(|existing| existing.id() == target_id)
            .ok_or_else(|| {
                anyhow::anyhow!("model discovery record {target_id:?} is not registered")
            })?;
        *existing = model_with_updated_taints(existing, taints)?;
        Ok(())
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
        let registry = self.registry.clone();

        let stream = async_stream::stream! {
            let mut known_instances = HashMap::<DiscoveryInstanceId, DiscoveryInstance>::new();

            loop {
                let current: HashMap<DiscoveryInstanceId, DiscoveryInstance> = {
                    let instances = registry.instances.lock().unwrap();
                    instances
                        .iter()
                        .filter(|instance| matches_query(instance, &query))
                        .cloned()
                        .map(|instance| (instance.id(), instance))
                        .collect()
                };

                let (events, reconciled) =
                    reconcile_discovery_snapshot(&known_instances, current);
                for event in events {
                    yield Ok(event);
                }

                known_instances = reconciled;
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        };

        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::TransportType;
    use futures::StreamExt;
    use tokio::time::{Duration, timeout};

    #[tokio::test]
    async fn watch_emits_same_id_endpoint_update() {
        let client = MockDiscovery::new(Some(1), SharedMockRegistry::new());
        let query = DiscoveryQuery::Endpoint {
            namespace: "ns".to_string(),
            component: "component".to_string(),
            endpoint: "endpoint".to_string(),
        };
        let spec = |transport: &str| DiscoverySpec::Endpoint {
            namespace: "ns".to_string(),
            component: "component".to_string(),
            endpoint: "endpoint".to_string(),
            transport: TransportType::Tcp(transport.to_string()),
            device_type: None,
            request_plane_codec: None,
        };
        let mut stream = client.list_and_watch(query.clone(), None).await.unwrap();

        let original = client.register(spec("127.0.0.1:8000")).await.unwrap();
        let event = timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("mock watch should emit the initial instance")
            .unwrap()
            .unwrap();
        assert_eq!(event, DiscoveryEvent::Added(original));

        let updated = client.register(spec("127.0.0.1:9000")).await.unwrap();
        let event = timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("mock watch should emit the updated instance")
            .unwrap()
            .unwrap();

        assert_eq!(event, DiscoveryEvent::Added(updated.clone()));
        assert_eq!(client.list(query).await.unwrap(), vec![updated]);
    }

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
    async fn model_taint_updates_use_the_authoritative_registry() {
        let client = MockDiscovery::new(Some(7), SharedMockRegistry::new());
        let model = client
            .register(DiscoverySpec::Model {
                namespace: "ns".to_string(),
                component: "worker".to_string(),
                endpoint: "generate".to_string(),
                card_json: serde_json::json!({
                    "display_name": "model",
                    "runtime_config": {"taints": ["a"]}
                }),
                model_suffix: None,
            })
            .await
            .unwrap();
        let DiscoveryInstanceId::Model(id) = model.id() else {
            unreachable!()
        };

        client
            .update_model_taints(id.clone(), HashSet::from(["b".to_string()]))
            .await
            .unwrap();
        client
            .update_model_taints(id.clone(), HashSet::from(["a".to_string()]))
            .await
            .unwrap();

        let stored = client
            .list(DiscoveryQuery::EndpointModels {
                namespace: "ns".to_string(),
                component: "worker".to_string(),
                endpoint: "generate".to_string(),
            })
            .await
            .unwrap()
            .pop()
            .unwrap();
        let DiscoveryInstance::Model { card_json, .. } = stored else {
            unreachable!()
        };
        assert_eq!(
            card_json["runtime_config"]["taints"],
            serde_json::json!(["a"])
        );

        client.unregister(model).await.unwrap();
        assert!(
            client
                .update_model_taints(id, HashSet::new())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn same_id_model_registration_preserves_updated_taints_without_duplicates() {
        let client = MockDiscovery::new(Some(7), SharedMockRegistry::new());
        let spec = DiscoverySpec::Model {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            endpoint: "generate".to_string(),
            card_json: serde_json::json!({
                "display_name": "model",
                "runtime_config": {"taints": ["initial"]}
            }),
            model_suffix: None,
        };
        let original = client.register(spec.clone()).await.unwrap();
        let DiscoveryInstanceId::Model(id) = original.id() else {
            unreachable!()
        };
        client
            .update_model_taints(id, HashSet::from(["updated".to_string()]))
            .await
            .unwrap();

        let replayed = client.register(spec).await.unwrap();
        let models = client
            .list(DiscoveryQuery::EndpointModels {
                namespace: "ns".to_string(),
                component: "worker".to_string(),
                endpoint: "generate".to_string(),
            })
            .await
            .unwrap();

        assert_eq!(models, vec![replayed.clone()]);
        let DiscoveryInstance::Model { card_json, .. } = replayed else {
            unreachable!()
        };
        assert_eq!(
            card_json["runtime_config"]["taints"],
            serde_json::json!(["updated"])
        );
    }

    #[tokio::test]
    async fn model_taint_update_rejects_foreign_worker_id() {
        let client = MockDiscovery::new(Some(7), SharedMockRegistry::new());
        let foreign_id = ModelCardInstanceId {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            endpoint: "generate".to_string(),
            instance_id: 8,
            model_suffix: None,
        };

        let error = client
            .update_model_taints(foreign_id, HashSet::new())
            .await
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("this discovery client owns worker 7")
        );
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
            device_type: None,
            request_plane_codec: None,
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
    async fn event_source_removal_is_publisher_specific() {
        use crate::discovery::{EventScope, EventSourceQuery};

        let client = MockDiscovery::new(Some(42), SharedMockRegistry::new());
        let endpoint = crate::protocols::EndpointId {
            namespace: "workers".to_string(),
            component: "backend".to_string(),
            name: "kv-state".to_string(),
        };
        let query = DiscoveryQuery::EventSources(EventSourceQuery::endpoint_topic(
            endpoint.clone(),
            "kv-events",
        ));
        let spec = |publisher_id, worker_id| DiscoverySpec::EventSource {
            scope: EventScope::Endpoint {
                endpoint: endpoint.clone(),
            },
            topic: "kv-events".to_string(),
            publisher_id,
            metadata: serde_json::json!({"worker_id": worker_id, "dp_rank": 0}),
        };

        let old = client.register(spec(100, 7)).await.unwrap();
        assert_eq!(client.register(spec(100, 7)).await.unwrap(), old);
        assert!(client.register(spec(100, 8)).await.is_err());
        assert_eq!(client.list(query.clone()).await.unwrap(), vec![old.clone()]);

        let current = client.register(spec(205, 7)).await.unwrap();
        assert_eq!(client.list(query.clone()).await.unwrap().len(), 2);

        client.unregister(old).await.unwrap();
        assert_eq!(client.list(query).await.unwrap(), vec![current]);
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
    async fn register_rejects_distinct_base_cards_with_same_source_path_on_same_endpoint() {
        let registry = SharedMockRegistry::new();
        let discovery1 = MockDiscovery::new(Some(1), registry.clone());
        let discovery2 = MockDiscovery::new(Some(2), registry);
        let spec = |display_name: &str| DiscoverySpec::Model {
            namespace: "ns".to_string(),
            component: "comp".to_string(),
            endpoint: "generate".to_string(),
            card_json: serde_json::json!({
                "display_name": display_name,
                "source_path": "org/base-model",
            }),
            model_suffix: None,
        };

        discovery1.register(spec("public-name-a")).await.unwrap();
        let err = discovery2
            .register(spec("public-name-b"))
            .await
            .unwrap_err();

        assert!(err.to_string().contains(
            "Cannot register model 'public-name-b' on endpoint 'ns/comp/generate': a different model 'public-name-a' is already registered there"
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
