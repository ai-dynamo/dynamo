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
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

/// Shared in-memory registry for mock discovery
#[derive(Clone, Default)]
pub struct SharedMockRegistry {
    state: Arc<Mutex<RegistryState>>,
}

impl SharedMockRegistry {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Default)]
struct RegistryState {
    instances: Vec<DiscoveryInstance>,
    /// Each watch receives the registry after every change, so a change made between two reads
    /// of its stream still reaches it.
    watchers: Vec<UnboundedSender<Arc<Vec<DiscoveryInstance>>>>,
}

impl RegistryState {
    fn publish(&mut self) {
        if self.watchers.is_empty() {
            return;
        }
        let instances = Arc::new(self.instances.clone());
        self.watchers
            .retain(|watcher| watcher.send(instances.clone()).is_ok());
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

fn query_snapshot(
    instances: &[DiscoveryInstance],
    query: &DiscoveryQuery,
) -> HashMap<DiscoveryInstanceId, DiscoveryInstance> {
    instances
        .iter()
        .filter(|instance| matches_query(instance, query))
        .cloned()
        .map(|instance| (instance.id(), instance))
        .collect()
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
        let mut state = self.registry.state.lock().unwrap();
        if let Some(existing) = state
            .instances
            .iter_mut()
            .find(|existing| existing.id() == instance_id)
        {
            match &instance {
                DiscoveryInstance::Endpoint(_) => {
                    *existing = instance.clone();
                    state.publish();
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
        state.instances.push(instance.clone());
        state.publish();

        Ok(instance)
    }

    async fn update_model_taints_internal(
        &self,
        id: ModelCardInstanceId,
        taints: HashSet<String>,
    ) -> Result<()> {
        let target_id = DiscoveryInstanceId::Model(id);
        let mut state = self.registry.state.lock().unwrap();
        let existing = state
            .instances
            .iter_mut()
            .find(|existing| existing.id() == target_id)
            .ok_or_else(|| {
                anyhow::anyhow!("model discovery record {target_id:?} is not registered")
            })?;
        *existing = model_with_updated_taints(existing, taints)?;
        state.publish();
        Ok(())
    }

    async fn unregister(&self, instance: DiscoveryInstance) -> Result<()> {
        let target_id = instance.id();

        let mut state = self.registry.state.lock().unwrap();
        state.instances.retain(|i| i.id() != target_id);
        state.publish();

        Ok(())
    }

    async fn list(&self, query: DiscoveryQuery) -> Result<Vec<DiscoveryInstance>> {
        let state = self.registry.state.lock().unwrap();
        Ok(state
            .instances
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
        let (changes_tx, mut changes) = tokio::sync::mpsc::unbounded_channel();
        // Every change publishes under this lock, so none falls between the snapshot and the
        // subscription.
        let initial = {
            let mut state = self.registry.state.lock().unwrap();
            state.watchers.push(changes_tx);
            query_snapshot(&state.instances, &query)
        };

        let stream = async_stream::stream! {
            let (events, mut known_instances) =
                reconcile_discovery_snapshot(&HashMap::new(), initial);
            for event in events {
                yield Ok(event);
            }

            while let Some(instances) = changes.recv().await {
                let (events, reconciled) = reconcile_discovery_snapshot(
                    &known_instances,
                    query_snapshot(&instances, &query),
                );
                for event in events {
                    yield Ok(event);
                }
                known_instances = reconciled;
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

    #[tokio::test]
    async fn watch_reports_an_unregister_that_follows_establishment() {
        let client = MockDiscovery::new(Some(1), SharedMockRegistry::new());
        let instance = client
            .register(DiscoverySpec::Endpoint {
                namespace: "ns".to_string(),
                component: "component".to_string(),
                endpoint: "endpoint".to_string(),
                transport: TransportType::Tcp("127.0.0.1:8000".to_string()),
                device_type: None,
                request_plane_codec: None,
            })
            .await
            .unwrap();

        let mut stream = client
            .list_and_watch(DiscoveryQuery::AllEndpoints, None)
            .await
            .unwrap();
        client.unregister(instance.clone()).await.unwrap();

        let added = timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("the first snapshot must reach the stream")
            .unwrap()
            .unwrap();
        assert_eq!(added, DiscoveryEvent::Added(instance.clone()));

        let removed = timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("an unregister after establishment must reach the stream")
            .unwrap()
            .unwrap();
        assert_eq!(removed, DiscoveryEvent::Removed(instance.id()));
    }

    #[tokio::test]
    async fn watch_reports_the_removal_of_an_instance_listed_after_establishment() {
        let client = MockDiscovery::new(Some(1), SharedMockRegistry::new());
        let mut stream = client
            .list_and_watch(DiscoveryQuery::AllEndpoints, None)
            .await
            .unwrap();

        let instance = client
            .register(DiscoverySpec::Endpoint {
                namespace: "ns".to_string(),
                component: "component".to_string(),
                endpoint: "endpoint".to_string(),
                transport: TransportType::Tcp("127.0.0.1:8000".to_string()),
                device_type: None,
                request_plane_codec: None,
            })
            .await
            .unwrap();
        let listed = client.list(DiscoveryQuery::AllEndpoints).await.unwrap();
        assert_eq!(listed, vec![instance.clone()]);
        client.unregister(instance.clone()).await.unwrap();

        let added = timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("a registration after establishment must reach the stream")
            .unwrap()
            .unwrap();
        assert_eq!(added, DiscoveryEvent::Added(instance.clone()));

        let removed = timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("a caller that listed the instance must see its removal")
            .unwrap()
            .unwrap();
        assert_eq!(removed, DiscoveryEvent::Removed(instance.id()));
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
    async fn register_non_lora_alias_compatibility() {
        for (name, source_a, source_b, compatible) in [
            ("alias-b", Some("org/base"), Some("org/base"), true),
            ("alias-a", Some("/mount/a"), Some("/mount/b"), true),
            ("alias-b", Some("org/base"), Some("org/other"), false),
            ("alias-b", Some("org/base"), None, false),
            ("alias-b", None, Some("org/base"), false),
            ("alias-b", Some(""), Some(""), false),
        ] {
            let registry = SharedMockRegistry::new();
            let discovery1 = MockDiscovery::new(Some(1), registry.clone());
            let discovery2 = MockDiscovery::new(Some(2), registry);
            let spec = |display_name: &str, source_path: Option<&str>| DiscoverySpec::Model {
                namespace: "ns".to_string(),
                component: "comp".to_string(),
                endpoint: "generate".to_string(),
                card_json: serde_json::json!({
                    "display_name": display_name,
                    "source_path": source_path,
                }),
                model_suffix: None,
            };
            discovery1
                .register(spec("alias-a", source_a))
                .await
                .unwrap();
            let result = discovery2.register(spec(name, source_b)).await;
            assert_eq!(
                result.is_ok(),
                compatible,
                "{name}: {source_a:?}, {source_b:?}: {result:?}"
            );
            if let Err(err) = result {
                assert!(
                    err.to_string()
                        .contains("a different model 'alias-a' is already registered there")
                );
            }
            let instances = discovery1
                .list(DiscoveryQuery::EndpointModels {
                    namespace: "ns".to_string(),
                    component: "comp".to_string(),
                    endpoint: "generate".to_string(),
                })
                .await
                .unwrap();
            assert_eq!(instances.len(), if compatible { 2 } else { 1 });
        }
    }

    #[tokio::test]
    async fn register_shared_source_requires_disjoint_served_names() {
        for (name_b, aliases_a, aliases_b, compatible) in [
            ("b", vec!["b"], vec![], false),
            ("b", vec![], vec!["a"], false),
            ("b", vec!["shared"], vec!["shared"], false),
            ("b", vec!["a", "extra-a"], vec!["b", "extra-b"], true),
            ("a", vec!["shared"], vec!["shared"], true),
        ] {
            let registry = SharedMockRegistry::new();
            let first = MockDiscovery::new(Some(1), registry.clone());
            let second = MockDiscovery::new(Some(2), registry);
            let spec = |name: &str, aliases: Vec<&str>| DiscoverySpec::Model {
                namespace: "ns".into(),
                component: "comp".into(),
                endpoint: "generate".into(),
                card_json: serde_json::json!({
                    "display_name": name,
                    "aliases": aliases,
                    "source_path": "org/base",
                }),
                model_suffix: None,
            };
            let incumbent = first.register(spec("a", aliases_a)).await.unwrap();
            let result = second.register(spec(name_b, aliases_b)).await;
            assert_eq!(result.is_ok(), compatible, "{result:?}");
            let instances = first
                .list(DiscoveryQuery::EndpointModels {
                    namespace: "ns".into(),
                    component: "comp".into(),
                    endpoint: "generate".into(),
                })
                .await
                .unwrap();
            assert!(instances.contains(&incumbent));
            assert_eq!(instances.len(), if compatible { 2 } else { 1 });
        }
    }

    #[tokio::test]
    async fn register_checks_every_existing_model() {
        // B is compatible with A by name and C by source, but A and C conflict.
        for order in [[0, 1, 2], [2, 1, 0]] {
            let registry = SharedMockRegistry::new();
            let cards = [("a", "/mount/a"), ("a", "/mount/b"), ("b", "/mount/b")];
            let mut accepted = Vec::new();
            for (position, index) in order.into_iter().enumerate() {
                let discovery = MockDiscovery::new(Some(index as u64 + 1), registry.clone());
                let (name, source) = cards[index];
                let result = discovery
                    .register(DiscoverySpec::Model {
                        namespace: "ns".into(),
                        component: "comp".into(),
                        endpoint: "generate".into(),
                        card_json: serde_json::json!({"display_name": name, "source_path": source}),
                        model_suffix: None,
                    })
                    .await;
                if position < 2 {
                    accepted.push(result.unwrap());
                } else {
                    assert!(result.is_err());
                    let instances = discovery
                        .list(DiscoveryQuery::EndpointModels {
                            namespace: "ns".into(),
                            component: "comp".into(),
                            endpoint: "generate".into(),
                        })
                        .await
                        .unwrap();
                    assert_eq!(instances.len(), accepted.len());
                    assert!(accepted.iter().all(|instance| instances.contains(instance)));
                }
            }
        }
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
    async fn register_base_and_lora_require_distinct_served_names() {
        for (adapter_name, compatible) in [("base", false), ("alias", false), ("adapter", true)] {
            for adapter_first in [false, true] {
                let registry = SharedMockRegistry::new();
                let first = MockDiscovery::new(Some(1), registry.clone());
                let second = MockDiscovery::new(Some(2), registry);
                let base = DiscoverySpec::Model {
                    namespace: "ns".into(),
                    component: "comp".into(),
                    endpoint: "generate".into(),
                    card_json: serde_json::json!({
                        "display_name": "base",
                        "aliases": ["alias"],
                        "source_path": "org/base",
                    }),
                    model_suffix: None,
                };
                let adapter = lora_model_spec(
                    "ns",
                    "comp",
                    "generate",
                    adapter_name,
                    "org/base",
                    adapter_name,
                );
                let (incumbent, newcomer) = if adapter_first {
                    (adapter, base)
                } else {
                    (base, adapter)
                };
                let incumbent = first.register(incumbent).await.unwrap();
                let result = second.register(newcomer).await;
                assert_eq!(
                    result.is_ok(),
                    compatible,
                    "{adapter_name}, adapter_first={adapter_first}: {result:?}"
                );
                let instances = first
                    .list(DiscoveryQuery::EndpointModels {
                        namespace: "ns".into(),
                        component: "comp".into(),
                        endpoint: "generate".into(),
                    })
                    .await
                    .unwrap();
                assert!(instances.contains(&incumbent));
                assert_eq!(instances.len(), if compatible { 2 } else { 1 });
            }
        }
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
