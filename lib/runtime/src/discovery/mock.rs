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

        // Cross-type matches return false
        (
            DiscoveryInstance::Endpoint(_),
            DiscoveryQuery::AllModels
            | DiscoveryQuery::NamespacedModels { .. }
            | DiscoveryQuery::ComponentModels { .. }
            | DiscoveryQuery::EndpointModels { .. },
        ) => false,
        (
            DiscoveryInstance::Model { .. },
            DiscoveryQuery::AllEndpoints
            | DiscoveryQuery::NamespacedEndpoints { .. }
            | DiscoveryQuery::ComponentEndpoints { .. }
            | DiscoveryQuery::Endpoint { .. },
        ) => false,
    }
}

#[async_trait]
impl Discovery for MockDiscovery {
    fn instance_id(&self) -> u64 {
        self.instance_id
    }

    async fn register(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
        let instance = spec.with_instance_id(self.instance_id);

        self.registry
            .instances
            .lock()
            .unwrap()
            .push(instance.clone());

        Ok(instance)
    }

    async fn unregister(&self, instance: DiscoveryInstance) -> Result<()> {
        let instance_id = instance.instance_id();

        self.registry
            .instances
            .lock()
            .unwrap()
            .retain(|i| i.instance_id() != instance_id);

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
        client1.register(spec.clone()).await.unwrap();

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
        registry.instances.lock().unwrap().retain(|i| match i {
            DiscoveryInstance::Endpoint(inst) => inst.instance_id != 1,
            DiscoveryInstance::Model { instance_id, .. } => *instance_id != 1,
        });

        let event = stream.next().await.unwrap().unwrap();
        match event {
            DiscoveryEvent::Removed(id) => {
                let endpoint_id = id.extract_endpoint_id().expect("Expected endpoint removal");
                assert_eq!(endpoint_id.instance_id, 1);
            }
            _ => panic!("Expected Removed event for instance-1"),
        }
    }
}
