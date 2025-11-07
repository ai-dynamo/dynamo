// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    DiscoveryClient, DiscoveryEvent, DiscoveryInstance, DiscoveryKey, DiscoverySpec,
    DiscoveryStream,
};
use crate::Result;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

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

/// Mock implementation of DiscoveryClient for testing
/// We can potentially remove this once we have KeyValueDiscoveryClient implemented
pub struct MockDiscoveryClient {
    instance_id: u64,
    registry: SharedMockRegistry,
}

impl MockDiscoveryClient {
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

/// Helper function to check if an instance matches a discovery key query
fn matches_key(instance: &DiscoveryInstance, key: &DiscoveryKey) -> bool {
    match (instance, key) {
        // Endpoint matching
        (DiscoveryInstance::Endpoint(_), DiscoveryKey::AllEndpoints) => true,
        (DiscoveryInstance::Endpoint(inst), DiscoveryKey::NamespacedEndpoints { namespace }) => {
            &inst.namespace == namespace
        }
        (
            DiscoveryInstance::Endpoint(inst),
            DiscoveryKey::ComponentEndpoints {
                namespace,
                component,
            },
        ) => &inst.namespace == namespace && &inst.component == component,
        (
            DiscoveryInstance::Endpoint(inst),
            DiscoveryKey::Endpoint {
                namespace,
                component,
                endpoint,
            },
        ) => {
            &inst.namespace == namespace
                && &inst.component == component
                && &inst.endpoint == endpoint
        }

        // ModelCard matching
        (DiscoveryInstance::ModelCard { .. }, DiscoveryKey::AllModelCards) => true,
        (
            DiscoveryInstance::ModelCard {
                namespace: inst_ns, ..
            },
            DiscoveryKey::NamespacedModelCards { namespace },
        ) => inst_ns == namespace,
        (
            DiscoveryInstance::ModelCard {
                namespace: inst_ns,
                component: inst_comp,
                ..
            },
            DiscoveryKey::ComponentModelCards {
                namespace,
                component,
            },
        ) => inst_ns == namespace && inst_comp == component,
        (
            DiscoveryInstance::ModelCard {
                namespace: inst_ns,
                component: inst_comp,
                endpoint: inst_ep,
                ..
            },
            DiscoveryKey::EndpointModelCards {
                namespace,
                component,
                endpoint,
            },
        ) => inst_ns == namespace && inst_comp == component && inst_ep == endpoint,

        // Cross-type matches return false
        (
            DiscoveryInstance::Endpoint(_),
            DiscoveryKey::AllModelCards
            | DiscoveryKey::NamespacedModelCards { .. }
            | DiscoveryKey::ComponentModelCards { .. }
            | DiscoveryKey::EndpointModelCards { .. },
        ) => false,
        (
            DiscoveryInstance::ModelCard { .. },
            DiscoveryKey::AllEndpoints
            | DiscoveryKey::NamespacedEndpoints { .. }
            | DiscoveryKey::ComponentEndpoints { .. }
            | DiscoveryKey::Endpoint { .. },
        ) => false,
    }
}

#[async_trait]
impl DiscoveryClient for MockDiscoveryClient {
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

    async fn list(&self, key: DiscoveryKey) -> Result<Vec<DiscoveryInstance>> {
        let instances = self.registry.instances.lock().unwrap();
        Ok(instances
            .iter()
            .filter(|instance| matches_key(instance, &key))
            .cloned()
            .collect())
    }

    async fn list_and_watch(&self, key: DiscoveryKey) -> Result<DiscoveryStream> {
        use std::collections::HashSet;

        let registry = self.registry.clone();

        let stream = async_stream::stream! {
            let mut known_instances = HashSet::new();

            loop {
                let current: Vec<_> = {
                    let instances = registry.instances.lock().unwrap();
                    instances
                        .iter()
                        .filter(|instance| matches_key(instance, &key))
                        .cloned()
                        .collect()
                };

                let current_ids: HashSet<_> = current.iter().map(|i| {
                    match i {
                        DiscoveryInstance::Endpoint(inst) => inst.instance_id,
                        DiscoveryInstance::ModelCard { instance_id, .. } => *instance_id,
                    }
                }).collect();

                // Emit Added events for new instances
                for instance in current {
                    let id = match &instance {
                        DiscoveryInstance::Endpoint(inst) => inst.instance_id,
                        DiscoveryInstance::ModelCard { instance_id, .. } => *instance_id,
                    };
                    if known_instances.insert(id) {
                        yield Ok(DiscoveryEvent::Added(instance));
                    }
                }

                // Emit Removed events for instances that are gone
                for id in known_instances.difference(&current_ids).cloned().collect::<Vec<_>>() {
                    yield Ok(DiscoveryEvent::Removed(id));
                    known_instances.remove(&id);
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
        let client1 = MockDiscoveryClient::new(Some(1), registry.clone());
        let client2 = MockDiscoveryClient::new(Some(2), registry.clone());

        let spec = DiscoverySpec::Endpoint {
            namespace: "test-ns".to_string(),
            component: "test-comp".to_string(),
            endpoint: "test-ep".to_string(),
            transport: crate::component::TransportType::NatsTcp("test-subject".to_string()),
        };

        let key = DiscoveryKey::Endpoint {
            namespace: "test-ns".to_string(),
            component: "test-comp".to_string(),
            endpoint: "test-ep".to_string(),
        };

        // Start watching
        let mut stream = client1.list_and_watch(key.clone()).await.unwrap();

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
            DiscoveryInstance::ModelCard { instance_id, .. } => *instance_id != 1,
        });

        let event = stream.next().await.unwrap().unwrap();
        match event {
            DiscoveryEvent::Removed(instance_id) => {
                assert_eq!(instance_id, 1);
            }
            _ => panic!("Expected Removed event for instance-1"),
        }
    }
}
