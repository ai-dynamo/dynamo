// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{DiscoveryClient, DiscoveryEvent, DiscoveryInstance, DiscoveryKey, DiscoveryStream};
use crate::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Shared in-memory registry for mock discovery
#[derive(Clone, Default)]
pub struct SharedMockRegistry {
    instances: Arc<Mutex<HashMap<DiscoveryKey, Vec<DiscoveryInstance>>>>,
}

impl SharedMockRegistry {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Mock implementation of DiscoveryClient for testing
/// We can potentially remove this once we have KeyValueDiscoveryClient implemented
pub struct MockDiscoveryClient {
    instance_id: String,
    registry: SharedMockRegistry,
}

impl MockDiscoveryClient {
    pub fn new(instance_id: impl Into<String>, registry: SharedMockRegistry) -> Self {
        Self {
            instance_id: instance_id.into(),
            registry,
        }
    }
}

#[async_trait]
impl DiscoveryClient for MockDiscoveryClient {
    fn instance_id(&self) -> String {
        self.instance_id.clone()
    }

    async fn serve(&self, key: DiscoveryKey) -> Result<DiscoveryInstance> {
        let instance = match &key {
            DiscoveryKey::Endpoint {
                namespace,
                component,
                endpoint,
            } => DiscoveryInstance::Endpoint {
                namespace: namespace.clone(),
                component: component.clone(),
                endpoint: endpoint.clone(),
                instance_id: self.instance_id.clone(),
            },
        };

        self.registry
            .instances
            .lock()
            .unwrap()
            .entry(key)
            .or_default()
            .push(instance.clone());

        Ok(instance)
    }

    async fn list_and_watch(&self, key: DiscoveryKey) -> Result<DiscoveryStream> {
        use std::collections::HashSet;

        let registry = self.registry.clone();

        let stream = async_stream::stream! {
            let mut known_instances = HashSet::new();

            loop {
                let current: Vec<_> = {
                    let instances = registry.instances.lock().unwrap();
                    instances.get(&key).cloned().unwrap_or_default()
                };

                let current_ids: HashSet<_> = current.iter().map(|i| {
                    match i {
                        DiscoveryInstance::Endpoint { instance_id, .. } => instance_id.clone(),
                    }
                }).collect();

                // Emit Added events for new instances
                for instance in current {
                    let id = match &instance {
                        DiscoveryInstance::Endpoint { instance_id, .. } => instance_id.clone(),
                    };
                    if known_instances.insert(id) {
                        yield Ok(DiscoveryEvent::Added(instance));
                    }
                }

                // Emit Removed events for instances that are gone
                for id in known_instances.difference(&current_ids).cloned().collect::<Vec<_>>() {
                    yield Ok(DiscoveryEvent::Removed(id.clone()));
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
        let client1 = MockDiscoveryClient::new("instance-1", registry.clone());
        let client2 = MockDiscoveryClient::new("instance-2", registry.clone());

        let key = DiscoveryKey::Endpoint {
            namespace: "test-ns".to_string(),
            component: "test-comp".to_string(),
            endpoint: "test-ep".to_string(),
        };

        // Start watching
        let mut stream = client1.list_and_watch(key.clone()).await.unwrap();

        // Add first instance
        client1.serve(key.clone()).await.unwrap();

        let event = stream.next().await.unwrap().unwrap();
        match event {
            DiscoveryEvent::Added(DiscoveryInstance::Endpoint { instance_id, .. }) => {
                assert_eq!(instance_id, "instance-1");
            }
            _ => panic!("Expected Added event for instance-1"),
        }

        // Add second instance
        client2.serve(key.clone()).await.unwrap();

        let event = stream.next().await.unwrap().unwrap();
        match event {
            DiscoveryEvent::Added(DiscoveryInstance::Endpoint { instance_id, .. }) => {
                assert_eq!(instance_id, "instance-2");
            }
            _ => panic!("Expected Added event for instance-2"),
        }

        // Remove first instance
        registry
            .instances
            .lock()
            .unwrap()
            .get_mut(&key)
            .unwrap()
            .retain(|i| match i {
                DiscoveryInstance::Endpoint { instance_id, .. } => instance_id != "instance-1",
            });

        let event = stream.next().await.unwrap().unwrap();
        match event {
            DiscoveryEvent::Removed(instance_id) => {
                assert_eq!(instance_id, "instance-1");
            }
            _ => panic!("Expected Removed event for instance-1"),
        }
    }
}
