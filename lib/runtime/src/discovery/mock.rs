use super::*;
use async_trait::async_trait;
use parking_lot::RwLock;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;
use uuid::Uuid;

/// Mock implementation of InstanceHandle
#[derive(Debug)]
pub struct MockInstanceHandle {
    instance_id: String,
    discovery: Arc<MockServiceDiscovery>,
    namespace: String,
    component: String,
}

#[async_trait]
impl InstanceHandle for MockInstanceHandle {
    fn instance_id(&self) -> &str {
        &self.instance_id
    }

    async fn set_metadata(&self, metadata: Value) -> Result<()> {
        let mut instances = self.discovery.instances.write();
        if let Some(instance) = instances.get_mut(&(self.namespace.clone(), self.component.clone())) {
            if let Some(inst) = instance.get_mut(&self.instance_id) {
                inst.metadata = metadata;
                Ok(())
            } else {
                Err(DiscoveryError::InstanceNotFound(self.instance_id.clone()))
            }
        } else {
            Err(DiscoveryError::InstanceNotFound(self.instance_id.clone()))
        }
    }

    async fn set_ready(&self, status: InstanceStatus) -> Result<()> {
        let instances = self.discovery.instances.write();
        if let Some(instance) = instances.get(&(self.namespace.clone(), self.component.clone())) {
            if let Some(inst) = instance.get(&self.instance_id) {
                // Clone instance for event
                let instance = Instance::new(inst.instance_id.clone(), inst.metadata.clone());
                
                match status {
                    InstanceStatus::Ready => {
                        // Notify watchers of new ready instance
                        let _ = self.discovery.event_tx.send(InstanceEvent::Added(instance));
                    }
                    InstanceStatus::NotReady => {
                        // Notify watchers instance is gone
                        let _ = self.discovery.event_tx.send(InstanceEvent::Removed(self.instance_id.clone()));
                    }
                }
                Ok(())
            } else {
                Err(DiscoveryError::InstanceNotFound(self.instance_id.clone()))
            }
        } else {
            Err(DiscoveryError::InstanceNotFound(self.instance_id.clone()))
        }
    }
}

/// Mock implementation of ServiceDiscovery
#[derive(Debug)]
pub struct MockServiceDiscovery {
    instances: RwLock<HashMap<(String, String), HashMap<String, Instance>>>,
    event_tx: broadcast::Sender<InstanceEvent>,
}

impl MockServiceDiscovery {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(100);
        Self {
            instances: RwLock::new(HashMap::new()),
            event_tx: tx,
        }
    }
}

impl Default for MockServiceDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for MockServiceDiscovery {
    fn clone(&self) -> Self {
        // Create new instance with empty state
        Self::new()
    }
}

#[async_trait]
impl ServiceDiscovery for MockServiceDiscovery {
    async fn register_instance(
        &self,
        namespace: &str,
        component: &str,
    ) -> Result<Box<dyn InstanceHandle>> {
        let instance_id = Uuid::new_v4().to_string();
        let instance = Instance::new(instance_id.clone(), Value::Null);
        
        let mut instances = self.instances.write();
        let component_instances = instances
            .entry((namespace.to_string(), component.to_string()))
            .or_default();
        component_instances.insert(instance_id.clone(), instance);

        Ok(Box::new(MockInstanceHandle {
            instance_id,
            discovery: Arc::new(Self::new()),
            namespace: namespace.to_string(),
            component: component.to_string(),
        }))
    }

    async fn list_instances(
        &self,
        namespace: &str,
        component: &str,
    ) -> Result<Vec<Instance>> {
        let instances = self.instances.read();
        Ok(instances
            .get(&(namespace.to_string(), component.to_string()))
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default())
    }

    async fn watch(
        &self,
        _namespace: &str,
        _component: &str,
    ) -> Result<broadcast::Receiver<InstanceEvent>> {
        Ok(self.event_tx.subscribe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_register_and_list() {
        let discovery = MockServiceDiscovery::new();
        
        // Register an instance
        let handle = discovery.register_instance("test", "comp").await.unwrap();
        
        // Set some metadata
        handle.set_metadata(json!({
            "transport": {
                "addr": "localhost:8080"
            }
        })).await.unwrap();
        
        // List instances (should be empty until ready)
        let instances = discovery.list_instances("test", "comp").await.unwrap();
        assert!(instances.is_empty());
        
        // Mark ready
        handle.set_ready(InstanceStatus::Ready).await.unwrap();
        
        // List again
        let instances = discovery.list_instances("test", "comp").await.unwrap();
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].instance_id, handle.instance_id());
    }

    #[tokio::test]
    async fn test_watch() {
        let discovery = MockServiceDiscovery::new();
        
        // Setup watch
        let mut watch = discovery.watch("test", "comp").await.unwrap();
        
        // Register and make ready
        let handle = discovery.register_instance("test", "comp").await.unwrap();
        handle.set_ready(InstanceStatus::Ready).await.unwrap();
        
        // Should get Added event
        match watch.recv().await.unwrap() {
            InstanceEvent::Added(instance) => {
                assert_eq!(instance.instance_id, handle.instance_id());
            }
            _ => panic!("Expected Added event"),
        }
        
        // Mark not ready
        handle.set_ready(InstanceStatus::NotReady).await.unwrap();
        
        // Should get Removed event
        match watch.recv().await.unwrap() {
            InstanceEvent::Removed(id) => {
                assert_eq!(id, handle.instance_id());
            }
            _ => panic!("Expected Removed event"),
        }
    }
}
