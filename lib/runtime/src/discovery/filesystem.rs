use super::*;
use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use uuid::Uuid;

/// Filesystem-based implementation of ServiceDiscovery for testing
#[derive(Debug, Clone)]
pub struct FilesystemServiceDiscovery {
    base_path: PathBuf,
    event_tx: Arc<broadcast::Sender<InstanceEvent>>,
    watcher_started: Arc<Mutex<bool>>,
}

impl FilesystemServiceDiscovery {
    pub fn new() -> Self {
        // Create temp directory for discovery files
        let base_path = std::env::temp_dir().join("dynamo_discovery");
        std::fs::create_dir_all(&base_path).ok();
        
        let (tx, _) = broadcast::channel(100);
        
        Self {
            base_path,
            event_tx: Arc::new(tx),
            watcher_started: Arc::new(Mutex::new(false)),
        }
    }
    
    fn instances_dir(&self) -> PathBuf {
        // Hardcode to "test/comp" as requested
        self.base_path.join("test").join("comp")
    }
    
    fn ensure_watcher_started(&self) {
        let mut started = self.watcher_started.lock();
        if *started {
            return;  // Already started
        }
        
        *started = true;
        drop(started);  // Release lock before spawning
        
        self.start_watcher();
    }
    
    fn start_watcher(&self) {
        let instances_dir = self.instances_dir();
        let event_tx = self.event_tx.clone();
        
        // Ensure directory exists
        std::fs::create_dir_all(&instances_dir).ok();
        
        tokio::spawn(async move {
            let mut known_instances: std::collections::HashSet<String> = std::collections::HashSet::new();
            
            loop {
                // Read current files
                let mut current_instances = std::collections::HashSet::new();
                
                if let Ok(entries) = std::fs::read_dir(&instances_dir) {
                    for entry in entries.flatten() {
                        if let Ok(filename) = entry.file_name().into_string() {
                            current_instances.insert(filename);
                        }
                    }
                }
                
                // Find new instances
                for instance_id in current_instances.difference(&known_instances) {
                    let instance = Instance::new(instance_id.clone(), Value::Null);
                    let _ = event_tx.send(InstanceEvent::Added(instance));
                }
                
                // Find removed instances
                for instance_id in known_instances.difference(&current_instances) {
                    let _ = event_tx.send(InstanceEvent::Removed(instance_id.clone()));
                }
                
                known_instances = current_instances;
                
                // Poll every 100ms
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });
    }
}

impl Default for FilesystemServiceDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for a registered instance
#[derive(Debug)]
pub struct FilesystemInstanceHandle {
    instance_id: String,
    instances_dir: PathBuf,
}

impl FilesystemInstanceHandle {
    fn instance_file(&self) -> PathBuf {
        self.instances_dir.join(&self.instance_id)
    }
}

impl Drop for FilesystemInstanceHandle {
    fn drop(&mut self) {
        // Clean up the instance file when handle is dropped
        let _ = std::fs::remove_file(self.instance_file());
    }
}

#[async_trait]
impl InstanceHandle for FilesystemInstanceHandle {
    fn instance_id(&self) -> &str {
        &self.instance_id
    }

    async fn set_metadata(&self, _metadata: Value) -> Result<()> {
        // Metadata is a no-op for this simple implementation
        Ok(())
    }

    async fn set_ready(&self, status: InstanceStatus) -> Result<()> {
        match status {
            InstanceStatus::Ready => {
                // Create file to mark instance as ready
                std::fs::write(self.instance_file(), self.instance_id.as_bytes())
                    .map_err(|e| DiscoveryError::RegistrationError(e.to_string()))?;
            }
            InstanceStatus::NotReady => {
                // Remove file to mark instance as not ready
                let _ = std::fs::remove_file(self.instance_file());
            }
        }
        Ok(())
    }
}

#[async_trait]
impl ServiceDiscovery for FilesystemServiceDiscovery {
    async fn register_instance(
        &self,
        _namespace: &str,
        _component: &str,
    ) -> Result<Box<dyn InstanceHandle>> {
        // Generate unique instance ID
        let instance_id = Uuid::new_v4().to_string();
        
        // Ensure directory exists
        let instances_dir = self.instances_dir();
        std::fs::create_dir_all(&instances_dir)
            .map_err(|e| DiscoveryError::RegistrationError(e.to_string()))?;
        
        Ok(Box::new(FilesystemInstanceHandle {
            instance_id,
            instances_dir,
        }))
    }

    async fn list_instances(
        &self,
        _namespace: &str,
        _component: &str,
    ) -> Result<Vec<Instance>> {
        let instances_dir = self.instances_dir();
        
        let mut instances = Vec::new();
        
        if let Ok(entries) = std::fs::read_dir(&instances_dir) {
            for entry in entries.flatten() {
                if let Ok(filename) = entry.file_name().into_string() {
                    instances.push(Instance::new(filename, Value::Null));
                }
            }
        }
        
        Ok(instances)
    }

    async fn watch(
        &self,
        _namespace: &str,
        _component: &str,
    ) -> Result<broadcast::Receiver<InstanceEvent>> {
        // Start watcher on first watch() call
        self.ensure_watcher_started();
        Ok(self.event_tx.subscribe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_filesystem_discovery() {
        let discovery = FilesystemServiceDiscovery::new();
        
        // Register an instance
        let handle = discovery.register_instance("test", "comp").await.unwrap();
        let instance_id = handle.instance_id().to_string();
        
        // Initially not ready (no file created)
        let instances = discovery.list_instances("test", "comp").await.unwrap();
        assert!(instances.is_empty());
        
        // Mark ready (creates file)
        handle.set_ready(InstanceStatus::Ready).await.unwrap();
        
        // Give watcher time to detect
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        // Should now be listed
        let instances = discovery.list_instances("test", "comp").await.unwrap();
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].instance_id, instance_id);
        
        // Drop handle (cleans up file)
        drop(handle);
        
        // Give watcher time to detect
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        // Should be gone
        let instances = discovery.list_instances("test", "comp").await.unwrap();
        assert!(instances.is_empty());
    }

    #[tokio::test]
    async fn test_filesystem_watch() {
        let discovery = FilesystemServiceDiscovery::new();
        
        // Setup watch
        let mut watch = discovery.watch("test", "comp").await.unwrap();
        
        // Register and make ready
        let handle = discovery.register_instance("test", "comp").await.unwrap();
        let instance_id = handle.instance_id().to_string();
        handle.set_ready(InstanceStatus::Ready).await.unwrap();
        
        // Should get Added event (within reasonable time)
        tokio::time::timeout(Duration::from_secs(1), async {
            match watch.recv().await.unwrap() {
                InstanceEvent::Added(instance) => {
                    assert_eq!(instance.instance_id, instance_id);
                }
                _ => panic!("Expected Added event"),
            }
        }).await.unwrap();
        
        // Mark not ready
        handle.set_ready(InstanceStatus::NotReady).await.unwrap();
        
        // Should get Removed event
        tokio::time::timeout(Duration::from_secs(1), async {
            match watch.recv().await.unwrap() {
                InstanceEvent::Removed(id) => {
                    assert_eq!(id, instance_id);
                }
                _ => panic!("Expected Removed event"),
            }
        }).await.unwrap();
    }
}

