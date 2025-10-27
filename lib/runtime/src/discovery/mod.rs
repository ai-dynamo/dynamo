use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;
use tokio::sync::broadcast;

// Re-export legacy types
mod legacy;
pub use legacy::{DiscoveryClient, Lease};

pub mod kubernetes;

#[derive(Error, Debug)]
pub enum DiscoveryError {
    #[error("instance not found: {0}")]
    InstanceNotFound(String),
    #[error("metadata error: {0}")]
    MetadataError(String),
    #[error("registration error: {0}")]
    RegistrationError(String),
    #[error("watch error: {0}")]
    WatchError(String),
}

pub type Result<T> = std::result::Result<T, DiscoveryError>;

/// Status of an instance
#[derive(Debug, Clone, PartialEq)]
pub enum InstanceStatus {
    Ready,
    NotReady,
}

/// Event emitted when instance status changes
#[derive(Debug, Clone)]
pub enum InstanceEvent {
    Added(Instance),
    Removed(String), // instance_id
}

/// Handle returned when registering a new instance
#[async_trait]
pub trait InstanceHandle: Send + Sync {
    /// Returns the unique identifier for this instance
    fn instance_id(&self) -> &str;

    /// Set metadata associated with this instance
    async fn set_metadata(&self, metadata: Value) -> Result<()>;

    /// Set the ready status of this instance
    async fn set_ready(&self, status: InstanceStatus) -> Result<()>;
}

/// Represents a discovered instance
#[derive(Debug, Clone)]
pub struct Instance {
    pub instance_id: String,
    pub metadata: Value,
}

// add a getter for the metadata
impl Instance {
    pub fn new(instance_id: String, metadata: Value) -> Self {
        Self {
            instance_id,
            metadata,
        }
    }
}

/// Convert a string instance_id to u64, either by parsing or hashing
pub fn instance_id_to_u64(id: &str) -> u64 {
    id.parse::<u64>().unwrap_or_else(|_| {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        id.hash(&mut hasher);
        hasher.finish()
    })
}

/// Main service discovery interface
#[async_trait]
pub trait ServiceDiscovery: Send + Sync + 'static {
    /// Register a new instance
    async fn register_instance(
        &self,
        namespace: &str,
        component: &str,
    ) -> Result<Box<dyn InstanceHandle>>;

    /// List all instances for a namespace/component
    async fn list_instances(&self, namespace: &str, component: &str) -> Result<Vec<Instance>>;

    /// Watch for instance changes
    async fn watch(
        &self,
        namespace: &str,
        component: &str,
    ) -> Result<broadcast::Receiver<InstanceEvent>>;
}
