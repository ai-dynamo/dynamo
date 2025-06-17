// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operational entities that provide distributed runtime capabilities for descriptors.
//!
//! This module implements the entity layer of Dynamo's two-tier architecture for
//! distributed component management. Entities wrap descriptors with a `DistributedRuntime`
//! handle to enable actual etcd operations and distributed coordination.
//!
//! # Architecture Overview
//!
//! The entity system mirrors the descriptor hierarchy with operational types:
//!
//! - [`Namespace`]: Operational namespace with discovery and hierarchical navigation
//! - [`Component`]: Operational component that can host endpoints and store data
//! - [`Endpoint`]: Operational endpoint with automatic instance ID management
//! - [`Path`]: Operational extended path for arbitrary data storage
//!
//! All entities:
//! - Embed a `DistributedRuntime` for etcd operations
//! - Implement `DiscoveryClient` for standardized storage access
//! - Provide navigation methods for traversing the hierarchy
//! - Can be created from descriptors via the `ToEntity` trait
//!
//! # Key Design Principles
//!
//! 1. **Separation of Concerns**: Descriptors handle data representation, entities handle operations
//! 2. **Immutable Descriptors**: Entities wrap immutable descriptors, never modify them
//! 3. **Factory Pattern**: Descriptors convert to entities via `ToEntity::to_entity()`
//! 4. **Navigation**: Entities provide methods to traverse the component hierarchy
//!
//! # Usage Examples
//!
//! ```ignore
//! use dynamo::runtime::{DistributedRuntime, EntityChain};
//! use dynamo::runtime::descriptor::Identifier;
//! use dynamo::runtime::entity::{ToEntity, DiscoveryClient};
//!
//! // Create entities using the fluent API
//! let ns = drt.namespace("production")?;
//! let comp = ns.component("gateway")?;
//! let ep = comp.endpoint("http")?;
//! let path = ep.path(&["v1", "config"])?;
//!
//! // Convert descriptors to entities
//! let id = Identifier::new_component("prod", "api")?;
//! let entity = id.to_entity(drt.clone())?;
//!
//! // Use DiscoveryClient for etcd operations
//! let storage = comp.storage()?;
//! storage.put(b"config_data".to_vec(), None).await?;
//! let values = storage.get().await?;
//! ```
//!
//! # Navigation and Chaining
//!
//! Entities support fluent navigation through the component hierarchy:
//!
//! ```ignore
//! // Start from runtime and chain down
//! let endpoint = drt.namespace("prod")?
//!     .component("api")?
//!     .endpoint("grpc")?;
//!
//! // Navigate from endpoint to extended paths
//! let metrics = endpoint.path(&["metrics", "cpu"])?;
//!
//! // Navigate up with parent() methods
//! let parent_path = metrics.parent()?;  // Returns Path for "metrics"
//! let parent_ns = ns.parent()?;         // Returns parent namespace
//! ```
//!
//! # Integration with DiscoveryClient
//!
//! All entities implement `DiscoveryClient`, providing standardized etcd operations:
//!
//! ```ignore
//! // Every entity can access its storage
//! let storage = entity.storage()?;
//!
//! // Perform etcd operations scoped to the entity's path
//! storage.create(data, lease_id).await?;  // Atomic create
//! storage.put(data, lease_id).await?;     // Create or update
//! storage.get().await?;                    // Retrieve
//! storage.delete(None).await?;            // Delete
//! storage.watch_prefix().await?;          // Watch for changes
//! ```

use crate::{
    descriptor::{
        Identifier, Instance, Keys, KeysBase, DescriptorError,
        BARRIER_KEYWORD, COMPONENT_KEYWORD, ENDPOINT_KEYWORD, PATH_KEYWORD,
    },
    DistributedRuntime,
};
use crate::traits::{DistributedRuntimeProvider, RuntimeProvider};
use std::fmt;
use crate::discovery::DiscoveryClient;

#[derive(Debug, thiserror::Error)]
pub enum EntityError {
    #[error("Invalid descriptor: {0}")]
    InvalidDescriptor(&'static str),
    #[error("Descriptor error: {0}")]
    DescriptorError(#[from] DescriptorError),
}

/// Factory trait for creating entities from descriptors
pub trait ToEntity {
    type Entity;

    fn to_entity(self, runtime: DistributedRuntime) -> Result<Self::Entity, EntityError>;
}

/// Operational namespace with distributed runtime
#[derive(Clone)]
pub struct Namespace {
    descriptor: Identifier,  // Always namespace-only
    runtime: DistributedRuntime,
}

impl Namespace {
    pub fn from_descriptor(descriptor: Identifier, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        Ok(Self { descriptor, runtime })
    }

    pub fn new(namespace: &str, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        let descriptor = Identifier::new_namespace(namespace)?;
        Self::from_descriptor(descriptor, runtime)
    }

    pub fn to_descriptor(&self) -> Identifier {
        self.descriptor.clone()
    }

    pub fn segments(&self) -> Vec<&str> {
        self.descriptor.namespace_name().split('.').collect()
    }

    /// Get parent namespace if not root
    pub fn parent(&self) -> Option<Namespace> {
        let segments = self.segments();
        if segments.len() <= 1 {
            return None;
        }

        let parent_path = segments[..segments.len()-1].join(".");
        Namespace::new(&parent_path, self.runtime.clone()).ok()
    }

    /// Create child namespace
    pub fn child(&self, name: &str) -> Result<Namespace, EntityError> {
        let child_path = format!("{}.{}", self.descriptor.namespace_name(), name);
        Namespace::new(&child_path, self.runtime.clone())
    }

    /// Chain to create a component
    pub fn component(&self, name: &str) -> Result<Component, EntityError> {
        Component::new(self.descriptor.namespace_name(), name, self.runtime.clone())
    }

    /// Chain to create a path
    pub fn path(&self, segments: &[&str]) -> Result<Path, EntityError> {
        let keys = Keys::from_identifier(
            self.descriptor.clone(),
            segments.iter().map(|s| s.to_string()).collect()
        )?;
        Path::from_descriptor(keys, self.runtime.clone())
    }
}

impl DistributedRuntimeProvider for Namespace {
    fn drt(&self) -> &DistributedRuntime {
        &self.runtime
    }
}

impl RuntimeProvider for Namespace {
    fn rt(&self) -> &crate::Runtime {
        self.runtime.rt()
    }
}

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.descriptor)
    }
}

impl DiscoveryClient for Namespace {
    fn etcd_key(&self) -> String {
        self.to_string()
    }
}

/// Operational component with distributed runtime
#[derive(Clone)]
pub struct Component {
    descriptor: Identifier,  // Must have component
    runtime: DistributedRuntime,
}

impl Component {
    pub fn from_descriptor(descriptor: Identifier, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        if descriptor.component_name().is_none() {
            return Err(EntityError::InvalidDescriptor("Descriptor must have component"));
        }
        Ok(Self { descriptor, runtime })
    }

    pub fn new(namespace: &str, component: &str, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        let descriptor = Identifier::new_component(namespace, component)?;
        Self::from_descriptor(descriptor, runtime)
    }

    pub fn to_descriptor(&self) -> Identifier {
        self.descriptor.clone()
    }

    /// Chain to create an endpoint
    pub fn endpoint(&self, name: &str) -> Result<Endpoint, EntityError> {
        Endpoint::new(
            self.descriptor.namespace_name(),
            self.descriptor.component_name().unwrap(),
            name,
            self.runtime.clone()
        )
    }

    /// Chain to create a path
    pub fn path(&self, segments: &[&str]) -> Result<Path, EntityError> {
        let keys = Keys::from_identifier(
            self.descriptor.clone(),
            segments.iter().map(|s| s.to_string()).collect()
        )?;
        Path::from_descriptor(keys, self.runtime.clone())
    }
}

impl DistributedRuntimeProvider for Component {
    fn drt(&self) -> &DistributedRuntime {
        &self.runtime
    }
}

impl RuntimeProvider for Component {
    fn rt(&self) -> &crate::Runtime {
        self.runtime.rt()
    }
}

impl fmt::Display for Component {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.descriptor)
    }
}

impl DiscoveryClient for Component {
    fn etcd_key(&self) -> String {
        self.to_string()
    }
}

/// Operational endpoint with distributed runtime
#[derive(Clone)]
pub struct Endpoint {
    descriptor: Instance,
    runtime: DistributedRuntime,
}

impl Endpoint {
    pub fn from_identifier(descriptor: Identifier, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        if descriptor.endpoint_name().is_none() {
            return Err(EntityError::InvalidDescriptor("Descriptor must have endpoint"));
        }
        let instance = if let Some(lease) = runtime.primary_lease() {
            Instance::new(descriptor, lease.id())?
        } else {
            Instance::new_static(descriptor)?
        };
        Ok(Self {
            descriptor: instance,
            runtime
        })
    }

    pub fn from_instance(instance: Instance, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        Ok(Self {
            descriptor: instance,
            runtime,
        })
    }

    pub fn new(namespace: &str, component: &str, endpoint: &str, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        let descriptor = Identifier::new_endpoint(namespace, component, endpoint)?;
        Self::from_identifier(descriptor, runtime)
    }

    pub fn to_descriptor(&self) -> Instance {
        self.descriptor.clone()
    }

    pub fn instance_id(&self) -> Option<i64> {
        self.descriptor.instance_id()
    }

    /// Chain to create a path
    pub fn path(&self, segments: &[&str]) -> Result<Path, EntityError> {
        let keys = Keys::from_instance(
            self.descriptor.clone(),
            segments.iter().map(|s| s.to_string()).collect()
        )?;
        Path::from_descriptor(keys, self.runtime.clone())
    }
}

impl DistributedRuntimeProvider for Endpoint {
    fn drt(&self) -> &DistributedRuntime {
        &self.runtime
    }
}

impl RuntimeProvider for Endpoint {
    fn rt(&self) -> &crate::Runtime {
        self.runtime.rt()
    }
}

impl fmt::Display for Endpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.descriptor)
    }
}

impl DiscoveryClient for Endpoint {
    fn etcd_key(&self) -> String {
        self.to_string()
    }
}

/// Operational path with extended segments and distributed runtime
#[derive(Clone)]
pub struct Path {
    descriptor: Keys,
    runtime: DistributedRuntime,
}

impl Path {
    /// Create from Keys descriptor
    pub fn from_descriptor(keys: Keys, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        Ok(Self { descriptor: keys, runtime })
    }

    /// Convert back to Keys descriptor
    pub fn to_descriptor(&self) -> Keys {
        self.descriptor.clone()
    }

    /// Get base entity (Namespace, Component, or Endpoint)
    pub fn base_entity(&self) -> BaseEntity {
        match self.descriptor.base() {
            KeysBase::Identifier(id) => {
                if id.endpoint_name().is_some() {
                    BaseEntity::Endpoint(Endpoint::from_identifier(id.clone(), self.runtime.clone()).unwrap())
                } else if id.component_name().is_some() {
                    BaseEntity::Component(Component::from_descriptor(id.clone(), self.runtime.clone()).unwrap())
                } else {
                    BaseEntity::Namespace(Namespace::from_descriptor(id.clone(), self.runtime.clone()).unwrap())
                }
            }
            KeysBase::Instance(inst) => {
                BaseEntity::Endpoint(Endpoint::from_instance(inst.clone(), self.runtime.clone()).unwrap())
            }
        }
    }

    /// Get parent path by removing the last segment
    /// Returns None if at the root (no segments)
    pub fn parent(&self) -> Option<Path> {
        let segments = self.descriptor.keys();
        if segments.is_empty() {
            return None;
        }

        let parent_segments = segments[..segments.len() - 1].to_vec();
        let keys = match self.descriptor.base() {
            KeysBase::Identifier(id) => Keys::from_identifier(id.clone(), parent_segments).ok()?,
            KeysBase::Instance(inst) => Keys::from_instance(inst.clone(), parent_segments).ok()?,
        };

        Some(Path {
            descriptor: keys,
            runtime: self.runtime.clone(),
        })
    }

    /// Create a child path by adding a segment
    pub fn child(&self, segment: &str) -> Result<Path, EntityError> {
        let mut segments = self.descriptor.keys().to_vec();
        segments.push(segment.to_string());

        let keys = match self.descriptor.base() {
            KeysBase::Identifier(id) => Keys::from_identifier(id.clone(), segments)?,
            KeysBase::Instance(inst) => Keys::from_instance(inst.clone(), segments)?,
        };

        Ok(Path {
            descriptor: keys,
            runtime: self.runtime.clone(),
        })
    }

    /// Get the path segments, for testing
    fn segments(&self) -> &[String] {
        self.descriptor.keys()
    }
}

impl DistributedRuntimeProvider for Path {
    fn drt(&self) -> &DistributedRuntime {
        &self.runtime
    }
}

impl RuntimeProvider for Path {
    fn rt(&self) -> &crate::Runtime {
        self.runtime.rt()
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.descriptor)
    }
}

impl DiscoveryClient for Path {
    fn etcd_key(&self) -> String {
        self.to_string()
    }
}

/// Base entity types that can be the foundation of a Path
pub enum BaseEntity {
    Namespace(Namespace),
    Component(Component),
    Endpoint(Endpoint),
}

/// Entity result from identifier conversion
pub enum IdentifierEntity {
    Namespace(Namespace),
    Component(Component),
    Endpoint(Endpoint),
}

// ToEntity implementations

impl ToEntity for Identifier {
    type Entity = IdentifierEntity;

    fn to_entity(self, runtime: DistributedRuntime) -> Result<Self::Entity, EntityError> {
        if self.endpoint_name().is_some() {
            Ok(IdentifierEntity::Endpoint(Endpoint::from_identifier(self, runtime)?))
        } else if self.component_name().is_some() {
            Ok(IdentifierEntity::Component(Component::from_descriptor(self, runtime)?))
        } else {
            Ok(IdentifierEntity::Namespace(Namespace::from_descriptor(self, runtime)?))
        }
    }
}

impl ToEntity for Instance {
    type Entity = Endpoint;

    fn to_entity(self, runtime: DistributedRuntime) -> Result<Self::Entity, EntityError> {
        Endpoint::from_instance(self, runtime)
    }
}

impl ToEntity for Keys {
    type Entity = Path;

    fn to_entity(self, runtime: DistributedRuntime) -> Result<Self::Entity, EntityError> {
        Path::from_descriptor(self, runtime)
    }
}

// Add extension trait for DistributedRuntime to start the chain
pub trait EntityChain {
    fn namespace(&self, name: &str) -> Result<Namespace, EntityError>;
    fn component(&self, namespace: &str, component: &str) -> Result<Component, EntityError>;
    fn endpoint(&self, namespace: &str, component: &str, endpoint: &str) -> Result<Endpoint, EntityError>;
}

impl EntityChain for DistributedRuntime {
    fn namespace(&self, name: &str) -> Result<Namespace, EntityError> {
        Namespace::new(name, self.clone())
    }

    fn component(&self, namespace: &str, component: &str) -> Result<Component, EntityError> {
        Component::new(namespace, component, self.clone())
    }

    fn endpoint(&self, namespace: &str, component: &str, endpoint: &str) -> Result<Endpoint, EntityError> {
        Endpoint::new(namespace, component, endpoint, self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Runtime;

    async fn create_test_runtime() -> DistributedRuntime {
        let runtime = Runtime::from_current().unwrap();
        DistributedRuntime::from_settings_without_discovery(runtime).await.unwrap()
    }

    #[tokio::test]
    async fn test_namespace_entity() {
        let drt = create_test_runtime().await;

        let ns = Namespace::new("production.api.v1", drt.clone()).unwrap();
        assert_eq!(ns.to_string(), "dynamo://production.api.v1");
        assert_eq!(ns.segments(), vec!["production", "api", "v1"]);

        // Test parent
        let parent = ns.parent().unwrap();
        assert_eq!(parent.to_string(), "dynamo://production.api");

        // Test child
        let child = ns.child("v2").unwrap();
        assert_eq!(child.to_string(), "dynamo://production.api.v1.v2");

        // Test conversion back to descriptor
        let desc = ns.to_descriptor();
        assert_eq!(desc.to_string(), "dynamo://production.api.v1");
    }

    #[tokio::test]
    async fn test_component_entity() {
        let drt = create_test_runtime().await;

        let comp = Component::new("production", "gateway", drt.clone()).unwrap();
        assert_eq!(comp.to_string(), "dynamo://production/_component_/gateway");

        // Test conversion back to descriptor
        let desc = comp.to_descriptor();
        assert_eq!(desc.to_string(), "dynamo://production/_component_/gateway");
    }

    #[tokio::test]
    async fn test_endpoint_entity() {
        let drt = create_test_runtime().await;

        let ep = Endpoint::new("production", "gateway", "http", drt.clone()).unwrap();
        assert_eq!(ep.to_string(), "dynamo://production/_component_/gateway/_endpoint_/http/_static_");
        assert_eq!(ep.instance_id(), None);

        // Test with instance
        let id = Identifier::new_endpoint("production", "gateway", "http").unwrap();
        let inst = Instance::new(id, 0x1234).unwrap();
        let ep_with_inst = Endpoint::from_instance(inst, drt).unwrap();
        assert_eq!(ep_with_inst.to_string(), "dynamo://production/_component_/gateway/_endpoint_/http:1234");
        assert_eq!(ep_with_inst.instance_id(), Some(0x1234));

        // Test conversion to instance
        let inst_opt = ep_with_inst.to_descriptor();
        assert_eq!(inst_opt.instance_id().unwrap(), 0x1234);
    }

    #[tokio::test]
    async fn test_path_entity() {
        let drt = create_test_runtime().await;

        let id = Identifier::new_component("production", "gateway").unwrap();
        let keys = Keys::from_identifier(id, vec!["v1".to_string(), "leader".to_string()]).unwrap();
        let path = Path::from_descriptor(keys, drt.clone()).unwrap();

        assert_eq!(path.to_string(), "dynamo://production/_component_/gateway/_path_/v1/leader");
        assert_eq!(path.segments(), &["v1", "leader"]);

        // Test base entity extraction
        match path.base_entity() {
            BaseEntity::Component(comp) => {
                assert_eq!(comp.to_string(), "dynamo://production/_component_/gateway");
            }
            _ => panic!("Expected component base entity"),
        }
    }

    #[tokio::test]
    async fn test_to_entity_trait() {
        let drt = create_test_runtime().await;

        // Test identifier to entity
        let id = Identifier::new_endpoint("ns1", "comp1", "ep1").unwrap();
        let entity = id.to_entity(drt.clone()).unwrap();
        match entity {
            IdentifierEntity::Endpoint(ep) => {
                assert_eq!(ep.to_string(), "dynamo://ns1/_component_/comp1/_endpoint_/ep1/_static_");
            }
            _ => panic!("Expected endpoint entity"),
        }

        // Test instance to entity
        let id = Identifier::new_endpoint("ns1", "comp1", "ep1").unwrap();
        let inst = Instance::new(id, 0x5678).unwrap();
        let ep = inst.to_entity(drt.clone()).unwrap();
        assert_eq!(ep.to_string(), "dynamo://ns1/_component_/comp1/_endpoint_/ep1:5678");
        assert_eq!(ep.instance_id(), Some(0x5678));

        // Test keys to entity
        let id = Identifier::new_namespace("ns1").unwrap();
        let keys = Keys::from_identifier(id, vec!["v1".to_string(), "config".to_string()]).unwrap();
        let path = keys.to_entity(drt).unwrap();
        assert_eq!(path.to_string(), "dynamo://ns1/_path_/v1/config");
        assert_eq!(path.segments(), &["v1", "config"]);
    }

    #[tokio::test]
    async fn test_chaining() {
        let drt = create_test_runtime().await;

        // Chain from namespace to component
        let comp = drt.namespace("production").unwrap()
            .component("gateway").unwrap();
        assert_eq!(comp.to_string(), "dynamo://production/_component_/gateway");

        // Chain from namespace to component to endpoint
        let ep = drt.namespace("production").unwrap()
            .component("gateway").unwrap()
            .endpoint("http").unwrap();
        assert_eq!(ep.to_string(), "dynamo://production/_component_/gateway/_endpoint_/http/_static_");

        // Chain from namespace to path
        let path = drt.namespace("production").unwrap()
            .path(&["v1", "config"]).unwrap();
        assert_eq!(path.to_string(), "dynamo://production/_path_/v1/config");

        // Chain from component to path
        let path = drt.namespace("production").unwrap()
            .component("gateway").unwrap()
            .path(&["v1", "config"]).unwrap();
        assert_eq!(path.to_string(), "dynamo://production/_component_/gateway/_path_/v1/config");

        // Chain from endpoint to path
        let path = drt.namespace("production").unwrap()
            .component("gateway").unwrap()
            .endpoint("http").unwrap()
            .path(&["v1", "config"]).unwrap();
        assert_eq!(path.to_string(), "dynamo://production/_component_/gateway/_endpoint_/http/_static_/_path_/v1/config");
    }

    #[tokio::test]
    async fn test_direct_entity_creation() {
        let drt = create_test_runtime().await;

        let comp = drt.component("production", "gateway").unwrap();
        assert_eq!(comp.to_string(), "dynamo://production/_component_/gateway");

        let ep = drt.endpoint("production", "gateway", "http").unwrap();
        assert_eq!(ep.to_string(), "dynamo://production/_component_/gateway/_endpoint_/http/_static_");

        let path = drt.component("production", "gateway").unwrap()
            .path(&["v1", "config"]).unwrap();
        assert_eq!(path.to_string(), "dynamo://production/_component_/gateway/_path_/v1/config");
    }

    #[tokio::test]
    async fn test_path_navigation() {
        let drt = create_test_runtime().await;

        // Create a path with multiple segments
        let path = drt.namespace("production").unwrap()
            .component("gateway").unwrap()
            .path(&["v1", "config", "settings"]).unwrap();

        assert_eq!(path.to_string(), "dynamo://production/_component_/gateway/_path_/v1/config/settings");
        assert_eq!(path.segments(), &["v1", "config", "settings"]);

        // Navigate up with parent()
        let parent = path.parent().unwrap();
        assert_eq!(parent.to_string(), "dynamo://production/_component_/gateway/_path_/v1/config");
        assert_eq!(parent.segments(), &["v1", "config"]);

        // Navigate up again
        let grandparent = parent.parent().unwrap();
        assert_eq!(grandparent.to_string(), "dynamo://production/_component_/gateway/_path_/v1");
        assert_eq!(grandparent.segments(), &["v1"]);

        // Navigate up to root (no segments)
        let root = grandparent.parent().unwrap();
        assert_eq!(root.to_string(), "dynamo://production/_component_/gateway/_path_");
        assert_eq!(root.segments(), &[] as &[String]);

        // Parent of root is None
        assert!(root.parent().is_none());

        // Navigate down with child()
        let child = root.child("api").unwrap();
        assert_eq!(child.to_string(), "dynamo://production/_component_/gateway/_path_/api");

        let grandchild = child.child("v2").unwrap();
        assert_eq!(grandchild.to_string(), "dynamo://production/_component_/gateway/_path_/api/v2");

        // Test with endpoint-based path
        let ep_path = drt.endpoint("prod", "svc", "http").unwrap()
            .path(&["metrics"]).unwrap();
        assert_eq!(ep_path.to_string(), "dynamo://prod/_component_/svc/_endpoint_/http/_static_/_path_/metrics");

        let ep_child = ep_path.child("cpu").unwrap();
        assert_eq!(ep_child.to_string(), "dynamo://prod/_component_/svc/_endpoint_/http/_static_/_path_/metrics/cpu");
    }

    #[tokio::test]
    async fn test_discovery_client_trait() {
        use crate::discovery::DiscoveryClient;
        use crate::distributed::DistributedConfig;

        // Helper to create test runtime with etcd
        async fn create_test_runtime_with_etcd() -> Result<DistributedRuntime, anyhow::Error> {
            let runtime = Runtime::from_current()?;
            let mut config = DistributedConfig::from_settings(false);
            config.etcd_config.etcd_url = vec!["http://localhost:2379".to_string()];
            DistributedRuntime::new(runtime, config).await
        }

        // Check if etcd is available
        if etcd_client::Client::connect(["localhost:2379"], None).await.is_err() {
            eprintln!("Skipping test: ETCD not available");
            return;
        }

        let drt = match create_test_runtime_with_etcd().await {
            Ok(drt) => drt,
            Err(_) => {
                eprintln!("Skipping test: Could not create runtime with etcd");
                return;
            }
        };

        // Test namespace can use DiscoveryClient
        let ns = drt.namespace("test.discovery").unwrap();
        assert_eq!(ns.etcd_key(), "dynamo://test.discovery");

        // Test namespace storage operations
        if let Ok(storage) = ns.storage() {
            let test_data = b"namespace data".to_vec();

            storage.put(test_data.clone(), None).await.unwrap();
            let values = storage.get().await.unwrap();
            assert_eq!(values[0].value(), &test_data);

            // Cleanup
            storage.delete(None).await.unwrap();
        }

        // Test component can use DiscoveryClient
        let comp = drt.component("test", "discovery").unwrap();
        assert_eq!(comp.etcd_key(), "dynamo://test/_component_/discovery");

        // Test component storage operations
        if let Ok(storage) = comp.storage() {
            let test_data = b"component data".to_vec();

            storage.put(test_data.clone(), None).await.unwrap();
            let values = storage.get().await.unwrap();
            assert_eq!(values[0].value(), &test_data);

            // Cleanup
            storage.delete(None).await.unwrap();
        }

        // Test endpoint can use DiscoveryClient
        let ep = drt.endpoint("test", "discovery", "http").unwrap();
        assert!(ep.etcd_key().starts_with("dynamo://test/_component_/discovery/_endpoint_/http"));

        // Test endpoint storage operations
        if let Ok(storage) = ep.storage() {
            let test_data = b"endpoint data".to_vec();

            storage.put(test_data.clone(), None).await.unwrap();
            let values = storage.get().await.unwrap();
            assert_eq!(values[0].value(), &test_data);

            // Cleanup
            storage.delete(None).await.unwrap();
        }

        // Test path can use DiscoveryClient
        let path = comp.path(&["v1", "config"]).unwrap();
        assert_eq!(path.etcd_key(), "dynamo://test/_component_/discovery/_path_/v1/config");

        // Test path storage operations
        if let Ok(storage) = path.storage() {
            let test_data = b"path data".to_vec();

            storage.put(test_data.clone(), None).await.unwrap();
            let values = storage.get().await.unwrap();
            assert_eq!(values[0].value(), &test_data);

            // Cleanup
            storage.delete(None).await.unwrap();
        }
    }
}
