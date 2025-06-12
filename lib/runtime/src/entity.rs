// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Entities for distributed runtime operations on descriptors
//! WIP

use crate::{
    descriptor::{
        Identifier, Instance, Keys, KeysBase, DescriptorError,
        BARRIER_KEYWORD, COMPONENT_KEYWORD, ENDPOINT_KEYWORD, PATH_KEYWORD,
    },
    DistributedRuntime,
};

/// Error type for entity operations
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
    /// Create from descriptor
    pub fn from_descriptor(descriptor: Identifier, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        if descriptor.component().is_some() || descriptor.endpoint().is_some() {
            return Err(EntityError::InvalidDescriptor("Expected namespace-only descriptor"));
        }
        Ok(Self { descriptor, runtime })
    }

    /// Direct construction (uses descriptor internally)
    pub fn new(namespace: &str, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        let descriptor = Identifier::new_namespace(namespace)?;
        Self::from_descriptor(descriptor, runtime)
    }

    /// Convert back to descriptor
    pub fn to_descriptor(&self) -> Identifier {
        self.descriptor.clone()
    }

    /// Get namespace hierarchy segments
    pub fn segments(&self) -> Vec<&str> {
        self.descriptor.namespace().split('.').collect()
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
        let child_path = format!("{}.{}", self.descriptor.namespace(), name);
        Namespace::new(&child_path, self.runtime.clone())
    }

    /// Access to runtime for operations
    pub fn runtime(&self) -> &DistributedRuntime {
        &self.runtime
    }

    /// Get the namespace name
    pub fn name(&self) -> &str {
        self.descriptor.namespace()
    }
}

/// Operational component with distributed runtime
#[derive(Clone)]
pub struct Component {
    descriptor: Identifier,  // Must have component
    runtime: DistributedRuntime,
}

impl Component {
    /// Create from descriptor
    pub fn from_descriptor(descriptor: Identifier, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        if descriptor.component().is_none() {
            return Err(EntityError::InvalidDescriptor("Descriptor must have component"));
        }
        if descriptor.endpoint().is_some() {
            return Err(EntityError::InvalidDescriptor("Expected component-only descriptor"));
        }
        Ok(Self { descriptor, runtime })
    }

    /// Direct construction
    pub fn new(namespace: &str, component: &str, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        let descriptor = Identifier::new_component(namespace, component)?;
        Self::from_descriptor(descriptor, runtime)
    }

    /// Convert back to descriptor
    pub fn to_descriptor(&self) -> Identifier {
        self.descriptor.clone()
    }

    /// Get parent namespace
    pub fn namespace(&self) -> Namespace {
        Namespace::new(self.descriptor.namespace(), self.runtime.clone()).unwrap()
    }

    /// Access to runtime for operations
    pub fn runtime(&self) -> &DistributedRuntime {
        &self.runtime
    }

    /// Get the component name
    pub fn name(&self) -> &str {
        self.descriptor.component().unwrap()
    }

    /// Get the namespace name
    pub fn namespace_name(&self) -> &str {
        self.descriptor.namespace()
    }
}

/// Operational endpoint with distributed runtime
#[derive(Clone)]
pub struct Endpoint {
    descriptor: Identifier,  // Must have endpoint
    instance_id: Option<i64>,
    runtime: DistributedRuntime,
}

impl Endpoint {
    /// Create from identifier descriptor
    pub fn from_identifier(descriptor: Identifier, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        if descriptor.endpoint().is_none() {
            return Err(EntityError::InvalidDescriptor("Descriptor must have endpoint"));
        }
        Ok(Self {
            descriptor,
            instance_id: None,
            runtime,
        })
    }

    /// Create from instance descriptor
    pub fn from_instance(instance: Instance, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        let descriptor = instance.identifier().clone();
        if descriptor.endpoint().is_none() {
            return Err(EntityError::InvalidDescriptor("Instance must have endpoint"));
        }
        Ok(Self {
            descriptor,
            instance_id: Some(instance.instance_id()),
            runtime,
        })
    }

    /// Direct construction
    pub fn new(namespace: &str, component: &str, endpoint: &str, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        let descriptor = Identifier::new_endpoint(namespace, component, endpoint)?;
        Self::from_identifier(descriptor, runtime)
    }

    /// Convert to descriptor (loses instance_id if present)
    pub fn to_descriptor(&self) -> Identifier {
        self.descriptor.clone()
    }

    /// Convert to instance descriptor if has instance_id
    pub fn to_instance(&self) -> Option<Instance> {
        self.instance_id.map(|id| Instance::new(self.descriptor.clone(), id).unwrap())
    }

    /// Get parent component
    pub fn component(&self) -> Component {
        Component::new(
            self.descriptor.namespace(),
            self.descriptor.component().unwrap(),
            self.runtime.clone()
        ).unwrap()
    }

    /// Access to runtime for operations
    pub fn runtime(&self) -> &DistributedRuntime {
        &self.runtime
    }

    /// Get the endpoint name
    pub fn name(&self) -> &str {
        self.descriptor.endpoint().unwrap()
    }

    /// Get the component name
    pub fn component_name(&self) -> &str {
        self.descriptor.component().unwrap()
    }

    /// Get the namespace name
    pub fn namespace_name(&self) -> &str {
        self.descriptor.namespace()
    }

    /// Get the instance ID if present
    pub fn instance_id(&self) -> Option<i64> {
        self.instance_id
    }
}

/// Operational path with extended segments and distributed runtime
#[derive(Clone)]
pub struct Path {
    keys: Keys,
    runtime: DistributedRuntime,
}

impl Path {
    /// Create from Keys descriptor
    pub fn from_keys(keys: Keys, runtime: DistributedRuntime) -> Result<Self, EntityError> {
        Ok(Self { keys, runtime })
    }

    /// Convert back to Keys descriptor
    pub fn to_keys(&self) -> Keys {
        self.keys.clone()
    }

    /// Check if path uses reserved keywords
    pub fn has_reserved_keyword(&self) -> bool {
        self.keys.keys().iter().any(|k| {
            k == BARRIER_KEYWORD || k == PATH_KEYWORD || k == COMPONENT_KEYWORD || k == ENDPOINT_KEYWORD
        })
    }

    /// Get base entity (Namespace, Component, or Endpoint)
    pub fn base_entity(&self) -> BaseEntity {
        match self.keys.base() {
            KeysBase::Identifier(id) => {
                if id.endpoint().is_some() {
                    BaseEntity::Endpoint(Endpoint::from_identifier(id.clone(), self.runtime.clone()).unwrap())
                } else if id.component().is_some() {
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

    /// Access to runtime for operations
    pub fn runtime(&self) -> &DistributedRuntime {
        &self.runtime
    }

    /// Get the additional path segments
    pub fn segments(&self) -> &[String] {
        self.keys.keys()
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
        if self.endpoint().is_some() {
            Ok(IdentifierEntity::Endpoint(Endpoint::from_identifier(self, runtime)?))
        } else if self.component().is_some() {
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
        Path::from_keys(self, runtime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Runtime;

    async fn create_test_runtime() -> DistributedRuntime {
        let runtime = Runtime::default();
        DistributedRuntime::from_settings_without_discovery(runtime).await.unwrap()
    }

    #[tokio::test]
    async fn test_namespace_entity() {
        let drt = create_test_runtime().await;

        let ns = Namespace::new("production.api.v1", drt.clone()).unwrap();
        assert_eq!(ns.name(), "production.api.v1");
        assert_eq!(ns.segments(), vec!["production", "api", "v1"]);

        // Test parent
        let parent = ns.parent().unwrap();
        assert_eq!(parent.name(), "production.api");

        // Test child
        let child = ns.child("v2").unwrap();
        assert_eq!(child.name(), "production.api.v1.v2");

        // Test conversion back to descriptor
        let desc = ns.to_descriptor();
        assert_eq!(desc.namespace(), "production.api.v1");
        assert_eq!(desc.component(), None);
        assert_eq!(desc.endpoint(), None);
    }

    #[tokio::test]
    async fn test_component_entity() {
        let drt = create_test_runtime().await;

        let comp = Component::new("production", "gateway", drt.clone()).unwrap();
        assert_eq!(comp.name(), "gateway");
        assert_eq!(comp.namespace_name(), "production");

        // Test parent namespace
        let ns = comp.namespace();
        assert_eq!(ns.name(), "production");

        // Test conversion back to descriptor
        let desc = comp.to_descriptor();
        assert_eq!(desc.namespace(), "production");
        assert_eq!(desc.component(), Some("gateway"));
        assert_eq!(desc.endpoint(), None);
    }

    #[tokio::test]
    async fn test_endpoint_entity() {
        let drt = create_test_runtime().await;

        let ep = Endpoint::new("production", "gateway", "http", drt.clone()).unwrap();
        assert_eq!(ep.name(), "http");
        assert_eq!(ep.component_name(), "gateway");
        assert_eq!(ep.namespace_name(), "production");
        assert_eq!(ep.instance_id(), None);

        // Test parent component
        let comp = ep.component();
        assert_eq!(comp.name(), "gateway");

        // Test with instance
        let id = Identifier::new_endpoint("production", "gateway", "http").unwrap();
        let inst = Instance::new(id, 0x1234).unwrap();
        let ep_with_inst = Endpoint::from_instance(inst, drt).unwrap();
        assert_eq!(ep_with_inst.instance_id(), Some(0x1234));

        // Test conversion to instance
        let inst_opt = ep_with_inst.to_instance();
        assert!(inst_opt.is_some());
        assert_eq!(inst_opt.unwrap().instance_id(), 0x1234);
    }

    #[tokio::test]
    async fn test_path_entity() {
        let drt = create_test_runtime().await;

        let id = Identifier::new_component("production", "gateway").unwrap();
        let keys = Keys::from_identifier(id, vec!["_barrier_".to_string(), "leader".to_string()]).unwrap();
        let path = Path::from_keys(keys, drt.clone()).unwrap();

        assert!(path.has_reserved_keyword());
        assert_eq!(path.segments(), &["_barrier_", "leader"]);

        // Test base entity extraction
        match path.base_entity() {
            BaseEntity::Component(comp) => {
                assert_eq!(comp.name(), "gateway");
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
                assert_eq!(ep.name(), "ep1");
            }
            _ => panic!("Expected endpoint entity"),
        }

        // Test instance to entity
        let id = Identifier::new_endpoint("ns1", "comp1", "ep1").unwrap();
        let inst = Instance::new(id, 0x5678).unwrap();
        let ep = inst.to_entity(drt.clone()).unwrap();
        assert_eq!(ep.instance_id(), Some(0x5678));

        // Test keys to entity
        let id = Identifier::new_namespace("ns1").unwrap();
        let keys = Keys::from_identifier(id, vec!["_path_".to_string(), "config".to_string()]).unwrap();
        let path = keys.to_entity(drt).unwrap();
        assert_eq!(path.segments(), &["_path_", "config"]);
    }
}
