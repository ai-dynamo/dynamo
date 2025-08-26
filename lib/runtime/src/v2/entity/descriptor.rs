// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Entity descriptor system for type-safe component identification and management
//!
//! This module provides a comprehensive descriptor system that enforces entity relationships
//! and naming conventions at compile time. It replaces string-based identities with
//! structured descriptor types that provide validation and type safety.

use std::fmt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during descriptor operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum DescriptorError {
    #[error("Invalid namespace name: {name}. Must contain only lowercase letters, numbers, hyphens, and underscores")]
    InvalidNamespaceName { name: String },
    
    #[error("Invalid component name: {name}. Must contain only lowercase letters, numbers, hyphens, and underscores")]
    InvalidComponentName { name: String },
    
    #[error("Invalid endpoint name: {name}. Must contain only lowercase letters, numbers, hyphens, and underscores")]
    InvalidEndpointName { name: String },
    
    #[error("Invalid instance type: {instance_type}")]
    InvalidInstanceType { instance_type: String },
    
    #[error("Empty name not allowed")]
    EmptyName,
    
    #[error("Parse error: {message}")]
    ParseError { message: String },
    
    #[error("Validation error: {message}")]
    ValidationError { message: String },

    #[error("Reserved prefix: {name}. Names starting with '_' are reserved for internal use")]
    ReservedPrefix { name: String },
}

/// Instance type for descriptors
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InstanceType {
    /// Distributed instance that can be discovered via etcd
    Distributed,
    /// Local instance that is static and not discoverable
    Local,
}

impl fmt::Display for InstanceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstanceType::Distributed => write!(f, "distributed"),
            InstanceType::Local => write!(f, "local"),
        }
    }
}

/// Validation trait for descriptor names
pub trait DescriptorValidation {
    /// Validate a name for this descriptor type
    fn validate_name(name: &str) -> Result<(), DescriptorError>;
}

/// Validates that a name contains only allowed characters and follows naming conventions
fn validate_descriptor_name(name: &str, allow_internal: bool) -> bool {
    if name.is_empty() {
        return false;
    }
    
    // Check for reserved prefix unless internal names are allowed
    if !allow_internal && name.starts_with('_') {
        return false;
    }
    
    // Allow lowercase letters, numbers, hyphens, and underscores
    name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

/// Namespace descriptor providing type-safe namespace identification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NamespaceDescriptor {
    name: String,
    instance_type: InstanceType,
    is_internal: bool,
}

impl NamespaceDescriptor {
    /// Create a new namespace descriptor with validation
    pub fn new(name: impl Into<String>) -> Result<Self, DescriptorError> {
        let name = name.into();
        Self::validate_name(&name)?;
        
        Ok(Self {
            is_internal: name.starts_with('_'),
            name,
            instance_type: InstanceType::Distributed,
        })
    }
    
    /// Create a new local namespace descriptor
    pub fn new_local(name: impl Into<String>) -> Result<Self, DescriptorError> {
        let name = name.into();
        Self::validate_name(&name)?;
        
        Ok(Self {
            is_internal: name.starts_with('_'),
            name,
            instance_type: InstanceType::Local,
        })
    }
    
    /// Create a new internal namespace descriptor (allows underscore prefix)
    pub fn new_internal(name: impl Into<String>) -> Result<Self, DescriptorError> {
        let name = name.into();
        Self::validate_internal_name(&name)?;
        
        Ok(Self {
            name,
            instance_type: InstanceType::Distributed,
            is_internal: true,
        })
    }
    
    /// Get the namespace name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the instance type
    pub fn instance_type(&self) -> &InstanceType {
        &self.instance_type
    }
    
    /// Check if this is a distributed namespace
    pub fn is_distributed(&self) -> bool {
        matches!(self.instance_type, InstanceType::Distributed)
    }
    
    /// Check if this is an internal namespace
    pub fn is_internal(&self) -> bool {
        self.is_internal
    }
    
    /// Convert to component descriptor
    pub fn component(&self, name: impl Into<String>) -> Result<ComponentDescriptor, DescriptorError> {
        ComponentDescriptor::new(self.clone(), name)
    }

    /// Validate internal namespace names (allows underscore prefix)
    fn validate_internal_name(name: &str) -> Result<(), DescriptorError> {
        if name.is_empty() {
            return Err(DescriptorError::EmptyName);
        }
        
        if !validate_descriptor_name(name, true) {
            return Err(DescriptorError::InvalidNamespaceName { 
                name: name.to_string() 
            });
        }
        
        Ok(())
    }
}

impl DescriptorValidation for NamespaceDescriptor {
    fn validate_name(name: &str) -> Result<(), DescriptorError> {
        if name.is_empty() {
            return Err(DescriptorError::EmptyName);
        }
        
        if name.starts_with('_') {
            return Err(DescriptorError::ReservedPrefix {
                name: name.to_string(),
            });
        }
        
        if !validate_descriptor_name(name, false) {
            return Err(DescriptorError::InvalidNamespaceName { 
                name: name.to_string() 
            });
        }
        
        Ok(())
    }
}

impl fmt::Display for NamespaceDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Component descriptor providing type-safe component identification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComponentDescriptor {
    namespace: NamespaceDescriptor,
    name: String,
}

impl ComponentDescriptor {
    /// Create a new component descriptor with validation
    pub fn new(
        namespace: NamespaceDescriptor, 
        name: impl Into<String>
    ) -> Result<Self, DescriptorError> {
        let name = name.into();
        Self::validate_name(&name)?;
        
        Ok(Self { namespace, name })
    }
    
    /// Get the component name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the namespace descriptor
    pub fn namespace(&self) -> &NamespaceDescriptor {
        &self.namespace
    }
    
    /// Get the full path (namespace.component)
    pub fn path(&self) -> String {
        format!("{}.{}", self.namespace.name(), self.name)
    }
    
    /// Convert to endpoint descriptor
    pub fn endpoint(&self, name: impl Into<String>) -> Result<EndpointDescriptor, DescriptorError> {
        EndpointDescriptor::new(self.clone(), name)
    }
}

impl DescriptorValidation for ComponentDescriptor {
    fn validate_name(name: &str) -> Result<(), DescriptorError> {
        if name.is_empty() {
            return Err(DescriptorError::EmptyName);
        }
        
        if name.starts_with('_') {
            return Err(DescriptorError::ReservedPrefix {
                name: name.to_string(),
            });
        }
        
        if !validate_descriptor_name(name, false) {
            return Err(DescriptorError::InvalidComponentName { 
                name: name.to_string() 
            });
        }
        
        Ok(())
    }
}

impl fmt::Display for ComponentDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.namespace, self.name)
    }
}

/// Endpoint descriptor providing type-safe endpoint identification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EndpointDescriptor {
    component: ComponentDescriptor,
    name: String,
}

impl EndpointDescriptor {
    /// Create a new endpoint descriptor with validation
    pub fn new(
        component: ComponentDescriptor, 
        name: impl Into<String>
    ) -> Result<Self, DescriptorError> {
        let name = name.into();
        Self::validate_name(&name)?;
        
        Ok(Self { component, name })
    }
    
    /// Get the endpoint name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the component descriptor
    pub fn component(&self) -> &ComponentDescriptor {
        &self.component
    }
    
    /// Get the namespace descriptor
    pub fn namespace(&self) -> &NamespaceDescriptor {
        self.component.namespace()
    }
    
    /// Get the full path (namespace.component.endpoint)
    pub fn path(&self) -> String {
        format!("{}.{}", self.component.path(), self.name)
    }
    
    /// Convert to instance descriptor
    pub fn instance(&self, instance_id: i64) -> InstanceDescriptor {
        InstanceDescriptor::new(self.clone(), instance_id)
    }
}

impl DescriptorValidation for EndpointDescriptor {
    fn validate_name(name: &str) -> Result<(), DescriptorError> {
        if name.is_empty() {
            return Err(DescriptorError::EmptyName);
        }
        
        if name.starts_with('_') {
            return Err(DescriptorError::ReservedPrefix {
                name: name.to_string(),
            });
        }
        
        if !validate_descriptor_name(name, false) {
            return Err(DescriptorError::InvalidEndpointName { 
                name: name.to_string() 
            });
        }
        
        Ok(())
    }
}

impl fmt::Display for EndpointDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.component, self.name)
    }
}

/// Instance descriptor providing type-safe instance identification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InstanceDescriptor {
    endpoint: EndpointDescriptor,
    instance_id: i64,
}

impl InstanceDescriptor {
    /// Create a new instance descriptor
    pub fn new(endpoint: EndpointDescriptor, instance_id: i64) -> Self {
        Self { endpoint, instance_id }
    }
    
    /// Get the instance ID
    pub fn instance_id(&self) -> i64 {
        self.instance_id
    }
    
    /// Get the endpoint descriptor
    pub fn endpoint(&self) -> &EndpointDescriptor {
        &self.endpoint
    }
    
    /// Get the component descriptor
    pub fn component(&self) -> &ComponentDescriptor {
        self.endpoint.component()
    }
    
    /// Get the namespace descriptor
    pub fn namespace(&self) -> &NamespaceDescriptor {
        self.endpoint.namespace()
    }
    
    /// Get the full path with instance ID
    pub fn path(&self) -> String {
        format!("{}:{:x}", self.endpoint.path(), self.instance_id)
    }
}

impl fmt::Display for InstanceDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{:x}", self.endpoint, self.instance_id)
    }
}

/// Path descriptor for general path operations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PathDescriptor {
    path: String,
}

impl PathDescriptor {
    /// Create a new path descriptor
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }
    
    /// Get the path string
    pub fn path(&self) -> &str {
        &self.path
    }
}

impl fmt::Display for PathDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path)
    }
}

/// Builder trait for creating descriptors with fluent API
pub trait DescriptorBuilder<T> {
    type Error;
    
    /// Build the descriptor with validation
    fn build(self) -> Result<T, Self::Error>;
}

/// Conversion utilities between descriptor types
impl From<&NamespaceDescriptor> for String {
    fn from(desc: &NamespaceDescriptor) -> Self {
        desc.name().to_string()
    }
}

impl From<&ComponentDescriptor> for String {
    fn from(desc: &ComponentDescriptor) -> Self {
        desc.path()
    }
}

impl From<&EndpointDescriptor> for String {
    fn from(desc: &EndpointDescriptor) -> Self {
        desc.path()
    }
}

impl From<&InstanceDescriptor> for String {
    fn from(desc: &InstanceDescriptor) -> Self {
        desc.path()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_descriptor_names() {
        let valid_names = vec![
            "test",
            "test-component",
            "test_endpoint", 
            "component123",
            "my-test_component-123",
        ];
        
        for name in valid_names {
            assert!(validate_descriptor_name(name, false), "Should be valid: {}", name);
        }
    }
    
    #[test]
    fn test_invalid_descriptor_names() {
        let invalid_names = vec![
            "",
            "Test",           // uppercase
            "test.component", // dot not allowed
            "test component", // space not allowed
            "test@component", // @ not allowed
            "test/component", // / not allowed
        ];
        
        for name in invalid_names {
            assert!(!validate_descriptor_name(name, false), "Should be invalid: {}", name);
        }
    }

    #[test]
    fn test_internal_names() {
        // Internal names with underscore prefix should be allowed when internal flag is true
        assert!(validate_descriptor_name("_internal", true));
        assert!(validate_descriptor_name("_test_name", true));
        
        // But not when internal flag is false
        assert!(!validate_descriptor_name("_internal", false));
        assert!(!validate_descriptor_name("_test_name", false));
    }
    
    #[test]
    fn test_namespace_descriptor() {
        let ns = NamespaceDescriptor::new("test-namespace").unwrap();
        assert_eq!(ns.name(), "test-namespace");
        assert!(ns.is_distributed());
        assert!(!ns.is_internal());
        
        let local_ns = NamespaceDescriptor::new_local("local-ns").unwrap();
        assert_eq!(local_ns.instance_type(), &InstanceType::Local);
        assert!(!local_ns.is_distributed());
        
        let internal_ns = NamespaceDescriptor::new_internal("_internal").unwrap();
        assert!(internal_ns.is_internal());
        assert_eq!(internal_ns.name(), "_internal");
    }

    #[test]
    fn test_namespace_reserved_prefix() {
        // Regular namespace creation should reject underscore prefix
        assert!(matches!(
            NamespaceDescriptor::new("_reserved"),
            Err(DescriptorError::ReservedPrefix { .. })
        ));
        
        // But internal namespace creation should allow it
        assert!(NamespaceDescriptor::new_internal("_internal").is_ok());
    }
    
    #[test]
    fn test_component_descriptor() {
        let ns = NamespaceDescriptor::new("test-ns").unwrap();
        let comp = ComponentDescriptor::new(ns, "test-component").unwrap();
        
        assert_eq!(comp.name(), "test-component");
        assert_eq!(comp.namespace().name(), "test-ns");
        assert_eq!(comp.path(), "test-ns.test-component");
    }

    #[test]
    fn test_component_reserved_prefix() {
        let ns = NamespaceDescriptor::new("test-ns").unwrap();
        assert!(matches!(
            ComponentDescriptor::new(ns, "_reserved"),
            Err(DescriptorError::ReservedPrefix { .. })
        ));
    }
    
    #[test]
    fn test_endpoint_descriptor() {
        let ns = NamespaceDescriptor::new("test-ns").unwrap();
        let comp = ComponentDescriptor::new(ns, "test-component").unwrap();
        let endpoint = EndpointDescriptor::new(comp, "test-endpoint").unwrap();
        
        assert_eq!(endpoint.name(), "test-endpoint");
        assert_eq!(endpoint.component().name(), "test-component");
        assert_eq!(endpoint.namespace().name(), "test-ns");
        assert_eq!(endpoint.path(), "test-ns.test-component.test-endpoint");
    }

    #[test]
    fn test_endpoint_reserved_prefix() {
        let ns = NamespaceDescriptor::new("test-ns").unwrap();
        let comp = ComponentDescriptor::new(ns, "test-component").unwrap();
        assert!(matches!(
            EndpointDescriptor::new(comp, "_reserved"),
            Err(DescriptorError::ReservedPrefix { .. })
        ));
    }
    
    #[test]
    fn test_instance_descriptor() {
        let ns = NamespaceDescriptor::new("test-ns").unwrap();
        let comp = ComponentDescriptor::new(ns, "test-component").unwrap();
        let endpoint = EndpointDescriptor::new(comp, "test-endpoint").unwrap();
        let instance = endpoint.instance(12345);
        
        assert_eq!(instance.instance_id(), 12345);
        assert_eq!(instance.endpoint().name(), "test-endpoint");
        assert_eq!(instance.path(), "test-ns.test-component.test-endpoint:3039");
    }
    
    #[test]
    fn test_fluent_api() {
        let instance = NamespaceDescriptor::new("my-namespace")
            .unwrap()
            .component("my-component")
            .unwrap()
            .endpoint("my-endpoint")
            .unwrap()
            .instance(456);
            
        assert_eq!(instance.path(), "my-namespace.my-component.my-endpoint:1c8");
    }

    #[test]
    fn test_internal_namespace_fluent_api() {
        let instance = NamespaceDescriptor::new_internal("_internal")
            .unwrap()
            .component("oscar")
            .unwrap()
            .endpoint("objects")
            .unwrap()
            .instance(789);
            
        assert_eq!(instance.path(), "_internal.oscar.objects:315");
    }
    
    #[test]
    fn test_invalid_names() {
        assert!(NamespaceDescriptor::new("Invalid-Name").is_err());
        assert!(NamespaceDescriptor::new("").is_err());
        assert!(NamespaceDescriptor::new("test.name").is_err());
        
        let ns = NamespaceDescriptor::new("valid-ns").unwrap();
        assert!(ComponentDescriptor::new(ns.clone(), "Invalid-Component").is_err());
        assert!(ComponentDescriptor::new(ns.clone(), "").is_err());
        
        let comp = ComponentDescriptor::new(ns, "valid-component").unwrap();
        assert!(EndpointDescriptor::new(comp.clone(), "Invalid-Endpoint").is_err());
        assert!(EndpointDescriptor::new(comp, "").is_err());
    }

    #[test]
    fn test_serialization() {
        let ns = NamespaceDescriptor::new("test").unwrap();
        let comp = ComponentDescriptor::new(ns, "component").unwrap();
        let endpoint = EndpointDescriptor::new(comp, "endpoint").unwrap();
        let instance = endpoint.instance(123);

        // Test serialization round-trip
        let serialized = serde_json::to_string(&instance).unwrap();
        let deserialized: InstanceDescriptor = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(instance, deserialized);
        assert_eq!(instance.path(), deserialized.path());
    }
}