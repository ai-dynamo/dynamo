// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Composition-based entity descriptor system for distributed component management
//!
//! This module provides a composition-based descriptor system where all descriptor types
//! wrap a central EntityDescriptor struct. This design enables natural path extension,
//! hierarchical namespaces, and type-safe transitions between descriptor levels.
//!
//! Design principles:
//! - Single EntityDescriptor core with optional fields
//! - Hierarchical namespace segments: ["prod", "api", "v1"]
//! - Built-in path extension via path_segments
//! - Type-safe wrapper structs for each descriptor level
//! - Fluent API for descriptor transitions

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;
use validator::{Validate, ValidationError};

use super::validation;

/// Errors that can occur during descriptor operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum DescriptorError {
    #[error(
        "Invalid namespace segment: '{segment}'. Must contain only lowercase letters, numbers, hyphens, and underscores"
    )]
    InvalidNamespaceSegment { segment: String },

    #[error(
        "Invalid component name: '{name}'. Must contain only lowercase letters, numbers, hyphens, and underscores"
    )]
    InvalidComponentName { name: String },

    #[error(
        "Invalid endpoint name: '{name}'. Must contain only lowercase letters, numbers, hyphens, and underscores"
    )]
    InvalidEndpointName { name: String },

    #[error(
        "Invalid path segment: '{segment}'. Must contain only lowercase letters, numbers, hyphens, underscores, and dots"
    )]
    InvalidPathSegment { segment: String },

    #[error("Empty namespace segments not allowed")]
    EmptyNamespace,

    #[error("Empty name not allowed")]
    EmptyName,

    #[error("Empty path segment not allowed")]
    EmptyPathSegment,

    #[error("Parse error: {message}")]
    ParseError { message: String },

    #[error("Validation error: {message}")]
    ValidationError { message: String },

    #[error("Reserved prefix: '{name}'. Names starting with '_' are reserved for internal use")]
    ReservedPrefix { name: String },

    #[error("Invalid transition: {message}")]
    InvalidTransition { message: String },

    #[error("Builder error: {message}")]
    BuilderError { message: String },
}

impl From<validator::ValidationError> for DescriptorError {
    fn from(err: validator::ValidationError) -> Self {
        let message = match err.code.as_ref() {
            "empty_identifier" => "Identifier cannot be empty".to_string(),
            "invalid_identifier_chars" => "Identifier contains invalid characters. Only lowercase letters, numbers, hyphens, and underscores are allowed".to_string(),
            "empty_path_segment" => "Path segment cannot be empty".to_string(),
            "invalid_path_segment_chars" => "Path segment contains invalid characters. Only lowercase letters, numbers, hyphens, underscores, and dots are allowed".to_string(),
            "reserved_internal_prefix" => "Names starting with '_' are reserved for internal use".to_string(),
            "empty_collection" => "Collection cannot be empty".to_string(),
            "component_name_too_long" => "Component name too long (max 63 characters)".to_string(),
            "endpoint_name_too_long" => "Endpoint name too long (max 63 characters)".to_string(),
            _ => format!("Validation error: {}", err.code),
        };
        DescriptorError::ValidationError { message }
    }
}

/// Convert ValidationErrors (multiple errors) to DescriptorError
impl From<validator::ValidationErrors> for DescriptorError {
    fn from(errors: validator::ValidationErrors) -> Self {
        // Take the first error or create a generic message
        if let Some((_field, field_errors)) = errors.field_errors().iter().next()
            && let Some(error) = field_errors.first()
        {
            return Self::from((*error).clone());
        }

        DescriptorError::ValidationError {
            message: "Multiple validation errors occurred".to_string(),
        }
    }
}

/// Convert derive_builder errors to DescriptorError
impl From<derive_builder::UninitializedFieldError> for DescriptorError {
    fn from(err: derive_builder::UninitializedFieldError) -> Self {
        DescriptorError::BuilderError {
            message: format!("Required field not set: {}", err),
        }
    }
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

/// Core entity descriptor using composition-based design
///
/// All specialized descriptors wrap this single struct, providing:
/// - Hierarchical namespace segments
/// - Optional component, endpoint, and instance fields
/// - Natural path extension via path_segments
/// - Type-safe transitions between descriptor levels
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityDescriptor {
    /// Hierarchical namespace segments like ["prod", "api", "v1"]
    namespace_segments: Vec<String>,
    /// Optional component name
    component: Option<String>,
    /// Optional endpoint name
    endpoint: Option<String>,
    /// Optional instance ID for distributed instances
    instance: Option<i64>,
    /// Additional path segments for extension
    path_segments: Vec<String>,
}

/// Builder for ergonomic descriptor construction
///
/// This builder allows constructing complex descriptors with single-point error handling,
/// replacing the need for multiple `.unwrap()` calls in descriptor chains.
///
/// # Example
/// ```rust
/// use dynamo_runtime::v2::DescriptorBuilder;
///
/// // Instead of:
/// // let instance = NamespaceDescriptor::new(&["prod"])
/// //     .unwrap()
/// //     .component("api")
/// //     .unwrap()
/// //     .endpoint("http")
/// //     .unwrap()
/// //     .instance(456);
///
/// // You can now write:
/// let spec = DescriptorBuilder::default()
///     .namespace_segments(vec!["prod".to_string()])
///     .component("api")
///     .endpoint("http")
///     .instance(456)
///     .build()
///     .unwrap();
/// let instance = spec.build_instance().unwrap();
/// ```
#[derive(Builder, Validate)]
#[builder(derive(Debug))]
#[validate(schema(function = "validate_descriptor_spec"))]
pub struct DescriptorSpec {
    /// Namespace segments (required)
    #[builder(setter(into))]
    #[validate(length(min = 1))]
    pub namespace_segments: Vec<String>,

    /// Optional component name
    #[builder(default, setter(strip_option, into))]
    pub component: Option<String>,

    /// Optional endpoint name
    #[builder(default, setter(strip_option, into))]
    pub endpoint: Option<String>,

    /// Optional instance ID
    #[builder(default, setter(strip_option))]
    pub instance: Option<i64>,

    /// Path extension segments
    #[builder(default, setter(into))]
    pub path_segments: Vec<String>,

    /// Whether this is an internal namespace (allows underscore prefixes)
    #[builder(default)]
    pub internal: bool,
}

/// Schema validation function for DescriptorSpec
fn validate_descriptor_spec(spec: &DescriptorSpec) -> Result<(), ValidationError> {
    // Validate namespace segments with internal prefix handling
    for segment in &spec.namespace_segments {
        validation::validate_namespace_segment(segment, spec.internal)
            .map_err(|_| ValidationError::new("invalid_namespace_segment"))?;
    }

    // Validate component name if provided
    if let Some(ref component) = spec.component {
        validation::validate_component_name(component)
            .map_err(|_| ValidationError::new("invalid_component_name"))?;
    }

    // Validate endpoint name if provided
    if let Some(ref endpoint) = spec.endpoint {
        validation::validate_endpoint_name(endpoint)
            .map_err(|_| ValidationError::new("invalid_endpoint_name"))?;
    }

    // Validate path segments if provided
    for segment in &spec.path_segments {
        validation::validate_path_segment(segment)
            .map_err(|_| ValidationError::new("invalid_path_segment"))?;
    }

    Ok(())
}

/// Convenience type alias for the builder
pub type DescriptorBuilder = DescriptorSpecBuilder;

impl DescriptorSpec {
    /// Build the EntityDescriptor from the spec
    pub fn build_entity(self) -> Result<EntityDescriptor, DescriptorError> {
        // Validate the spec before building
        self.validate()?;

        Ok(EntityDescriptor {
            namespace_segments: self.namespace_segments,
            component: self.component,
            endpoint: self.endpoint,
            instance: self.instance,
            path_segments: self.path_segments,
        })
    }

    /// Build a NamespaceDescriptor
    pub fn build_namespace(self) -> Result<NamespaceDescriptor, DescriptorError> {
        Ok(NamespaceDescriptor(self.build_entity()?))
    }

    /// Build a ComponentDescriptor (requires component to be set)
    pub fn build_component(self) -> Result<ComponentDescriptor, DescriptorError> {
        if self.component.is_none() {
            return Err(DescriptorError::BuilderError {
                message: "Component name is required for ComponentDescriptor".to_string(),
            });
        }
        Ok(ComponentDescriptor(self.build_entity()?))
    }

    /// Build an EndpointDescriptor (requires component and endpoint to be set)
    pub fn build_endpoint(self) -> Result<EndpointDescriptor, DescriptorError> {
        if self.component.is_none() {
            return Err(DescriptorError::BuilderError {
                message: "Component name is required for EndpointDescriptor".to_string(),
            });
        }
        if self.endpoint.is_none() {
            return Err(DescriptorError::BuilderError {
                message: "Endpoint name is required for EndpointDescriptor".to_string(),
            });
        }
        Ok(EndpointDescriptor(self.build_entity()?))
    }

    /// Build an InstanceDescriptor (requires component, endpoint, and instance to be set)
    pub fn build_instance(self) -> Result<InstanceDescriptor, DescriptorError> {
        if self.component.is_none() {
            return Err(DescriptorError::BuilderError {
                message: "Component name is required for InstanceDescriptor".to_string(),
            });
        }
        if self.endpoint.is_none() {
            return Err(DescriptorError::BuilderError {
                message: "Endpoint name is required for InstanceDescriptor".to_string(),
            });
        }
        if self.instance.is_none() {
            return Err(DescriptorError::BuilderError {
                message: "Instance ID is required for InstanceDescriptor".to_string(),
            });
        }
        Ok(InstanceDescriptor(self.build_entity()?))
    }
}

impl EntityDescriptor {
    /// Create a new entity descriptor with namespace segments
    pub fn new_namespace(segments: &[&str]) -> Result<Self, DescriptorError> {
        if segments.is_empty() {
            return Err(DescriptorError::EmptyNamespace);
        }

        let validated_segments = segments
            .iter()
            .map(|s| {
                Self::validate_namespace_segment(s)?;
                Ok(s.to_string())
            })
            .collect::<Result<Vec<_>, DescriptorError>>()?;

        Ok(Self {
            namespace_segments: validated_segments,
            component: None,
            endpoint: None,
            instance: None,
            path_segments: Vec::new(),
        })
    }

    /// Create an internal namespace (allows underscore prefix)
    pub fn new_internal_namespace(segments: &[&str]) -> Result<Self, DescriptorError> {
        if segments.is_empty() {
            return Err(DescriptorError::EmptyNamespace);
        }

        let validated_segments = segments
            .iter()
            .map(|s| {
                Self::validate_internal_segment(s)?;
                Ok(s.to_string())
            })
            .collect::<Result<Vec<_>, DescriptorError>>()?;

        Ok(Self {
            namespace_segments: validated_segments,
            component: None,
            endpoint: None,
            instance: None,
            path_segments: Vec::new(),
        })
    }

    /// Add a component to this descriptor
    pub fn with_component(mut self, name: &str) -> Result<Self, DescriptorError> {
        Self::validate_component_name(name)?;
        self.component = Some(name.to_string());
        Ok(self)
    }

    /// Add an endpoint to this descriptor
    pub fn with_endpoint(mut self, name: &str) -> Result<Self, DescriptorError> {
        Self::validate_endpoint_name(name)?;
        self.endpoint = Some(name.to_string());
        Ok(self)
    }

    /// Add an instance ID to this descriptor
    pub fn with_instance(mut self, instance_id: i64) -> Self {
        self.instance = Some(instance_id);
        self
    }

    /// Add path segments to this descriptor
    pub fn with_path(mut self, segments: &[&str]) -> Result<Self, DescriptorError> {
        let validated_segments = segments
            .iter()
            .map(|s| {
                Self::validate_path_segment(s)?;
                Ok(s.to_string())
            })
            .collect::<Result<Vec<_>, DescriptorError>>()?;

        self.path_segments.extend(validated_segments);
        Ok(self)
    }

    /// Get namespace segments
    pub fn namespace_segments(&self) -> &[String] {
        &self.namespace_segments
    }

    /// Get component name if present
    pub fn component(&self) -> Option<&str> {
        self.component.as_deref()
    }

    /// Get endpoint name if present
    pub fn endpoint(&self) -> Option<&str> {
        self.endpoint.as_deref()
    }

    /// Get instance ID if present
    pub fn instance(&self) -> Option<i64> {
        self.instance
    }

    /// Get path segments
    pub fn path_segments(&self) -> &[String] {
        &self.path_segments
    }

    /// Generate the full path string
    pub fn path_string(&self) -> String {
        let mut parts = Vec::new();

        // Add namespace segments
        parts.extend(self.namespace_segments.iter().cloned());

        // Add component if present
        if let Some(ref component) = self.component {
            parts.push(component.clone());
        }

        // Add endpoint if present
        if let Some(ref endpoint) = self.endpoint {
            parts.push(endpoint.clone());
        }

        // Add instance if present
        if let Some(instance_id) = self.instance {
            parts.push(format!("instance-{:x}", instance_id));
        }

        // Add path segments
        parts.extend(self.path_segments.iter().cloned());

        parts.join(".")
    }

    /// Generate etcd key with dynamo:// prefix
    pub fn etcd_key(&self) -> String {
        format!("dynamo://{}", self.path_string())
    }

    /// Check if this uses an internal namespace
    pub fn is_internal(&self) -> bool {
        self.namespace_segments
            .first()
            .is_some_and(|s| s.starts_with('_'))
    }

    // Validation functions
    /// Public validation method for namespace segments
    pub fn validate_namespace_segment(segment: &str) -> Result<(), DescriptorError> {
        validation::validate_namespace_segment(segment, false)?;
        Ok(())
    }

    /// Public validation method for internal namespace segments
    pub fn validate_internal_segment(segment: &str) -> Result<(), DescriptorError> {
        validation::validate_namespace_segment(segment, true)?;
        Ok(())
    }

    /// Public validation method for component names
    pub fn validate_component_name(name: &str) -> Result<(), DescriptorError> {
        validation::validate_component_name(name)?;
        Ok(())
    }

    /// Public validation method for endpoint names
    pub fn validate_endpoint_name(name: &str) -> Result<(), DescriptorError> {
        validation::validate_endpoint_name(name)?;
        Ok(())
    }

    /// Public validation method for path segments
    pub fn validate_path_segment(segment: &str) -> Result<(), DescriptorError> {
        validation::validate_path_segment(segment)?;
        Ok(())
    }

    /// Public validation method for object names (Oscar compatibility)
    pub fn validate_object_name(name: &str) -> Result<(), DescriptorError> {
        validation::validate_object_name(name)?;
        Ok(())
    }
}

/// Namespace descriptor wrapper providing type-safe namespace operations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NamespaceDescriptor(EntityDescriptor);

impl NamespaceDescriptor {
    /// Create a new namespace descriptor
    pub fn new(segments: &[&str]) -> Result<Self, DescriptorError> {
        let entity = EntityDescriptor::new_namespace(segments)?;
        Ok(NamespaceDescriptor(entity))
    }

    /// Create a new internal namespace descriptor (allows underscore prefix)
    pub fn new_internal(segments: &[&str]) -> Result<Self, DescriptorError> {
        let entity = EntityDescriptor::new_internal_namespace(segments)?;
        Ok(NamespaceDescriptor(entity))
    }

    /// Get namespace segments
    pub fn segments(&self) -> &[String] {
        self.0.namespace_segments()
    }

    /// Get the namespace name (first segment for backwards compatibility)
    pub fn name(&self) -> &str {
        &self.0.namespace_segments()[0]
    }

    /// Check if this is an internal namespace
    pub fn is_internal(&self) -> bool {
        self.0.is_internal()
    }

    /// Convert to component descriptor
    pub fn component(&self, name: &str) -> Result<ComponentDescriptor, DescriptorError> {
        let entity = self.0.clone().with_component(name)?;
        Ok(ComponentDescriptor(entity))
    }

    /// Get the inner EntityDescriptor
    pub fn entity(&self) -> &EntityDescriptor {
        &self.0
    }
}

impl fmt::Display for NamespaceDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.namespace_segments().join("."))
    }
}

/// Component descriptor wrapper providing type-safe component operations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComponentDescriptor(EntityDescriptor);

impl ComponentDescriptor {
    /// Create a new component descriptor from namespace
    pub fn new(
        namespace: NamespaceDescriptor,
        component_name: &str,
    ) -> Result<Self, DescriptorError> {
        let entity = namespace.0.with_component(component_name)?;
        Ok(ComponentDescriptor(entity))
    }

    /// Get component name
    pub fn name(&self) -> Option<&str> {
        self.0.component()
    }

    /// Get namespace descriptor
    pub fn namespace(&self) -> NamespaceDescriptor {
        let ns_segments = self
            .0
            .namespace_segments()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let entity = if self.0.is_internal() {
            EntityDescriptor::new_internal_namespace(&ns_segments).unwrap()
        } else {
            EntityDescriptor::new_namespace(&ns_segments).unwrap()
        };
        NamespaceDescriptor(entity)
    }

    /// Get the full path (namespace.component)
    pub fn path(&self) -> String {
        self.0.path_string()
    }

    /// Convert to endpoint descriptor
    pub fn endpoint(&self, name: &str) -> Result<EndpointDescriptor, DescriptorError> {
        let entity = self.0.clone().with_endpoint(name)?;
        Ok(EndpointDescriptor(entity))
    }

    /// Get the inner EntityDescriptor
    pub fn entity(&self) -> &EntityDescriptor {
        &self.0
    }
}

impl fmt::Display for ComponentDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.path_string())
    }
}

/// Endpoint descriptor wrapper providing type-safe endpoint operations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EndpointDescriptor(EntityDescriptor);

impl EndpointDescriptor {
    /// Create a new endpoint descriptor from component
    pub fn new(
        component: ComponentDescriptor,
        endpoint_name: &str,
    ) -> Result<Self, DescriptorError> {
        let entity = component.0.with_endpoint(endpoint_name)?;
        Ok(EndpointDescriptor(entity))
    }

    /// Get endpoint name
    pub fn name(&self) -> Option<&str> {
        self.0.endpoint()
    }

    /// Get component descriptor
    pub fn component(&self) -> ComponentDescriptor {
        let ns_segments = self
            .0
            .namespace_segments()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let mut entity = if self.0.is_internal() {
            EntityDescriptor::new_internal_namespace(&ns_segments).unwrap()
        } else {
            EntityDescriptor::new_namespace(&ns_segments).unwrap()
        };
        if let Some(component) = self.0.component() {
            entity = entity.with_component(component).unwrap();
        }
        ComponentDescriptor(entity)
    }

    /// Get namespace descriptor
    pub fn namespace(&self) -> NamespaceDescriptor {
        let ns_segments = self
            .0
            .namespace_segments()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let entity = if self.0.is_internal() {
            EntityDescriptor::new_internal_namespace(&ns_segments).unwrap()
        } else {
            EntityDescriptor::new_namespace(&ns_segments).unwrap()
        };
        NamespaceDescriptor(entity)
    }

    /// Get the full path (namespace.component.endpoint)
    pub fn path(&self) -> String {
        self.0.path_string()
    }

    /// Convert to instance descriptor
    pub fn instance(&self, instance_id: i64) -> InstanceDescriptor {
        let entity = self.0.clone().with_instance(instance_id);
        InstanceDescriptor(entity)
    }

    /// Get the inner EntityDescriptor
    pub fn entity(&self) -> &EntityDescriptor {
        &self.0
    }
}

impl fmt::Display for EndpointDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.path_string())
    }
}

/// Instance descriptor wrapper providing type-safe instance operations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InstanceDescriptor(EntityDescriptor);

impl InstanceDescriptor {
    /// Create a new instance descriptor from endpoint
    pub fn new(endpoint: EndpointDescriptor, instance_id: i64) -> Self {
        let entity = endpoint.0.with_instance(instance_id);
        InstanceDescriptor(entity)
    }

    /// Get the instance ID
    pub fn instance_id(&self) -> Option<i64> {
        self.0.instance()
    }

    /// Get endpoint descriptor
    pub fn endpoint(&self) -> EndpointDescriptor {
        let ns_segments = self
            .0
            .namespace_segments()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let mut entity = if self.0.is_internal() {
            EntityDescriptor::new_internal_namespace(&ns_segments).unwrap()
        } else {
            EntityDescriptor::new_namespace(&ns_segments).unwrap()
        };
        if let Some(component) = self.0.component() {
            entity = entity.with_component(component).unwrap();
        }
        if let Some(endpoint) = self.0.endpoint() {
            entity = entity.with_endpoint(endpoint).unwrap();
        }
        EndpointDescriptor(entity)
    }

    /// Get component descriptor
    pub fn component(&self) -> ComponentDescriptor {
        let ns_segments = self
            .0
            .namespace_segments()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let mut entity = if self.0.is_internal() {
            EntityDescriptor::new_internal_namespace(&ns_segments).unwrap()
        } else {
            EntityDescriptor::new_namespace(&ns_segments).unwrap()
        };
        if let Some(component) = self.0.component() {
            entity = entity.with_component(component).unwrap();
        }
        ComponentDescriptor(entity)
    }

    /// Get namespace descriptor
    pub fn namespace(&self) -> NamespaceDescriptor {
        let ns_segments = self
            .0
            .namespace_segments()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let entity = if self.0.is_internal() {
            EntityDescriptor::new_internal_namespace(&ns_segments).unwrap()
        } else {
            EntityDescriptor::new_namespace(&ns_segments).unwrap()
        };
        NamespaceDescriptor(entity)
    }

    /// Get the full path with instance ID
    pub fn path(&self) -> String {
        self.0.path_string()
    }

    /// Get the inner EntityDescriptor
    pub fn entity(&self) -> &EntityDescriptor {
        &self.0
    }
}

impl fmt::Display for InstanceDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.path_string())
    }
}

/// Path descriptor wrapper providing path extension capabilities
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PathDescriptor(EntityDescriptor);

impl PathDescriptor {
    /// Create a path descriptor from any other descriptor type
    pub fn from_namespace(namespace: NamespaceDescriptor) -> Self {
        PathDescriptor(namespace.0)
    }

    pub fn from_component(component: ComponentDescriptor) -> Self {
        PathDescriptor(component.0)
    }

    pub fn from_endpoint(endpoint: EndpointDescriptor) -> Self {
        PathDescriptor(endpoint.0)
    }

    pub fn from_instance(instance: InstanceDescriptor) -> Self {
        PathDescriptor(instance.0)
    }

    /// Extend this path with additional segments
    pub fn extend(&self, segments: &[&str]) -> Result<Self, DescriptorError> {
        let entity = self.0.clone().with_path(segments)?;
        Ok(PathDescriptor(entity))
    }

    /// Add a single segment to this path
    pub fn with_segment(&self, segment: &str) -> Result<Self, DescriptorError> {
        self.extend(&[segment])
    }

    /// Get the full path string
    pub fn path(&self) -> String {
        self.0.path_string()
    }

    /// Generate etcd key with dynamo:// prefix
    pub fn etcd_key(&self) -> String {
        self.0.etcd_key()
    }

    /// Get path segments
    pub fn segments(&self) -> &[String] {
        self.0.path_segments()
    }

    /// Get the inner EntityDescriptor
    pub fn entity(&self) -> &EntityDescriptor {
        &self.0
    }
}

impl fmt::Display for PathDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.path_string())
    }
}

/// Trait for converting descriptors to PathDescriptor for extension
pub trait ToPath {
    fn to_path(self) -> PathDescriptor;
}

impl ToPath for NamespaceDescriptor {
    fn to_path(self) -> PathDescriptor {
        PathDescriptor::from_namespace(self)
    }
}

impl ToPath for ComponentDescriptor {
    fn to_path(self) -> PathDescriptor {
        PathDescriptor::from_component(self)
    }
}

impl ToPath for EndpointDescriptor {
    fn to_path(self) -> PathDescriptor {
        PathDescriptor::from_endpoint(self)
    }
}

impl ToPath for InstanceDescriptor {
    fn to_path(self) -> PathDescriptor {
        PathDescriptor::from_instance(self)
    }
}

/// Conversion utilities between descriptor types and strings
impl From<&NamespaceDescriptor> for String {
    fn from(desc: &NamespaceDescriptor) -> Self {
        desc.to_string()
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

impl From<&PathDescriptor> for String {
    fn from(desc: &PathDescriptor) -> Self {
        desc.path()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_descriptor_creation() {
        let entity = EntityDescriptor::new_namespace(&["prod", "api"]).unwrap();
        assert_eq!(entity.namespace_segments(), &["prod", "api"]);
        assert!(entity.component().is_none());
        assert!(entity.endpoint().is_none());
        assert!(entity.instance().is_none());
        assert!(entity.path_segments().is_empty());
    }

    #[test]
    fn test_internal_namespace_creation() {
        let entity = EntityDescriptor::new_internal_namespace(&["_internal", "oscar"]).unwrap();
        assert_eq!(entity.namespace_segments(), &["_internal", "oscar"]);
        assert!(entity.is_internal());
    }

    #[test]
    fn test_fluent_descriptor_building() {
        let entity = EntityDescriptor::new_namespace(&["prod", "api"])
            .unwrap()
            .with_component("gateway")
            .unwrap()
            .with_endpoint("http")
            .unwrap()
            .with_instance(12345)
            .with_path(&["health", "check"])
            .unwrap();

        assert_eq!(entity.namespace_segments(), &["prod", "api"]);
        assert_eq!(entity.component(), Some("gateway"));
        assert_eq!(entity.endpoint(), Some("http"));
        assert_eq!(entity.instance(), Some(12345));
        assert_eq!(entity.path_segments(), &["health", "check"]);
    }

    #[test]
    fn test_path_generation() {
        let entity = EntityDescriptor::new_internal_namespace(&["_internal", "oscar"])
            .unwrap()
            .with_component("objects")
            .unwrap()
            .with_path(&["tokenizer.json-a1b2c3d4", "metadata"])
            .unwrap();

        let expected_path = "_internal.oscar.objects.tokenizer.json-a1b2c3d4.metadata";
        assert_eq!(entity.path_string(), expected_path);
        assert_eq!(entity.etcd_key(), format!("dynamo://{}", expected_path));
    }

    #[test]
    fn test_wrapper_descriptor_creation() {
        let ns = NamespaceDescriptor::new_internal(&["_internal", "oscar"]).unwrap();
        let comp = ns.component("objects").unwrap();
        let endpoint = comp.endpoint("registry").unwrap();
        let instance = endpoint.instance(123);

        assert_eq!(ns.segments(), &["_internal", "oscar"]);
        assert_eq!(comp.name(), Some("objects"));
        assert_eq!(endpoint.name(), Some("registry"));
        assert_eq!(instance.instance_id(), Some(123));
    }

    #[test]
    fn test_path_extension() {
        let ns = NamespaceDescriptor::new_internal(&["_internal", "oscar"]).unwrap();
        let path = ns
            .to_path()
            .extend(&["objects", "tokenizer.json-a1b2c3d4"])
            .unwrap()
            .with_segment("metadata")
            .unwrap();

        let expected = "_internal.oscar.objects.tokenizer.json-a1b2c3d4.metadata";
        assert_eq!(path.path(), expected);
        assert_eq!(path.etcd_key(), format!("dynamo://{}", expected));
    }

    #[test]
    fn test_descriptor_transitions() {
        let instance = NamespaceDescriptor::new(&["prod"])
            .unwrap()
            .component("api")
            .unwrap()
            .endpoint("http")
            .unwrap()
            .instance(456);

        // Test going back down the hierarchy
        let endpoint = instance.endpoint();
        let component = instance.component();
        let namespace = instance.namespace();

        assert_eq!(namespace.name(), "prod");
        assert_eq!(component.name(), Some("api"));
        assert_eq!(endpoint.name(), Some("http"));
    }

    #[test]
    fn test_validation() {
        // Valid cases
        assert!(EntityDescriptor::new_namespace(&["valid-name"]).is_ok());
        assert!(EntityDescriptor::new_internal_namespace(&["_internal"]).is_ok());

        // Invalid cases
        assert!(EntityDescriptor::new_namespace(&[]).is_err()); // Empty
        assert!(EntityDescriptor::new_namespace(&["Invalid-Name"]).is_err()); // Uppercase
        assert!(EntityDescriptor::new_namespace(&["_internal"]).is_err()); // Reserved prefix
        assert!(EntityDescriptor::new_namespace(&["name.with.dots"]).is_err()); // Dots
    }

    #[test]
    fn test_descriptor_builder_ergonomics() {
        // Before: Multiple .unwrap() calls with error handling at each step
        // let instance = NamespaceDescriptor::new(&["prod"])
        //     .unwrap()
        //     .component("api")
        //     .unwrap()
        //     .endpoint("http")
        //     .unwrap()
        //     .instance(456);

        // After: Single error handling point with builder pattern
        let instance = DescriptorBuilder::default()
            .namespace_segments(vec!["prod".to_string()])
            .component("api")
            .endpoint("http")
            .instance(456)
            .build()
            .unwrap()
            .build_instance()
            .unwrap();

        assert_eq!(instance.instance_id(), Some(456));
        assert_eq!(instance.namespace().segments(), &["prod"]);
        assert_eq!(instance.component().name(), Some("api"));
        assert_eq!(instance.endpoint().name(), Some("http"));
    }

    #[test]
    fn test_descriptor_builder_component_only() {
        let component = DescriptorBuilder::default()
            .namespace_segments(vec!["test".to_string()])
            .component("service")
            .build()
            .unwrap()
            .build_component()
            .unwrap();

        assert_eq!(component.namespace().segments(), &["test"]);
        assert_eq!(component.name(), Some("service"));
    }

    #[test]
    fn test_descriptor_builder_validation() {
        // Test empty namespace segments (validation happens in build_namespace)
        let spec = DescriptorBuilder::default()
            .namespace_segments(Vec::<String>::new())
            .build()
            .unwrap();
        let result = spec.build_namespace();
        assert!(result.is_err());

        // Test invalid component name (validation happens in build_component)
        let spec = DescriptorBuilder::default()
            .namespace_segments(vec!["test".to_string()])
            .component("Invalid-Component!")
            .build()
            .unwrap();
        let result = spec.build_component();
        assert!(result.is_err());

        // Test endpoint without component
        let spec = DescriptorBuilder::default()
            .namespace_segments(vec!["test".to_string()])
            .endpoint("http")
            .build()
            .unwrap();

        let endpoint_result = spec.build_endpoint();
        assert!(endpoint_result.is_err());
    }

    #[test]
    fn test_descriptor_builder_internal_namespace() {
        let namespace = DescriptorBuilder::default()
            .namespace_segments(vec!["_internal".to_string(), "oscar".to_string()])
            .internal(true)
            .build()
            .unwrap()
            .build_namespace()
            .unwrap();

        assert!(namespace.is_internal());
        assert_eq!(namespace.segments(), &["_internal", "oscar"]);
    }
}
