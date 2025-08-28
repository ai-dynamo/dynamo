// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Oscar object descriptors with caller context mirroring
//!
//! This module provides Oscar-specific object management that mirrors the caller's
//! descriptor context within the _internal.oscar namespace. Objects are associated
//! with the caller's namespace/component/endpoint but not instance-specific.
//!
//! Key concepts:
//! - Caller context mirroring: ns1.foo.generate → _internal.oscar.ns1.foo.generate
//! - Path-based object keys: tokenizer.json-a1b2c3d4/metadata
//! - Lease attachment tracking: tokenizer.json-a1b2c3d4/attaches/lease_123
//! - Content-addressable storage with BLAKE3 hashing

use crate::ContentHash;
use dynamo_runtime::v2::{
    DescriptorError, NamespaceDescriptor, ComponentDescriptor, EndpointDescriptor, 
    InstanceDescriptor, PathDescriptor
};
use dynamo_runtime::v2::entity::ToPath;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Errors specific to Oscar object operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum OscarDescriptorError {
    #[error("Runtime descriptor error: {0}")]
    Runtime(#[from] DescriptorError),
    
    #[error("Builder validation error: {0}")]
    Builder(String),
    
    #[error("Invalid object name: '{name}'. Object names must be 1-255 characters and contain only lowercase letters, numbers, hyphens, underscores, and dots")]
    InvalidObjectName { name: String },
    
    #[error("Object name too long: {length} characters (max 255)")]
    ObjectNameTooLong { length: usize },
    
    #[error("Invalid caller context: {message}")]
    InvalidCallerContext { message: String },
    
    #[error("Missing required context: {field}")]
    MissingRequiredContext { field: String },
}

impl From<derive_builder::UninitializedFieldError> for OscarDescriptorError {
    fn from(err: derive_builder::UninitializedFieldError) -> Self {
        OscarDescriptorError::Builder(err.to_string())
    }
}

impl From<CallerContextSpecBuilderError> for OscarDescriptorError {
    fn from(err: CallerContextSpecBuilderError) -> Self {
        OscarDescriptorError::Builder(err.to_string())
    }
}

impl From<ObjectDescriptorSpecBuilderError> for OscarDescriptorError {
    fn from(err: ObjectDescriptorSpecBuilderError) -> Self {
        OscarDescriptorError::Builder(err.to_string())
    }
}

/// Object name with Oscar-specific validation (allows dots for file-like names)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectName {
    name: String,
}

impl ObjectName {
    /// Create a new object name with validation
    pub fn new(name: impl Into<String>) -> Result<Self, OscarDescriptorError> {
        let name = name.into();
        Self::validate(&name)?;
        Ok(Self { name })
    }
    
    /// Get the object name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get slugified version for use in etcd keys (dots → underscores)
    pub fn slugified(&self) -> String {
        self.name.replace('.', "_")
    }
    
    /// Validate object name according to Oscar rules
    fn validate(name: &str) -> Result<(), OscarDescriptorError> {
        if name.is_empty() {
            return Err(OscarDescriptorError::InvalidObjectName {
                name: name.to_string(),
            });
        }
        
        if name.len() > 255 {
            return Err(OscarDescriptorError::ObjectNameTooLong {
                length: name.len(),
            });
        }
        
        // Object names are more permissive than component names - allow dots
        let is_valid = name.chars().all(|c| 
            c.is_ascii_lowercase() || 
            c.is_ascii_digit() || 
            c == '-' || 
            c == '_' || 
            c == '.'  // Allow dots for file-like naming
        );
        
        if !is_valid {
            return Err(OscarDescriptorError::InvalidObjectName {
                name: name.to_string(),
            });
        }
        
        Ok(())
    }
}

impl fmt::Display for ObjectName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Context information about the caller registering an object
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CallerContext {
    /// The caller's descriptor (can be namespace, component, or endpoint - not instance)
    descriptor: CallerDescriptor,
}

/// The type of descriptor the caller is using
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CallerDescriptor {
    /// Caller is at namespace level
    Namespace(NamespaceDescriptor),
    /// Caller is at component level  
    Component(ComponentDescriptor),
    /// Caller is at endpoint level
    Endpoint(EndpointDescriptor),
}

impl CallerContext {
    /// Create caller context from a namespace descriptor
    pub fn from_namespace(namespace: NamespaceDescriptor) -> Self {
        Self {
            descriptor: CallerDescriptor::Namespace(namespace),
        }
    }
    
    /// Create caller context from a component descriptor
    pub fn from_component(component: ComponentDescriptor) -> Self {
        Self {
            descriptor: CallerDescriptor::Component(component),
        }
    }
    
    /// Create caller context from an endpoint descriptor
    pub fn from_endpoint(endpoint: EndpointDescriptor) -> Self {
        Self {
            descriptor: CallerDescriptor::Endpoint(endpoint),
        }
    }
    
    /// Create caller context from an instance descriptor (extracts endpoint)
    pub fn from_instance(instance: InstanceDescriptor) -> Self {
        Self {
            descriptor: CallerDescriptor::Endpoint(instance.endpoint()),
        }
    }
    
    /// Get the caller's namespace segments  
    pub fn namespace_segments(&self) -> Vec<String> {
        match &self.descriptor {
            CallerDescriptor::Namespace(ns) => ns.segments().to_vec(),
            CallerDescriptor::Component(comp) => comp.namespace().segments().to_vec(), 
            CallerDescriptor::Endpoint(ep) => ep.namespace().segments().to_vec(),
        }
    }
    
    /// Get the caller's component name if present
    pub fn component_name(&self) -> Option<String> {
        match &self.descriptor {
            CallerDescriptor::Namespace(_) => None,
            CallerDescriptor::Component(comp) => comp.name().map(|s| s.to_string()),
            CallerDescriptor::Endpoint(ep) => ep.component().name().map(|s| s.to_string()),
        }
    }
    
    /// Get the caller's endpoint name if present  
    pub fn endpoint_name(&self) -> Option<String> {
        match &self.descriptor {
            CallerDescriptor::Namespace(_) => None,
            CallerDescriptor::Component(_) => None,
            CallerDescriptor::Endpoint(ep) => ep.name().map(|s| s.to_string()),
        }
    }
}

/// Builder specification for creating CallerContext with ergonomic API
#[derive(Builder)]
#[builder(build_fn(validate = "Self::validate"))]
#[builder(derive(Debug))]
pub struct CallerContextSpec {
    /// Namespace segments for the caller
    #[builder(setter(into))]
    pub namespace_segments: Vec<String>,
    
    /// Optional component name
    #[builder(default, setter(strip_option, into))]
    pub component: Option<String>,
    
    /// Optional endpoint name
    #[builder(default, setter(strip_option, into))]
    pub endpoint: Option<String>,
}

impl CallerContextSpecBuilder {
    /// Validate the builder state before creating CallerContext
    fn validate(&self) -> Result<(), String> {
        // Check that namespace_segments is present and not empty
        if let Some(ref segments) = self.namespace_segments {
            if segments.is_empty() {
                return Err("Namespace segments cannot be empty".to_string());
            }
        }
        Ok(())
    }
}

impl CallerContextSpec {
    /// Build a CallerContext from this specification
    pub fn build_context(self) -> Result<CallerContext, OscarDescriptorError> {
        // Create the appropriate descriptor based on what's specified
        if let Some(endpoint_name) = self.endpoint {
            // Create endpoint descriptor
            let segments_str: Vec<&str> = self.namespace_segments.iter().map(|s| s.as_str()).collect();
            let ns = NamespaceDescriptor::new(&segments_str)?;
            let component_name = self.component.ok_or_else(|| {
                OscarDescriptorError::MissingRequiredContext {
                    field: "component".to_string(),
                }
            })?;
            let comp = ns.component(&component_name)?;
            let endpoint = comp.endpoint(&endpoint_name)?;
            Ok(CallerContext::from_endpoint(endpoint))
        } else if let Some(component_name) = self.component {
            // Create component descriptor
            let segments_str: Vec<&str> = self.namespace_segments.iter().map(|s| s.as_str()).collect();
            let ns = NamespaceDescriptor::new(&segments_str)?;
            let comp = ns.component(&component_name)?;
            Ok(CallerContext::from_component(comp))
        } else {
            // Create namespace descriptor
            let segments_str: Vec<&str> = self.namespace_segments.iter().map(|s| s.as_str()).collect();
            let ns = NamespaceDescriptor::new(&segments_str)?;
            Ok(CallerContext::from_namespace(ns))
        }
    }
}

/// Type alias for convenient builder access
pub type CallerContextBuilder = CallerContextSpecBuilder;

/// Oscar object descriptor with caller context mirroring
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectDescriptor {
    /// The object name
    object_name: ObjectName,
    /// Content hash for content-addressable storage
    content_hash: ContentHash,
    /// Caller context for namespace mirroring
    caller_context: CallerContext,
}

impl ObjectDescriptor {
    /// Create a new object descriptor with caller context
    pub fn create(
        object_name: impl Into<String>,
        content_hash: ContentHash,
        caller_context: CallerContext,
    ) -> Result<Self, OscarDescriptorError> {
        let object_name = ObjectName::new(object_name)?;
        
        Ok(Self {
            object_name,
            content_hash,
            caller_context,
        })
    }
    
    /// Get the object name
    pub fn object_name(&self) -> &ObjectName {
        &self.object_name
    }
    
    /// Get the content hash
    pub fn content_hash(&self) -> &ContentHash {
        &self.content_hash
    }
    
    /// Get the caller context
    pub fn caller_context(&self) -> &CallerContext {
        &self.caller_context
    }
    
    /// Get the content hash (alias for backward compatibility)
    pub fn hash(&self) -> &ContentHash {
        &self.content_hash
    }
    
    /// Generate the Oscar namespace mirroring the caller's context
    /// Example: caller ns1.foo.generate → _internal.oscar.ns1
    fn oscar_namespace(&self) -> Result<NamespaceDescriptor, OscarDescriptorError> {
        let caller_ns_segments = self.caller_context.namespace_segments();
        
        // Build Oscar namespace: [_internal, oscar, ...caller_namespace_segments]
        let mut oscar_segments = vec!["_internal", "oscar"];
        oscar_segments.extend(caller_ns_segments.iter().map(|s| s.as_str()));
        
        Ok(NamespaceDescriptor::new_internal(&oscar_segments)?)
    }
    
    /// Generate Oscar descriptor mirroring caller's context
    /// Examples:
    /// - Caller ns1 → _internal.oscar.ns1
    /// - Caller ns1.foo → _internal.oscar.ns1.foo
    /// - Caller ns1.foo.generate → _internal.oscar.ns1.foo.generate
    fn oscar_descriptor(&self) -> Result<PathDescriptor, OscarDescriptorError> {
        let oscar_ns = self.oscar_namespace()?;
        
        let mut oscar_desc = oscar_ns.to_path();
        
        // Add component if caller has one
        if let Some(component_name) = self.caller_context.component_name() {
            oscar_desc = oscar_desc.with_segment(&component_name)?;
        }
        
        // Add endpoint if caller has one  
        if let Some(endpoint_name) = self.caller_context.endpoint_name() {
            oscar_desc = oscar_desc.with_segment(&endpoint_name)?;
        }
        
        Ok(oscar_desc)
    }
    
    /// Generate object key with hash suffix
    /// Example: tokenizer.json + hash_a1b2c3d4... → tokenizer_json-a1b2c3d4
    fn object_key(&self) -> String {
        let hash_suffix = &self.content_hash.to_string()[..8]; // First 8 chars of hash
        format!("{}-{}", self.object_name.slugified(), hash_suffix)
    }
    
    /// Generate metadata key path
    /// Example: dynamo://_internal.oscar.ns1.foo.generate.tokenizer_json-a1b2c3d4.metadata
    pub fn metadata_key(&self) -> Result<String, OscarDescriptorError> {
        let oscar_desc = self.oscar_descriptor()?;
        let object_key = self.object_key();
        let metadata_path = oscar_desc
            .with_segment(&object_key)?
            .with_segment("metadata")?;
        Ok(metadata_path.etcd_key())
    }
    
    /// Generate lease attachment key path  
    /// Example: dynamo://_internal.oscar.ns1.foo.generate.tokenizer_json-a1b2c3d4.attaches.lease_123456789
    pub fn lease_attachment_key(&self, lease_id: i64) -> Result<String, OscarDescriptorError> {
        let oscar_desc = self.oscar_descriptor()?;
        let object_key = self.object_key();
        let lease_key = format!("lease_{}", lease_id);
        let attachment_path = oscar_desc
            .with_segment(&object_key)?
            .with_segment("attaches")?
            .with_segment(&lease_key)?;
        Ok(attachment_path.etcd_key())
    }
    
    /// Generate the base object path for prefix operations
    /// Example: dynamo://_internal.oscar.ns1.foo.generate.tokenizer_json-a1b2c3d4
    pub fn object_path(&self) -> Result<String, OscarDescriptorError> {
        let oscar_desc = self.oscar_descriptor()?;
        let object_key = self.object_key();
        let object_path = oscar_desc.with_segment(&object_key)?;
        Ok(object_path.etcd_key())
    }
}

impl fmt::Display for ObjectDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.object_name)
    }
}

/// Builder specification for creating ObjectDescriptor with ergonomic API
#[derive(Builder)]
#[builder(build_fn(validate = "Self::validate"))]
#[builder(derive(Debug))]
pub struct ObjectDescriptorSpec {
    /// Object name
    #[builder(setter(into))]
    pub object_name: String,
    
    /// Content hash
    pub content_hash: ContentHash,
    
    /// Namespace segments for the caller
    #[builder(setter(into))]
    pub namespace_segments: Vec<String>,
    
    /// Optional component name
    #[builder(default, setter(strip_option, into))]
    pub component: Option<String>,
    
    /// Optional endpoint name
    #[builder(default, setter(strip_option, into))]
    pub endpoint: Option<String>,
}

impl ObjectDescriptorSpecBuilder {
    /// Validate the builder state before creating ObjectDescriptor
    fn validate(&self) -> Result<(), String> {
        // Check that namespace_segments is present and not empty
        if let Some(ref segments) = self.namespace_segments {
            if segments.is_empty() {
                return Err("Namespace segments cannot be empty".to_string());
            }
        }
        
        // Check that object_name is present and valid
        if let Some(name) = &self.object_name {
            ObjectName::validate(name).map_err(|e| e.to_string())?;
        }
        
        Ok(())
    }
}

impl ObjectDescriptorSpec {
    /// Build an ObjectDescriptor from this specification
    pub fn build_descriptor(self) -> Result<ObjectDescriptor, OscarDescriptorError> {
        // Create the caller context first
        let mut context_builder = CallerContextBuilder::default();
        context_builder.namespace_segments(self.namespace_segments);
        
        if let Some(component) = self.component {
            context_builder.component(component);
        }
        
        if let Some(endpoint) = self.endpoint {
            context_builder.endpoint(endpoint);
        }
        
        let context_spec = context_builder.build()?;
        let caller_context = context_spec.build_context()?;
        
        // Create the object descriptor
        ObjectDescriptor::create(self.object_name, self.content_hash, caller_context)
    }
}

/// Type alias for convenient builder access
pub type ObjectDescriptorBuilder = ObjectDescriptorSpecBuilder;

// Conversion helpers for ObjectName
impl TryFrom<&str> for ObjectName {
    type Error = OscarDescriptorError;
    
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        ObjectName::new(value)
    }
}

impl TryFrom<String> for ObjectName {
    type Error = OscarDescriptorError;
    
    fn try_from(value: String) -> Result<Self, Self::Error> {
        ObjectName::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ObjectHasher;
    
    fn test_hash() -> ContentHash {
        ObjectHasher::hash(b"test data")
    }
    
    #[test]
    fn test_object_name_validation() {
        // Valid names
        assert!(ObjectName::new("simple").is_ok());
        assert!(ObjectName::new("with-hyphens").is_ok());
        assert!(ObjectName::new("with_underscores").is_ok());
        assert!(ObjectName::new("with.dots").is_ok());
        assert!(ObjectName::new("tokenizer.json").is_ok());
        assert!(ObjectName::new("model-v1_final.bin").is_ok());
        
        // Invalid names
        assert!(ObjectName::new("").is_err());
        assert!(ObjectName::new("With-Capital").is_err());
        assert!(ObjectName::new("with spaces").is_err());
        assert!(ObjectName::new("with/slash").is_err());
        assert!(ObjectName::new("a".repeat(256)).is_err()); // Too long
    }
    
    #[test]
    fn test_object_name_slugification() {
        let name = ObjectName::new("tokenizer.json").unwrap();
        assert_eq!(name.name(), "tokenizer.json");
        assert_eq!(name.slugified(), "tokenizer_json");
    }
    
    #[test]
    fn test_caller_context_creation() {
        // From namespace
        let ns = NamespaceDescriptor::new(&["ns1"]).unwrap();
        let context = CallerContext::from_namespace(ns.clone());
        assert_eq!(context.namespace_segments(), &["ns1"]);
        assert!(context.component_name().is_none());
        assert!(context.endpoint_name().is_none());
        
        // From component  
        let comp = ns.component("foo").unwrap();
        let context = CallerContext::from_component(comp.clone());
        assert_eq!(context.namespace_segments(), &["ns1"]);
        assert_eq!(context.component_name(), Some("foo".to_string()));
        assert!(context.endpoint_name().is_none());
        
        // From endpoint
        let endpoint = comp.endpoint("generate").unwrap();
        let context = CallerContext::from_endpoint(endpoint.clone());
        assert_eq!(context.namespace_segments(), &["ns1"]);
        assert_eq!(context.component_name(), Some("foo".to_string()));
        assert_eq!(context.endpoint_name(), Some("generate".to_string()));
        
        // From instance (should extract endpoint)
        let instance = endpoint.instance(123);
        let context = CallerContext::from_instance(instance);
        assert_eq!(context.namespace_segments(), &["ns1"]);
        assert_eq!(context.component_name(), Some("foo".to_string()));
        assert_eq!(context.endpoint_name(), Some("generate".to_string()));
    }
    
    #[test]
    fn test_oscar_namespace_mirroring() {
        let hash = test_hash();
        
        // Test namespace-level caller
        let caller_ns = NamespaceDescriptor::new(&["ns1"]).unwrap();
        let context = CallerContext::from_namespace(caller_ns);
        let object = ObjectDescriptor::create("tokenizer.json", hash.clone(), context).unwrap();
        
        let oscar_ns = object.oscar_namespace().unwrap();
        assert_eq!(oscar_ns.segments(), &["_internal", "oscar", "ns1"]);
        assert!(oscar_ns.is_internal());
    }
    
    #[test]
    fn test_metadata_key_generation() {
        let hash = test_hash();
        let hash_prefix = &hash.to_string()[..8];
        
        // Test with namespace-level caller
        let caller_ns = NamespaceDescriptor::new(&["ns1"]).unwrap();
        let context = CallerContext::from_namespace(caller_ns);
        let object = ObjectDescriptor::create("tokenizer.json", hash.clone(), context).unwrap();
        
        let metadata_key = object.metadata_key().unwrap();
        let expected = format!("dynamo://_internal.oscar.ns1.tokenizer_json-{}.metadata", hash_prefix);
        assert_eq!(metadata_key, expected);
    }
    
    #[test]
    fn test_metadata_key_with_component() {
        let hash = test_hash();
        let hash_prefix = &hash.to_string()[..8];
        
        // Test with component-level caller
        let caller_ns = NamespaceDescriptor::new(&["ns1"]).unwrap();
        let caller_comp = caller_ns.component("foo").unwrap();
        let context = CallerContext::from_component(caller_comp);
        let object = ObjectDescriptor::create("tokenizer.json", hash.clone(), context).unwrap();
        
        let metadata_key = object.metadata_key().unwrap();
        let expected = format!("dynamo://_internal.oscar.ns1.foo.tokenizer_json-{}.metadata", hash_prefix);
        assert_eq!(metadata_key, expected);
    }
    
    #[test]
    fn test_metadata_key_with_endpoint() {
        let hash = test_hash();
        let hash_prefix = &hash.to_string()[..8];
        
        // Test with endpoint-level caller
        let caller_ns = NamespaceDescriptor::new(&["ns1"]).unwrap();
        let caller_comp = caller_ns.component("foo").unwrap();
        let caller_endpoint = caller_comp.endpoint("generate").unwrap();
        let context = CallerContext::from_endpoint(caller_endpoint);
        let object = ObjectDescriptor::create("tokenizer.json", hash.clone(), context).unwrap();
        
        let metadata_key = object.metadata_key().unwrap();
        let expected = format!("dynamo://_internal.oscar.ns1.foo.generate.tokenizer_json-{}.metadata", hash_prefix);
        assert_eq!(metadata_key, expected);
    }
    
    #[test]
    fn test_lease_attachment_key() {
        let hash = test_hash();
        let hash_prefix = &hash.to_string()[..8];
        
        let caller_ns = NamespaceDescriptor::new(&["ns1"]).unwrap();
        let caller_comp = caller_ns.component("foo").unwrap();
        let caller_endpoint = caller_comp.endpoint("generate").unwrap();
        let context = CallerContext::from_endpoint(caller_endpoint);
        let object = ObjectDescriptor::create("tokenizer.json", hash.clone(), context).unwrap();
        
        let lease_key = object.lease_attachment_key(123456789).unwrap();
        let expected = format!("dynamo://_internal.oscar.ns1.foo.generate.tokenizer_json-{}.attaches.lease_123456789", hash_prefix);
        assert_eq!(lease_key, expected);
    }
    
    #[test]
    fn test_hierarchical_namespace_mirroring() {
        let hash = test_hash();
        
        // Test multi-level namespace
        let caller_ns = NamespaceDescriptor::new_internal(&["_system", "cache", "v1"]).unwrap();
        let context = CallerContext::from_namespace(caller_ns);
        let object = ObjectDescriptor::create("data.bin", hash.clone(), context).unwrap();
        
        let oscar_ns = object.oscar_namespace().unwrap();
        assert_eq!(oscar_ns.segments(), &["_internal", "oscar", "_system", "cache", "v1"]);
    }
    
    #[test]
    fn test_object_path_prefix() {
        let hash = test_hash();
        let hash_prefix = &hash.to_string()[..8];
        
        let caller_ns = NamespaceDescriptor::new(&["ns1"]).unwrap();
        let context = CallerContext::from_namespace(caller_ns);
        let object = ObjectDescriptor::create("model.bin", hash.clone(), context).unwrap();
        
        let object_path = object.object_path().unwrap();
        let expected = format!("dynamo://_internal.oscar.ns1.model_bin-{}", hash_prefix);
        assert_eq!(object_path, expected);
    }
    
    #[test]
    fn test_display_implementations() {
        let hash = test_hash();
        let name = ObjectName::new("test.model").unwrap();
        let caller_ns = NamespaceDescriptor::new(&["test"]).unwrap();
        let context = CallerContext::from_namespace(caller_ns);
        let object = ObjectDescriptor::create("test.model", hash, context).unwrap();
        
        assert_eq!(name.to_string(), "test.model");
        assert_eq!(object.to_string(), "test.model");
    }
    
    #[test]
    fn test_conversion_traits() {
        let name_from_str = ObjectName::try_from("valid.name").unwrap();
        assert_eq!(name_from_str.name(), "valid.name");
        
        let name_from_string = ObjectName::try_from("valid.name".to_string()).unwrap();
        assert_eq!(name_from_string.name(), "valid.name");
    }
    
    #[test]
    fn test_caller_context_builder_namespace() {
        // Test namespace-level builder
        let context = CallerContextBuilder::default()
            .namespace_segments(vec!["prod".to_string()])
            .build()
            .unwrap()
            .build_context()
            .unwrap();
        
        assert_eq!(context.namespace_segments(), &["prod"]);
        assert!(context.component_name().is_none());
        assert!(context.endpoint_name().is_none());
    }
    
    #[test]
    fn test_caller_context_builder_component() {
        // Test component-level builder
        let context = CallerContextBuilder::default()
            .namespace_segments(vec!["prod".to_string()])
            .component("api")
            .build()
            .unwrap()
            .build_context()
            .unwrap();
        
        assert_eq!(context.namespace_segments(), &["prod"]);
        assert_eq!(context.component_name(), Some("api".to_string()));
        assert!(context.endpoint_name().is_none());
    }
    
    #[test]
    fn test_caller_context_builder_endpoint() {
        // Test endpoint-level builder
        let context = CallerContextBuilder::default()
            .namespace_segments(vec!["prod".to_string()])
            .component("api")
            .endpoint("http")
            .build()
            .unwrap()
            .build_context()
            .unwrap();
        
        assert_eq!(context.namespace_segments(), &["prod"]);
        assert_eq!(context.component_name(), Some("api".to_string()));
        assert_eq!(context.endpoint_name(), Some("http".to_string()));
    }
    
    #[test]
    fn test_object_descriptor_builder_ergonomics() {
        let hash = test_hash();
        
        // Demonstrate the improved ergonomics - single error handling point
        let object = ObjectDescriptorBuilder::default()
            .object_name("tokenizer.json")
            .content_hash(hash.clone())
            .namespace_segments(vec!["prod".to_string()])
            .component("api")
            .endpoint("http")
            .build()
            .unwrap()
            .build_descriptor()
            .unwrap();
        
        // Verify the object was created correctly
        assert_eq!(object.object_name().name(), "tokenizer.json");
        assert_eq!(object.content_hash(), &hash);
        
        // Check the generated key
        let hash_prefix = &hash.to_string()[..8];
        let metadata_key = object.metadata_key().unwrap();
        let expected = format!("dynamo://_internal.oscar.prod.api.http.tokenizer_json-{}.metadata", hash_prefix);
        assert_eq!(metadata_key, expected);
    }
    
    #[test]
    fn test_builder_validation_errors() {
        let hash = test_hash();
        
        // Test empty namespace segments validation
        let result = CallerContextBuilder::default()
            .namespace_segments(Vec::<String>::new())
            .build();
        assert!(result.is_err());
        
        // Test missing component when endpoint is provided
        let result = CallerContextBuilder::default()
            .namespace_segments(vec!["prod".to_string()])
            .endpoint("http")
            .build();
        
        if let Ok(spec) = result {
            let context_result = spec.build_context();
            assert!(context_result.is_err());
        }
        
        // Test invalid object name validation
        let result = ObjectDescriptorBuilder::default()
            .object_name("Invalid Name!")
            .content_hash(hash)
            .namespace_segments(vec!["prod".to_string()])
            .build();
        assert!(result.is_err());
    }
}