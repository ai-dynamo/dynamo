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
use dynamo_runtime::v2::entity::EntityDescriptor;
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
/// Uses descriptor path system where object name is the final path segment
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectDescriptor {
    /// Object path descriptor (includes object name as final segment)
    object_path: PathDescriptor,
    /// Content hash for content-addressable storage
    content_hash: ContentHash,
}

impl ObjectDescriptor {
    /// Create a new object descriptor with caller context
    /// Object name becomes the final path segment in the Oscar namespace hierarchy
    pub fn create(
        object_name: impl Into<String>,
        content_hash: ContentHash,
        caller_context: CallerContext,
    ) -> Result<Self, OscarDescriptorError> {
        let object_name_str = object_name.into();
        
        // Validate object name as a path segment (allows dots)
        EntityDescriptor::validate_object_name(&object_name_str)?;
        
        // Build Oscar namespace mirroring caller's context: _internal.oscar.{caller_namespace}
        let caller_ns_segments = caller_context.namespace_segments();
        let mut oscar_segments = vec!["_internal", "oscar"];
        oscar_segments.extend(caller_ns_segments.iter().map(|s| s.as_str()));
        
        // Start with Oscar namespace
        let oscar_ns = NamespaceDescriptor::new_internal(&oscar_segments)?;
        let mut oscar_path = oscar_ns.to_path();
        
        // Add component if caller has one
        if let Some(component_name) = caller_context.component_name() {
            oscar_path = oscar_path.with_segment(&component_name)?;
        }
        
        // Add endpoint if caller has one  
        if let Some(endpoint_name) = caller_context.endpoint_name() {
            oscar_path = oscar_path.with_segment(&endpoint_name)?;
        }
        
        // Add object name as final path segment
        let object_path = oscar_path.with_segment(&object_name_str)?;
        
        Ok(Self {
            object_path,
            content_hash,
        })
    }
    
    /// Get the object name (final path segment)
    pub fn object_name(&self) -> &str {
        self.object_path.segments().last()
            .expect("Object path must have at least one segment (the object name)")
    }
    
    /// Get the content hash
    pub fn content_hash(&self) -> &ContentHash {
        &self.content_hash
    }
    
    /// Get the Oscar namespace descriptor that mirrors the caller
    pub fn oscar_namespace(&self) -> Result<NamespaceDescriptor, OscarDescriptorError> {
        // Extract Oscar namespace from object path
        // The Oscar path structure is: _internal.oscar.{caller_namespace_segments}...object_name
        let namespace_segments = self.object_path.entity().namespace_segments();
        
        // Find where the caller namespace starts (after _internal.oscar)
        if namespace_segments.len() < 2 || 
           namespace_segments[0] != "_internal" || 
           namespace_segments[1] != "oscar" {
            return Err(OscarDescriptorError::InvalidCallerContext {
                message: "Object path does not contain valid Oscar namespace".to_string(),
            });
        }
        
        // Convert Vec<String> to Vec<&str> for NamespaceDescriptor::new_internal
        let segments_ref: Vec<&str> = namespace_segments.iter().map(|s| s.as_str()).collect();
        Ok(NamespaceDescriptor::new_internal(&segments_ref)?)
    }
    
    /// Get the content hash (alias for backward compatibility)
    pub fn hash(&self) -> &ContentHash {
        &self.content_hash
    }
    
    /// Get the object path descriptor
    pub fn path_descriptor(&self) -> &PathDescriptor {
        &self.object_path
    }
    
    /// Generate the base object path for prefix operations
    /// Example: dynamo://_internal.oscar.ns1.foo.generate.tokenizer_json-a1b2c3d4
    pub fn object_path(&self) -> Result<String, OscarDescriptorError> {
        self.object_key_path()
    }
    
    /// Generate object key with hash suffix  
    /// Example: tokenizer.json + hash_a1b2c3d4... → tokenizer_json-a1b2c3d4
    fn object_key(&self) -> String {
        let object_name = self.object_name();
        let hash_suffix = &self.content_hash.to_string()[..8]; // First 8 chars of hash
        let slugified_name = object_name.replace('.', "_"); // Convert dots to underscores for etcd
        format!("{}-{}", slugified_name, hash_suffix)
    }
    
    /// Generate metadata key path using existing object path structure
    /// Example: dynamo://_internal.oscar.ns1.foo.generate.tokenizer_json-a1b2c3d4.metadata  
    pub fn metadata_key(&self) -> Result<String, OscarDescriptorError> {
        let object_key = self.object_key();
        
        // The object path contains all segments: namespace + component + endpoint + object_name
        // We need to rebuild the path with object_key instead of object_name
        let namespace_segments = self.object_path.entity().namespace_segments();
        let path_segments = self.object_path.segments();
        
        // Build full path excluding the last segment (object name)
        let segments_ref: Vec<&str> = namespace_segments.iter().map(|s| s.as_str()).collect();
        let oscar_ns = NamespaceDescriptor::new_internal(&segments_ref)?;
        let mut key_path = oscar_ns.to_path();
        
        // Add all path segments except the last one (which is the object name)
        if path_segments.len() > 1 {
            for segment in &path_segments[..path_segments.len()-1] {
                key_path = key_path.with_segment(segment)?;
            }
        }
        
        // Add object key and metadata
        let metadata_path = key_path
            .with_segment(&object_key)?
            .with_segment("metadata")?;
        Ok(metadata_path.etcd_key())
    }
    
    /// Generate lease attachment key path
    /// Example: dynamo://_internal.oscar.ns1.foo.generate.tokenizer_json-a1b2c3d4.attaches.lease_123456789
    pub fn lease_attachment_key(&self, lease_id: i64) -> Result<String, OscarDescriptorError> {
        let object_key = self.object_key();
        
        // The object path contains all segments: namespace + component + endpoint + object_name
        // We need to rebuild the path with object_key instead of object_name
        let namespace_segments = self.object_path.entity().namespace_segments();
        let path_segments = self.object_path.segments();
        
        // Build full path excluding the last segment (object name)
        let segments_ref: Vec<&str> = namespace_segments.iter().map(|s| s.as_str()).collect();
        let oscar_ns = NamespaceDescriptor::new_internal(&segments_ref)?;
        let mut key_path = oscar_ns.to_path();
        
        // Add all path segments except the last one (which is the object name)
        if path_segments.len() > 1 {
            for segment in &path_segments[..path_segments.len()-1] {
                key_path = key_path.with_segment(segment)?;
            }
        }
        
        // Add object key, attaches, and lease ID
        let lease_key = format!("lease_{}", lease_id);
        let attachment_path = key_path
            .with_segment(&object_key)?
            .with_segment("attaches")?
            .with_segment(&lease_key)?;
        Ok(attachment_path.etcd_key())
    }
    
    /// Generate the base object key path for prefix operations
    /// Example: dynamo://_internal.oscar.ns1.foo.generate.tokenizer_json-a1b2c3d4
    pub fn object_key_path(&self) -> Result<String, OscarDescriptorError> {
        let object_key = self.object_key();
        
        // The object path contains all segments: namespace + component + endpoint + object_name
        // We need to rebuild the path with object_key instead of object_name
        let namespace_segments = self.object_path.entity().namespace_segments();
        let path_segments = self.object_path.segments();
        
        // Build full path excluding the last segment (object name)
        let segments_ref: Vec<&str> = namespace_segments.iter().map(|s| s.as_str()).collect();
        let oscar_ns = NamespaceDescriptor::new_internal(&segments_ref)?;
        let mut key_path = oscar_ns.to_path();
        
        // Add all path segments except the last one (which is the object name)
        if path_segments.len() > 1 {
            for segment in &path_segments[..path_segments.len()-1] {
                key_path = key_path.with_segment(segment)?;
            }
        }
        
        // Add object key
        let key_path = key_path.with_segment(&object_key)?;
        Ok(key_path.etcd_key())
    }
}

impl fmt::Display for ObjectDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.object_name())
    }
}

/// Builder specification for creating ObjectDescriptor with ergonomic API
#[derive(Builder)]
#[builder(derive(Debug))]
pub struct ObjectDescriptorSpec {
    /// Object name (will become final path segment)
    #[builder(setter(into))]
    pub object_name: String,
    
    /// Content hash
    pub content_hash: ContentHash,
    
    /// Caller context descriptor (can be namespace, component, or endpoint level)
    pub caller_context: CallerContext,
}

impl ObjectDescriptorSpec {
    /// Build an ObjectDescriptor from this specification
    pub fn build_descriptor(self) -> Result<ObjectDescriptor, OscarDescriptorError> {
        ObjectDescriptor::create(self.object_name, self.content_hash, self.caller_context)
    }
}

/// Type alias for convenient builder access
pub type ObjectDescriptorBuilder = ObjectDescriptorSpecBuilder;


#[cfg(test)]
mod tests {
    use super::*;
    use crate::ObjectHasher;
    
    fn test_hash() -> ContentHash {
        ObjectHasher::hash(b"test data")
    }
    
    #[test]
    fn test_object_name_validation() {
        // Object name validation now handled by EntityDescriptor::validate_object_name
        // Valid names
        assert!(EntityDescriptor::validate_object_name("simple").is_ok());
        assert!(EntityDescriptor::validate_object_name("with-hyphens").is_ok());
        assert!(EntityDescriptor::validate_object_name("with_underscores").is_ok());
        assert!(EntityDescriptor::validate_object_name("with.dots").is_ok());
        assert!(EntityDescriptor::validate_object_name("tokenizer.json").is_ok());
        assert!(EntityDescriptor::validate_object_name("model-v1_final.bin").is_ok());
        
        // Invalid names
        assert!(EntityDescriptor::validate_object_name("").is_err());
        assert!(EntityDescriptor::validate_object_name("With-Capital").is_err());
        assert!(EntityDescriptor::validate_object_name("with spaces").is_err());
        assert!(EntityDescriptor::validate_object_name("with/slash").is_err());
        assert!(EntityDescriptor::validate_object_name(&"a".repeat(256)).is_err()); // Too long
    }
    
    #[test]
    fn test_object_name_slugification() {
        let name = "tokenizer.json";
        let slugified = name.replace('.', "_");
        assert_eq!(slugified, "tokenizer_json");
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
        let caller_ns = NamespaceDescriptor::new(&["test"]).unwrap();
        let context = CallerContext::from_namespace(caller_ns);
        let object = ObjectDescriptor::create("test.model", hash, context).unwrap();
        
        assert_eq!(object.to_string(), "test.model");
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
        
        // Build caller context first
        let caller_context = CallerContextBuilder::default()
            .namespace_segments(vec!["prod".to_string()])
            .component("api")
            .endpoint("http")
            .build()
            .unwrap()
            .build_context()
            .unwrap();
        
        // Demonstrate the improved ergonomics - single error handling point
        let object = ObjectDescriptorBuilder::default()
            .object_name("tokenizer.json")
            .content_hash(hash.clone())
            .caller_context(caller_context)
            .build()
            .unwrap()
            .build_descriptor()
            .unwrap();
        
        // Verify the object was created correctly
        assert_eq!(object.object_name(), "tokenizer.json");
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
        
        // Test invalid object name validation through create method
        let caller_context = CallerContextBuilder::default()
            .namespace_segments(vec!["prod".to_string()])
            .build()
            .unwrap()
            .build_context()
            .unwrap();
            
        let result = ObjectDescriptor::create("Invalid Name!", hash, caller_context);
        assert!(result.is_err());
    }
}