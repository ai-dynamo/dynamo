// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Oscar-specific descriptors built on dynamo-runtime v2 descriptor system
//!
//! This module provides Oscar-specific entity descriptors that integrate with the
//! dynamo-runtime v2 descriptor system while adding Oscar-specific validation and
//! functionality for object sharing and management.

use crate::{ContentHash, OscarError, OscarResult};
use dynamo_runtime::v2::{DescriptorError, NamespaceDescriptor, ComponentDescriptor, EndpointDescriptor};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Errors specific to Oscar descriptors
#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum OscarDescriptorError {
    #[error("Runtime descriptor error: {0}")]
    Runtime(#[from] DescriptorError),
    
    #[error("Invalid object name: {name}. Object names must be 1-255 characters and contain only lowercase letters, numbers, hyphens, and underscores")]
    InvalidObjectName { name: String },
    
    #[error("Invalid object hash: {hash}")]
    InvalidObjectHash { hash: String },
    
    #[error("Object name too long: {length} characters (max 255)")]
    ObjectNameTooLong { length: usize },
}

/// Object name with Oscar-specific validation
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
        
        // Object names follow same rules as component names but are more permissive
        let is_valid = name.chars().all(|c| 
            c.is_ascii_lowercase() || 
            c.is_ascii_digit() || 
            c == '-' || 
            c == '_' || 
            c == '.'  // Allow dots in object names for file-like naming
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

/// Oscar service descriptor - represents the internal Oscar service in the `_internal` namespace
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OscarDescriptor {
    endpoint: EndpointDescriptor,
}

impl OscarDescriptor {
    /// Create Oscar descriptor for the internal Oscar service
    pub fn new() -> Result<Self, OscarDescriptorError> {
        let namespace = NamespaceDescriptor::new_internal("_internal")?;
        let component = namespace.component("oscar")?;
        let endpoint = component.endpoint("objects")?;
        
        Ok(Self { endpoint })
    }
    
    /// Get the underlying endpoint descriptor
    pub fn endpoint(&self) -> &EndpointDescriptor {
        &self.endpoint
    }
    
    /// Get the namespace (should be "_internal")
    pub fn namespace(&self) -> &NamespaceDescriptor {
        self.endpoint.namespace()
    }
    
    /// Get the component (should be "oscar") 
    pub fn component(&self) -> &ComponentDescriptor {
        self.endpoint.component()
    }
    
    /// Get the full path for this Oscar descriptor
    pub fn path(&self) -> String {
        self.endpoint.path()
    }
    
    /// Create object descriptor for a specific object
    pub fn object(&self, name: ObjectName, hash: ContentHash) -> ObjectDescriptor {
        ObjectDescriptor::new(self.clone(), name, hash)
    }
}

impl Default for OscarDescriptor {
    fn default() -> Self {
        Self::new().expect("Oscar descriptor should always be valid")
    }
}

impl fmt::Display for OscarDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.endpoint)
    }
}

/// Object descriptor representing a specific object in Oscar
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectDescriptor {
    oscar: OscarDescriptor,
    name: ObjectName,
    hash: ContentHash,
}

impl ObjectDescriptor {
    /// Create a new object descriptor
    pub fn new(oscar: OscarDescriptor, name: ObjectName, hash: ContentHash) -> Self {
        Self { oscar, name, hash }
    }
    
    /// Create object descriptor with validation
    pub fn create(
        object_name: impl Into<String>, 
        hash: ContentHash
    ) -> Result<Self, OscarDescriptorError> {
        let oscar = OscarDescriptor::new()?;
        let name = ObjectName::new(object_name)?;
        Ok(Self::new(oscar, name, hash))
    }
    
    /// Get the Oscar service descriptor
    pub fn oscar(&self) -> &OscarDescriptor {
        &self.oscar
    }
    
    /// Get the object name
    pub fn object_name(&self) -> &ObjectName {
        &self.name
    }
    
    /// Get the content hash
    pub fn hash(&self) -> &ContentHash {
        &self.hash
    }
    
    /// Get the full path for this object
    pub fn path(&self) -> String {
        format!("{}.{}", self.oscar.path(), self.name)
    }
    
    /// Generate etcd key for object metadata
    pub fn metadata_key(&self) -> String {
        format!(
            "dynamo://_internal/oscar/objects/{}/{}",
            dynamo_runtime::slug::Slug::slugify(self.name.name()),
            self.hash.to_hex()
        )
    }
    
    /// Generate etcd key for lease reference
    pub fn lease_reference_key(&self, lease_id: i64) -> String {
        format!(
            "dynamo://_internal/oscar/leases/{:x}/objects/{}/{}",
            lease_id,
            dynamo_runtime::slug::Slug::slugify(self.name.name()),
            self.hash.to_hex()
        )
    }
}

impl fmt::Display for ObjectDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.oscar, self.name)
    }
}

/// Conversion utilities
impl From<&ObjectDescriptor> for String {
    fn from(desc: &ObjectDescriptor) -> Self {
        desc.path()
    }
}

impl TryFrom<&str> for ObjectName {
    type Error = OscarDescriptorError;
    
    fn try_from(name: &str) -> Result<Self, Self::Error> {
        Self::new(name)
    }
}

impl TryFrom<String> for ObjectName {
    type Error = OscarDescriptorError;
    
    fn try_from(name: String) -> Result<Self, Self::Error> {
        Self::new(name)
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
        assert!(ObjectName::new("model-v1.2.3").is_ok());
        assert!(ObjectName::new("dataset_2024-01-01.csv").is_ok());
        
        // Invalid names
        assert!(ObjectName::new("").is_err());
        assert!(ObjectName::new("With-Capital").is_err());
        assert!(ObjectName::new("with spaces").is_err());
        assert!(ObjectName::new("with/slashes").is_err());
        assert!(ObjectName::new("with@symbols").is_err());
        
        // Too long name
        let long_name = "x".repeat(256);
        assert!(matches!(
            ObjectName::new(long_name),
            Err(OscarDescriptorError::ObjectNameTooLong { length: 256 })
        ));
    }

    #[test]
    fn test_oscar_descriptor() {
        let oscar = OscarDescriptor::new().unwrap();
        
        assert_eq!(oscar.namespace().name(), "_internal");
        assert_eq!(oscar.component().name(), "oscar");
        assert_eq!(oscar.endpoint().name(), "objects");
        assert_eq!(oscar.path(), "_internal.oscar.objects");
        assert!(oscar.namespace().is_internal());
    }

    #[test]
    fn test_object_descriptor() {
        let hash = test_hash();
        let object = ObjectDescriptor::create("my-model.bin", hash.clone()).unwrap();
        
        assert_eq!(object.object_name().name(), "my-model.bin");
        assert_eq!(object.hash(), &hash);
        assert_eq!(object.path(), "_internal.oscar.objects.my-model.bin");
    }

    #[test]
    fn test_object_descriptor_key_generation() {
        let hash = test_hash();
        let object = ObjectDescriptor::create("test-object", hash.clone()).unwrap();
        
        let metadata_key = object.metadata_key();
        assert!(metadata_key.starts_with("dynamo://_internal/oscar/objects/"));
        assert!(metadata_key.contains("test-object"));
        assert!(metadata_key.ends_with(&hash.to_hex()));
        
        let lease_key = object.lease_reference_key(0xabc123);
        assert!(lease_key.starts_with("dynamo://_internal/oscar/leases/abc123/objects/"));
        assert!(lease_key.contains("test-object"));
        assert!(lease_key.ends_with(&hash.to_hex()));
    }

    #[test]
    fn test_object_name_slugification_in_keys() {
        let hash = test_hash();
        // Use a valid object name but test that it gets slugified in the key
        let object = ObjectDescriptor::create("test.model-v1_final", hash.clone()).unwrap();
        
        let metadata_key = object.metadata_key();
        
        // The name should be valid and preserved
        assert_eq!(object.object_name().name(), "test.model-v1_final");
        // The slugified version (dots become underscores) should appear in the key
        assert!(metadata_key.contains("test_model-v1_final"));
    }

    #[test]
    fn test_serialization() {
        let hash = test_hash();
        let original = ObjectDescriptor::create("test-object", hash).unwrap();
        
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: ObjectDescriptor = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(original, deserialized);
        assert_eq!(original.path(), deserialized.path());
    }

    #[test]
    fn test_display_implementations() {
        let hash = test_hash();
        let name = ObjectName::new("test-object").unwrap();
        let oscar = OscarDescriptor::new().unwrap();
        let object = ObjectDescriptor::new(oscar.clone(), name.clone(), hash);
        
        assert_eq!(name.to_string(), "test-object");
        assert_eq!(oscar.to_string(), "_internal.oscar.objects");
        assert_eq!(object.to_string(), "_internal.oscar.objects.test-object");
    }

    #[test] 
    fn test_conversion_traits() {
        let hash = test_hash();
        let object = ObjectDescriptor::create("test", hash).unwrap();
        
        let path_string: String = (&object).into();
        assert_eq!(path_string, object.path());
        
        let name_from_str = ObjectName::try_from("valid-name").unwrap();
        assert_eq!(name_from_str.name(), "valid-name");
        
        let name_from_string = ObjectName::try_from("valid-name".to_string()).unwrap();
        assert_eq!(name_from_string.name(), "valid-name");
    }
}