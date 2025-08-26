// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Oscar etcd key structure and utilities.
//!
//! Key hierarchy:
//! ```text
//! dynamo://_internal/oscar/objects/<object-name>/<blake3-hash>
//!   -> Object metadata (hash, size, created_at, ref_count)
//!
//! dynamo://_internal/oscar/leases/<lease-id>/objects/<object-name>/<blake3-hash>  
//!   -> Lease reference (hash value for validation)
//! ```

use crate::{ContentHash, OscarError, OscarResult};
use dynamo_runtime::slug::Slug;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Oscar's root prefix in the etcd namespace
pub const OSCAR_ROOT_PREFIX: &str = "dynamo://_internal/oscar";

/// Objects namespace for object metadata
pub const OBJECTS_PREFIX: &str = "objects";

/// Leases namespace for lease references
pub const LEASES_PREFIX: &str = "leases";

/// Oscar key types for different data stored in etcd
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OscarKeyType {
    /// Object metadata key: `dynamo://_internal/oscar/objects/<name>/<hash>`
    ObjectMetadata {
        object_name: String,
        hash: ContentHash,
    },
    /// Lease reference key: `dynamo://_internal/oscar/leases/<lease-id>/objects/<name>/<hash>`
    LeaseReference {
        lease_id: i64,
        object_name: String, 
        hash: ContentHash,
    },
}

impl OscarKeyType {
    /// Generate the full etcd key path for this key type
    pub fn to_key(&self) -> String {
        match self {
            OscarKeyType::ObjectMetadata { object_name, hash } => {
                let safe_name = Slug::slugify(object_name);
                format!("{}/{}/{}/{}", OSCAR_ROOT_PREFIX, OBJECTS_PREFIX, safe_name, hash.to_hex())
            }
            OscarKeyType::LeaseReference { lease_id, object_name, hash } => {
                let safe_name = Slug::slugify(object_name);
                format!(
                    "{}/{}/{:x}/{}/{}/{}",
                    OSCAR_ROOT_PREFIX,
                    LEASES_PREFIX,
                    lease_id,
                    OBJECTS_PREFIX,
                    safe_name,
                    hash.to_hex()
                )
            }
        }
    }

    /// Parse an etcd key back into an OscarKeyType
    pub fn from_key(key: &str) -> OscarResult<Self> {
        if !key.starts_with(OSCAR_ROOT_PREFIX) {
            return Err(OscarError::InvalidOperation {
                reason: format!("Key does not start with Oscar prefix: {}", key),
            });
        }

        let suffix = &key[OSCAR_ROOT_PREFIX.len()..];
        let parts: Vec<&str> = suffix.trim_start_matches('/').split('/').collect();

        match parts.get(0) {
            Some(&OBJECTS_PREFIX) if parts.len() == 3 => {
                // objects/<name>/<hash>
                let object_name = parts[1].to_string();
                let hash = ContentHash::from_hex(parts[2])
                    .map_err(|_| OscarError::InvalidOperation {
                        reason: format!("Invalid hash in object key: {}", parts[2]),
                    })?;
                
                Ok(OscarKeyType::ObjectMetadata { object_name, hash })
            }
            Some(&LEASES_PREFIX) if parts.len() == 5 => {
                // leases/<lease-id>/objects/<name>/<hash>
                let lease_id = i64::from_str_radix(parts[1], 16)
                    .map_err(|_| OscarError::InvalidOperation {
                        reason: format!("Invalid lease ID in key: {}", parts[1]),
                    })?;
                    
                if parts[2] != OBJECTS_PREFIX {
                    return Err(OscarError::InvalidOperation {
                        reason: format!("Expected 'objects' segment, got: {}", parts[2]),
                    });
                }
                
                let object_name = parts[3].to_string();
                let hash = ContentHash::from_hex(parts[4])
                    .map_err(|_| OscarError::InvalidOperation {
                        reason: format!("Invalid hash in lease key: {}", parts[4]),
                    })?;
                
                Ok(OscarKeyType::LeaseReference { lease_id, object_name, hash })
            }
            _ => Err(OscarError::InvalidOperation {
                reason: format!("Invalid key format: {}", key),
            }),
        }
    }

    /// Get the prefix for watching all object metadata keys
    pub fn objects_prefix() -> String {
        format!("{}/{}", OSCAR_ROOT_PREFIX, OBJECTS_PREFIX)
    }

    /// Get the prefix for watching all lease references for a specific lease
    pub fn lease_prefix(lease_id: i64) -> String {
        format!("{}/{}/{:x}", OSCAR_ROOT_PREFIX, LEASES_PREFIX, lease_id)
    }

    /// Get the prefix for watching all lease references
    pub fn all_leases_prefix() -> String {
        format!("{}/{}", OSCAR_ROOT_PREFIX, LEASES_PREFIX)
    }
}

/// Metadata stored for each object in etcd
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObjectMetadata {
    /// Content hash of the object
    pub hash: ContentHash,
    /// Size of the object in bytes
    pub size: usize,
    /// When the object was first registered
    pub created_at: SystemTime,
    /// Current reference count (updated via prefix watching)
    pub ref_count: u32,
    /// Storage backend where object data is stored
    pub storage_backend: String,
    /// Storage key/path for retrieving object data
    pub storage_key: String,
}

impl ObjectMetadata {
    /// Create new object metadata for initial registration
    pub fn new(
        hash: ContentHash,
        size: usize,
        storage_backend: String,
        storage_key: String,
    ) -> Self {
        Self {
            hash,
            size,
            created_at: SystemTime::now(),
            ref_count: 0, // Will be updated by reference counting system
            storage_backend,
            storage_key,
        }
    }
}

/// Reference entry stored for each lease pointing to an object
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LeaseReference {
    /// The hash this lease references (for validation)
    pub hash: ContentHash,
    /// When this lease reference was created
    pub created_at: SystemTime,
    /// Additional metadata about how this lease uses the object
    pub metadata: LeaseMetadata,
}

impl LeaseReference {
    /// Create a new lease reference
    pub fn new(hash: ContentHash, metadata: LeaseMetadata) -> Self {
        Self {
            hash,
            created_at: SystemTime::now(),
            metadata,
        }
    }
}

/// Additional metadata about how a lease uses an object
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LeaseMetadata {
    /// Purpose or usage description
    pub usage: String,
    /// Optional tags for classification
    pub tags: Vec<String>,
}

/// Utility functions for Oscar key operations
pub struct OscarKeys;

impl OscarKeys {
    /// Create an object metadata key
    pub fn object_metadata_key(object_name: &str, hash: &ContentHash) -> String {
        OscarKeyType::ObjectMetadata {
            object_name: object_name.to_string(),
            hash: hash.clone(),
        }.to_key()
    }

    /// Create a lease reference key
    pub fn lease_reference_key(
        lease_id: i64,
        object_name: &str,
        hash: &ContentHash,
    ) -> String {
        OscarKeyType::LeaseReference {
            lease_id,
            object_name: object_name.to_string(),
            hash: hash.clone(),
        }.to_key()
    }

    /// Validate object name for use in keys
    pub fn validate_object_name(name: &str) -> OscarResult<()> {
        if name.is_empty() {
            return Err(OscarError::InvalidOperation {
                reason: "Object name cannot be empty".to_string(),
            });
        }
        
        if name.len() > 255 {
            return Err(OscarError::InvalidOperation {
                reason: format!("Object name too long: {} chars (max 255)", name.len()),
            });
        }
        
        // Additional validation can be added here
        Ok(())
    }

    /// Generate a unique object name from user-provided name and content hash
    pub fn generate_object_name(user_name: &str, hash: &ContentHash) -> String {
        // Use first 8 chars of hash for uniqueness while keeping human-readable name
        let hash_suffix = &hash.to_hex()[0..8];
        format!("{}-{}", Slug::slugify(user_name), hash_suffix)
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
    fn test_object_metadata_key_generation() {
        let hash = test_hash();
        let key = OscarKeys::object_metadata_key("my-object", &hash);
        
        assert!(key.starts_with("dynamo://_internal/oscar/objects/"));
        assert!(key.ends_with(&hash.to_hex()));
        assert!(key.contains("my-object"));
    }

    #[test]
    fn test_lease_reference_key_generation() {
        let hash = test_hash();
        let lease_id = 0xabc123;
        let key = OscarKeys::lease_reference_key(lease_id, "my-object", &hash);
        
        assert!(key.starts_with("dynamo://_internal/oscar/leases/abc123/objects/"));
        assert!(key.ends_with(&hash.to_hex()));
        assert!(key.contains("my-object"));
    }

    #[test]
    fn test_key_type_to_key_object_metadata() {
        let hash = test_hash();
        let key_type = OscarKeyType::ObjectMetadata {
            object_name: "test-object".to_string(),
            hash: hash.clone(),
        };
        
        let key = key_type.to_key();
        let expected = format!("dynamo://_internal/oscar/objects/test-object/{}", hash.to_hex());
        assert_eq!(key, expected);
    }

    #[test]
    fn test_key_type_to_key_lease_reference() {
        let hash = test_hash();
        let lease_id = 0xdeadbeef;
        let key_type = OscarKeyType::LeaseReference {
            lease_id,
            object_name: "test-object".to_string(),
            hash: hash.clone(),
        };
        
        let key = key_type.to_key();
        let expected = format!(
            "dynamo://_internal/oscar/leases/deadbeef/objects/test-object/{}",
            hash.to_hex()
        );
        assert_eq!(key, expected);
    }

    #[test]
    fn test_key_type_from_key_object_metadata() {
        let hash = test_hash();
        let key = format!("dynamo://_internal/oscar/objects/test-object/{}", hash.to_hex());
        
        let key_type = OscarKeyType::from_key(&key).unwrap();
        match key_type {
            OscarKeyType::ObjectMetadata { object_name, hash: parsed_hash } => {
                assert_eq!(object_name, "test-object");
                assert_eq!(parsed_hash, hash);
            }
            _ => panic!("Expected ObjectMetadata key type"),
        }
    }

    #[test]
    fn test_key_type_from_key_lease_reference() {
        let hash = test_hash();
        let lease_id = 0xabc123;
        let key = format!(
            "dynamo://_internal/oscar/leases/abc123/objects/test-object/{}",
            hash.to_hex()
        );
        
        let key_type = OscarKeyType::from_key(&key).unwrap();
        match key_type {
            OscarKeyType::LeaseReference { lease_id: parsed_lease_id, object_name, hash: parsed_hash } => {
                assert_eq!(parsed_lease_id, lease_id);
                assert_eq!(object_name, "test-object");
                assert_eq!(parsed_hash, hash);
            }
            _ => panic!("Expected LeaseReference key type"),
        }
    }

    #[test]
    fn test_key_parsing_invalid_prefix() {
        let result = OscarKeyType::from_key("invalid://prefix/objects/test/hash");
        assert!(result.is_err());
    }

    #[test]
    fn test_key_parsing_invalid_format() {
        let result = OscarKeyType::from_key("dynamo://_internal/oscar/invalid/format");
        assert!(result.is_err());
    }

    #[test]
    fn test_key_parsing_invalid_hash() {
        let result = OscarKeyType::from_key("dynamo://_internal/oscar/objects/test/invalid-hash");
        assert!(result.is_err());
    }

    #[test]
    fn test_prefix_functions() {
        assert_eq!(
            OscarKeyType::objects_prefix(),
            "dynamo://_internal/oscar/objects"
        );
        
        assert_eq!(
            OscarKeyType::lease_prefix(0xabc123),
            "dynamo://_internal/oscar/leases/abc123"
        );
        
        assert_eq!(
            OscarKeyType::all_leases_prefix(),
            "dynamo://_internal/oscar/leases"
        );
    }

    #[test]
    fn test_object_name_validation() {
        // Valid names
        assert!(OscarKeys::validate_object_name("valid-name").is_ok());
        assert!(OscarKeys::validate_object_name("a").is_ok());
        
        // Invalid names
        assert!(OscarKeys::validate_object_name("").is_err());
        assert!(OscarKeys::validate_object_name(&"x".repeat(256)).is_err());
    }

    #[test]
    fn test_object_name_generation() {
        let hash = test_hash();
        let name = OscarKeys::generate_object_name("My Object!", &hash);
        
        // Should be slugified and include hash prefix
        assert!(name.starts_with("my_object_"));
        assert!(name.contains(&hash.to_hex()[0..8]));
        assert_eq!(name.len(), "my_object_".len() + 8 + 1); // +1 for dash
    }

    #[test]
    fn test_object_name_slugification() {
        let hash = test_hash();
        let name = OscarKeys::generate_object_name("Special!@#$%^&*()Chars", &hash);
        
        // Should only contain valid slug characters
        assert!(name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-'));
    }

    #[test]
    fn test_object_metadata_serialization() {
        let hash = test_hash();
        let metadata = ObjectMetadata::new(
            hash.clone(),
            1024,
            "minio".to_string(),
            "bucket/object-key".to_string(),
        );
        
        let serialized = serde_json::to_string(&metadata).unwrap();
        let deserialized: ObjectMetadata = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(metadata.hash, deserialized.hash);
        assert_eq!(metadata.size, deserialized.size);
        assert_eq!(metadata.storage_backend, deserialized.storage_backend);
        assert_eq!(metadata.storage_key, deserialized.storage_key);
    }

    #[test]
    fn test_lease_reference_serialization() {
        let hash = test_hash();
        let lease_ref = LeaseReference::new(
            hash.clone(),
            LeaseMetadata {
                usage: "model weights".to_string(),
                tags: vec!["ml".to_string(), "inference".to_string()],
            },
        );
        
        let serialized = serde_json::to_string(&lease_ref).unwrap();
        let deserialized: LeaseReference = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(lease_ref.hash, deserialized.hash);
        assert_eq!(lease_ref.metadata, deserialized.metadata);
    }

    #[test]
    fn test_key_roundtrip_object_metadata() {
        let hash = test_hash();
        let original = OscarKeyType::ObjectMetadata {
            object_name: "test-object".to_string(),
            hash: hash.clone(),
        };
        
        let key = original.to_key();
        let parsed = OscarKeyType::from_key(&key).unwrap();
        
        assert_eq!(original, parsed);
    }

    #[test]
    fn test_key_roundtrip_lease_reference() {
        let hash = test_hash();
        let original = OscarKeyType::LeaseReference {
            lease_id: 0xfeedface,
            object_name: "test-object".to_string(),
            hash: hash.clone(),
        };
        
        let key = original.to_key();
        let parsed = OscarKeyType::from_key(&key).unwrap();
        
        assert_eq!(original, parsed);
    }
}