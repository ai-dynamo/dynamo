// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use blake3::Hash;
use serde::{Deserialize, Serialize};
use std::fmt;
use crate::{OscarError, OscarResult, MAX_OBJECT_SIZE};

/// Content-addressable hash for objects using BLAKE3.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    /// Create a new hash from bytes.
    pub fn new(hash: Hash) -> Self {
        Self(*hash.as_bytes())
    }

    /// Get the hash as bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to hex string.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Parse from hex string.
    pub fn from_hex(hex: &str) -> Result<Self, hex::FromHexError> {
        let bytes = hex::decode(hex)?;
        if bytes.len() != 32 {
            return Err(hex::FromHexError::InvalidStringLength);
        }
        let mut array = [0u8; 32];
        array.copy_from_slice(&bytes);
        Ok(Self(array))
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl Serialize for ContentHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for ContentHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let hex_str = String::deserialize(deserializer)?;
        Self::from_hex(&hex_str).map_err(serde::de::Error::custom)
    }
}

/// BLAKE3 hasher for objects.
pub struct ObjectHasher;

impl ObjectHasher {
    /// Hash the given data.
    pub fn hash(data: &[u8]) -> ContentHash {
        ContentHash::new(blake3::hash(data))
    }

    /// Hash data from multiple chunks.
    pub fn hash_chunks<I>(chunks: I) -> ContentHash
    where
        I: IntoIterator<Item = bytes::Bytes>,
    {
        let mut hasher = blake3::Hasher::new();
        for chunk in chunks {
            hasher.update(&chunk);
        }
        ContentHash::new(hasher.finalize())
    }

    /// Validate that data size is within the maximum object size limit.
    pub fn validate_size(data: &[u8]) -> OscarResult<()> {
        if data.len() > MAX_OBJECT_SIZE {
            return Err(OscarError::ObjectTooLarge {
                size: data.len(),
                max_size: MAX_OBJECT_SIZE,
            });
        }
        Ok(())
    }

    /// Hash data with size validation.
    pub fn hash_with_validation(data: &[u8]) -> OscarResult<ContentHash> {
        Self::validate_size(data)?;
        Ok(Self::hash(data))
    }

    /// Verify that computed hash matches the provided hash.
    pub fn verify_hash(data: &[u8], expected_hash: &ContentHash) -> OscarResult<()> {
        let computed_hash = Self::hash(data);
        if computed_hash == *expected_hash {
            Ok(())
        } else {
            Err(OscarError::HashMismatch {
                expected: expected_hash.to_hex(),
                computed: computed_hash.to_hex(),
            })
        }
    }

    /// Hash file contents with size validation.
    pub fn hash_file<P: AsRef<std::path::Path>>(path: P) -> OscarResult<ContentHash> {
        let data = std::fs::read(path.as_ref())
            .map_err(|e| OscarError::IoError {
                path: path.as_ref().to_path_buf(),
                error: e,
            })?;
        Self::hash_with_validation(&data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_content_hash_creation() {
        let data = b"hello world";
        let hash = ObjectHasher::hash(data);
        assert_eq!(hash.as_bytes().len(), 32);
    }

    #[test]
    fn test_content_hash_hex_conversion() {
        let data = b"hello world";
        let hash = ObjectHasher::hash(data);
        let hex_str = hash.to_hex();
        assert_eq!(hex_str.len(), 64); // 32 bytes * 2 hex chars per byte
        
        let parsed_hash = ContentHash::from_hex(&hex_str).unwrap();
        assert_eq!(hash, parsed_hash);
    }

    #[test]
    fn test_content_hash_display() {
        let data = b"hello world";
        let hash = ObjectHasher::hash(data);
        let display_str = format!("{}", hash);
        assert_eq!(display_str, hash.to_hex());
    }

    #[test]
    fn test_content_hash_serialization() {
        let data = b"hello world";
        let hash = ObjectHasher::hash(data);
        
        let serialized = serde_json::to_string(&hash).unwrap();
        let deserialized: ContentHash = serde_json::from_str(&serialized).unwrap();
        assert_eq!(hash, deserialized);
    }

    #[test]
    fn test_size_validation_within_limit() {
        let small_data = vec![0u8; 1024]; // 1KB
        assert!(ObjectHasher::validate_size(&small_data).is_ok());
    }

    #[test]
    fn test_size_validation_at_limit() {
        let max_data = vec![0u8; MAX_OBJECT_SIZE]; // Exactly 32MiB
        assert!(ObjectHasher::validate_size(&max_data).is_ok());
    }

    #[test]
    fn test_size_validation_exceeds_limit() {
        let oversized_data = vec![0u8; MAX_OBJECT_SIZE + 1]; // 32MiB + 1 byte
        let result = ObjectHasher::validate_size(&oversized_data);
        assert!(result.is_err());
        
        if let Err(OscarError::ObjectTooLarge { size, max_size }) = result {
            assert_eq!(size, MAX_OBJECT_SIZE + 1);
            assert_eq!(max_size, MAX_OBJECT_SIZE);
        } else {
            panic!("Expected ObjectTooLarge error");
        }
    }

    #[test]
    fn test_hash_with_validation_success() {
        let data = b"test data";
        let result = ObjectHasher::hash_with_validation(data);
        assert!(result.is_ok());
        
        let hash = result.unwrap();
        assert_eq!(hash, ObjectHasher::hash(data));
    }

    #[test]
    fn test_hash_with_validation_too_large() {
        let oversized_data = vec![0u8; MAX_OBJECT_SIZE + 1];
        let result = ObjectHasher::hash_with_validation(&oversized_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_hash_verification_success() {
        let data = b"test verification";
        let hash = ObjectHasher::hash(data);
        assert!(ObjectHasher::verify_hash(data, &hash).is_ok());
    }

    #[test]
    fn test_hash_verification_failure() {
        let data = b"test verification";
        let hash = ObjectHasher::hash(data);
        let different_data = b"different data";
        
        let result = ObjectHasher::verify_hash(different_data, &hash);
        assert!(result.is_err());
        
        if let Err(OscarError::HashMismatch { expected, computed }) = result {
            assert_eq!(expected, hash.to_hex());
            assert_eq!(computed, ObjectHasher::hash(different_data).to_hex());
        } else {
            panic!("Expected HashMismatch error");
        }
    }

    #[test]
    fn test_hash_chunks_consistency() {
        let data1 = bytes::Bytes::from("hello ");
        let data2 = bytes::Bytes::from("world");
        let chunks = vec![data1, data2];
        
        let chunked_hash = ObjectHasher::hash_chunks(chunks);
        let direct_hash = ObjectHasher::hash(b"hello world");
        
        assert_eq!(chunked_hash, direct_hash);
    }

    #[test]
    fn test_file_hashing() {
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let test_data = b"file content for hashing";
        temp_file.write_all(test_data).unwrap();
        
        let file_hash = ObjectHasher::hash_file(temp_file.path()).unwrap();
        let direct_hash = ObjectHasher::hash(test_data);
        
        assert_eq!(file_hash, direct_hash);
    }

    #[test]
    fn test_file_hashing_too_large() {
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let oversized_data = vec![0u8; MAX_OBJECT_SIZE + 1];
        temp_file.write_all(&oversized_data).unwrap();
        
        let result = ObjectHasher::hash_file(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_file_hashing_nonexistent() {
        let result = ObjectHasher::hash_file("/nonexistent/path");
        assert!(result.is_err());
    }

    #[test] 
    fn test_hex_parsing_invalid_length() {
        let short_hex = "abc123";
        let result = ContentHash::from_hex(short_hex);
        assert!(result.is_err());
    }

    #[test]
    fn test_hex_parsing_invalid_chars() {
        let invalid_hex = "zz".repeat(32); // 64 chars but invalid hex
        let result = ContentHash::from_hex(&invalid_hex);
        assert!(result.is_err());
    }
}