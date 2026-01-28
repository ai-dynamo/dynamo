// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//!
//! This module provides [`RemoteStorage`] - a NIXL-compatible memory region
//! for remote storage transfers via NIXL's OBJ backend.
//!
//! ## Usage
//!
//! ```ignore
//! use dynamo_llm::block_manager::storage::{RemoteStorage, ObjectStorage};
//! use dynamo_llm::block_manager::block::transfer::remote::RemoteKey;
//!
//! // Create an object storage region (legacy style)
//! let storage = ObjectStorage::new("my-bucket", 0x1234567890abcdef, 4096 * 128);
//!
//! // Create from RemoteKey (new unified style)
//! let key = RemoteKey::object("my-bucket", "block-001");
//! let storage = RemoteStorage::from_remote_key(&key, 4096 * 128);
//!
//! // Register with NIXL agent
//! let handle = agent.register_memory(&storage, None)?;
//! ```
use std::fmt;

use crate::block_manager::block::transfer::remote::RemoteKey;

/// Result type for RemoteStorage operations.
pub type Result<T> = std::result::Result<T, RemoteStorageError>;

/// Error type for RemoteStorage operations.
#[derive(Debug)]
pub struct RemoteStorageError(String);

impl RemoteStorageError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl std::fmt::Display for RemoteStorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RemoteStorageError: {}", self.0)
    }
}

impl std::error::Error for RemoteStorageError {}

/// Legacy error type alias for backward compatibility.
pub type ObjectStorageError = RemoteStorageError;

/// Remote storage region for NIXL transfers.
///
/// This struct represents a region of remote storage that can be registered
/// with NIXL for transfers. The storage is identified by a bucket and key.
///
/// The `device_id` field is derived from hashing the `RemoteKey` and is used
/// by NIXL to identify the storage location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteStorage {
    bucket: String,
    key: String,
    size: usize,
    /// Pre-computed NIXL device_id for consistency with RemoteKey
    device_id: u64,
}

impl RemoteStorage {
    /// Create a new remote storage region from a sequence hash.
    ///
    /// # Arguments
    /// * `bucket` - Object bucket name
    /// * `sequence_hash` - The sequence hash (used as both key and device_id)
    /// * `size` - Size of the region in bytes
    pub fn new(bucket: impl Into<String>, sequence_hash: u64, size: usize) -> Self {
        Self {
            bucket: bucket.into(),
            key: format!("{:016x}", sequence_hash),
            size,
            device_id: sequence_hash,
        }
    }

    /// Create from a RemoteKey.
    ///
    /// The device_id is the sequence hash embedded in the key.
    pub fn from_remote_key(remote_key: &RemoteKey, size: usize) -> Self {
        Self {
            bucket: remote_key.location().to_string(),
            key: remote_key.key_str().to_string(),
            size,
            device_id: remote_key.nixl_device_id(),
        }
    }

    /// Create an object storage region (legacy API).
    ///
    /// # Arguments
    /// * `bucket` - Object bucket name
    /// * `key` - Object key (u64 numeric identifier, typically a sequence hash)
    /// * `size` - Size of the region in bytes
    ///
    /// # Returns
    /// `Result<Self>` for API consistency
    pub fn new_object(bucket: impl Into<String>, key: u64, size: usize) -> Result<Self> {
        Ok(Self::new(bucket, key, size))
    }

    /// Get the bucket name.
    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    /// Get the key string.
    pub fn key_str(&self) -> &str {
        &self.key
    }

    /// Get the pre-computed NIXL device_id.
    ///
    /// This is derived from the RemoteKey hash and is consistent across
    /// RemoteStorage and RemoteKey for the same bucket+key.
    pub fn device_id(&self) -> u64 {
        self.device_id
    }

    /// Get the size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the object key as u64 (legacy).
    /// Returns the device_id for backward compatibility.
    pub fn key(&self) -> u64 {
        self.device_id
    }
}

// Implement nixl_sys traits for direct registration with NIXL agent

impl nixl_sys::MemoryRegion for RemoteStorage {
    unsafe fn as_ptr(&self) -> *const u8 {
        // Remote storage doesn't use direct memory pointers
        std::ptr::null()
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl nixl_sys::NixlDescriptor for RemoteStorage {
    fn mem_type(&self) -> nixl_sys::MemType {
        nixl_sys::MemType::Object
    }

    fn device_id(&self) -> u64 {
        self.device_id
    }
}

impl fmt::Display for RemoteStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "obj://{}/{}", self.bucket, self.key)
    }
}

// Implement core Storage trait for interoperability with block manager
impl super::Storage for RemoteStorage {
    fn storage_type(&self) -> super::StorageType {
        super::StorageType::Nixl
    }

    fn addr(&self) -> u64 {
        0
    }

    fn size(&self) -> usize {
        self.size
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        std::ptr::null()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        std::ptr::null_mut()
    }
}

// Remote storage is remote - accessed via NIXL transfers
impl super::Remote for RemoteStorage {}

/// Type alias for backward compatibility.
///
/// New code should use `RemoteStorage` directly.
pub type ObjectStorage = RemoteStorage;

// Legacy constructor wrapper for backward compatibility
impl ObjectStorage {
    /// Create a new object storage region (legacy API).
    ///
    /// This is equivalent to `RemoteStorage::new_object()`.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(bucket: impl Into<String>, key: u64, size: usize) -> Result<RemoteStorage> {
        RemoteStorage::new_object(bucket, key, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_storage_creation() {
        let storage = ObjectStorage::new("test-bucket", 0x1234, 4096).unwrap();
        assert_eq!(storage.bucket(), "test-bucket");
        assert_eq!(storage.key_str(), "0000000000001234");
        assert_eq!(storage.device_id(), 0x1234);
        assert_eq!(storage.size(), 4096);
    }

    #[test]
    fn test_remote_storage_from_key() {
        let key = RemoteKey::object_from_hash("my-bucket", 0xABCD);
        let storage = RemoteStorage::from_remote_key(&key, 8192);
        assert_eq!(storage.bucket(), "my-bucket");
        assert_eq!(storage.key_str(), "000000000000abcd");
        assert_eq!(storage.device_id(), 0xABCD);
        assert_eq!(storage.size(), 8192);
    }

    #[test]
    fn test_nixl_descriptor() {
        use nixl_sys::NixlDescriptor;

        let storage = RemoteStorage::new("bucket", 0x5678, 8192);
        assert_eq!(storage.mem_type(), nixl_sys::MemType::Object);
        assert_eq!(storage.device_id(), 0x5678);
    }

    #[test]
    fn test_device_id_is_sequence_hash() {
        // device_id should be the sequence hash directly
        let hash: u64 = 0x123456789ABCDEF0;
        let storage = RemoteStorage::new("my-bucket", hash, 4096);
        assert_eq!(storage.device_id(), hash);

        // Same via RemoteKey
        let key = RemoteKey::object_from_hash("my-bucket", hash);
        let storage2 = RemoteStorage::from_remote_key(&key, 4096);
        assert_eq!(storage2.device_id(), hash);
    }

    #[test]
    fn test_display() {
        let storage = RemoteStorage::new("bucket", 0xFF, 1024);
        assert_eq!(format!("{}", storage), "obj://bucket/00000000000000ff");
    }
}
