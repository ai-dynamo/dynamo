// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//!
//! This module provides [`ObjectStorage`] - a NIXL-compatible memory region
//! for object storage transfers via NIXL's OBJ backend.
//!
//! ## Usage
//!
//! ```ignore
//! use dynamo_llm::block_manager::storage::ObjectStorage;
//!
//! // Create an object storage region for a specific key
//! let storage = ObjectStorage::new("my-bucket", 0x1234567890abcdef, 4096 * 128);
//!
//! // Register with NIXL agent
//! let handle = agent.register_memory(&storage, None)?;
//! ```
use std::fmt;

/// Result type for ObjectStorage operations.
pub type Result<T> = std::result::Result<T, ObjectStorageError>;

/// Error type for ObjectStorage operations.
#[derive(Debug)]
pub struct ObjectStorageError(String);

impl std::fmt::Display for ObjectStorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ObjectStorageError: {}", self.0)
    }
}

impl std::error::Error for ObjectStorageError {}

/// Object storage region for NIXL OBJ backend.
///
/// Represents a region of object storage that can be registered with NIXL
/// for transfers. The object is identified by a bucket name and a u64 key.
pub struct ObjectStorage {
    /// Size of the object storage region in bytes
    size: usize,
    /// Object key (u64 identifier used by NIXL for object identification)
    ///
    /// NIXL's OBJ backend uses this as the device_id for transfer descriptors.
    key: u64,
    /// Object bucket name
    bucket: String,
}

impl ObjectStorage {
    /// Create a new object storage region.
    ///
    /// # Arguments
    /// * `bucket` - Object bucket name
    /// * `key` - Object key (u64 numeric identifier, typically a sequence hash)
    /// * `size` - Size of the region in bytes
    ///
    /// # Returns
    /// `Result<Self>` for API consistency
    pub fn new(bucket: impl Into<String>, key: u64, size: usize) -> Result<Self> {
        Ok(Self {
            bucket: bucket.into(),
            key,
            size,
        })
    }

    /// Get the object key.
    pub fn key(&self) -> u64 {
        self.key
    }

    /// Get the bucket name.
    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    /// Get the size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

// Implement nixl_sys traits for direct registration with NIXL agent

impl nixl_sys::MemoryRegion for ObjectStorage {
    unsafe fn as_ptr(&self) -> *const u8 {
        // Object storage doesn't use direct memory pointers
        std::ptr::null()
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl nixl_sys::NixlDescriptor for ObjectStorage {
    fn mem_type(&self) -> nixl_sys::MemType {
        nixl_sys::MemType::Object
    }

    fn device_id(&self) -> u64 {
        self.key
    }
}

impl fmt::Debug for ObjectStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObjectStorage")
            .field("bucket", &self.bucket)
            .field("key", &format!("{:#018x}", self.key))
            .field("size", &self.size)
            .finish()
    }
}

// Implement core Storage trait for interoperability with block manager
impl super::Storage for ObjectStorage {
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

// Object storage is remote - accessed via NIXL transfers
impl super::Remote for ObjectStorage {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_storage_creation() {
        let storage = ObjectStorage::new("test-bucket", 0x1234, 4096).unwrap();
        assert_eq!(storage.bucket(), "test-bucket");
        assert_eq!(storage.key(), 0x1234);
        assert_eq!(storage.size(), 4096);
    }

    #[test]
    fn test_nixl_descriptor() {
        use nixl_sys::NixlDescriptor;

        let storage = ObjectStorage::new("bucket", 0xABCD, 8192).unwrap();
        assert_eq!(storage.mem_type(), nixl_sys::MemType::Object);
        assert_eq!(storage.device_id(), 0xABCD);
    }
}
