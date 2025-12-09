// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Object storage memory region
//!
//! This module provides memory-mapped access to object storage via NIXL's OBJ plugin.

use super::{MemoryDescriptor, Result, StorageKind, nixl::NixlDescriptor};
use std::any::Any;
use std::fmt;

/// Object storage backed by NIXL's OBJ plugin
///
/// This represents a region of memory that is backed by object storage.
/// NIXL's OBJ plugin handles the actual transfers to/from Object.
pub struct ObjectStorage {
    /// Base address (byte offset within object for partial reads)
    addr: usize,
    /// Size of the object storage region in bytes
    size: usize,
    /// Object key as string
    key: String,
    /// Device ID derived from the object key (xxhash64 of key bytes)
    device_id: u64,
    /// Object bucket name
    bucket: String,
}

impl ObjectStorage {
    /// Create a new object storage region
    ///
    /// # Arguments
    /// * `bucket` - Object Bucket Name
    /// * `key` - Object Key as string (e.g., "{sequence_hash}_{tp_rank}")
    /// * `size` - Size of the region in bytes
    ///
    /// # Example
    /// ```ignore
    /// use dynamo_memory::ObjectStorage;
    ///
    /// // Key format: "{sequence_hash}_{tp_rank}"
    /// let storage = ObjectStorage::new("my-bucket", "12345_0", 4096)?;
    /// ```
    pub fn new(bucket: impl Into<String>, key: impl Into<String>, size: usize) -> Result<Self> {
        let key = key.into();
        let device_id = xxhash_rust::xxh3::xxh3_64(key.as_bytes());
        Ok(Self {
            addr: 0,
            bucket: bucket.into(),
            key,
            device_id,
            size,
        })
    }

    /// Get the device_id for a given object key.
    ///
    /// This is useful when building transfer descriptor lists that need
    /// to reference the same device_id used during registration.
    pub fn device_id(key: &str) -> u64 {
        xxhash_rust::xxh3::xxh3_64(key.as_bytes())
    }

    /// Set the byte offset for reading from this object.
    ///
    /// For multipart uploads, each worker can read a different portion:
    /// ```ignore
    /// // Worker 1: reads bytes 0-5MB
    /// ObjectStorage::new("bucket", "key_0", 5MB)?.with_offset(0);
    ///
    /// // Worker 2: reads bytes 5-10MB
    /// ObjectStorage::new("bucket", "key_0", 5MB)?.with_offset(5 * 1024 * 1024);
    /// ```
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.addr = offset;
        self
    }

    /// Get the object key
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Get the bucket name
    pub fn bucket(&self) -> &str {
        &self.bucket
    }
}

impl MemoryDescriptor for ObjectStorage {
    fn addr(&self) -> usize {
        self.addr
    }

    fn size(&self) -> usize {
        self.size
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::Object(self.device_id as u128)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

// Support for NIXL registration via NixlCompatible (legacy)
impl super::nixl::NixlCompatible for ObjectStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        (
            self.addr as *const u8,
            self.size,
            nixl_sys::MemType::Object,
            self.device_id,
        )
    }
}

// Direct NIXL descriptor support with metadata
impl nixl_sys::MemoryRegion for ObjectStorage {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.addr as *const u8
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
        self.device_id
    }

    fn metadata(&self) -> Vec<u8> {
        // OBJ backend expects UTF-8 string as object key
        self.key.clone().into_bytes()
    }
}

impl fmt::Debug for ObjectStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObjectStorage")
            .field("addr", &format!("0x{:x}", self.addr))
            .field("size", &self.size)
            .field("key", &self.key)
            .field("device_id", &self.device_id)
            .field("bucket", &self.bucket)
            .finish()
    }
}
