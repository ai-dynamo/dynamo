// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Object storage memory region
//!
//! This module provides memory-mapped access to object storage via NIXL's OBJ plugin.

use super::{MemoryDescriptor, Result, StorageKind, nixl::NixlDescriptor};
use std::any::Any;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

use super::ObjectKey;
/// Object storage backed by NIXL's OBJ plugin
///
/// This represents a region of memory that is backed by object storage.
/// NIXL's OBJ plugin handles the actual transfers to/from Object.
pub struct ObjectStorage {
    /// Base address (virtual, managed by NIXL)
    addr: usize,
    /// Size of the object storage region in bytes
    size: usize,
    /// Object key (u64 identifier used by NIXL for object identification)
    ///
    /// This is a numeric identifier that uniquely identifies the object within the bucket.
    /// NIXL's OBJ backend uses this as the device_id for transfer descriptors.
    key: ObjectKey,
    /// Object bucket name
    bucket: String,
}

impl ObjectStorage {
    /// Create a new object storage region
    ///
    /// # Arguments
    /// * `bucket` - Object Bucket Name
    /// * `key` - Object Key (u64 numeric identifier for the object)
    /// * `size` - Size of the region in bytes
    ///
    /// # Returns
    /// A new ObjectStorage instance
    ///
    /// # Example
    /// ```ignore
    /// use dynamo_memory::ObjectStorage;
    ///
    /// let storage = ObjectStorage::new("my-bucket", 1234567890u64, 4096 * 128)?;
    /// ```
    pub fn new(bucket: impl Into<String>, key: ObjectKey, size: usize) -> Result<Self> {
        Ok(Self {
            addr: 0,
            bucket: bucket.into(),
            key,
            size,
        })
    }

    /// Set the byte offset for reading from this object.
    ///
    /// For multipart uploads, each worker can read a different portion:
    /// ```ignore
    /// // Worker 1: reads bytes 0-5MB
    /// ObjectStorage::new("bucket", key, 5MB)?.with_offset(0);
    ///
    /// // Worker 2: reads bytes 5-10MB
    /// ObjectStorage::new("bucket", key, 5MB)?.with_offset(5 * 1024 * 1024);
    /// ```
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.addr = offset;
        self
    }

    pub fn key(&self) -> ObjectKey {
        self.key
    }

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
        StorageKind::Object(self.key() as u128)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        // Return None so that register_with_nixl actually calls agent.register_memory()
        // The actual registration parameters come from nixl_params()
        None
    }
}

// Support for NIXL registration
impl super::nixl::NixlCompatible for ObjectStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        // device_id = bucket hash (register once per bucket)
        // Object keys are passed via opt_args.customParam per transfer
        // addr = byte offset for S3 Range requests (set via with_offset())
        let mut hasher = DefaultHasher::new();
        self.bucket().hash(&mut hasher);
        let bucket_id = hasher.finish();

        (
            self.addr as *const u8,  // Byte offset for S3 Range header
            self.size,
            nixl_sys::MemType::Object,
            bucket_id,
        )
    }
}

impl fmt::Debug for ObjectStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("ObjectStorage");
        debug
            .field("addr", &format!("0x{:x}", self.addr))
            .field("size", &self.size)
            .field("key", &format!("{:x}", self.key))
            .field("bucket", &self.bucket)
            .finish()
    }
}
