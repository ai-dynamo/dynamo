// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Object storage memory region
//!
//! This module provides memory-mapped access to object storage via NIXL's OBJ plugin.

use super::{MemoryRegion, Result, StorageKind};
use std::any::Any;
use std::fmt;

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
    key: u64,
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
    pub fn new(bucket: impl Into<String>, key: u64, size: usize) -> Result<Self> {

        Ok(Self { addr: 0, bucket: bucket.into(), key, size })
    }

    pub fn key(&self) -> u64 {
        self.key
    }

    pub fn bucket(&self) -> &str {
        &self.bucket
    }
}

impl MemoryRegion for ObjectStorage {
    fn addr(&self) -> usize {
        self.addr
    }

    fn size(&self) -> usize {
        self.size
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::Object(self.key())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<crate::block_manager::v2::memory::NixlDescriptor> {
        Some(crate::block_manager::v2::memory::NixlDescriptor {
            addr: self.addr() as u64,
            size: self.size(),
            mem_type: nixl_sys::MemType::Object,
            device_id: self.key(),
        })
    }
}

impl super::registered::NixlCompatible for ObjectStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        (
            std::ptr::null(),
            self.size,
            nixl_sys::MemType::Object,
            self.key(),
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
