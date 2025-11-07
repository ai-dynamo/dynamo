// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clean, minimal storage API for v2 block manager.
//!
//! This module provides a simplified storage abstraction with:
//! - Single trait for type erasure (`MemoryRegion`)
//! - Concrete storage types (no trait implementations required)
//! - Composition-based NIXL registration via `NixlRegistered<T>` wrapper
//! - RAII with proper drop ordering (registration handle drops before memory)

pub mod actions;

mod device;
mod disk;
mod pinned;
mod registered;
mod system;
mod torch;

#[cfg(test)]
mod tests;

pub use device::DeviceStorage;
pub use disk::DiskStorage;
pub use pinned::PinnedStorage;
pub use registered::{
    NixlCompatible, NixlDescriptor, NixlRegistered, RegisteredView, register_with_nixl,
};
pub use system::SystemStorage;
pub use torch::{TorchDevice, TorchTensor};

use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

/// Result type for storage operations.
pub type Result<T> = std::result::Result<T, StorageError>;

/// Errors that can occur during storage operations.
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("allocation failed: {0}")]
    AllocationFailed(String),

    #[error("registration failed: {0}")]
    RegistrationFailed(String),

    #[error("operation failed: {0}")]
    OperationFailed(String),

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    // #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),

    #[error("NIXL error: {0}")]
    Nixl(#[from] nixl_sys::NixlError),
}

/// Storage type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageKind {
    /// System memory (malloc)
    System,

    /// CUDA pinned host memory
    // #[cfg(feature = "cuda")]
    Pinned,

    /// CUDA device memory with device ID
    // #[cfg(feature = "cuda")]
    Device(u32),

    /// Disk-backed memory (mmap)
    Disk(u64),
}

/// Core trait for memory regions that can be type-erased.
///
/// This is the only trait in the storage API. Concrete storage types
/// implement this trait to enable type erasure via `Arc<dyn MemoryRegion>`.
pub trait MemoryRegion: Send + Sync + fmt::Debug {
    /// Base address of the memory region.
    fn addr(&self) -> usize;

    /// Size of the memory region in bytes.
    fn size(&self) -> usize;

    /// Type of storage backing this region.
    fn storage_kind(&self) -> StorageKind;

    /// Enable downcasting to concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Get the NIXL descriptor for this memory region.
    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

/// Type-erased memory region for use in layouts.
pub type OwnedMemoryRegion = Arc<dyn MemoryRegion>;

/// Helper function to convert concrete storage to type-erased form.
pub fn erase_storage<S: MemoryRegion + 'static>(storage: S) -> OwnedMemoryRegion {
    Arc::new(storage)
}

/// Simple memory region descriptor.
#[derive(Debug)]
pub struct OffsetMemoryRegion {
    base: OwnedMemoryRegion,
    offset: usize,
    len: usize,
}

impl OffsetMemoryRegion {
    /// Create a new offset view into an existing memory region.
    ///
    /// Returns an error if the offset and length exceed the bounds of the base region.
    pub fn new(base: OwnedMemoryRegion, offset: usize, len: usize) -> Result<Self> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| StorageError::Unsupported("offset overflow".into()))?;
        if end > base.size() {
            return Err(StorageError::Unsupported(
                "offset region exceeds base allocation bounds".into(),
            ));
        }
        Ok(Self { base, offset, len })
    }

    /// Get the offset relative to the base mapping.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the length of the offset region.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the offset region is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Access the underlying base region.
    pub fn base(&self) -> &OwnedMemoryRegion {
        &self.base
    }
}

impl MemoryRegion for OffsetMemoryRegion {
    fn addr(&self) -> usize {
        self.base.addr() + self.offset
    }

    fn size(&self) -> usize {
        self.len
    }

    fn storage_kind(&self) -> StorageKind {
        self.base.storage_kind()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryDescriptor {
    pub addr: usize,
    pub size: usize,
}

impl MemoryDescriptor {
    pub fn new(addr: usize, size: usize) -> Self {
        Self { addr, size }
    }

    #[inline]
    pub fn addr(&self) -> usize {
        self.addr
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
}

impl actions::Slice for MemoryDescriptor {
    fn as_slice(&self) -> Result<&[u8]> {
        Ok(unsafe { std::slice::from_raw_parts(self.addr as *const u8, self.size) })
    }
}
