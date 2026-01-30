// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clean, minimal storage API for v2 block manager.
//!
//! This module provides a simplified storage abstraction with:
//! - Single trait for type erasure (`MemoryDescription`)
//! - Concrete storage types (no trait implementations required)
//! - Composition-based NIXL registration via `NixlRegistered<T>` wrapper
//! - RAII with proper drop ordering (registration handle drops before memory)

pub mod actions;
pub mod arena;
pub mod nixl;
pub mod offset;
pub mod pool;
pub mod prelude;

mod device;
mod disk;
mod pinned;
mod system;
mod torch;

#[cfg(test)]
mod tests;

pub use arena::{ArenaAllocator, ArenaBuffer, ArenaError};
pub use device::DeviceStorage;
pub use disk::DiskStorage;
pub use pinned::PinnedStorage;
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
/// implement this trait to enable type erasure via `Arc<dyn MemoryDescription>`.
pub trait MemoryDescription: Send + Sync + fmt::Debug {
    /// Base address of the memory region.
    fn addr(&self) -> usize;

    /// Size of the memory region in bytes.
    fn size(&self) -> usize;

    /// Type of storage backing this region.
    fn storage_kind(&self) -> StorageKind;

    /// Enable downcasting to concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Get the NIXL descriptor for this memory region.
    fn nixl_descriptor(&self) -> Option<nixl::NixlDescriptor>;
}

/// Type-erased memory region for use in layouts.
#[derive(Clone)]
pub struct Buffer(Arc<dyn MemoryDescription>);

impl MemoryDescription for Buffer {
    fn addr(&self) -> usize {
        self.0.addr()
    }
    fn size(&self) -> usize {
        self.0.size()
    }
    fn storage_kind(&self) -> StorageKind {
        self.0.storage_kind()
    }
    fn as_any(&self) -> &dyn Any {
        self.0.as_any()
    }
    fn nixl_descriptor(&self) -> Option<nixl::NixlDescriptor> {
        self.0.nixl_descriptor()
    }
}

impl std::ops::Deref for Buffer {
    type Target = dyn MemoryDescription;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("addr", &self.addr())
            .field("size", &self.size())
            .field("kind", &self.storage_kind())
            .finish()
    }
}

/// Helper function to convert concrete storage to type-erased form.
pub fn create_buffer<S: MemoryDescription + 'static>(memory: S) -> Buffer {
    Buffer(Arc::new(memory))
}

/// An unowned contiguous chunk of memory, not storage specific.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryRegion {
    /// Start address of the memory region.
    pub addr: usize,

    /// Size of the memory region in bytes.
    pub size: usize,
}

impl MemoryRegion {
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
