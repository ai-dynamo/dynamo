// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clean, minimal storage API for v2 block manager.
//!
//! This module provides a simplified storage abstraction with:
//! - Single trait for type erasure (`MemoryDescriptor`)
//! - Concrete storage types (no trait implementations required)
//! - Composition-based NIXL registration via `NixlRegistered<T>` wrapper
//! - RAII with proper drop ordering (registration handle drops before memory)

#![deny(missing_docs)]

pub mod actions;
pub mod arena;
pub mod nixl;
#[cfg(target_os = "linux")]
pub mod numa;

/// Offset-based buffer views into underlying storage.
pub mod offset;

/// CUDA memory pool utilities.
pub mod pool;

/// Common imports for working with memory types.
pub mod prelude;

mod device;
#[cfg(target_os = "linux")]
mod disk;
mod external;
mod pinned;
mod system;
mod tensor;

#[cfg(test)]
mod tests;

pub use arena::{ArenaAllocator, ArenaBuffer, ArenaError};
pub use device::DeviceStorage;
#[cfg(target_os = "linux")]
pub use disk::DiskStorage;
pub use external::ExternalDeviceMemory;
#[cfg(target_os = "linux")]
pub use numa::{NumaNode, is_numa_enabled};
pub use offset::OffsetBuffer;
pub use pinned::PinnedStorage;
pub use pool::{CudaMemPool, CudaMemPoolBuilder};
pub use system::SystemStorage;
pub use tensor::{TensorDescriptor, TensorDescriptorExt};

use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

/// Result type for storage operations.
pub type Result<T> = std::result::Result<T, StorageError>;

/// Core trait for memory regions that can be type-erased.
///
/// This is the only trait in the storage API. Concrete storage types
/// implement this trait to enable type erasure via `Arc<dyn MemoryDescriptor>`.
pub trait MemoryDescriptor: Send + Sync + fmt::Debug {
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

/// Errors that can occur during storage operations.
#[derive(Debug, Error)]
#[allow(missing_docs)]
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

impl StorageKind {
    /// Returns the CUDA device index if this is device memory.
    pub fn cuda_device_index(&self) -> Option<u32> {
        match self {
            StorageKind::Device(idx) => Some(*idx),
            _ => None,
        }
    }

    /// Returns true if this is CUDA device memory.
    pub fn is_cuda(&self) -> bool {
        matches!(self, StorageKind::Device(_))
    }

    /// Returns true if this is system memory (malloc).
    pub fn is_system(&self) -> bool {
        matches!(self, StorageKind::System)
    }

    /// Returns true if this is CUDA pinned host memory.
    pub fn is_pinned(&self) -> bool {
        matches!(self, StorageKind::Pinned)
    }

    /// Returns true if this is disk-backed memory.
    pub fn is_disk(&self) -> bool {
        matches!(self, StorageKind::Disk(_))
    }
}

/// Type-erased memory region for use in layouts.
#[derive(Clone)]
pub struct Buffer(Arc<dyn MemoryDescriptor>);

impl Buffer {
    /// Wraps a concrete storage type into a type-erased [`Buffer`].
    ///
    /// This is the primary way to create a `Buffer` from any type that
    /// implements [`MemoryDescriptor`].
    pub fn new<S: MemoryDescriptor + 'static>(memory: S) -> Self {
        Buffer(Arc::new(memory))
    }
}

impl MemoryDescriptor for Buffer {
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
    type Target = dyn MemoryDescriptor;

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
pub fn create_buffer<S: MemoryDescriptor + 'static>(memory: S) -> Buffer {
    Buffer(Arc::new(memory))
}

impl Buffer {
    /// Create a Buffer from an existing Arc<dyn MemoryDescriptor>.
    pub fn from_arc(arc: Arc<dyn MemoryDescriptor>) -> Self {
        Buffer(arc)
    }
}

// From implementations for ergonomic Buffer creation
impl From<Arc<dyn MemoryDescriptor>> for Buffer {
    fn from(arc: Arc<dyn MemoryDescriptor>) -> Self {
        Buffer::from_arc(arc)
    }
}

impl From<Arc<dyn nixl::NixlMemory + Send + Sync>> for Buffer {
    fn from(arc: Arc<dyn nixl::NixlMemory + Send + Sync>) -> Self {
        // Arc<dyn NixlMemory> implements MemoryDescriptor, so we can wrap it
        Buffer::new(arc)
    }
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
    /// Creates a new memory region with the given base address and size.
    pub fn new(addr: usize, size: usize) -> Self {
        Self { addr, size }
    }

    /// Returns the base address of this memory region.
    #[inline]
    pub fn addr(&self) -> usize {
        self.addr
    }

    /// Returns the size of this memory region in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get a slice view of this memory region.
    ///
    /// # Safety
    /// This is unsafe because:
    /// - The caller must ensure the memory region is valid and properly initialized
    /// - The caller must ensure no mutable references exist to this memory
    /// - The caller must ensure the memory remains valid for the lifetime of the slice
    #[cfg(feature = "unsafe-slices")]
    pub unsafe fn as_slice(&self) -> Result<&[u8]> {
        if self.size == 0 {
            return Ok(&[]);
        }
        // SAFETY: Caller guarantees memory is valid
        unsafe {
            Ok(std::slice::from_raw_parts(
                self.addr as *const u8,
                self.size,
            ))
        }
    }

    /// Get a mutable slice view of this memory region.
    ///
    /// # Safety
    /// This is unsafe because:
    /// - The caller must ensure the memory region is valid and properly initialized
    /// - The caller must ensure no other references (mutable or immutable) exist to this memory
    /// - The caller must ensure the memory remains valid for the lifetime of the slice
    #[cfg(feature = "unsafe-slices")]
    pub unsafe fn as_slice_mut(&mut self) -> Result<&mut [u8]> {
        if self.size == 0 {
            return Ok(&mut []);
        }
        // SAFETY: Caller guarantees memory is valid and exclusively accessible
        unsafe {
            Ok(std::slice::from_raw_parts_mut(
                self.addr as *mut u8,
                self.size,
            ))
        }
    }
}
