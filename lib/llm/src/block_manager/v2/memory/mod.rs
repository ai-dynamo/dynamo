// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Re-export memory types from dynamo-memory with backwards-compatible names.
//!
//! This module previously contained its own storage implementations, which have
//! been consolidated into the `dynamo-memory` crate. Types are re-exported here
//! with compatibility aliases to preserve the existing API.

// Re-export actions module from dynamo-memory
pub use dynamo_memory::actions;

// Keep local torch types (unique to block_manager)
mod torch;
pub use torch::{TorchDevice, TorchTensor};

// Keep tests
#[cfg(test)]
mod tests;

// === Core types with name aliases ===

/// The core trait (was `MemoryRegion` here, now `MemoryDescriptor` in dynamo-memory).
pub use dynamo_memory::MemoryDescriptor as MemoryRegion;

/// The simple descriptor struct (was `MemoryDescriptor` here, now `MemoryRegion` in dynamo-memory).
pub use dynamo_memory::MemoryRegion as MemoryDescriptor;

// === Storage types (same names) ===
pub use dynamo_memory::{
    DeviceStorage, DiskStorage, PinnedStorage, StorageError, StorageKind, SystemStorage,
};

// === NIXL types ===
pub use dynamo_memory::nixl::{
    NixlCompatible, NixlDescriptor, NixlRegistered, RegisteredView, register_with_nixl,
};

// === Compatibility aliases ===

/// Result type for storage operations.
pub type Result<T> = std::result::Result<T, StorageError>;

/// Type-erased memory region for use in layouts.
pub type OwnedMemoryRegion = std::sync::Arc<dyn MemoryRegion>;

/// Helper function to convert concrete storage to type-erased form.
pub fn erase_storage<S: MemoryRegion + 'static>(storage: S) -> OwnedMemoryRegion {
    std::sync::Arc::new(storage)
}

/// An offset view into an existing memory region.
///
/// This wraps an `OwnedMemoryRegion` with an offset and length to represent
/// a sub-region of the original allocation.
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}
