// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    Any, Buffer, MemoryDescription, Result, StorageError, StorageKind, nixl::NixlDescriptor,
};

/// An [`OffsetBuffer`] is a new [`Buffer`]-like object that represents a sub-region (still contiguous)
/// within an existing [`Buffer`].
#[derive(Clone)]
pub struct OffsetBuffer {
    base: Buffer,
    offset: usize,
    size: usize,
}

impl std::fmt::Debug for OffsetBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OffsetBuffer")
            .field("base", &self.base)
            .field("offset", &self.offset)
            .field("size", &self.size)
            .finish()
    }
}

impl OffsetBuffer {
    /// Create a new offset view into an existing memory region.
    ///
    /// Returns an error if the offset and length exceed the bounds of the base region.
    pub fn new(base: Buffer, offset: usize, size: usize) -> Result<Self> {
        let end = offset
            .checked_add(size)
            .ok_or_else(|| StorageError::Unsupported("offset overflow".into()))?;
        if end > base.size() {
            return Err(StorageError::Unsupported(
                "offset region exceeds base allocation bounds".into(),
            ));
        }
        Ok(Self { base, offset, size })
    }

    /// Get the offset relative to the base mapping.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Access the underlying base region.
    pub fn base(&self) -> &Buffer {
        &self.base
    }
}

impl MemoryDescription for OffsetBuffer {
    fn addr(&self) -> usize {
        self.base.addr() + self.offset
    }

    fn size(&self) -> usize {
        self.size
    }

    fn storage_kind(&self) -> StorageKind {
        self.base.storage_kind()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        let mut descriptor = self.base.nixl_descriptor()?;
        descriptor.addr = self.addr() as u64;
        descriptor.size = self.size();
        Some(descriptor)
    }
}
