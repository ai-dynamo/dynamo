// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    Any, Buffer, MemoryDescriptor, Result, StorageError, StorageKind, nixl::NixlDescriptor,
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

    /// Creates an offset buffer from an absolute address within the base region.
    pub fn from_inner_address(base: Buffer, address: usize, size: usize) -> Result<Self> {
        // Use checked arithmetic to prevent overflow
        let end = address
            .checked_add(size)
            .ok_or_else(|| StorageError::Unsupported("address + size overflow".into()))?;
        let base_end = base
            .addr()
            .checked_add(base.size())
            .ok_or_else(|| StorageError::Unsupported("base address + size overflow".into()))?;

        // Verify address is within the base region
        if address < base.addr() || end > base_end {
            return Err(StorageError::Unsupported("address out of bounds".into()));
        }

        let offset = address - base.addr();
        Self::new(base, offset, size)
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

impl MemoryDescriptor for OffsetBuffer {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SystemStorage;

    fn create_test_buffer(size: usize) -> Buffer {
        Buffer::new(SystemStorage::new(size).expect("allocation failed"))
    }

    #[test]
    fn test_offset_buffer_new_valid() {
        let base = create_test_buffer(1024);
        let offset_buf = OffsetBuffer::new(base, 100, 200).expect("should succeed");
        assert_eq!(offset_buf.offset(), 100);
        assert_eq!(offset_buf.size(), 200);
    }

    #[test]
    fn test_offset_buffer_new_zero_offset() {
        let base = create_test_buffer(1024);
        let offset_buf = OffsetBuffer::new(base.clone(), 0, 1024).expect("should succeed");
        assert_eq!(offset_buf.offset(), 0);
        assert_eq!(offset_buf.size(), 1024);
        assert_eq!(offset_buf.addr(), base.addr());
    }

    #[test]
    fn test_offset_buffer_new_at_end() {
        let base = create_test_buffer(1024);
        // Offset at exact end with zero size should succeed
        let offset_buf = OffsetBuffer::new(base, 1024, 0).expect("should succeed");
        assert_eq!(offset_buf.offset(), 1024);
        assert_eq!(offset_buf.size(), 0);
    }

    #[test]
    fn test_offset_buffer_new_invalid_offset() {
        let base = create_test_buffer(1024);
        // Offset beyond bounds
        let result = OffsetBuffer::new(base, 1025, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_offset_buffer_new_invalid_size() {
        let base = create_test_buffer(1024);
        // Size exceeds remaining space
        let result = OffsetBuffer::new(base, 100, 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_offset_buffer_new_size_overflow() {
        let base = create_test_buffer(1024);
        // offset + size would overflow usize
        let result = OffsetBuffer::new(base, usize::MAX, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_offset_buffer_from_inner_address_valid() {
        let base = create_test_buffer(1024);
        let base_addr = base.addr();
        let offset_buf =
            OffsetBuffer::from_inner_address(base, base_addr + 100, 200).expect("should succeed");
        assert_eq!(offset_buf.offset(), 100);
        assert_eq!(offset_buf.size(), 200);
    }

    #[test]
    fn test_offset_buffer_from_inner_address_at_start() {
        let base = create_test_buffer(1024);
        let base_addr = base.addr();
        let offset_buf = OffsetBuffer::from_inner_address(base.clone(), base_addr, 1024)
            .expect("should succeed");
        assert_eq!(offset_buf.offset(), 0);
        assert_eq!(offset_buf.addr(), base.addr());
    }

    #[test]
    fn test_offset_buffer_from_inner_address_overflow() {
        let base = create_test_buffer(1024);
        // address + size would overflow
        let result = OffsetBuffer::from_inner_address(base, usize::MAX, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_offset_buffer_from_inner_address_out_of_bounds_before() {
        let base = create_test_buffer(1024);
        let base_addr = base.addr();
        // Address before base region
        let result = OffsetBuffer::from_inner_address(base, base_addr.saturating_sub(1), 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_offset_buffer_from_inner_address_out_of_bounds_after() {
        let base = create_test_buffer(1024);
        let base_addr = base.addr();
        // End address beyond base region
        let result = OffsetBuffer::from_inner_address(base, base_addr + 900, 200);
        assert!(result.is_err());
    }

    #[test]
    fn test_offset_buffer_accessors() {
        let base = create_test_buffer(1024);
        let base_addr = base.addr();
        let offset_buf = OffsetBuffer::new(base, 256, 512).expect("should succeed");

        assert_eq!(offset_buf.offset(), 256);
        assert_eq!(offset_buf.base().addr(), base_addr);
        assert_eq!(offset_buf.base().size(), 1024);
    }

    #[test]
    fn test_offset_buffer_memory_descriptor_addr() {
        let base = create_test_buffer(1024);
        let base_addr = base.addr();
        let offset_buf = OffsetBuffer::new(base, 100, 200).expect("should succeed");

        // addr() should return base_addr + offset
        assert_eq!(offset_buf.addr(), base_addr + 100);
    }

    #[test]
    fn test_offset_buffer_memory_descriptor_size() {
        let base = create_test_buffer(1024);
        let offset_buf = OffsetBuffer::new(base, 100, 200).expect("should succeed");
        assert_eq!(offset_buf.size(), 200);
    }

    #[test]
    fn test_offset_buffer_memory_descriptor_storage_kind() {
        let base = create_test_buffer(1024);
        let base_kind = base.storage_kind();
        let offset_buf = OffsetBuffer::new(base, 100, 200).expect("should succeed");

        // storage_kind should match the base
        assert_eq!(offset_buf.storage_kind(), base_kind);
    }

    #[test]
    fn test_offset_buffer_as_any() {
        let base = create_test_buffer(1024);
        let offset_buf = OffsetBuffer::new(base, 100, 200).expect("should succeed");

        // Should be able to downcast to OffsetBuffer
        let any_ref = offset_buf.as_any();
        assert!(any_ref.downcast_ref::<OffsetBuffer>().is_some());
    }

    #[test]
    fn test_offset_buffer_clone() {
        let base = create_test_buffer(1024);
        let offset_buf = OffsetBuffer::new(base, 100, 200).expect("should succeed");
        let cloned = offset_buf.clone();

        assert_eq!(offset_buf.addr(), cloned.addr());
        assert_eq!(offset_buf.size(), cloned.size());
        assert_eq!(offset_buf.offset(), cloned.offset());
    }

    #[test]
    fn test_offset_buffer_debug() {
        let base = create_test_buffer(1024);
        let offset_buf = OffsetBuffer::new(base, 100, 200).expect("should succeed");
        let debug_str = format!("{:?}", offset_buf);

        assert!(debug_str.contains("OffsetBuffer"));
        assert!(debug_str.contains("offset"));
        assert!(debug_str.contains("size"));
    }

    #[test]
    fn test_offset_buffer_nixl_descriptor_none() {
        // SystemStorage doesn't have a NIXL descriptor
        let base = create_test_buffer(1024);
        let offset_buf = OffsetBuffer::new(base, 100, 200).expect("should succeed");

        // Should return None since base has no NIXL descriptor
        assert!(offset_buf.nixl_descriptor().is_none());
    }
}
