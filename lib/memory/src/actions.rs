// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Storage actions.

use super::{MemoryDescriptor, StorageError};

/// Extension trait for storage types that support memory setting operations
pub trait Memset: MemoryDescriptor {
    /// Sets a region of memory to a specific value
    ///
    /// # Arguments
    /// * `value` - The value to set (will be truncated to u8)
    /// * `offset` - Offset in bytes from the start of the storage
    /// * `size` - Number of bytes to set
    ///
    /// # Safety
    /// The caller must ensure:
    /// - offset + size <= self.size()
    /// - No other references exist to the memory region being set
    fn memset(&mut self, value: u8, offset: usize, size: usize) -> Result<(), StorageError>;
}

/// Extension trait for storage types that support slicing operations
pub trait Slice: MemoryDescriptor + 'static {
    /// Returns an immutable byte slice view of the entire storage region
    ///
    /// # Safety
    /// This is an unsafe method. The caller must ensure:
    /// - The memory region remains valid for the lifetime of the returned slice
    /// - The memory region is properly initialized
    /// - No concurrent mutable access occurs while the slice is in use
    /// - The memory backing this storage remains valid (implementors with owned
    ///   memory satisfy this, but care must be taken with unowned memory regions)
    unsafe fn as_slice(&self) -> Result<&[u8], StorageError>;

    /// Returns an immutable byte slice view of a subregion
    ///
    /// # Arguments
    /// * `offset` - Offset in bytes from the start of the storage
    /// * `len` - Number of bytes to slice
    ///
    /// # Safety
    /// The caller must ensure:
    /// - offset + len <= self.size()
    /// - The memory region is valid and initialized
    /// - No concurrent mutable access occurs while the slice is in use
    fn slice(&self, offset: usize, len: usize) -> Result<&[u8], StorageError> {
        // SAFETY: Caller guarantees memory validity per trait's safety contract
        let slice = unsafe { self.as_slice()? };

        // validate offset and len
        if offset.saturating_add(len) > slice.len() {
            return Err(StorageError::Unsupported("slice out of bounds".into()));
        }

        slice
            .get(offset..offset.saturating_add(len))
            .ok_or_else(|| StorageError::Unsupported("slice out of bounds".into()))
    }

    /// Returns a typed immutable slice view of the entire storage region
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The memory region is valid and initialized
    /// - The memory is properly aligned for type T
    /// - The size is a multiple of `size_of::<T>()`
    /// - No concurrent mutable access occurs while the slice is in use
    /// - The data represents valid values of type T
    fn as_slice_typed<T: Sized>(&self) -> Result<&[T], StorageError> {
        // SAFETY: Caller guarantees memory validity per trait's safety contract
        let bytes = unsafe { self.as_slice()? };
        let ptr = bytes.as_ptr() as *const T;
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            return Err(StorageError::Unsupported(
                "zero-sized types are not supported".into(),
            ));
        }
        let len = bytes.len() / elem_size;

        if !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()) {
            return Err(StorageError::Unsupported(format!(
                "memory not aligned for type (required alignment: {})",
                std::mem::align_of::<T>()
            )));
        }

        if bytes.len() % elem_size != 0 {
            return Err(StorageError::Unsupported(format!(
                "size {} is not a multiple of type size {}",
                bytes.len(),
                elem_size
            )));
        }

        // SAFETY: Caller guarantees memory is valid, aligned, and properly initialized for T
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Returns a typed immutable slice view of a subregion
    ///
    /// # Arguments
    /// * `offset` - Offset in bytes from the start of the storage
    /// * `len` - Number of elements of type T to slice
    ///
    /// # Safety
    /// The caller must ensure:
    /// - offset + (len * size_of::<T>()) <= self.size()
    /// - offset is properly aligned for type T
    /// - The memory region is valid and initialized
    /// - No concurrent mutable access occurs while the slice is in use
    /// - The data represents valid values of type T
    fn slice_typed<T: Sized>(&self, offset: usize, len: usize) -> Result<&[T], StorageError> {
        let type_size = std::mem::size_of::<T>();
        let byte_len = len
            .checked_mul(type_size)
            .ok_or_else(|| StorageError::Unsupported("length overflow".into()))?;

        let bytes = self.slice(offset, byte_len)?;
        let ptr = bytes.as_ptr() as *const T;

        if !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()) {
            return Err(StorageError::Unsupported(format!(
                "memory not aligned for type (required alignment: {})",
                std::mem::align_of::<T>()
            )));
        }

        // SAFETY: Caller guarantees memory is valid, aligned, and properly initialized for T
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }
}

/// Extension trait for storage types that support mutable slicing operations.
pub trait SliceMut: MemoryDescriptor + 'static {
    /// Returns a mutable byte slice view of the entire storage region
    ///
    /// # Safety
    /// This is an unsafe method. The caller must ensure:
    /// - The memory region remains valid for the lifetime of the returned slice
    /// - The memory region is valid and accessible
    /// - No other references (mutable or immutable) exist to this memory region
    /// - The memory backing this storage remains valid (implementors with owned
    ///   memory satisfy this, but care must be taken with unowned memory regions)
    unsafe fn as_slice_mut(&mut self) -> Result<&mut [u8], StorageError>;

    /// Returns a mutable byte slice view of a subregion
    ///
    /// # Arguments
    /// * `offset` - Offset in bytes from the start of the storage
    /// * `len` - Number of bytes to slice
    ///
    /// # Safety
    /// The caller must ensure:
    /// - offset + len <= self.size()
    /// - The memory region is valid
    /// - No other references (mutable or immutable) exist to this memory region
    fn slice_mut(&mut self, offset: usize, len: usize) -> Result<&mut [u8], StorageError> {
        // SAFETY: Caller guarantees memory validity per trait's safety contract
        let slice = unsafe { self.as_slice_mut()? };

        // validate offset and len
        if offset.saturating_add(len) > slice.len() {
            return Err(StorageError::Unsupported("slice out of bounds".into()));
        }

        slice
            .get_mut(offset..offset.saturating_add(len))
            .ok_or_else(|| StorageError::Unsupported("slice out of bounds".into()))
    }

    /// Returns a typed mutable slice view of the entire storage region
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The memory region is valid
    /// - The memory is properly aligned for type T
    /// - The size is a multiple of `size_of::<T>()`
    /// - No other references (mutable or immutable) exist to this memory region
    fn as_slice_typed_mut<T: Sized>(&mut self) -> Result<&mut [T], StorageError> {
        // SAFETY: Caller guarantees memory validity per trait's safety contract
        let bytes = unsafe { self.as_slice_mut()? };
        let ptr = bytes.as_mut_ptr() as *mut T;
        let len = bytes.len() / std::mem::size_of::<T>();

        if !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()) {
            return Err(StorageError::Unsupported(format!(
                "memory not aligned for type (required alignment: {})",
                std::mem::align_of::<T>()
            )));
        }

        if bytes.len() % std::mem::size_of::<T>() != 0 {
            return Err(StorageError::Unsupported(format!(
                "size {} is not a multiple of type size {}",
                bytes.len(),
                std::mem::size_of::<T>()
            )));
        }

        // SAFETY: Caller guarantees memory is valid, aligned, and no aliasing
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
    }

    /// Returns a typed mutable slice view of a subregion
    ///
    /// # Arguments
    /// * `offset` - Offset in bytes from the start of the storage
    /// * `len` - Number of elements of type T to slice
    ///
    /// # Safety
    /// The caller must ensure:
    /// - offset + (len * size_of::<T>()) <= self.size()
    /// - offset is properly aligned for type T
    /// - The memory region is valid
    /// - No other references (mutable or immutable) exist to this memory region
    fn slice_typed_mut<T: Sized>(
        &mut self,
        offset: usize,
        len: usize,
    ) -> Result<&mut [T], StorageError> {
        let type_size = std::mem::size_of::<T>();
        let byte_len = len
            .checked_mul(type_size)
            .ok_or_else(|| StorageError::Unsupported("length overflow".into()))?;

        let bytes = self.slice_mut(offset, byte_len)?;
        let ptr = bytes.as_mut_ptr() as *mut T;

        if !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()) {
            return Err(StorageError::Unsupported(format!(
                "memory not aligned for type (required alignment: {})",
                std::mem::align_of::<T>()
            )));
        }

        // SAFETY: Caller guarantees memory is valid, aligned, and no aliasing
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SystemStorage;

    // Helper to create a test storage
    fn create_storage(size: usize) -> SystemStorage {
        SystemStorage::new(size).expect("allocation failed")
    }

    // ========== Memset tests ==========

    #[test]
    fn test_memset_full_region() {
        let mut storage = create_storage(1024);
        storage
            .memset(0xAB, 0, 1024)
            .expect("memset should succeed");

        let slice = unsafe { storage.as_slice().expect("as_slice should succeed") };
        assert!(slice.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn test_memset_partial_region() {
        let mut storage = create_storage(1024);
        // First fill with 0x00
        storage
            .memset(0x00, 0, 1024)
            .expect("memset should succeed");
        // Then fill middle region with 0xFF
        storage
            .memset(0xFF, 100, 200)
            .expect("memset should succeed");

        let slice = unsafe { storage.as_slice().expect("as_slice should succeed") };
        // Check before region
        assert!(slice[..100].iter().all(|&b| b == 0x00));
        // Check filled region
        assert!(slice[100..300].iter().all(|&b| b == 0xFF));
        // Check after region
        assert!(slice[300..].iter().all(|&b| b == 0x00));
    }

    #[test]
    fn test_memset_at_end() {
        let mut storage = create_storage(1024);
        // Fill the last 100 bytes
        storage
            .memset(0x42, 924, 100)
            .expect("memset should succeed");

        let slice = unsafe { storage.as_slice().expect("as_slice should succeed") };
        assert!(slice[924..].iter().all(|&b| b == 0x42));
    }

    #[test]
    fn test_memset_zero_size() {
        let mut storage = create_storage(1024);
        // Zero-size memset should succeed (no-op)
        storage
            .memset(0xFF, 500, 0)
            .expect("zero-size memset should succeed");
    }

    #[test]
    fn test_memset_out_of_bounds() {
        let mut storage = create_storage(1024);
        // Try to write beyond the storage
        let result = storage.memset(0xFF, 900, 200);
        assert!(result.is_err());
    }

    #[test]
    fn test_memset_offset_overflow() {
        let mut storage = create_storage(1024);
        // offset + size would overflow
        let result = storage.memset(0xFF, usize::MAX, 1);
        assert!(result.is_err());
    }

    // ========== Slice tests ==========

    #[test]
    fn test_as_slice_full() {
        let mut storage = create_storage(1024);
        storage
            .memset(0xCD, 0, 1024)
            .expect("memset should succeed");

        let slice = unsafe { storage.as_slice().expect("as_slice should succeed") };
        assert_eq!(slice.len(), 1024);
        assert!(slice.iter().all(|&b| b == 0xCD));
    }

    #[test]
    fn test_slice_partial() {
        let mut storage = create_storage(1024);
        storage
            .memset(0x00, 0, 1024)
            .expect("memset should succeed");
        storage
            .memset(0xAA, 100, 50)
            .expect("memset should succeed");

        let partial = storage.slice(100, 50).expect("slice should succeed");
        assert_eq!(partial.len(), 50);
        assert!(partial.iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn test_slice_at_start() {
        let storage = create_storage(1024);
        let slice = storage.slice(0, 100).expect("slice should succeed");
        assert_eq!(slice.len(), 100);
    }

    #[test]
    fn test_slice_at_end() {
        let storage = create_storage(1024);
        let slice = storage.slice(924, 100).expect("slice should succeed");
        assert_eq!(slice.len(), 100);
    }

    #[test]
    fn test_slice_zero_length() {
        let storage = create_storage(1024);
        let slice = storage
            .slice(500, 0)
            .expect("zero-length slice should succeed");
        assert!(slice.is_empty());
    }

    #[test]
    fn test_slice_out_of_bounds() {
        let storage = create_storage(1024);
        let result = storage.slice(900, 200);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_offset_overflow() {
        let storage = create_storage(1024);
        // offset + len would overflow when using saturating_add
        let result = storage.slice(usize::MAX, 1);
        assert!(result.is_err());
    }

    // ========== Typed slice tests ==========

    #[test]
    fn test_as_slice_typed_u32() {
        let mut storage = create_storage(1024);
        // Fill with known pattern
        storage
            .memset(0x00, 0, 1024)
            .expect("memset should succeed");

        let typed: &[u32] = storage
            .as_slice_typed()
            .expect("typed slice should succeed");
        assert_eq!(typed.len(), 256); // 1024 / 4
        assert!(typed.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_as_slice_typed_u64() {
        let storage = create_storage(1024);
        let typed: &[u64] = storage
            .as_slice_typed()
            .expect("typed slice should succeed");
        assert_eq!(typed.len(), 128); // 1024 / 8
    }

    #[test]
    fn test_slice_typed_partial() {
        let mut storage = create_storage(1024);
        storage
            .memset(0x00, 0, 1024)
            .expect("memset should succeed");

        // Slice 10 u32 elements starting at offset 0
        let typed: &[u32] = storage
            .slice_typed(0, 10)
            .expect("typed slice should succeed");
        assert_eq!(typed.len(), 10);
    }

    #[test]
    fn test_slice_typed_with_offset() {
        let storage = create_storage(1024);
        // Slice starting at offset 64 (aligned for u64)
        let typed: &[u64] = storage
            .slice_typed(64, 5)
            .expect("typed slice should succeed");
        assert_eq!(typed.len(), 5);
    }

    #[test]
    fn test_as_slice_typed_zst_error() {
        let storage = create_storage(1024);
        // Zero-sized types should fail
        let result: Result<&[()], _> = storage.as_slice_typed();
        assert!(result.is_err());
    }

    #[test]
    fn test_as_slice_typed_size_not_multiple() {
        // Create storage with size not divisible by 4
        let storage = create_storage(1023);
        let result: Result<&[u32], _> = storage.as_slice_typed();
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_typed_length_overflow() {
        let storage = create_storage(1024);
        // len * size_of::<u64>() would overflow
        let result: Result<&[u64], _> = storage.slice_typed(0, usize::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_typed_out_of_bounds() {
        let storage = create_storage(1024);
        // Request more elements than available
        let result: Result<&[u64], _> = storage.slice_typed(0, 200);
        assert!(result.is_err());
    }
}
