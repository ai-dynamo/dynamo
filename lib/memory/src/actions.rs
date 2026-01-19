// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Storage actions.

use super::{MemoryDescription, StorageError};

/// Extension trait for storage types that support memory setting operations
pub trait Memset: MemoryDescription {
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
pub trait Slice: MemoryDescription + 'static {
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

pub trait SliceMut: MemoryDescription + 'static {
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
