// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! System memory storage backed by malloc.

use super::{MemoryDescriptor, Result, StorageError, StorageKind, actions, nixl::NixlDescriptor};
use std::any::Any;
use std::ptr::NonNull;

/// System memory allocated via malloc.
#[derive(Debug)]
pub struct SystemStorage {
    ptr: NonNull<u8>,
    len: usize,
}

unsafe impl Send for SystemStorage {}
unsafe impl Sync for SystemStorage {}

impl SystemStorage {
    /// Allocate new system memory of the given size.
    pub fn new(len: usize) -> Result<Self> {
        if len == 0 {
            return Err(StorageError::AllocationFailed(
                "zero-sized allocations are not supported".into(),
            ));
        }

        let mut ptr: *mut libc::c_void = std::ptr::null_mut();

        // We need 4KB alignment here for NIXL disk transfers to work.
        // The O_DIRECT flag is required for GDS.
        // However, a limitation of this flag is that all operations involving disk
        // (both read and write) must be page-aligned.
        // Pinned memory is already page-aligned, so we only need to align system memory.
        // TODO(jthomson04): Is page size always 4KB?

        // SAFETY: malloc returns suitably aligned memory or null on failure.
        let result = unsafe { libc::posix_memalign(&mut ptr, 4096, len) };
        if result != 0 {
            return Err(StorageError::AllocationFailed(format!(
                "posix_memalign failed for size {}",
                len
            )));
        }
        let ptr = NonNull::new(ptr as *mut u8).ok_or_else(|| {
            StorageError::AllocationFailed(format!("malloc failed for size {}", len))
        })?;

        // Zero-initialize the memory
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, len);
        }

        Ok(Self { ptr, len })
    }

    /// Get a pointer to the underlying memory.
    ///
    /// # Safety
    /// The caller must ensure the pointer is not used after this storage is dropped.
    pub unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Get a mutable pointer to the underlying memory.
    ///
    /// # Safety
    /// The caller must ensure the pointer is not used after this storage is dropped
    /// and that there are no other references to this memory.
    pub unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for SystemStorage {
    fn drop(&mut self) {
        // SAFETY: pointer was allocated by malloc.
        unsafe {
            libc::free(self.ptr.as_ptr() as *mut libc::c_void);
        }
    }
}

impl MemoryDescriptor for SystemStorage {
    fn addr(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    fn size(&self) -> usize {
        self.len
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::System
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

// Support for NIXL registration
impl super::nixl::NixlCompatible for SystemStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        (self.ptr.as_ptr(), self.len, nixl_sys::MemType::Dram, 0)
    }
}

impl actions::Memset for SystemStorage {
    fn memset(&mut self, value: u8, offset: usize, size: usize) -> Result<()> {
        let end = offset
            .checked_add(size)
            .ok_or_else(|| StorageError::OperationFailed("memset: offset overflow".into()))?;
        if end > self.len {
            return Err(StorageError::OperationFailed(
                "memset: offset + size > storage size".into(),
            ));
        }
        unsafe {
            let ptr = self.ptr.as_ptr().add(offset);
            std::ptr::write_bytes(ptr, value, size);
        }
        Ok(())
    }
}

impl actions::Slice for SystemStorage {
    unsafe fn as_slice(&self) -> Result<&[u8]> {
        // SAFETY: SystemStorage owns the memory allocated via the global allocator.
        // The memory remains valid as long as this SystemStorage instance exists.
        // The ptr is guaranteed to be valid for `self.len` bytes.
        // Caller must ensure no concurrent mutable access per trait contract.
        // SAFETY: The pointer is valid, properly aligned, and points to `self.len` bytes.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::{Memset, Slice};

    #[test]
    fn test_system_storage_new() {
        let storage = SystemStorage::new(1024).expect("allocation should succeed");
        assert_eq!(storage.size(), 1024);
        assert!(storage.addr() != 0);
    }

    #[test]
    fn test_system_storage_zero_size_fails() {
        let result = SystemStorage::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_system_storage_storage_kind() {
        let storage = SystemStorage::new(1024).unwrap();
        assert_eq!(storage.storage_kind(), StorageKind::System);
    }

    #[test]
    fn test_system_storage_as_any() {
        let storage = SystemStorage::new(1024).unwrap();
        let any = storage.as_any();
        assert!(any.downcast_ref::<SystemStorage>().is_some());
    }

    #[test]
    fn test_system_storage_nixl_descriptor() {
        let storage = SystemStorage::new(1024).unwrap();
        // Unregistered storage has no NIXL descriptor
        assert!(storage.nixl_descriptor().is_none());
    }

    #[test]
    fn test_system_storage_as_ptr() {
        let storage = SystemStorage::new(1024).unwrap();
        unsafe {
            let ptr = storage.as_ptr();
            assert!(!ptr.is_null());
            assert_eq!(ptr as usize, storage.addr());
        }
    }

    #[test]
    fn test_system_storage_as_mut_ptr() {
        let mut storage = SystemStorage::new(1024).unwrap();
        unsafe {
            let ptr = storage.as_mut_ptr();
            assert!(!ptr.is_null());
            assert_eq!(ptr as usize, storage.addr());

            // Write and read back to verify the pointer works
            *ptr = 0xAB;
            assert_eq!(*ptr, 0xAB);
        }
    }

    #[test]
    fn test_system_storage_zero_initialized() {
        let storage = SystemStorage::new(1024).unwrap();
        unsafe {
            let slice = storage.as_slice().unwrap();
            // Memory should be zero-initialized
            assert!(slice.iter().all(|&b| b == 0));
        }
    }

    #[test]
    fn test_system_storage_memset_and_read() {
        let mut storage = SystemStorage::new(1024).unwrap();
        storage.memset(0xCD, 0, 1024).unwrap();

        unsafe {
            let slice = storage.as_slice().unwrap();
            assert!(slice.iter().all(|&b| b == 0xCD));
        }
    }

    #[test]
    fn test_system_storage_multiple_allocations_independent() {
        let storage1 = SystemStorage::new(512).unwrap();
        let storage2 = SystemStorage::new(512).unwrap();

        // Different allocations should have different addresses
        assert_ne!(storage1.addr(), storage2.addr());
    }

    #[test]
    fn test_system_storage_alignment() {
        let storage = SystemStorage::new(1024).unwrap();
        // posix_memalign allocates with 4096-byte alignment
        assert!(storage.addr().is_multiple_of(4096));
    }

    #[test]
    fn test_system_storage_nixl_compatible() {
        use crate::nixl::NixlCompatible;

        let storage = SystemStorage::new(2048).unwrap();
        let (ptr, size, mem_type, device_id) = storage.nixl_params();

        assert_eq!(ptr as usize, storage.addr());
        assert_eq!(size, 2048);
        assert_eq!(mem_type, nixl_sys::MemType::Dram);
        assert_eq!(device_id, 0);
    }

    #[test]
    fn test_system_storage_large_allocation() {
        // Allocate 1MB to test larger sizes
        let storage = SystemStorage::new(1024 * 1024).unwrap();
        assert_eq!(storage.size(), 1024 * 1024);
    }

    #[test]
    fn test_system_storage_debug() {
        let storage = SystemStorage::new(1024).unwrap();
        let debug_str = format!("{:?}", storage);
        assert!(debug_str.contains("SystemStorage"));
    }
}
