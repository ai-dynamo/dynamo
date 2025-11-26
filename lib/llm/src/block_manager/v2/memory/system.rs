// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! System memory storage backed by malloc.

use super::{MemoryRegion, Result, StorageError, StorageKind, actions};
use std::any::Any;
use std::ptr::NonNull;

use nix::libc;

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

impl MemoryRegion for SystemStorage {
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
}

// Support for NIXL registration
impl super::registered::NixlCompatible for SystemStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        (self.ptr.as_ptr(), self.len, nixl_sys::MemType::Dram, 0)
    }
}

impl actions::Memset for SystemStorage {
    fn memset(&mut self, value: u8, offset: usize, size: usize) -> Result<()> {
        if offset + size > self.len {
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
    fn as_slice(&self) -> Result<&[u8]> {
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) })
    }
}
