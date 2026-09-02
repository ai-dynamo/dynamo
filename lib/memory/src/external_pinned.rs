// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA registration wrapper for externally owned host memory.

use crate::nixl::{MemType, NixlCompatible, NixlDescriptor};
use crate::{MemoryDescriptor, Result, StorageError, StorageKind};
use cudarc::driver::CudaContext;
use cudarc::driver::result::DriverError;
use std::any::Any;
use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;

/// Externally allocated host memory registered with CUDA for DMA.
///
/// The underlying mapping remains owned by the caller. Dropping this value only
/// unregisters the pages from CUDA; it never calls `cuMemFreeHost` or unmaps the
/// address. The caller must keep the mapping alive until this wrapper is dropped.
pub struct ExternalPinnedStorage {
    ptr: NonNull<u8>,
    len: usize,
    ctx: Arc<CudaContext>,
}

// SAFETY: construction requires the caller to keep the external mapping alive;
// CUDA registration makes the address stable for the wrapper's lifetime.
unsafe impl Send for ExternalPinnedStorage {}
unsafe impl Sync for ExternalPinnedStorage {}

impl ExternalPinnedStorage {
    /// Register an externally owned host region for CUDA DMA.
    ///
    /// # Safety
    ///
    /// `ptr..ptr+len` must be a live, writable host mapping which remains valid
    /// until this value is dropped. No other owner may unregister the same range.
    pub unsafe fn new(ptr: *mut u8, len: usize, device_id: u32) -> Result<Self> {
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            StorageError::RegistrationFailed("external host pointer is null".into())
        })?;
        if len == 0 {
            return Err(StorageError::RegistrationFailed(
                "external host region is empty".into(),
            ));
        }
        if len > isize::MAX as usize || (ptr.as_ptr() as usize).checked_add(len).is_none() {
            return Err(StorageError::RegistrationFailed(
                "external host region address range overflows".into(),
            ));
        }

        let ctx = crate::device::cuda_context(device_id)?;
        ctx.bind_to_thread().map_err(StorageError::Cuda)?;
        let flags = cudarc::driver::sys::CU_MEMHOSTREGISTER_PORTABLE
            | cudarc::driver::sys::CU_MEMHOSTREGISTER_DEVICEMAP;
        unsafe {
            cudarc::driver::sys::cuMemHostRegister_v2(ptr.as_ptr().cast::<c_void>(), len, flags)
                .result()
                .map_err(StorageError::Cuda)?;
        }
        Ok(Self { ptr, len, ctx })
    }
}

impl Drop for ExternalPinnedStorage {
    fn drop(&mut self) {
        if let Err(error) = self.ctx.bind_to_thread() {
            tracing::error!(?error, "failed to bind CUDA context before host unregister");
            return;
        }
        let result: std::result::Result<(), DriverError> = unsafe {
            cudarc::driver::sys::cuMemHostUnregister(self.ptr.as_ptr().cast::<c_void>()).result()
        };
        if let Err(error) = result {
            tracing::error!(?error, "failed to unregister external pinned memory");
        }
    }
}

impl fmt::Debug for ExternalPinnedStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalPinnedStorage")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl MemoryDescriptor for ExternalPinnedStorage {
    fn addr(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    fn size(&self) -> usize {
        self.len
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::Pinned
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

impl NixlCompatible for ExternalPinnedStorage {
    fn nixl_params(&self) -> (*const u8, usize, MemType, u64) {
        (self.ptr.as_ptr(), self.len, MemType::Dram, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_regions_before_opening_cuda() {
        let null = unsafe { ExternalPinnedStorage::new(std::ptr::null_mut(), 1, 0) };
        assert!(matches!(null, Err(StorageError::RegistrationFailed(_))));

        let empty = unsafe { ExternalPinnedStorage::new(NonNull::dangling().as_ptr(), 0, 0) };
        assert!(matches!(empty, Err(StorageError::RegistrationFailed(_))));
    }
}
