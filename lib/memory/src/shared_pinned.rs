// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared pinned host memory backed by `/dev/shm/` files.
//!
//! Enables GPU-accessible shared host memory regions that multiple processes
//! can map simultaneously. Uses `mmap(MAP_SHARED)` on a `/dev/shm/` file
//! and registers the region with CUDA via `cuMemHostRegister` for GPU access.
//!
//! # Use Case
//!
//! A shared KVBM G2 host-memory cache where multiple TP=1 GPU instances on
//! the same node can `cudaMemcpyAsync` or do direct kernel reads/writes
//! against the same host memory region.
//!
//! # Architecture
//!
//! 1. **Owner** creates the shared file and memory region
//! 2. **Joiners** open the existing file and map it
//! 3. Both register with CUDA for GPU access via `cuMemHostRegister`
//! 4. Each gets a device pointer via `cuMemHostGetDevicePointer`
//! 5. GPU kernels or `cudaMemcpyAsync` use the device pointer

use std::path::{Path, PathBuf};

use cudarc::driver::sys::{self, CUresult};

use super::{Result, StorageError};

/// A pinned host memory region backed by a shared file, GPU-accessible.
///
/// Multiple processes can map the same file and access the memory from
/// both CPU and GPU. The underlying file lives on a tmpfs (e.g., `/dev/shm/`)
/// for low-latency access.
#[derive(Debug)]
pub struct SharedPinnedStorage {
    /// Path to the shared memory file.
    path: PathBuf,
    /// mmap'd host pointer.
    ptr: *mut u8,
    /// Size of the region in bytes.
    size: usize,
    /// GPU-accessible device pointer (from cuMemHostGetDevicePointer).
    device_ptr: u64,
    /// Whether this instance created the file (owner cleans up on drop).
    is_owner: bool,
}

unsafe impl Send for SharedPinnedStorage {}
unsafe impl Sync for SharedPinnedStorage {}

/// Create a new shared pinned region (owner — creates the file).
///
/// # Arguments
/// * `path` - Path for the shared memory file (e.g., `/dev/shm/gms_cache`)
/// * `size` - Size in bytes
/// * `device` - CUDA device index for GPU registration
pub fn create_shared_pinned(path: &Path, size: usize, device: i32) -> Result<SharedPinnedStorage> {
    if size == 0 {
        return Err(StorageError::AllocationFailed(
            "zero-sized shared pinned allocation not supported".into(),
        ));
    }

    // Create and size the file
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    file.set_len(size as u64)?;

    // mmap with MAP_SHARED
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            std::os::unix::io::AsRawFd::as_raw_fd(&file),
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(StorageError::AllocationFailed(format!(
            "mmap failed for {:?}: {}",
            path,
            std::io::Error::last_os_error()
        )));
    }
    let ptr = ptr as *mut u8;

    // Register with CUDA for GPU access
    let device_ptr = match register_with_cuda(ptr, size, device) {
        Ok(dp) => dp,
        Err(e) => {
            unsafe { libc::munmap(ptr as *mut libc::c_void, size) };
            std::fs::remove_file(path).ok();
            return Err(e);
        }
    };

    Ok(SharedPinnedStorage {
        path: path.to_path_buf(),
        ptr,
        size,
        device_ptr,
        is_owner: true,
    })
}

/// Join an existing shared pinned region (joiner — opens existing file).
///
/// # Arguments
/// * `path` - Path to the existing shared memory file
/// * `device` - CUDA device index for GPU registration
pub fn open_shared_pinned(path: &Path, device: i32) -> Result<SharedPinnedStorage> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?;

    let metadata = file.metadata()?;
    let size = metadata.len() as usize;
    if size == 0 {
        return Err(StorageError::AllocationFailed(
            "shared pinned file has zero size".into(),
        ));
    }

    // mmap with MAP_SHARED
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            std::os::unix::io::AsRawFd::as_raw_fd(&file),
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(StorageError::AllocationFailed(format!(
            "mmap failed for {:?}: {}",
            path,
            std::io::Error::last_os_error()
        )));
    }
    let ptr = ptr as *mut u8;

    // Register with CUDA
    let device_ptr = match register_with_cuda(ptr, size, device) {
        Ok(dp) => dp,
        Err(e) => {
            unsafe { libc::munmap(ptr as *mut libc::c_void, size) };
            return Err(e);
        }
    };

    Ok(SharedPinnedStorage {
        path: path.to_path_buf(),
        ptr,
        size,
        device_ptr,
        is_owner: false,
    })
}

/// CUDA host register flags.
const CU_MEMHOSTREGISTER_PORTABLE: u32 = 0x01;
const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 0x02;

/// Register a host pointer with CUDA and get the device-accessible pointer.
fn register_with_cuda(ptr: *mut u8, size: usize, device: i32) -> Result<u64> {
    // Ensure CUDA context is active for the target device
    let ctx = crate::device::cuda_context(device as u32)?;
    ctx.bind_to_thread().map_err(StorageError::Cuda)?;

    let flags = CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_PORTABLE;

    let result =
        unsafe { sys::cuMemHostRegister_v2(ptr as *mut std::ffi::c_void, size, flags as u32) };
    if result != CUresult::CUDA_SUCCESS {
        return Err(StorageError::OperationFailed(format!(
            "cuMemHostRegister_v2 failed: {result:?}"
        )));
    }

    let mut device_ptr: u64 = 0;
    let result = unsafe {
        sys::cuMemHostGetDevicePointer_v2(
            &mut device_ptr,
            ptr as *mut std::ffi::c_void,
            0,
        )
    };
    if result != CUresult::CUDA_SUCCESS {
        // Unregister on failure
        unsafe { sys::cuMemHostUnregister(ptr as *mut std::ffi::c_void) };
        return Err(StorageError::OperationFailed(format!(
            "cuMemHostGetDevicePointer_v2 failed: {result:?}"
        )));
    }

    Ok(device_ptr)
}

impl SharedPinnedStorage {
    /// Path to the shared memory file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Size of the region in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Host pointer to the shared memory.
    ///
    /// # Safety
    /// The caller must ensure the pointer is not used after this storage is dropped.
    pub unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Mutable host pointer to the shared memory.
    ///
    /// # Safety
    /// The caller must ensure exclusive access and that the pointer is not used
    /// after this storage is dropped.
    pub unsafe fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// GPU-accessible device pointer for `cudaMemcpyAsync` or kernel access.
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    /// Whether this instance owns (created) the shared file.
    pub fn is_owner(&self) -> bool {
        self.is_owner
    }
}

impl Drop for SharedPinnedStorage {
    fn drop(&mut self) {
        // Unregister from CUDA
        let result =
            unsafe { sys::cuMemHostUnregister(self.ptr as *mut std::ffi::c_void) };
        if result != CUresult::CUDA_SUCCESS {
            tracing::debug!("cuMemHostUnregister failed: {result:?}");
        }

        // Unmap
        let result = unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.size) };
        if result != 0 {
            tracing::debug!(
                "munmap failed: {}",
                std::io::Error::last_os_error()
            );
        }

        // Owner deletes the file
        if self.is_owner {
            if let Err(e) = std::fs::remove_file(&self.path) {
                tracing::debug!("failed to remove shared pinned file {:?}: {e}", self.path);
            }
        }
    }
}
