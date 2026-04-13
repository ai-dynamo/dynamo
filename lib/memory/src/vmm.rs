// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA Virtual Memory Management (VMM) primitives.
//!
//! Safe wrappers around `cudarc::driver::sys::*` VMM functions for physical
//! memory allocation, FD export/import, virtual address reservation, and mapping.
//!
//! The key design insight from the GPU Memory Service is that the **server**
//! only creates physical allocations and exports FDs — it never maps memory
//! to virtual addresses. This means the server doesn't need a CUDA context
//! and can survive GPU driver restarts.
//!
//! **Clients** import the FDs, reserve VA space, map, and set access.

use std::os::fd::{FromRawFd, OwnedFd, RawFd};

use cudarc::driver::sys::{self, CUresult};

use super::{Result, StorageError};

/// A physical GPU memory allocation handle.
///
/// Created by [`vmm_create`], this represents physical memory that can be
/// exported as a POSIX file descriptor for sharing across processes.
/// The handle does NOT map to any virtual address — that is done separately.
#[derive(Debug)]
pub struct VmmHandle {
    /// Raw CUDA allocation handle (CUmemGenericAllocationHandle).
    pub handle: u64,
    /// Requested size in bytes.
    pub size: usize,
    /// Actual size after alignment to VMM granularity.
    pub aligned_size: usize,
}

/// A virtual address mapping of a physical allocation.
///
/// Created by [`vmm_map`], this represents a VA range backed by a physical handle.
/// Must be unmapped via [`vmm_unmap`] before freeing the VA or releasing the handle.
#[derive(Debug)]
pub struct VmmMapping {
    /// Virtual address of the mapping.
    pub va: u64,
    /// Size of the mapping in bytes.
    pub size: usize,
}

/// Check a CUDA result, converting failures to [`StorageError`].
fn check(result: CUresult, op: &str) -> Result<()> {
    if result != CUresult::CUDA_SUCCESS {
        Err(StorageError::OperationFailed(format!(
            "{op}: CUDA error {result:?}"
        )))
    } else {
        Ok(())
    }
}

/// Build the standard VMM allocation properties for a device.
///
/// Sets pinned allocation type, device location, and POSIX FD handle type.
fn make_alloc_prop(device: i32) -> sys::CUmemAllocationProp {
    let mut prop: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes =
        sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop
}

/// Get the VMM allocation granularity for a device.
///
/// Returns the minimum alignment in bytes (typically 2 MiB). All VMM
/// allocations must be a multiple of this value.
pub fn get_allocation_granularity(device: i32) -> Result<usize> {
    let prop = make_alloc_prop(device);
    let mut granularity: usize = 0;

    let result = unsafe {
        sys::cuMemGetAllocationGranularity(
            &mut granularity,
            &prop,
            sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
    };
    check(result, "cuMemGetAllocationGranularity")?;
    Ok(granularity)
}

/// Align a size up to VMM granularity.
///
/// Returns `None` if the aligned size would overflow `usize`.
pub fn align_to_granularity(size: usize, granularity: usize) -> Option<usize> {
    size.checked_add(granularity - 1)
        .map(|s| (s / granularity) * granularity)
}

/// Create a physical GPU memory allocation (no VA mapping).
///
/// Uses `cuMemCreate` to allocate physical memory that can be exported
/// as a POSIX file descriptor for sharing with other processes.
///
/// The returned handle does NOT have a virtual address mapping. Use
/// [`vmm_reserve_va`] + [`vmm_map`] + [`vmm_set_access`] on the client
/// side to make the memory accessible.
pub fn vmm_create(size: usize, device: i32) -> Result<VmmHandle> {
    let granularity = get_allocation_granularity(device)?;
    let aligned_size = align_to_granularity(size, granularity).ok_or_else(|| {
        StorageError::AllocationFailed(format!(
            "size {size} overflows when aligned to granularity {granularity}"
        ))
    })?;
    let prop = make_alloc_prop(device);

    let mut handle: u64 = 0;
    let result = unsafe { sys::cuMemCreate(&mut handle, aligned_size, &prop, 0) };
    check(result, "cuMemCreate")?;

    Ok(VmmHandle {
        handle,
        size,
        aligned_size,
    })
}

/// Export a VMM allocation as a POSIX file descriptor.
///
/// The returned `OwnedFd` can be sent to another process via Unix domain socket
/// SCM_RIGHTS. The receiving process imports it with [`vmm_import_fd`].
///
/// The `OwnedFd` closes the FD automatically on drop, preventing FD leaks.
pub fn vmm_export_fd(handle: &VmmHandle) -> Result<OwnedFd> {
    let mut fd: i32 = 0;
    let result = unsafe {
        sys::cuMemExportToShareableHandle(
            &mut fd as *mut i32 as *mut std::ffi::c_void,
            handle.handle,
            sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            0,
        )
    };
    check(result, "cuMemExportToShareableHandle")?;
    Ok(unsafe { OwnedFd::from_raw_fd(fd as RawFd) })
}

/// Import a VMM allocation from a POSIX file descriptor.
///
/// The FD is typically received via SCM_RIGHTS from the GPU Memory Service.
/// After import, the caller should close the FD.
///
/// Note: The imported handle has no size information — the caller must
/// track the size separately (received via the protocol).
pub fn vmm_import_fd(fd: RawFd, size: usize, aligned_size: usize) -> Result<VmmHandle> {
    let mut handle: u64 = 0;
    let fd_val = fd as i32;
    let result = unsafe {
        sys::cuMemImportFromShareableHandle(
            &mut handle,
            &fd_val as *const i32 as *mut std::ffi::c_void,
            sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        )
    };
    check(result, "cuMemImportFromShareableHandle")?;

    Ok(VmmHandle {
        handle,
        size,
        aligned_size,
    })
}

/// Release a physical memory allocation.
///
/// The handle must not be mapped to any virtual address when released.
/// This consumes the handle and prevents the Drop impl from double-freeing.
pub fn vmm_release(mut handle: VmmHandle) -> Result<()> {
    let h = handle.handle;
    handle.handle = 0; // Prevent Drop from releasing again
    let result = unsafe { sys::cuMemRelease(h) };
    check(result, "cuMemRelease")
}

impl Drop for VmmHandle {
    fn drop(&mut self) {
        if self.handle != 0 {
            let result = unsafe { sys::cuMemRelease(self.handle) };
            if result != CUresult::CUDA_SUCCESS {
                tracing::warn!("cuMemRelease failed in drop: {result:?}");
            }
            self.handle = 0;
        }
    }
}

/// Reserve a virtual address range for VMM mapping.
///
/// Returns the base virtual address. The range is not backed by physical
/// memory until [`vmm_map`] is called.
pub fn vmm_reserve_va(size: usize, granularity: usize) -> Result<u64> {
    let mut va: u64 = 0;
    let result = unsafe { sys::cuMemAddressReserve(&mut va, size, granularity, 0, 0) };
    check(result, "cuMemAddressReserve")?;
    Ok(va)
}

/// Free a reserved virtual address range.
///
/// The range must be unmapped first via [`vmm_unmap`].
pub fn vmm_free_va(va: u64, size: usize) -> Result<()> {
    let result = unsafe { sys::cuMemAddressFree(va, size) };
    check(result, "cuMemAddressFree")
}

/// Map a physical allocation to a virtual address range.
///
/// The VA must have been reserved via [`vmm_reserve_va`] with sufficient size.
/// After mapping, call [`vmm_set_access`] to enable GPU access.
pub fn vmm_map(va: u64, handle: &VmmHandle) -> Result<VmmMapping> {
    let result = unsafe { sys::cuMemMap(va, handle.aligned_size, 0, handle.handle, 0) };
    check(result, "cuMemMap")?;

    Ok(VmmMapping {
        va,
        size: handle.aligned_size,
    })
}

/// Unmap a physical allocation from a virtual address range.
pub fn vmm_unmap(mapping: VmmMapping) -> Result<()> {
    let result = unsafe { sys::cuMemUnmap(mapping.va, mapping.size) };
    check(result, "cuMemUnmap")
}

/// Set GPU access permissions on a mapped virtual address range.
///
/// Must be called after [`vmm_map`] to enable GPU read/write access.
///
/// # Arguments
/// * `va` - Base virtual address (must be mapped)
/// * `size` - Size of the region
/// * `device` - CUDA device index that should have access
/// * `read_write` - If true, grants read+write; if false, grants read-only
pub fn vmm_set_access(va: u64, size: usize, device: i32, read_write: bool) -> Result<()> {
    let mut desc: sys::CUmemAccessDesc = unsafe { std::mem::zeroed() };
    desc.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = device;
    desc.flags = if read_write {
        sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    } else {
        sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ
    };

    let result = unsafe { sys::cuMemSetAccess(va, size, &desc, 1) };
    check(result, "cuMemSetAccess")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_to_granularity() {
        assert_eq!(align_to_granularity(0, 2 * 1024 * 1024), Some(0));
        assert_eq!(align_to_granularity(1, 2 * 1024 * 1024), Some(2 * 1024 * 1024));
        assert_eq!(
            align_to_granularity(2 * 1024 * 1024, 2 * 1024 * 1024),
            Some(2 * 1024 * 1024)
        );
        assert_eq!(
            align_to_granularity(2 * 1024 * 1024 + 1, 2 * 1024 * 1024),
            Some(4 * 1024 * 1024)
        );
        assert_eq!(align_to_granularity(100, 64), Some(128));
        assert_eq!(align_to_granularity(64, 64), Some(64));
        // Overflow case
        assert_eq!(align_to_granularity(usize::MAX, 2 * 1024 * 1024), None);
        assert_eq!(align_to_granularity(usize::MAX - 1, 2 * 1024 * 1024), None);
    }

    #[cfg(feature = "testing-cuda")]
    #[test]
    fn test_get_granularity() {
        let granularity = get_allocation_granularity(0).unwrap();
        assert!(granularity > 0);
        // Typically 2 MiB
        assert!(granularity.is_power_of_two());
    }

    #[cfg(feature = "testing-cuda")]
    #[test]
    fn test_vmm_create_export_release() {
        use std::os::fd::AsRawFd;

        let handle = vmm_create(4 * 1024 * 1024, 0).unwrap();
        assert!(handle.aligned_size >= handle.size);
        assert!(handle.handle != 0);

        let owned_fd = vmm_export_fd(&handle).unwrap();
        assert!(owned_fd.as_raw_fd() >= 0);
        // OwnedFd closes on drop

        vmm_release(handle).unwrap();
    }

    #[cfg(feature = "testing-cuda")]
    #[test]
    fn test_vmm_full_lifecycle() {
        // Create physical allocation
        let handle = vmm_create(4 * 1024 * 1024, 0).unwrap();
        let size = handle.aligned_size;

        // Reserve VA
        let granularity = get_allocation_granularity(0).unwrap();
        let va = vmm_reserve_va(size, granularity).unwrap();
        assert!(va != 0);

        // Map
        let mapping = vmm_map(va, &handle).unwrap();
        assert_eq!(mapping.va, va);

        // Set access
        vmm_set_access(va, size, 0, true).unwrap();

        // Unmap
        vmm_unmap(mapping).unwrap();

        // Free VA
        vmm_free_va(va, size).unwrap();

        // Release physical memory
        vmm_release(handle).unwrap();
    }
}
