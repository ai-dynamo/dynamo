// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust FFI wrappers for the SYCL block/universal permute kernels (XPU).
//!
//! This mirrors `tensor_kernels.rs` (CUDA) but links against
//! `libkvbm_kernels_sycl.so` built from `sycl/tensor_permute_kernel.cpp`.
//!
//! Key differences from the CUDA version:
//! * Uses `elem_size` (bytes per element) instead of a dtype enum — the SYCL
//!   kernel treats all types as raw byte moves (no arithmetic on element values).
//! * The "stream" parameter is an opaque `*mut c_void` pointing to a
//!   `sycl::queue` — the caller must ensure the pointer remains valid for the
//!   duration of the kernel execution.
//! * Return type is `i32` (0 = success) instead of `cudaError_t`.

use std::ffi::c_void;

use crate::BlockLayout;

#[allow(dead_code)]
unsafe extern "C" {
    fn kvbm_kernels_sycl_launch_universal_from_block(
        universal_ptrs: *const *mut c_void,
        block_ptrs: *const *const c_void,
        num_blocks: usize,
        nh: usize,
        nl: usize,
        no: usize,
        nt: usize,
        hd: usize,
        nl_full: usize,
        nl_offset: usize,
        elem_size: usize,
        layout: i32,
        queue: *mut c_void,
    ) -> i32;

    fn kvbm_kernels_sycl_launch_block_from_universal(
        universal_ptrs: *const *const c_void,
        block_ptrs: *const *mut c_void,
        num_blocks: usize,
        nh: usize,
        nl: usize,
        no: usize,
        nt: usize,
        hd: usize,
        nl_full: usize,
        nl_offset: usize,
        elem_size: usize,
        layout: i32,
        queue: *mut c_void,
    ) -> i32;

    fn kvbm_kernels_sycl_launch_vectorized_copy(
        src_ptrs: *mut *mut c_void,
        dst_ptrs: *mut *mut c_void,
        copy_size_bytes: usize,
        num_pairs: i32,
        queue: *mut c_void,
    ) -> i32;

    fn kvbm_kernels_sycl_launch_nhd_hnd_transpose(
        src_ptrs: *const *const c_void,
        dst_ptrs: *const *mut c_void,
        num_blocks: usize,
        nl: usize,
        no: usize,
        nt: usize,
        nh: usize,
        hd: usize,
        elem_size: usize,
        src_layout: i32,
        queue: *mut c_void,
    ) -> i32;

}

/// Copy `num_blocks` stacks of NHD/HND tensors into universal form (XPU).
///
/// * `universal_ptrs` – device-accessible pointer to `num_blocks` universal bases.
/// * `block_ptrs` – device-accessible pointer to a flattened `[num_blocks][nl*no]`
///   table of chunk pointers.
/// * `nh, nl, no, nt, hd` – logical dimensions of each universal tensor.
/// * `elem_size` – bytes per element (2=f16/bf16, 4=f32, 8=f64).
/// * `layout` – chunk inner layout (NHD or HND).
/// * `queue` – opaque `sycl::queue*` for kernel submission.
///
/// Returns 0 on success, non-zero on error.
///
/// # Safety
/// - All pointer arrays must be device-accessible with correct lengths.
/// - `queue` must be a valid `sycl::queue*` that outlives the kernel execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sycl_universal_from_block(
    universal_ptrs: *const *mut c_void,
    block_ptrs: *const *const c_void,
    num_blocks: usize,
    nh: usize,
    nl: usize,
    no: usize,
    nt: usize,
    hd: usize,
    nl_full: usize,
    nl_offset: usize,
    elem_size: usize,
    layout: BlockLayout,
    queue: *mut c_void,
) -> i32 {
    unsafe {
        kvbm_kernels_sycl_launch_universal_from_block(
            universal_ptrs,
            block_ptrs,
            num_blocks,
            nh,
            nl,
            no,
            nt,
            hd,
            nl_full,
            nl_offset,
            elem_size,
            layout as i32,
            queue,
        )
    }
}

/// Copy `num_blocks` universal tensors back into their block stacks (XPU).
///
/// # Safety
/// Same requirements as [`sycl_universal_from_block`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn sycl_block_from_universal(
    universal_ptrs: *const *const c_void,
    block_ptrs: *const *mut c_void,
    num_blocks: usize,
    nh: usize,
    nl: usize,
    no: usize,
    nt: usize,
    hd: usize,
    nl_full: usize,
    nl_offset: usize,
    elem_size: usize,
    layout: BlockLayout,
    queue: *mut c_void,
) -> i32 {
    unsafe {
        kvbm_kernels_sycl_launch_block_from_universal(
            universal_ptrs,
            block_ptrs,
            num_blocks,
            nh,
            nl,
            no,
            nt,
            hd,
            nl_full,
            nl_offset,
            elem_size,
            layout as i32,
            queue,
        )
    }
}

/// Launch vectorized copy between device-visible pointer pairs (XPU / SYCL).
///
/// Mirrors [`kvbm_kernels::vectorized_copy`] (CUDA) but uses a `sycl::queue*`
/// instead of a `cudaStream_t`.
///
/// # Arguments
/// * `src_ptrs` - Device-accessible pointer to array of source pointers
/// * `dst_ptrs` - Device-accessible pointer to array of destination pointers
/// * `copy_size_bytes` - Size of each copy in bytes (same for all pairs)
/// * `num_pairs` - Number of pointer pairs to copy
/// * `queue` - Opaque `sycl::queue*` for kernel submission
///
/// # Safety
/// - All pointers in src/dst arrays must be valid device-visible pointers.
/// - The pointer arrays must reside in device memory with ≥ `num_pairs` entries.
/// - `queue` must be a valid `sycl::queue*` that outlives the kernel execution.
/// NHD ↔ HND transpose for operational block stacks (XPU).
///
/// Mirrors `nhd_hnd_transpose` (CUDA) — both src and dst are block stacks of
/// shape `[nl, no][nt, nh, hd]` (NHD) or `[nl, no][nh, nt, hd]` (HND); the
/// transform swaps the inner `(nt, nh)` order while keeping `hd` contiguous.
/// `dst_layout` is implicitly the opposite of `src_layout`.
///
/// * `src_ptrs` – device-accessible `[num_blocks*nl*no]` table of src chunk pointers.
/// * `dst_ptrs` – device-accessible `[num_blocks*nl*no]` table of dst chunk pointers.
/// * `nl, no, nt, nh, hd` – logical chunk dimensions.
/// * `elem_size` – bytes per element (2=f16/bf16, 4=f32, 8=f64).
/// * `src_layout` – chunk inner layout of the source (dst is the opposite).
/// * `queue` – opaque `sycl::queue*` for kernel submission.
///
/// Returns 0 on success, non-zero on error.
///
/// # Safety
/// - All pointer arrays must be device-accessible with correct lengths.
/// - `queue` must be a valid `sycl::queue*` that outlives the kernel execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sycl_nhd_hnd_transpose(
    src_ptrs: *const *const c_void,
    dst_ptrs: *const *mut c_void,
    num_blocks: usize,
    nl: usize,
    no: usize,
    nt: usize,
    nh: usize,
    hd: usize,
    elem_size: usize,
    src_layout: BlockLayout,
    queue: *mut c_void,
) -> i32 {
    unsafe {
        kvbm_kernels_sycl_launch_nhd_hnd_transpose(
            src_ptrs,
            dst_ptrs,
            num_blocks,
            nl,
            no,
            nt,
            nh,
            hd,
            elem_size,
            src_layout as i32,
            queue,
        )
    }
}

pub unsafe fn sycl_vectorized_copy(
    src_ptrs: *mut *mut c_void,
    dst_ptrs: *mut *mut c_void,
    copy_size_bytes: usize,
    num_pairs: i32,
    queue: *mut c_void,
) -> i32 {
    unsafe {
        kvbm_kernels_sycl_launch_vectorized_copy(
            src_ptrs,
            dst_ptrs,
            copy_size_bytes,
            num_pairs,
            queue,
        )
    }
}
