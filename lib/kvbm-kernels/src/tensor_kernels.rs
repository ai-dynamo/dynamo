// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Safe-ish wrappers around the CUDA block/universal packing kernels.
//!
//! The core ideas:
//! * A “block” represents the stack of `nl * no` tensors arranged either as NHD
//!   (inner axes `[nt, nh, hd]`) or HND (inner axes `[nh, nt, hd]`).
//! * A “universal” tensor is `[nh, nl, no, nt, hd]` stored contiguously.
//! * An “operational” tensor is `[nl, no, inner]` with `inner = nt * nh * hd`.
//!
//! All pointer-list parameters (e.g. `universal_ptrs`, `src_ptrs`) must be
//! device-accessible: allocated via `cudaMalloc` (device memory) or
//! `cudaMallocHost` / `cuMemHostRegister` (pinned/registered/page-locked host memory).
//!
//! Host code calls these helpers with flattened pointer tables so a single
//! launch can move many logical blocks in one go.

#![allow(clippy::missing_safety_doc)]
use std::ffi::c_void;

use cudarc::runtime::sys::{cudaError_t, cudaStream_t};

/// Numeric tags passed across the FFI boundary to select the CUDA template.
#[cfg(feature = "permute_kernels")]
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorDataType {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
    F64 = 3,
}

/// Identifies how each `[nt, nh, hd]` chunk is laid out in device memory.
#[cfg(feature = "permute_kernels")]
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockLayout {
    NHD = 0,
    HND = 1,
}

#[cfg(feature = "permute_kernels")]
#[allow(dead_code)]
unsafe extern "C" {
    fn kvbm_kernels_launch_universal_from_block(
        universal_ptrs: *const *mut c_void,
        block_ptrs: *const *const c_void,
        num_blocks: usize,
        nh: usize,
        nl: usize,
        no: usize,
        nt: usize,
        hd: usize,
        dtype: i32,
        layout: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn kvbm_kernels_launch_block_from_universal(
        universal_ptrs: *const *const c_void,
        block_ptrs: *const *mut c_void,
        num_blocks: usize,
        nh: usize,
        nl: usize,
        no: usize,
        nt: usize,
        hd: usize,
        dtype: i32,
        layout: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

/// Controls how `memcpy_batch` dispatches copies.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemcpyBatchMode {
    /// Try cudaMemcpyBatchAsync, fall back to individual cudaMemcpyAsync on failure.
    BatchedWithFallback = 0,
    /// Only use individual cudaMemcpyAsync loop (never attempt batch API).
    FallbackOnly = 1,
    /// Try cudaMemcpyBatchAsync, return error on failure (no fallback).
    BatchWithoutFallback = 2,
}

#[allow(dead_code)]
unsafe extern "C" {
    fn kvbm_kernels_launch_vectorized_copy(
        src_ptrs: *mut *mut c_void,
        dst_ptrs: *mut *mut c_void,
        copy_size_bytes: usize,
        num_pairs: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn kvbm_kernels_memcpy_batch(
        src_ptrs: *const *const c_void,
        dst_ptrs: *const *mut c_void,
        size_per_copy: usize,
        num_copies: usize,
        mode: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn kvbm_kernels_has_memcpy_batch_async() -> bool;
    fn kvbm_kernels_is_stub_build() -> bool;
}

/// Check if cudaMemcpyBatchAsync is available.
///
/// Returns true if the library was compiled with CUDA 12.9+ which provides
/// the `cudaMemcpyBatchAsync` API for efficient batched memory transfers.
pub fn is_memcpy_batch_available() -> bool {
    unsafe { kvbm_kernels_has_memcpy_batch_async() }
}

/// Check if this library was built with stub kernels (no real CUDA).
///
/// Returns `true` if the library is using stubs that will abort on actual CUDA calls.
/// Returns `false` if real CUDA kernels are available.
///
/// Downstream crates should use this to skip CUDA tests at runtime:
/// ```ignore
/// #[test]
/// fn my_cuda_test() {
///     if kvbm_kernels::is_using_stubs() {
///         eprintln!("Skipping CUDA test: stub kernels in use");
///         return;
///     }
///     // ... actual CUDA test code ...
/// }
/// ```
pub fn is_using_stubs() -> bool {
    unsafe { kvbm_kernels_is_stub_build() }
}

/// Batched memcpy using cudaMemcpyBatchAsync (CUDA 12.9+) and/or individual cudaMemcpyAsync.
///
/// Takes HOST arrays of src/dst pointers - no device allocation needed.
/// Direction is auto-determined by CUDA from pointer types using cudaMemcpyDefault.
///
/// The `mode` parameter controls dispatch:
/// - [`MemcpyBatchMode::BatchedWithFallback`]: try batch API, fall back to individual copies on error
/// - [`MemcpyBatchMode::FallbackOnly`]: always use individual cudaMemcpyAsync loop
/// - [`MemcpyBatchMode::BatchWithoutFallback`]: try batch API, return error if unavailable
///
/// # Safety
/// - `src_ptrs` must point to a valid array of `num_copies` source pointers
/// - `dst_ptrs` must point to a valid array of `num_copies` destination pointers
/// - Each source/destination pointer pair must have at least `size_per_copy` bytes accessible
/// - `stream` must be a valid CUDA stream handle
pub unsafe fn memcpy_batch(
    src_ptrs: *const *const c_void,
    dst_ptrs: *const *mut c_void,
    size_per_copy: usize,
    num_copies: usize,
    mode: MemcpyBatchMode,
    stream: cudaStream_t,
) -> cudaError_t {
    unsafe {
        kvbm_kernels_memcpy_batch(
            src_ptrs,
            dst_ptrs,
            size_per_copy,
            num_copies,
            mode as i32,
            stream,
        )
    }
}

/// Copy `num_blocks` stacks of NHD/HND tensors into universal form.
///
/// * `universal_ptrs` – device-accessible pointer to `num_blocks` universal bases.
/// * `block_ptrs` – device-accessible pointer to a flattened `[num_blocks][nl*no]`
///   table of chunk pointers.
/// * `nh, nl, no, nt, hd` – logical dimensions of each universal tensor.
/// * `stream` – CUDA stream used for the launch.
#[cfg(feature = "permute_kernels")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn universal_from_block(
    universal_ptrs: *const *mut c_void,
    block_ptrs: *const *const c_void,
    num_blocks: usize,
    nh: usize,
    nl: usize,
    no: usize,
    nt: usize,
    hd: usize,
    dtype: TensorDataType,
    layout: BlockLayout,
    stream: cudaStream_t,
) -> cudaError_t {
    unsafe {
        kvbm_kernels_launch_universal_from_block(
            universal_ptrs,
            block_ptrs,
            num_blocks,
            nh,
            nl,
            no,
            nt,
            hd,
            dtype as i32,
            layout as i32,
            stream,
        )
    }
}

/// Copy `num_blocks` universal tensors back into their block stacks.
#[cfg(feature = "permute_kernels")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn block_from_universal(
    universal_ptrs: *const *const c_void,
    block_ptrs: *const *mut c_void,
    num_blocks: usize,
    nh: usize,
    nl: usize,
    no: usize,
    nt: usize,
    hd: usize,
    dtype: TensorDataType,
    layout: BlockLayout,
    stream: cudaStream_t,
) -> cudaError_t {
    unsafe {
        kvbm_kernels_launch_block_from_universal(
            universal_ptrs,
            block_ptrs,
            num_blocks,
            nh,
            nl,
            no,
            nt,
            hd,
            dtype as i32,
            layout as i32,
            stream,
        )
    }
}

/// Launch vectorized copy between arbitrary device-visible pointer pairs.
///
/// This kernel automatically selects optimal vectorization (4/8/16 bytes) based on
/// pointer alignment. It is useful for copying between non-contiguous memory regions
/// where each pair has the same copy size.
///
/// Both source and destination pointers may refer to any device-visible memory,
/// including device allocations (`cudaMalloc`) and pinned host memory
/// (`cudaMallocHost` / `cudaHostAlloc`). CUDA unified addressing resolves the
/// actual location at runtime.
///
/// # Arguments
/// * `src_ptrs` - Device-accessible pointer to array of source pointers (each pointing to device-visible memory)
/// * `dst_ptrs` - Device-accessible pointer to array of destination pointers (each pointing to device-visible memory)
/// * `copy_size_bytes` - Size of each copy in bytes (same for all pairs)
/// * `num_pairs` - Number of pointer pairs to copy
/// * `stream` - CUDA stream for async execution
///
/// # Safety
/// - All pointers in the src/dst arrays must be valid device-visible pointers
///   (device memory or pinned host memory)
/// - Each pointer must have at least `copy_size_bytes` bytes accessible
/// - The pointer arrays themselves must be in device memory with at least `num_pairs` entries
/// - `stream` must be a valid CUDA stream handle
pub unsafe fn vectorized_copy(
    src_ptrs: *mut *mut c_void,
    dst_ptrs: *mut *mut c_void,
    copy_size_bytes: usize,
    num_pairs: i32,
    stream: cudaStream_t,
) -> cudaError_t {
    unsafe {
        kvbm_kernels_launch_vectorized_copy(src_ptrs, dst_ptrs, copy_size_bytes, num_pairs, stream)
    }
}

// Tests are gated to only run when:
// 1. testing-cuda feature is enabled
// 2. permute_kernels feature is enabled (tests use universal kernels)
// 3. NOT using stub kernels (stub_kernels cfg is set by build.rs when no nvcc)
#[cfg(all(
    test,
    feature = "testing-cuda",
    feature = "permute_kernels",
    not(stub_kernels)
))]
mod tests {
    use super::*;
    use cudarc::driver::result::memset_d8_async;
    use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut, DriverError};
    use cudarc::runtime::sys as cuda_runtime;

    #[test]
    fn universal_roundtrip() -> Result<(), DriverError> {
        let device_count = match CudaContext::device_count() {
            Ok(count) => count,
            Err(_) => return Ok(()),
        };
        if device_count <= 0 {
            return Ok(());
        }

        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let stream_raw = stream.cu_stream() as cuda_runtime::cudaStream_t;

        let nh = 2usize;
        let nl = 2usize;
        let no = 2usize;
        let nt = 3usize;
        let hd = 4usize;
        let inner = nt * nh * hd;
        let chunk_count = nl * no;
        let block_volume = nh * nl * no * nt * hd;
        let num_blocks = 2usize;

        let dtype = TensorDataType::F32;
        let layout = BlockLayout::NHD;

        let mut host_block_chunks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_blocks);
        let mut block_slices: Vec<Vec<CudaSlice<f32>>> = Vec::with_capacity(num_blocks);
        let mut block_ptr_values: Vec<usize> = Vec::with_capacity(num_blocks * chunk_count);

        for block_idx in 0..num_blocks {
            let mut host_chunks_for_block = Vec::with_capacity(chunk_count);
            let mut slices_for_block = Vec::with_capacity(chunk_count);
            for chunk_idx in 0..chunk_count {
                let global_idx = block_idx * chunk_count + chunk_idx;
                let mut host_chunk = Vec::with_capacity(inner);
                for offset in 0..inner {
                    host_chunk.push((global_idx * inner + offset) as f32 + 0.25f32);
                }
                let slice = stream.clone_htod(&host_chunk)?;
                {
                    let (ptr_raw, _guard) = slice.device_ptr(&stream);
                    block_ptr_values.push(ptr_raw as usize);
                }
                slices_for_block.push(slice);
                host_chunks_for_block.push(host_chunk);
            }
            block_slices.push(slices_for_block);
            host_block_chunks.push(host_chunks_for_block);
        }

        let block_ptrs = stream.clone_htod(block_ptr_values.as_slice())?;

        let mut universal_slices = Vec::with_capacity(num_blocks);
        let mut universal_ptr_values = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let mut slice = unsafe { stream.alloc::<f32>(block_volume)? };
            {
                let (ptr_raw, _guard) = slice.device_ptr_mut(&stream);
                universal_ptr_values.push(ptr_raw as usize);
                unsafe {
                    memset_d8_async(
                        ptr_raw,
                        0xDE,
                        block_volume * std::mem::size_of::<f32>(),
                        stream.cu_stream(),
                    )?;
                }
            }
            universal_slices.push(slice);
        }
        let universal_ptrs = stream.clone_htod(universal_ptr_values.as_slice())?;

        // Block -> Universal
        {
            let (block_ptrs_raw, _block_guard) = block_ptrs.device_ptr(&stream);
            let block_ptrs_ptr = block_ptrs_raw as usize as *const *const c_void;
            let (universal_ptrs_raw, _univ_guard) = universal_ptrs.device_ptr(&stream);
            let universal_ptrs_ptr = universal_ptrs_raw as usize as *const *mut c_void;

            let status = unsafe {
                super::universal_from_block(
                    universal_ptrs_ptr,
                    block_ptrs_ptr,
                    num_blocks,
                    nh,
                    nl,
                    no,
                    nt,
                    hd,
                    dtype,
                    layout,
                    stream_raw,
                )
            };
            assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);
        }
        stream.synchronize()?;

        let inner_offset = |nt_idx: usize, nh_idx: usize, hd_idx: usize| match layout {
            BlockLayout::NHD => ((nt_idx * nh) + nh_idx) * hd + hd_idx,
            BlockLayout::HND => ((nh_idx * nt) + nt_idx) * hd + hd_idx,
        };

        for (block_idx, universal_slice) in universal_slices.iter().enumerate().take(num_blocks) {
            let host_universal = stream.clone_dtoh(universal_slice)?;
            for nh_idx in 0..nh {
                for nl_idx in 0..nl {
                    for no_idx in 0..no {
                        for nt_idx in 0..nt {
                            for hd_idx in 0..hd {
                                let universal_index =
                                    ((((nh_idx * nl + nl_idx) * no + no_idx) * nt + nt_idx) * hd)
                                        + hd_idx;
                                let chunk_idx = nl_idx * no + no_idx;
                                let offset = inner_offset(nt_idx, nh_idx, hd_idx);
                                let expected = ((block_idx * chunk_count + chunk_idx) * inner
                                    + offset) as f32
                                    + 0.25f32;
                                let value = host_universal[universal_index];
                                assert!(
                                    (value - expected).abs() < 1e-5,
                                    "universal mismatch block {} [{} {} {} {} {}]: {} vs {}",
                                    block_idx,
                                    nh_idx,
                                    nl_idx,
                                    no_idx,
                                    nt_idx,
                                    hd_idx,
                                    value,
                                    expected
                                );
                            }
                        }
                    }
                }
            }
        }

        // Universal -> Block (poison-fill destination before reverse pass)
        for block in &mut block_slices {
            for slice in block {
                let (dptr, _guard) = slice.device_ptr_mut(&stream);
                unsafe {
                    memset_d8_async(
                        dptr,
                        0xDE,
                        inner * std::mem::size_of::<f32>(),
                        stream.cu_stream(),
                    )?;
                }
            }
        }
        stream.synchronize()?;

        {
            let (block_ptrs_raw, _block_guard) = block_ptrs.device_ptr(&stream);
            let block_ptrs_mut = block_ptrs_raw as usize as *const *mut c_void;
            let (universal_ptrs_raw, _univ_guard) = universal_ptrs.device_ptr(&stream);
            let universal_ptrs_const = universal_ptrs_raw as usize as *const *const c_void;
            let status = unsafe {
                super::block_from_universal(
                    universal_ptrs_const,
                    block_ptrs_mut,
                    num_blocks,
                    nh,
                    nl,
                    no,
                    nt,
                    hd,
                    dtype,
                    layout,
                    stream_raw,
                )
            };
            assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);
        }
        stream.synchronize()?;

        for block_idx in 0..num_blocks {
            for chunk_idx in 0..chunk_count {
                let host_chunk = stream.clone_dtoh(&block_slices[block_idx][chunk_idx])?;
                for (inner_idx, value) in host_chunk.iter().enumerate() {
                    let expected = host_block_chunks[block_idx][chunk_idx][inner_idx];
                    assert!(
                        (value - expected).abs() < 1e-5,
                        "block mismatch block {} chunk {} offset {}: {} vs {}",
                        block_idx,
                        chunk_idx,
                        inner_idx,
                        value,
                        expected
                    );
                }
            }
        }

        Ok(())
    }

    /// Test the vectorized copy kernel directly with aligned data.
    #[test]
    fn test_vectorized_copy_aligned() -> Result<(), DriverError> {
        let device_count = match CudaContext::device_count() {
            Ok(count) => count,
            Err(_) => return Ok(()),
        };
        if device_count <= 0 {
            return Ok(());
        }

        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let stream_raw = stream.cu_stream() as cuda_runtime::cudaStream_t;

        // Create test data - 8-byte aligned for vectorized copy
        let num_pairs = 4;
        let copy_size = 256usize; // 256 bytes, divisible by 16 for int4 vectorization

        // Source data
        let mut src_slices = Vec::with_capacity(num_pairs);
        let mut src_ptr_values = Vec::with_capacity(num_pairs);
        let mut expected_data = Vec::with_capacity(num_pairs);

        for i in 0..num_pairs {
            let data: Vec<u8> = (0..copy_size)
                .map(|j| ((i * copy_size + j) % 256) as u8)
                .collect();
            expected_data.push(data.clone());
            let slice = stream.clone_htod(&data)?;
            {
                let (ptr, _guard) = slice.device_ptr(&stream);
                src_ptr_values.push(ptr as usize);
            }
            src_slices.push(slice);
        }

        // Destination buffers
        let mut dst_slices = Vec::with_capacity(num_pairs);
        let mut dst_ptr_values = Vec::with_capacity(num_pairs);

        for _ in 0..num_pairs {
            let mut slice = unsafe { stream.alloc::<u8>(copy_size)? };
            {
                let (ptr, _guard) = slice.device_ptr_mut(&stream);
                dst_ptr_values.push(ptr as usize);
            }
            dst_slices.push(slice);
        }

        // Upload pointer arrays to device
        let src_ptrs = stream.clone_htod(&src_ptr_values)?;
        let dst_ptrs = stream.clone_htod(&dst_ptr_values)?;

        // Launch vectorized copy
        {
            let (src_ptrs_raw, _src_guard) = src_ptrs.device_ptr(&stream);
            let (dst_ptrs_raw, _dst_guard) = dst_ptrs.device_ptr(&stream);

            let status = unsafe {
                super::vectorized_copy(
                    src_ptrs_raw as usize as *mut *mut c_void,
                    dst_ptrs_raw as usize as *mut *mut c_void,
                    copy_size,
                    num_pairs as i32,
                    stream_raw,
                )
            };
            assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);
        }
        stream.synchronize()?;

        // Verify results
        for i in 0..num_pairs {
            let result = stream.clone_dtoh(&dst_slices[i])?;
            assert_eq!(result, expected_data[i], "Mismatch at pair {}", i);
        }

        Ok(())
    }
}
