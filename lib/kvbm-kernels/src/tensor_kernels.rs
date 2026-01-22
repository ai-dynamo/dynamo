// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Safe-ish wrappers around the CUDA block/universal packing kernels.
//!
//! The core ideas:
//! * A “block” represents the stack of `nl * no` tensors arranged either as NHD
//!   (inner axes `[nt, nh, hd]`) or HND (inner axes `[nh, nt, hd]`).
//! * A “universal” tensor is `[nh, nl, no, nt, hd]` stored contiguously.
//! * An “operational” tensor is `[nl, no, inner]` with `inner = nt * nh * hd`.
//!
//! Host code calls these helpers with flattened pointer tables so a single
//! launch can move many logical blocks in one go.

#![allow(dead_code)]
#![allow(clippy::missing_safety_doc)]
use std::ffi::c_void;

use cudarc::runtime::sys::{cudaError_t, cudaStream_t};

/// Numeric tags passed across the FFI boundary to select the CUDA template.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorDataType {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
    F64 = 3,
}

/// Identifies how each `[nt, nh, hd]` chunk is laid out in device memory.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockLayout {
    NHD = 0,
    HND = 1,
}

/// Direction flag for copying between block stacks and operational buffers.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationalCopyDirection {
    BlockToOperational = 0,
    OperationalToBlock = 1,
}

/// Selects how the operational copy should move data.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationalCopyBackend {
    /// Auto-select the best backend based on data alignment and CUDA version.
    /// Priority: vectorized kernel (if 8-byte aligned) -> batch copy (CUDA 12.9+) -> memcpy async.
    Auto = 0,
    /// Force the vectorized 64-bit kernel (requires 8-byte aligned data).
    VectorizedKernel = 1,
    /// Force the dtype-specific kernel path.
    KernelOnly = 2,
    /// Issue one cudaMemcpyAsync per chunk.
    MemcpyAsync = 3,
    /// Invoke cudaMemcpyBatchAsync directly.
    MemcpyBatch = 4,
}

unsafe extern "C" {
    fn kvbm_kernels_launch_universal_from_block(
        universal_ptrs_device: *const *mut c_void,
        block_ptrs_device: *const *const c_void,
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
        universal_ptrs_device: *const *const c_void,
        block_ptrs_device: *const *mut c_void,
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

    fn kvbm_kernels_launch_operational_copy(
        block_ptrs_host: *const *const c_void,
        block_ptrs_device: *const *const c_void,
        operational_ptrs_host: *const *mut c_void,
        operational_ptrs_device: *const *const c_void,
        num_blocks: usize,
        nl: usize,
        no: usize,
        inner: usize,
        elem_size: usize,
        dtype: i32,
        direction: i32,
        backend: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn kvbm_kernels_launch_vectorized_copy(
        src_ptrs_device: *mut *mut c_void,
        dst_ptrs_device: *mut *mut c_void,
        copy_size_bytes: usize,
        num_pairs: i32,
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
///     if dynamo_kvbm_kernels::is_using_stubs() {
///         eprintln!("Skipping CUDA test: stub kernels in use");
///         return;
///     }
///     // ... actual CUDA test code ...
/// }
/// ```
pub fn is_using_stubs() -> bool {
    unsafe { kvbm_kernels_is_stub_build() }
}

/// Copy `num_blocks` stacks of NHD/HND tensors into universal form.
///
/// * `universal_device_ptrs` – device pointer to `num_blocks` universal bases.
/// * `block_device_ptrs` – device pointer to a flattened `[num_blocks][nl*no]`
///   table of chunk pointers.
/// * `nh, nl, no, nt, hd` – logical dimensions of each universal tensor.
/// * `stream` – CUDA stream used for the launch.
#[allow(clippy::too_many_arguments)]
pub unsafe fn universal_from_block(
    universal_device_ptrs: *const *mut c_void,
    block_device_ptrs: *const *const c_void,
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
            universal_device_ptrs,
            block_device_ptrs,
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
#[allow(clippy::too_many_arguments)]
pub unsafe fn block_from_universal(
    universal_device_ptrs: *const *const c_void,
    block_device_ptrs: *const *mut c_void,
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
            universal_device_ptrs,
            block_device_ptrs,
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

/// Launch vectorized copy between arbitrary pointer pairs.
///
/// This kernel automatically selects optimal vectorization (4/8/16 bytes) based on
/// pointer alignment. It is useful for copying between non-contiguous memory regions
/// where each pair has the same copy size.
///
/// # Arguments
/// * `src_ptrs_device` - Device pointer to array of source pointers
/// * `dst_ptrs_device` - Device pointer to array of destination pointers
/// * `copy_size_bytes` - Size of each copy in bytes
/// * `num_pairs` - Number of pointer pairs to copy
/// * `stream` - CUDA stream for async execution
///
/// # Safety
/// The caller must ensure all pointers are valid device pointers and the copy
/// sizes do not exceed the allocated memory at each pointer.
pub unsafe fn vectorized_copy(
    src_ptrs_device: *mut *mut c_void,
    dst_ptrs_device: *mut *mut c_void,
    copy_size_bytes: usize,
    num_pairs: i32,
    stream: cudaStream_t,
) -> cudaError_t {
    unsafe {
        kvbm_kernels_launch_vectorized_copy(
            src_ptrs_device,
            dst_ptrs_device,
            copy_size_bytes,
            num_pairs,
            stream,
        )
    }
}

/// Copy between block stacks and operational buffers for `num_blocks`.
///
/// In `Auto` mode, the priority order is:
/// 1. Vectorized 64-bit kernel (if data is 8-byte aligned)
/// 2. `cudaMemcpyBatchAsync` (CUDA 12.9+)
/// 3. `cudaMemcpyAsync` (fallback)
///
/// The `backend` parameter lets callers force a specific path:
/// - `Auto` - automatic selection based on alignment and CUDA version
/// - `VectorizedKernel` - force 64-bit vectorized kernel (requires 8-byte alignment)
/// - `KernelOnly` - force dtype-specific kernel
/// - `MemcpyAsync` - force per-chunk cudaMemcpyAsync
/// - `MemcpyBatch` - force cudaMemcpyBatchAsync
#[allow(clippy::too_many_arguments)]
pub unsafe fn operational_copy(
    block_ptrs_host: *const *const c_void,
    block_ptrs_device: *const *const c_void,
    operational_ptrs_host: *const *mut c_void,
    operational_ptrs_device: *const *const c_void,
    num_blocks: usize,
    nl: usize,
    no: usize,
    inner: usize,
    elem_size: usize,
    dtype: TensorDataType,
    direction: OperationalCopyDirection,
    backend: OperationalCopyBackend,
    stream: cudaStream_t,
) -> cudaError_t {
    unsafe {
        kvbm_kernels_launch_operational_copy(
            block_ptrs_host,
            block_ptrs_device,
            operational_ptrs_host,
            operational_ptrs_device,
            num_blocks,
            nl,
            no,
            inner,
            elem_size,
            dtype as i32,
            direction as i32,
            backend as i32,
            stream,
        )
    }
}

// Tests are gated to only run when:
// 1. testing-cuda feature is enabled
// 2. NOT using stub kernels (stub_kernels cfg is set by build.rs when no nvcc)
#[cfg(all(test, feature = "testing-cuda", not(stub_kernels)))]
mod tests {
    use super::*;
    use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut, DriverError};
    use cudarc::runtime::sys as cuda_runtime;

    #[test]
    fn fused_copy_roundtrip() -> Result<(), DriverError> {
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
        let operational_volume = chunk_count * inner;
        let num_blocks = 2usize;

        let dtype = TensorDataType::F32;
        let layout = BlockLayout::NHD;

        let mut host_block_chunks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_blocks);
        let mut block_slices: Vec<Vec<CudaSlice<f32>>> = Vec::with_capacity(num_blocks);
        let mut block_ptrs_host: Vec<*const c_void> = Vec::with_capacity(num_blocks * chunk_count);
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
                    block_ptrs_host.push(ptr_raw as usize as *const c_void);
                    block_ptr_values.push(ptr_raw as usize);
                }
                slices_for_block.push(slice);
                host_chunks_for_block.push(host_chunk);
            }
            block_slices.push(slices_for_block);
            host_block_chunks.push(host_chunks_for_block);
        }

        let block_ptrs_device = stream.clone_htod(block_ptr_values.as_slice())?;

        let mut universal_slices = Vec::with_capacity(num_blocks);
        let mut universal_ptr_values = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let mut slice = unsafe { stream.alloc::<f32>(block_volume)? };
            {
                let (ptr_raw, _guard) = slice.device_ptr_mut(&stream);
                universal_ptr_values.push(ptr_raw as usize);
            }
            universal_slices.push(slice);
        }
        let universal_ptrs_device = stream.clone_htod(universal_ptr_values.as_slice())?;

        let mut operational_slices = Vec::with_capacity(num_blocks);
        let mut operational_ptrs_host = Vec::with_capacity(num_blocks);
        let mut operational_ptr_values = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let mut slice = unsafe { stream.alloc::<f32>(operational_volume)? };
            {
                let (ptr_raw, _guard) = slice.device_ptr_mut(&stream);
                operational_ptrs_host.push(ptr_raw as usize as *mut c_void);
                operational_ptr_values.push(ptr_raw as usize);
            }
            operational_slices.push(slice);
        }
        let operational_ptrs_device = stream.clone_htod(operational_ptr_values.as_slice())?;

        // Block -> Universal
        {
            let (block_ptrs_device_raw, _block_guard) = block_ptrs_device.device_ptr(&stream);
            let block_ptrs_device_ptr = block_ptrs_device_raw as usize as *const *const c_void;
            let (universal_ptrs_device_raw, _univ_guard) =
                universal_ptrs_device.device_ptr(&stream);
            let universal_ptrs_device_ptr =
                universal_ptrs_device_raw as usize as *const *mut c_void;

            let status = unsafe {
                super::universal_from_block(
                    universal_ptrs_device_ptr,
                    block_ptrs_device_ptr,
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

        // Universal -> Block
        for block in &mut block_slices {
            for slice in block {
                stream.memset_zeros(slice)?;
            }
        }
        stream.synchronize()?;

        {
            let (block_ptrs_device_raw, _block_guard) = block_ptrs_device.device_ptr(&stream);
            let block_ptrs_device_mut = block_ptrs_device_raw as usize as *const *mut c_void;
            let (universal_ptrs_device_raw, _univ_guard) =
                universal_ptrs_device.device_ptr(&stream);
            let universal_ptrs_device_const =
                universal_ptrs_device_raw as usize as *const *const c_void;
            let status = unsafe {
                super::block_from_universal(
                    universal_ptrs_device_const,
                    block_ptrs_device_mut,
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

        // Block -> Operational
        {
            let (block_ptrs_device_raw, _block_guard) = block_ptrs_device.device_ptr(&stream);
            let block_ptrs_device_ptr = block_ptrs_device_raw as usize as *const *const c_void;
            let (operational_ptrs_device_raw, _op_guard) =
                operational_ptrs_device.device_ptr(&stream);
            let operational_ptrs_device_ptr =
                operational_ptrs_device_raw as usize as *const *const c_void;
            let status = unsafe {
                super::operational_copy(
                    block_ptrs_host.as_ptr(),
                    block_ptrs_device_ptr,
                    operational_ptrs_host.as_ptr(),
                    operational_ptrs_device_ptr,
                    num_blocks,
                    nl,
                    no,
                    inner,
                    std::mem::size_of::<f32>(),
                    dtype,
                    OperationalCopyDirection::BlockToOperational,
                    OperationalCopyBackend::Auto,
                    stream_raw,
                )
            };
            assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);
        }
        stream.synchronize()?;

        for block_idx in 0..num_blocks {
            let host_operational = stream.clone_dtoh(&operational_slices[block_idx])?;
            for chunk_idx in 0..chunk_count {
                for inner_idx in 0..inner {
                    let expected = host_block_chunks[block_idx][chunk_idx][inner_idx];
                    let value = host_operational[chunk_idx * inner + inner_idx];
                    assert!(
                        (value - expected).abs() < 1e-5,
                        "operational pack mismatch block {} chunk {} offset {}: {} vs {}",
                        block_idx,
                        chunk_idx,
                        inner_idx,
                        value,
                        expected
                    );
                }
            }
        }

        // Operational -> Block
        for block in &mut block_slices {
            for slice in block {
                stream.memset_zeros(slice)?;
            }
        }
        stream.synchronize()?;

        {
            let (block_ptrs_device_raw, _block_guard) = block_ptrs_device.device_ptr(&stream);
            let (operational_ptrs_device_raw, _op_guard) =
                operational_ptrs_device.device_ptr(&stream);
            let operational_ptrs_device_const =
                operational_ptrs_device_raw as usize as *const *const c_void;
            let status = unsafe {
                super::operational_copy(
                    block_ptrs_host.as_ptr(),
                    block_ptrs_device_raw as usize as *const *const c_void,
                    operational_ptrs_host.as_ptr(),
                    operational_ptrs_device_const,
                    num_blocks,
                    nl,
                    no,
                    inner,
                    std::mem::size_of::<f32>(),
                    dtype,
                    OperationalCopyDirection::OperationalToBlock,
                    OperationalCopyBackend::Auto,
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
                        "operational unpack mismatch block {} chunk {} offset {}: {} vs {}",
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
            let (ptr, _guard) = slice.device_ptr(&stream);
            src_ptr_values.push(ptr as usize);
            src_slices.push(slice);
        }

        // Destination buffers
        let mut dst_slices = Vec::with_capacity(num_pairs);
        let mut dst_ptr_values = Vec::with_capacity(num_pairs);

        for _ in 0..num_pairs {
            let mut slice = unsafe { stream.alloc::<u8>(copy_size)? };
            let (ptr, _guard) = slice.device_ptr_mut(&stream);
            dst_ptr_values.push(ptr as usize);
            dst_slices.push(slice);
        }

        // Upload pointer arrays to device
        let src_ptrs_device = stream.clone_htod(&src_ptr_values)?;
        let dst_ptrs_device = stream.clone_htod(&dst_ptr_values)?;

        // Launch vectorized copy
        {
            let (src_ptrs_raw, _src_guard) = src_ptrs_device.device_ptr(&stream);
            let (dst_ptrs_raw, _dst_guard) = dst_ptrs_device.device_ptr(&stream);

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

    /// Test operational copy with explicit VectorizedKernel backend.
    #[test]
    fn test_operational_copy_vectorized_backend() -> Result<(), DriverError> {
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

        // Use dimensions that result in 8-byte aligned data for vectorized path
        let nl = 2usize;
        let no = 2usize;
        let inner = 32usize; // 32 elements * 4 bytes = 128 bytes (8-byte aligned)
        let chunk_count = nl * no;
        let operational_volume = chunk_count * inner;
        let num_blocks = 2usize;

        let dtype = TensorDataType::F32;

        // Create block chunks
        let mut host_block_chunks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_blocks);
        let mut block_slices: Vec<Vec<CudaSlice<f32>>> = Vec::with_capacity(num_blocks);
        let mut block_ptrs_host: Vec<*const c_void> = Vec::with_capacity(num_blocks * chunk_count);
        let mut block_ptr_values: Vec<usize> = Vec::with_capacity(num_blocks * chunk_count);

        for block_idx in 0..num_blocks {
            let mut host_chunks_for_block = Vec::with_capacity(chunk_count);
            let mut slices_for_block = Vec::with_capacity(chunk_count);
            for chunk_idx in 0..chunk_count {
                let global_idx = block_idx * chunk_count + chunk_idx;
                let mut host_chunk = Vec::with_capacity(inner);
                for offset in 0..inner {
                    host_chunk.push((global_idx * inner + offset) as f32 + 0.5f32);
                }
                let slice = stream.clone_htod(&host_chunk)?;
                {
                    let (ptr_raw, _guard) = slice.device_ptr(&stream);
                    block_ptrs_host.push(ptr_raw as usize as *const c_void);
                    block_ptr_values.push(ptr_raw as usize);
                }
                slices_for_block.push(slice);
                host_chunks_for_block.push(host_chunk);
            }
            block_slices.push(slices_for_block);
            host_block_chunks.push(host_chunks_for_block);
        }

        let block_ptrs_device = stream.clone_htod(block_ptr_values.as_slice())?;

        // Create operational buffers
        let mut operational_slices = Vec::with_capacity(num_blocks);
        let mut operational_ptrs_host = Vec::with_capacity(num_blocks);
        let mut operational_ptr_values = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let mut slice = unsafe { stream.alloc::<f32>(operational_volume)? };
            {
                let (ptr_raw, _guard) = slice.device_ptr_mut(&stream);
                operational_ptrs_host.push(ptr_raw as usize as *mut c_void);
                operational_ptr_values.push(ptr_raw as usize);
            }
            operational_slices.push(slice);
        }
        let operational_ptrs_device = stream.clone_htod(operational_ptr_values.as_slice())?;

        // Block -> Operational using VectorizedKernel backend
        {
            let (block_ptrs_device_raw, _block_guard) = block_ptrs_device.device_ptr(&stream);
            let block_ptrs_device_ptr = block_ptrs_device_raw as usize as *const *const c_void;
            let (operational_ptrs_device_raw, _op_guard) =
                operational_ptrs_device.device_ptr(&stream);
            let operational_ptrs_device_ptr =
                operational_ptrs_device_raw as usize as *const *const c_void;
            let status = unsafe {
                super::operational_copy(
                    block_ptrs_host.as_ptr(),
                    block_ptrs_device_ptr,
                    operational_ptrs_host.as_ptr(),
                    operational_ptrs_device_ptr,
                    num_blocks,
                    nl,
                    no,
                    inner,
                    std::mem::size_of::<f32>(),
                    dtype,
                    OperationalCopyDirection::BlockToOperational,
                    OperationalCopyBackend::VectorizedKernel,
                    stream_raw,
                )
            };
            assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);
        }
        stream.synchronize()?;

        // Verify pack results
        for block_idx in 0..num_blocks {
            let host_operational = stream.clone_dtoh(&operational_slices[block_idx])?;
            for chunk_idx in 0..chunk_count {
                for inner_idx in 0..inner {
                    let expected = host_block_chunks[block_idx][chunk_idx][inner_idx];
                    let value = host_operational[chunk_idx * inner + inner_idx];
                    assert!(
                        (value - expected).abs() < 1e-5,
                        "vectorized pack mismatch block {} chunk {} offset {}: {} vs {}",
                        block_idx,
                        chunk_idx,
                        inner_idx,
                        value,
                        expected
                    );
                }
            }
        }

        // Clear block data and test unpack
        for block in &mut block_slices {
            for slice in block {
                stream.memset_zeros(slice)?;
            }
        }
        stream.synchronize()?;

        // Operational -> Block using VectorizedKernel backend
        {
            let (block_ptrs_device_raw, _block_guard) = block_ptrs_device.device_ptr(&stream);
            let (operational_ptrs_device_raw, _op_guard) =
                operational_ptrs_device.device_ptr(&stream);
            let status = unsafe {
                super::operational_copy(
                    block_ptrs_host.as_ptr(),
                    block_ptrs_device_raw as usize as *const *const c_void,
                    operational_ptrs_host.as_ptr(),
                    operational_ptrs_device_raw as usize as *const *const c_void,
                    num_blocks,
                    nl,
                    no,
                    inner,
                    std::mem::size_of::<f32>(),
                    dtype,
                    OperationalCopyDirection::OperationalToBlock,
                    OperationalCopyBackend::VectorizedKernel,
                    stream_raw,
                )
            };
            assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);
        }
        stream.synchronize()?;

        // Verify unpack results
        for block_idx in 0..num_blocks {
            for chunk_idx in 0..chunk_count {
                let host_chunk = stream.clone_dtoh(&block_slices[block_idx][chunk_idx])?;
                for (inner_idx, value) in host_chunk.iter().enumerate() {
                    let expected = host_block_chunks[block_idx][chunk_idx][inner_idx];
                    assert!(
                        (value - expected).abs() < 1e-5,
                        "vectorized unpack mismatch block {} chunk {} offset {}: {} vs {}",
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
}
