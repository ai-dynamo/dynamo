// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `memcpy_batch` and the always-available query helpers
//! (`is_memcpy_batch_available`, `is_using_stubs`).
//!
//! These don't require `permute_kernels` — the functions are unconditionally
//! linked regardless of feature flags.
//!
//! Functional tests use pinned-host -> device -> pinned-host roundtrips (H2D + D2H)
//! to match the transfer patterns that `cudaMemcpyBatchAsync` is designed for.

#![cfg(all(feature = "testing-cuda", not(stub_kernels)))]

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DriverError};
use cudarc::runtime::sys as cuda_runtime;
use kvbm_kernels::{MemcpyBatchMode, is_memcpy_batch_available, is_using_stubs, memcpy_batch};

// Direct FFI for cudaMallocHost / cudaFreeHost.
// We bypass cudarc's runtime::sys because cudarc eagerly resolves ALL runtime
// symbols on first use, and CUDA 13.x removed `cudaGetDeviceProperties_v2`
// which causes a panic.  Our test binary links against libcudart directly
// (through kvbm-kernels' build.rs), so these symbols are always available.
unsafe extern "C" {
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> u32;
    fn cudaFreeHost(ptr: *mut c_void) -> u32;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cuda_setup() -> Option<(Arc<CudaStream>, cuda_runtime::cudaStream_t)> {
    let count = CudaContext::device_count().ok()?;
    if count == 0 {
        return None;
    }
    let ctx = CudaContext::new(0).ok()?;
    // Use a non-default stream — cudaMemcpyBatchAsync does not accept the
    // NULL (default) stream.  A real CUstream from the driver API works fine.
    let stream = ctx.new_stream().ok()?;
    let raw = stream.cu_stream() as cuda_runtime::cudaStream_t;
    Some((stream, raw))
}

/// Allocate `len` zero bytes on device, return slice + raw device address.
fn alloc_device_zeroed(
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<(CudaSlice<u8>, usize), DriverError> {
    let slice = stream.alloc_zeros::<u8>(len)?;
    let addr = {
        let (ptr, _guard) = slice.device_ptr(stream);
        ptr as usize
    };
    Ok((slice, addr))
}

/// RAII wrapper around pinned host memory allocated with `cudaMallocHost`.
struct PinnedBuffer {
    ptr: *mut c_void,
    len: usize,
}

impl PinnedBuffer {
    /// Allocate `len` bytes of pinned host memory, zeroed.
    fn new_zeroed(len: usize) -> Self {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { cudaMallocHost(&mut ptr, len) };
        assert_eq!(err, 0, "cudaMallocHost failed with error {err}");
        // Zero the buffer
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, len) };
        Self { ptr, len }
    }

    /// Allocate pinned host memory and fill it from `data`.
    fn from_data(data: &[u8]) -> Self {
        let buf = Self::new_zeroed(data.len());
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.ptr as *mut u8, data.len());
        }
        buf
    }

    fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    fn as_const_ptr(&self) -> *const c_void {
        self.ptr as *const c_void
    }

    /// Read contents back as a `Vec<u8>`.
    fn to_vec(&self) -> Vec<u8> {
        let mut v = vec![0u8; self.len];
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr as *const u8, v.as_mut_ptr(), self.len);
        }
        v
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFreeHost(self.ptr);
            }
        }
    }
}

/// Run a pinned-host -> device -> pinned-host roundtrip via two `memcpy_batch` calls.
///
/// 1. Batch H2D: copy from `src_pinned` buffers to `device` buffers
/// 2. Batch D2H: copy from `device` buffers to `dst_pinned` (zeroed) buffers
/// 3. Assert `dst_pinned` contents match original data
///
/// Returns both batch statuses for the caller to inspect.
fn h2d_d2h_roundtrip(
    stream: &Arc<CudaStream>,
    raw: cuda_runtime::cudaStream_t,
    data_sets: &[Vec<u8>],
    copy_size: usize,
    mode: MemcpyBatchMode,
) -> Result<(cuda_runtime::cudaError, cuda_runtime::cudaError), DriverError> {
    let num_pairs = data_sets.len();

    // Source: pinned host buffers filled with known data
    let src_pinned: Vec<PinnedBuffer> = data_sets
        .iter()
        .map(|d| PinnedBuffer::from_data(d))
        .collect();

    // Device buffers (zeroed)
    let mut dev_slices = Vec::with_capacity(num_pairs);
    let mut dev_addrs = Vec::with_capacity(num_pairs);
    for _ in 0..num_pairs {
        let (s, a) = alloc_device_zeroed(stream, copy_size)?;
        dev_slices.push(s);
        dev_addrs.push(a);
    }

    // Destination: zeroed pinned host buffers
    let dst_pinned: Vec<PinnedBuffer> = (0..num_pairs)
        .map(|_| PinnedBuffer::new_zeroed(copy_size))
        .collect();

    // Build pointer arrays for H2D: src = pinned host, dst = device
    let h2d_src_ptrs: Vec<*const c_void> = src_pinned.iter().map(|b| b.as_const_ptr()).collect();
    let h2d_dst_ptrs: Vec<*mut c_void> = dev_addrs.iter().map(|&a| a as *mut c_void).collect();

    let h2d_status = unsafe {
        memcpy_batch(
            h2d_src_ptrs.as_ptr() as *const *const c_void,
            h2d_dst_ptrs.as_ptr() as *const *mut c_void,
            copy_size,
            num_pairs,
            mode,
            raw,
        )
    };

    if h2d_status != cuda_runtime::cudaError::cudaSuccess {
        return Ok((h2d_status, cuda_runtime::cudaError::cudaSuccess));
    }

    // Build pointer arrays for D2H: src = device, dst = pinned host
    let d2h_src_ptrs: Vec<*const c_void> = dev_addrs.iter().map(|&a| a as *const c_void).collect();
    let d2h_dst_ptrs: Vec<*mut c_void> = dst_pinned.iter().map(|b| b.as_ptr()).collect();

    let d2h_status = unsafe {
        memcpy_batch(
            d2h_src_ptrs.as_ptr() as *const *const c_void,
            d2h_dst_ptrs.as_ptr() as *const *mut c_void,
            copy_size,
            num_pairs,
            mode,
            raw,
        )
    };

    if d2h_status != cuda_runtime::cudaError::cudaSuccess {
        return Ok((h2d_status, d2h_status));
    }

    // Synchronize before reading back
    stream.synchronize()?;

    // Verify roundtrip: dst_pinned should match original data
    for (i, (dst, expected)) in dst_pinned.iter().zip(data_sets.iter()).enumerate() {
        let result = dst.to_vec();
        assert_eq!(
            result,
            *expected,
            "roundtrip mismatch at pair {i}: first differing byte at position {}",
            result
                .iter()
                .zip(expected.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(result.len())
        );
    }

    Ok((h2d_status, d2h_status))
}

// ---------------------------------------------------------------------------
// Query function tests
// ---------------------------------------------------------------------------

#[test]
fn stubs_not_active() {
    // Since the file is gated on not(stub_kernels), this must be false.
    assert!(!is_using_stubs());
}

#[test]
fn availability_is_consistent() {
    // Just ensure it doesn't crash and returns a stable value.
    let a = is_memcpy_batch_available();
    let b = is_memcpy_batch_available();
    assert_eq!(a, b);
    eprintln!(
        "cudaMemcpyBatchAsync available: {} (CUDA {}12.9)",
        a,
        if a { ">=" } else { "<" }
    );
}

// ---------------------------------------------------------------------------
// memcpy_batch edge cases (work regardless of CUDA version)
// ---------------------------------------------------------------------------

#[test]
fn memcpy_batch_zero_copies_noop() -> Result<(), DriverError> {
    let (_stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    // All modes should treat zero copies as a no-op
    for mode in [
        MemcpyBatchMode::BatchedWithFallback,
        MemcpyBatchMode::FallbackOnly,
        MemcpyBatchMode::BatchWithoutFallback,
    ] {
        let status = unsafe {
            memcpy_batch(
                std::ptr::null(),
                std::ptr::null(),
                128,
                0, // num_copies = 0
                mode,
                raw,
            )
        };
        assert_eq!(
            status,
            cuda_runtime::cudaError::cudaSuccess,
            "mode={mode:?}"
        );
    }
    Ok(())
}

#[test]
fn memcpy_batch_zero_size_noop() -> Result<(), DriverError> {
    let (_stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    // All modes should treat zero size as a no-op
    for mode in [
        MemcpyBatchMode::BatchedWithFallback,
        MemcpyBatchMode::FallbackOnly,
        MemcpyBatchMode::BatchWithoutFallback,
    ] {
        let status = unsafe {
            memcpy_batch(
                std::ptr::null(),
                std::ptr::null(),
                0, // size_per_copy = 0
                5,
                mode,
                raw,
            )
        };
        assert_eq!(
            status,
            cuda_runtime::cudaError::cudaSuccess,
            "mode={mode:?}"
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// memcpy_batch functional tests — H2D + D2H roundtrip with pinned memory
//
// Each test runs across all three MemcpyBatchMode variants:
// - BatchedWithFallback: always works (batch or fallback)
// - FallbackOnly: always works (individual cudaMemcpyAsync)
// - BatchWithoutFallback: only works when batch API is available (CUDA 12.9+)
// ---------------------------------------------------------------------------

/// Run a roundtrip test across all modes, handling BatchWithoutFallback gracefully
/// when the batch API is not available.
fn run_all_modes(
    stream: &Arc<CudaStream>,
    raw: cuda_runtime::cudaStream_t,
    data_sets: &[Vec<u8>],
    copy_size: usize,
) -> Result<(), DriverError> {
    let batch_available = is_memcpy_batch_available();

    for mode in [
        MemcpyBatchMode::BatchedWithFallback,
        MemcpyBatchMode::FallbackOnly,
        MemcpyBatchMode::BatchWithoutFallback,
    ] {
        let (h2d, d2h) = h2d_d2h_roundtrip(stream, raw, data_sets, copy_size, mode)?;

        if mode == MemcpyBatchMode::BatchWithoutFallback && !batch_available {
            // Expected to fail when batch API is not available
            eprintln!("  {mode:?}: batch API unavailable, got h2d={h2d:?} (expected non-success)");
            continue;
        }

        assert_eq!(
            h2d,
            cuda_runtime::cudaError::cudaSuccess,
            "H2D failed with mode={mode:?}"
        );
        assert_eq!(
            d2h,
            cuda_runtime::cudaError::cudaSuccess,
            "D2H failed with mode={mode:?}"
        );
        eprintln!("  {mode:?}: OK");
    }
    Ok(())
}

/// Single H2D + D2H roundtrip via `memcpy_batch` (all modes).
#[test]
fn memcpy_batch_single_copy() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let copy_size = 256;
    let data: Vec<u8> = (0..copy_size as u16).map(|i| (i % 256) as u8).collect();
    run_all_modes(&stream, raw, &[data], copy_size)
}

/// Multiple independent H2D + D2H roundtrips in one batch call (all modes).
#[test]
fn memcpy_batch_multiple_copies() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let num_pairs = 8;
    let copy_size = 512;
    let data_sets: Vec<Vec<u8>> = (0..num_pairs)
        .map(|i| {
            (0..copy_size)
                .map(|j| ((i * 31 + j * 7) % 256) as u8)
                .collect()
        })
        .collect();

    run_all_modes(&stream, raw, &data_sets, copy_size)
}

/// Large copy (1 MiB per pair) to exercise alignment paths (all modes).
#[test]
fn memcpy_batch_large_copy() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let copy_size = 1 << 20; // 1 MiB
    let num_pairs = 3;
    let data_sets: Vec<Vec<u8>> = (0..num_pairs)
        .map(|i| (0..copy_size).map(|j| ((i + j) % 251) as u8).collect())
        .collect();

    run_all_modes(&stream, raw, &data_sets, copy_size)
}

/// Non-power-of-two copy size (regression guard for alignment assumptions, all modes).
#[test]
fn memcpy_batch_odd_size() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let copy_size = 999; // not aligned to anything useful
    let num_pairs = 4;
    let data_sets: Vec<Vec<u8>> = (0..num_pairs)
        .map(|i| (0..copy_size).map(|j| ((i * 13 + j) % 256) as u8).collect())
        .collect();

    run_all_modes(&stream, raw, &data_sets, copy_size)
}

/// Many small pairs to stress the batch dispatch path (all modes).
#[test]
fn memcpy_batch_many_pairs() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let num_pairs = 256;
    let copy_size = 64;
    let data_sets: Vec<Vec<u8>> = (0..num_pairs)
        .map(|i| (0..copy_size).map(|j| ((i + j) % 256) as u8).collect())
        .collect();

    run_all_modes(&stream, raw, &data_sets, copy_size)
}

// ---------------------------------------------------------------------------
// Diagnostic: mirrors NVIDIA benchmark calling pattern exactly
// ---------------------------------------------------------------------------
