// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `memcpy_batch` and the always-available query helpers
//! (`is_memcpy_batch_available`, `is_using_stubs`).
//!
//! These don't require `permute_kernels` — the functions are unconditionally
//! linked regardless of feature flags.

#![cfg(all(feature = "testing-cuda", not(stub_kernels)))]

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DriverError};
use cudarc::runtime::sys as cuda_runtime;
use dynamo_kvbm_kernels::{is_memcpy_batch_available, is_using_stubs, memcpy_batch};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cuda_setup() -> Option<(Arc<CudaStream>, cuda_runtime::cudaStream_t)> {
    let count = CudaContext::device_count().ok()?;
    if count == 0 {
        return None;
    }
    let ctx = CudaContext::new(0).ok()?;
    let stream = ctx.default_stream();
    let raw = stream.cu_stream() as cuda_runtime::cudaStream_t;
    Some((stream, raw))
}

/// Allocate a device buffer, fill it with `data`, and return the slice + its
/// raw device address.
fn upload(stream: &Arc<CudaStream>, data: &[u8]) -> Result<(CudaSlice<u8>, usize), DriverError> {
    let slice = stream.clone_htod(data)?;
    let addr = {
        let (ptr, _guard) = slice.device_ptr(stream);
        ptr as usize
    };
    Ok((slice, addr))
}

/// Allocate `len` zero bytes on device, return slice + raw device address.
fn alloc_zeroed(
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

    let status = unsafe {
        memcpy_batch(
            std::ptr::null(),
            std::ptr::null(),
            128,
            0, // num_copies = 0
            raw,
        )
    };
    assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);
    Ok(())
}

#[test]
fn memcpy_batch_zero_size_noop() -> Result<(), DriverError> {
    let (_stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let status = unsafe {
        memcpy_batch(
            std::ptr::null(),
            std::ptr::null(),
            0, // size_per_copy = 0
            5,
            raw,
        )
    };
    assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);
    Ok(())
}

// ---------------------------------------------------------------------------
// memcpy_batch functional tests — gated on is_memcpy_batch_available()
// ---------------------------------------------------------------------------

/// Single device-to-device copy via `memcpy_batch`.
#[test]
fn memcpy_batch_single_copy() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };
    if !is_memcpy_batch_available() {
        eprintln!("Skipping: cudaMemcpyBatchAsync not available");
        return Ok(());
    }

    let data: Vec<u8> = (0..256u16).map(|i| (i % 256) as u8).collect();
    let (_src_slice, src_addr) = upload(&stream, &data)?;
    let (dst_slice, dst_addr) = alloc_zeroed(&stream, data.len())?;

    let src_ptrs: [*const c_void; 1] = [src_addr as *const c_void];
    let dst_ptrs: [*mut c_void; 1] = [dst_addr as *mut c_void];

    let status = unsafe {
        memcpy_batch(
            src_ptrs.as_ptr() as *const *const c_void,
            dst_ptrs.as_ptr() as *const *mut c_void,
            data.len(),
            1,
            raw,
        )
    };
    assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);
    stream.synchronize()?;

    let result = stream.clone_dtoh(&dst_slice)?;
    assert_eq!(result, data);
    Ok(())
}

/// Multiple independent copies in one batch call.
#[test]
fn memcpy_batch_multiple_copies() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };
    if !is_memcpy_batch_available() {
        eprintln!("Skipping: cudaMemcpyBatchAsync not available");
        return Ok(());
    }

    let num_pairs = 8;
    let copy_size = 512;

    let mut src_slices = Vec::with_capacity(num_pairs);
    let mut dst_slices = Vec::with_capacity(num_pairs);
    let mut src_ptrs: Vec<*const c_void> = Vec::with_capacity(num_pairs);
    let mut dst_ptrs: Vec<*mut c_void> = Vec::with_capacity(num_pairs);
    let mut expected: Vec<Vec<u8>> = Vec::with_capacity(num_pairs);

    for i in 0..num_pairs {
        let data: Vec<u8> = (0..copy_size)
            .map(|j| ((i * 31 + j * 7) % 256) as u8)
            .collect();
        let (s, sa) = upload(&stream, &data)?;
        let (d, da) = alloc_zeroed(&stream, copy_size)?;
        src_ptrs.push(sa as *const c_void);
        dst_ptrs.push(da as *mut c_void);
        src_slices.push(s);
        dst_slices.push(d);
        expected.push(data);
    }

    let status = unsafe {
        memcpy_batch(
            src_ptrs.as_ptr() as *const *const c_void,
            dst_ptrs.as_ptr() as *const *mut c_void,
            copy_size,
            num_pairs,
            raw,
        )
    };
    assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);
    stream.synchronize()?;

    for (i, (dst, exp)) in dst_slices.iter().zip(expected.iter()).enumerate() {
        let result = stream.clone_dtoh(dst)?;
        assert_eq!(result, *exp, "mismatch at pair {i}");
    }
    Ok(())
}

/// Large copy to exercise alignment / vectorization paths.
#[test]
fn memcpy_batch_large_copy() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };
    if !is_memcpy_batch_available() {
        eprintln!("Skipping: cudaMemcpyBatchAsync not available");
        return Ok(());
    }

    let copy_size = 1 << 20; // 1 MiB
    let num_pairs = 3;

    let mut src_slices = Vec::with_capacity(num_pairs);
    let mut dst_slices = Vec::with_capacity(num_pairs);
    let mut src_ptrs: Vec<*const c_void> = Vec::with_capacity(num_pairs);
    let mut dst_ptrs: Vec<*mut c_void> = Vec::with_capacity(num_pairs);

    for i in 0..num_pairs {
        let data: Vec<u8> = (0..copy_size).map(|j| ((i + j) % 251) as u8).collect();
        let (s, sa) = upload(&stream, &data)?;
        let (d, da) = alloc_zeroed(&stream, copy_size)?;
        src_ptrs.push(sa as *const c_void);
        dst_ptrs.push(da as *mut c_void);
        src_slices.push(s);
        dst_slices.push(d);
    }

    let status = unsafe {
        memcpy_batch(
            src_ptrs.as_ptr() as *const *const c_void,
            dst_ptrs.as_ptr() as *const *mut c_void,
            copy_size,
            num_pairs,
            raw,
        )
    };
    assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);
    stream.synchronize()?;

    for (i, (src, dst)) in src_slices.iter().zip(dst_slices.iter()).enumerate() {
        let src_host = stream.clone_dtoh(src)?;
        let dst_host = stream.clone_dtoh(dst)?;
        assert_eq!(src_host, dst_host, "mismatch at pair {i}");
    }
    Ok(())
}

/// Non-power-of-two copy size (regression guard for alignment assumptions).
#[test]
fn memcpy_batch_odd_size() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };
    if !is_memcpy_batch_available() {
        eprintln!("Skipping: cudaMemcpyBatchAsync not available");
        return Ok(());
    }

    let copy_size = 999; // not aligned to anything useful
    let num_pairs = 4;

    let mut src_slices = Vec::with_capacity(num_pairs);
    let mut dst_slices = Vec::with_capacity(num_pairs);
    let mut src_ptrs: Vec<*const c_void> = Vec::with_capacity(num_pairs);
    let mut dst_ptrs: Vec<*mut c_void> = Vec::with_capacity(num_pairs);
    let mut expected: Vec<Vec<u8>> = Vec::with_capacity(num_pairs);

    for i in 0..num_pairs {
        let data: Vec<u8> = (0..copy_size).map(|j| ((i * 13 + j) % 256) as u8).collect();
        let (s, sa) = upload(&stream, &data)?;
        let (d, da) = alloc_zeroed(&stream, copy_size)?;
        src_ptrs.push(sa as *const c_void);
        dst_ptrs.push(da as *mut c_void);
        src_slices.push(s);
        dst_slices.push(d);
        expected.push(data);
    }

    let status = unsafe {
        memcpy_batch(
            src_ptrs.as_ptr() as *const *const c_void,
            dst_ptrs.as_ptr() as *const *mut c_void,
            copy_size,
            num_pairs,
            raw,
        )
    };
    assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);
    stream.synchronize()?;

    for (i, (dst, exp)) in dst_slices.iter().zip(expected.iter()).enumerate() {
        let result = stream.clone_dtoh(dst)?;
        assert_eq!(result, *exp, "mismatch at pair {i}");
    }
    Ok(())
}

/// Many small pairs to stress the batch dispatch path.
#[test]
fn memcpy_batch_many_pairs() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };
    if !is_memcpy_batch_available() {
        eprintln!("Skipping: cudaMemcpyBatchAsync not available");
        return Ok(());
    }

    let num_pairs = 256;
    let copy_size = 64;

    let mut src_slices = Vec::with_capacity(num_pairs);
    let mut dst_slices = Vec::with_capacity(num_pairs);
    let mut src_ptrs: Vec<*const c_void> = Vec::with_capacity(num_pairs);
    let mut dst_ptrs: Vec<*mut c_void> = Vec::with_capacity(num_pairs);

    for i in 0..num_pairs {
        let data: Vec<u8> = (0..copy_size).map(|j| ((i + j) % 256) as u8).collect();
        let (s, sa) = upload(&stream, &data)?;
        let (d, da) = alloc_zeroed(&stream, copy_size)?;
        src_ptrs.push(sa as *const c_void);
        dst_ptrs.push(da as *mut c_void);
        src_slices.push(s);
        dst_slices.push(d);
    }

    let status = unsafe {
        memcpy_batch(
            src_ptrs.as_ptr() as *const *const c_void,
            dst_ptrs.as_ptr() as *const *mut c_void,
            copy_size,
            num_pairs,
            raw,
        )
    };
    assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);
    stream.synchronize()?;

    // Spot-check a few pairs rather than all 256.
    for i in [0, 1, 127, 255] {
        let src_host = stream.clone_dtoh(&src_slices[i])?;
        let dst_host = stream.clone_dtoh(&dst_slices[i])?;
        assert_eq!(src_host, dst_host, "mismatch at pair {i}");
    }
    Ok(())
}
