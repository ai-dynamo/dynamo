// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for SYCL XPU tensor permute kernel roundtrips.
//!
//! Mirrors kernel_roundtrip.rs (CUDA) using the oneAPI-rs safe SYCL API:
//! - Memory: SyclContext::alloc_device<T> (typed USM slices)
//! - Transfer: SyclQueue::memcpy_sync (auto-detect direction)
//! - Kernel dispatch: sycl_universal_from_block / sycl_block_from_universal
//!   via FFI with queue.raw_queue_ptr()
//! - Sync: SyclQueue::synchronize
//!
//! Run with:
//!   cargo test -p kvbm-kernels --features testing-xpu-sycl \
//!       --test sycl_kernel_roundtrip -- --nocapture

#![cfg(all(feature = "testing-xpu-sycl", feature = "xpu-sycl-permute"))]

use std::ffi::c_void;
use std::fmt::Debug;
use std::sync::Arc;

use half::{f16, bf16};
use kvbm_kernels::{
    BlockLayout, sycl_block_from_universal, sycl_nhd_hnd_transpose, sycl_universal_from_block,
    sycl_vectorized_copy,
};
use oneapi_rs::sycl::SyclQueue;

// ---------------------------------------------------------------------------
// TestDtype trait
// ---------------------------------------------------------------------------

trait TestDtype: oneapi_rs::sycl::DeviceRepr + Clone + Debug + Default + Send + 'static {
    const ELEM_SIZE: usize;
    const ATOL: f64;
    const RTOL: f64;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl TestDtype for f32 {
    const ELEM_SIZE: usize = 4;
    const ATOL: f64 = 1e-5;
    const RTOL: f64 = 1e-5;
    fn from_f64(v: f64) -> Self { v as f32 }
    fn to_f64(self) -> f64 { self as f64 }
}

impl TestDtype for f64 {
    const ELEM_SIZE: usize = 8;
    const ATOL: f64 = 1e-12;
    const RTOL: f64 = 1e-12;
    fn from_f64(v: f64) -> Self { v }
    fn to_f64(self) -> f64 { self }
}

impl TestDtype for f16 {
    const ELEM_SIZE: usize = 2;
    const ATOL: f64 = 1e-2;
    const RTOL: f64 = 1e-2;
    fn from_f64(v: f64) -> Self { f16::from_f64(v) }
    fn to_f64(self) -> f64 { f16::to_f64(self) }
}

impl TestDtype for bf16 {
    const ELEM_SIZE: usize = 2;
    const ATOL: f64 = 1e-2;
    const RTOL: f64 = 1e-2;
    fn from_f64(v: f64) -> Self { bf16::from_f64(v) }
    fn to_f64(self) -> f64 { bf16::to_f64(self) }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Approximate equality check for two flat buffers.
fn assert_close<T: TestDtype>(actual: &[T], expected: &[T], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a64 = a.clone().to_f64();
        let e64 = e.clone().to_f64();
        let diff = (a64 - e64).abs();
        let tol = T::ATOL + T::RTOL * e64.abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: {a64} vs {e64} (diff={diff}, tol={tol})"
        );
    }
}

/// Compute the within-chunk flat element offset for a given (nt, nh, hd) index.
fn inner_offset(layout: BlockLayout, nt_idx: usize, nh_idx: usize, hd_idx: usize,
                nt: usize, nh: usize, hd: usize) -> usize {
    match layout {
        BlockLayout::NHD => ((nt_idx * nh) + nh_idx) * hd + hd_idx,
        BlockLayout::HND => ((nh_idx * nt) + nt_idx) * hd + hd_idx,
    }
}

/// Create a SYCL queue for device 0. Returns None if no XPU device available.
fn sycl_setup() -> Option<Arc<SyclQueue>> {
    SyclQueue::new_for_device_ordinal(0).ok()
}

// ---------------------------------------------------------------------------
// Core roundtrip test (SYCL path — mirrors CUDA kernel_roundtrip.rs)
// ---------------------------------------------------------------------------

fn block_universal_roundtrip_inner<T: TestDtype>(layout: BlockLayout) {
    let queue = match sycl_setup() {
        Some(q) => q,
        None => {
            eprintln!("No XPU/SYCL device found, skipping test");
            return;
        }
    };

    // Dimensions matching the CUDA tests
    let nh = 3usize;
    let nl = 2usize;
    let no = 2usize;
    let nt = 4usize;
    let hd = 5usize;
    let nb = 3usize;
    let inner = nt * nh * hd;
    let chunk_count = nl * no;
    let universal_volume = nh * nl * no * nt * hd;
    let elem_size = T::ELEM_SIZE;

    // Generate deterministic reference data on host
    let mut host_block_chunks: Vec<Vec<Vec<T>>> = Vec::with_capacity(nb);

    for block_idx in 0..nb {
        let mut chunks = Vec::with_capacity(chunk_count);
        for chunk_idx in 0..chunk_count {
            let global_idx = block_idx * chunk_count + chunk_idx;
            let chunk: Vec<T> = (0..inner)
                .map(|off| T::from_f64((global_idx * inner + off) as f64 + 0.25))
                .collect();
            chunks.push(chunk);
        }
        host_block_chunks.push(chunks);
    }

    // Compute reference universal tensors from the block data
    let mut ref_universals: Vec<Vec<T>> = Vec::with_capacity(nb);
    for block_idx in 0..nb {
        let mut universal = vec![T::from_f64(0.0); universal_volume];
        for nh_idx in 0..nh {
            for nl_idx in 0..nl {
                for no_idx in 0..no {
                    for nt_idx in 0..nt {
                        for hd_idx in 0..hd {
                            let uni_idx = ((((nh_idx * nl + nl_idx) * no + no_idx) * nt + nt_idx) * hd) + hd_idx;
                            let chunk_idx = nl_idx * no + no_idx;
                            let offset = inner_offset(layout, nt_idx, nh_idx, hd_idx, nt, nh, hd);
                            universal[uni_idx] = host_block_chunks[block_idx][chunk_idx][offset].clone();
                        }
                    }
                }
            }
        }
        ref_universals.push(universal);
    }

    // --- Allocate device memory via SYCL USM (typed SyclSlice<T>) ---
    //
    // The returned `SyclSlice` holds an `Arc<SyclContext>`
    // so it outlives the specific queue.

    // Upload block chunks to device
    use oneapi_rs::sycl::SyclSlice;
    let mut block_device_bufs: Vec<SyclSlice<T>> = Vec::with_capacity(nb * chunk_count);

    for block_idx in 0..nb {
        for chunk_idx in 0..chunk_count {
            let mut buf = unsafe { queue.context().alloc_device::<T>(queue.device(), inner) }
                .expect("device alloc failed");

            // H2D: copy host chunk data to device
            queue.memcpy_sync(
                host_block_chunks[block_idx][chunk_idx].as_slice(),
                &mut buf,
            ).expect("H2D memcpy failed");

            block_device_bufs.push(buf);
        }
    }

    // Allocate universal output buffers on device
    let mut universal_device_bufs: Vec<SyclSlice<T>> = Vec::with_capacity(nb);
    for _ in 0..nb {
        let buf = unsafe { queue.context().alloc_device::<T>(queue.device(), universal_volume) }
            .expect("device alloc failed");
        universal_device_bufs.push(buf);
    }

    // Build pointer tables as raw u64 arrays, then upload to device as u64 buffers.
    // block_ptrs: [nb * chunk_count] pointers to device chunk data
    let block_ptr_values: Vec<u64> = block_device_bufs
        .iter()
        .map(|buf| buf.as_mut_ptr() as u64)
        .collect();
    let universal_ptr_values: Vec<u64> = universal_device_bufs
        .iter()
        .map(|buf| buf.as_mut_ptr() as u64)
        .collect();

    let mut block_ptrs_dev = unsafe { queue.context().alloc_device::<u64>(queue.device(), block_ptr_values.len()) }
        .expect("device alloc failed");
    queue.memcpy_sync(block_ptr_values.as_slice(), &mut block_ptrs_dev)
        .expect("block ptrs H2D failed");

    let mut universal_ptrs_dev = unsafe { queue.context().alloc_device::<u64>(queue.device(), universal_ptr_values.len()) }
        .expect("device alloc failed");
    queue.memcpy_sync(universal_ptr_values.as_slice(), &mut universal_ptrs_dev)
        .expect("universal ptrs H2D failed");

    queue.synchronize().expect("sync failed");

    // --- Forward: blocks -> universal (SYCL kernel dispatch) ---
    let queue_ptr = queue.raw_queue_ptr();
    let status = unsafe {
        sycl_universal_from_block(
            universal_ptrs_dev.as_mut_ptr() as *const *mut c_void,
            block_ptrs_dev.as_mut_ptr() as *const *const c_void,
            nb, nh, nl, no, nt, hd, nl, 0, elem_size,
            layout,
            queue_ptr,
        )
    };
    assert_eq!(status, 0, "sycl_universal_from_block returned {}", status);
    queue.synchronize().expect("sync after forward kernel failed");

    // --- Read back universal buffers and verify against reference ---
    for block_idx in 0..nb {
        let readback = queue
            .clone_dtoh(&universal_device_bufs[block_idx])
            .expect("D2H clone failed");
        assert_close(
            &readback,
            &ref_universals[block_idx],
            &format!("block[{block_idx}] universal_from_block"),
        );
    }
    eprintln!(
        "SYCL forward (block→universal) verified for {:?} layout, elem_size={}",
        layout, elem_size,
    );

    // --- Reverse: universal -> blocks (SYCL kernel dispatch) ---
    // Zero out block device memory first
    for buf in &mut block_device_bufs {
        let zeros = vec![T::from_f64(0.0); inner];
        queue.memcpy_sync(zeros.as_slice(), buf).expect("zero fill failed");
    }
    queue.synchronize().expect("sync after zero fill failed");

    let status = unsafe {
        sycl_block_from_universal(
            universal_ptrs_dev.as_mut_ptr() as *const *const c_void,
            block_ptrs_dev.as_mut_ptr() as *const *mut c_void,
            nb, nh, nl, no, nt, hd, nl, 0, elem_size,
            layout,
            queue_ptr,
        )
    };
    assert_eq!(status, 0, "sycl_block_from_universal returned {}", status);
    queue.synchronize().expect("sync after reverse kernel failed");

    // --- Read back block chunks and verify full roundtrip ---
    for block_idx in 0..nb {
        for chunk_idx in 0..chunk_count {
            let buf_idx = block_idx * chunk_count + chunk_idx;
            let readback = queue
                .clone_dtoh(&block_device_bufs[buf_idx])
                .expect("D2H clone failed");
            assert_close(
                &readback,
                &host_block_chunks[block_idx][chunk_idx],
                &format!("block[{block_idx}][{chunk_idx}] full roundtrip"),
            );
        }
    }
    eprintln!(
        "SYCL full roundtrip (block→universal→block) verified for {:?} layout, elem_size={}",
        layout, elem_size,
    );
}

// ---------------------------------------------------------------------------
// Test matrix
// ---------------------------------------------------------------------------

macro_rules! sycl_block_universal_test {
    ($name:ident, $ty:ty, $layout:expr) => {
        #[test]
        fn $name() {
            block_universal_roundtrip_inner::<$ty>($layout)
        }
    };
}

sycl_block_universal_test!(sycl_roundtrip_nhd_f32, f32, BlockLayout::NHD);
sycl_block_universal_test!(sycl_roundtrip_nhd_f64, f64, BlockLayout::NHD);
sycl_block_universal_test!(sycl_roundtrip_hnd_f32, f32, BlockLayout::HND);
sycl_block_universal_test!(sycl_roundtrip_hnd_f64, f64, BlockLayout::HND);
sycl_block_universal_test!(sycl_roundtrip_nhd_f16, f16, BlockLayout::NHD);
sycl_block_universal_test!(sycl_roundtrip_nhd_bf16, bf16, BlockLayout::NHD);
sycl_block_universal_test!(sycl_roundtrip_hnd_f16, f16, BlockLayout::HND);
sycl_block_universal_test!(sycl_roundtrip_hnd_bf16, bf16, BlockLayout::HND);

// ---------------------------------------------------------------------------
// NHD ↔ HND transpose (SYCL path — mirrors CUDA kernel_roundtrip.rs)
// ---------------------------------------------------------------------------
//
// Each test starts from independent ground truth (a deterministic universal
// tensor projected into the src layout) and verifies the kernel output against
// the *opposite* layout's ground truth — not against the kernel's own inverse
// pass. A pure round-trip would silently pass a symmetric bug shared by both
// directions, since the kernel is a single FFI symbol with a runtime layout
// switch.

/// Project a `[nh, nl, no, nt, hd]` universal tensor into `nl * no` flat block
/// chunks, each laid out per `layout` (NHD = `[nt, nh, hd]`, HND = `[nh, nt, hd]`).
fn project_blocks<T: TestDtype>(
    universal: &[T], layout: BlockLayout,
    nh: usize, nl: usize, no: usize, nt: usize, hd: usize,
) -> Vec<Vec<T>> {
    let mut blocks = Vec::with_capacity(nl * no);
    for nl_idx in 0..nl {
        for no_idx in 0..no {
            let mut chunk = vec![T::from_f64(0.0); nt * nh * hd];
            for nt_idx in 0..nt {
                for nh_idx in 0..nh {
                    for hd_idx in 0..hd {
                        let uni_idx = ((((nh_idx * nl + nl_idx) * no + no_idx) * nt + nt_idx) * hd) + hd_idx;
                        let off = inner_offset(layout, nt_idx, nh_idx, hd_idx, nt, nh, hd);
                        chunk[off] = universal[uni_idx].clone();
                    }
                }
            }
            blocks.push(chunk);
        }
    }
    blocks
}

fn nhd_hnd_transpose_inner<T: TestDtype>(src_layout: BlockLayout) {
    let queue = match sycl_setup() {
        Some(q) => q,
        None => {
            eprintln!("No XPU/SYCL device found, skipping test");
            return;
        }
    };

    let nh = 3usize;
    let nl = 2usize;
    let no = 2usize;
    let nt = 4usize;
    let hd = 5usize;
    let nb = 3usize;
    let chunk_volume = nh * nt * hd;
    let universal_volume = nh * nl * no * nt * hd;
    let chunk_count = nl * no;
    let elem_size = T::ELEM_SIZE;

    let dst_layout = match src_layout {
        BlockLayout::NHD => BlockLayout::HND,
        BlockLayout::HND => BlockLayout::NHD,
    };

    // Generate deterministic universal tensors and project into both layouts.
    let universals: Vec<Vec<T>> = (0..nb)
        .map(|block_idx| {
            (0..universal_volume)
                .map(|i| T::from_f64((block_idx * universal_volume + i) as f64 * 0.5 - 1.0))
                .collect()
        })
        .collect();

    let src_blocks: Vec<Vec<Vec<T>>> = universals
        .iter()
        .map(|u| project_blocks::<T>(u, src_layout, nh, nl, no, nt, hd))
        .collect();
    let dst_blocks_expected: Vec<Vec<Vec<T>>> = universals
        .iter()
        .map(|u| project_blocks::<T>(u, dst_layout, nh, nl, no, nt, hd))
        .collect();

    use oneapi_rs::sycl::SyclSlice;

    // Upload src chunks to device.
    let mut src_device_bufs: Vec<SyclSlice<T>> = Vec::with_capacity(nb * chunk_count);
    for block in &src_blocks {
        for chunk in block {
            let mut buf = unsafe { queue.context().alloc_device::<T>(queue.device(), chunk_volume) }
                .expect("device alloc failed");
            queue.memcpy_sync(chunk.as_slice(), &mut buf).expect("H2D src chunk");
            src_device_bufs.push(buf);
        }
    }

    // Allocate dst chunks on device, zero-filled so a misbehaving kernel that
    // touches only some elements is detected by assert_close on the rest.
    let mut dst_device_bufs: Vec<SyclSlice<T>> = Vec::with_capacity(nb * chunk_count);
    let zeros = vec![T::from_f64(0.0); chunk_volume];
    for _ in 0..(nb * chunk_count) {
        let mut buf = unsafe { queue.context().alloc_device::<T>(queue.device(), chunk_volume) }
            .expect("device alloc failed");
        queue.memcpy_sync(zeros.as_slice(), &mut buf).expect("dst zero-fill");
        dst_device_bufs.push(buf);
    }

    let src_ptr_values: Vec<u64> = src_device_bufs.iter().map(|b| b.as_mut_ptr() as u64).collect();
    let dst_ptr_values: Vec<u64> = dst_device_bufs.iter().map(|b| b.as_mut_ptr() as u64).collect();

    let mut src_ptrs_dev = unsafe { queue.context().alloc_device::<u64>(queue.device(), src_ptr_values.len()) }
        .expect("device alloc failed");
    queue.memcpy_sync(src_ptr_values.as_slice(), &mut src_ptrs_dev)
        .expect("src ptrs H2D");

    let mut dst_ptrs_dev = unsafe { queue.context().alloc_device::<u64>(queue.device(), dst_ptr_values.len()) }
        .expect("device alloc failed");
    queue.memcpy_sync(dst_ptr_values.as_slice(), &mut dst_ptrs_dev)
        .expect("dst ptrs H2D");

    queue.synchronize().expect("sync before kernel");

    let queue_ptr = queue.raw_queue_ptr();
    let status = unsafe {
        sycl_nhd_hnd_transpose(
            src_ptrs_dev.as_mut_ptr() as *const *const c_void,
            dst_ptrs_dev.as_mut_ptr() as *const *mut c_void,
            nb, nl, no, nt, nh, hd, elem_size,
            src_layout,
            queue_ptr,
        )
    };
    assert_eq!(status, 0, "sycl_nhd_hnd_transpose returned {}", status);
    queue.synchronize().expect("sync after kernel");

    // Verify each dst chunk matches the opposite-layout ground truth.
    for block_idx in 0..nb {
        for chunk_idx in 0..chunk_count {
            let buf_idx = block_idx * chunk_count + chunk_idx;
            let readback = queue.clone_dtoh(&dst_device_bufs[buf_idx]).expect("D2H");
            assert_close(
                &readback,
                &dst_blocks_expected[block_idx][chunk_idx],
                &format!(
                    "transpose {:?}->{:?} block {} chunk {}",
                    src_layout, dst_layout, block_idx, chunk_idx
                ),
            );
        }
    }
    eprintln!(
        "SYCL nhd_hnd_transpose verified for {:?} -> {:?}, elem_size={}",
        src_layout, dst_layout, elem_size,
    );
}

macro_rules! sycl_nhd_hnd_test {
    ($name:ident, $ty:ty, $src_layout:expr) => {
        #[test]
        fn $name() {
            nhd_hnd_transpose_inner::<$ty>($src_layout)
        }
    };
}

sycl_nhd_hnd_test!(sycl_nhd_hnd_transpose_nhd_to_hnd_f16, f16, BlockLayout::NHD);
sycl_nhd_hnd_test!(sycl_nhd_hnd_transpose_nhd_to_hnd_bf16, bf16, BlockLayout::NHD);
sycl_nhd_hnd_test!(sycl_nhd_hnd_transpose_nhd_to_hnd_f32, f32, BlockLayout::NHD);
sycl_nhd_hnd_test!(sycl_nhd_hnd_transpose_nhd_to_hnd_f64, f64, BlockLayout::NHD);
sycl_nhd_hnd_test!(sycl_nhd_hnd_transpose_hnd_to_nhd_f16, f16, BlockLayout::HND);
sycl_nhd_hnd_test!(sycl_nhd_hnd_transpose_hnd_to_nhd_bf16, bf16, BlockLayout::HND);
sycl_nhd_hnd_test!(sycl_nhd_hnd_transpose_hnd_to_nhd_f32, f32, BlockLayout::HND);
sycl_nhd_hnd_test!(sycl_nhd_hnd_transpose_hnd_to_nhd_f64, f64, BlockLayout::HND);

/// Empty batch should be a noop — the C++ launcher returns 0 for num_blocks==0.
#[test]
fn sycl_empty_batch_noop() {
    let queue = match sycl_setup() {
        Some(q) => q,
        None => {
            eprintln!("No XPU/SYCL device found, skipping test");
            return;
        }
    };
    let queue_ptr = queue.raw_queue_ptr();
    let status = unsafe {
        sycl_universal_from_block(
            std::ptr::null(),
            std::ptr::null(),
            0, 1, 1, 1, 1, 1, 1, 0, 4,
            BlockLayout::NHD,
            queue_ptr,
        )
    };
    assert_eq!(status, 0, "empty batch should return 0, got {}", status);
    eprintln!("SYCL empty batch noop: confirmed launcher returns 0 for num_blocks=0");
}

// ===========================================================================
// vectorized_copy SYCL test
// ===========================================================================

#[test]
fn sycl_vectorized_copy_roundtrip() {
    let queue = match sycl_setup() {
        Some(q) => q,
        None => {
            eprintln!("No XPU/SYCL device found, skipping test");
            return;
        }
    };

    let num_pairs: i32 = 4;
    let copy_size: usize = 1024;

    // Generate deterministic test data
    let mut src_host_data: Vec<Vec<u8>> = Vec::new();
    let mut src_bufs: Vec<oneapi_rs::sycl::SyclSlice<u8>> = Vec::new();
    let mut dst_bufs: Vec<oneapi_rs::sycl::SyclSlice<u8>> = Vec::new();

    for i in 0..num_pairs as usize {
        let pattern: Vec<u8> = (0..copy_size).map(|b| ((i * 37 + b * 7) & 0xFF) as u8).collect();
        let mut src = unsafe { queue.context().alloc_device::<u8>(queue.device(), copy_size)}.expect("alloc src");
        queue.memcpy_sync(pattern.as_slice(), &mut src).expect("H2D src");
        let dst = unsafe { queue.context().alloc_device::<u8>(queue.device(), copy_size)}.expect("alloc dst");

        src_host_data.push(pattern);
        src_bufs.push(src);
        dst_bufs.push(dst);
    }

    // Build pointer arrays (as u64 device addresses)
    let src_ptrs: Vec<u64> = src_bufs.iter().map(|b| b.as_mut_ptr() as u64).collect();
    let dst_ptrs: Vec<u64> = dst_bufs.iter().map(|b| b.as_mut_ptr() as u64).collect();

    let mut src_ptrs_dev = unsafe { queue.context().alloc_device::<u64>(queue.device(), num_pairs as usize) }
        .expect("alloc src_ptrs");
    queue.memcpy_sync(src_ptrs.as_slice(), &mut src_ptrs_dev).expect("H2D src_ptrs");

    let mut dst_ptrs_dev = unsafe { queue.context().alloc_device::<u64>(queue.device(), num_pairs as usize) }
        .expect("alloc dst_ptrs");
    queue.memcpy_sync(dst_ptrs.as_slice(), &mut dst_ptrs_dev).expect("H2D dst_ptrs");

    queue.synchronize().expect("sync before kernel");

    // Dispatch sycl_vectorized_copy
    let queue_ptr = queue.raw_queue_ptr();
    let status = unsafe {
        sycl_vectorized_copy(
            src_ptrs_dev.as_mut_ptr() as *mut *mut std::ffi::c_void,
            dst_ptrs_dev.as_mut_ptr() as *mut *mut std::ffi::c_void,
            copy_size,
            num_pairs,
            queue_ptr,
        )
    };
    assert_eq!(status, 0, "sycl_vectorized_copy returned {}", status);
    queue.synchronize().expect("sync after kernel");

    // Verify each dst matches src
    for i in 0..num_pairs as usize {
        let readback = queue.clone_dtoh(&dst_bufs[i]).expect("D2H");
        assert_eq!(readback, src_host_data[i], "vectorized_copy pair {i} mismatch");
    }
    eprintln!("SYCL vectorized_copy: {} pairs × {} bytes — PASS", num_pairs, copy_size);
}
