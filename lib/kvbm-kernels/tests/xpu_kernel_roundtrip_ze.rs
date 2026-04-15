// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Level-Zero XPU SPIR-V kernel roundtrips.
//!
//! Uses Level-Zero directly to test the OpenCL SPIR-V kernels:
//!   - vectorized_copy.spv  — bulk device-to-device copy
//!   - tensor_permute.spv   — block ↔ universal layout permute
//!
//! API stack:
//! - Memory:  ze::Context::alloc_device / alloc_host
//! - Transfer: ImmediateCommandList::append_memcpy
//! - Kernel:   Module::from_spirv → Kernel → append_launch_kernel
//! - Sync:     host_synchronize
//!
//! For SYCL C++ kernel tests, see xpu_kernel_roundtrip_sycl.rs.
//!
//! Run with:
//!   cargo test -p kvbm-kernels \
//!       --features xpu_permute_kernels,testing-xpu \
//!       --test xpu_kernel_roundtrip_ze -- --nocapture

#![cfg(all(feature = "testing-xpu", feature = "xpu_permute_kernels"))]

use std::ffi::c_void;
use std::fmt::Debug;

use kvbm_kernels::XpuBlockLayout;
use half::{f16, bf16};
use level_zero as ze;

/// Embedded SPIR-V binaries (same ones used by the ze backend at runtime).
static VECTORIZED_COPY_SPIRV: &[u8] =
    include_bytes!("../opencl/vectorized_copy.spv");
static TENSOR_PERMUTE_SPIRV: &[u8] =
    include_bytes!("../opencl/tensor_permute.spv");

const COPY_WG_SIZE: u32 = 128;
const COPY_MAX_WGS: u32 = 65535;

// ---------------------------------------------------------------------------
// TestDtype trait
// ---------------------------------------------------------------------------

trait TestDtype: Clone + Debug + Send + 'static {
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

fn inner_offset(layout: XpuBlockLayout, nt_idx: usize, nh_idx: usize, hd_idx: usize,
                nt: usize, nh: usize, hd: usize) -> usize {
    match layout {
        XpuBlockLayout::NHD => ((nt_idx * nh) + nh_idx) * hd + hd_idx,
        XpuBlockLayout::HND => ((nh_idx * nt) + nt_idx) * hd + hd_idx,
    }
}

/// Helper: upload a host byte slice to a new device buffer.
fn upload(cmd: &ze::ImmediateCommandList, ctx: &ze::Context, dev: &ze::Device,
          data: &[u8]) -> ze::DeviceBuffer {
    let buf = ctx.alloc_device(dev, data.len(), 64).expect("alloc_device");
    cmd.append_memcpy(buf.as_mut_ptr(), data.as_ptr() as *const c_void, data.len())
        .expect("H2D");
    buf
}

/// Helper: download device buffer contents to a host Vec<u8>.
fn download(cmd: &ze::ImmediateCommandList, buf: &ze::DeviceBuffer, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    cmd.append_memcpy(out.as_mut_ptr() as *mut c_void, buf.as_mut_ptr() as *const c_void, len)
        .expect("D2H");
    cmd.host_synchronize(u64::MAX).expect("sync");
    out
}

/// Initialize Level Zero, pick device 0.
fn ze_setup() -> Option<(ze::Context, ze::Device)> {
    ze::init().ok()?;
    let drivers = ze::drivers().ok()?;
    let driver = drivers.into_iter().next()?;
    let devices = driver.devices().ok()?;
    let device = devices.into_iter().next()?;
    let context = ze::Context::create(&driver).ok()?;
    Some((context, device))
}

// ===========================================================================
// 1. vectorized_copy SPIR-V test
// ===========================================================================

#[test]
fn ze_vectorized_copy_spv() {
    let (context, device) = match ze_setup() {
        Some(s) => s,
        None => { eprintln!("No XPU device, skipping"); return; }
    };

    let module = match context.create_module_from_spirv(&device, VECTORIZED_COPY_SPIRV, None) {
        Ok(m) => m,
        Err(e) => { eprintln!("SPIR-V load failed: {:?}, skipping", e); return; }
    };
    let kernel = module.create_kernel("vectorized_copy").expect("create_kernel");
    kernel.set_indirect_access(
        ze::KERNEL_INDIRECT_ACCESS_FLAG_HOST | ze::KERNEL_INDIRECT_ACCESS_FLAG_DEVICE,
    ).expect("set_indirect_access");

    let cmd_list = context.create_immediate_command_list(&device).expect("cmd list");

    // Setup: 4 pairs of 1024-byte buffers with known patterns.
    let num_pairs: i32 = 4;
    let copy_size: usize = 1024;

    let mut src_host_data: Vec<Vec<u8>> = Vec::new();
    let mut src_bufs: Vec<ze::DeviceBuffer> = Vec::new();
    let mut dst_bufs: Vec<ze::DeviceBuffer> = Vec::new();
    let mut src_ptrs: Vec<u64> = Vec::new();
    let mut dst_ptrs: Vec<u64> = Vec::new();

    for i in 0..num_pairs as usize {
        let pattern: Vec<u8> = (0..copy_size).map(|b| ((i * 37 + b * 7) & 0xFF) as u8).collect();
        let src = upload(&cmd_list, &context, &device, &pattern);
        let dst = context.alloc_device(&device, copy_size, 64).expect("alloc dst");

        src_ptrs.push(src.as_mut_ptr() as u64);
        dst_ptrs.push(dst.as_mut_ptr() as u64);
        src_host_data.push(pattern);
        src_bufs.push(src);
        dst_bufs.push(dst);
    }

    // Upload pointer arrays to device
    let src_ptrs_bytes = unsafe {
        std::slice::from_raw_parts(src_ptrs.as_ptr() as *const u8, src_ptrs.len() * 8)
    };
    let dst_ptrs_bytes = unsafe {
        std::slice::from_raw_parts(dst_ptrs.as_ptr() as *const u8, dst_ptrs.len() * 8)
    };
    let src_ptrs_dev = upload(&cmd_list, &context, &device, src_ptrs_bytes);
    let dst_ptrs_dev = upload(&cmd_list, &context, &device, dst_ptrs_bytes);
    cmd_list.host_synchronize(u64::MAX).expect("sync uploads");

    // Dispatch vectorized_copy kernel
    kernel.set_group_size(COPY_WG_SIZE, 1, 1).expect("set_group_size");
    // args: src_addrs, dst_addrs, copy_size_in_bytes, num_pairs
    let copy_size_u64 = copy_size as u64;
    kernel.set_arg(0, &(src_ptrs_dev.as_mut_ptr() as u64)).expect("arg0");
    kernel.set_arg(1, &(dst_ptrs_dev.as_mut_ptr() as u64)).expect("arg1");
    kernel.set_arg(2, &copy_size_u64).expect("arg2");
    kernel.set_arg(3, &num_pairs).expect("arg3");

    let grid_dim = std::cmp::min(num_pairs as u32, COPY_MAX_WGS);
    cmd_list.append_launch_kernel(&kernel, ze::GroupCount { x: grid_dim, y: 1, z: 1 })
        .expect("launch");
    cmd_list.host_synchronize(u64::MAX).expect("sync kernel");

    // Verify each dst matches src
    for i in 0..num_pairs as usize {
        let readback = download(&cmd_list, &dst_bufs[i], copy_size);
        assert_eq!(readback, src_host_data[i], "vectorized_copy pair {i} mismatch");
    }
    eprintln!("ze_vectorized_copy_spv: {} pairs × {} bytes — PASS", num_pairs, copy_size);
}

// ===========================================================================
// 2. tensor_permute SPIR-V test (universal_from_block + block_from_universal)
// ===========================================================================

fn ze_permute_roundtrip_inner<T: TestDtype>(layout: XpuBlockLayout) {
    let (context, device) = match ze_setup() {
        Some(s) => s,
        None => { eprintln!("No XPU device, skipping"); return; }
    };

    let module = match context.create_module_from_spirv(&device, TENSOR_PERMUTE_SPIRV, None) {
        Ok(m) => m,
        Err(e) => { eprintln!("tensor_permute SPIR-V load failed: {:?}, skipping", e); return; }
    };
    let k_fwd = module.create_kernel("universal_from_block").expect("fwd kernel");
    let k_rev = module.create_kernel("block_from_universal").expect("rev kernel");
    for k in [&k_fwd, &k_rev] {
        k.set_indirect_access(
            ze::KERNEL_INDIRECT_ACCESS_FLAG_HOST | ze::KERNEL_INDIRECT_ACCESS_FLAG_DEVICE,
        ).expect("set_indirect_access");
    }

    let cmd_list = context.create_immediate_command_list(&device).expect("cmd list");

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

    // Compute reference universal tensors
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

    // Upload block chunks to device
    let mut block_device_bufs: Vec<ze::DeviceBuffer> = Vec::with_capacity(nb * chunk_count);
    let mut block_ptr_values: Vec<u64> = Vec::with_capacity(nb * chunk_count);

    for block_idx in 0..nb {
        for chunk_idx in 0..chunk_count {
            let byte_size = inner * elem_size;
            let host_bytes = unsafe {
                std::slice::from_raw_parts(
                    host_block_chunks[block_idx][chunk_idx].as_ptr() as *const u8,
                    byte_size,
                )
            };
            let buf = upload(&cmd_list, &context, &device, host_bytes);
            block_ptr_values.push(buf.as_mut_ptr() as u64);
            block_device_bufs.push(buf);
        }
    }

    // Allocate universal output buffers
    let mut universal_device_bufs: Vec<ze::DeviceBuffer> = Vec::with_capacity(nb);
    let mut universal_ptr_values: Vec<u64> = Vec::with_capacity(nb);
    for _ in 0..nb {
        let buf = context.alloc_device(&device, universal_volume * elem_size, 64)
            .expect("alloc universal");
        universal_ptr_values.push(buf.as_mut_ptr() as u64);
        universal_device_bufs.push(buf);
    }

    // Upload pointer tables
    let block_ptrs_bytes = unsafe {
        std::slice::from_raw_parts(block_ptr_values.as_ptr() as *const u8, block_ptr_values.len() * 8)
    };
    let universal_ptrs_bytes = unsafe {
        std::slice::from_raw_parts(universal_ptr_values.as_ptr() as *const u8, universal_ptr_values.len() * 8)
    };
    let block_ptrs_dev = upload(&cmd_list, &context, &device, block_ptrs_bytes);
    let universal_ptrs_dev = upload(&cmd_list, &context, &device, universal_ptrs_bytes);
    cmd_list.host_synchronize(u64::MAX).expect("sync uploads");

    // --- Forward: universal_from_block SPIR-V dispatch ---
    let total_elements = (nb * universal_volume) as u64;
    let wg_size: u32 = 256;
    let num_wgs = std::cmp::min(
        ((total_elements + wg_size as u64 - 1) / wg_size as u64) as u32,
        COPY_MAX_WGS,
    );

    k_fwd.set_group_size(wg_size, 1, 1).expect("group_size");
    // Kernel args match the .cl signature: universal_ptrs, block_ptrs, num_blocks, nh, nl, no, nt, hd, elem_size, layout
    k_fwd.set_arg(0, &(universal_ptrs_dev.as_mut_ptr() as u64)).expect("arg0");
    k_fwd.set_arg(1, &(block_ptrs_dev.as_mut_ptr() as u64)).expect("arg1");
    k_fwd.set_arg(2, &(nb as u64)).expect("arg2");
    k_fwd.set_arg(3, &(nh as u64)).expect("arg3");
    k_fwd.set_arg(4, &(nl as u64)).expect("arg4");
    k_fwd.set_arg(5, &(no as u64)).expect("arg5");
    k_fwd.set_arg(6, &(nt as u64)).expect("arg6");
    k_fwd.set_arg(7, &(hd as u64)).expect("arg7");
    k_fwd.set_arg(8, &(elem_size as u64)).expect("arg8");
    k_fwd.set_arg(9, &(layout as i32)).expect("arg9");

    cmd_list.append_launch_kernel(&k_fwd, ze::GroupCount { x: num_wgs, y: 1, z: 1 })
        .expect("launch fwd");
    cmd_list.host_synchronize(u64::MAX).expect("sync fwd");

    // Verify forward results
    for block_idx in 0..nb {
        let byte_size = universal_volume * elem_size;
        let bytes = download(&cmd_list, &universal_device_bufs[block_idx], byte_size);
        let readback: &[T] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const T, universal_volume)
        };
        assert_close(
            readback,
            &ref_universals[block_idx],
            &format!("block[{block_idx}] universal_from_block (SPIR-V)"),
        );
    }
    eprintln!("ZE forward (block→universal) SPIR-V verified for {:?} elem_size={}", layout, elem_size);

    // --- Reverse: block_from_universal SPIR-V dispatch ---
    // Zero out block buffers first
    for buf in &block_device_bufs {
        let byte_size = inner * elem_size;
        let zeros = vec![0u8; byte_size];
        cmd_list.append_memcpy(buf.as_mut_ptr(), zeros.as_ptr() as *const c_void, byte_size)
            .expect("zero fill");
    }
    cmd_list.host_synchronize(u64::MAX).expect("sync zero");

    k_rev.set_group_size(wg_size, 1, 1).expect("group_size");
    k_rev.set_arg(0, &(universal_ptrs_dev.as_mut_ptr() as u64)).expect("arg0");
    k_rev.set_arg(1, &(block_ptrs_dev.as_mut_ptr() as u64)).expect("arg1");
    k_rev.set_arg(2, &(nb as u64)).expect("arg2");
    k_rev.set_arg(3, &(nh as u64)).expect("arg3");
    k_rev.set_arg(4, &(nl as u64)).expect("arg4");
    k_rev.set_arg(5, &(no as u64)).expect("arg5");
    k_rev.set_arg(6, &(nt as u64)).expect("arg6");
    k_rev.set_arg(7, &(hd as u64)).expect("arg7");
    k_rev.set_arg(8, &(elem_size as u64)).expect("arg8");
    k_rev.set_arg(9, &(layout as i32)).expect("arg9");

    cmd_list.append_launch_kernel(&k_rev, ze::GroupCount { x: num_wgs, y: 1, z: 1 })
        .expect("launch rev");
    cmd_list.host_synchronize(u64::MAX).expect("sync rev");

    // Verify full roundtrip
    let mut buf_idx = 0;
    for block_idx in 0..nb {
        for chunk_idx in 0..chunk_count {
            let byte_size = inner * elem_size;
            let bytes = download(&cmd_list, &block_device_bufs[buf_idx], byte_size);
            let readback: &[T] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const T, inner)
            };
            assert_close(
                readback,
                &host_block_chunks[block_idx][chunk_idx],
                &format!("block[{block_idx}][{chunk_idx}] roundtrip (SPIR-V)"),
            );
            buf_idx += 1;
        }
    }
    eprintln!(
        "ZE full roundtrip (block→universal→block) SPIR-V verified for {:?} elem_size={}",
        layout, elem_size,
    );
}

// ---------------------------------------------------------------------------
// Test matrix
// ---------------------------------------------------------------------------

macro_rules! ze_permute_test {
    ($name:ident, $ty:ty, $layout:expr) => {
        #[test]
        fn $name() {
            ze_permute_roundtrip_inner::<$ty>($layout)
        }
    };
}

ze_permute_test!(ze_permute_nhd_f32, f32, XpuBlockLayout::NHD);
ze_permute_test!(ze_permute_nhd_f64, f64, XpuBlockLayout::NHD);
ze_permute_test!(ze_permute_hnd_f32, f32, XpuBlockLayout::HND);
ze_permute_test!(ze_permute_hnd_f64, f64, XpuBlockLayout::HND);
ze_permute_test!(ze_permute_nhd_f16, f16, XpuBlockLayout::NHD);
ze_permute_test!(ze_permute_nhd_bf16, bf16, XpuBlockLayout::NHD);
ze_permute_test!(ze_permute_hnd_f16, f16, XpuBlockLayout::HND);
ze_permute_test!(ze_permute_hnd_bf16, bf16, XpuBlockLayout::HND);
