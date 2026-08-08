// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for KVTC quantization kernels.
//!
//! Tests FP8 and IntX roundtrip accuracy, min/max reduction correctness,
//! and the range-iterating orchestrators with self-describing compressed format.

#![cfg(all(feature = "testing-cuda", feature = "kvtc_kernels", not(stub_kernels)))]

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DriverError};
use cudarc::runtime::sys as cuda_runtime;
use kvbm_kernels::kvtc_kernels::*;

// Direct FFI for pinned memory — avoids cudarc runtime symbol resolution issues on CUDA 13.x
unsafe extern "C" {
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> u32;
    fn cudaFreeHost(ptr: *mut c_void) -> u32;
}

struct PinnedBuffer {
    ptr: *mut c_void,
    len: usize,
}

impl PinnedBuffer {
    fn new_zeroed(len: usize) -> Self {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { cudaMallocHost(&mut ptr, len) };
        assert_eq!(err, 0, "cudaMallocHost failed with error {err}");
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, len) };
        Self { ptr, len }
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr as *mut u8
    }

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

fn cuda_setup() -> Option<(Arc<CudaStream>, cuda_runtime::cudaStream_t)> {
    let count = CudaContext::device_count().ok()?;
    if count == 0 {
        return None;
    }
    let ctx = CudaContext::new(0).ok()?;
    let stream = ctx.new_stream().ok()?;
    let raw = stream.cu_stream() as cuda_runtime::cudaStream_t;
    Some((stream, raw))
}

// ---------------------------------------------------------------------------
// FP8 roundtrip
// ---------------------------------------------------------------------------

#[test]
fn fp8_roundtrip() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let batch = 4usize;
    let total_features = 32usize;
    let start_feature = 0usize;
    let range_features = total_features;

    // Generate test data in a range FP8 can represent well: [-100, 100]
    let mut host_data = vec![0.0f32; batch * total_features];
    for (i, v) in host_data.iter_mut().enumerate() {
        *v = ((i as f32) * 7.3 - 50.0).sin() * 100.0;
    }

    let input = stream.clone_htod(&host_data)?;
    let packed_size = batch * range_features; // 1 byte per element
    let mut output_slice = unsafe { stream.alloc::<u8>(packed_size)? };
    let mut decompressed = stream.alloc_zeros::<f32>(batch * total_features)?;

    // Quantize
    {
        let (in_ptr, _g1) = input.device_ptr(&stream);
        let (out_ptr, _g2) = output_slice.device_ptr_mut(&stream);
        let err = unsafe {
            quantize_fp8(
                in_ptr as *const f32,
                out_ptr as *mut u8,
                batch,
                total_features,
                start_feature,
                range_features,
                raw,
            )
        };
        assert_eq!(err, cuda_runtime::cudaError::cudaSuccess);
    }

    // Dequantize
    {
        let (packed_ptr, _g1) = output_slice.device_ptr(&stream);
        let (dec_ptr, _g2) = decompressed.device_ptr_mut(&stream);
        let err = unsafe {
            dequantize_fp8(
                packed_ptr as *const u8,
                dec_ptr as *mut f32,
                batch,
                total_features,
                start_feature,
                range_features,
                raw,
            )
        };
        assert_eq!(err, cuda_runtime::cudaError::cudaSuccess);
    }

    stream.synchronize()?;

    let result = stream.clone_dtoh(&decompressed)?;
    for (i, (&original, &reconstructed)) in host_data.iter().zip(result.iter()).enumerate() {
        if original.is_nan() {
            assert!(reconstructed.is_nan(), "Expected NaN at index {i}");
            continue;
        }
        if original == 0.0 {
            assert_eq!(reconstructed, 0.0, "Expected zero at index {i}");
            continue;
        }
        // FP8 E4M3FN has ~12.5% relative error for most values
        let rel_err = ((original - reconstructed) / original).abs();
        assert!(
            rel_err < 0.15,
            "FP8 roundtrip error too large at index {i}: original={original}, reconstructed={reconstructed}, rel_err={rel_err}"
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// IntX roundtrip tests
// ---------------------------------------------------------------------------

fn intx_roundtrip_test(int_bits: i32) -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let batch = 4usize;
    let total_features = 32usize;
    let start_feature = 0usize;
    let range_features = total_features;

    // Data in [0, 100] range
    let mut host_data = vec![0.0f32; batch * total_features];
    for (i, v) in host_data.iter_mut().enumerate() {
        *v = (i as f32) * 100.0 / (batch * total_features) as f32;
    }

    let input = stream.clone_htod(&host_data)?;

    // Min/max workspace
    let mut min_vals = stream.alloc_zeros::<f32>(batch)?;
    let mut max_vals = stream.alloc_zeros::<f32>(batch)?;

    // Compute min/max
    {
        let (in_ptr, _g1) = input.device_ptr(&stream);
        let (min_ptr, _g2) = min_vals.device_ptr_mut(&stream);
        let (max_ptr, _g3) = max_vals.device_ptr_mut(&stream);
        let err = unsafe {
            minmax_reduce(
                in_ptr as *const f32,
                min_ptr as *mut f32,
                max_ptr as *mut f32,
                batch,
                total_features,
                start_feature,
                range_features,
                raw,
            )
        };
        assert_eq!(err, cuda_runtime::cudaError::cudaSuccess);
    }

    // Packed output
    let slots_per_byte = 8 / int_bits as usize;
    let bytes_per_row = (range_features + slots_per_byte - 1) / slots_per_byte;
    let packed_size = batch * bytes_per_row;
    let mut packed = unsafe { stream.alloc::<u8>(packed_size)? };

    // Quantize
    {
        let (in_ptr, _g1) = input.device_ptr(&stream);
        let (min_ptr, _g2) = min_vals.device_ptr(&stream);
        let (max_ptr, _g3) = max_vals.device_ptr(&stream);
        let (out_ptr, _g4) = packed.device_ptr_mut(&stream);
        let err = unsafe {
            quantize_intx(
                in_ptr as *const f32,
                min_ptr as *const f32,
                max_ptr as *const f32,
                out_ptr as *mut u8,
                batch,
                total_features,
                start_feature,
                range_features,
                int_bits,
                raw,
            )
        };
        assert_eq!(err, cuda_runtime::cudaError::cudaSuccess);
    }

    // Dequantize
    let mut decompressed = stream.alloc_zeros::<f32>(batch * total_features)?;
    {
        let (packed_ptr, _g1) = packed.device_ptr(&stream);
        let (min_ptr, _g2) = min_vals.device_ptr(&stream);
        let (max_ptr, _g3) = max_vals.device_ptr(&stream);
        let (dec_ptr, _g4) = decompressed.device_ptr_mut(&stream);
        let err = unsafe {
            dequantize_intx(
                packed_ptr as *const u8,
                min_ptr as *const f32,
                max_ptr as *const f32,
                dec_ptr as *mut f32,
                batch,
                total_features,
                start_feature,
                range_features,
                int_bits,
                raw,
            )
        };
        assert_eq!(err, cuda_runtime::cudaError::cudaSuccess);
    }

    stream.synchronize()?;

    // Verify: error should be within one quantization step
    let host_min = stream.clone_dtoh(&min_vals)?;
    let host_max = stream.clone_dtoh(&max_vals)?;
    let result = stream.clone_dtoh(&decompressed)?;

    let max_quant = ((1 << int_bits) - 1) as f32;
    for row in 0..batch {
        let interval = host_max[row] - host_min[row];
        let step = if interval < 1e-10 {
            1.0
        } else {
            interval / max_quant
        };
        for col in 0..range_features {
            let idx = row * total_features + col;
            let err = (host_data[idx] - result[idx]).abs();
            assert!(
                err <= step + 1e-5,
                "IntX({int_bits}) roundtrip error at [{row},{col}]: original={}, reconstructed={}, err={err}, step={step}",
                host_data[idx],
                result[idx]
            );
        }
    }

    Ok(())
}

#[test]
fn intx_roundtrip_8bit() -> Result<(), DriverError> {
    intx_roundtrip_test(8)
}

#[test]
fn intx_roundtrip_4bit() -> Result<(), DriverError> {
    intx_roundtrip_test(4)
}

#[test]
fn intx_roundtrip_2bit() -> Result<(), DriverError> {
    intx_roundtrip_test(2)
}

#[test]
fn intx_roundtrip_1bit() -> Result<(), DriverError> {
    intx_roundtrip_test(1)
}

// ---------------------------------------------------------------------------
// Min/max reduction correctness
// ---------------------------------------------------------------------------

#[test]
fn minmax_reduce_correctness() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let batch = 3usize;
    let total_features = 16usize;

    // Known data: each row has a distinct min and max
    let host_data: Vec<f32> = vec![
        // Row 0: min=-5.0 max=10.0
        1.0, 3.0, -5.0, 7.0, 10.0, 2.0, -1.0, 0.0, 4.0, 6.0, 8.0, -3.0, 9.0, -2.0, 5.0, -4.0,
        // Row 1: min=-100.0 max=200.0
        0.0, 200.0, -100.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, -50.0, 100.0,
        // Row 2: all same value 42.0
        42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0,
        42.0,
    ];

    let input = stream.clone_htod(&host_data)?;
    let mut min_vals = stream.alloc_zeros::<f32>(batch)?;
    let mut max_vals = stream.alloc_zeros::<f32>(batch)?;

    {
        let (in_ptr, _g1) = input.device_ptr(&stream);
        let (min_ptr, _g2) = min_vals.device_ptr_mut(&stream);
        let (max_ptr, _g3) = max_vals.device_ptr_mut(&stream);
        let err = unsafe {
            minmax_reduce(
                in_ptr as *const f32,
                min_ptr as *mut f32,
                max_ptr as *mut f32,
                batch,
                total_features,
                0,
                total_features,
                raw,
            )
        };
        assert_eq!(err, cuda_runtime::cudaError::cudaSuccess);
    }
    stream.synchronize()?;

    let host_min = stream.clone_dtoh(&min_vals)?;
    let host_max = stream.clone_dtoh(&max_vals)?;

    assert_eq!(host_min[0], -5.0);
    assert_eq!(host_max[0], 10.0);
    assert_eq!(host_min[1], -100.0);
    assert_eq!(host_max[1], 200.0);
    assert_eq!(host_min[2], 42.0);
    assert_eq!(host_max[2], 42.0);

    Ok(())
}

// ---------------------------------------------------------------------------
// Range header serialization roundtrip (pure Rust, no GPU)
// ---------------------------------------------------------------------------

#[test]
fn range_header_roundtrip() {
    let header = KvtcRangeHeader {
        quant_type: KvtcQuantType::IntX as i32,
        int_bits: 4,
        start_idx: 100,
        end_idx: 200,
        packed_data_bytes: 1234,
        metadata_bytes: 56,
    };

    let header_size = std::mem::size_of::<KvtcRangeHeader>();
    let mut buf = vec![0u8; header_size];

    unsafe {
        std::ptr::copy_nonoverlapping(
            &header as *const KvtcRangeHeader as *const u8,
            buf.as_mut_ptr(),
            header_size,
        );
    }

    let mut restored = unsafe { std::mem::zeroed::<KvtcRangeHeader>() };
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.as_ptr(),
            &mut restored as *mut KvtcRangeHeader as *mut u8,
            header_size,
        );
    }

    assert_eq!(restored.quant_type, KvtcQuantType::IntX as i32);
    assert_eq!(restored.int_bits, 4);
    assert_eq!(restored.start_idx, 100);
    assert_eq!(restored.end_idx, 200);
    assert_eq!(restored.packed_data_bytes, 1234);
    assert_eq!(restored.metadata_bytes, 56);
}

// ---------------------------------------------------------------------------
// Multi-range orchestrator roundtrip
// ---------------------------------------------------------------------------

#[test]
fn multi_range_quantize_dequantize() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let batch = 4usize;
    let total_features = 64usize;

    // Two ranges: FP8 for first 32 features, 4-bit IntX for last 32
    let ranges = vec![
        KvtcTypeRange {
            start_idx: 0,
            end_idx: 32,
            quant_type: KvtcQuantType::Fp8,
            int_bits: 0,
        },
        KvtcTypeRange {
            start_idx: 32,
            end_idx: 64,
            quant_type: KvtcQuantType::IntX,
            int_bits: 4,
        },
    ];

    // Generate input data
    let mut host_data = vec![0.0f32; batch * total_features];
    for (i, v) in host_data.iter_mut().enumerate() {
        *v = ((i as f32) * 3.7 - 100.0).sin() * 50.0;
    }

    let input = stream.clone_htod(&host_data)?;

    // Allocate output buffer (pinned host for realistic use case)
    let compressed_size = kvtc_compressed_size(&ranges, batch);
    let output_buf = PinnedBuffer::new_zeroed(compressed_size + 256); // extra padding

    // Min/max workspace
    let mut minmax_ws = stream.alloc_zeros::<f32>(batch * 2)?;

    // Quantize ranges
    let bytes_written = {
        let (in_ptr, _g1) = input.device_ptr(&stream);
        let (mm_ptr, _g2) = minmax_ws.device_ptr_mut(&stream);
        unsafe {
            kvtc_quantize_ranges(
                in_ptr as *const f32,
                output_buf.as_mut_ptr(),
                &ranges,
                batch,
                total_features,
                mm_ptr as *mut f32,
                raw,
            )
        }
    };
    stream.synchronize()?;
    let bytes_written = bytes_written.expect("kvtc_quantize_ranges failed");
    assert_eq!(bytes_written, compressed_size, "Compressed size mismatch");

    // Dequantize ranges
    let mut decompressed = stream.alloc_zeros::<f32>(batch * total_features)?;
    {
        let (dec_ptr, _g1) = decompressed.device_ptr_mut(&stream);
        let (mm_ptr, _g2) = minmax_ws.device_ptr_mut(&stream);
        let result = unsafe {
            kvtc_dequantize_ranges(
                output_buf.as_mut_ptr() as *const u8,
                dec_ptr as *mut f32,
                ranges.len(),
                batch,
                total_features,
                mm_ptr as *mut f32,
                raw,
            )
        };
        result.expect("kvtc_dequantize_ranges failed");
    }
    stream.synchronize()?;

    let result = stream.clone_dtoh(&decompressed)?;

    // Verify FP8 range (features 0..32): ~12.5% relative error
    for row in 0..batch {
        for col in 0..32 {
            let idx = row * total_features + col;
            let orig = host_data[idx];
            let recon = result[idx];
            if orig == 0.0 {
                assert!(
                    recon.abs() < 0.5,
                    "FP8 range: expected near-zero at [{row},{col}], got {recon}"
                );
            } else {
                let rel_err = ((orig - recon) / orig).abs();
                assert!(
                    rel_err < 0.15,
                    "FP8 range error at [{row},{col}]: orig={orig}, recon={recon}, rel_err={rel_err}"
                );
            }
        }
    }

    // Verify IntX 4-bit range (features 32..64): within quantization step
    // We need per-row min/max to compute step size, so just check generous bound
    for row in 0..batch {
        let row_slice = &host_data[row * total_features + 32..row * total_features + 64];
        let row_min = row_slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let row_max = row_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let interval = row_max - row_min;
        let step = if interval < 1e-10 {
            1.0
        } else {
            interval / 15.0
        };

        for col in 32..64 {
            let idx = row * total_features + col;
            let err = (host_data[idx] - result[idx]).abs();
            assert!(
                err <= step + 1e-4,
                "IntX 4-bit error at [{row},{col}]: orig={}, recon={}, err={err}, step={step}",
                host_data[idx],
                result[idx]
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 2: Gather/scatter roundtrip (f32)
// ---------------------------------------------------------------------------

#[test]
fn gather_scatter_f32_roundtrip() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let num_blocks = 3usize;
    let features = 16usize;

    // Create scattered blocks: each block is a separate device allocation
    let mut host_blocks: Vec<Vec<f32>> = Vec::new();
    for b in 0..num_blocks {
        let block: Vec<f32> = (0..features)
            .map(|f| (b * features + f) as f32 * 1.5 + 0.7)
            .collect();
        host_blocks.push(block);
    }

    // Upload each block separately
    let device_blocks: Vec<CudaSlice<f32>> = host_blocks
        .iter()
        .map(|b| stream.clone_htod(b))
        .collect::<Result<_, _>>()?;

    // Build device pointer array
    let block_ptrs: Vec<u64> = device_blocks
        .iter()
        .map(|b| {
            let (ptr, _guard) = b.device_ptr(&stream);
            ptr as u64
        })
        .collect();
    let device_ptr_array = stream.clone_htod(&block_ptrs)?;

    // Mean vector (all zeros for simple roundtrip)
    let mean = vec![0.0f32; features];
    let device_mean = stream.clone_htod(&mean)?;

    // Output workspace
    let mut output = stream.alloc_zeros::<f32>(num_blocks * features)?;

    // Gather
    {
        let (ptr_arr, _g1) = device_ptr_array.device_ptr(&stream);
        let (mean_ptr, _g2) = device_mean.device_ptr(&stream);
        let (out_ptr, _g3) = output.device_ptr_mut(&stream);
        let err = unsafe {
            gather_mean_subtract(
                ptr_arr as *const *const c_void,
                mean_ptr as *const f32,
                out_ptr as *mut f32,
                num_blocks,
                features,
                features,
                TensorDataType::F32,
                raw,
            )
        };
        assert_eq!(err, cuda_runtime::cudaError::cudaSuccess);
    }

    // Allocate output blocks for scatter
    let mut out_blocks: Vec<CudaSlice<f32>> = Vec::new();
    for _ in 0..num_blocks {
        out_blocks.push(stream.alloc_zeros::<f32>(features)?);
    }
    let out_ptrs: Vec<u64> = out_blocks
        .iter_mut()
        .map(|b| {
            let (ptr, _guard) = b.device_ptr_mut(&stream);
            ptr as u64
        })
        .collect();
    let device_out_ptr_array = stream.clone_htod(&out_ptrs)?;

    // Scatter
    {
        let (out_gathered_ptr, _g1) = output.device_ptr(&stream);
        let (mean_ptr, _g2) = device_mean.device_ptr(&stream);
        let (ptr_arr, _g3) = device_out_ptr_array.device_ptr(&stream);
        let err = unsafe {
            mean_add_scatter(
                out_gathered_ptr as *const f32,
                mean_ptr as *const f32,
                ptr_arr as *const *mut c_void,
                num_blocks,
                features,
                features,
                TensorDataType::F32,
                raw,
            )
        };
        assert_eq!(err, cuda_runtime::cudaError::cudaSuccess);
    }

    stream.synchronize()?;

    // Verify each block matches original
    for (b, (orig, out_block)) in host_blocks.iter().zip(out_blocks.iter()).enumerate() {
        let result = stream.clone_dtoh(out_block)?;
        for (f, (&expected, &actual)) in orig.iter().zip(result.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-5,
                "Block {b} feature {f}: expected {expected}, got {actual}"
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 2: Gather/scatter with mean subtraction/addition
// ---------------------------------------------------------------------------

#[test]
fn gather_scatter_mean_roundtrip() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let num_blocks = 4usize;
    let features = 32usize;

    // Create blocks with known data
    let mut host_blocks: Vec<Vec<f32>> = Vec::new();
    for b in 0..num_blocks {
        let block: Vec<f32> = (0..features)
            .map(|f| ((b * features + f) as f32 * 2.3).sin() * 10.0)
            .collect();
        host_blocks.push(block);
    }

    // Non-trivial mean
    let mean: Vec<f32> = (0..features).map(|f| f as f32 * 0.5 - 8.0).collect();

    let device_blocks: Vec<CudaSlice<f32>> = host_blocks
        .iter()
        .map(|b| stream.clone_htod(b))
        .collect::<Result<_, _>>()?;
    let block_ptrs: Vec<u64> = device_blocks
        .iter()
        .map(|b| {
            let (p, _) = b.device_ptr(&stream);
            p as u64
        })
        .collect();
    let device_ptr_array = stream.clone_htod(&block_ptrs)?;
    let device_mean = stream.clone_htod(&mean)?;
    let mut centered = stream.alloc_zeros::<f32>(num_blocks * features)?;

    // Gather + mean subtract
    {
        let (pa, _g1) = device_ptr_array.device_ptr(&stream);
        let (mp, _g2) = device_mean.device_ptr(&stream);
        let (op, _g3) = centered.device_ptr_mut(&stream);
        let err = unsafe {
            gather_mean_subtract(
                pa as *const *const c_void,
                mp as *const f32,
                op as *mut f32,
                num_blocks,
                features,
                features,
                TensorDataType::F32,
                raw,
            )
        };
        assert_eq!(err, cuda_runtime::cudaError::cudaSuccess);
    }

    // Verify centered values
    stream.synchronize()?;
    let centered_host = stream.clone_dtoh(&centered)?;
    for b in 0..num_blocks {
        for f in 0..features {
            let expected = host_blocks[b][f] - mean[f];
            let actual = centered_host[b * features + f];
            assert!(
                (expected - actual).abs() < 1e-5,
                "Centered [{b},{f}]: expected {expected}, got {actual}"
            );
        }
    }

    // Mean-add + scatter back
    let mut out_blocks: Vec<CudaSlice<f32>> = Vec::new();
    for _ in 0..num_blocks {
        out_blocks.push(stream.alloc_zeros::<f32>(features)?);
    }
    let out_ptrs: Vec<u64> = out_blocks
        .iter_mut()
        .map(|b| {
            let (p, _) = b.device_ptr_mut(&stream);
            p as u64
        })
        .collect();
    let device_out_ptr_array = stream.clone_htod(&out_ptrs)?;

    {
        let (cp, _g1) = centered.device_ptr(&stream);
        let (mp, _g2) = device_mean.device_ptr(&stream);
        let (pa, _g3) = device_out_ptr_array.device_ptr(&stream);
        let err = unsafe {
            mean_add_scatter(
                cp as *const f32,
                mp as *const f32,
                pa as *const *mut c_void,
                num_blocks,
                features,
                features,
                TensorDataType::F32,
                raw,
            )
        };
        assert_eq!(err, cuda_runtime::cudaError::cudaSuccess);
    }

    stream.synchronize()?;

    for (b, (orig, out_block)) in host_blocks.iter().zip(out_blocks.iter()).enumerate() {
        let result = stream.clone_dtoh(out_block)?;
        for (f, (&expected, &actual)) in orig.iter().zip(result.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-4,
                "Roundtrip [{b},{f}]: expected {expected}, got {actual}"
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 2: Full compress/decompress with identity projection
// ---------------------------------------------------------------------------

#[test]
fn identity_projection_roundtrip() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let num_blocks = 4usize;
    let features = 16usize;
    let pca_components = features; // Identity: same dimensionality

    // Identity projection matrix [features, pca_components] = I
    let mut projection = vec![0.0f32; features * pca_components];
    for i in 0..features {
        projection[i * pca_components + i] = 1.0;
    }
    let device_projection = stream.clone_htod(&projection)?;

    // Zero mean (simplest case)
    let mean = vec![0.0f32; features];
    let device_mean = stream.clone_htod(&mean)?;

    // Create scattered blocks
    let mut host_blocks: Vec<Vec<f32>> = Vec::new();
    for b in 0..num_blocks {
        let block: Vec<f32> = (0..features)
            .map(|f| ((b * features + f) as f32 * 1.7).sin() * 5.0)
            .collect();
        host_blocks.push(block);
    }

    let device_blocks: Vec<CudaSlice<f32>> = host_blocks
        .iter()
        .map(|b| stream.clone_htod(b))
        .collect::<Result<_, _>>()?;
    let in_ptrs: Vec<u64> = device_blocks
        .iter()
        .map(|b| {
            let (p, _) = b.device_ptr(&stream);
            p as u64
        })
        .collect();
    let device_in_ptrs = stream.clone_htod(&in_ptrs)?;

    // Config: single FP8 range covering all components
    let ranges = vec![KvtcTypeRange {
        start_idx: 0,
        end_idx: pca_components,
        quant_type: KvtcQuantType::Fp8,
        int_bits: 0,
    }];

    let (mean_ptr, _gm) = device_mean.device_ptr(&stream);
    let (proj_ptr, _gp) = device_projection.device_ptr(&stream);

    let config = KvtcConfig {
        mean: mean_ptr as *const f32,
        projection: proj_ptr as *const f32,
        features,
        pca_components,
        ranges: ranges.clone(),
    };

    // Allocate workspace
    let ws_bytes = kvtc_workspace_size(features, pca_components, num_blocks);
    let mut workspace = stream.alloc_zeros::<f32>(ws_bytes / 4)?;
    let mut minmax_ws = stream.alloc_zeros::<f32>(num_blocks * 2)?;

    // Compressed output buffer (pinned host)
    let compressed_size = kvtc_compressed_size(&ranges, num_blocks);
    let output_buf = PinnedBuffer::new_zeroed(compressed_size + 256);

    // Create cuBLAS handle
    let cublas = unsafe { kvtc_create_cublas_handle().expect("cuBLAS create failed") };

    // Compress
    let bytes_written = {
        let (in_pa, _g1) = device_in_ptrs.device_ptr(&stream);
        let (ws_ptr, _g2) = workspace.device_ptr_mut(&stream);
        let (mm_ptr, _g3) = minmax_ws.device_ptr_mut(&stream);
        unsafe {
            kvtc_compress(
                in_pa as *const *const c_void,
                &config,
                output_buf.as_mut_ptr(),
                ws_ptr as *mut f32,
                mm_ptr as *mut f32,
                num_blocks,
                features,
                TensorDataType::F32,
                cublas,
                raw,
            )
        }
    };
    stream.synchronize()?;
    let bytes_written = bytes_written.expect("kvtc_compress failed");
    assert_eq!(bytes_written, compressed_size);

    // Decompress into new scattered blocks
    let mut out_blocks: Vec<CudaSlice<f32>> = Vec::new();
    for _ in 0..num_blocks {
        out_blocks.push(stream.alloc_zeros::<f32>(features)?);
    }
    let out_ptrs: Vec<u64> = out_blocks
        .iter_mut()
        .map(|b| {
            let (p, _) = b.device_ptr_mut(&stream);
            p as u64
        })
        .collect();
    let device_out_ptrs = stream.clone_htod(&out_ptrs)?;

    {
        let (out_pa, _g1) = device_out_ptrs.device_ptr(&stream);
        let (ws_ptr, _g2) = workspace.device_ptr_mut(&stream);
        let (mm_ptr, _g3) = minmax_ws.device_ptr_mut(&stream);
        unsafe {
            kvtc_decompress(
                output_buf.as_mut_ptr() as *const u8,
                &config,
                out_pa as *const *mut c_void,
                ws_ptr as *mut f32,
                mm_ptr as *mut f32,
                num_blocks,
                features,
                TensorDataType::F32,
                cublas,
                raw,
            )
        }
        .expect("kvtc_decompress failed");
    }
    stream.synchronize()?;

    // Verify: identity projection + FP8 quantization → ~12.5% relative error
    for (b, (orig, out_block)) in host_blocks.iter().zip(out_blocks.iter()).enumerate() {
        let result = stream.clone_dtoh(out_block)?;
        for (f, (&expected, &actual)) in orig.iter().zip(result.iter()).enumerate() {
            if expected == 0.0 {
                assert!(
                    actual.abs() < 0.5,
                    "Identity roundtrip [{b},{f}]: expected ~0, got {actual}"
                );
            } else {
                let rel_err = ((expected - actual) / expected).abs();
                assert!(
                    rel_err < 0.15,
                    "Identity roundtrip [{b},{f}]: expected {expected}, got {actual}, rel_err={rel_err}"
                );
            }
        }
    }

    unsafe { kvtc_destroy_cublas_handle(cublas) };
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 2: Full compress/decompress with orthogonal projection + mean
// ---------------------------------------------------------------------------

#[test]
fn orthogonal_projection_roundtrip() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let num_blocks = 4usize;
    let features = 16usize;
    let pca_components = 8usize; // Dimensionality reduction: 16 → 8

    // Build a simple orthogonal projection via Gram-Schmidt on random-ish vectors.
    // For reproducibility, use deterministic "random" values.
    let mut basis = vec![vec![0.0f32; features]; pca_components];
    for (i, row) in basis.iter_mut().enumerate() {
        for (j, val) in row.iter_mut().enumerate() {
            // Deterministic pseudo-random using a simple hash
            *val = ((i * 137 + j * 31 + 17) as f32 * 0.618).sin();
        }
    }

    // Gram-Schmidt orthonormalization
    for i in 0..pca_components {
        for j in 0..i {
            let dot: f32 = (0..features).map(|k| basis[i][k] * basis[j][k]).sum();
            for k in 0..features {
                basis[i][k] -= dot * basis[j][k];
            }
        }
        let norm: f32 = (0..features)
            .map(|k| basis[i][k] * basis[i][k])
            .sum::<f32>()
            .sqrt();
        assert!(norm > 1e-6, "Gram-Schmidt produced zero vector at step {i}");
        for k in 0..features {
            basis[i][k] /= norm;
        }
    }

    // Projection matrix [features, pca_components] row-major
    // Each column is a basis vector: projection[feat][comp] = basis[comp][feat]
    let mut projection = vec![0.0f32; features * pca_components];
    for feat in 0..features {
        for comp in 0..pca_components {
            projection[feat * pca_components + comp] = basis[comp][feat];
        }
    }

    let device_projection = stream.clone_htod(&projection)?;

    // Non-zero mean
    let mean: Vec<f32> = (0..features)
        .map(|f| (f as f32 * 0.3).cos() * 2.0)
        .collect();
    let device_mean = stream.clone_htod(&mean)?;

    // Create scattered blocks
    let mut host_blocks: Vec<Vec<f32>> = Vec::new();
    for b in 0..num_blocks {
        let block: Vec<f32> = (0..features)
            .map(|f| ((b * features + f) as f32 * 0.97).sin() * 3.0 + mean[f])
            .collect();
        host_blocks.push(block);
    }

    let device_blocks: Vec<CudaSlice<f32>> = host_blocks
        .iter()
        .map(|b| stream.clone_htod(b))
        .collect::<Result<_, _>>()?;
    let in_ptrs: Vec<u64> = device_blocks
        .iter()
        .map(|b| {
            let (p, _) = b.device_ptr(&stream);
            p as u64
        })
        .collect();
    let device_in_ptrs = stream.clone_htod(&in_ptrs)?;

    // Config: 8-bit IntX for all components (high precision for testing)
    let ranges = vec![KvtcTypeRange {
        start_idx: 0,
        end_idx: pca_components,
        quant_type: KvtcQuantType::IntX,
        int_bits: 8,
    }];

    let (mean_ptr, _gm) = device_mean.device_ptr(&stream);
    let (proj_ptr, _gp) = device_projection.device_ptr(&stream);

    let config = KvtcConfig {
        mean: mean_ptr as *const f32,
        projection: proj_ptr as *const f32,
        features,
        pca_components,
        ranges: ranges.clone(),
    };

    let ws_bytes = kvtc_workspace_size(features, pca_components, num_blocks);
    let mut workspace = stream.alloc_zeros::<f32>(ws_bytes / 4)?;
    let mut minmax_ws = stream.alloc_zeros::<f32>(num_blocks * 2)?;

    let compressed_size = kvtc_compressed_size(&ranges, num_blocks);
    let output_buf = PinnedBuffer::new_zeroed(compressed_size + 256);

    let cublas = unsafe { kvtc_create_cublas_handle().expect("cuBLAS create failed") };

    // Compress
    let bytes_written = {
        let (in_pa, _g1) = device_in_ptrs.device_ptr(&stream);
        let (ws_ptr, _g2) = workspace.device_ptr_mut(&stream);
        let (mm_ptr, _g3) = minmax_ws.device_ptr_mut(&stream);
        unsafe {
            kvtc_compress(
                in_pa as *const *const c_void,
                &config,
                output_buf.as_mut_ptr(),
                ws_ptr as *mut f32,
                mm_ptr as *mut f32,
                num_blocks,
                features,
                TensorDataType::F32,
                cublas,
                raw,
            )
        }
    };
    stream.synchronize()?;
    let bytes_written = bytes_written.expect("kvtc_compress failed");
    assert_eq!(bytes_written, compressed_size);

    // Decompress
    let mut out_blocks: Vec<CudaSlice<f32>> = Vec::new();
    for _ in 0..num_blocks {
        out_blocks.push(stream.alloc_zeros::<f32>(features)?);
    }
    let out_ptrs: Vec<u64> = out_blocks
        .iter_mut()
        .map(|b| {
            let (p, _) = b.device_ptr_mut(&stream);
            p as u64
        })
        .collect();
    let device_out_ptrs = stream.clone_htod(&out_ptrs)?;

    {
        let (out_pa, _g1) = device_out_ptrs.device_ptr(&stream);
        let (ws_ptr, _g2) = workspace.device_ptr_mut(&stream);
        let (mm_ptr, _g3) = minmax_ws.device_ptr_mut(&stream);
        unsafe {
            kvtc_decompress(
                output_buf.as_mut_ptr() as *const u8,
                &config,
                out_pa as *const *mut c_void,
                ws_ptr as *mut f32,
                mm_ptr as *mut f32,
                num_blocks,
                features,
                TensorDataType::F32,
                cublas,
                raw,
            )
        }
        .expect("kvtc_decompress failed");
    }
    stream.synchronize()?;

    // With dimensionality reduction (16→8), we lose information in the null space.
    // Compute the expected reconstruction: project into PCA space and back.
    // reconstructed = (data - mean) @ P @ P^T + mean
    for (b, (orig, out_block)) in host_blocks.iter().zip(out_blocks.iter()).enumerate() {
        let result = stream.clone_dtoh(out_block)?;

        // Compute expected: project to PCA space and back (ignoring quantization)
        let centered: Vec<f32> = orig.iter().zip(mean.iter()).map(|(v, m)| v - m).collect();

        // projected = centered @ projection [features, pca_comp]
        let mut projected = vec![0.0f32; pca_components];
        for c in 0..pca_components {
            for f in 0..features {
                projected[c] += centered[f] * projection[f * pca_components + c];
            }
        }

        // reconstructed = projected @ projection^T + mean
        let mut expected = vec![0.0f32; features];
        for f in 0..features {
            for c in 0..pca_components {
                expected[f] += projected[c] * projection[f * pca_components + c];
            }
            expected[f] += mean[f];
        }

        // The error comes from:
        // 1. Projection loss (null space truncation) — exact
        // 2. IntX 8-bit quantization — small per-step error
        for f in 0..features {
            let err = (expected[f] - result[f]).abs();
            // IntX 8-bit has 255 levels, so quantization error is bounded by step/2.
            // Allow generous tolerance since PCA + quantization compound errors.
            assert!(
                err < 1.0,
                "Ortho roundtrip [{b},{f}]: expected {:.4}, got {:.4}, err={err:.4}",
                expected[f],
                result[f]
            );
        }
    }

    unsafe { kvtc_destroy_cublas_handle(cublas) };
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 3: Full pipeline multi-range with PCA (mixed FP8 + IntX + mean)
// ---------------------------------------------------------------------------

#[test]
fn full_pipeline_multi_range() -> Result<(), DriverError> {
    let (stream, raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let num_blocks = 6usize;
    let features = 32usize;
    let pca_components = 24usize; // Dimensionality reduction: 32 → 24

    // Build orthonormal projection via Gram-Schmidt
    let mut basis = vec![vec![0.0f32; features]; pca_components];
    for (i, row) in basis.iter_mut().enumerate() {
        for (j, val) in row.iter_mut().enumerate() {
            *val = ((i * 97 + j * 53 + 7) as f32 * 1.234).sin();
        }
    }
    for i in 0..pca_components {
        for j in 0..i {
            let dot: f32 = (0..features).map(|k| basis[i][k] * basis[j][k]).sum();
            for k in 0..features {
                basis[i][k] -= dot * basis[j][k];
            }
        }
        let norm: f32 = (0..features)
            .map(|k| basis[i][k] * basis[i][k])
            .sum::<f32>()
            .sqrt();
        for k in 0..features {
            basis[i][k] /= norm;
        }
    }

    let mut projection = vec![0.0f32; features * pca_components];
    for feat in 0..features {
        for comp in 0..pca_components {
            projection[feat * pca_components + comp] = basis[comp][feat];
        }
    }

    let device_projection = stream.clone_htod(&projection)?;
    let mean: Vec<f32> = (0..features)
        .map(|f| (f as f32 * 0.7).sin() * 5.0)
        .collect();
    let device_mean = stream.clone_htod(&mean)?;

    // Create scattered blocks with varied data
    let mut host_blocks: Vec<Vec<f32>> = Vec::new();
    for b in 0..num_blocks {
        let block: Vec<f32> = (0..features)
            .map(|f| {
                let x = (b * features + f) as f32;
                (x * 0.37).sin() * 8.0 + (x * 1.1).cos() * 3.0 + mean[f]
            })
            .collect();
        host_blocks.push(block);
    }

    let device_blocks: Vec<CudaSlice<f32>> = host_blocks
        .iter()
        .map(|b| stream.clone_htod(b))
        .collect::<Result<_, _>>()?;
    let in_ptrs: Vec<u64> = device_blocks
        .iter()
        .map(|b| {
            let (p, _) = b.device_ptr(&stream);
            p as u64
        })
        .collect();
    let device_in_ptrs = stream.clone_htod(&in_ptrs)?;

    // Mixed quantization: FP8 for first 8 components, 4-bit IntX for next 8, 2-bit IntX for last 8
    let ranges = vec![
        KvtcTypeRange {
            start_idx: 0,
            end_idx: 8,
            quant_type: KvtcQuantType::Fp8,
            int_bits: 0,
        },
        KvtcTypeRange {
            start_idx: 8,
            end_idx: 16,
            quant_type: KvtcQuantType::IntX,
            int_bits: 4,
        },
        KvtcTypeRange {
            start_idx: 16,
            end_idx: 24,
            quant_type: KvtcQuantType::IntX,
            int_bits: 2,
        },
    ];

    let (mean_ptr, _gm) = device_mean.device_ptr(&stream);
    let (proj_ptr, _gp) = device_projection.device_ptr(&stream);

    // Use KvtcConfig::new constructor (Phase 3 API)
    let config = KvtcConfig::new(
        mean_ptr as *const f32,
        proj_ptr as *const f32,
        features,
        pca_components,
        ranges,
    );

    // Use method-style helpers
    let ws_bytes = config.workspace_size(num_blocks);
    let compressed_size = config.compressed_size(num_blocks);

    let mut workspace = stream.alloc_zeros::<f32>(ws_bytes / 4)?;
    let mut minmax_ws = stream.alloc_zeros::<f32>(num_blocks * 2)?;
    let output_buf = PinnedBuffer::new_zeroed(compressed_size + 256);

    let cublas = unsafe { kvtc_create_cublas_handle().expect("cuBLAS create failed") };

    // Compress
    let bytes_written = {
        let (in_pa, _g1) = device_in_ptrs.device_ptr(&stream);
        let (ws_ptr, _g2) = workspace.device_ptr_mut(&stream);
        let (mm_ptr, _g3) = minmax_ws.device_ptr_mut(&stream);
        unsafe {
            kvtc_compress(
                in_pa as *const *const c_void,
                &config,
                output_buf.as_mut_ptr(),
                ws_ptr as *mut f32,
                mm_ptr as *mut f32,
                num_blocks,
                features,
                TensorDataType::F32,
                cublas,
                raw,
            )
        }
    };
    stream.synchronize()?;
    let bytes_written = bytes_written.expect("kvtc_compress failed");
    assert_eq!(bytes_written, compressed_size, "Compressed size mismatch");

    // Decompress into new scattered blocks
    let mut out_blocks: Vec<CudaSlice<f32>> = Vec::new();
    for _ in 0..num_blocks {
        out_blocks.push(stream.alloc_zeros::<f32>(features)?);
    }
    let out_ptrs: Vec<u64> = out_blocks
        .iter_mut()
        .map(|b| {
            let (p, _) = b.device_ptr_mut(&stream);
            p as u64
        })
        .collect();
    let device_out_ptrs = stream.clone_htod(&out_ptrs)?;

    {
        let (out_pa, _g1) = device_out_ptrs.device_ptr(&stream);
        let (ws_ptr, _g2) = workspace.device_ptr_mut(&stream);
        let (mm_ptr, _g3) = minmax_ws.device_ptr_mut(&stream);
        unsafe {
            kvtc_decompress(
                output_buf.as_mut_ptr() as *const u8,
                &config,
                out_pa as *const *mut c_void,
                ws_ptr as *mut f32,
                mm_ptr as *mut f32,
                num_blocks,
                features,
                TensorDataType::F32,
                cublas,
                raw,
            )
        }
        .expect("kvtc_decompress failed");
    }
    stream.synchronize()?;

    // Compute expected reconstruction on CPU (project + quantize-approximate + inverse project)
    // Just check that reconstruction is reasonable — exact tolerance depends on quantization
    for (b, (orig, out_block)) in host_blocks.iter().zip(out_blocks.iter()).enumerate() {
        let result = stream.clone_dtoh(out_block)?;

        // Compute expected via CPU: center → project → inverse-project → uncenter
        let centered: Vec<f32> = orig.iter().zip(mean.iter()).map(|(v, m)| v - m).collect();
        let mut projected = vec![0.0f32; pca_components];
        for c in 0..pca_components {
            for f in 0..features {
                projected[c] += centered[f] * projection[f * pca_components + c];
            }
        }
        let mut expected = vec![0.0f32; features];
        for f in 0..features {
            for c in 0..pca_components {
                expected[f] += projected[c] * projection[f * pca_components + c];
            }
            expected[f] += mean[f];
        }

        // The reconstruction error comes from PCA truncation + quantization.
        // With 32→24 features and mixed quantization (FP8 + 4bit + 2bit),
        // expect moderate but bounded error.
        let mut max_err: f32 = 0.0;
        for f in 0..features {
            let err = (expected[f] - result[f]).abs();
            max_err = max_err.max(err);
        }
        // With 2-bit quantization in some ranges, errors can be significant
        // but should still be bounded.
        assert!(
            max_err < 5.0,
            "Block {b}: max reconstruction error {max_err:.4} exceeds tolerance"
        );
    }

    unsafe { kvtc_destroy_cublas_handle(cublas) };
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 3: KvtcConfig method helpers
// ---------------------------------------------------------------------------

#[test]
fn config_helper_methods() {
    let ranges = vec![
        KvtcTypeRange {
            start_idx: 0,
            end_idx: 50,
            quant_type: KvtcQuantType::Fp8,
            int_bits: 0,
        },
        KvtcTypeRange {
            start_idx: 50,
            end_idx: 100,
            quant_type: KvtcQuantType::IntX,
            int_bits: 4,
        },
    ];

    let config = KvtcConfig::new(std::ptr::null(), std::ptr::null(), 256, 100, ranges);

    let batch = 8;
    assert_eq!(
        config.compressed_size(batch),
        kvtc_compressed_size(&config.ranges, batch)
    );
    assert_eq!(
        config.workspace_size(batch),
        kvtc_workspace_size(256, 100, batch)
    );
}

// ---------------------------------------------------------------------------
// Compressed size helper consistency
// ---------------------------------------------------------------------------

#[test]
fn compressed_size_helper() {
    let ranges = vec![
        KvtcTypeRange {
            start_idx: 0,
            end_idx: 100,
            quant_type: KvtcQuantType::Fp8,
            int_bits: 0,
        },
        KvtcTypeRange {
            start_idx: 100,
            end_idx: 200,
            quant_type: KvtcQuantType::IntX,
            int_bits: 4,
        },
        KvtcTypeRange {
            start_idx: 200,
            end_idx: 300,
            quant_type: KvtcQuantType::IntX,
            int_bits: 2,
        },
    ];

    let batch = 8;
    let header_size = std::mem::size_of::<KvtcRangeHeader>();

    // FP8: 8*100 = 800 packed bytes, 0 metadata
    // IntX 4-bit: 8*100 = 800 elements, 2 values/byte = 400 bytes, metadata = 8*2*4 = 64
    // IntX 2-bit: 8*100 = 800 elements, 4 values/byte = 200 bytes, metadata = 8*2*4 = 64
    let expected = 3 * header_size + (800 + 0) + (400 + 64) + (200 + 64);
    assert_eq!(kvtc_compressed_size(&ranges, batch), expected);
}
