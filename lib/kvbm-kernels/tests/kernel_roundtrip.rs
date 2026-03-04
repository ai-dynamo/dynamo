// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for CUDA tensor packing kernel roundtrips.
//!
//! Mirrors the Python tests in `lib/bindings/kvbm/tests/test_tensor_kernels.py`
//! using ndarray for reference permutations and cudarc for GPU memory management.

#![cfg(all(
    feature = "testing-cuda",
    feature = "permute_kernels",
    not(stub_kernels)
))]

use std::ffi::c_void;
use std::fmt::Debug;
use std::sync::Arc;

use cudarc::driver::result::memset_d8_async;
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, DriverError,
    ValidAsZeroBits,
};
use cudarc::runtime::sys as cuda_runtime;
use half::{bf16, f16};
use kvbm_kernels::{BlockLayout, TensorDataType, block_from_universal, universal_from_block};
use ndarray::{Array5, s};
use rand::Rng;

// ---------------------------------------------------------------------------
// TestDtype trait — bridges Rust types to kernel enums + tolerances
// ---------------------------------------------------------------------------

trait TestDtype: Clone + Debug + DeviceRepr + ValidAsZeroBits + 'static {
    const DTYPE: TensorDataType;
    const ATOL: f64;
    const RTOL: f64;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl TestDtype for f16 {
    const DTYPE: TensorDataType = TensorDataType::F16;
    const ATOL: f64 = 1e-2;
    const RTOL: f64 = 1e-2;

    fn from_f64(v: f64) -> Self {
        f16::from_f64(v)
    }
    fn to_f64(self) -> f64 {
        f16::to_f64(self)
    }
}

impl TestDtype for bf16 {
    const DTYPE: TensorDataType = TensorDataType::BF16;
    const ATOL: f64 = 1e-2;
    const RTOL: f64 = 1e-2;

    fn from_f64(v: f64) -> Self {
        bf16::from_f64(v)
    }
    fn to_f64(self) -> f64 {
        bf16::to_f64(self)
    }
}

impl TestDtype for f32 {
    const DTYPE: TensorDataType = TensorDataType::F32;
    const ATOL: f64 = 1e-5;
    const RTOL: f64 = 1e-5;

    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl TestDtype for f64 {
    const DTYPE: TensorDataType = TensorDataType::F64;
    const ATOL: f64 = 1e-12;
    const RTOL: f64 = 1e-12;

    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reference permutation using ndarray, mirrors the Python `_make_blocks()`.
///
/// Takes a `[nh, nl, no, nt, hd]` universal tensor and produces `nl * no` flat
/// block chunks, each with layout-dependent axis ordering.
fn make_blocks<T: TestDtype>(universal: &Array5<T>, layout: BlockLayout) -> Vec<Vec<T>> {
    let (_nh, nl, no, _nt, _hd) = universal.dim();
    let mut blocks = Vec::with_capacity(nl * no);
    for l in 0..nl {
        for o in 0..no {
            // Slice out [nh, nt, hd] for this (layer, outer) pair.
            let chunk = universal.slice(s![.., l, o, .., ..]);
            let flat = match layout {
                BlockLayout::NHD => {
                    // [nh, nt, hd] -> [nt, nh, hd]
                    let permuted = chunk.permuted_axes([1, 0, 2]);
                    permuted.as_standard_layout().as_slice().unwrap().to_vec()
                }
                BlockLayout::HND => {
                    // [nh, nt, hd] — identity permutation
                    chunk.as_standard_layout().as_slice().unwrap().to_vec()
                }
            };
            blocks.push(flat);
        }
    }
    blocks
}

/// Element-wise comparison with dtype-aware tolerance (mirrors `torch.allclose`).
fn assert_close<T: TestDtype>(actual: &[T], expected: &[T], context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a_f64 = a.clone().to_f64();
        let e_f64 = e.clone().to_f64();
        let diff = (a_f64 - e_f64).abs();
        let tol = T::ATOL + T::RTOL * e_f64.abs();
        assert!(
            diff <= tol,
            "{context}[{i}]: {a_f64} vs {e_f64} (diff={diff}, tol={tol})"
        );
    }
}

/// Set up a CUDA context and stream.  Returns `None` if no GPU is available.
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

// ---------------------------------------------------------------------------
// GPU allocation helpers
// ---------------------------------------------------------------------------

/// Upload block chunks to GPU, returning the slices (kept alive) and a device
/// pointer table suitable for the kernel FFI.
fn upload_blocks<T: TestDtype>(
    stream: &Arc<CudaStream>,
    ref_blocks: &[Vec<Vec<T>>],
) -> Result<(Vec<Vec<CudaSlice<T>>>, CudaSlice<usize>), DriverError> {
    let nb = ref_blocks.len();
    let chunks_per_batch = ref_blocks.first().map_or(0, |b| b.len());
    let mut all_slices: Vec<Vec<CudaSlice<T>>> = Vec::with_capacity(nb);
    let mut ptr_values: Vec<usize> = Vec::with_capacity(nb * chunks_per_batch);

    for batch in ref_blocks {
        let mut slices = Vec::with_capacity(batch.len());
        for chunk in batch {
            let slice = stream.clone_htod(chunk)?;
            {
                let (ptr, _guard) = slice.device_ptr(stream);
                ptr_values.push(ptr as usize);
            }
            slices.push(slice);
        }
        all_slices.push(slices);
    }

    let ptrs_device = stream.clone_htod(ptr_values.as_slice())?;
    Ok((all_slices, ptrs_device))
}

/// Allocate `count` poison-filled (0xDE) device buffers of `volume` elements each.
/// Returns the slices and a device pointer table.
fn alloc_buffers<T: TestDtype>(
    stream: &Arc<CudaStream>,
    count: usize,
    volume: usize,
) -> Result<(Vec<CudaSlice<T>>, CudaSlice<usize>), DriverError> {
    let mut slices: Vec<CudaSlice<T>> = Vec::with_capacity(count);
    let mut ptr_values: Vec<usize> = Vec::with_capacity(count);
    let byte_count = volume * std::mem::size_of::<T>();

    for _ in 0..count {
        let mut slice = unsafe { stream.alloc::<T>(volume)? };
        {
            let (ptr, _guard) = slice.device_ptr_mut(stream);
            ptr_values.push(ptr as usize);
            unsafe {
                memset_d8_async(ptr, 0xDE, byte_count, stream.cu_stream())?;
            }
        }
        slices.push(slice);
    }

    let ptrs_device = stream.clone_htod(ptr_values.as_slice())?;
    Ok((slices, ptrs_device))
}

/// Poison-fill (0xDE) all block chunk slices. `chunk_volume` is the element count per chunk.
fn poison_fill_blocks<T: TestDtype>(
    stream: &Arc<CudaStream>,
    block_slices: &mut [Vec<CudaSlice<T>>],
    chunk_volume: usize,
) -> Result<(), DriverError> {
    let byte_count = chunk_volume * std::mem::size_of::<T>();
    for batch in block_slices.iter_mut() {
        for slice in batch.iter_mut() {
            let (dptr, _guard) = slice.device_ptr_mut(stream);
            unsafe {
                memset_d8_async(dptr, 0xDE, byte_count, stream.cu_stream())?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// block <-> universal roundtrip
// ---------------------------------------------------------------------------

fn block_universal_roundtrip_inner<T: TestDtype>(layout: BlockLayout) -> Result<(), DriverError> {
    let (stream, stream_raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    // Dimensions matching the Python test.
    let nh = 3usize;
    let nl = 2usize;
    let no = 2usize;
    let nt = 4usize;
    let hd = 5usize;
    let nb = 3usize;
    let universal_volume = nh * nl * no * nt * hd;

    // Generate random universal tensors and compute reference blocks.
    let mut rng = rand::rng();
    let universals: Vec<Array5<T>> = (0..nb)
        .map(|_| {
            Array5::from_shape_fn((nh, nl, no, nt, hd), |_| {
                T::from_f64(rng.random::<f64>() * 2.0 - 1.0)
            })
        })
        .collect();

    let ref_blocks: Vec<Vec<Vec<T>>> = universals.iter().map(|u| make_blocks(u, layout)).collect();

    // Upload reference blocks to GPU.
    let (mut block_slices, block_ptrs) = upload_blocks(&stream, &ref_blocks)?;

    // Allocate universal output buffers on GPU.
    let (universal_slices, universal_ptrs) = alloc_buffers::<T>(&stream, nb, universal_volume)?;

    // --- Forward: blocks -> universal ---
    {
        let (bp, _g1) = block_ptrs.device_ptr(&stream);
        let (up, _g2) = universal_ptrs.device_ptr(&stream);
        let status = unsafe {
            universal_from_block(
                up as usize as *const *mut c_void,
                bp as usize as *const *const c_void,
                nb,
                nh,
                nl,
                no,
                nt,
                hd,
                T::DTYPE,
                layout,
                stream_raw,
            )
        };
        assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);
    }
    stream.synchronize()?;

    // Verify each universal buffer matches the original tensor.
    for (i, (slice, expected)) in universal_slices.iter().zip(universals.iter()).enumerate() {
        let host = stream.clone_dtoh(slice)?;
        let expected_flat: Vec<T> = expected.as_standard_layout().as_slice().unwrap().to_vec();
        assert_close::<T>(&host, &expected_flat, &format!("universal batch {i}"));
    }

    // --- Reverse: poison-fill blocks, then universal -> blocks ---
    poison_fill_blocks(&stream, &mut block_slices, nh * nt * hd)?;
    stream.synchronize()?;

    {
        let (bp, _g1) = block_ptrs.device_ptr(&stream);
        let (up, _g2) = universal_ptrs.device_ptr(&stream);
        let status = unsafe {
            block_from_universal(
                up as usize as *const *const c_void,
                bp as usize as *const *mut c_void,
                nb,
                nh,
                nl,
                no,
                nt,
                hd,
                T::DTYPE,
                layout,
                stream_raw,
            )
        };
        assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);
    }
    stream.synchronize()?;

    for (bi, (batch, ref_batch)) in block_slices.iter().zip(ref_blocks.iter()).enumerate() {
        for (ci, (slice, expected)) in batch.iter().zip(ref_batch.iter()).enumerate() {
            let host = stream.clone_dtoh(slice)?;
            assert_close::<T>(&host, expected, &format!("block batch {bi} chunk {ci}"));
        }
    }

    Ok(())
}

macro_rules! block_universal_test {
    ($name:ident, $ty:ty, $layout:expr) => {
        #[test]
        fn $name() -> Result<(), DriverError> {
            block_universal_roundtrip_inner::<$ty>($layout)
        }
    };
}

block_universal_test!(block_universal_roundtrip_nhd_f16, f16, BlockLayout::NHD);
block_universal_test!(block_universal_roundtrip_nhd_bf16, bf16, BlockLayout::NHD);
block_universal_test!(block_universal_roundtrip_nhd_f32, f32, BlockLayout::NHD);
block_universal_test!(block_universal_roundtrip_nhd_f64, f64, BlockLayout::NHD);
block_universal_test!(block_universal_roundtrip_hnd_f16, f16, BlockLayout::HND);
block_universal_test!(block_universal_roundtrip_hnd_bf16, bf16, BlockLayout::HND);
block_universal_test!(block_universal_roundtrip_hnd_f32, f32, BlockLayout::HND);
block_universal_test!(block_universal_roundtrip_hnd_f64, f64, BlockLayout::HND);

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// All kernel functions with num_blocks=0 should be a noop returning cudaSuccess.
#[test]
fn empty_batch_noop() -> Result<(), DriverError> {
    let (_stream, stream_raw) = match cuda_setup() {
        Some(s) => s,
        None => return Ok(()),
    };

    let null_mut = std::ptr::null() as *const *mut c_void;
    let null_const = std::ptr::null() as *const *const c_void;

    // universal_from_block
    let status = unsafe {
        universal_from_block(
            null_mut,
            null_const,
            0,
            1,
            1,
            1,
            1,
            1,
            TensorDataType::F32,
            BlockLayout::NHD,
            stream_raw,
        )
    };
    assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);

    // block_from_universal
    let status = unsafe {
        block_from_universal(
            null_const,
            null_mut,
            0,
            1,
            1,
            1,
            1,
            1,
            TensorDataType::F32,
            BlockLayout::NHD,
            stream_raw,
        )
    };
    assert_eq!(status, cuda_runtime::cudaError::cudaSuccess);

    Ok(())
}

// ---------------------------------------------------------------------------
// CPU-only validation of make_blocks reference implementation
// ---------------------------------------------------------------------------

/// Verify `make_blocks` for NHD layout against first-principles index arithmetic.
/// Uses deterministic position-encoded values so each element maps to a unique expected value.
#[test]
fn make_blocks_reference_nhd() {
    let nh = 3usize;
    let nl = 2usize;
    let no = 2usize;
    let nt = 4usize;
    let hd = 5usize;

    let universal =
        Array5::from_shape_fn((nh, nl, no, nt, hd), |(nh_i, nl_i, no_i, nt_i, hd_i)| {
            ((((nh_i * nl + nl_i) * no + no_i) * nt + nt_i) * hd + hd_i) as f32
        });

    let blocks = make_blocks(&universal, BlockLayout::NHD);
    assert_eq!(blocks.len(), nl * no);

    for nl_i in 0..nl {
        for no_i in 0..no {
            let block = &blocks[nl_i * no + no_i];
            assert_eq!(block.len(), nt * nh * hd);
            for nt_i in 0..nt {
                for nh_i in 0..nh {
                    for hd_i in 0..hd {
                        // NHD block offset: [nt, nh, hd]
                        let offset = (nt_i * nh + nh_i) * hd + hd_i;
                        let expected =
                            ((((nh_i * nl + nl_i) * no + no_i) * nt + nt_i) * hd + hd_i) as f32;
                        assert_eq!(
                            block[offset], expected,
                            "NHD mismatch at nl={nl_i} no={no_i} nt={nt_i} nh={nh_i} hd={hd_i}"
                        );
                    }
                }
            }
        }
    }
}

/// Verify `make_blocks` for HND layout against first-principles index arithmetic.
#[test]
fn make_blocks_reference_hnd() {
    let nh = 3usize;
    let nl = 2usize;
    let no = 2usize;
    let nt = 4usize;
    let hd = 5usize;

    let universal =
        Array5::from_shape_fn((nh, nl, no, nt, hd), |(nh_i, nl_i, no_i, nt_i, hd_i)| {
            ((((nh_i * nl + nl_i) * no + no_i) * nt + nt_i) * hd + hd_i) as f32
        });

    let blocks = make_blocks(&universal, BlockLayout::HND);
    assert_eq!(blocks.len(), nl * no);

    for nl_i in 0..nl {
        for no_i in 0..no {
            let block = &blocks[nl_i * no + no_i];
            assert_eq!(block.len(), nh * nt * hd);
            for nh_i in 0..nh {
                for nt_i in 0..nt {
                    for hd_i in 0..hd {
                        // HND block offset: [nh, nt, hd]
                        let offset = (nh_i * nt + nt_i) * hd + hd_i;
                        let expected =
                            ((((nh_i * nl + nl_i) * no + no_i) * nt + nt_i) * hd + hd_i) as f32;
                        assert_eq!(
                            block[offset], expected,
                            "HND mismatch at nl={nl_i} no={no_i} nh={nh_i} nt={nt_i} hd={hd_i}"
                        );
                    }
                }
            }
        }
    }
}
