// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KVTC (KV Cache Transform Coding) quantization kernels.
//!
//! Provides FP8 and IntX quantization/dequantization with a self-describing
//! compressed format. These are the building blocks for the full KVTC compression
//! pipeline (Phase 2 adds PCA projection via cuBLAS).

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::c_void;

use cudarc::runtime::sys::{cudaError_t, cudaStream_t};

/// Quantization type for a range of PCA components.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvtcQuantType {
    Fp8 = 0,
    IntX = 1,
}

/// Describes one range of PCA components and how to quantize it.
#[derive(Clone, Debug)]
pub struct KvtcTypeRange {
    pub start_idx: usize,
    pub end_idx: usize,
    pub quant_type: KvtcQuantType,
    pub int_bits: u32,
}

/// Header written per-range in the compressed output buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KvtcRangeHeader {
    pub quant_type: i32,
    pub int_bits: i32,
    pub start_idx: u64,
    pub end_idx: u64,
    pub packed_data_bytes: u64,
    pub metadata_bytes: u64,
}

// Direct FFI for cudaMemcpyAsync — bypasses cudarc's runtime::sys lazy loader
// which panics on CUDA 13.x due to removed cudaGetDeviceProperties_v2 symbol.
// We link against libcudart directly through build.rs, so this is always available.
unsafe extern "C" {
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

const CUDA_MEMCPY_DEFAULT: i32 = 4; // cudaMemcpyDefault

// cuBLAS direct FFI — bypasses cudarc's dlopen-based loader.
// We link against libcublas directly through build.rs.
pub type CublasHandle = *mut c_void;

unsafe extern "C" {
    fn cublasSgemm_v2(
        handle: CublasHandle,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: *const f32,
        c: *mut f32,
        ldc: i32,
    ) -> i32;

    fn cublasCreate_v2(handle: *mut CublasHandle) -> i32;
    fn cublasDestroy_v2(handle: CublasHandle) -> i32;
    fn cublasSetStream_v2(handle: CublasHandle, stream: cudaStream_t) -> i32;
}

const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const CUBLAS_STATUS_SUCCESS: i32 = 0;

/// Data type tag for gather/scatter kernels.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorDataType {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
    F64 = 3,
}

unsafe extern "C" {
    fn kvbm_kernels_kvtc_gather_mean_subtract(
        block_ptrs: *const *const c_void,
        mean: *const f32,
        output: *mut f32,
        num_blocks: usize,
        features: usize,
        block_stride: usize,
        input_dtype: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn kvbm_kernels_kvtc_mean_add_scatter(
        input: *const f32,
        mean: *const f32,
        block_ptrs: *const *mut c_void,
        num_blocks: usize,
        features: usize,
        block_stride: usize,
        output_dtype: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

unsafe extern "C" {
    fn kvbm_kernels_kvtc_quantize_fp8(
        input: *const f32,
        output: *mut u8,
        batch: usize,
        total_features: usize,
        start_feature: usize,
        range_features: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn kvbm_kernels_kvtc_dequantize_fp8(
        input: *const u8,
        output: *mut f32,
        batch: usize,
        total_features: usize,
        start_feature: usize,
        range_features: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn kvbm_kernels_kvtc_minmax_reduce(
        input: *const f32,
        min_vals: *mut f32,
        max_vals: *mut f32,
        batch: usize,
        total_features: usize,
        start_feature: usize,
        range_features: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn kvbm_kernels_kvtc_quantize_intx(
        input: *const f32,
        min_vals: *const f32,
        max_vals: *const f32,
        output: *mut u8,
        batch: usize,
        total_features: usize,
        start_feature: usize,
        range_features: usize,
        int_bits: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn kvbm_kernels_kvtc_dequantize_intx(
        input: *const u8,
        min_vals: *const f32,
        max_vals: *const f32,
        output: *mut f32,
        batch: usize,
        total_features: usize,
        start_feature: usize,
        range_features: usize,
        int_bits: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

pub unsafe fn quantize_fp8(
    input: *const f32,
    output: *mut u8,
    batch: usize,
    total_features: usize,
    start_feature: usize,
    range_features: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    unsafe {
        kvbm_kernels_kvtc_quantize_fp8(
            input,
            output,
            batch,
            total_features,
            start_feature,
            range_features,
            stream,
        )
    }
}

pub unsafe fn dequantize_fp8(
    input: *const u8,
    output: *mut f32,
    batch: usize,
    total_features: usize,
    start_feature: usize,
    range_features: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    unsafe {
        kvbm_kernels_kvtc_dequantize_fp8(
            input,
            output,
            batch,
            total_features,
            start_feature,
            range_features,
            stream,
        )
    }
}

pub unsafe fn minmax_reduce(
    input: *const f32,
    min_vals: *mut f32,
    max_vals: *mut f32,
    batch: usize,
    total_features: usize,
    start_feature: usize,
    range_features: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    unsafe {
        kvbm_kernels_kvtc_minmax_reduce(
            input,
            min_vals,
            max_vals,
            batch,
            total_features,
            start_feature,
            range_features,
            stream,
        )
    }
}

pub unsafe fn quantize_intx(
    input: *const f32,
    min_vals: *const f32,
    max_vals: *const f32,
    output: *mut u8,
    batch: usize,
    total_features: usize,
    start_feature: usize,
    range_features: usize,
    int_bits: i32,
    stream: cudaStream_t,
) -> cudaError_t {
    unsafe {
        kvbm_kernels_kvtc_quantize_intx(
            input,
            min_vals,
            max_vals,
            output,
            batch,
            total_features,
            start_feature,
            range_features,
            int_bits,
            stream,
        )
    }
}

pub unsafe fn dequantize_intx(
    input: *const u8,
    min_vals: *const f32,
    max_vals: *const f32,
    output: *mut f32,
    batch: usize,
    total_features: usize,
    start_feature: usize,
    range_features: usize,
    int_bits: i32,
    stream: cudaStream_t,
) -> cudaError_t {
    unsafe {
        kvbm_kernels_kvtc_dequantize_intx(
            input,
            min_vals,
            max_vals,
            output,
            batch,
            total_features,
            start_feature,
            range_features,
            int_bits,
            stream,
        )
    }
}

pub unsafe fn gather_mean_subtract(
    block_ptrs: *const *const c_void,
    mean: *const f32,
    output: *mut f32,
    num_blocks: usize,
    features: usize,
    block_stride: usize,
    input_dtype: TensorDataType,
    stream: cudaStream_t,
) -> cudaError_t {
    unsafe {
        kvbm_kernels_kvtc_gather_mean_subtract(
            block_ptrs,
            mean,
            output,
            num_blocks,
            features,
            block_stride,
            input_dtype as i32,
            stream,
        )
    }
}

pub unsafe fn mean_add_scatter(
    input: *const f32,
    mean: *const f32,
    block_ptrs: *const *mut c_void,
    num_blocks: usize,
    features: usize,
    block_stride: usize,
    output_dtype: TensorDataType,
    stream: cudaStream_t,
) -> cudaError_t {
    unsafe {
        kvbm_kernels_kvtc_mean_add_scatter(
            input,
            mean,
            block_ptrs,
            num_blocks,
            features,
            block_stride,
            output_dtype as i32,
            stream,
        )
    }
}

/// Compute packed byte size for a single range.
pub fn kvtc_range_packed_size(range: &KvtcTypeRange, batch_size: usize) -> usize {
    let range_features = range.end_idx - range.start_idx;
    let total_elements = batch_size * range_features;
    match range.quant_type {
        KvtcQuantType::Fp8 => total_elements, // 1 byte per element
        KvtcQuantType::IntX => {
            let slots_per_byte = 8 / range.int_bits as usize;
            total_elements.div_ceil(slots_per_byte)
        }
    }
}

/// Compute metadata size for a single range (min/max floats for IntX).
pub fn kvtc_range_metadata_size(range: &KvtcTypeRange, batch_size: usize) -> usize {
    match range.quant_type {
        KvtcQuantType::Fp8 => 0,
        KvtcQuantType::IntX => batch_size * 2 * std::mem::size_of::<f32>(), // min + max per row
    }
}

/// Total compressed buffer size for all ranges.
pub fn kvtc_compressed_size(ranges: &[KvtcTypeRange], batch_size: usize) -> usize {
    let header_size = std::mem::size_of::<KvtcRangeHeader>();
    ranges
        .iter()
        .map(|r| {
            header_size
                + kvtc_range_metadata_size(r, batch_size)
                + kvtc_range_packed_size(r, batch_size)
        })
        .sum()
}

/// Full compressor configuration.
///
/// All device pointers must remain valid for the lifetime of any
/// `kvtc_compress` / `kvtc_decompress` call using this config.
pub struct KvtcConfig {
    pub mean: *const f32,       // [features] device ptr
    pub projection: *const f32, // [features, pca_components] device ptr, row-major
    pub features: usize,
    pub pca_components: usize,
    pub ranges: Vec<KvtcTypeRange>,
}

impl KvtcConfig {
    /// Create a new KVTC configuration.
    ///
    /// # Arguments
    /// * `mean` - Device pointer to [features] float32 mean vector
    /// * `projection` - Device pointer to [features, pca_components] float32 projection matrix (row-major)
    /// * `features` - Input feature dimension
    /// * `pca_components` - PCA output dimension (must equal sum of range widths)
    /// * `ranges` - Quantization ranges covering [0, pca_components)
    pub fn new(
        mean: *const f32,
        projection: *const f32,
        features: usize,
        pca_components: usize,
        ranges: Vec<KvtcTypeRange>,
    ) -> Self {
        Self {
            mean,
            projection,
            features,
            pca_components,
            ranges,
        }
    }

    /// Compute compressed output size in bytes for a given batch size.
    pub fn compressed_size(&self, batch_size: usize) -> usize {
        kvtc_compressed_size(&self.ranges, batch_size)
    }

    /// Compute workspace size in bytes for a given batch size.
    pub fn workspace_size(&self, batch_size: usize) -> usize {
        kvtc_workspace_size(self.features, self.pca_components, batch_size)
    }
}

/// Workspace size in bytes for compress/decompress.
pub fn kvtc_workspace_size(features: usize, pca_components: usize, batch_size: usize) -> usize {
    // centered [batch, features] + pca_output [batch, pca_components], both f32
    (batch_size * features + batch_size * pca_components) * std::mem::size_of::<f32>()
}

/// Create a cuBLAS handle. Caller must destroy with kvtc_destroy_cublas_handle.
pub unsafe fn kvtc_create_cublas_handle() -> Result<CublasHandle, i32> {
    let mut handle: CublasHandle = std::ptr::null_mut();
    let status = cublasCreate_v2(&mut handle);
    if status != CUBLAS_STATUS_SUCCESS {
        return Err(status);
    }
    Ok(handle)
}

/// Destroy a cuBLAS handle.
pub unsafe fn kvtc_destroy_cublas_handle(handle: CublasHandle) {
    cublasDestroy_v2(handle);
}

/// Quantize ranges and write self-describing compressed output.
///
/// `input` is [batch, total_features] float32 on device.
/// `output` is the destination buffer (can be pinned host or device).
/// `minmax_workspace` is [2 * batch] float32 on device (for IntX min/max).
///
/// Returns total bytes written to output.
pub unsafe fn kvtc_quantize_ranges(
    input: *const f32,
    output: *mut u8,
    ranges: &[KvtcTypeRange],
    batch: usize,
    total_features: usize,
    minmax_workspace: *mut f32,
    stream: cudaStream_t,
) -> Result<usize, cudaError_t> {
    use cudarc::runtime::sys::cudaError::cudaSuccess;
    let header_size = std::mem::size_of::<KvtcRangeHeader>();
    let mut offset = 0usize;

    for range in ranges {
        let range_features = range.end_idx - range.start_idx;
        let packed_size = kvtc_range_packed_size(range, batch);
        let metadata_size = kvtc_range_metadata_size(range, batch);

        // Write header
        let header = KvtcRangeHeader {
            quant_type: range.quant_type as i32,
            int_bits: range.int_bits as i32,
            start_idx: range.start_idx as u64,
            end_idx: range.end_idx as u64,
            packed_data_bytes: packed_size as u64,
            metadata_bytes: metadata_size as u64,
        };
        std::ptr::copy_nonoverlapping(
            &header as *const KvtcRangeHeader as *const u8,
            output.add(offset),
            header_size,
        );
        offset += header_size;

        match range.quant_type {
            KvtcQuantType::Fp8 => {
                let err = quantize_fp8(
                    input,
                    output.add(offset),
                    batch,
                    total_features,
                    range.start_idx,
                    range_features,
                    stream,
                );
                if err != cudaSuccess {
                    return Err(err);
                }
                offset += packed_size;
            }
            KvtcQuantType::IntX => {
                // 1. Compute min/max
                let min_vals = minmax_workspace;
                let max_vals = minmax_workspace.add(batch);
                let err = minmax_reduce(
                    input,
                    min_vals,
                    max_vals,
                    batch,
                    total_features,
                    range.start_idx,
                    range_features,
                    stream,
                );
                if err != cudaSuccess {
                    return Err(err);
                }

                // 2. Write min/max metadata to output
                // min/max are on device, output may be pinned host — use cudaMemcpyAsync.
                {
                    let err = cudaMemcpyAsync(
                        output.add(offset) as *mut c_void,
                        min_vals as *const c_void,
                        batch * std::mem::size_of::<f32>(),
                        CUDA_MEMCPY_DEFAULT,
                        stream,
                    );
                    if err != cudaSuccess {
                        return Err(err);
                    }
                    let err = cudaMemcpyAsync(
                        output.add(offset + batch * std::mem::size_of::<f32>()) as *mut c_void,
                        max_vals as *const c_void,
                        batch * std::mem::size_of::<f32>(),
                        CUDA_MEMCPY_DEFAULT,
                        stream,
                    );
                    if err != cudaSuccess {
                        return Err(err);
                    }
                }
                offset += metadata_size;

                // 3. Quantize
                let err = quantize_intx(
                    input,
                    min_vals,
                    max_vals,
                    output.add(offset),
                    batch,
                    total_features,
                    range.start_idx,
                    range_features,
                    range.int_bits as i32,
                    stream,
                );
                if err != cudaSuccess {
                    return Err(err);
                }
                offset += packed_size;
            }
        }
    }

    Ok(offset)
}

/// Dequantize from self-describing compressed buffer back to float32.
///
/// `input` is the compressed buffer (can be pinned host or device).
/// `output` is [batch, total_features] float32 on device.
/// `minmax_workspace` is [2 * batch] float32 on device (for IntX min/max).
pub unsafe fn kvtc_dequantize_ranges(
    input: *const u8,
    output: *mut f32,
    num_ranges: usize,
    batch: usize,
    total_features: usize,
    minmax_workspace: *mut f32,
    stream: cudaStream_t,
) -> Result<(), cudaError_t> {
    use cudarc::runtime::sys::cudaError::cudaSuccess;
    let header_size = std::mem::size_of::<KvtcRangeHeader>();
    let mut offset = 0usize;

    for _ in 0..num_ranges {
        // Read header
        let mut header = std::mem::zeroed::<KvtcRangeHeader>();
        std::ptr::copy_nonoverlapping(
            input.add(offset),
            &mut header as *mut KvtcRangeHeader as *mut u8,
            header_size,
        );
        offset += header_size;

        let range_features = (header.end_idx - header.start_idx) as usize;
        let start_feature = header.start_idx as usize;

        if header.quant_type == KvtcQuantType::Fp8 as i32 {
            let err = dequantize_fp8(
                input.add(offset),
                output,
                batch,
                total_features,
                start_feature,
                range_features,
                stream,
            );
            if err != cudaSuccess {
                return Err(err);
            }
            offset += header.packed_data_bytes as usize;
        } else {
            // IntX: read min/max metadata, then dequantize
            let min_vals = minmax_workspace;
            let max_vals = minmax_workspace.add(batch);

            // Copy min/max from input to device workspace
            {
                let err = cudaMemcpyAsync(
                    min_vals as *mut c_void,
                    input.add(offset) as *const c_void,
                    batch * std::mem::size_of::<f32>(),
                    CUDA_MEMCPY_DEFAULT,
                    stream,
                );
                if err != cudaSuccess {
                    return Err(err);
                }
                let err = cudaMemcpyAsync(
                    max_vals as *mut c_void,
                    input.add(offset + batch * std::mem::size_of::<f32>()) as *const c_void,
                    batch * std::mem::size_of::<f32>(),
                    CUDA_MEMCPY_DEFAULT,
                    stream,
                );
                if err != cudaSuccess {
                    return Err(err);
                }
            }
            offset += header.metadata_bytes as usize;

            let err = dequantize_intx(
                input.add(offset),
                min_vals,
                max_vals,
                output,
                batch,
                total_features,
                start_feature,
                range_features,
                header.int_bits,
                stream,
            );
            if err != cudaSuccess {
                return Err(err);
            }
            offset += header.packed_data_bytes as usize;
        }
    }

    Ok(())
}

/// Compress KV cache blocks using KVTC pipeline.
///
/// 1. Gather from scattered block pointers + mean-subtract → workspace [batch, features]
/// 2. cuBLAS SGEMM: PCA projection → workspace [batch, pca_components]
/// 3. Quantize ranges → self-describing compressed output
///
/// Returns bytes written to output.
pub unsafe fn kvtc_compress(
    block_ptrs: *const *const c_void,
    config: &KvtcConfig,
    output: *mut u8,
    workspace: *mut f32,
    minmax_workspace: *mut f32,
    num_blocks: usize,
    block_stride: usize,
    input_dtype: TensorDataType,
    cublas_handle: CublasHandle,
    stream: cudaStream_t,
) -> Result<usize, cudaError_t> {
    use cudarc::runtime::sys::cudaError::cudaSuccess;

    if num_blocks == 0 {
        return Ok(0);
    }

    let features = config.features;
    let pca_components = config.pca_components;

    // Workspace layout: [centered | pca_output]
    let centered = workspace;
    let pca_output = workspace.add(num_blocks * features);

    // 1. Gather + mean-subtract
    let err = gather_mean_subtract(
        block_ptrs,
        config.mean,
        centered,
        num_blocks,
        features,
        block_stride,
        input_dtype,
        stream,
    );
    if err != cudaSuccess {
        return Err(err);
    }

    // 2. cuBLAS SGEMM: pca_output = centered @ projection
    // Row-major reinterpreted as col-major:
    //   projection [features, pca_comp] rm = [pca_comp, features] cm
    //   centered   [num_blocks, features] rm = [features, num_blocks] cm
    //   pca_output [num_blocks, pca_comp] rm = [pca_comp, num_blocks] cm
    // C = A * B: m=pca_comp, n=num_blocks, k=features
    {
        let status = cublasSetStream_v2(cublas_handle, stream);
        if status != CUBLAS_STATUS_SUCCESS {
            // Map cublas error to cudaErrorUnknown
            return Err(std::mem::transmute::<i32, cudaError_t>(999));
        }

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasSgemm_v2(
            cublas_handle,
            CUBLAS_OP_N,           // transa
            CUBLAS_OP_N,           // transb
            pca_components as i32, // m
            num_blocks as i32,     // n
            features as i32,       // k
            &alpha,
            config.projection,     // A [pca_comp, features] cm
            pca_components as i32, // lda
            centered,              // B [features, num_blocks] cm
            features as i32,       // ldb
            &beta,
            pca_output,            // C [pca_comp, num_blocks] cm
            pca_components as i32, // ldc
        );
        if status != CUBLAS_STATUS_SUCCESS {
            return Err(std::mem::transmute::<i32, cudaError_t>(999));
        }
    }

    // 3. Quantize ranges
    kvtc_quantize_ranges(
        pca_output,
        output,
        &config.ranges,
        num_blocks,
        pca_components,
        minmax_workspace,
        stream,
    )
}

/// Decompress KVTC compressed buffer back to KV cache blocks.
///
/// 1. Dequantize ranges → workspace [batch, pca_components]
/// 2. cuBLAS SGEMM: inverse PCA → workspace [batch, features]
/// 3. Mean-add + scatter to block pointers
pub unsafe fn kvtc_decompress(
    input: *const u8,
    config: &KvtcConfig,
    block_ptrs: *const *mut c_void,
    workspace: *mut f32,
    minmax_workspace: *mut f32,
    num_blocks: usize,
    block_stride: usize,
    output_dtype: TensorDataType,
    cublas_handle: CublasHandle,
    stream: cudaStream_t,
) -> Result<(), cudaError_t> {
    use cudarc::runtime::sys::cudaError::cudaSuccess;

    if num_blocks == 0 {
        return Ok(());
    }

    let features = config.features;
    let pca_components = config.pca_components;

    // Workspace layout: [pca_output | reconstructed]
    let pca_output = workspace;
    let reconstructed = workspace.add(num_blocks * pca_components);

    // 1. Dequantize ranges
    kvtc_dequantize_ranges(
        input,
        pca_output,
        config.ranges.len(),
        num_blocks,
        pca_components,
        minmax_workspace,
        stream,
    )?;

    // 2. cuBLAS SGEMM: reconstructed = pca_output @ projection^T
    // Row-major reinterpreted as col-major:
    //   projection [features, pca_comp] rm = [pca_comp, features] cm
    //   pca_output [num_blocks, pca_comp] rm = [pca_comp, num_blocks] cm
    //   reconstructed [num_blocks, features] rm = [features, num_blocks] cm
    // C = A^T * B: m=features, n=num_blocks, k=pca_comp
    {
        let status = cublasSetStream_v2(cublas_handle, stream);
        if status != CUBLAS_STATUS_SUCCESS {
            return Err(std::mem::transmute::<i32, cudaError_t>(999));
        }

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasSgemm_v2(
            cublas_handle,
            CUBLAS_OP_T,           // transa - transpose projection
            CUBLAS_OP_N,           // transb
            features as i32,       // m
            num_blocks as i32,     // n
            pca_components as i32, // k
            &alpha,
            config.projection, // A [pca_comp, features] cm, transposed to [features, pca_comp]
            pca_components as i32, // lda (leading dim of A before transpose)
            pca_output,        // B [pca_comp, num_blocks] cm
            pca_components as i32, // ldb
            &beta,
            reconstructed,   // C [features, num_blocks] cm
            features as i32, // ldc
        );
        if status != CUBLAS_STATUS_SUCCESS {
            return Err(std::mem::transmute::<i32, cudaError_t>(999));
        }
    }

    // 3. Mean-add + scatter
    let err = mean_add_scatter(
        reconstructed,
        config.mean,
        block_ptrs,
        num_blocks,
        features,
        block_stride,
        output_dtype,
        stream,
    );
    if err != cudaSuccess {
        return Err(err);
    }

    Ok(())
}
