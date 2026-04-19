// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod tensor_kernels;

// Always available - core transfer functionality
pub use tensor_kernels::{
    MemcpyBatchMode, is_memcpy_batch_available, is_using_stubs, memcpy_batch, vectorized_copy,
};

// Permute kernels - data layout transformation (requires permute_kernels feature)
#[cfg(feature = "permute_kernels")]
pub use tensor_kernels::{BlockLayout, TensorDataType, block_from_universal, universal_from_block};

// KVTC kernels - KV cache compression via PCA + quantization
#[cfg(feature = "kvtc_kernels")]
pub mod kvtc_kernels;

#[cfg(feature = "kvtc_kernels")]
pub use kvtc_kernels::{
    CublasHandle, KvtcConfig, KvtcQuantType, KvtcRangeHeader, KvtcTypeRange, kvtc_compress,
    kvtc_compressed_size, kvtc_create_cublas_handle, kvtc_decompress, kvtc_destroy_cublas_handle,
    kvtc_workspace_size,
};

// Re-export TensorDataType from kvtc_kernels when permute_kernels isn't enabled
// (permute_kernels already exports it from tensor_kernels)
#[cfg(all(feature = "kvtc_kernels", not(feature = "permute_kernels")))]
pub use kvtc_kernels::TensorDataType;
