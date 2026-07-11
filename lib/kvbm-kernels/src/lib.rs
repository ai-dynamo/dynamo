// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod tensor_kernels;

#[cfg(feature = "xpu-sycl")]
pub mod tensor_kernels_sycl;

pub use tensor_kernels::{
    BlockLayout, MemcpyBatchMode, TensorDataType, block_from_universal, is_memcpy_batch_available,
    is_using_stubs, memcpy_batch, nhd_hnd_transpose, universal_from_block, vectorized_copy,
};

// SYCL core transfer - queue-based vectorized copy (requires xpu-sycl feature)
#[cfg(feature = "xpu-sycl")]
pub use tensor_kernels_sycl::sycl_vectorized_copy;

// SYCL permute kernels - queue-based layout transforms (requires xpu-sycl-permute feature)
#[cfg(feature = "xpu-sycl-permute")]
pub use tensor_kernels_sycl::{
    sycl_block_from_universal, sycl_nhd_hnd_transpose, sycl_universal_from_block,
};
