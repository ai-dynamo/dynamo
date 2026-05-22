// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod tensor_kernels;

#[cfg(feature = "xpu-sycl")]
pub mod tensor_kernels_sycl;


// Always available - core transfer functionality
pub use tensor_kernels::{
    BlockLayout, MemcpyBatchMode, is_memcpy_batch_available, is_using_stubs, memcpy_batch,
    vectorized_copy,
};

// Permute kernels - data layout transformation (requires permute_kernels feature)
#[cfg(feature = "permute_kernels")]
pub use tensor_kernels::{TensorDataType, block_from_universal, universal_from_block};

// SYCL core transfer - queue-based vectorized copy (requires xpu-sycl feature)
#[cfg(feature = "xpu-sycl")]
pub use tensor_kernels_sycl::sycl_vectorized_copy;

// SYCL permute kernels - queue-based layout transforms (requires xpu-sycl-permute feature)
#[cfg(feature = "xpu-sycl-permute")]
pub use tensor_kernels_sycl::{sycl_block_from_universal, sycl_universal_from_block};
