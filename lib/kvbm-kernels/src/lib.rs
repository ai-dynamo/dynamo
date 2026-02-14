// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod tensor_kernels;

// Always available - core transfer functionality
pub use tensor_kernels::{
    MemcpyBatchMode, is_memcpy_batch_available, is_using_stubs, memcpy_batch,
    memcpy_batch_diagnostic, vectorized_copy,
};

// Permute kernels - data layout transformation (requires permute_kernels feature)
#[cfg(feature = "permute_kernels")]
pub use tensor_kernels::{
    BlockLayout, OperationalCopyBackend, OperationalCopyDirection, TensorDataType,
    block_from_universal, operational_copy, universal_from_block,
};
