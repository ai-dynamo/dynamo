// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V vectorized_copy kernel for Level Zero / Intel XPU.
//!
//! Mirrors the CUDA `kvbm_kernels_vectorized_copy_kernel` from
//! `cuda/tensor_kernels.cu`.  The OpenCL C source lives in
//! `opencl/vectorized_copy.cl`.
//!
//! # Usage (with syclrc)
//!
//! ```rust,ignore
//! use kvbm_kernels::ze_vectorized_copy as vc;
//! use syclrc::level_zero::ze::safe::{ZeModule, ZeKernel};
//!
//! let module = ZeModule::from_spirv(&dev, vc::SPIRV, None)?;
//! let module = std::sync::Arc::new(module);
//! let kernel = ZeKernel::new(&module, vc::KERNEL_NAME)?;
//! kernel.set_group_size(vc::WORK_GROUP_SIZE, 1, 1)?;
//! ```

/// Pre-compiled SPIR-V binary of the `vectorized_copy` kernel.
///
/// Generated from `opencl/vectorized_copy.cl` via:
/// ```text
/// ocloc compile -file vectorized_copy.cl -device pvc \
///               -out_dir . -options "-cl-std=CL2.0"
/// ```
pub const SPIRV: &[u8] = include_bytes!("../opencl/vectorized_copy.spv");

/// Entry-point name of the kernel inside the SPIR-V module.
pub const KERNEL_NAME: &std::ffi::CStr = c"vectorized_copy";

/// Recommended work-group (thread-block) size — matches the CUDA launch config.
pub const WORK_GROUP_SIZE: u32 = 128;

/// Maximum number of work-groups to dispatch.
///
/// When `num_pairs > MAX_GROUPS`, the kernel grid-strides over the remaining
/// pairs (identical to the CUDA `min(num_pairs, 65535)` launch pattern).
pub const MAX_GROUPS: u32 = 65535;
