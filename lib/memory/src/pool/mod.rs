// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Memory pool for efficient device memory allocation in hot paths.

pub mod cuda;

#[cfg(feature = "level-zero")]
pub mod ze;

pub use cuda::{CudaMemPool, CudaMemPoolBuilder};

#[cfg(feature = "level-zero")]
pub use ze::{ZeMemPool, ZeMemPoolBuilder};

#[cfg(feature = "sycl")]
pub mod sycl;

#[cfg(feature = "sycl")]
pub use sycl::{SyclMemPool, SyclMemPoolBuilder};
