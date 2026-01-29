// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Memory pool for efficient device memory allocation in hot paths.

pub mod cuda;

pub use cuda::{CudaMemPool, CudaMemPoolBuilder};
