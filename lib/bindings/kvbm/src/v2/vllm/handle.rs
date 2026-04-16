// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opaque handle wrapping `Arc<BlockManager<G1>>` so it can be passed
//! from `PyConnectorLeader` to `PyRustKvCacheManager` without exposing
//! internal types to Python.

use std::sync::Arc;

use kvbm_engine::G1;
use kvbm_logical::manager::BlockManager;
use pyo3::prelude::*;

#[pyclass(name = "G1BlockManagerHandle")]
pub struct PyG1BlockManagerHandle {
    pub(crate) inner: Arc<BlockManager<G1>>,
}

impl PyG1BlockManagerHandle {
    pub fn new(inner: Arc<BlockManager<G1>>) -> Self {
        Self { inner }
    }
}
