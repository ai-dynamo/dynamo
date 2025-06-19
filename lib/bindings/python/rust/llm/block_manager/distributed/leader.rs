// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use llm_rs::block_manager::distributed::{KvbmLeader as KvbmLeaderImpl, KvbmLeaderConfig};

#[pyclass]
pub struct KvbmLeader {
    _impl: Arc<KvbmLeaderImpl>,
}

#[pymethods]
impl KvbmLeader {
    #[new]
    #[pyo3(signature = (barrier_id, bytes_per_block, world_size))]
    fn new(barrier_id: String, bytes_per_block: usize, world_size: usize) -> PyResult<Self> {
        let config = KvbmLeaderConfig::builder()
            .barrier_id(barrier_id)
            .bytes_per_block(bytes_per_block)
            .world_size(world_size)
            .build()
            .map_err(to_pyerr)?;

        let leader = KvbmLeaderImpl::new(config).map_err(to_pyerr)?;

        Ok(Self {
            _impl: Arc::new(leader),
        })
    }
}
