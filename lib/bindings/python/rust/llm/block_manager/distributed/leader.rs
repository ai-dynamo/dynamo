// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use llm_rs::block_manager::distributed::KvbmLeader as KvbmLeaderImpl;


#[pyclass]
pub struct KvbmLeader {
    _impl: Arc<KvbmLeaderImpl>,
}

#[pymethods]
impl KvbmLeader {
    #[new]
    #[pyo3(signature = (barrier_id, world_size=1))]
    fn new(barrier_id: String, world_size: usize) -> PyResult<Self> {

        let leader = KvbmLeaderImpl::new(barrier_id, world_size).map_err(to_pyerr)?;

        Ok(Self {
            _impl: Arc::new(leader),
        })
    }
}
