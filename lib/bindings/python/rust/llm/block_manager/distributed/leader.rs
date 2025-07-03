// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use utils::get_barrier_id;

use derive_getters::Dissolve;
use llm_rs::block_manager::distributed::{KvbmLeader as KvbmLeaderImpl, KvbmLeaderConfig};

fn compute_num_blocks(env_var: &str, bytes_per_block: usize) -> usize {
    let cache_size_gb = std::env::var(env_var)
        .unwrap_or_default()
        .parse::<usize>()
        .unwrap_or(0);
    (cache_size_gb * 1_000_000_000) / bytes_per_block
}

#[pyclass]
#[derive(Clone, Dissolve)]
pub struct KvbmLeader {
    leader: Arc<KvbmLeaderImpl>,
    rt: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl KvbmLeader {
    #[new]
    #[pyo3(signature = (bytes_per_block, world_size))]
    fn new(bytes_per_block: usize, world_size: usize) -> PyResult<Self> {
        let num_host_blocks = compute_num_blocks("DYNAMO_KVBM_CPU_CACHE", bytes_per_block);
        let num_disk_blocks = compute_num_blocks("DYNAMO_KVBM_DISK_CACHE", bytes_per_block);

        let barrier_id = get_barrier_id();

        let config = KvbmLeaderConfig::builder()
            .barrier_id(barrier_id)
            .num_host_blocks(num_host_blocks)
            .num_disk_blocks(num_disk_blocks)
            .world_size(world_size)
            .build()
            .map_err(to_pyerr)?;

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(to_pyerr)?;

        let leader =
            rt.block_on(async move { KvbmLeaderImpl::new(config).await.map_err(to_pyerr) })?;

        Ok(Self {
            leader: Arc::new(leader),
            rt: Arc::new(rt),
        })
    }
}
