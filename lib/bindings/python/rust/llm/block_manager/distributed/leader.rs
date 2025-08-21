// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use utils::get_barrier_id;

use derive_getters::Dissolve;
use llm_rs::block_manager::distributed::{KvbmLeader as KvbmLeaderImpl, KvbmLeaderConfig};

const CPU_CACHE: &str = "DYN_KVBM_CPU_CACHE_GB";
const CPU_CACHE_OVERRIDE: &str = "DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS";

const DISK_CACHE: &str = "DYN_KVBM_DISK_CACHE_GB";
const DISK_CACHE_OVERRIDE: &str = "DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS";

const LEADER_WORKER_INIT_TIMEOUT_SECS: &str = "DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS";
const DEFAULT_INIT_TIMEOUT_SECS: u64 = 120;

fn compute_num_blocks(cache_size_key: &str, override_key: &str, bytes_per_block: usize) -> usize {
    if let Ok(override_num_blocks) = std::env::var(override_key) {
        override_num_blocks.parse::<usize>().unwrap_or(0)
    } else {
        let cache_size_gb = std::env::var(cache_size_key)
            .unwrap_or_default()
            .parse::<f64>()
            .unwrap_or(0.0);
        ((cache_size_gb * 1_000_000_000.0) / bytes_per_block as f64) as usize
    }
}

fn get_leader_init_timeout_secs(override_key: &str) -> u64 {
    std::env::var(override_key)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_INIT_TIMEOUT_SECS)
}

#[pyclass]
#[derive(Clone, Dissolve)]
pub struct KvbmLeader {
    leader: Arc<KvbmLeaderImpl>,
    drt: DistributedRuntime,
}

impl KvbmLeader {
    pub fn get_inner(&self) -> Arc<KvbmLeaderImpl> {
        self.leader.clone()
    }
}

#[pymethods]
impl KvbmLeader {
    #[new]
    #[pyo3(signature = (bytes_per_block, world_size, drt))]
    fn new(bytes_per_block: usize, world_size: usize, drt: DistributedRuntime) -> PyResult<Self> {
        let num_host_blocks = compute_num_blocks(CPU_CACHE, CPU_CACHE_OVERRIDE, bytes_per_block);
        let num_disk_blocks = compute_num_blocks(DISK_CACHE, DISK_CACHE_OVERRIDE, bytes_per_block);

        let barrier_id = get_barrier_id();
        let leader_init_timeout_sec: u64 =
            get_leader_init_timeout_secs(LEADER_WORKER_INIT_TIMEOUT_SECS);

        let config = KvbmLeaderConfig::builder()
            .barrier_id(barrier_id)
            .num_host_blocks(num_host_blocks)
            .num_disk_blocks(num_disk_blocks)
            .world_size(world_size)
            .leader_init_timeout_secs(leader_init_timeout_sec)
            .drt(drt.inner().clone())
            .build()
            .map_err(to_pyerr)?;

        let rt = drt.inner().runtime().primary();

        let leader =
            rt.block_on(async move { KvbmLeaderImpl::new(config).await.map_err(to_pyerr) })?;

        Ok(Self {
            leader: Arc::new(leader),
            drt,
        })
    }
}
