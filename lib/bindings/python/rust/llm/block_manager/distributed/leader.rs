// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use utils::get_barrier_id_prefix;

use derive_getters::Dissolve;
use llm_rs::block_manager::distributed::{
    KvbmLeader as KvbmLeaderImpl, KvbmLeaderConfig, KvbmLeaderNumBlocksConfig,
};

const CPU_CACHE: &str = "DYN_KVBM_CPU_CACHE_GB";
const CPU_CACHE_OVERRIDE: &str = "DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS";

const DISK_CACHE: &str = "DYN_KVBM_DISK_CACHE_GB";
const DISK_CACHE_OVERRIDE: &str = "DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS";

const LEADER_WORKER_INIT_TIMEOUT_SECS: &str = "DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS";
const DEFAULT_INIT_TIMEOUT_SECS: u64 = 120;

fn read_env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.trim().parse::<usize>().ok()
}

fn read_cache_size_float(key: &str) -> f64 {
    std::env::var(key)
        .unwrap_or_default()
        .parse::<f64>()
        .unwrap_or(0.0)
}

fn get_blocks_config(cache_size_key: &str, override_key: &str) -> KvbmLeaderNumBlocksConfig {
    if let Some(nblocks) = read_env_usize(override_key) {
        // Optional: still read cache size for observability, but override takes precedence.
        let cache_gb: f64 = read_cache_size_float(cache_size_key);
        return KvbmLeaderNumBlocksConfig {
            cache_size_in_gb: cache_gb,
            num_blocks_overriden: nblocks,
        };
    }

    // No override -> compute from cache size (in GB)
    let cache_gb: f64 = read_cache_size_float(cache_size_key);
    KvbmLeaderNumBlocksConfig {
        cache_size_in_gb: cache_gb,
        num_blocks_overriden: 0,
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
    #[pyo3(signature = (world_size, drt))]
    fn new(world_size: usize, drt: DistributedRuntime) -> PyResult<Self> {
        let barrier_id_prefix = get_barrier_id_prefix();
        let leader_init_timeout_sec: u64 =
            get_leader_init_timeout_secs(LEADER_WORKER_INIT_TIMEOUT_SECS);

        let config = KvbmLeaderConfig::builder()
            .barrier_id_prefix(barrier_id_prefix)
            .world_size(world_size)
            .leader_init_timeout_secs(leader_init_timeout_sec)
            .drt(drt.inner().clone())
            .host_blocks_config(get_blocks_config(CPU_CACHE, CPU_CACHE_OVERRIDE))
            .disk_blocks_config(get_blocks_config(DISK_CACHE, DISK_CACHE_OVERRIDE))
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
