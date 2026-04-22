// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use derive_getters::Dissolve;
use llm_rs::block_manager::distributed::{
    KvbmLeader as KvbmLeaderImpl, KvbmLeaderConfig, KvbmLeaderNumBlocksConfig,
};
use utils::{get_leader_zmq_ack_url, get_leader_zmq_pub_url};

use dynamo_runtime::config::environment_names::kvbm::cpu_cache as env_cpu_cache;
use dynamo_runtime::config::environment_names::kvbm::disk_cache as env_disk_cache;
use dynamo_runtime::config::environment_names::kvbm::leader as env_kvbm_leader;

const DEFAULT_INIT_TIMEOUT_SECS: u64 = 1800;

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
    drt: Option<Arc<rs::DistributedRuntime>>,
}

impl KvbmLeader {
    pub fn get_inner(&self) -> Arc<KvbmLeaderImpl> {
        self.leader.clone()
    }
}

#[pymethods]
impl KvbmLeader {
    #[new]
    #[pyo3(signature = (world_size, drt=None))]
    fn new(world_size: usize, drt: Option<PyObject>) -> PyResult<Self> {
        let drt: Option<Arc<rs::DistributedRuntime>> = Python::with_gil(|py| {
            if let Some(obj) = drt {
                extract_distributed_runtime_from_obj(py, obj)
            } else {
                Ok(None)
            }
        })?;

        let leader_init_timeout_sec: u64 =
            get_leader_init_timeout_secs(env_kvbm_leader::DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS);

        let config = KvbmLeaderConfig::builder()
            .world_size(world_size)
            .leader_init_timeout_secs(leader_init_timeout_sec)
            .host_blocks_config(get_blocks_config(
                env_cpu_cache::DYN_KVBM_CPU_CACHE_GB,
                env_cpu_cache::DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS,
            ))
            .disk_blocks_config(get_blocks_config(
                env_disk_cache::DYN_KVBM_DISK_CACHE_GB,
                env_disk_cache::DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS,
            ))
            .leader_pub_url(get_leader_zmq_pub_url())
            .leader_ack_url(get_leader_zmq_ack_url())
            .build()
            .map_err(to_pyerr)?;

        config.sanity_check().map_err(to_pyerr)?;

        let rt = get_current_tokio_handle();

        let leader =
            rt.block_on(async move { KvbmLeaderImpl::new(config).await.map_err(to_pyerr) })?;

        Ok(Self {
            leader: Arc::new(leader),
            drt,
        })
    }
}
