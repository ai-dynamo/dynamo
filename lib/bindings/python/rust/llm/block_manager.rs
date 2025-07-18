// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;
use dynamo_llm::block_manager::block::{
    data::logical::distributed_leader_worker::DistributedLeaderWorkerResources, locality::Logical,
};
use dynamo_llm::block_manager::controller::client::ControlClient;
use dynamo_llm::block_manager::controller::{CacheLevel, Controller};
use dynamo_llm::block_manager::{BasicMetadata, BlockParallelismStrategy};
use pyo3::PyResult;
use tokio_util::sync::CancellationToken;

mod distributed;

pub mod vllm;

/// Add bingings from this crate to the provided module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BlockManager>()?;
    m.add_class::<distributed::KvbmWorker>()?;
    m.add_class::<distributed::KvbmLeader>()?;
    m.add_class::<BlockManagerClient>()?;
    m.add_class::<PoolStatus>()?;
    m.add_class::<ResetBlocksResponse>()?;

    vllm::add_to_module(m)?;

    Ok(())
}

type VllmBlockManager = dynamo_llm::block_manager::KvBlockManager<
    Logical<DistributedLeaderWorkerResources>,
    BasicMetadata,
>;

type VllmController = Arc<
    dynamo_llm::block_manager::controller::Controller<
        Logical<DistributedLeaderWorkerResources>,
        BasicMetadata,
    >,
>;

#[pyclass]
#[derive(Clone)]
pub struct BlockManager {
    inner: VllmBlockManager,
    _rt: Arc<tokio::runtime::Runtime>,
    _controller: Option<VllmController>,
}

#[pymethods]
impl BlockManager {
    #[new]
    #[pyo3(signature = (worker_id, leader = None, page_size = 32, device_num_blocks = 16))]
    fn new(
        worker_id: u64,
        leader: Option<distributed::KvbmLeader>,
        page_size: usize,
        device_num_blocks: usize,
    ) -> PyResult<Self> {
        let cancel_token = CancellationToken::new();
        let mut config = dynamo_llm::block_manager::KvBlockManagerConfig::builder().runtime(
            dynamo_llm::block_manager::KvManagerRuntimeConfig::builder()
                .worker_id(worker_id)
                .cancellation_token(cancel_token.clone())
                .build()
                .map_err(to_pyerr)?,
        );

        tracing::info!("Using {} device blocks", device_num_blocks);

        let model_config = dynamo_llm::block_manager::KvManagerModelConfig::builder()
            .num_layers(1)
            .outer_dim(1)
            .page_size(page_size)
            .inner_dim(1);

        config = config.model(model_config.build().map_err(to_pyerr)?);

        config = config.device_layout(
            dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                .num_blocks(device_num_blocks)
                .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                .build()
                .map_err(to_pyerr)?,
        );

        let (leader, rt) = if let Some(leader) = leader {
            let (leader, rt) = leader.dissolve();
            if leader.num_host_blocks() > 0 {
                tracing::info!("Using {} host blocks", leader.num_host_blocks());
                config = config.host_layout(
                    dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                        .num_blocks(leader.num_host_blocks())
                        .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                        .build()
                        .map_err(to_pyerr)?,
                );
            }

            if leader.num_disk_blocks() > 0 {
                tracing::info!("Using {} disk blocks", leader.num_disk_blocks());
                config = config.disk_layout(
                    dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                        .num_blocks(leader.num_disk_blocks())
                        .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                        .build()
                        .map_err(to_pyerr)?,
                );
            }
            (Some(leader), rt)
        } else {
            tracing::info!("Leader not provided. Block transfer functionality will be disabled.");
            (
                None,
                Arc::new(
                    tokio::runtime::Builder::new_multi_thread()
                        .enable_all()
                        .build()
                        .map_err(to_pyerr)?,
                ),
            )
        };

        let config = config.build().map_err(to_pyerr)?;
        Ok(BlockManager {
            inner: rt
                .block_on(async {
                    let resources =
                        DistributedLeaderWorkerResources::new(leader, cancel_token.child_token())?;

                    dynamo_llm::block_manager::KvBlockManager::<
                        Logical<DistributedLeaderWorkerResources>,
                        BasicMetadata,
                    >::new(config, resources)
                    .await
                })
                .map_err(to_pyerr)?,
            _rt: rt,
            _controller: None,
        })
    }

    fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    fn init_controller(&mut self, component: Component) -> PyResult<()> {
        if self._controller.is_some() {
            tracing::warn!("Controller already initialized. Ignoring init_controller call.");
            return Ok(());
        }

        let block_manager = self.inner.clone();
        let controller = self
            ._rt
            .block_on(Controller::new(block_manager, component.inner.clone()))
            .map_err(to_pyerr)?;

        self._controller = Some(Arc::new(controller));

        let instance_id = component
            .inner
            .drt()
            .primary_lease()
            .map(|lease| lease.id())
            .ok_or_else(|| to_pyerr(anyhow::anyhow!("no instance id")))?;

        tracing::info!(
            "Dynamo KVBM Controller: {}.{}:{}",
            component.inner.namespace().name(),
            component.inner.name(),
            instance_id
        );

        Ok(())
    }
}

impl BlockManager {
    #[inline(always)]
    pub fn get_block_manager(&self) -> &VllmBlockManager {
        &self.inner
    }
}

#[pyclass]
pub struct BlockManagerClient {
    inner: ControlClient,
}

#[pymethods]
impl BlockManagerClient {
    // #[staticmethod]
    // fn new<'p>(
    //     py: Python<'p>,
    //     component: Component,
    //     instance_id: i64,
    // ) -> PyResult<Bound<'p, PyAny>> {
    //     pyo3_async_runtimes::tokio::future_into_py(py, async move {
    //         let client = ControlClient::new(component.inner, instance_id)
    //             .await
    //             .map_err(to_pyerr)?;
    //         Ok(BlockManagerClient { inner: client })
    //     })
    // }

    #[new]
    fn new(component: Component, instance_id: i64) -> PyResult<Self> {
        let client = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(ControlClient::new(component.inner, instance_id))
            .map_err(to_pyerr)?;
        Ok(BlockManagerClient { inner: client })
    }

    fn reset_pool(&self, cache_level: String) -> PyResult<()> {
        let cache_level = Self::cache_level_from_str(&cache_level).map_err(to_pyerr)?;
        pyo3_async_runtimes::tokio::get_runtime()
            .block_on(self.inner.reset_pool(cache_level))
            .map_err(to_pyerr)
    }

    fn reset_blocks(&self, cache_level: String, blocks: Vec<u64>) -> PyResult<ResetBlocksResponse> {
        let cache_level = Self::cache_level_from_str(&cache_level).map_err(to_pyerr)?;
        let response = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(self.inner.reset_blocks(cache_level, blocks))
            .map_err(to_pyerr)?;
        Ok(ResetBlocksResponse { inner: response })
    }

    fn status(&self, cache_level: String) -> PyResult<PoolStatus> {
        let cache_level = Self::cache_level_from_str(&cache_level).map_err(to_pyerr)?;
        let status = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(self.inner.status(cache_level))
            .map_err(to_pyerr)?;
        Ok(PoolStatus { inner: status })
    }

    fn reset_all_pools(&self) -> PyResult<()> {
        pyo3_async_runtimes::tokio::get_runtime()
            .block_on(self.inner.reset_all_pools())
            .map_err(to_pyerr)
    }
}

impl BlockManagerClient {
    // convert string to cache level
    fn cache_level_from_str(cache_level: &str) -> anyhow::Result<CacheLevel> {
        match cache_level.to_uppercase().as_str() {
            "G1" => Ok(CacheLevel::G1),
            "G2" => Ok(CacheLevel::G2),
            "G3" => Ok(CacheLevel::G3),
            _ => anyhow::bail!("Invalid cache level: allowed values are G1, G2, G3"),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PoolStatus {
    inner: dynamo_llm::block_manager::pool::PoolStatus,
}

#[pymethods]
impl PoolStatus {
    fn active_blocks(&self) -> Vec<u64> {
        self.inner.active_blocks.clone()
    }

    fn inactive_blocks(&self) -> Vec<u64> {
        self.inner.inactive_blocks.clone()
    }

    fn empty_blocks(&self) -> usize {
        self.inner.empty_blocks
    }
}

#[pyclass]
pub struct ResetBlocksResponse {
    inner: dynamo_llm::block_manager::pool::ResetBlocksResponse,
}

#[pymethods]
impl ResetBlocksResponse {
    fn reset_blocks(&self) -> Vec<u64> {
        self.inner.reset_blocks.clone()
    }

    fn not_found_blocks(&self) -> Vec<u64> {
        self.inner.not_found.clone()
    }

    fn not_reset_blocks(&self) -> Vec<u64> {
        self.inner.not_reset.clone()
    }
}
