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
use dynamo_llm::block_manager::{BasicMetadata, BlockParallelismStrategy};
use pyo3::PyResult;
use tokio_util::sync::CancellationToken;

mod controller;
mod distributed;

pub mod vllm;

/// Add bingings from this crate to the provided module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BlockManager>()?;
    m.add_class::<distributed::KvbmWorker>()?;
    m.add_class::<distributed::KvbmLeader>()?;
    m.add_class::<controller::BlockManagerClient>()?;
    m.add_class::<controller::BlockPoolStatus>()?;
    m.add_class::<controller::ResetBlocksResponse>()?;

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
    drt: DistributedRuntime,
    _controller: Option<VllmController>,
}

// TODO: This is in desperate need of a massive refactor. We bind and instantiate this in Python, but we never actually use it.
#[pymethods]
#[allow(unused_variables)]
impl BlockManager {
    #[new]
    #[pyo3(signature = (worker_id, leader = None, page_size = 32, num_device_blocks = None, disable_device_pool = false))]
    fn new(
        worker_id: u64,
        leader: Option<distributed::KvbmLeader>,
        page_size: usize,
        num_device_blocks: Option<usize>,
        disable_device_pool: bool,
    ) -> PyResult<Self> {
        let cancel_token = CancellationToken::new();
        let mut config = dynamo_llm::block_manager::KvBlockManagerConfig::builder().runtime(
            dynamo_llm::block_manager::KvManagerRuntimeConfig::builder()
                .worker_id(worker_id)
                .cancellation_token(cancel_token.clone())
                .build()
                .map_err(to_pyerr)?,
        );

        let model_config = dynamo_llm::block_manager::KvManagerModelConfig::builder()
            .num_layers(1)
            .outer_dim(1)
            .page_size(page_size)
            .inner_dim(1);

        config = config.model(model_config.build().map_err(to_pyerr)?);

        let (leader, drt) = if let Some(leader) = leader {
            let (leader, rt) = leader.dissolve();

            if !disable_device_pool {
                config = config.device_layout(
                    dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                        .num_blocks(leader.num_device_blocks())
                        .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                        .build()
                        .map_err(to_pyerr)?,
                );
            }

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

            // let num_device_blocks = num_device_blocks
            //     .expect("num_device_blocks must be provided if leader is not provided");

            // config = config.device_layout(
            //     dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
            //         .num_blocks(num_device_blocks)
            //         .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
            //         .build()
            //         .map_err(to_pyerr)?,
            // );

            unimplemented!("Leader not provided");
            // (
            //     None,
            //     Arc::new(
            //         tokio::runtime::Builder::new_multi_thread()
            //             .enable_all()
            //             .build()
            //             .map_err(to_pyerr)?,
            //     ),
            // )
        };

        let rt = drt.inner().runtime().primary();

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
            drt,
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
            .drt
            .inner()
            .runtime()
            .primary()
            .block_on(controller::Controller::new(
                block_manager,
                component.inner.clone(),
            ))
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
