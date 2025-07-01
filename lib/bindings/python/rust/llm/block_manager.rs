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

// mod block;
// mod block_list;
// mod dlpack;
// mod layer;

mod distributed;

pub mod vllm;

/// Add bingings from this crate to the provided module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_class::<layer::Layer>()?;
    // m.add_class::<block::Block>()?;
    // m.add_class::<block_list::BlockList>()?;
    m.add_class::<BlockManager>()?;
    m.add_class::<distributed::KvbmWorker>()?;
    m.add_class::<distributed::KvbmLeader>()?;

    vllm::add_to_module(m)?;

    Ok(())
}

pub fn map_dtype(dtype: &str) -> anyhow::Result<dynamo_llm::common::dtype::DType> {
    Ok(match dtype {
        "fp8" | "FP8" => dynamo_llm::common::dtype::DType::FP8,
        "fp16" | "FP16" => dynamo_llm::common::dtype::DType::FP16,
        "bf16" | "BF16" => dynamo_llm::common::dtype::DType::BF16,
        "fp32" | "FP32" => dynamo_llm::common::dtype::DType::FP32,
        "u8" | "U8" => dynamo_llm::common::dtype::DType::U8,
        "u16" | "U16" => dynamo_llm::common::dtype::DType::U16,
        "u32" | "U32" => dynamo_llm::common::dtype::DType::U32,
        "u64" | "U64" => dynamo_llm::common::dtype::DType::U64,
        "i8" | "I8" => dynamo_llm::common::dtype::DType::I8,
        "i16" | "I16" => dynamo_llm::common::dtype::DType::I16,
        "i32" | "I32" => dynamo_llm::common::dtype::DType::I32,
        "i64" | "I64" => dynamo_llm::common::dtype::DType::I64,
        _ => return Err(anyhow::anyhow!("Unsupported dtype: {}", dtype)),
    })
}

type VllmBlockManager = dynamo_llm::block_manager::KvBlockManager<
    Logical<DistributedLeaderWorkerResources>,
    BasicMetadata,
>;

#[pyclass]
#[derive(Clone)]
pub struct BlockManager {
    inner: Arc<VllmBlockManager>,
    _rt: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl BlockManager {
    #[new]
    #[pyo3(signature = (worker_id, leader, page_size, device_num_blocks))]
    fn new(
        worker_id: u64,
        leader: distributed::KvbmLeader,
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

        let config = config.build().map_err(to_pyerr)?;
        Ok(BlockManager {
            inner: Arc::from(
                rt.block_on(async {
                    let resources =
                        DistributedLeaderWorkerResources::new(leader, cancel_token.child_token())?;

                    dynamo_llm::block_manager::KvBlockManager::<
                        Logical<DistributedLeaderWorkerResources>,
                        BasicMetadata,
                    >::new(config, resources)
                    .await
                })
                .map_err(to_pyerr)?,
            ),
            _rt: rt,
        })
    }

    fn block_size(&self) -> usize {
        self.inner.block_size()
    }
}

impl BlockManager {
    #[inline(always)]
    pub fn get_block_manager(&self) -> &VllmBlockManager {
        self.inner.as_ref()
    }
}
