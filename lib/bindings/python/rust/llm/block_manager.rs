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
use pyo3::PyResult;

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

#[pyclass]
#[derive(Clone)]
pub struct BlockManager {
    inner: Arc<dynamo_llm::block_manager::ReferenceBlockManager>,
}

#[pymethods]
impl BlockManager {
    #[new]
    #[pyo3(signature = (worker_id, num_layer, outer_dim, page_size, inner_dim, dtype=None, host_num_blocks=None, device_num_blocks=None, device_id=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        worker_id: u64,
        num_layer: usize,
        outer_dim: usize,
        page_size: usize,
        inner_dim: usize,
        dtype: Option<String>,
        host_num_blocks: Option<usize>,
        device_num_blocks: Option<usize>,
        device_id: usize,
    ) -> PyResult<Self> {
        let mut config = dynamo_llm::block_manager::KvBlockManagerConfig::builder().runtime(
            dynamo_llm::block_manager::KvManagerRuntimeConfig::builder()
                .worker_id(worker_id)
                .build()
                .map_err(to_pyerr)?,
        );
        let mut model_config = dynamo_llm::block_manager::KvManagerModelConfig::builder()
            .num_layers(num_layer)
            .outer_dim(outer_dim)
            .page_size(page_size)
            .inner_dim(inner_dim);
        let mut dtype_ = dynamo_llm::common::dtype::DType::FP16; // Default in block_manager config
        if let Some(dtype_str) = dtype {
            dtype_ = map_dtype(&dtype_str).map_err(to_pyerr)?;
        }
        model_config = model_config.dtype(dtype_);
        config = config.model(model_config.build().map_err(to_pyerr)?);
        if let Some(host_num_blocks) = host_num_blocks {
            config = config.host_layout(
                dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                    .num_blocks(host_num_blocks)
                    .allocator(
                        dynamo_llm::block_manager::storage::PinnedAllocator::new()
                            .map_err(to_pyerr)?,
                    )
                    .build()
                    .map_err(to_pyerr)?,
            );
        }
        if let Some(device_num_blocks) = device_num_blocks {
            config = config.device_layout(
                dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                    .num_blocks(device_num_blocks)
                    .allocator(
                        dynamo_llm::block_manager::storage::DeviceAllocator::new(device_id)
                            .map_err(to_pyerr)?,
                    )
                    .build()
                    .map_err(to_pyerr)?,
            );
        }
        let config = config.build().map_err(to_pyerr)?;
        let tokio_runtime = pyo3_async_runtimes::tokio::get_runtime();
        Ok(BlockManager {
            inner: Arc::from(
                tokio_runtime
                    .block_on(async {
                        dynamo_llm::block_manager::ReferenceBlockManager::new(config).await
                    })
                    .map_err(to_pyerr)?,
            ),
        })
    }

    fn block_size(&self) -> usize {
        self.inner.block_size()
    }
}

impl BlockManager {
    #[inline(always)]
    pub fn get_block_manager(
        &self,
    ) -> &dynamo_llm::block_manager::KvBlockManager<dynamo_llm::block_manager::BasicMetadata> {
        self.inner.as_ref()
    }
}
