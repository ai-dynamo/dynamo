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

#![cfg(feature = "block-manager")]
// Silence warnings about deprecated features (like pyo3::IntoPy::into_py)
#![allow(deprecated)]

use super::*;
use pyo3::PyResult;
use tokio;

mod block;
mod block_list;

/// Add bingings from this crate to the provided module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<block::Block>()?;
    m.add_class::<block_list::BlockList>()?;
    m.add_class::<BlockManager>()?;
    Ok(())
}

#[pyclass]
pub struct BlockManager {
    // TODO: Can this be implicitly created and referenced?
    tokio_runtime: tokio::runtime::Runtime,
    // Block manager
    inner: Arc<dynamo_llm::block_manager::ReferenceBlockManager>,
    // TODO: Metadata should be stored in the block manager?
    dtype: dynamo_llm::common::dtype::DType,
    device_id: usize,
}

#[pymethods]
impl BlockManager {
    #[new]
    #[pyo3(signature = (worker_id, num_layer, page_size, inner_dim, dtype=None, host_num_blocks=None, device_num_blocks=None, device_id=0))]
    fn new(
        worker_id: u64,
        num_layer: usize,
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
                .unwrap(),
        );
        let mut model_config = dynamo_llm::block_manager::KvManagerModelConfig::builder()
            .num_layers(num_layer)
            .page_size(page_size)
            .inner_dim(inner_dim);
        let mut dtype_ = dynamo_llm::common::dtype::DType::FP16; // Default in block_manager config
        if let Some(dtype_str) = dtype {
            dtype_ = match dtype_str.as_str() {
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
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unsupported dtype: {}",
                        dtype_str
                    )))
                }
            };
        }
        model_config = model_config.dtype(dtype_.clone());
        config = config.model(model_config.build().unwrap());
        if let Some(host_num_blocks) = host_num_blocks {
            config = config.host_layout(
                dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                    .num_blocks(host_num_blocks)
                    .allocator(dynamo_llm::block_manager::storage::PinnedAllocator::new().unwrap())
                    .build()
                    .unwrap(),
            );
        }
        if let Some(device_num_blocks) = device_num_blocks {
            config = config.device_layout(
                dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                    .num_blocks(device_num_blocks)
                    .allocator(
                        dynamo_llm::block_manager::storage::DeviceAllocator::new(device_id)
                            .unwrap(),
                    )
                    .build()
                    .unwrap(),
            );
        }
        let config = config.build().unwrap();
        let tokio_runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let block_manager = tokio_runtime.block_on(async {
            dynamo_llm::block_manager::ReferenceBlockManager::new(config).unwrap()
        });
        Ok(BlockManager {
            tokio_runtime: tokio_runtime,
            inner: Arc::from(block_manager),
            dtype: dtype_,
            device_id: device_id,
        })
    }

    fn allocate_host_blocks_blocking(&self, count: usize) -> PyResult<block_list::BlockList> {
        let blocks = self
            .inner
            .host()
            .unwrap()
            .allocate_blocks_blocking(count)
            .unwrap();
        // Wrap each block in an enum accounting for Pinned & Device block
        let blocks = blocks
            .into_iter()
            .map(|b| block::BlockType::Pinned(b))
            .collect();
        Ok(block_list::BlockList::from_rust(
            blocks,
            self.dtype.clone(),
            self.device_id,
        ))
    }

    fn allocate_device_blocks_blocking(&self, count: usize) -> PyResult<block_list::BlockList> {
        let blocks = self
            .inner
            .device()
            .unwrap()
            .allocate_blocks_blocking(count)
            .unwrap();
        // Wrap each block in an enum accounting for Pinned & Device block
        let blocks = blocks
            .into_iter()
            .map(|b| block::BlockType::Device(b))
            .collect();
        Ok(block_list::BlockList::from_rust(
            blocks,
            self.dtype.clone(),
            self.device_id,
        ))
    }
}
