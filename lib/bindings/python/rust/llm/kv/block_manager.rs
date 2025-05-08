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

// Silence warnings about deprecated features (like pyo3::IntoPy::into_py)
#![allow(deprecated)]

use super::*;
use pyo3::PyResult;

#[pyclass]
pub struct BlockManager {
    inner: Arc<dynamo_llm::block_manager::ReferenceBlockManager>,
}

#[pymethods]
impl BlockManager {
    #[new]
    #[pyo3(signature = (worker_id, num_layer, page_size, inner_dim, host_num_blocks, device_num_blocks))]
    fn new(
        worker_id: u64,
        num_layer: usize,
        page_size: usize,
        inner_dim: usize,
        host_num_blocks: usize,
        device_num_blocks: usize,
    ) -> PyResult<Self> {
        let config = dynamo_llm::block_manager::KvBlockManagerConfig::builder()
            .runtime(
                dynamo_llm::block_manager::KvManagerRuntimeConfig::builder()
                    .worker_id(worker_id)
                    .build()
                    .unwrap(),
            )
            .model(
                dynamo_llm::block_manager::KvManagerModelConfig::builder()
                    .num_layers(num_layer)
                    .page_size(page_size)
                    .inner_dim(inner_dim)
                    .build()
                    .unwrap(),
            )
            .host_layout(
                dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                    .num_blocks(host_num_blocks)
                    .allocator(dynamo_llm::block_manager::storage::PinnedAllocator::default())
                    .build()
                    .unwrap(),
            )
            .device_layout(
                dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                    .num_blocks(device_num_blocks)
                    .allocator(dynamo_llm::block_manager::storage::DeviceAllocator::new(0).unwrap())
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();

        Ok(BlockManager {
            inner: Arc::from(dynamo_llm::block_manager::ReferenceBlockManager::new(config).unwrap()),
        })
    }

    fn allocate_blocks(&self, count: usize) -> PyResult<block_list::BlockList> {
        let blocks = self.inner.host().unwrap().allocate_blocks_blocking(count).unwrap();
        Ok(block_list::BlockList::from_rust(blocks))
    }
}
