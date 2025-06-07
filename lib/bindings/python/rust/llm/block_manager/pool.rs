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

use super::*;
use pyo3::{PyResult, Python};
use std::sync::Arc;

// Type erasure enum for BlockPool
pub enum BlockPoolType {
    PinnedPool(
        Arc<
            dynamo_llm::block_manager::BlockPool<
                dynamo_llm::block_manager::storage::PinnedStorage,
                dynamo_llm::block_manager::block::BasicMetadata,
            >,
        >,
    ),
    DevicePool(
        Arc<
            dynamo_llm::block_manager::BlockPool<
                dynamo_llm::block_manager::storage::DeviceStorage,
                dynamo_llm::block_manager::block::BasicMetadata,
            >,
        >,
    ),
}

impl BlockPoolType {
    // Helper methods for common operations with type dispatch
    pub async fn allocate_blocks(
        &self,
        count: usize,
    ) -> Result<Vec<super::block::BlockType>, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            BlockPoolType::PinnedPool(pool) => {
                let blocks = pool.allocate_blocks(count).await?;
                Ok(blocks
                    .into_iter()
                    .map(super::block::BlockType::Pinned)
                    .collect())
            }
            BlockPoolType::DevicePool(pool) => {
                let blocks = pool.allocate_blocks(count).await?;
                Ok(blocks
                    .into_iter()
                    .map(super::block::BlockType::Device)
                    .collect())
            }
        }
    }

    pub fn allocate_blocks_blocking(
        &self,
        count: usize,
    ) -> Result<Vec<super::block::BlockType>, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            BlockPoolType::PinnedPool(pool) => {
                let blocks = pool.allocate_blocks_blocking(count)?;
                Ok(blocks
                    .into_iter()
                    .map(super::block::BlockType::Pinned)
                    .collect())
            }
            BlockPoolType::DevicePool(pool) => {
                let blocks = pool.allocate_blocks_blocking(count)?;
                Ok(blocks
                    .into_iter()
                    .map(super::block::BlockType::Device)
                    .collect())
            }
        }
    }
}

#[pyclass]
pub struct BlockPool {
    inner: BlockPoolType,
    pool_type: String,
    dtype: dynamo_llm::common::dtype::DType,
    device_id: usize,
}

impl BlockPool {
    pub fn from_host_pool(
        pool: Arc<
            dynamo_llm::block_manager::BlockPool<
                dynamo_llm::block_manager::storage::PinnedStorage,
                dynamo_llm::block_manager::block::BasicMetadata,
            >,
        >,
        dtype: dynamo_llm::common::dtype::DType,
        device_id: usize,
    ) -> Self {
        Self {
            inner: BlockPoolType::PinnedPool(pool),
            pool_type: "pinned".to_string(),
            dtype,
            device_id,
        }
    }

    pub fn from_device_pool(
        pool: Arc<
            dynamo_llm::block_manager::BlockPool<
                dynamo_llm::block_manager::storage::DeviceStorage,
                dynamo_llm::block_manager::block::BasicMetadata,
            >,
        >,
        dtype: dynamo_llm::common::dtype::DType,
        device_id: usize,
    ) -> Self {
        Self {
            inner: BlockPoolType::DevicePool(pool),
            pool_type: "device".to_string(),
            dtype,
            device_id,
        }
    }
}

#[pymethods]
impl BlockPool {
    #[getter]
    fn pool_type(&self) -> &str {
        &self.pool_type
    }

    #[getter]
    fn dtype(&self) -> String {
        format!("{:?}", self.dtype)
    }

    #[getter]
    fn device_id(&self) -> usize {
        self.device_id
    }

    // Synchronous methods
    fn allocate_blocks_blocking(&self, count: usize) -> PyResult<super::block_list::BlockList> {
        let blocks = self
            .inner
            .allocate_blocks_blocking(count)
            .map_err(to_pyerr)?;
        Ok(super::block_list::BlockList::from_rust(
            blocks,
            self.dtype,
            self.device_id,
        ))
    }

    // Async methods (following your pattern)
    fn allocate_blocks<'py>(&self, py: Python<'py>, count: usize) -> PyResult<Bound<'py, PyAny>> {
        let inner = match &self.inner {
            BlockPoolType::PinnedPool(pool) => BlockPoolType::PinnedPool(pool.clone()),
            BlockPoolType::DevicePool(pool) => BlockPoolType::DevicePool(pool.clone()),
        };
        let dtype = self.dtype;
        let device_id = self.device_id;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let blocks = inner.allocate_blocks(count).await.map_err(to_pyerr)?;
            Ok(super::block_list::BlockList::from_rust(
                blocks, dtype, device_id,
            ))
        })
    }
}
