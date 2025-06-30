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

use std::sync::Arc;

use dynamo_llm::block_manager as bm;

use crate::to_pyerr;

type DeviceStorageType = bm::storage::DeviceStorage;

#[derive(Debug)]
pub enum BlockListType {
    Immutable(Vec<bm::block::ImmutableBlock<DeviceStorageType, bm::block::locality::Local, bm::BasicMetadata>>),
    Mutable(Vec<bm::block::MutableBlock<DeviceStorageType, bm::block::locality::Local, bm::BasicMetadata>>),
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct KvbmBlockList {
    blocks: Arc<std::sync::Mutex<Option<BlockListType>>>,
    count: usize,
}

impl KvbmBlockList {
    pub fn new(blocks: BlockListType) -> Self {
        let count = match &blocks {
            BlockListType::Immutable(blocks) => blocks.len(),
            BlockListType::Mutable(blocks) => blocks.len(),
        };

        Self {
            blocks: Arc::new(std::sync::Mutex::new(Some(blocks))),
            count,
        }
    }

    pub fn take_blocks(&self) -> Option<BlockListType> {
        let mut blocks = self.blocks.lock().unwrap();
        blocks.take()
    }
}

#[pymethods]
impl KvbmBlockList {
    pub fn get_block_id(&self, block_idx: usize) -> PyResult<usize> {
        let blocks = self.blocks.lock().unwrap();
        let block_id = match &*blocks {
            Some(BlockListType::Immutable(blocks)) => blocks.get(block_idx).map(|b| b.block_id()),
            Some(BlockListType::Mutable(blocks)) => blocks.get(block_idx).map(|b| b.block_id()),
            None => None,
        };

        block_id.ok_or_else(|| to_pyerr("block not found"))
    }

    pub fn get_block_hash(&self, block_idx: usize) -> PyResult<Option<u64>> {
        let blocks = self.blocks.lock().unwrap();
        let sequence_hash = match &*blocks {
            Some(BlockListType::Immutable(blocks)) => Some(
                blocks
                    .get(block_idx)
                    .ok_or_else(|| to_pyerr("block not found"))?
                    .sequence_hash(),
            ),
            Some(BlockListType::Mutable(blocks)) => blocks
                .get(block_idx)
                .ok_or_else(|| to_pyerr("block not found"))?
                .sequence_hash()
                .ok(),
            None => None,
        };

        Ok(sequence_hash)
    }

    pub fn block_count(&self) -> usize {
        self.count
    }
}

/// vLLM has a KVCacheBlock object which holds the block ID and sequence hash information.
/// The way vLLM computes the sequence hash is different than the way Dynamo computes it;
/// however, vLLM does provide the necessary information within the `BlockHashType` to
/// extract the tokens ids for the block so we can compute our own sequence hash.
///
/// This object represents a converted `KVCacheBlock` object into something we can directly
/// use in rust.
#[pyclass]
#[derive(Debug, Clone)]
pub struct BlockState {
    pub block_id: usize,
    pub tokens: Option<Vec<u32>>,
}

#[pymethods]
impl BlockState {
    #[new]
    #[pyo3(signature = (block_id, tokens = None))]
    pub fn new(block_id: usize, tokens: Option<Vec<u32>>) -> Self {
        Self { block_id, tokens }
    }

    pub fn block_id(&self) -> usize {
        self.block_id
    }
}

#[pyclass]
#[derive(Debug, Clone, Default)]
pub struct BlockStates {
    pub states: Vec<BlockState>,
}

#[pymethods]
impl BlockStates {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    #[pyo3(signature = (block_id, tokens = None))]
    pub fn emplace_back(&mut self, block_id: usize, tokens: Option<Vec<u32>>) {
        self.states.push(BlockState::new(block_id, tokens));
    }

    pub fn push_back(&mut self, state: BlockState) {
        self.states.push(state);
    }

    pub fn block_ids(&self) -> Vec<usize> {
        self.states.iter().map(|s| s.block_id).collect()
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }
}

impl From<Vec<BlockState>> for BlockStates {
    fn from(states: Vec<BlockState>) -> Self {
        Self { states }
    }
}
