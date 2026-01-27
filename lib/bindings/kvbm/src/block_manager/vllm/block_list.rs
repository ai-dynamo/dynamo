// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use std::sync::Arc;

use dynamo_llm::block_manager as bm;
use dynamo_llm::block_manager::block::data::logical::distributed_leader_worker::DistributedLeaderWorkerResources;
use dynamo_llm::block_manager::block::locality::Logical;

use crate::to_pyerr;

type DeviceStorageType = bm::storage::DeviceStorage;
type HostStorageType = bm::storage::PinnedStorage;
type DiskStorageType = bm::storage::DiskStorage;

#[derive(Debug)]
pub enum BlockListType {
    ImmutableDevice(
        Vec<
            bm::block::ImmutableBlock<
                DeviceStorageType,
                Logical<DistributedLeaderWorkerResources>,
                bm::BasicMetadata,
            >,
        >,
    ),
    MutableDevice(
        Vec<
            bm::block::MutableBlock<
                DeviceStorageType,
                Logical<DistributedLeaderWorkerResources>,
                bm::BasicMetadata,
            >,
        >,
    ),
    ImmutableHost(
        Vec<
            bm::block::ImmutableBlock<
                HostStorageType,
                Logical<DistributedLeaderWorkerResources>,
                bm::BasicMetadata,
            >,
        >,
    ),
    MutableHost(
        Vec<
            bm::block::MutableBlock<
                HostStorageType,
                Logical<DistributedLeaderWorkerResources>,
                bm::BasicMetadata,
            >,
        >,
    ),
    ImmutableDisk(
        Vec<
            bm::block::ImmutableBlock<
                DiskStorageType,
                Logical<DistributedLeaderWorkerResources>,
                bm::BasicMetadata,
            >,
        >,
    ),
    MutableDisk(
        Vec<
            bm::block::MutableBlock<
                DiskStorageType,
                Logical<DistributedLeaderWorkerResources>,
                bm::BasicMetadata,
            >,
        >,
    ),
}

#[pyclass]
#[derive(Clone)]
pub struct KvbmBlockList {
    blocks: Arc<std::sync::Mutex<Option<BlockListType>>>,
    count: usize,
}

impl KvbmBlockList {
    pub fn new(blocks: BlockListType) -> Self {
        let count = match &blocks {
            BlockListType::ImmutableDevice(blocks) => blocks.len(),
            BlockListType::MutableDevice(blocks) => blocks.len(),
            BlockListType::ImmutableHost(blocks) => blocks.len(),
            BlockListType::MutableHost(blocks) => blocks.len(),
            BlockListType::ImmutableDisk(blocks) => blocks.len(),
            BlockListType::MutableDisk(blocks) => blocks.len(),
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
            Some(BlockListType::ImmutableDevice(blocks)) => {
                blocks.get(block_idx).map(|b| b.block_id())
            }
            Some(BlockListType::MutableDevice(blocks)) => {
                blocks.get(block_idx).map(|b| b.block_id())
            }
            Some(BlockListType::ImmutableHost(blocks)) => {
                blocks.get(block_idx).map(|b| b.block_id())
            }
            Some(BlockListType::MutableHost(blocks)) => blocks.get(block_idx).map(|b| b.block_id()),
            Some(BlockListType::ImmutableDisk(blocks)) => {
                blocks.get(block_idx).map(|b| b.block_id())
            }
            Some(BlockListType::MutableDisk(blocks)) => blocks.get(block_idx).map(|b| b.block_id()),
            None => None,
        };

        block_id.ok_or_else(|| to_pyerr("block not found"))
    }

    pub fn get_block_hash(&self, block_idx: usize) -> PyResult<Option<u64>> {
        let blocks = self.blocks.lock().unwrap();
        let sequence_hash = match &*blocks {
            Some(BlockListType::ImmutableDevice(blocks)) => Some(
                blocks
                    .get(block_idx)
                    .ok_or_else(|| to_pyerr("block not found"))?
                    .sequence_hash(),
            ),
            Some(BlockListType::MutableDevice(blocks)) => blocks
                .get(block_idx)
                .ok_or_else(|| to_pyerr("block not found"))?
                .sequence_hash()
                .ok(),
            Some(BlockListType::ImmutableHost(blocks)) => Some(
                blocks
                    .get(block_idx)
                    .ok_or_else(|| to_pyerr("block not found"))?
                    .sequence_hash(),
            ),
            Some(BlockListType::MutableHost(blocks)) => blocks
                .get(block_idx)
                .ok_or_else(|| to_pyerr("block not found"))?
                .sequence_hash()
                .ok(),
            Some(BlockListType::ImmutableDisk(blocks)) => Some(
                blocks
                    .get(block_idx)
                    .ok_or_else(|| to_pyerr("block not found"))?
                    .sequence_hash(),
            ),
            Some(BlockListType::MutableDisk(blocks)) => blocks
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

    pub fn get_block_ids(&self) -> Vec<usize> {
        let blocks = self.blocks.lock().unwrap();
        match &*blocks {
            Some(BlockListType::ImmutableDevice(blocks)) => {
                blocks.iter().map(|b| b.block_id()).collect()
            }
            Some(BlockListType::MutableDevice(blocks)) => {
                blocks.iter().map(|b| b.block_id()).collect()
            }
            Some(BlockListType::ImmutableHost(blocks)) => {
                blocks.iter().map(|b| b.block_id()).collect()
            }
            Some(BlockListType::MutableHost(blocks)) => {
                blocks.iter().map(|b| b.block_id()).collect()
            }
            Some(BlockListType::ImmutableDisk(blocks)) => {
                blocks.iter().map(|b| b.block_id()).collect()
            }
            Some(BlockListType::MutableDisk(blocks)) => {
                blocks.iter().map(|b| b.block_id()).collect()
            }
            None => Vec::new(),
        }
    }

    pub fn get_block_hashes(&self) -> Vec<u64> {
        let blocks = self.blocks.lock().unwrap();
        match &*blocks {
            Some(BlockListType::ImmutableDevice(blocks)) => {
                blocks.iter().map(|b| b.sequence_hash()).collect()
            }

            Some(BlockListType::ImmutableHost(blocks)) => {
                blocks.iter().map(|b| b.sequence_hash()).collect()
            }

            Some(BlockListType::ImmutableDisk(blocks)) => {
                blocks.iter().map(|b| b.sequence_hash()).collect()
            }

            _ => Vec::new(),
        }
    }

    pub fn get_block_types(&self) -> Vec<String> {
        let blocks = self.blocks.lock().unwrap();
        match &*blocks {
            Some(BlockListType::ImmutableDevice(_)) => vec!["ImmutableDevice".to_string()],
            Some(BlockListType::MutableDevice(_)) => vec!["MutableDevice".to_string()],
            Some(BlockListType::ImmutableHost(_)) => vec!["ImmutableHost".to_string()],
            Some(BlockListType::MutableHost(_)) => vec!["MutableHost".to_string()],
            Some(BlockListType::ImmutableDisk(_)) => vec!["ImmutableDisk".to_string()],
            Some(BlockListType::MutableDisk(_)) => vec!["MutableDisk".to_string()],
            None => Vec::new(),
        }
    }
}

impl std::fmt::Debug for KvbmBlockList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KvbmBlockList(count: {}; block_types: {:?}; block_ids: {:?}; block_hashes: {:?})",
            self.count,
            self.get_block_types(),
            self.get_block_ids(),
            self.get_block_hashes()
        )
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
#[derive(Clone, Default)]
pub struct BlockStates {
    pub states: Vec<BlockState>,
}

impl std::fmt::Debug for BlockStates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let block_ids = self.states.iter().map(|s| s.block_id).collect::<Vec<_>>();
        write!(f, "BlockStates(block_ids: {:?})", block_ids)
    }
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
