use pyo3::prelude::*;
use std::sync::Arc;

use dynamo_llm::block_manager::{self as bm, block::BlockIdentifier};
use dynamo_llm::tokens::{compute_hash_v2, TokenBlockSequence, Tokens};

use crate::to_pyerr;

pub type DeviceStorageType = bm::storage::DeviceStorage;

/// Request Inputs
#[pyclass]
#[derive(Debug, Clone)]
pub struct KvRequest {
    lora_name: Option<String>,
    salt_hash: u64,
    tbs: Arc<TokenBlockSequence>,
}

impl KvRequest {
    pub fn tbs(&self) -> Arc<TokenBlockSequence> {
        self.tbs.clone()
    }
}

#[pymethods]
impl KvRequest {
    #[new]
    #[pyo3(signature = (tokens, block_size, lora_name=None, salt_hash=None))]
    fn new(
        tokens: Vec<usize>,
        block_size: usize,
        lora_name: Option<String>,
        salt_hash: Option<String>,
    ) -> Self {
        let tokens: Tokens = tokens
            .into_iter()
            .map(|t| t as u32)
            .collect::<Vec<_>>()
            .into();

        // compute salt
        #[derive(serde::Serialize)]
        struct Salt {
            #[serde(skip_serializing_if = "Option::is_none")]
            salt: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            lora_name: Option<String>,
        }

        let salt = Salt {
            salt: salt_hash,
            lora_name: lora_name.clone(),
        };

        let salt_bytes = serde_json::to_vec(&salt).unwrap();
        let salt_hash = compute_hash_v2(&salt_bytes, 0);

        let sequence = Arc::new(TokenBlockSequence::new(tokens, block_size, Some(salt_hash)));

        Self {
            lora_name,
            salt_hash,
            tbs: sequence,
        }
    }

    pub fn sequence_hashes(&self) -> Vec<u64> {
        self.tbs
            .blocks()
            .iter()
            .map(|b| b.sequence_hash())
            .collect()
    }
}

#[derive(Debug)]
pub enum BlockListType {
    Immutable(Vec<bm::block::ImmutableBlock<DeviceStorageType, bm::BasicMetadata>>),
    Mutable(Vec<bm::block::MutableBlock<DeviceStorageType, bm::BasicMetadata>>),
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct DynamoKvBlockList {
    blocks: Arc<std::sync::Mutex<Option<BlockListType>>>,
    count: usize,
}

impl DynamoKvBlockList {
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
impl DynamoKvBlockList {
    fn get_block_id(&self, block_idx: usize) -> PyResult<usize> {
        let blocks = self.blocks.lock().unwrap();
        let block_id = match &*blocks {
            Some(BlockListType::Immutable(blocks)) => blocks.get(block_idx).map(|b| b.block_id()),
            Some(BlockListType::Mutable(blocks)) => blocks.get(block_idx).map(|b| b.block_id()),
            None => None,
        };

        Ok(block_id.ok_or_else(|| to_pyerr("block not found"))?)
    }

    fn get_block_hash(&self, block_idx: usize) -> PyResult<Option<u64>> {
        let blocks = self.blocks.lock().unwrap();
        let block_id = match &*blocks {
            Some(BlockListType::Immutable(blocks)) => blocks
                .get(block_idx)
                .ok_or(to_pyerr("block not found"))?
                .sequence_hash()
                .ok(),
            Some(BlockListType::Mutable(blocks)) => blocks
                .get(block_idx)
                .ok_or(to_pyerr("block not found"))?
                .sequence_hash()
                .ok(),
            None => None,
        };

        Ok(block_id)
    }

    #[pyo3(name = "__len__")]
    fn len(&self) -> usize {
        self.count
    }
}
