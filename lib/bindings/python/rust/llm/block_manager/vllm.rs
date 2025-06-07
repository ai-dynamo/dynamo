// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, VecDeque},
    sync::Mutex,
};

use derive_getters::Dissolve;
use pyo3::{prelude::*, wrap_pymodule};

use dynamo_llm::{
    block_manager::{
        block::{BlockId, BlockIdentifier, ImmutableBlock, MutableBlock},
        pool::BlockPool,
        BasicMetadata, DeviceStorage, KvBlockManager, Storage,
    },
    tokens::{SaltHash, SequenceHash, TokenBlockSequence, Tokens},
};

use crate::llm::block_manager::BlockManager as PyBlockManager;

use crate::to_pyerr;

mod block_list;
mod request;
mod slot;

pub use block_list::{BlockListType, BlockState, BlockStates, KvbmBlockList};
pub use request::KvbmRequest;
pub use slot::{Slot, SlotPosition};

#[pymodule]
fn _vllm_integration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KvbmCacheManager>()?;
    m.add_class::<KvbmRequest>()?;
    m.add_class::<KvbmBlockList>()?;
    m.add_class::<BlockState>()?;
    m.add_class::<BlockStates>()?;
    m.add_class::<SlotUpdate>()?;
    Ok(())
}

/// Add bingings from this crate to the provided module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(_vllm_integration))?;
    Ok(())
}

#[pyclass]
pub struct KvbmCacheManager {
    block_manager: PyBlockManager,
    slot_manager: Mutex<SlotManager<String>>,
}

impl KvbmCacheManager {
    #[inline(always)]
    pub fn block_manager(&self) -> &KvBlockManager<BasicMetadata> {
        self.block_manager.get_block_manager()
    }
}

#[pymethods]
impl KvbmCacheManager {
    #[new]
    #[pyo3(signature = (block_manager))]
    pub fn new(block_manager: PyBlockManager) -> PyResult<Self> {
        let slot_manager = Mutex::new(SlotManager::new(block_manager.block_size()));
        Ok(Self {
            block_manager,
            slot_manager,
        })
    }

    /// Create a new slot for the given request ID.
    /// This is used to create a new slot for the request.
    pub fn create_slot(
        &self,
        request: KvbmRequest,
        tokens: Vec<u32>,
    ) -> PyResult<Vec<SequenceHash>> {
        let mut slot_manager = self.slot_manager.lock().map_err(to_pyerr)?;
        slot_manager
            .create_slot(&request.request_id, request.salt_hash, tokens)
            .map_err(to_pyerr)
    }

    /// Returns the number of tokens that have been computed/accepted for the given request.
    pub fn num_computed_tokens(&self, request_id: String) -> PyResult<usize> {
        let slot_manager = self.slot_manager.lock().map_err(to_pyerr)?;
        slot_manager
            .num_tokens(&request_id, SlotPosition::Computed)
            .map_err(to_pyerr)
    }

    /// Get the computed blocks for the given sequence hashes.
    /// This is used to get the blocks for the request.
    pub fn get_computed_blocks(
        &self,
        sequence_hashes: Vec<SequenceHash>,
    ) -> PyResult<KvbmBlockList> {
        let blocks = self
            .block_manager()
            .device()
            .unwrap()
            .match_sequence_hashes_blocking(&sequence_hashes)
            .map_err(to_pyerr)?;

        Ok(KvbmBlockList::new(BlockListType::Immutable(blocks)))
    }

    /// Updates the slot manager with the current request state and allocates new blocks if needed.
    pub fn alloctate_slots(&self, update: SlotUpdate) -> PyResult<Option<BlockStates>> {
        self.slot_manager
            .lock()
            .map_err(to_pyerr)?
            .update_slot(update.dissolve(), self.block_manager())
            .map_err(to_pyerr)
    }
}

#[derive(Debug, Clone, Dissolve)]
pub struct GenericSlotUpdate<R> {
    /// The request ID.
    pub request_id: R,

    /// External state about the number of computed tokens in the request.
    /// This should match the slots expectation.
    pub request_num_tokens: usize,

    /// The tokens to append to the sequence.
    /// After the tokens are appendend, the internal sequence length should match `request_num_token`
    pub tokens_to_append: Vec<usize>,

    /// The number of new tokens which advances the sequence state.
    /// This is the number of tokens which will be computed in the near future.
    /// When [BaseKvCacheManager::update_slot] is called again, these tokens will be committed.
    pub num_new_tokens: usize,

    /// The number of new computed tokens in the request.
    /// The `num_new_tokens / block_size` should be equal to the length of the `new_computed_blocks`,
    /// it may have a remainder for the partial block state.
    pub num_new_computed_tokens: Option<usize>,

    /// The new computed blocks which advance the sequence state.
    pub new_computed_blocks: Option<BlockStates>,

    /// The number of lookahead blocks to cache.
    pub num_lookahead_blocks: Option<usize>,

    /// Whether to delay caching the blocks.
    pub delay_cache_blocks: Option<bool>,
}

#[pyclass]
#[derive(Debug, Clone, Dissolve)]
pub struct SlotUpdate(pub GenericSlotUpdate<String>);

#[pymethods]
impl SlotUpdate {
    #[new]
    #[pyo3(signature = (request_id, request_num_tokens, tokens_to_append, num_new_tokens, num_new_computed_tokens=None, new_computed_blocks=None, num_lookahead_blocks=None, delay_cache_blocks=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: String,
        request_num_tokens: usize,
        tokens_to_append: Vec<usize>,
        num_new_tokens: usize,
        num_new_computed_tokens: Option<usize>,
        new_computed_blocks: Option<BlockStates>,
        num_lookahead_blocks: Option<usize>,
        delay_cache_blocks: Option<bool>,
    ) -> Self {
        let update = GenericSlotUpdate {
            request_id,
            request_num_tokens,
            tokens_to_append,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_blocks,
            delay_cache_blocks,
        };

        SlotUpdate(update)
    }
}

pub trait RequestKey:
    std::hash::Hash + std::cmp::Eq + Clone + std::fmt::Debug + Send + Sync + 'static
{
}

impl RequestKey for String {}

#[derive(Debug, thiserror::Error)]
pub enum SlotError {
    #[error("slot already exists")]
    AlreadyExists,

    #[error("slot not found")]
    NotFound,

    #[error("slot error: {0}")]
    Error(String),
}

impl SlotError {
    pub fn from_str(msg: &str) -> Self {
        Self::Error(msg.to_string())
    }
}

pub struct SlotManager<R: RequestKey> {
    slots: HashMap<R, Slot<DeviceStorage>>,
    block_size: usize,
}

impl<R: RequestKey> SlotManager<R> {
    /// Creates a new slot manager.
    pub fn new(block_size: usize) -> Self {
        Self {
            slots: HashMap::new(),
            block_size,
        }
    }

    /// Returns the number of tokens in the sequence for the given request ID.
    pub fn num_tokens(&self, request_id: &R, position: SlotPosition) -> Result<usize, SlotError> {
        let slot = self.slots.get(request_id).ok_or(SlotError::NotFound)?;
        Ok(slot.num_tokens(position))
    }

    /// Creates a new slot for the given request ID.
    /// This will populate the slot with the prefill tokens in the block sequence.
    pub fn create_slot(
        &mut self,
        request_id: &R,
        salt_hash: SaltHash,
        tokens: Vec<u32>,
    ) -> Result<Vec<SequenceHash>, SlotError> {
        // let token_count = tokens.len();

        // if !self.slots.contains_key(request_id) {
        //     self.slots.insert(
        //         request_id.clone(),
        //         Slot::new(tokens.into(), self.block_size, salt_hash),
        //     );
        // }

        // let slot = self.slots.get(request_id).ok_or(SlotError::NotFound)?;

        // let sequence_hashes = slot
        //     .sequence
        //     .blocks()
        //     .iter()
        //     .map(|b| b.sequence_hash())
        //     .collect();

        // Ok(sequence_hashes)
        unimplemented!()
    }

    pub fn update_slot(
        &mut self,
        update: GenericSlotUpdate<R>,
        bm: &KvBlockManager<BasicMetadata>,
    ) -> Result<Option<BlockStates>, SlotError> {
        // let (
        //     request_id,
        //     request_num_tokens,
        //     tokens_to_append,
        //     num_new_tokens,
        //     num_new_computed_tokens,
        //     new_computed_blocks,
        //     num_lookahead_blocks,
        //     delay_cache_blocks,
        // ) = update.dissolve();

        // // check conditions:
        // // - if new_computed_blocks is provided, then tokens_to_append

        // let slot = self.slots.get_mut(&request_id).ok_or(SlotError::NotFound)?;

        // slot.update_sequence(tokens_to_append)?;
        // debug_assert_eq!(slot.sequence.total_tokens(), request_num_tokens);

        // match new_computed_blocks {
        //     // decode
        //     None => {
        //         slot.update_blocks(bm.device().unwrap())?;
        //     }
        //     Some(new_computed_blocks) => {
        //         slot.update_blocks(bm.device().unwrap())?;
        //     }
        // }

        // // 3. allocate new blocks if needed
        // let required_tokens = slot.sequence.total_tokens() + num_new_tokens;
        // let num_blocks = required_tokens.div_ceil(self.block_size);
        // let num_new_blocks = num_blocks - (slot.immutable.len() + slot.mutable.len());

        // let new_blocks = bm
        //     .device()
        //     .unwrap()
        //     .allocate_blocks_blocking(num_new_blocks)
        //     .ok();

        // let mut block_descriptors = BlockStates::new();

        // if let Some(new_blocks) = new_blocks {
        //     new_blocks.into_iter().for_each(|block| {
        //         block_descriptors.push_back(BlockState::new(block.block_id(), None));
        //         slot.mutable.push_back(block);
        //     });
        // } else {
        //     return Ok(None);
        // }

        // Ok(Some(block_descriptors))

        unimplemented!()
    }
}
