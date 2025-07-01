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
        block::data::logical::distributed_leader_worker::DistributedLeaderWorkerResources,
        block::locality::{LocalityProvider, Logical},
        block::{BlockId, ImmutableBlock, MutableBlock},
        pool::BlockPool,
        BasicMetadata, DeviceStorage, Storage,
    },
    tokens::{SaltHash, SequenceHash, TokenBlockSequence, Tokens},
};

// use crate::llm::block_manager::BlockManager as PyBlockManager;
use crate::llm::block_manager::BlockManager as PyBlockManager;
use crate::llm::block_manager::VllmBlockManager;

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
    m.add_class::<KvCacheEvent>()?;
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

#[pyclass]
pub struct KvCacheEvent {}

impl KvbmCacheManager {
    #[inline(always)]
    pub fn block_manager(&self) -> &VllmBlockManager {
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

    pub fn has_slot(&self, request_id: String) -> PyResult<bool> {
        let slot_manager = self.slot_manager.lock().map_err(to_pyerr)?;
        Ok(slot_manager.has_slot(&request_id))
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

    /// Returns the number of tokens that have been computed for the given request.
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

        Ok(KvbmBlockList::new(BlockListType::ImmutableDevice(blocks)))
    }

    /// Get the number of offloaded computed blocks for the given sequence hashes.
    pub fn get_num_offloaded_computed_blocks(
        &self,
        sequence_hashes: Vec<SequenceHash>,
    ) -> PyResult<(usize, usize)> {
        let num_host_blocks = if let Some(host) = self.block_manager().host() {
            host.match_sequence_hashes_blocking(&sequence_hashes)
                .map_err(to_pyerr)?
                .len()
        } else {
            0
        };

        let num_disk_blocks = if let Some(disk) = self.block_manager().disk() {
            disk.match_sequence_hashes_blocking(&sequence_hashes)
                .map_err(to_pyerr)?
                .len()
        } else {
            0
        };

        tracing::debug!(
            "in get_num_offloaded_computed_blocks, found {} host blocks and {} disk blocks",
            num_host_blocks,
            num_disk_blocks,
        );

        Ok((num_host_blocks, num_disk_blocks))
    }

    /// Updates the slot manager with the current request state and allocates new blocks if needed.
    /// Returns the new blocks if they were allocated, otherwise returns None.
    pub fn alloctate_slots(&self, update: SlotUpdate) -> PyResult<Option<BlockStates>> {
        self.slot_manager
            .lock()
            .map_err(to_pyerr)?
            .update_slot(update.dissolve(), self.block_manager())
            .map_err(to_pyerr)
    }

    pub fn free(&self, request_id: String) -> PyResult<()> {
        self.slot_manager
            .lock()
            .map_err(to_pyerr)?
            .free_blocks(&request_id)
            .map_err(to_pyerr)
    }

    pub fn reset_prefix_cache(&self) -> PyResult<()> {
        Err(to_pyerr("reset_prefix_cache is not implemented"))
    }

    pub fn get_num_common_prefix_blocks(
        &self,
        _request_id: String,
        _num_running_requests: usize,
    ) -> PyResult<usize> {
        Err(to_pyerr("get_num_common_prefix_blocks is not implemented"))
    }

    /// Free the entire slot for the given request ID.
    pub fn free_block_hashes(&self, request_id: String) -> PyResult<()> {
        self.slot_manager
            .lock()
            .map_err(to_pyerr)?
            .drop_slot(&request_id)
            .map_err(to_pyerr)
    }

    pub fn take_events(&self) -> PyResult<Vec<KvCacheEvent>> {
        // we don't need events
        Ok(vec![])
    }

    pub fn get_block_ids(&self, request_id: String) -> PyResult<Vec<BlockId>> {
        self.slot_manager
            .lock()
            .map_err(to_pyerr)?
            .get_block_ids(&request_id)
            .map_err(to_pyerr)
    }

    pub fn usage(&self) -> PyResult<f64> {
        let pool = self.block_manager().device().unwrap();
        let inuse = pool.total_blocks() - pool.available_blocks();
        let usage: f64 = inuse as f64 / pool.total_blocks() as f64;
        Ok(usage)
    }

    #[pyo3(signature = (host_blocks=None, disk_blocks=None))]
    pub fn onboard_blocks(
        &self,
        host_blocks: Option<KvbmBlockList>,
        disk_blocks: Option<KvbmBlockList>,
    ) -> PyResult<(Option<KvbmBlockList>, Option<KvbmBlockList>)> {
        let host_onboard_blocks = if let Some(host_blocks) = host_blocks {
            let host_blocks = host_blocks
                .take_blocks()
                .ok_or(to_pyerr("host_blocks has already been taken"))?;
            let BlockListType::ImmutableHost(blocks) = host_blocks else {
                return Err(to_pyerr("Blocks are not immutable host blocks"));
            };

            let device_blocks = self
                .block_manager()
                .onboard_blocks(blocks)
                .blocking_recv()
                .map_err(to_pyerr)?
                .map_err(to_pyerr)?;
            Some(KvbmBlockList::new(BlockListType::ImmutableDevice(
                device_blocks,
            )))
        } else {
            None
        };

        let disk_onboard_blocks = if let Some(disk_blocks) = disk_blocks {
            let disk_blocks = disk_blocks
                .take_blocks()
                .ok_or(to_pyerr("disk_blocks has already been taken"))?;
            let BlockListType::ImmutableDisk(blocks) = disk_blocks else {
                return Err(to_pyerr("Blocks are not immutable disk blocks"));
            };

            let device_blocks = self
                .block_manager()
                .onboard_blocks(blocks)
                .blocking_recv()
                .map_err(to_pyerr)?
                .map_err(to_pyerr)?;
            Some(KvbmBlockList::new(BlockListType::ImmutableDevice(
                device_blocks,
            )))
        } else {
            None
        };

        Ok((host_onboard_blocks, disk_onboard_blocks))
    }
}

#[derive(Debug, Clone, Dissolve)]
pub struct GenericSlotUpdate<R> {
    /// The request ID.
    pub request_id: R,

    /// External state about the number of tokens in the request.
    /// This should match the slots expectation.
    pub request_num_tokens: usize,

    /// External state about the number of computed tokens in the request.
    /// This should match the slots expectation.
    pub request_num_computed_tokens: usize,

    /// The tokens to append to the sequence.
    /// After the tokens are appendend, the internal sequence length should match `request_num_tokens`.
    pub tokens_to_append: Vec<u32>,

    /// The number of new tokens which advances the sequence state.
    /// This is the number of tokens which will be computed in the near future.
    /// When [BaseKvCacheManager::update_slot] is called again, these tokens will be committed.
    pub num_new_tokens: usize,

    /// The number of new computed tokens in the request.
    /// The `num_new_tokens / block_size` should be equal to the length of the `new_computed_blocks`,
    /// it may have a remainder for the partial block state.
    /// Note: this field is solely tied to the `new_computed_blocks` field and not used when `tokens_to_append` is provided.
    /// The name might be confusing, but the name matched the vLLM implementation.
    pub num_new_computed_tokens: Option<usize>,

    /// The new computed blocks which advance the sequence state.
    pub new_computed_blocks: Option<KvbmBlockList>,

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
    #[pyo3(signature = (request_id, request_num_tokens, request_num_computed_tokens, tokens_to_append, num_new_tokens, num_new_computed_tokens=None, new_computed_blocks=None, num_lookahead_blocks=None, delay_cache_blocks=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: String,
        request_num_tokens: usize,
        request_num_computed_tokens: usize,
        tokens_to_append: Vec<u32>,
        num_new_tokens: usize,
        num_new_computed_tokens: Option<usize>,
        new_computed_blocks: Option<KvbmBlockList>,
        num_lookahead_blocks: Option<usize>,
        delay_cache_blocks: Option<bool>,
    ) -> Self {
        let update = GenericSlotUpdate {
            request_id,
            request_num_tokens,
            request_num_computed_tokens,
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
    std::hash::Hash
    + std::cmp::Eq
    + std::fmt::Debug
    + std::fmt::Display
    + tracing::Value
    + Clone
    + Send
    + Sync
    + 'static
{
}

impl RequestKey for String {}

#[derive(Debug, thiserror::Error)]
pub enum SlotError {
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
    slots: HashMap<R, Slot<DeviceStorage, Logical<DistributedLeaderWorkerResources>>>,
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

    /// Returns true if the slot manager has a slot for the given request ID.
    pub fn has_slot(&self, request_id: &R) -> bool {
        self.slots.contains_key(request_id)
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
        tracing::debug!(request_id, "creating slot");

        if !self.slots.contains_key(request_id) {
            self.slots.insert(
                request_id.clone(),
                Slot::new(tokens.into(), self.block_size, salt_hash),
            );
        }

        let slot = self.slots.get(request_id).ok_or(SlotError::NotFound)?;
        Ok(slot.sequence_hashes(SlotPosition::All))
    }

    pub fn update_slot(
        &mut self,
        update: GenericSlotUpdate<R>,
        bm: &VllmBlockManager,
    ) -> Result<Option<BlockStates>, SlotError> {
        let (
            request_id,
            _request_num_tokens,
            request_num_computed_tokens,
            tokens_to_append,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_blocks,
            delay_cache_blocks,
        ) = update.dissolve();

        // TODO(ryan): add support for lookahead blocks
        if num_lookahead_blocks.is_some() {
            return Err(SlotError::Error(
                "num_lookahead_blocks is not supported".to_string(),
            ));
        }

        // TODO: add support for delay_cache_blocks
        if delay_cache_blocks.unwrap_or(false) {
            return Err(SlotError::Error(
                "delay_cache_blocks is not supported".to_string(),
            ));
        }

        let slot = self.slots.get_mut(&request_id).ok_or(SlotError::NotFound)?;

        // we always apply the matched blocks to the beginning of the sequence; however,
        // if we fail to allocate the requested new blocks, vllm treats the request as never started,
        // so we need to drop the applied immutable block. however, if we have successfully advanced
        // the sequence state, then we rely on the scheduler to free any held blocks.
        let first_allocation = slot.first_allocation();

        // first apply any new computed blocks
        // these are the blocks that were matched to the sequence hashes
        // this will advance the computed position of the slot
        if let Some(matched_blocks) = new_computed_blocks {
            let blocks = matched_blocks.take_blocks();
            match blocks {
                Some(BlockListType::ImmutableDevice(blocks)) => {
                    tracing::debug!(
                        request_id,
                        "applying {} cache-hit tokens",
                        blocks.len() * self.block_size
                    );
                    slot.apply_computed_blocks(blocks)?;
                }
                Some(BlockListType::MutableDevice(_blocks)) => {
                    panic!(
                        "impossibility: mutable blocks were provided instead of immutable blocks"
                    );
                }
                Some(BlockListType::ImmutableHost(_blocks)) => {
                    panic!("ImmutableHost should not be provided");
                }
                Some(BlockListType::MutableHost(_blocks)) => {
                    panic!("MutableHost should not be provided");
                }
                Some(BlockListType::ImmutableDisk(_blocks)) => {
                    panic!("ImmutableDisk should not be provided");
                }
                Some(BlockListType::MutableDisk(_blocks)) => {
                    panic!("MutableDisk should not be provided");
                }
                None => {
                    panic!("impossibility: block list was none; possible taken previously");
                }
            }
        } else {
            tracing::debug!(request_id, "applying {} tokens", tokens_to_append.len());
            slot.apply_computed_tokens(tokens_to_append, bm.device().unwrap())?;
        }

        debug_assert_eq!(
            slot.num_tokens(SlotPosition::Computed),
            request_num_computed_tokens + num_new_computed_tokens.unwrap_or(0)
        );

        // 3. allocate new blocks if needed
        let new_blocks = slot
            .allocate_blocks(num_new_tokens, bm.device().unwrap())
            .map(|new_block_ids| {
                new_block_ids
                    .into_iter()
                    .map(|block_id| BlockState::new(block_id, None))
                    .collect::<Vec<BlockState>>()
                    .into()
            });

        match new_blocks {
            Some(new_blocks) => Ok(Some(new_blocks)),
            None => {
                // could not allocate new blocks and we reset the slot
                // note: we could free the blocks here; however, apply_computed_blocks always resets the
                // immutable block list, avoiding the free_blocks() here allows us to hold the reference count on
                // the blocks we intend to reuse
                if first_allocation {
                    slot.free_blocks();
                }
                Ok(None)
            }
        }
    }

    pub fn get_block_ids(&self, request_id: &R) -> Result<Vec<BlockId>, SlotError> {
        let slot = self.slots.get(request_id).ok_or(SlotError::NotFound)?;
        Ok(slot.get_block_ids())
    }

    pub fn free_blocks(&mut self, request_id: &R) -> Result<(), SlotError> {
        let slot = self.slots.get_mut(request_id).ok_or(SlotError::NotFound)?;
        slot.free_blocks();
        Ok(())
    }

    pub fn drop_slot(&mut self, request_id: &R) -> Result<(), SlotError> {
        self.slots.remove(request_id).ok_or(SlotError::NotFound)?;
        Ok(())
    }
}
