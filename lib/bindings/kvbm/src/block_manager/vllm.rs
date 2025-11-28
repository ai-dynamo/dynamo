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
        BasicMetadata, DeviceStorage, Storage,
        block::{
            BlockId, ImmutableBlock, MutableBlock,
            data::logical::distributed_leader_worker::DistributedLeaderWorkerResources,
            locality::{LocalityProvider, Logical},
        },
        pool::{BlockPool, BlockPoolError},
    },
    tokens::{SaltHash, SequenceHash, TokenBlockSequence, Tokens},
};

// use crate::block_manager::BlockManager as PyBlockManager;
use crate::block_manager::BlockManager as PyBlockManager;
use crate::block_manager::VllmBlockManager;

use crate::to_pyerr;

mod block_list;
mod connector;
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

    m.add_class::<connector::worker::PyKvConnectorWorker>()?;
    m.add_class::<connector::leader::PyKvConnectorLeader>()?;
    m.add_class::<connector::SchedulerOutput>()?;
    // TODO: use TRTLLM own integration module
    m.add_class::<connector::trtllm_worker::PyTrtllmKvConnectorWorker>()?;
    m.add_class::<connector::trtllm_leader::PyTrtllmKvConnectorLeader>()?;
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
    #[tracing::instrument(level = "debug", skip(self))]
    pub fn num_computed_tokens(&self, request_id: String) -> PyResult<usize> {
        let slot_manager = self.slot_manager.lock().map_err(to_pyerr)?;
        slot_manager
            .num_tokens(&request_id, SlotPosition::Computed)
            .map_err(to_pyerr)
    }

    /// Get the computed blocks for the given sequence hashes.
    /// This is used to get the blocks for the request.
    #[tracing::instrument(level = "debug", skip(self), ret)]
    pub fn get_computed_blocks(
        &self,
        sequence_hashes: Vec<SequenceHash>,
    ) -> PyResult<KvbmBlockList> {
        // Unfortunately, we cannot associate the sequence hashes with the request ID due to the calling
        // structure of the vLLM scheduler.

        let blocks = self
            .block_manager()
            .device()
            .unwrap()
            .match_sequence_hashes_blocking(&sequence_hashes)
            .map_err(to_pyerr)?;

        Ok(KvbmBlockList::new(BlockListType::ImmutableDevice(blocks)))
    }

    /// Get the number of matched tokens that can be loaded from the external connector.
    /// This is used to implement the `get_num_new_matched_tokens` in the vLLM Connector API.
    ///
    /// Note: we unpack the id and the num_tokens from the vLLM `Request` so we can hold state
    /// in the slot manager as well as determine if the matches are on full block boundaries.
    pub fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> PyResult<(usize, bool)> {
        let mut slot_manager = self.slot_manager.lock().map_err(to_pyerr)?;
        slot_manager
            .get_num_new_matched_tokens(
                &request_id,
                request_num_tokens,
                num_computed_tokens,
                self.block_manager(),
            )
            .map_err(to_pyerr)
    }

    /// Updates the slot manager with the current request state and allocates new blocks if needed.
    /// Returns the new blocks if they were allocated, otherwise returns None.
    #[tracing::instrument(level = "debug", skip(self), fields(update = ?update), ret)]
    pub fn allocate_slots(&self, update: SlotUpdate) -> PyResult<Option<BlockStates>> {
        self.slot_manager
            .lock()
            .map_err(to_pyerr)?
            .update_slot(update.dissolve(), self.block_manager())
            .map_err(to_pyerr)
    }

    pub fn free(&self, request_id: String) -> PyResult<()> {
        let mut slot_manager = self.slot_manager.lock().map_err(to_pyerr)?;
        slot_manager.free_blocks(&request_id);
        Ok(())
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
        let mut slot_manager = self.slot_manager.lock().map_err(to_pyerr)?;
        slot_manager.drop_slot(&request_id);
        Ok(())
    }

    pub fn take_events(&self) -> PyResult<Vec<KvCacheEvent>> {
        // we don't need events
        Ok(vec![])
    }

    pub fn get_block_ids(&self, request_id: String) -> PyResult<Vec<BlockId>> {
        Ok(self
            .slot_manager
            .lock()
            .map_err(to_pyerr)?
            .get_block_ids(&request_id)
            .inspect_err(|e| match e {
                SlotError::NotFound => {
                    tracing::warn!(request_id, "slot was never allocated for this request");
                }
                _ => {
                    tracing::error!(request_id, "failed to get block ids: {:?}", e);
                }
            })
            .unwrap_or_default())
    }

    pub fn usage(&self) -> PyResult<f64> {
        let pool = self.block_manager().device().unwrap();
        let inuse = pool.total_blocks() - pool.available_blocks();
        let usage: f64 = inuse as f64 / pool.total_blocks() as f64;
        Ok(usage)
    }

    pub fn trigger_onboard(&self, request_id: String) -> PyResult<()> {
        self.slot_manager
            .lock()
            .map_err(to_pyerr)?
            .trigger_onboard(&request_id, self.block_manager())
            .map_err(to_pyerr)
    }

    pub fn reset_prefix_cache(&self) -> bool {
        match self._reset_prefix_cache() {
            Ok(_) => true,
            Err(e) => {
                tracing::error!("failed to reset prefix cache: {:?}", e);
                false
            }
        }
    }
}

impl KvbmCacheManager {
    #[tracing::instrument(level = "debug", skip(self), ret)]
    fn _reset_prefix_cache(&self) -> Result<(), SlotError> {
        let manager = self.block_manager();

        if let Some(disk) = manager.disk() {
            disk.reset_blocking()?;
            tracing::debug!("reset disk prefix cache");
        }

        if let Some(host) = manager.host() {
            host.reset_blocking()?;
            tracing::debug!("reset host prefix cache");
        }

        if let Some(device) = manager.device() {
            device.reset_blocking()?;
            tracing::debug!("reset device prefix cache");
        }

        Ok(())
    }
}

#[derive(Clone, Dissolve)]
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

impl std::fmt::Debug for GenericSlotUpdate<String> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tokens_display = if self.tokens_to_append.len() > 8 {
            format!(
                "[{:?}...{:?}]",
                &self.tokens_to_append[..3],
                &self.tokens_to_append[self.tokens_to_append.len() - 3..]
            )
        } else {
            format!("{:?}", self.tokens_to_append)
        };

        write!(
            f,
            "GenericSlotUpdate(request_id: {}, request_num_tokens: {}, request_num_computed_tokens: {}, tokens_to_append: {}, num_new_tokens: {}, num_new_computed_tokens: {:?}, new_computed_blocks: {:?}, num_lookahead_blocks: {:?}, delay_cache_blocks: {:?})",
            self.request_id,
            self.request_num_tokens,
            self.request_num_computed_tokens,
            tokens_display,
            self.num_new_tokens,
            self.num_new_computed_tokens,
            self.new_computed_blocks,
            self.num_lookahead_blocks,
            self.delay_cache_blocks
        )
    }
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

    #[error(transparent)]
    BlockPoolError(#[from] BlockPoolError),
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
    #[tracing::instrument(level = "debug", skip_all, fields(request_id = %request_id))]
    pub fn create_slot(
        &mut self,
        request_id: &R,
        salt_hash: SaltHash,
        tokens: Vec<u32>,
    ) -> Result<Vec<SequenceHash>, SlotError> {
        if !self.slots.contains_key(request_id) {
            self.slots.insert(
                request_id.clone(),
                Slot::new(tokens.into(), self.block_size, salt_hash),
            );
            tracing::debug!(
                request_id,
                "created slot; total slots: {}",
                self.slots.len()
            );
        }

        let slot = self.slots.get(request_id).ok_or(SlotError::NotFound)?;
        Ok(slot.sequence_hashes(SlotPosition::All))
    }

    #[tracing::instrument(level = "debug", skip_all, fields(request_id = %update.request_id))]
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
            match matched_blocks.take_blocks() {
                Some(BlockListType::ImmutableDevice(blocks)) => {
                    tracing::debug!(
                        request_id,
                        "applying {} cache-hit tokens",
                        blocks.len() * self.block_size
                    );
                    slot.initialize_with_device_matches(blocks)?;
                }
                _ => {
                    panic!("logic error: block list was not immutable device");
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

    #[tracing::instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub fn free_blocks(&mut self, request_id: &R) {
        if let Some(slot) = self.slots.get_mut(request_id) {
            slot.free_blocks();
        } else {
            // Request ID may not be found if the client aborts the request.
            tracing::debug!(
                request_id,
                "request id {} not found in the slot manager",
                request_id
            );
        }
    }

    #[tracing::instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub fn drop_slot(&mut self, request_id: &R) {
        match self.slots.remove(request_id) {
            Some(slot) => {
                let isl = slot.num_tokens(SlotPosition::Prefill);
                let isl_device = slot.num_blocks_cached_from_device() * self.block_size;
                let isl_host = slot.num_blocks_cached_from_host() * self.block_size;
                let isl_disk = slot.num_blocks_cached_from_disk() * self.block_size;
                tracing::info!(
                    request_id,
                    "request complete isl: {} - cache hits: device: {}, host: {}, disk: {} - prefilled: {}",
                    isl,
                    isl_device,
                    isl_host,
                    isl_disk,
                    isl - (isl_device + isl_host + isl_disk)
                );
            }
            None => {
                tracing::debug!(
                    request_id,
                    "request id {} not found in the slot manager during drop",
                    request_id
                );
            }
        }
    }

    #[tracing::instrument(level = "debug", skip(self, block_manager), ret)]
    pub fn get_num_new_matched_tokens(
        &mut self,
        request_id: &R,
        request_num_tokens: usize,
        num_computed_tokens: usize,
        block_manager: &VllmBlockManager,
    ) -> Result<(usize, bool), SlotError> {
        let slot = self.slots.get_mut(request_id).ok_or(SlotError::NotFound)?;

        // the number of device matched tokens should be less than or equal to the number of tokens in the request
        assert!(num_computed_tokens <= request_num_tokens);

        // early exit if we cannot match full block
        if (request_num_tokens - num_computed_tokens) < self.block_size {
            return Ok((0, false));
        }

        // num_computed_tokens represents the number of tokens already on the device
        // this much be a multiple of the block size
        let num_device_blocks = num_computed_tokens / self.block_size;
        debug_assert_eq!(num_computed_tokens % self.block_size, 0);

        // get the sequence hashes for the device matched tokens
        let sequence_hashes = slot.sequence_hashes(SlotPosition::All);
        assert!(sequence_hashes.len() >= num_device_blocks);

        if let Some(host) = block_manager.host() {
            host.touch_blocks_blocking(&sequence_hashes)?;
        }

        if let Some(disk) = block_manager.disk() {
            disk.touch_blocks_blocking(&sequence_hashes)?;
        }

        // we start matching non-device blocks after the device blocks
        let search_offset = num_device_blocks;

        let mut host_blocks = block_manager
            .host()
            .map(|host| host.match_sequence_hashes_blocking(&sequence_hashes[search_offset..]))
            .transpose()?
            .unwrap_or_default();

        let num_matched_host_blocks = host_blocks.len();

        // advance the search offset by the number of matched host blocks
        let search_offset = search_offset + num_matched_host_blocks;

        // start at host offset
        let mut disk_blocks = block_manager
            .disk()
            .map(|disk| disk.match_sequence_hashes_blocking(&sequence_hashes[search_offset..]))
            .transpose()?
            .unwrap_or_default();

        let num_matched_disk_blocks = disk_blocks.len();

        let num_matched_blocks = num_matched_host_blocks + num_matched_disk_blocks;

        tracing::debug!(
            "matched {} host blocks and {} disk blocks; {} total blocks",
            num_matched_host_blocks,
            num_matched_disk_blocks,
            num_matched_blocks
        );

        // early exit if we did not match any blocks
        if num_matched_blocks == 0 {
            return Ok((0, false));
        }

        let mut num_new_matched_tokens = num_matched_blocks * self.block_size;

        // we are on a block boundary, so we need to throw away the last block
        if num_computed_tokens + num_new_matched_tokens == request_num_tokens {
            tracing::debug!(
                request_id,
                "on a block boundary, throwing away the last block"
            );

            // we should have matched at least one block
            assert!(!host_blocks.is_empty() || !disk_blocks.is_empty());

            // pop from disk, or if there are none, then from host
            if disk_blocks.is_empty() {
                host_blocks.pop();
            } else {
                disk_blocks.pop();
            }

            // decrement the number of new matched tokens by the block size
            num_new_matched_tokens -= self.block_size;
        }

        slot.store_onboard_blocks(host_blocks, disk_blocks);

        Ok((num_new_matched_tokens, false))
    }

    #[tracing::instrument(level = "debug", skip(self, block_manager), ret)]
    pub fn trigger_onboard(
        &mut self,
        request_id: &R,
        block_manager: &VllmBlockManager,
    ) -> Result<(), SlotError> {
        let slot = self.slots.get_mut(request_id).ok_or(SlotError::NotFound)?;
        slot.trigger_onboard(block_manager)?;
        Ok(())
    }
}
