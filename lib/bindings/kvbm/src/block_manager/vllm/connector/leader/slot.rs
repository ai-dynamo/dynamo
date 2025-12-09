// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{any::Any, cmp::max, sync::Arc};

use dynamo_llm::{
    block_manager::{
        BlockPool, NixlRegisterableStorage, PinnedStorage, Storage,
        block::{BasicMetadata, BlockMetadata, MutableBlock, locality::LocalityProvider},
        config::should_bypass_cpu_cache,
        connector::protocol::{LeaderTransferRequest, RequestType, TransferType},
        distributed::{BlockTransferPool, BlockTransferRequest, KvbmLeader, ObjectStorageConfig},
        v2::logical::ObjectRegistry,
    },
    tokens::TokenBlock,
};
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use tokio_util::sync::CancellationToken;

use crate::block_manager::cache_stats::CacheStatsTracker;
use crate::{get_current_cancel_token, get_current_tokio_handle};

use super::*;

#[derive(Debug, thiserror::Error)]
pub enum SlotError {
    #[error("slot not found")]
    NotFound,

    #[error("slot is in an invalid state: {0}")]
    InvalidState(String),

    #[error("slot operation failed: {0}")]
    InvalidOperation(String),

    #[error(transparent)]
    BlockPoolError(#[from] BlockPoolError),
}

pub trait SlotManager<R: RequestKey>: Send + Sync {
    type SlotType: Slot + ?Sized;

    fn has_slot(&self, request_id: &R) -> bool;

    /// Create a new slot for the given request ID, initial tokens and salt hash.
    fn create_slot(
        &self,
        request_id: &R,
        tokens: Vec<u32>,
        salt_hash: SaltHash,
    ) -> Result<(), SlotError>;

    fn get_slot(&self, request_id: &R) -> Result<Arc<Mutex<Self::SlotType>>, SlotError>;
    fn remove_slot(&self, request_id: &R) -> Result<(), SlotError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SlotState {
    /// The slot was not scheduled in the previous iteration.
    Initialized,

    /// The slot is prepared to load kv blocks from external storage; however, the onboarding operation
    /// has not been triggered yet. The usize is the number of tokens that are ready for onboarding.
    OnboardStaged(usize),

    /// The slot is actively copying blocks to device storage from some external storage(s).
    /// The usize is the number of tokens that are being onboarded.
    Onboarding(usize),

    /// The slot is actively prefilling the sequence.
    Prefilling,

    /// The slot is skipped prefill.
    SkippedPrefill,

    /// The slot is actively participating in a forward pass which will result in one more more tokens
    /// to be applied to the sequence.
    Decoding,

    /// The slot is skipped decoding.
    SkippedDecode,

    /// The slot is marked as finished, but not all resources have been released.
    Finishing,

    /// The slot is finished and all resources have been released.
    Finished,

    /// The slot is preempted and is waiting for the next iteration to resume.
    Preempted,
}

/// Staging info for object storage blocks, including pre-allocated host bounce buffers.
/// This mirrors how disk staging works - we reserve resources at match time.
pub struct StagedObjectBlocks {
    /// Object keys: (sequence_hash, object_key) for S3 lookup
    pub object_keys: Vec<(u64, u64)>,
    /// Pre-allocated host blocks for bounce buffers (Object → Host → GPU)
    /// These are held until transfer completes, preventing resource contention.
    pub bounce_blocks: Vec<MutableBlock<PinnedStorage, VllmLocality, BasicMetadata>>,
}

impl StagedObjectBlocks {
    pub fn new(object_keys: Vec<(u64, u64)>, bounce_blocks: Vec<MutableBlock<PinnedStorage, VllmLocality, BasicMetadata>>) -> Self {
        Self { object_keys, bounce_blocks }
    }

    pub fn len(&self) -> usize {
        self.object_keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.object_keys.is_empty()
    }

    /// Get bounce block IDs for transfer
    pub fn bounce_block_ids(&self) -> Vec<usize> {
        self.bounce_blocks.iter().map(|b| b.block_id()).collect()
    }
}

#[allow(dead_code)]
pub trait Slot: std::fmt::Debug {
    fn request_id(&self) -> &str;

    fn state(&self) -> SlotState;

    fn sequence(&self) -> &TokenBlockSequence;

    /// The number of tokens that have been computed on the device, i.e. the number of tokens for which we have ownership
    /// of computed kv blocks in the device storage.
    fn computed_tokens(&self) -> usize;

    fn apply_scheduler_output(
        &mut self,
        tokens: &[u32],
        block_ids: &[usize],
        num_computed_tokens: usize,
        num_scheduled_tokens: usize,
    ) -> Result<(), SlotError>;

    // TRT-LLM does not include scheduled tokens in the scheduler output.
    // Ideally, we should have a dedicated implementation for the TRT-LLM slot.
    // However, since only this single function needs to be rewritten for now,
    // we keep it as a separate function in Slot.
    fn apply_scheduler_output_with_computed_position(
        &mut self,
        tokens: &[u32],
        block_ids: &[usize],
        computed_position: usize,
        is_new_request: bool,
    ) -> Result<(), SlotError>;

    fn record_start_iteration(&mut self, iteration: u64) -> Result<(), SlotError>;

    fn mark_as_prefilling(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_decoding(&mut self, iteration: u64) -> Result<(), SlotError>;

    fn mark_as_finished(&mut self, iteration: u64) -> Result<(), SlotError>;

    /// The number of device blocks that have been allocated to the slot.
    fn num_device_blocks_allocated(&self) -> usize;

    /// Find all possible block matches for remaining known tokens in some local storage, i.e. look up and take ownership
    /// of any kv blocks for tokens in the isl that are not already in memory on the device, but on some local storage.
    ///
    /// If external tokens are matched, then the slot will transition to the [`SlotState::Onboarding`] state.
    /// `num_computed_tokens` is the number of tokens that have been computed on the device, this indicated the number of
    /// blocks in the ISL sequence that we should skip before we start looking for matches.
    fn acquire_local_matches(&mut self, num_computed_tokens: usize) -> Result<(), SlotError>;

    /// Trigger the onboarding operation for the slot.
    fn trigger_onboarding(&mut self, num_external_tokens: usize) -> Result<(), SlotError>;

    /// Take all pending operations for the slot.
    fn take_pending_operations(&mut self) -> Option<Vec<WorkerTransferRequest>>;

    /// Record the number of tokens that were cached on the device.
    fn record_cached_device_tokens(&mut self, num_tokens: usize);

    /// Record the number of tokens that were cached on the host.
    fn record_cached_host_tokens(&mut self, num_tokens: usize);

    /// Record the number of tokens that were cached on the disk.
    fn record_cached_disk_tokens(&mut self, num_tokens: usize);

    /// Record the number of tokens that were cached in object storage.
    fn record_cached_object_tokens(&mut self, num_tokens: usize);

    /// Reset the slot after preemption.
    fn reset_after_preemption(&mut self);

    /// Reset the slot.
    fn reset(&mut self);

    /// Get a mutable reference to the slot as a dynamic Any.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait ExternallyManagedDeviceSlot: Slot {
    /// Since we do not control the device pool, nor do we have insight in how the device pool is managed,
    /// we must accept external updates to the computed position.
    fn advance_computed_position(&mut self, num_tokens: usize) -> Result<(), SlotError>;

    /// Append the given block ids to the slot.
    ///
    /// The external device block manager has provided a set of mutable blocks to the slot.
    fn append_mutable_device_blocks(&mut self, block_ids: &[BlockId]) -> Result<(), SlotError>;
}

pub struct ConnectorSlotManager<R: RequestKey> {
    slots: Mutex<HashMap<R, Arc<Mutex<VllmConnectorSlot>>>>,
    block_manager: VllmBlockManager,
    /// use this to issue [`LocalTransferRequest`]s to the transfer engine
    xfer_tx: mpsc::UnboundedSender<LocalTransferRequest>,
    _transfer_engine_handle: Option<CriticalTaskExecutionHandle>,
    /// Cache statistics tracker
    cache_stats: Arc<CacheStatsTracker>,
    /// KVBM metrics for exposing cache hit rates
    kvbm_metrics: KvbmMetrics,
    /// Object storage registry (sequence hash → object key mapping)
    object_registry: Arc<ObjectRegistry>,
}

impl std::fmt::Debug for ConnectorSlotManager<String> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectorSlotManager").finish()
    }
}

impl<R: RequestKey> ConnectorSlotManager<R> {
    pub fn new(
        block_manager: VllmBlockManager,
        leader: Arc<KvbmLeader>,
        kvbm_metrics: KvbmMetrics,
        identifier: Option<String>,
    ) -> Self {
        let cache_stats = Arc::new(CacheStatsTracker::new(identifier));
        let kvbm_metrics_clone = kvbm_metrics.clone();
        let cache_stats_clone = cache_stats.clone();

        // Spawn a background task to periodically update metrics and log cache hit rates
        let handle = get_current_tokio_handle();
        handle.spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
            loop {
                interval.tick().await;
                // Update Prometheus metrics
                let host_rate = cache_stats_clone.host_hit_rate();
                let disk_rate = cache_stats_clone.disk_hit_rate();
                let object_rate = cache_stats_clone.object_hit_rate();
                kvbm_metrics_clone.update_cache_hit_rates(host_rate, disk_rate, object_rate);
                // Also log cache hit rates periodically
                cache_stats_clone.maybe_log();
            }
        });
        tracing::debug!(
            "creating slot manager with block size: {}",
            block_manager.block_size()
        );

        // Create object registry early so it can be shared with transfer engine
        let object_registry = Arc::new(ObjectRegistry::new());

        let (xfer_tx, xfer_rx) = mpsc::unbounded_channel();

        let mut xfer_engine = LocalTransferEngine::new(
            block_manager.clone(),
            leader,
            xfer_rx,
            object_registry.clone(),
        );
        let primary_token = get_current_cancel_token();
        let primary_token_clone = primary_token.clone();
        let runtime_primary = get_current_tokio_handle();
        let runtime_primary_clone = runtime_primary.clone();
        let kvbm_metrics_clone = kvbm_metrics.clone();

        let xfer_engine_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token| async move {
                xfer_engine
                    .execute(
                        cancellation_token,
                        runtime_primary_clone,
                        primary_token_clone,
                        kvbm_metrics_clone,
                    )
                    .await
            },
            primary_token,
            "LocalTransferEngine",
            &runtime_primary,
        )
        .unwrap();

        Self {
            slots: Mutex::new(HashMap::new()),
            block_manager,
            xfer_tx,
            _transfer_engine_handle: Some(xfer_engine_task),
            cache_stats,
            kvbm_metrics: kvbm_metrics.clone(),
            object_registry,
        }
    }

}

impl<R: RequestKey> SlotManager<R> for ConnectorSlotManager<R> {
    type SlotType = dyn ExternallyManagedDeviceSlot;

    fn has_slot(&self, request_id: &R) -> bool {
        self.slots.lock().unwrap().contains_key(request_id)
    }

    fn create_slot(
        &self,
        request_id: &R,
        tokens: Vec<u32>,
        salt_hash: SaltHash,
    ) -> Result<(), SlotError> {
        tracing::debug!(
            "creating slot with request_id: {}, num_tokens: {}",
            request_id,
            tokens.len()
        );
        let slot = VllmConnectorSlot::new(
            request_id.to_string(),
            tokens.into(),
            salt_hash,
            self.block_manager.clone(),
            self.xfer_tx.clone(),
            self.cache_stats.clone(),
            self.object_registry.clone(),
        );
        self.slots
            .lock()
            .unwrap()
            .insert(request_id.clone(), Arc::new(Mutex::new(slot)));
        Ok(())
    }

    fn get_slot(&self, request_id: &R) -> Result<Arc<Mutex<Self::SlotType>>, SlotError> {
        let slots = self.slots.lock().unwrap();
        let slot = slots.get(request_id).ok_or(SlotError::NotFound)?;
        Ok(slot.clone())
    }

    fn remove_slot(&self, request_id: &R) -> Result<(), SlotError> {
        self.slots.lock().unwrap().remove(request_id);
        Ok(())
    }
}

impl<R: RequestKey> Drop for ConnectorSlotManager<R> {
    fn drop(&mut self) {
        if let Some(task) = self._transfer_engine_handle.take() {
            task.cancel();
            task.detach();
        }
    }
}

pub struct VllmConnectorSlot {
    request_id: String,

    /// The state of the slot.
    state: SlotState,

    // /// Current position in the sequence of tokens that have been computed.
    // /// When the slot is initialized, we populate the sequence with the prefill tokens.
    // /// However, those tokens are not yet prefilled, so they are not yet represented
    // /// in the sequence_position.
    // computed_position: usize,
    /// The sequence of token blocks
    sequence: TokenBlockSequence,

    /// The mutable blocks id (device)
    device_blocks: Vec<BlockId>,

    /// Blocks to be onboarded from the host
    /// We must hold these blocks in the slot state until the scheduler trigger the onboarding.
    staging_from_host: Option<Vec<ImmutableBlock<PinnedStorage, VllmLocality, BasicMetadata>>>,

    /// Blocks to be onboarded from the disk
    /// We must hold these blocks in the slot state until the scheduler trigger the onboarding.
    staging_from_disk: Option<Vec<ImmutableBlock<DiskStorage, VllmLocality, BasicMetadata>>>,

    /// Object storage blocks with pre-allocated host bounce buffers.
    /// Like disk, we reserve resources at match time to prevent contention.
    staging_from_object: Option<StagedObjectBlocks>,

    /// The number of blocks cached from the device
    tokens_cached_from_device: usize,

    /// The number of blocks cached from the host
    tokens_cached_from_host: usize,

    /// The number of blocks cached from the disk
    tokens_cached_from_disk: usize,

    /// The number of tokens cached from object storage
    tokens_cached_from_object: usize,

    /// Phantom data to ensure the storage type is correct.
    block_manager: VllmBlockManager,

    block_size: usize,

    iteration_first_scheduled: Option<u64>,

    pending_operations: Option<Vec<WorkerTransferRequest>>,

    /// Tracks whether operations have been sent to the worker.
    /// Used to ensure we return `true` from `mark_as_finished()` even after
    /// `pending_operations` is emptied by `take_pending_operations()`.
    has_sent_operations: bool,

    /// use this to issue [`LocalTransferRequest`]s to the transfer engine
    xfer_tx: mpsc::UnboundedSender<LocalTransferRequest>,

    /// This is the current position for which we are applying some number of active/scheduled tokens.
    /// On application, then we decide what actions we take.
    /// This the point that we will call our generic policy object.
    current_position: usize,

    /// The number of blocks that have been evaluated by the policy.
    /// Each policy evaluation will skip the already evaluated blocks.
    evaluated_blocks: usize,

    /// Whether we actually performed a cache lookup for this request
    performed_cache_lookup: bool,

    /// Total number of blocks queried from host/disk cache
    total_blocks_queried: usize,

    /// Cache statistics tracker for this KVBM instance
    cache_stats: Arc<CacheStatsTracker>,

    /// Object storage registry for S3/GCS block tracking
    object_registry: Arc<ObjectRegistry>,
}

impl VllmConnectorSlot {
    fn new(
        request_id: String,
        tokens: Tokens,
        salt_hash: SaltHash,
        block_manager: VllmBlockManager,
        xfer_tx: mpsc::UnboundedSender<LocalTransferRequest>,
        cache_stats: Arc<CacheStatsTracker>,
        object_registry: Arc<ObjectRegistry>,
    ) -> Self {
        assert!(!tokens.is_empty(), "tokens must be non-empty");
        let block_size = block_manager.block_size();
        debug_assert!(block_size.is_power_of_two() && block_size <= 1024);
        let sequence = TokenBlockSequence::new(tokens, block_size as u32, Some(salt_hash));

        Self {
            request_id,
            sequence,
            block_manager,
            block_size,
            xfer_tx,
            // default values
            state: SlotState::Initialized,
            iteration_first_scheduled: None,
            current_position: 0,
            evaluated_blocks: 0,
            device_blocks: Vec::new(),
            staging_from_host: None,
            staging_from_disk: None,
            staging_from_object: None,
            pending_operations: None,
            has_sent_operations: false,
            tokens_cached_from_device: 0,
            tokens_cached_from_host: 0,
            tokens_cached_from_disk: 0,
            tokens_cached_from_object: 0,
            performed_cache_lookup: false,
            total_blocks_queried: 0,
            cache_stats,
            object_registry,
        }
    }

    fn mark_as_skipped_prefill(&mut self) -> Result<(), SlotError> {
        if self.state != SlotState::Prefilling {
            return Err(SlotError::InvalidState(format!(
                "cannot mark slot as skipped prefill in state {:?}",
                self.state
            )));
        }
        self.state = SlotState::SkippedPrefill;
        Ok(())
    }

    fn mark_as_skipped_decode(&mut self) -> Result<(), SlotError> {
        if self.state != SlotState::Decoding {
            return Err(SlotError::InvalidState(format!(
                "cannot mark slot as skipped decode in state {:?}",
                self.state
            )));
        }
        self.state = SlotState::SkippedDecode;
        Ok(())
    }

    pub fn mark_as_skipped(&mut self) -> Result<(), SlotError> {
        match self.state {
            SlotState::Prefilling => self.mark_as_skipped_prefill(),
            SlotState::Decoding => self.mark_as_skipped_decode(),
            SlotState::SkippedPrefill => Ok(()), // already skipped
            SlotState::SkippedDecode => Ok(()),  // already skipped
            _ => {
                tracing::debug!(
                    "slot is in the {:?} state; will not explicitly mark as skipped, request_id: {}",
                    self.state,
                    self.request_id
                );
                Ok(())
            }
        }
    }
}

impl std::fmt::Debug for VllmConnectorSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VllmConnectorSlot")
            .field("state", &self.state)
            .field("current_position", &self.current_position)
            .field("num_tokens", &self.sequence.total_tokens())
            .finish()
    }
}

impl Slot for VllmConnectorSlot {
    fn request_id(&self) -> &str {
        &self.request_id
    }

    fn state(&self) -> SlotState {
        self.state
    }

    fn reset_after_preemption(&mut self) {
        assert!(self.staging_from_disk.is_none());
        assert!(self.staging_from_host.is_none());
        assert!(self.staging_from_object.is_none());
        assert!(self.pending_operations.is_none());

        self.state = SlotState::Preempted;
        self.iteration_first_scheduled = None;
        self.current_position = 0;
        self.evaluated_blocks = 0;
        self.device_blocks.clear();
        self.tokens_cached_from_device = 0;
        self.tokens_cached_from_host = 0;
        self.tokens_cached_from_disk = 0;
        self.tokens_cached_from_object = 0;
        self.performed_cache_lookup = false;
        self.total_blocks_queried = 0;
    }

    fn reset(&mut self) {
        self.reset_after_preemption();
        self.state = SlotState::Initialized;
    }

    fn mark_as_prefilling(&mut self, _iteration: u64) -> Result<(), SlotError> {
        self.state = SlotState::Prefilling;
        Ok(())
    }

    fn mark_as_decoding(&mut self, _iteration: u64) -> Result<(), SlotError> {
        self.state = SlotState::Decoding;
        Ok(())
    }

    fn record_cached_device_tokens(&mut self, num_tokens: usize) {
        self.tokens_cached_from_device = num_tokens;
        tracing::debug!("recording {} cached device tokens", num_tokens,);
    }

    fn record_cached_host_tokens(&mut self, num_tokens: usize) {
        self.tokens_cached_from_host = num_tokens;
        tracing::debug!("recording {} cached host tokens", num_tokens);
    }

    fn record_cached_disk_tokens(&mut self, num_tokens: usize) {
        self.tokens_cached_from_disk = num_tokens;
        tracing::debug!("recording {} cached disk tokens", num_tokens);
    }

    fn record_cached_object_tokens(&mut self, num_tokens: usize) {
        self.tokens_cached_from_object = num_tokens;
        tracing::debug!("recording {} cached object storage tokens", num_tokens);
    }

    #[tracing::instrument(level = "debug", skip_all, fields(request_id = self.request_id.as_str()))]
    fn apply_scheduler_output(
        &mut self,
        tokens: &[u32],
        block_ids: &[BlockId],
        num_computed_tokens: usize,
        num_scheduled_tokens: usize,
    ) -> Result<(), SlotError> {
        if !tokens.is_empty() {
            tracing::debug!(
                "appending {} newly decoded tokens to sequence",
                tokens.len()
            );
            self.state = SlotState::Decoding;
            self.sequence.extend(tokens.into()).unwrap();
        } else {
            self.state = SlotState::Prefilling;
        }

        // Use max to advance both current_position and evaluated_blocks at least by num_computed_tokens.
        // This logic is to prevent redundant block offloading.
        self.current_position = max(self.current_position, num_computed_tokens);
        self.evaluated_blocks = max(self.evaluated_blocks, num_computed_tokens / self.block_size);

        // apply new block_ids
        if !block_ids.is_empty() {
            tracing::debug!("assigning {} new device blocks slot", block_ids.len());
            self.device_blocks.extend(block_ids);
        }

        // we should have enough device blocks to cover the newly scheduled tokens
        let next_position = self.current_position + num_scheduled_tokens;
        assert!(
            next_position <= self.device_blocks.len() * self.block_size,
            "next_position: {} > device_blocks.len() {} * block_size {}",
            next_position,
            self.device_blocks.len(),
            self.block_size
        );

        if next_position > self.sequence.total_tokens() {
            // vllm stopped providing tokens, so we are done
            self.state = SlotState::Decoding;
            tracing::debug!(
                "connector source stopped providing tokens; no further evaluation possible"
            );
            return Ok(());
        }

        // now we decide what we should do from the current position to the num_scheduled_tokens
        tracing::debug!(
            "applying kv cache policy at current_position: {}; num_scheduled_tokens: {}; num_evaluated_blocks: {}",
            self.current_position,
            num_scheduled_tokens,
            self.evaluated_blocks
        );

        // TODO(ryan) - apply policy
        let next_position = self.current_position + num_scheduled_tokens;

        debug_assert!(next_position / self.block_size >= self.evaluated_blocks);

        let num_candidate_blocks = (next_position / self.block_size) - self.evaluated_blocks;

        tracing::debug!(
            "evaluating policy with the following parameters: state: {:?}; current_position: {}; num_candidate_blocks: {}; num_scheduled_tokens: {}",
            self.state,
            self.current_position,
            num_candidate_blocks,
            num_scheduled_tokens
        );

        if num_candidate_blocks != 0 {
            // do we have a mechanism for skipping gpu cache hit blocks?  not sure yet.
            // for now, offload all the blocks to the host
            let offload_block_ids: Vec<usize> = self
                .device_blocks
                .iter()
                .skip(self.evaluated_blocks)
                .take(num_candidate_blocks)
                .copied()
                .collect::<Vec<_>>();

            assert_eq!(
                offload_block_ids.len(),
                num_candidate_blocks,
                "device block overflow - candidate blocks exceed block count at offset {}",
                self.evaluated_blocks
            );

            let offload_token_blocks: Vec<TokenBlock> = self
                .sequence
                .blocks()
                .iter()
                .skip(self.evaluated_blocks)
                .take(num_candidate_blocks)
                .cloned()
                .collect::<Vec<_>>();

            self.offload_blocks(&offload_block_ids, &offload_token_blocks)
                .expect("failed to offload blocks");

            self.evaluated_blocks += num_candidate_blocks;
        }

        // done applying policy
        tracing::debug!(
            "done applying kv cache policy at current_position: {}; num_scheduled_tokens: {}",
            self.current_position,
            num_scheduled_tokens
        );

        // advance current and computed position
        self.current_position += num_scheduled_tokens;

        Ok(())
    }

    #[tracing::instrument(level = "debug", skip_all, fields(request_id = self.request_id.as_str()))]
    fn apply_scheduler_output_with_computed_position(
        &mut self,
        tokens: &[u32],
        block_ids: &[usize],
        computed_position: usize,
        is_new_request: bool,
    ) -> Result<(), SlotError> {
        // TRTLLM's KV Connector Manager will have (computed_position - external matches)
        // in onborading case
        if computed_position < self.current_position {
            tracing::debug!(
                "computed_position={} < current_position={}, so we are onboarding during prefilling phase",
                computed_position,
                self.current_position
            );
            return Ok(());
        }

        // now we decide what we should do for the new computed tokens
        tracing::debug!(
            "applying scheduler output, computed_position={}, sequence_total_tokens={}",
            computed_position,
            self.sequence.total_tokens()
        );

        if computed_position < self.sequence.total_tokens() {
            // no need to apply new tokens, since it's applied when created the slot during prefilling
            self.state = SlotState::Prefilling;
        } else {
            tracing::debug!(
                "appending {} newly decoded tokens to sequence",
                tokens.len()
            );
            self.sequence.extend(tokens.into()).unwrap();
            self.state = SlotState::Decoding;
        }

        // apply new block_ids, this should be applied for both prefilling and decoding
        // because this is unknown when creating the slot
        if !block_ids.is_empty() {
            tracing::debug!("assigning {} new device blocks slot", block_ids.len());
            self.device_blocks.extend(block_ids);
        }

        // This approach is fragile, but it’s the only way currently to skip evaluating
        // the device matched blocks and to avoid offloading them again.
        // TODO: Consider adding an indicator in the scheduler output to distinguish between
        // matched and unmatched device blocks/tokens from the scheduler.
        let maybe_have_device_matched_blocks =
            is_new_request && computed_position > 0 && self.evaluated_blocks == 0;

        if maybe_have_device_matched_blocks {
            self.evaluated_blocks = (computed_position + 1) / self.block_size;
        }

        let num_candidate_blocks =
            ((computed_position + 1) / self.block_size).saturating_sub(self.evaluated_blocks);

        if num_candidate_blocks > 0 {
            // do we have a mechanism for skipping gpu cache hit blocks?  not sure yet.
            // for now, offload all the blocks to the host
            let offload_block_ids: Vec<usize> = self
                .device_blocks
                .iter()
                .skip(self.evaluated_blocks)
                .take(num_candidate_blocks)
                .copied()
                .collect::<Vec<_>>();

            assert_eq!(
                offload_block_ids.len(),
                num_candidate_blocks,
                "device block overflow - candidate blocks exceed block count at offset {}",
                self.evaluated_blocks
            );

            let offload_token_blocks: Vec<TokenBlock> = self
                .sequence
                .blocks()
                .iter()
                .skip(self.evaluated_blocks)
                .take(num_candidate_blocks)
                .cloned()
                .collect::<Vec<_>>();

            self.offload_blocks(&offload_block_ids, &offload_token_blocks)
                .expect("failed to offload blocks");

            self.evaluated_blocks += num_candidate_blocks;
        }

        // done applying policy
        tracing::debug!(
            "done applying kv cache policy at current_position: {}; computed_position: {}",
            self.current_position,
            computed_position,
        );

        // advance current position to computed position
        self.current_position = computed_position;

        Ok(())
    }

    fn record_start_iteration(&mut self, iteration: u64) -> Result<(), SlotError> {
        if self.iteration_first_scheduled.is_none() {
            self.iteration_first_scheduled = Some(iteration);
        }
        Ok(())
    }

    fn mark_as_finished(&mut self, _iteration: u64) -> Result<(), SlotError> {
        // Report cache statistics if we performed a cache lookup
        if self.performed_cache_lookup {
            let block_size = self.block_size;

            // Convert cached tokens to blocks (rounding up)
            let host_blocks = (self.tokens_cached_from_host + block_size - 1) / block_size;
            let disk_blocks = (self.tokens_cached_from_disk + block_size - 1) / block_size;
            let object_blocks = (self.tokens_cached_from_object + block_size - 1) / block_size;

            tracing::debug!(
                request_id = %self.request_id,
                "Reporting cache stats: host_blocks={}, disk_blocks={}, object_blocks={}, total_blocks_queried={}, tokens_from_host={}, tokens_from_disk={}, tokens_from_object={}",
                host_blocks,
                disk_blocks,
                object_blocks,
                self.total_blocks_queried,
                self.tokens_cached_from_host,
                self.tokens_cached_from_disk,
                self.tokens_cached_from_object
            );

            self.cache_stats
                .record(host_blocks, disk_blocks, object_blocks, self.total_blocks_queried);
        }

        // Check if there are any pending operations (unsent) OR if operations were sent to worker
        let has_pending_ops = self
            .pending_operations
            .as_ref()
            .map(|ops| !ops.is_empty())
            .unwrap_or(false);

        // Operations are "in flight" if we have unsent operations OR we've already sent some to the worker.
        // The worker is responsible for tracking completion of sent operations.
        let has_inflight_operations = has_pending_ops || self.has_sent_operations;

        if has_inflight_operations {
            // There are in-flight operations - need to wait for worker to complete them
            self.state = SlotState::Finishing;
            tracing::debug!(
                request_id = %self.request_id,
                pending_operations = self.pending_operations.as_ref().map(|v| v.len()).unwrap_or(0),
                has_sent_operations = self.has_sent_operations,
                "request set to finish (with in-flight operations): cached_gpu_tokens: {}; cached_host_tokens: {}; cached_disk_tokens: {}; cached_object_tokens: {}",
                self.tokens_cached_from_device,
                self.tokens_cached_from_host,
                self.tokens_cached_from_disk,
                self.tokens_cached_from_object
            );
        } else {
            // No pending or in-flight operations - can immediately mark as finished
            self.state = SlotState::Finished;
            tracing::debug!(
                request_id = %self.request_id,
                "request set to finished (no in-flight operations): cached_gpu_tokens: {}; cached_host_tokens: {}; cached_disk_tokens: {}; cached_object_tokens: {}",
                self.tokens_cached_from_device,
                self.tokens_cached_from_host,
                self.tokens_cached_from_disk,
                self.tokens_cached_from_object
            );
        }
        Ok(())
    }

    fn sequence(&self) -> &TokenBlockSequence {
        &self.sequence
    }

    fn computed_tokens(&self) -> usize {
        self.current_position
    }

    fn num_device_blocks_allocated(&self) -> usize {
        self.device_blocks.len()
    }

    fn take_pending_operations(&mut self) -> Option<Vec<WorkerTransferRequest>> {
        let ops = self.pending_operations.take();
        if ops.is_some() {
            // Mark that we've sent operations to the worker.
            // This ensures mark_as_finished() returns Finishing state
            // so worker can properly track completion.
            self.has_sent_operations = true;
        }
        ops
    }

    #[tracing::instrument(level = "debug", skip_all)]
    fn acquire_local_matches(&mut self, num_computed_tokens: usize) -> Result<(), SlotError> {
        if matches!(self.state(), SlotState::OnboardStaged(_)) {
            tracing::debug!("slot is already in the OnboardStaged state; skipping lookup");
            return Ok(());
        }

        if !matches!(self.state(), SlotState::Initialized | SlotState::Preempted) {
            return Err(SlotError::InvalidOperation(format!(
                "slot must be in the NotScheduled or Preempted state to acquire local matches; got {:?}",
                self.state()
            )));
        }

        if matches!(self.state(), SlotState::Preempted) {
            tracing::info!("slot is in the Preempted state; we get another chance to match");
        }

        let block_size = self.block_manager.block_size();
        let num_computed_blocks = num_computed_tokens / block_size;
        debug_assert!(num_computed_tokens % block_size == 0);

        let sequence_hashes = self
            .sequence()
            .blocks()
            .iter()
            .map(|b| b.sequence_hash())
            .collect::<Vec<_>>();

        // we start matching non-device blocks after the device blocks
        let search_offset = num_computed_blocks;

        // Calculate how many blocks we're querying from host/disk
        let blocks_to_lookup = &sequence_hashes[search_offset..];

        tracing::debug!("matching against {} block hashes", blocks_to_lookup.len());

        // If there are no blocks to lookup (GPU has everything), return early
        if blocks_to_lookup.is_empty() {
            tracing::debug!(
                request_id = %self.request_id,
                "no blocks to lookup from host/disk; GPU has all blocks"
            );
            // Still mark that we performed a lookup (even though we didn't need to query)
            self.performed_cache_lookup = true;
            self.total_blocks_queried = 0;
            return Ok(());
        }

        // Mark that we're performing a cache lookup and track the total blocks
        self.performed_cache_lookup = true;
        self.total_blocks_queried = blocks_to_lookup.len();

        tracing::debug!(
            request_id = %self.request_id,
            "Starting cache lookup: querying {} blocks from host/disk (num_computed_blocks={})",
            blocks_to_lookup.len(),
            num_computed_blocks
        );

        // Limit host pool matching to leave headroom for bounce buffers
        // Reserve blocks = 2 * batch_size (enough for concurrent transfers)
        let bounce_reserve = dynamo_llm::block_manager::offload::max_transfer_batch_size() * 2;
        let host_available = self
            .block_manager
            .host()
            .map(|h| h.available_blocks() as usize)
            .unwrap_or(0);

        // Only match blocks if we have more than the reserve
        let max_host_match = host_available.saturating_sub(bounce_reserve);
        let host_lookup_limit = blocks_to_lookup.len().min(max_host_match);

        if host_lookup_limit < blocks_to_lookup.len() {
            tracing::debug!(
                request_id = %self.request_id,
                requested = blocks_to_lookup.len(),
                limit = host_lookup_limit,
                available = host_available,
                reserve = bounce_reserve,
                "Limiting host pool matching to preserve bounce buffer headroom"
            );
        }

        let host_lookup_slice = &blocks_to_lookup[..host_lookup_limit];

        let mut host_blocks = self
            .block_manager
            .host()
            .filter(|_| !host_lookup_slice.is_empty())
            .map(|host| host.match_sequence_hashes_blocking(host_lookup_slice))
            .transpose()?
            .unwrap_or_default();

        let num_matched_host_blocks = host_blocks.len();

        self.record_cached_host_tokens(num_matched_host_blocks * block_size);

        // advance the search offset by the number of matched host blocks
        let search_offset = search_offset + num_matched_host_blocks;

        // start at host offset
        let mut disk_blocks = self
            .block_manager
            .disk()
            .map(|disk| disk.match_sequence_hashes_blocking(&sequence_hashes[search_offset..]))
            .transpose()?
            .unwrap_or_default();

        let num_matched_disk_blocks = disk_blocks.len();
        self.record_cached_disk_tokens(num_matched_disk_blocks * block_size);

        // advance the search offset by the number of matched disk blocks
        let search_offset = search_offset + num_matched_disk_blocks;

        // Object storage onboarding is disabled - offload only mode.
        // Blocks offloaded to S3 are persisted but not retrieved for cache hits.

        // Limit object matches to transfer batch size
        // Object→GPU transfers use host pool blocks as bounce buffers
        let blocks_per_batch = dynamo_llm::block_manager::offload::max_transfer_batch_size();

        let object_lookup_slice = &sequence_hashes[search_offset..];
        let object_lookup_limit = object_lookup_slice.len().min(blocks_per_batch);

        if object_lookup_limit < object_lookup_slice.len() {
            tracing::debug!(
                request_id = %self.request_id,
                requested = object_lookup_slice.len(),
                limit = object_lookup_limit,
                blocks_per_batch = blocks_per_batch,
                "Limiting object storage matching to batch size"
            );
        }

        // Check object storage for remaining blocks (limited to batch size)
        let object_matches = self
            .object_registry
            .match_sequence_hashes(&object_lookup_slice[..object_lookup_limit]);

        let num_matched_object_blocks = object_matches.len();
        self.record_cached_object_tokens(num_matched_object_blocks * block_size);

        // let object_matches: Vec<(u64, u64)> = Vec::new();
        // let num_matched_object_blocks = 0;
        // let _ = search_offset; // silence unused variable warning

        let num_matched_blocks =
            num_matched_host_blocks + num_matched_disk_blocks + num_matched_object_blocks;

        tracing::debug!(
            "successfully matched {} host, {} disk, {} object blocks; {} total",
            num_matched_host_blocks,
            num_matched_disk_blocks,
            num_matched_object_blocks,
            num_matched_blocks
        );

        // early exit if we did not match any blocks
        if num_matched_blocks == 0 {
            return Ok(());
        }

        let mut num_new_matched_tokens = num_matched_blocks * block_size;
        let mut object_matches = object_matches;

        // we are on a block boundary, so we need to throw away the last block
        if (num_computed_tokens + num_new_matched_tokens) == self.sequence().total_tokens() {
            tracing::debug!("on a block boundary, throwing away the last block");

            // we should have matched at least one block
            assert!(
                !host_blocks.is_empty() || !disk_blocks.is_empty() || !object_matches.is_empty()
            );

            // pop from object first, then disk, then host
            if !object_matches.is_empty() {
                object_matches.pop();
            } else if !disk_blocks.is_empty() {
                disk_blocks.pop();
            } else {
                host_blocks.pop();
            }

            // decrement the number of new matched tokens by the block size
            num_new_matched_tokens -= block_size;
        }

        // early exit if we need to onboard 0 blocks (after potentially dropping the last block)
        if num_new_matched_tokens == 0 {
            return Ok(());
        }

        self.staging_from_host = if !host_blocks.is_empty() {
            Some(host_blocks)
        } else {
            None
        };
        self.staging_from_disk = if !disk_blocks.is_empty() {
            Some(disk_blocks)
        } else {
            None
        };

        // Store object matches for later - actual host block allocation happens during transfer
        // No blocking allocation here to avoid stalling the vLLM scheduler
        self.staging_from_object = if !object_matches.is_empty() {
            tracing::debug!(
                request_id = %self.request_id,
                num_object_matches = object_matches.len(),
                "Staged {} object blocks for onboarding (host blocks will be allocated during transfer)",
                object_matches.len()
            );
            // Store object matches without pre-allocated bounce blocks
            // Bounce blocks will be allocated from host pool during the actual transfer
            Some(StagedObjectBlocks::new(object_matches, Vec::new()))
        } else {
            None
        };

        self.state = SlotState::OnboardStaged(num_new_matched_tokens);

        Ok(())
    }

    fn trigger_onboarding(&mut self, num_external_tokens: usize) -> Result<(), SlotError> {
        if !matches!(self.state(), SlotState::OnboardStaged(_)) {
            return Err(SlotError::InvalidOperation(format!(
                "slot must be in the OnboardStaged state to trigger onboarding; got {:?}",
                self.state()
            )));
        }

        debug_assert_eq!(self.evaluated_blocks, 0);
        debug_assert_eq!(self.current_position % self.block_size, 0);
        debug_assert_eq!(num_external_tokens % self.block_size, 0);

        let num_computed_blocks = self.current_position / self.block_size;

        // shift the evaluated blocks position to the end of the computed/cached blocks
        self.evaluated_blocks = num_computed_blocks;

        // match the host / disk blocks to the newly assigned mutable device blocks
        if let Some(host_blocks) = self.staging_from_host.take() {
            let num_host_blocks = host_blocks.len();

            // get device block ids
            let dst_block_ids = self
                .device_blocks
                .iter()
                .skip(self.evaluated_blocks)
                .take(num_host_blocks)
                .copied()
                .collect::<Vec<_>>();

            debug_assert_eq!(dst_block_ids.len(), num_host_blocks);

            // construct offload requests - transfer engine + worker
            let src_blocks = Box::new(AnyImmutableBlocks::<PinnedStorage, _, _>::new(host_blocks));

            self.onboard_blocks(src_blocks, dst_block_ids)?;

            // shift the evaluated blocks position to the end of the computed/cached blocks
            self.evaluated_blocks += num_host_blocks;
        }

        if let Some(disk_blocks) = self.staging_from_disk.take() {
            let num_disk_blocks = disk_blocks.len();

            // get device block ids
            let dst_block_ids = self
                .device_blocks
                .iter()
                .skip(self.evaluated_blocks)
                .take(num_disk_blocks)
                .copied()
                .collect::<Vec<_>>();

            debug_assert_eq!(dst_block_ids.len(), num_disk_blocks);

            // construct offload requests - transfer engine + worker
            let src_blocks = Box::new(AnyImmutableBlocks::<DiskStorage, _, _>::new(disk_blocks));

            self.onboard_blocks(src_blocks, dst_block_ids)?;

            // shift the evaluated blocks position to the end of the computed/cached blocks
            self.evaluated_blocks += num_disk_blocks;
        }

        if let Some(staged) = self.staging_from_object.take() {
            let num_object_blocks = staged.len();

            // get device block ids
            let dst_block_ids = self
                .device_blocks
                .iter()
                .skip(self.evaluated_blocks)
                .take(num_object_blocks)
                .copied()
                .collect::<Vec<_>>();

            debug_assert_eq!(dst_block_ids.len(), num_object_blocks);

            // Get pre-allocated bounce block IDs (if any)
            let bounce_block_ids = if !staged.bounce_blocks.is_empty() {
                let ids = staged.bounce_block_ids();
                tracing::debug!(
                    request_id = %self.request_id,
                    num_bounce = ids.len(),
                    "Using pre-allocated bounce blocks for Object→GPU transfer"
                );
                Some(ids)
            } else {
                tracing::debug!(
                    request_id = %self.request_id,
                    "No pre-allocated bounce blocks, will use fallback allocator"
                );
                None
            };

            // Object storage onboarding uses (sequence_hash, object_key) pairs
            let src_blocks = Box::new(ObjectStorageBlocks::new(staged.object_keys));

            self.onboard_blocks_with_bounce(src_blocks, dst_block_ids, bounce_block_ids)?;

            // Keep bounce blocks alive until transfer is submitted
            // (they'll be dropped when staged goes out of scope after this block)
            drop(staged.bounce_blocks);

            // shift the evaluated blocks position to the end of the computed/cached blocks
            self.evaluated_blocks += num_object_blocks;
        }

        self.state = SlotState::Onboarding(num_external_tokens);
        self.advance_computed_position(num_external_tokens)?;

        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ExternallyManagedDeviceSlot for VllmConnectorSlot {
    fn advance_computed_position(&mut self, num_tokens: usize) -> Result<(), SlotError> {
        if self.current_position + num_tokens > self.sequence().total_tokens() {
            return Err(SlotError::InvalidOperation(format!(
                "cannot advance computed position from {} by {num_tokens} tokens, total tokens is {}",
                self.current_position,
                self.sequence().total_tokens()
            )));
        }

        tracing::debug!(
            "advancing computed position by {} tokens from {} to {}",
            num_tokens,
            self.current_position,
            self.current_position + num_tokens
        );

        self.current_position += num_tokens;
        Ok(())
    }

    #[tracing::instrument(level = "debug", skip_all, fields(request_id = self.request_id))]
    fn append_mutable_device_blocks(&mut self, block_ids: &[BlockId]) -> Result<(), SlotError> {
        let count = block_ids.len();
        self.device_blocks.extend(block_ids);
        tracing::debug!(
            "appended {} mutable device blocks to slot; total device blocks: {}",
            count,
            self.num_device_blocks_allocated()
        );

        Ok(())
    }
}

impl VllmConnectorSlot {
    /// this method does two things which are related:
    /// 1. creates transfer engine offload request
    /// 2. creates matching connector worker transfer request
    ///
    /// these requests share the same uuid.
    ///
    /// the worker request triggers the transfer when sufficient forward pass progress has been made.
    fn offload_blocks(
        &mut self,
        block_ids: &[BlockId],
        token_blocks: &[TokenBlock],
    ) -> Result<(), SlotError> {
        // Check if slot is in Finishing state before creating operations
        // If we're finishing, don't create new operations
        if matches!(self.state, SlotState::Finishing | SlotState::Finished) {
            return Ok(());
        }

        assert!(block_ids.len() == token_blocks.len());
        let operation_id = uuid::Uuid::new_v4();

        let xfer_req = LocalTransferRequest::Offload(LocalOffloadRequest::new(
            self.request_id.clone(),
            block_ids.to_vec(),
            token_blocks.to_vec(),
            operation_id,
        ));

        let worker_req = WorkerTransferRequest {
            request_id: self.request_id.clone(),
            uuid: operation_id,
            transfer_type: TransferType::Store,
            request_type: RequestType::Scheduled,
        };

        if let Err(e) = self.xfer_tx.send(xfer_req) {
            tracing::error!("Failed to send transfer request: {:?}", e);
            return Err(SlotError::InvalidOperation(format!(
                "Transfer engine unavailable: {}; aborting offload",
                e
            )));
        }

        self.append_pending_operation(worker_req);

        tracing::debug!(
            request_id = self.request_id,
            operation_id = %operation_id,
            "offloading {} blocks to host",
            block_ids.len()
        );

        Ok(())
    }

    fn onboard_blocks(
        &mut self,
        src_blocks: Box<dyn AnyBlocks>,
        dst_block_ids: Vec<BlockId>,
    ) -> Result<(), SlotError> {
        debug_assert_eq!(src_blocks.len(), dst_block_ids.len());

        let num_blocks = src_blocks.len();
        let src_storage_pool = src_blocks.storage_pool();
        let operation_id = uuid::Uuid::new_v4();

        let xfer_req = LocalTransferRequest::Onboard(LocalOnboardRequest::new(
            self.request_id.clone(),
            src_blocks,
            dst_block_ids,
            operation_id,
        ));

        let worker_req = WorkerTransferRequest {
            request_id: self.request_id.clone(),
            uuid: operation_id,
            transfer_type: TransferType::Load,
            request_type: RequestType::Immediate,
        };

        if let Err(e) = self.xfer_tx.send(xfer_req) {
            tracing::error!("Failed to send transfer request: {:?}", e);
            return Err(SlotError::InvalidOperation(format!(
                "Transfer engine unavailable: {}; aborting offload",
                e
            )));
        }

        self.append_pending_operation(worker_req);

        tracing::debug!(
            request_id = self.request_id,
            operation_id = %operation_id,
            "start onboarding {} blocks from {:?} to device",
            num_blocks,
            src_storage_pool,
        );

        Ok(())
    }

    /// Onboard blocks with pre-allocated bounce buffers (for Object→GPU transfers)
    fn onboard_blocks_with_bounce(
        &mut self,
        src_blocks: Box<dyn AnyBlocks>,
        dst_block_ids: Vec<BlockId>,
        bounce_block_ids: Option<Vec<BlockId>>,
    ) -> Result<(), SlotError> {
        debug_assert_eq!(src_blocks.len(), dst_block_ids.len());

        let num_blocks = src_blocks.len();
        let src_storage_pool = src_blocks.storage_pool();
        let operation_id = uuid::Uuid::new_v4();
        let has_bounce_blocks = bounce_block_ids.is_some();

        let xfer_req = LocalTransferRequest::Onboard(LocalOnboardRequest::with_bounce_blocks(
            self.request_id.clone(),
            src_blocks,
            dst_block_ids,
            operation_id,
            bounce_block_ids,
        ));

        let worker_req = WorkerTransferRequest {
            request_id: self.request_id.clone(),
            uuid: operation_id,
            transfer_type: TransferType::Load,
            request_type: RequestType::Immediate,
        };

        if let Err(e) = self.xfer_tx.send(xfer_req) {
            tracing::error!("Failed to send transfer request: {:?}", e);
            return Err(SlotError::InvalidOperation(format!(
                "Transfer engine unavailable: {}; aborting onboard",
                e
            )));
        }

        self.append_pending_operation(worker_req);

        tracing::debug!(
            request_id = self.request_id,
            operation_id = %operation_id,
            has_bounce_blocks = has_bounce_blocks,
            "start onboarding {} blocks from {:?} to device (with bounce)",
            num_blocks,
            src_storage_pool,
        );

        Ok(())
    }

    fn append_pending_operation(&mut self, operation: WorkerTransferRequest) {
        if let Some(pending_operations) = self.pending_operations.as_mut() {
            pending_operations.push(operation);
        } else {
            self.pending_operations = Some(vec![operation]);
        }
    }
}

enum LocalTransferRequest {
    Offload(LocalOffloadRequest),
    Onboard(LocalOnboardRequest),
}

struct LocalOffloadRequest {
    request_id: String,
    block_ids: Vec<BlockId>,
    token_blocks: Vec<TokenBlock>,
    operation_id: uuid::Uuid,
}

impl LocalOffloadRequest {
    pub fn new(
        request_id: String,
        block_ids: Vec<BlockId>,
        token_blocks: Vec<TokenBlock>,
        operation_id: uuid::Uuid,
    ) -> Self {
        debug_assert!(block_ids.len() == token_blocks.len());
        Self {
            request_id,
            block_ids,
            token_blocks,
            operation_id,
        }
    }
}

struct LocalOnboardRequest {
    request_id: String,
    src_blocks: Box<dyn AnyBlocks>,
    dst_block_ids: Vec<BlockId>,
    operation_id: uuid::Uuid,
    /// Pre-allocated bounce block IDs from host pool (for Object→GPU transfers)
    bounce_block_ids: Option<Vec<BlockId>>,
}

impl LocalOnboardRequest {
    pub fn new(
        request_id: String,
        src_blocks: Box<dyn AnyBlocks>,
        dst_block_ids: Vec<BlockId>,
        operation_id: uuid::Uuid,
    ) -> Self {
        debug_assert!(src_blocks.len() == dst_block_ids.len());
        Self {
            request_id,
            src_blocks,
            dst_block_ids,
            operation_id,
            bounce_block_ids: None,
        }
    }

    pub fn with_bounce_blocks(
        request_id: String,
        src_blocks: Box<dyn AnyBlocks>,
        dst_block_ids: Vec<BlockId>,
        operation_id: uuid::Uuid,
        bounce_block_ids: Option<Vec<BlockId>>,
    ) -> Self {
        debug_assert!(src_blocks.len() == dst_block_ids.len());
        Self {
            request_id,
            src_blocks,
            dst_block_ids,
            operation_id,
            bounce_block_ids,
        }
    }
}

// ============================================================================
// Timing and Filtering Helpers
// ============================================================================

/// Extension trait for Duration to simplify timing logs.
trait DurationExt {
    /// Convert duration to milliseconds as f64.
    fn as_ms(&self) -> f64;
}

impl DurationExt for std::time::Duration {
    fn as_ms(&self) -> f64 {
        self.as_secs_f64() * 1000.0
    }
}

/// Result of filtering blocks that are already in object storage.
struct FilteredOffloadBlocks {
    /// Block IDs that need to be offloaded (not already in object storage)
    block_ids: Vec<BlockId>,
    /// Token blocks corresponding to block_ids
    token_blocks: Vec<TokenBlock>,
    /// Sequence hashes for the filtered blocks (computed once, reused)
    sequence_hashes: Vec<u64>,
    /// Count of blocks that were already in object storage
    already_offloaded_count: usize,
}

impl FilteredOffloadBlocks {
    /// Returns true if there are no blocks to offload
    fn is_empty(&self) -> bool {
        self.block_ids.is_empty()
    }

    /// Number of blocks to offload
    fn len(&self) -> usize {
        self.block_ids.len()
    }

    /// Create block pairs for transfer: (device_block_id, object_key)
    /// For object storage, the "destination block ID" is the sequence hash.
    fn to_block_pairs(&self) -> Vec<(usize, usize)> {
        self.block_ids
            .iter()
            .zip(self.sequence_hashes.iter())
            .map(|(&src, &hash)| (src, hash as usize))
            .collect()
    }
}

/// Filter out blocks that are already in object storage.
///
/// Returns the filtered blocks along with pre-computed sequence hashes
/// and count of already-offloaded blocks.
fn filter_offload_blocks(
    offload_req: &LocalOffloadRequest,
    object_registry: &ObjectRegistry,
    request_id: &str,
    operation_id: &uuid::Uuid,
) -> FilteredOffloadBlocks {
    let capacity = offload_req.block_ids.len();
    let mut block_ids = Vec::with_capacity(capacity);
    let mut token_blocks = Vec::with_capacity(capacity);
    let mut sequence_hashes = Vec::with_capacity(capacity);
    let mut already_offloaded_count = 0usize;

    for (block_id, token_block) in offload_req
        .block_ids
        .iter()
        .zip(offload_req.token_blocks.iter())
    {
        let sequence_hash = token_block.sequence_hash();
        if object_registry.contains(sequence_hash) {
            already_offloaded_count += 1;
            tracing::debug!(
                target: "object_transfer_timing",
                request_id = request_id,
                operation_id = %operation_id,
                sequence_hash = sequence_hash,
                "Skipping block - already in object storage"
            );
        } else {
            block_ids.push(*block_id);
            token_blocks.push(token_block.clone());
            sequence_hashes.push(sequence_hash);
        }
    }

    FilteredOffloadBlocks {
        block_ids,
        token_blocks,
        sequence_hashes,
        already_offloaded_count,
    }
}

// ============================================================================
// Transfer Context and Helpers
// ============================================================================

/// Context for transfer operations, providing common state and helper methods.
struct TransferContext<'a> {
    request_id: String,
    operation_id: uuid::Uuid,
    leader: &'a Arc<KvbmLeader>,
    start_time: std::time::Instant,
}

impl<'a> TransferContext<'a> {
    fn new(request_id: String, operation_id: uuid::Uuid, leader: &'a Arc<KvbmLeader>) -> Self {
        Self {
            request_id,
            operation_id,
            leader,
            start_time: std::time::Instant::now(),
        }
    }

    /// Total elapsed time since context creation.
    fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Mark the operation as complete without performing any transfer.
    /// Use this when the data is already present at the destination (e.g., already offloaded to object storage).
    /// This ensures the scheduler's completion counter is incremented even when no actual transfer occurs.
    async fn mark_already_complete(&self) -> anyhow::Result<()> {
        // Create a minimal request just to trigger the scheduler notification
        let request = BlockTransferRequest {
            from_pool: BlockTransferPool::Device,
            to_pool: BlockTransferPool::Device, // No-op: same pool
            blocks: vec![], // Empty: no blocks to transfer
            connector_req: Some(LeaderTransferRequest {
                request_id: self.request_id.clone(),
                uuid: self.operation_id,
                requirement: None,
                request_type: RequestType::Immediate, // Use Immediate for instant completion
            }),
            bounce_block_ids: None,
        };

        // Send the request which will immediately mark as complete
        let notify = self.leader.transfer_blocks_request(request).await?;
        notify.await.map_err(|_| {
            anyhow::anyhow!("Failed to mark operation as already complete")
        })?;

        Ok(())
    }

    /// Create a BlockTransferRequest with standard connector metadata.
    fn make_request(
        &self,
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
        request_type: RequestType,
    ) -> BlockTransferRequest {
        BlockTransferRequest {
            from_pool,
            to_pool,
            blocks,
            connector_req: Some(LeaderTransferRequest {
                request_id: self.request_id.clone(),
                uuid: self.operation_id,
                requirement: None,
                request_type,
            }),
            bounce_block_ids: None,
        }
    }

    /// Create a BlockTransferRequest with external bounce block IDs.
    /// Use this for Device↔Object transfers when bounce blocks are allocated from host pool.
    fn make_request_with_bounce(
        &self,
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
        request_type: RequestType,
        bounce_block_ids: Vec<usize>,
    ) -> BlockTransferRequest {
        BlockTransferRequest {
            from_pool,
            to_pool,
            blocks,
            connector_req: Some(LeaderTransferRequest {
                request_id: self.request_id.clone(),
                uuid: self.operation_id,
                requirement: None,
                request_type,
            }),
            bounce_block_ids: Some(bounce_block_ids),
        }
    }

    /// Create a BlockTransferRequest WITHOUT connector metadata.
    /// This is for internal intermediate transfers that shouldn't trigger scheduler completion.
    fn make_internal_request(
        &self,
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
    ) -> BlockTransferRequest {
        BlockTransferRequest {
            from_pool,
            to_pool,
            blocks,
            connector_req: None, // No scheduler notification
            bounce_block_ids: None,
        }
    }

    /// Execute an internal transfer request (no scheduler notification).
    /// Used for intermediate transfers in multi-hop operations (e.g., Object→Host in write-through).
    async fn execute_internal(
        &self,
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
    ) -> anyhow::Result<std::time::Duration> {
        let start = std::time::Instant::now();
        let request = self.make_internal_request(from_pool, to_pool, blocks);

        let notify = self.leader.transfer_blocks_request(request).await?;

        // Wait for worker ACK with timeout to prevent indefinite hangs
        match tokio::time::timeout(std::time::Duration::from_secs(60), notify).await {
            Ok(result) => result.map_err(|_| {
                anyhow::anyhow!(
                    "Internal transfer {:?}→{:?} notification failed (channel closed)",
                    from_pool,
                    to_pool
                )
            })?,
            Err(_) => {
                return Err(anyhow::anyhow!(
                    "Internal transfer {:?}→{:?} timed out after 60s",
                    from_pool,
                    to_pool
                ));
            }
        };

        Ok(start.elapsed())
    }

    /// Execute a transfer request and wait for completion.
    async fn execute_transfer(
        &self,
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
        request_type: RequestType,
    ) -> anyhow::Result<std::time::Duration> {
        let start = std::time::Instant::now();
        let request = self.make_request(from_pool, to_pool, blocks, request_type);

        let notify = self.leader.transfer_blocks_request(request).await?;
        notify.await.map_err(|_| {
            anyhow::anyhow!(
                "Transfer {:?}→{:?} notification failed",
                from_pool,
                to_pool
            )
        })?;

        Ok(start.elapsed())
    }

    /// Execute a scheduled transfer (waits for worker scheduling).
    async fn execute_scheduled(
        &self,
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
    ) -> anyhow::Result<std::time::Duration> {
        self.execute_transfer(from_pool, to_pool, blocks, RequestType::Scheduled)
            .await
    }

    /// Execute an immediate transfer (no scheduling delay).
    async fn execute_immediate(
        &self,
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
    ) -> anyhow::Result<std::time::Duration> {
        self.execute_transfer(from_pool, to_pool, blocks, RequestType::Immediate)
            .await
    }

    /// Execute an immediate transfer with external bounce block IDs.
    /// Use this for Device↔Object transfers when bounce blocks are allocated from host pool.
    async fn execute_immediate_with_bounce(
        &self,
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
        bounce_block_ids: Vec<usize>,
    ) -> anyhow::Result<std::time::Duration> {
        let start = std::time::Instant::now();
        let request = self.make_request_with_bounce(
            from_pool,
            to_pool,
            blocks,
            RequestType::Immediate,
            bounce_block_ids,
        );

        let notify = self.leader.transfer_blocks_request(request).await?;
        notify.await.map_err(|_| {
            anyhow::anyhow!(
                "Transfer {:?}→{:?} notification failed",
                from_pool,
                to_pool
            )
        })?;

        Ok(start.elapsed())
    }
}

/// Destination tier for offload operations.
#[derive(Debug, Clone, Copy)]
enum OffloadDestination {
    /// Offload to host memory (G2)
    Host,
    /// Offload directly to disk, bypassing host (G3)
    Disk,
    /// Offload to object storage (G4)
    Object,
    /// Offload to object storage with write-through to host cache
    ObjectWriteThrough,
}

impl OffloadDestination {
    /// Determine the offload destination based on configuration.
    fn from_config() -> Self {
        if ObjectStorageConfig::is_offload_enabled() {
            if ObjectStorageConfig::is_write_through_enabled() {
                Self::ObjectWriteThrough
            } else {
                Self::Object
            }
        } else if should_bypass_cpu_cache() {
            Self::Disk
        } else {
            Self::Host
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Disk => "disk",
            Self::Object => "object_storage",
            Self::ObjectWriteThrough => "object_storage+host_cache",
        }
    }
}

/// Helper to create block pairs from parallel iterators.
fn zip_to_pairs<T, U>(src: impl Iterator<Item = T>, dst: impl Iterator<Item = U>) -> Vec<(usize, usize)>
where
    T: Into<usize>,
    U: Into<usize>,
{
    src.zip(dst).map(|(s, d)| (s.into(), d.into())).collect()
}

struct LocalTransferEngine {
    block_manager: VllmBlockManager,
    leader: Arc<KvbmLeader>,
    xfer_rx: mpsc::UnboundedReceiver<LocalTransferRequest>,
    object_registry: Arc<ObjectRegistry>,
}

impl LocalTransferEngine {
    pub fn new(
        block_manager: VllmBlockManager,
        leader: Arc<KvbmLeader>,
        xfer_rx: mpsc::UnboundedReceiver<LocalTransferRequest>,
        object_registry: Arc<ObjectRegistry>,
    ) -> Self {
        Self {
            block_manager,
            leader,
            xfer_rx,
            object_registry,
        }
    }

    // build an adapted TaskTracker:
    // https://docs.rs/tokio-util/latest/tokio_util/task/task_tracker/struct.TaskTracker.html
    //
    // this should track completions via atomic counters using the dynamo prometheus metrics
    // - critical_tasks: labels - success, failure, cancelled
    //
    // should spawn any task/future that returns either any task that can be converted to a
    // Result<CompletionStatus, String> where CompletionStatus is an enum with Ok and Cancelled.
    // anyhow::Result<()> can be considered non-cancellable and coerced to Ok(CompletionStatus::Ok)
    // tasks allowed to cancel should return a CompletionStatus.
    //
    // This should be a composable unit that we can layer on specialized types of critical tasks
    // with their own sets of custom metrics.
    async fn execute(
        &mut self,
        cancellation_token: CancellationToken,
        task_handle: Handle,
        task_token: CancellationToken,
        kvbm_metrics: KvbmMetrics,
    ) -> anyhow::Result<()> {
        let (onboard_tx, mut onboard_rx) = mpsc::unbounded_channel::<LocalOnboardRequest>();
        let (offload_tx, mut offload_rx) = mpsc::unbounded_channel::<LocalOffloadRequest>();

        // Clone resources needed for tasks
        let block_manager_offload = self.block_manager.clone();
        let block_manager_onboard = self.block_manager.clone();
        let leader_offload = Arc::clone(&self.leader);
        let leader_onboard = Arc::clone(&self.leader);
        let object_registry_offload = Arc::clone(&self.object_registry);
        let object_registry_onboard = Arc::clone(&self.object_registry);

        let kvbm_metrics_onboard = kvbm_metrics.clone();
        let kvbm_metrics_offload = kvbm_metrics.clone();

        let onboard_task = CriticalTaskExecutionHandle::new_with_runtime(
            move |cancellation_token_onboard| async move {
                use futures::stream::{FuturesUnordered, StreamExt};

                // Track in-flight onboard operations for parallel execution
                let mut in_flight: FuturesUnordered<tokio::task::JoinHandle<()>> = FuturesUnordered::new();

                tracing::info!("LocalOnboardTask: starting (unlimited concurrency)");

                loop {
                    tokio::select! {
                        biased;

                        _ = cancellation_token_onboard.cancelled() => {
                        tracing::debug!("LocalOnboardTask: received cancellation signal");
                        break;
                    }

                        // Drain completed futures first (non-blocking)
                        Some(result) = in_flight.next(), if !in_flight.is_empty() => {
                            if let Err(e) = result {
                                tracing::error!("LocalOnboardTask: onboard task panicked: {:?}", e);
                            }
                            tracing::debug!(
                                in_flight = in_flight.len(),
                                "LocalOnboardTask: onboard completed, {} still in flight",
                                in_flight.len()
                            );
                        }

                        // Accept new requests (no concurrency limit)
                        Some(req) = onboard_rx.recv() => {
                            let leader = Arc::clone(&leader_onboard);
                            let block_manager = block_manager_onboard.clone();
                            let object_registry = Arc::clone(&object_registry_onboard);
                            let metrics = kvbm_metrics_onboard.clone();

                            tracing::debug!(
                                in_flight = in_flight.len() + 1,
                                "LocalOnboardTask: spawning onboard, {} will be in flight",
                                in_flight.len() + 1
                            );

                            // Spawn onboard as concurrent task
                            let handle = tokio::spawn(async move {
                                if let Err(e) = process_onboard_request(
                                    req,
                                    &block_manager,
                                    &leader,
                                    &object_registry,
                                    metrics,
                                ).await {
                        tracing::error!("LocalOnboardTask: error processing request: {:?}", e);
                    }
                            });

                            in_flight.push(handle);
                        }

                        else => {
                            // Channel closed or at capacity - wait for in-flight to complete
                            if in_flight.is_empty() {
                                break;
                            }
                        }
                    }
                }

                // Drain remaining in-flight operations
                while let Some(result) = in_flight.next().await {
                    if let Err(e) = result {
                        tracing::error!("LocalOnboardTask: onboard task panicked during drain: {:?}", e);
                    }
                }

                tracing::info!("LocalOnboardTask: shutdown complete");
                Ok(())
            },
            task_token.clone(),
            "LocalOnboardTask",
            &task_handle,
        )
        .unwrap();
        let offload_task = CriticalTaskExecutionHandle::new_with_runtime(
            move |cancellation_token_offload| async move {
                use futures::stream::{FuturesUnordered, StreamExt};

                // Track in-flight offload operations for parallel execution
                let mut in_flight: FuturesUnordered<tokio::task::JoinHandle<()>> = FuturesUnordered::new();

                tracing::info!("LocalOffloadTask: starting (unlimited concurrency)");

                loop {
                    tokio::select! {
                        biased;

                        _ = cancellation_token_offload.cancelled() => {
                        tracing::debug!("LocalOffloadTask: received cancellation signal");
                        break;
                    }

                        // Drain completed futures first (non-blocking)
                        Some(result) = in_flight.next(), if !in_flight.is_empty() => {
                            if let Err(e) = result {
                                tracing::error!("LocalOffloadTask: offload task panicked: {:?}", e);
                            }
                            tracing::debug!(
                                in_flight = in_flight.len(),
                                "LocalOffloadTask: offload completed, {} still in flight",
                                in_flight.len()
                            );
                        }

                        // Accept new requests (no concurrency limit)
                        Some(req) = offload_rx.recv() => {
                    let request_id = req.request_id.clone();
                    let operation_id = req.operation_id;
                            let block_manager = block_manager_offload.clone();
                            let leader = Arc::clone(&leader_offload);
                            let object_registry = Arc::clone(&object_registry_offload);
                            let metrics = kvbm_metrics_offload.clone();

                            tracing::debug!(
                                request_id = %request_id,
                                operation_id = %operation_id,
                                in_flight = in_flight.len() + 1,
                                "LocalOffloadTask: spawning offload, {} will be in flight",
                                in_flight.len() + 1
                            );

                            // Spawn offload as concurrent task (fire-and-forget within the pool)
                            let handle = tokio::spawn(async move {
                    if let Err(e) = process_offload_request(
                        req,
                                    &block_manager,
                                    &leader,
                                    &object_registry,
                                    metrics,
                    )
                    .await
                    {
                                    tracing::error!(
                                        request_id = %request_id,
                                        operation_id = %operation_id,
                                        "LocalOffloadTask: error processing request: {:?}", e
                                    );

                        // Create a fake/immediate transfer request that completes instantly.
                        // Otherwise, worker side might stuck and cause memory leak.
                        let fake_xfer = BlockTransferRequest {
                                        from_pool: BlockTransferPool::Device,
                                        to_pool: BlockTransferPool::Host,
                                        blocks: vec![],
                            connector_req: Some(LeaderTransferRequest {
                                request_id: request_id.clone(),
                                uuid: operation_id,
                                requirement: None,
                                            request_type: RequestType::Immediate,
                            }),
                            bounce_block_ids: None,
                        };

                                    match leader.transfer_blocks_request(fake_xfer).await {
                            Ok(notify_receiver) => {
                                let _ = notify_receiver.await;
                            }
                                        Err(_xfer_err) => {}
                                    }
                                }
                            });

                            in_flight.push(handle);
                        }

                        else => {
                            // Channel closed or at capacity - wait for in-flight to complete
                            if in_flight.is_empty() {
                                break;
                            }
                        }
                    }
                }

                // Drain remaining in-flight operations
                while let Some(result) = in_flight.next().await {
                    if let Err(e) = result {
                        tracing::error!("LocalOffloadTask: offload task panicked during drain: {:?}", e);
                    }
                }

                tracing::info!("LocalOffloadTask: shutdown complete");
                Ok(())
            },
            task_token,
            "LocalOffloadTask",
            &task_handle,
        )
        .unwrap();

        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    tracing::debug!("LocalTransferEngine: received cancellation signal");
                    break;
                }
                req = self.xfer_rx.recv() => {
                    match req {
                        Some(req) => {
                            match req {
                                LocalTransferRequest::Offload(offload_req) => {
                                    if let Err(e) = offload_tx.send(offload_req) {
                                        tracing::error!("LocalTransferEngine: error sending offload request: {:?}", e);
                                    }
                                }
                                LocalTransferRequest::Onboard(onboard_req) => {
                                    if let Err(e) = onboard_tx.send(onboard_req) {
                                        tracing::error!("LocalTransferEngine: error sending onboard request: {:?}", e);
                                    }
                                }
                            }
                        }
                        None => {
                            tracing::debug!("LocalTransferEngine: channel closed");
                            break;
                        }
                    }
                }
            }
        }

        tracing::debug!("LocalTransferEngine: shutting down");

        // drop all tx channels
        drop(onboard_tx);
        drop(offload_tx);

        onboard_task.cancel();
        offload_task.cancel();

        if let Err(e) = onboard_task.join().await {
            tracing::error!("LocalOnboardTask failed: {:?}", e);
        }
        if let Err(e) = offload_task.join().await {
            tracing::error!("LocalOffloadTask failed: {:?}", e);
        }

        tracing::debug!("LocalTransferEngine: shutdown complete");
        Ok(())
    }
}

async fn process_offload_request(
    offload_req: LocalOffloadRequest,
    block_manager: &VllmBlockManager,
    leader: &Arc<KvbmLeader>,
    object_registry: &Arc<ObjectRegistry>,
    kvbm_metrics: KvbmMetrics,
) -> anyhow::Result<()> {
    let ctx = TransferContext::new(
        offload_req.request_id.clone(),
        offload_req.operation_id,
        leader,
    );
    let num_blocks = offload_req.block_ids.len();
    let destination = OffloadDestination::from_config();

    tracing::debug!(
        target: "object_transfer_timing",
        request_id = %ctx.request_id,
        operation_id = %ctx.operation_id,
        num_blocks = num_blocks,
        destination = destination.as_str(),
        "SLOT_OFFLOAD_START: {} blocks to {}",
        num_blocks,
        destination.as_str()
    );

    // Update metrics based on destination
    match destination {
        OffloadDestination::Object | OffloadDestination::ObjectWriteThrough => {
            kvbm_metrics.offload_blocks_d2o.inc_by(num_blocks as u64);
            kvbm_metrics
                .offload_bytes_object
                .inc_by((num_blocks * leader.bytes_per_block()) as u64);
        }
        OffloadDestination::Disk => {
            kvbm_metrics.offload_blocks_d2d.inc_by(num_blocks as u64);
        }
        OffloadDestination::Host => {
            kvbm_metrics.offload_blocks_d2h.inc_by(num_blocks as u64);
        }
    }

    // Execute the appropriate offload path
    match destination {
        OffloadDestination::ObjectWriteThrough | OffloadDestination::Object => {
            process_offload_to_object(offload_req, block_manager, &ctx, object_registry, destination)
                .await?;
        }
        OffloadDestination::Disk => {
            process_offload_to_storage(
                offload_req,
                block_manager.disk().unwrap(),
                BlockTransferPool::Disk,
                &ctx,
            )
            .await?;
        }
        OffloadDestination::Host => {
            process_offload_to_storage(
                offload_req,
                block_manager.host().unwrap(),
                BlockTransferPool::Host,
                &ctx,
            )
            .await?;
        }
    }

    tracing::debug!(
        target: "object_transfer_timing",
        request_id = %ctx.request_id,
        operation_id = %ctx.operation_id,
        num_blocks = num_blocks,
        destination = destination.as_str(),
        total_ms = ctx.elapsed().as_ms(),
        "SLOT_OFFLOAD_DONE: {} blocks to {} in {:.2}ms",
        num_blocks,
        destination.as_str(),
        ctx.elapsed().as_ms()
    );

    Ok(())
}

async fn process_offload_to_storage<S, L, M>(
    offload_req: LocalOffloadRequest,
    storage_pool: &dyn BlockPool<S, L, M>,
    transfer_pool: BlockTransferPool,
    ctx: &TransferContext<'_>,
) -> anyhow::Result<()>
where
    S: Storage + NixlRegisterableStorage,
    L: LocalityProvider,
    M: BlockMetadata,
{
    let storage_name = format!("{:?}", transfer_pool).to_lowercase();
    let num_blocks_requested = offload_req.block_ids.len();

    let blocks = storage_pool
        .allocate_blocks(num_blocks_requested)
        .await?;

    let allocated_ids: Vec<usize> = blocks.iter().map(|b| b.block_id()).collect();
    let block_pairs = zip_to_pairs(
        offload_req.block_ids.iter().copied(),
        allocated_ids.iter().copied(),
    );

    let mut blocks_to_register = Vec::with_capacity(blocks.len());
    for (mut block, token_block) in blocks.into_iter().zip(offload_req.token_blocks.into_iter()) {
        block
            .apply_token_block(token_block)
            .map_err(|e| anyhow::anyhow!("failed to apply token block: {:?}", e))?;
        blocks_to_register.push(block);
    }

    // 2. Execute the transfer
    ctx.execute_scheduled(BlockTransferPool::Device, transfer_pool, block_pairs)
        .await?;

    // 3. Register blocks in the pool
    let registered = storage_pool.register_blocks(blocks_to_register).await?;

    tracing::debug!(
        request_id = %ctx.request_id,
        operation_id = %ctx.operation_id,
        "offload to {}: registered {} blocks",
        storage_name,
        registered.len()
    );

    Ok(())
}

/// Offload blocks to object storage (S3/GCS).
///
/// For `ObjectWriteThrough`: GPU → Host (cached) → Object Storage
/// For `Object`: GPU → Object Storage (direct via bounce buffer)
async fn process_offload_to_object(
    offload_req: LocalOffloadRequest,
    block_manager: &VllmBlockManager,
    ctx: &TransferContext<'_>,
    object_registry: &Arc<ObjectRegistry>,
    destination: OffloadDestination,
) -> anyhow::Result<()> {
    let write_through = matches!(destination, OffloadDestination::ObjectWriteThrough);

    // Filter out blocks already in object storage
    let filtered = filter_offload_blocks(&offload_req, object_registry, &ctx.request_id, &ctx.operation_id);

    if filtered.is_empty() {
        tracing::debug!(
            target: "object_transfer_timing",
            request_id = %ctx.request_id,
            operation_id = %ctx.operation_id,
            already_offloaded = filtered.already_offloaded_count,
            write_through = write_through,
            "OBJ_OFFLOAD_SKIP: all {} blocks already in object storage",
            filtered.already_offloaded_count
        );
        // Even though no transfer is needed, we must still mark the operation as complete
        // so the scheduler increments the completion counter and the worker can proceed.
        ctx.mark_already_complete().await?;
        return Ok(());
    }

    let num_blocks = filtered.len();

    tracing::debug!(
        target: "object_transfer_timing",
        request_id = %ctx.request_id,
        operation_id = %ctx.operation_id,
        num_blocks = num_blocks,
        already_offloaded = filtered.already_offloaded_count,
        write_through = write_through,
        "OBJ_OFFLOAD: {} blocks, {} already present, write_through={}",
        num_blocks,
        filtered.already_offloaded_count,
        write_through
    );

    if write_through {
        // Write-through path: GPU → Host → Object
        let host_pool = block_manager.host().ok_or_else(|| {
            anyhow::anyhow!("Host pool required for write-through but not available")
        })?;

        // Check which blocks already exist in host cache to avoid duplicates
        let existing_host_blocks = host_pool
            .match_sequence_hashes(&filtered.sequence_hashes)
            .await
            .unwrap_or_default();

        let existing_hashes: std::collections::HashSet<u64> = existing_host_blocks
            .iter()
            .map(|b| b.sequence_hash())
            .collect();

        // Partition into: already in host vs need to transfer
        let mut need_transfer_gpu_ids = Vec::new();
        let mut need_transfer_hashes = Vec::new();
        let mut need_transfer_tokens = Vec::new();
        let mut already_in_host_block_ids = Vec::new();
        let mut already_in_host_hashes = Vec::new();

        for i in 0..filtered.len() {
            let hash = filtered.sequence_hashes[i];
            if existing_hashes.contains(&hash) {
                // Block already in host cache - just use it for Host → Object
                if let Some(block) = existing_host_blocks.iter().find(|b| b.sequence_hash() == hash) {
                    already_in_host_block_ids.push(block.block_id());
                    already_in_host_hashes.push(hash);
                }
            } else {
                // Block not in host cache - need GPU → Host transfer
                need_transfer_gpu_ids.push(filtered.block_ids[i]);
                need_transfer_hashes.push(hash);
                need_transfer_tokens.push(filtered.token_blocks[i].clone());
            }
        }

        let num_already_cached = already_in_host_block_ids.len();
        let num_to_transfer = need_transfer_gpu_ids.len();

        tracing::debug!(
            request_id = %ctx.request_id,
            operation_id = %ctx.operation_id,
            already_in_host = num_already_cached,
            need_transfer = num_to_transfer,
            "Write-through offload: {} blocks already in host cache, {} need GPU→Host transfer",
            num_already_cached,
            num_to_transfer
        );

        // Allocate and transfer only blocks not already in host
        let mut new_host_block_ids = Vec::new();
        let mut new_host_hashes = Vec::new();

        if !need_transfer_gpu_ids.is_empty() {

            let host_blocks = host_pool
                .allocate_blocks(num_to_transfer)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to allocate host blocks: {:?}", e))?;

            new_host_block_ids = host_blocks.iter().map(|b| b.block_id()).collect();

            // Apply token metadata to host blocks
            let mut blocks_to_register = Vec::with_capacity(num_to_transfer);
            for (mut block, token_block) in host_blocks.into_iter().zip(need_transfer_tokens.iter()) {
                block
                    .apply_token_block(token_block.clone())
                    .map_err(|e| anyhow::anyhow!("Failed to apply token block: {:?}", e))?;
                blocks_to_register.push(block);
            }

            // GPU → Host (only for blocks not already in host cache)
            let gpu_to_host = zip_to_pairs(
                need_transfer_gpu_ids.iter().copied(),
                new_host_block_ids.iter().copied(),
            );

            // Use execute_internal for intermediate transfer to avoid double-completion signal
            ctx.execute_internal(BlockTransferPool::Device, BlockTransferPool::Host, gpu_to_host)
                .await?;

            // Register new host blocks
            host_pool.register_blocks(blocks_to_register).await?;

            new_host_hashes = need_transfer_hashes;
        }

        // Host → Object: transfer ALL blocks (both already-cached and newly-transferred)
        // Combine the block IDs and hashes
        let all_host_block_ids: Vec<usize> = already_in_host_block_ids
            .iter()
            .chain(new_host_block_ids.iter())
            .copied()
            .collect();
        let all_hashes: Vec<u64> = already_in_host_hashes
            .iter()
            .chain(new_host_hashes.iter())
            .copied()
            .collect();

        if !all_host_block_ids.is_empty() {
            let host_to_obj = zip_to_pairs(
                all_host_block_ids.iter().copied(),
                all_hashes.iter().map(|&h| h as usize),
            );
            ctx.execute_immediate(BlockTransferPool::Host, BlockTransferPool::Object, host_to_obj)
                .await?;
        }
    } else {
        // Direct path: GPU → Object
        ctx.execute_scheduled(
            BlockTransferPool::Device,
            BlockTransferPool::Object,
            filtered.to_block_pairs(),
        )
        .await?;
    }

    // Register sequence hashes in object registry
    for hash in &filtered.sequence_hashes {
        object_registry.register_with_hash_as_key(*hash);
    }

    tracing::debug!(
        target: "object_transfer_timing",
        request_id = %ctx.request_id,
        operation_id = %ctx.operation_id,
        num_blocks = num_blocks,
        write_through = write_through,
        total_ms = ctx.elapsed().as_ms(),
        "OBJ_OFFLOAD: {} blocks in {:.2}ms (write_through={})",
        num_blocks,
        ctx.elapsed().as_ms(),
        write_through
    );

    Ok(())
}

async fn process_onboard_request(
    onboard_req: LocalOnboardRequest,
    block_manager: &VllmBlockManager,
    leader: &Arc<KvbmLeader>,
    _object_registry: &Arc<ObjectRegistry>,
    kvbm_metrics: KvbmMetrics,
) -> anyhow::Result<()> {
    let ctx = TransferContext::new(
        onboard_req.request_id.clone(),
        onboard_req.operation_id,
        leader,
    );
    let num_blocks = onboard_req.src_blocks.len();
    let source_pool = onboard_req.src_blocks.storage_pool();

    // Update metrics based on source
    match source_pool {
        BlockTransferPool::Host => {
            kvbm_metrics.onboard_blocks_h2d.inc_by(num_blocks as u64);
        }
        BlockTransferPool::Disk => {
            kvbm_metrics.onboard_blocks_d2d.inc_by(num_blocks as u64);
        }
        BlockTransferPool::Object => {
            kvbm_metrics.onboard_blocks_o2d.inc_by(num_blocks as u64);
            kvbm_metrics
                .onboard_bytes_object
                .inc_by((num_blocks * leader.bytes_per_block()) as u64);
        }
        _ => {}
    }

    tracing::debug!(
        target: "object_transfer_timing",
        request_id = %ctx.request_id,
        operation_id = %ctx.operation_id,
        num_blocks = num_blocks,
        source = ?source_pool,
        "SLOT_ONBOARD_START: {} blocks from {:?}",
        num_blocks,
        source_pool
    );

    // Use dedicated object storage handler
    if source_pool == BlockTransferPool::Object {
        return process_onboard_from_object(onboard_req, block_manager, &ctx).await;
    }

    // Standard onboard: Source → Device (immediate transfer)
    let block_pairs = zip_to_pairs(
        onboard_req.src_blocks.block_ids().iter().copied(),
        onboard_req.dst_block_ids.iter().copied(),
    );

    let transfer_elapsed = ctx
        .execute_immediate(source_pool, BlockTransferPool::Device, block_pairs)
        .await?;

    tracing::debug!(
        target: "object_transfer_timing",
        request_id = %ctx.request_id,
        operation_id = %ctx.operation_id,
        num_blocks = num_blocks,
        source = ?source_pool,
        transfer_ms = transfer_elapsed.as_ms(),
        total_ms = ctx.elapsed().as_ms(),
        "SLOT_ONBOARD_DONE: {} blocks from {:?} in {:.2}ms",
        num_blocks,
        source_pool,
        ctx.elapsed().as_ms()
    );

    Ok(())
}

/// Onboard blocks from object storage.
///
/// When read-through caching is enabled (via write-through config):
///   - Checks host cache for hits first
///   - For misses: Object → Host (cached) → Device
///   - For hits: Host → Device (fast path)
/// When direct mode:
///   - Simple Object → Device transfer
async fn process_onboard_from_object(
    onboard_req: LocalOnboardRequest,
    block_manager: &VllmBlockManager,
    ctx: &TransferContext<'_>,
) -> anyhow::Result<()> {
    let num_blocks = onboard_req.src_blocks.len();
    let src_block_ids = onboard_req.src_blocks.block_ids();
    let sequence_hashes: Vec<u64> = src_block_ids.iter().map(|&id| id as u64).collect();

    let read_through = ObjectStorageConfig::is_write_through_enabled();

    // Direct path: Object → Device (requires bounce buffer from host pool)
    if !read_through {
        let host_pool = block_manager.host().ok_or_else(|| {
            anyhow::anyhow!("Host pool required for Object→GPU transfers (for bounce buffers)")
        })?;

        // Try to allocate bounce blocks from host pool
        match host_pool.allocate_blocks(num_blocks).await {
            Ok(bounce_blocks) => {
                let bounce_ids: Vec<usize> = bounce_blocks.iter().map(|b| b.block_id()).collect();
                let pairs = zip_to_pairs(
                    sequence_hashes.iter().map(|&h| h as usize),
                    onboard_req.dst_block_ids.iter().copied(),
                );

                tracing::debug!(
                    request_id = %ctx.request_id,
                    num_bounce = bounce_ids.len(),
                    "Allocated bounce blocks from host pool for direct Object→GPU"
                );

                ctx.execute_immediate_with_bounce(
                    BlockTransferPool::Object,
                    BlockTransferPool::Device,
                    pairs,
                    bounce_ids,
                ).await?;

                tracing::debug!(
                    target: "object_transfer_timing",
                    request_id = %ctx.request_id,
                    operation_id = %ctx.operation_id,
                    num_blocks = num_blocks,
                    total_ms = ctx.elapsed().as_ms(),
                    "OBJ_ONBOARD: {} blocks in {:.2}ms (direct)",
                    num_blocks,
                    ctx.elapsed().as_ms()
                );
            }
            Err(e) => {
                tracing::warn!(
                    request_id = %ctx.request_id,
                    error = %e,
                    num_blocks = num_blocks,
                    "Host pool exhausted, cannot onboard from object storage - skipping {} blocks",
                    num_blocks
                );
                // Skip the object transfer - vLLM will recompute these blocks
                // Still mark operation as complete so worker doesn't get stuck
                ctx.mark_already_complete().await?;
            }
        }
        return Ok(());
    }

    // Read-through path: use host cache
    let host_pool = block_manager.host().ok_or_else(|| {
        anyhow::anyhow!("Host pool required for read-through onboard")
    })?;

    // Check host cache for hits
    let match_start = std::time::Instant::now();

    // Add timeout to match_sequence_hashes to prevent hanging on block pool deadlocks
    let cached_blocks = match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        host_pool.match_sequence_hashes(&sequence_hashes)
    ).await {
        Ok(result) => result.unwrap_or_default(),
        Err(_) => {
            // Fallback to empty hits (will force reload)
            Default::default()
        }
    };

    let cache_hits = cached_blocks.len();
    let cache_misses = num_blocks - cache_hits;

    tracing::debug!(
        target: "object_transfer_timing",
        request_id = %ctx.request_id,
        operation_id = %ctx.operation_id,
        num_blocks = num_blocks,
        cache_hits = cache_hits,
        cache_misses = cache_misses,
        elapsed_ms = match_start.elapsed().as_millis(),
        "OBJ_ONBOARD_RT: {} blocks | {} hits, {} misses | match={:.2}ms",
        num_blocks,
        cache_hits,
        cache_misses,
        match_start.elapsed().as_secs_f64() * 1000.0
    );

    // Partition blocks into cache hits vs misses
    let mut hit_pairs = Vec::with_capacity(cache_hits);
    let mut miss_hashes = Vec::with_capacity(cache_misses);
    let mut miss_dst_ids = Vec::with_capacity(cache_misses);

    let mut cached_idx = 0;
    for (i, &src_hash) in sequence_hashes.iter().enumerate() {
        if cached_idx < cached_blocks.len() && cached_blocks[cached_idx].sequence_hash() == src_hash {
            hit_pairs.push((cached_blocks[cached_idx].block_id(), onboard_req.dst_block_ids[i]));
            cached_idx += 1;
        } else {
            miss_hashes.push(src_hash);
            miss_dst_ids.push(onboard_req.dst_block_ids[i]);
        }
    }

    // Handle cache misses: try Object → Host → GPU, fallback to Object → GPU
    let mut miss_host_ids = Vec::new();
    let mut use_direct_fallback = false;

    // Hold host blocks until Host→GPU transfer completes to prevent reallocation
    let mut allocated_host_blocks = None;

    if !miss_hashes.is_empty() {

        match host_pool.allocate_blocks(cache_misses).await {
            Ok(host_blocks) => {

                miss_host_ids = host_blocks.iter().map(|b| b.block_id()).collect();
                // Object → Host
                let obj_to_host = zip_to_pairs(
                    miss_hashes.iter().map(|&h| h as usize),
                    miss_host_ids.iter().copied(),
                );


                // Use execute_internal to avoid triggering scheduler completion for this intermediate step
                ctx.execute_internal(BlockTransferPool::Object, BlockTransferPool::Host, obj_to_host)
                    .await?;

                // Keep blocks allocated until after Host→GPU transfer
                allocated_host_blocks = Some(host_blocks);
            }
            Err(_) => {
                use_direct_fallback = true;
            }
        }
    }

    // Transfer cache hits: Host → GPU
    // Use execute_internal (no scheduler notification) to avoid double-counting.
    // We'll send exactly ONE completion notification at the end of this function.
    if !hit_pairs.is_empty() {
        ctx.execute_internal(BlockTransferPool::Host, BlockTransferPool::Device, hit_pairs)
            .await?;
    }

    // Track whether we've sent a completion notification to scheduler
    let mut completion_sent = false;

    // Transfer cache misses to GPU
    // Hold bounce blocks until transfer completes (if we allocate them)
    let mut allocated_bounce_blocks = None;

    if !miss_hashes.is_empty() {
        if use_direct_fallback {
            // Direct: Object → GPU (requires bounce buffer from host pool)
            let pairs = zip_to_pairs(
                miss_hashes.iter().map(|&h| h as usize),
                miss_dst_ids.iter().copied(),
            );

            // Try to allocate bounce blocks from host pool for staging
            let batch_size = dynamo_llm::block_manager::offload::max_transfer_batch_size();
            let num_bounce_needed = cache_misses.min(batch_size);

            tracing::debug!(
                request_id = %ctx.request_id,
                num_blocks = num_bounce_needed,
                "Allocating bounce blocks from host pool for Object→GPU"
            );

            match host_pool.allocate_blocks(num_bounce_needed).await {
                Ok(bounce_blocks) => {
                    let bounce_ids: Vec<usize> = bounce_blocks.iter().map(|b| b.block_id()).collect();
                    tracing::debug!(
                        request_id = %ctx.request_id,
                        num_blocks = bounce_ids.len(),
                        "Allocated bounce blocks from host pool for Object→GPU: {:?}",
                        bounce_ids
                    );

                    let xfer_start = std::time::Instant::now();
                    // Add timeout to execute_immediate to detect transfer hangs
                    let result = tokio::time::timeout(
                        std::time::Duration::from_secs(60),
                        ctx.execute_immediate_with_bounce(
                            BlockTransferPool::Object,
                            BlockTransferPool::Device,
                            pairs,
                            bounce_ids,
                        )
                    ).await;

                    match result {
                        Ok(Ok(_)) => {
                            tracing::info!(
                                target: "object_transfer_timing",
                                request_id = %ctx.request_id,
                                elapsed_ms = xfer_start.elapsed().as_millis(),
                                "OBJ_DIRECT_XFER_COMPLETE"
                            );
                            completion_sent = true;
                        },
                        Ok(Err(e)) => tracing::error!("Direct transfer failed: {:?}", e),
                        Err(_) => tracing::error!("Direct transfer timed out after 60s"),
                    }

                    // Keep bounce blocks allocated until transfer completes
                    allocated_bounce_blocks = Some(bounce_blocks);
                }
                Err(e) => {
                    // Host pool exhausted - cannot transfer from object storage
                    tracing::warn!(
                        request_id = %ctx.request_id,
                        error = %e,
                        num_blocks = cache_misses,
                        "Host pool exhausted, cannot onboard from object storage - skipping {} blocks",
                        cache_misses
                    );
                    // Skip the object transfer - vLLM will recompute these blocks
                    // Still mark operation as complete so worker doesn't get stuck
                    ctx.mark_already_complete().await?;
                    completion_sent = true;
                }
            }
        } else {
            // Via host: Host → GPU
            let pairs = zip_to_pairs(miss_host_ids.iter().copied(), miss_dst_ids.iter().copied());

            ctx.execute_immediate(BlockTransferPool::Host, BlockTransferPool::Device, pairs)
                .await?;
            completion_sent = true;
        }
    }

    // Now safe to release host blocks back to pool
    drop(allocated_host_blocks);
    drop(allocated_bounce_blocks);

    // Ensure exactly ONE completion notification is sent per operation.
    // If we only had cache hits (no misses), we haven't sent one yet.
    if !completion_sent {
        tracing::debug!(
            request_id = %ctx.request_id,
            operation_id = %ctx.operation_id,
            "Sending completion for cache-hit-only onboard"
        );
        ctx.mark_already_complete().await?;
    }

    tracing::debug!(
        target: "object_transfer_timing",
        request_id = %ctx.request_id,
        operation_id = %ctx.operation_id,
        num_blocks = num_blocks,
        cache_hits = cache_hits,
        cache_misses = cache_misses,
        direct_fallback = use_direct_fallback,
        total_ms = ctx.elapsed().as_ms(),
        "OBJ_ONBOARD_RT: {} blocks | {} hits, {} misses | {:.2}ms",
        num_blocks,
        cache_hits,
        cache_misses,
        ctx.elapsed().as_ms()
    );

    Ok(())
}

// todo move to core lib
pub trait AnyBlocks: Send {
    fn len(&self) -> usize;
    fn storage_pool(&self) -> BlockTransferPool;
    fn block_ids(&self) -> Vec<BlockId>;
}

struct AnyImmutableBlocks<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    blocks: Vec<ImmutableBlock<S, L, M>>,
    storage_pool: BlockTransferPool,
}

impl<L: LocalityProvider, M: BlockMetadata> AnyImmutableBlocks<PinnedStorage, L, M> {
    pub fn new(blocks: Vec<ImmutableBlock<PinnedStorage, L, M>>) -> Self {
        Self {
            blocks,
            storage_pool: BlockTransferPool::Host,
        }
    }
}

impl<L: LocalityProvider, M: BlockMetadata> AnyImmutableBlocks<DiskStorage, L, M> {
    pub fn new(blocks: Vec<ImmutableBlock<DiskStorage, L, M>>) -> Self {
        Self {
            blocks,
            storage_pool: BlockTransferPool::Disk,
        }
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> AnyImmutableBlocks<S, L, M> {
    pub fn storage_pool(&self) -> BlockTransferPool {
        self.storage_pool
    }

    pub fn block_ids(&self) -> Vec<BlockId> {
        self.blocks.iter().map(|b| b.block_id()).collect()
    }

    fn len(&self) -> usize {
        self.blocks.len()
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> AnyBlocks for AnyImmutableBlocks<S, L, M> {
    fn len(&self) -> usize {
        self.len()
    }

    fn storage_pool(&self) -> BlockTransferPool {
        self.storage_pool()
    }

    fn block_ids(&self) -> Vec<BlockId> {
        self.block_ids()
    }
}

/// Object storage blocks for onboarding from S3/GCS.
///
/// Unlike ImmutableBlocks, object storage blocks are represented as
/// (sequence_hash, object_key) pairs since objects are dynamically allocated.
struct ObjectStorageBlocks {
    /// Vec of (sequence_hash, object_key) pairs
    keys: Vec<(u64, u64)>,
}

impl ObjectStorageBlocks {
    pub fn new(keys: Vec<(u64, u64)>) -> Self {
        Self { keys }
    }
}

impl AnyBlocks for ObjectStorageBlocks {
    fn len(&self) -> usize {
        self.keys.len()
    }

    fn storage_pool(&self) -> BlockTransferPool {
        BlockTransferPool::Object
    }

    fn block_ids(&self) -> Vec<BlockId> {
        // For object storage, the "block ID" is the object key
        self.keys.iter().map(|(_hash, key)| *key as BlockId).collect()
    }
}
