// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{any::Any, cmp::max, sync::Arc, time::Duration};

use dynamo_runtime::config::environment_names::kvbm::remote_storage as env_g4;
use once_cell::sync::Lazy;

/// Default maximum concurrent H2O (host-to-object) transfers.
const DEFAULT_MAX_CONCURRENT_H2O: usize = 8;

/// Default timeout in seconds for G4 (remote storage) transfers.
const DEFAULT_G4_TRANSFER_TIMEOUT_SECS: u64 = 30;

/// Maximum concurrent H2O transfers - cached from env var.
static MAX_CONCURRENT_H2O: Lazy<usize> = Lazy::new(|| {
    std::env::var(env_g4::DYN_KVBM_MAX_CONCURRENT_H2O)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_CONCURRENT_H2O)
});

/// Timeout for G4 transfers - cached from env var.
static G4_TRANSFER_TIMEOUT: Lazy<Duration> = Lazy::new(|| {
    let secs: u64 = std::env::var(env_g4::DYN_KVBM_G4_TRANSFER_TIMEOUT_SECS)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_G4_TRANSFER_TIMEOUT_SECS);
    Duration::from_secs(secs)
});

use dynamo_llm::{
    block_manager::{
        BlockPool, NixlRegisterableStorage, Storage,
        block::{BlockMetadata, locality::LocalityProvider},
        config::{RemoteStorageConfig, should_bypass_cpu_cache},
        connector::protocol::{LeaderTransferRequest, RequestType, TransferType},
        distributed::{
            BlockTransferPool, BlockTransferRequest, KvbmLeader, RemoteHashOperationsSync,
            vllm as vllm_int,
        },
        pool::{PinGuard, PinRegistry},
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
    /// Reference to the leader for G4 operations
    leader: Arc<KvbmLeader>,
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

        let (xfer_tx, xfer_rx) = mpsc::unbounded_channel();

        let leader_for_engine = leader.clone();
        let mut xfer_engine =
            LocalTransferEngine::new(block_manager.clone(), leader_for_engine, xfer_rx);
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
            leader,
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
            self.leader.clone(),
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

    /// Sequence hashes to be onboarded from g4 storage
    staging_from_g4: Option<Vec<u64>>,

    /// The number of blocks cached from the device
    tokens_cached_from_device: usize,

    /// The number of blocks cached from the host
    tokens_cached_from_host: usize,

    /// The number of blocks cached from the disk
    tokens_cached_from_disk: usize,

    /// The number of blocks cached from G4 object storage
    tokens_cached_from_g4: usize,

    /// Phantom data to ensure the storage type is correct.
    block_manager: VllmBlockManager,

    block_size: usize,

    iteration_first_scheduled: Option<u64>,

    pending_operations: Option<Vec<WorkerTransferRequest>>,

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

    /// Skip G4 (remote) lookup on retry after a failed onboard.
    /// This prevents infinite retry loops when remote storage has stale registry entries.
    skip_g4_on_retry: bool,

    /// Flag indicating the slot just recovered from a failed transfer.
    /// When true, `apply_scheduler_output` should ignore vLLM's `num_computed_tokens`
    /// since it reflects pre-failure state, not our reset state.
    recovered_from_failed_transfer: bool,

    /// G4 hashes with their positions that were attempted but may have failed.
    /// Used to invalidate stale registry entries when onboard fails.
    /// Stores (sequence_hash, position) pairs where position is the index in the offloaded sequence.
    attempted_g4_hashes: Option<Vec<(u64, u32)>>,

    /// Cache statistics tracker for this KVBM instance
    cache_stats: Arc<CacheStatsTracker>,

    // Reference to the leader for g4 operations
    leader: Arc<KvbmLeader>,
}

impl VllmConnectorSlot {
    fn new(
        request_id: String,
        tokens: Tokens,
        salt_hash: SaltHash,
        block_manager: VllmBlockManager,
        xfer_tx: mpsc::UnboundedSender<LocalTransferRequest>,
        cache_stats: Arc<CacheStatsTracker>,
        leader: Arc<KvbmLeader>,
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
            staging_from_g4: None,
            pending_operations: None,
            tokens_cached_from_device: 0,
            tokens_cached_from_host: 0,
            tokens_cached_from_disk: 0,
            tokens_cached_from_g4: 0,
            performed_cache_lookup: false,
            total_blocks_queried: 0,
            skip_g4_on_retry: false,
            recovered_from_failed_transfer: false,
            attempted_g4_hashes: None,
            cache_stats,
            leader,
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
        // Preemption can happen at any time, including:
        // - After acquire_local_matches staged blocks but before trigger_onboarding
        // - While there are pending transfer operations
        // Using assert! would crash vLLM in production. Instead, gracefully clean up.
        if self.staging_from_disk.is_some() {
            tracing::warn!(request_id = %self.request_id, "Preemption while disk blocks staged");
            self.staging_from_disk.take();
        }
        if self.staging_from_host.is_some() {
            tracing::warn!(request_id = %self.request_id, "Preemption while host blocks staged");
            self.staging_from_host.take();
        }
        if self.staging_from_g4.is_some() {
            tracing::warn!(target: "kvbm-g4", request_id = %self.request_id, "preemption while hashes staged");
            self.staging_from_g4.take();
        }
        if self.pending_operations.is_some() {
            tracing::warn!(
                request_id = %self.request_id,
                pending_ops = self.pending_operations.as_ref().map(|o| o.len()).unwrap_or(0),
                "Preemption while operations pending"
            );
            self.pending_operations.take();
        }

        self.state = SlotState::Preempted;
        self.iteration_first_scheduled = None;
        self.current_position = 0;
        self.evaluated_blocks = 0;
        self.device_blocks.clear();
        self.tokens_cached_from_device = 0;
        self.tokens_cached_from_host = 0;
        self.tokens_cached_from_disk = 0;
        self.tokens_cached_from_g4 = 0;
        self.performed_cache_lookup = false;
        self.total_blocks_queried = 0;
        self.skip_g4_on_retry = false;
        self.attempted_g4_hashes = None;
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

    #[tracing::instrument(level = "debug", skip_all, fields(request_id = self.request_id.as_str()))]
    fn apply_scheduler_output(
        &mut self,
        tokens: &[u32],
        block_ids: &[BlockId],
        num_computed_tokens: usize,
        num_scheduled_tokens: usize,
    ) -> Result<(), SlotError> {
        // Handle recovery from failed onboard prior to processing the scheduler output.
        // When a transfer fails and vLLM reschedules the request, apply_scheduler_output
        // is called BEFORE acquire_local_matches. We need to detect the failed state here
        // and reset, otherwise our stale current_position will cause capacity errors.
        if matches!(self.state, SlotState::Onboarding(_)) {
            tracing::warn!(
                request_id = %self.request_id,
                current_position = self.current_position,
                device_blocks = self.device_blocks.len(),
                num_computed_tokens = num_computed_tokens,
                "Detected Onboarding state in apply_scheduler_output - recovering from failed transfer"
            );
            // Reset slot state for retry.
            // Do not clear device_blocks
            self.current_position = 0;
            self.evaluated_blocks = 0;
            self.tokens_cached_from_device = 0;
            self.tokens_cached_from_host = 0;
            self.tokens_cached_from_disk = 0;
            self.tokens_cached_from_g4 = 0;
            self.performed_cache_lookup = false;
            self.total_blocks_queried = 0;
            self.skip_g4_on_retry = true;
            self.recovered_from_failed_transfer = true;
            self.pending_operations.take();
            self.attempted_g4_hashes.take();
        }

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

        // apply new block_ids
        if !block_ids.is_empty() {
            tracing::debug!("assigning {} new device blocks slot", block_ids.len());
            self.device_blocks.extend(block_ids);
        }

        // After recovery, vLLM's num_computed_tokens reflects pre-failure state.
        // Use our reset position instead.
        let effective_computed_tokens = if self.recovered_from_failed_transfer {
            tracing::info!(
                request_id = %self.request_id,
                vllm_computed_tokens = num_computed_tokens,
                our_position = self.current_position,
                device_blocks = self.device_blocks.len(),
                "Ignoring vLLM's stale num_computed_tokens after recovery"
            );
            self.recovered_from_failed_transfer = false;
            self.current_position
        } else {
            num_computed_tokens
        };

        // Use max to advance both current_position and evaluated_blocks at least by effective_computed_tokens.
        // This logic is to prevent redundant block offloading.
        self.current_position = max(self.current_position, effective_computed_tokens);
        self.evaluated_blocks = max(
            self.evaluated_blocks,
            self.current_position / self.block_size,
        );

        // we should have enough device blocks to cover the newly scheduled tokens
        let next_position = self.current_position + num_scheduled_tokens;
        let capacity = self.device_blocks.len() * self.block_size;
        if next_position > capacity {
            // This can happen when vLLM's state is out of sync with ours (e.g., after recovery).
            // Return an error instead of panicking - vLLM will handle the retry.
            tracing::error!(
                request_id = %self.request_id,
                next_position = next_position,
                capacity = capacity,
                device_blocks = self.device_blocks.len(),
                block_size = self.block_size,
                current_position = self.current_position,
                num_scheduled_tokens = num_scheduled_tokens,
                "Insufficient device blocks for scheduled tokens - state sync issue with vLLM"
            );
            return Err(SlotError::InvalidOperation(format!(
                "Insufficient device blocks: need {} slots but have {} (current_pos={}, scheduled={})",
                next_position, capacity, self.current_position, num_scheduled_tokens
            )));
        }

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
            let object_blocks = (self.tokens_cached_from_g4 + block_size - 1) / block_size;

            tracing::debug!(
                request_id = %self.request_id,
                host_blocks = host_blocks,
                disk_blocks = disk_blocks,
                object_blocks = object_blocks,
                total_blocks_queried = self.total_blocks_queried,
                "Reporting cache stats"
            );

            self.cache_stats.record(
                host_blocks,
                disk_blocks,
                object_blocks,
                self.total_blocks_queried,
            );
        }

        // Check if there are any pending operations
        let has_pending_ops = self
            .pending_operations
            .as_ref()
            .map(|ops| !ops.is_empty())
            .unwrap_or(false);

        if has_pending_ops {
            // There are pending operations - need to wait for them to complete
            self.state = SlotState::Finishing;
            tracing::debug!(
                request_id = %self.request_id,
                pending_operations = self.pending_operations.as_ref().unwrap().len(),
                "request set to finish (with pending operations): cached_gpu_tokens: {}; cached_host_tokens: {}; cached_disk_tokens: {}",
                self.tokens_cached_from_device,
                self.tokens_cached_from_host,
                self.tokens_cached_from_disk
            );
        } else {
            // No pending operations - can immediately mark as finished
            self.state = SlotState::Finished;
            tracing::debug!(
                request_id = %self.request_id,
                "request set to finished (no pending operations): cached_gpu_tokens: {}; cached_host_tokens: {}; cached_disk_tokens: {}",
                self.tokens_cached_from_device,
                self.tokens_cached_from_host,
                self.tokens_cached_from_disk
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
        self.pending_operations.take()
    }

    #[tracing::instrument(level = "debug", skip_all)]
    fn acquire_local_matches(&mut self, num_computed_tokens: usize) -> Result<(), SlotError> {
        if matches!(self.state(), SlotState::OnboardStaged(_)) {
            tracing::debug!("slot is already in the OnboardStaged state; skipping lookup");
            return Ok(());
        }

        // Handle recovery from failed onboard - vLLM rescheduled the request
        if matches!(self.state(), SlotState::Onboarding(_)) {
            // Remove stale G4 hashes from registry to prevent other workers from hitting the same error
            if let Some(stale_hash_positions) = self.attempted_g4_hashes.take() {
                tracing::warn!(
                    target: "kvbm-g4",
                    request_id = %self.request_id,
                    num_stale_hashes = stale_hash_positions.len(),
                    stale_hash_positions = ?stale_hash_positions,
                    "onboard failed - removing stale hashes from registry"
                );

                // Remove stale entries from the registry (fire-and-forget)
                if let Some(handle) = self.leader.remote_handle() {
                    let worker_id = self.leader.worker_id();
                    handle.remove_hashes_with_positions_blocking(&stale_hash_positions, worker_id);
                    tracing::info!(
                        target: "kvbm-g4",
                        request_id = %self.request_id,
                        num_removed = stale_hash_positions.len(),
                        "removed stale hashes from registry"
                    );
                }
            }

            tracing::warn!(
                target: "kvbm-g4",
                request_id = %self.request_id,
                state = ?self.state(),
                "slot in onboarding state during acquire_local_matches; recovering from failed onboard - will skip lookup on retry"
            );
            // Clean up any pending operations from the failed onboard
            let _ = self.pending_operations.take();
            // Reset slot state to allow retry - staging fields should already be None
            // since trigger_onboarding consumed them with .take()
            self.state = SlotState::Preempted;
            self.iteration_first_scheduled = None;
            self.current_position = 0;
            self.evaluated_blocks = 0;
            self.device_blocks.clear();
            self.tokens_cached_from_device = 0;
            self.tokens_cached_from_host = 0;
            self.tokens_cached_from_disk = 0;
            self.tokens_cached_from_g4 = 0;
            self.performed_cache_lookup = false;
            self.total_blocks_queried = 0;
            // Skip G4 (remote) lookup on retry to prevent infinite loops
            // when remote storage has stale registry entries (NoSuchKey errors)
            self.skip_g4_on_retry = true;
            // Mark that we've recovered - next apply_scheduler_output should ignore
            // vLLM's stale num_computed_tokens
            self.recovered_from_failed_transfer = true;
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

        // we should do this opportunistically after this operation is done
        // ideally it was triggered by the match_sequence_hashes_blocking calls directly

        // if let Some(host) = self.block_manager.host() {
        //     host.touch_blocks_blocking(&sequence_hashes)?;
        // }

        // if let Some(disk) = self.block_manager.disk() {
        //     disk.touch_blocks_blocking(&sequence_hashes)?;
        // }

        let mut host_blocks = self
            .block_manager
            .host()
            .map(|host| host.match_sequence_hashes_blocking(blocks_to_lookup))
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

        // Remote registry lookup with TP consensus (G4/object storage)
        let search_offset_g4 = search_offset + num_matched_disk_blocks;
        let mut g4_hashes = if self.skip_g4_on_retry {
            tracing::info!(target: "kvbm-g4", request_id = %self.request_id, "skipping - previous failure");
            vec![]
        } else if let Some(handle) = self.leader.remote_handle() {
            let remaining = &sequence_hashes[search_offset_g4..];
            if !remaining.is_empty() {
                // TP-aware lookup: queries all workers, returns consensus
                let matched = vllm_int::match_prefix_tp_blocking(
                    &handle,
                    remaining,
                    self.leader.world_size(),
                );
                if !matched.is_empty() {
                    tracing::debug!(target: "kvbm-g4", matched = matched.len(), remaining = remaining.len(), "cache hit");
                }
                matched
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        let num_matched_g4_blocks = g4_hashes.len();
        self.tokens_cached_from_g4 = num_matched_g4_blocks * block_size;

        let num_matched_blocks =
            num_matched_host_blocks + num_matched_disk_blocks + num_matched_g4_blocks;

        tracing::debug!(
            "successfully matched {} host, {} disk, {} g4 blocks; {} total blocks",
            num_matched_host_blocks,
            num_matched_disk_blocks,
            num_matched_g4_blocks,
            num_matched_blocks
        );

        // early exit if we did not match any blocks
        if num_matched_blocks == 0 {
            return Ok(());
        }

        let mut num_new_matched_tokens = num_matched_blocks * block_size;

        // we are on a block boundary, so we need to throw away the last block
        if (num_computed_tokens + num_new_matched_tokens) == self.sequence().total_tokens() {
            tracing::debug!("on a block boundary, throwing away the last block");

            // we should have matched at least one block
            assert!(!host_blocks.is_empty() || !disk_blocks.is_empty() || !g4_hashes.is_empty());

            // pop from g4 first, then disk, then host
            if !g4_hashes.is_empty() {
                g4_hashes.pop();
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
        self.staging_from_g4 = if !g4_hashes.is_empty() {
            Some(g4_hashes)
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

        if let Some(g4_hashes) = self.staging_from_g4.take() {
            let num_g4_blocks = g4_hashes.len();

            // get device block ids
            let dst_block_ids = self
                .device_blocks
                .iter()
                .skip(self.evaluated_blocks)
                .take(num_g4_blocks)
                .copied()
                .collect::<Vec<_>>();

            debug_assert_eq!(dst_block_ids.len(), num_g4_blocks);

            // Store hashes with positions for invalidation on failure
            // Positions are 0, 1, 2... relative to the offloaded sequence (matching registration)
            self.attempted_g4_hashes = Some(
                g4_hashes
                    .iter()
                    .enumerate()
                    .map(|(pos, &hash)| (hash, pos as u32))
                    .collect(),
            );

            // G4 onboard uses a different path - sends G4OnboardRequest to worker
            self.onboard_from_g4(g4_hashes, dst_block_ids)?;

            // shift the evaluated blocks position to the end of the computed/cached blocks
            self.evaluated_blocks += num_g4_blocks;
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
            self.block_size,
        ));

        let worker_req = WorkerTransferRequest {
            request_id: self.request_id.clone(),
            uuid: operation_id,
            transfer_type: TransferType::Store,
            request_type: RequestType::Scheduled,
            block_ids: block_ids.to_vec(),
        };

        if let Err(e) = self.xfer_tx.send(xfer_req) {
            tracing::error!("Failed to send transfer request: {:?}", e);
            return Err(SlotError::InvalidOperation(format!(
                "Transfer engine unavailable: {}; aborting offload",
                e
            )));
        }

        self.append_pending_operation(worker_req);

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
            dst_block_ids.clone(),
            operation_id,
        ));

        let worker_req = WorkerTransferRequest {
            request_id: self.request_id.clone(),
            uuid: operation_id,
            transfer_type: TransferType::Load,
            request_type: RequestType::Immediate,
            block_ids: dst_block_ids,
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

    /// Onboard blocks from G4 storage.
    ///
    /// Unlike host/disk onboarding, G4 onboarding sends a G4OnboardRequest
    /// to the worker, which handles the G4->Host->Device transfer atomically.
    fn onboard_from_g4(
        &mut self,
        sequence_hashes: Vec<u64>,
        device_block_ids: Vec<BlockId>,
    ) -> Result<(), SlotError> {
        debug_assert_eq!(sequence_hashes.len(), device_block_ids.len());

        let (params, worker_req) = vllm_int::onboard_from_g4(
            self.request_id.clone(),
            sequence_hashes,
            device_block_ids,
            self.block_size,
        );

        let xfer_req = LocalTransferRequest::Remote(RemoteTransferRequest::from_g4_params(&params));

        self.xfer_tx.send(xfer_req).map_err(|e| {
            tracing::error!(target: "kvbm-g4", "failed to send request: {:?}", e);
            SlotError::InvalidOperation(format!("Transfer engine unavailable: {}", e))
        })?;

        self.append_pending_operation(worker_req);
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
    Remote(RemoteTransferRequest),
}

struct LocalOffloadRequest {
    request_id: String,
    block_ids: Vec<BlockId>,
    token_blocks: Vec<TokenBlock>,
    operation_id: uuid::Uuid,
    sequence_hashes: Vec<u64>,
    block_size: usize,
}

impl LocalOffloadRequest {
    pub fn new(
        request_id: String,
        block_ids: Vec<BlockId>,
        token_blocks: Vec<TokenBlock>,
        operation_id: uuid::Uuid,
        block_size: usize,
    ) -> Self {
        debug_assert!(block_ids.len() == token_blocks.len());
        // Extract sequence hashes from token blocks for G4 write-through
        let sequence_hashes = token_blocks.iter().map(|tb| tb.sequence_hash()).collect();
        Self {
            request_id,
            block_ids,
            token_blocks,
            operation_id,
            sequence_hashes,
            block_size,
        }
    }
}

struct LocalOnboardRequest {
    request_id: String,
    src_blocks: Box<dyn AnyBlocks>,
    dst_block_ids: Vec<BlockId>,
    operation_id: uuid::Uuid,
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
        }
    }
}

struct RemoteTransferRequest {
    request_id: String,
    sequence_hashes: Vec<u64>,
    device_block_ids: Vec<BlockId>,
    host_block_ids: Option<Vec<BlockId>>,
    operation_id: uuid::Uuid,
    block_size: usize,
    is_onboard: bool,
    /// Pin guard ID - when set, the corresponding PinGuard must be released
    /// after the transfer completes. This prevents host blocks from being
    /// evicted during transfers.
    pin_id: Option<uuid::Uuid>,
}

impl RemoteTransferRequest {
    pub fn from_g4_params(params: &vllm_int::G4OnboardParams) -> Self {
        Self {
            request_id: params.request_id.clone(),
            sequence_hashes: params.sequence_hashes.clone(),
            device_block_ids: params.device_block_ids.clone(),
            host_block_ids: None,
            operation_id: params.operation_id,
            block_size: params.block_size,
            is_onboard: true,
            pin_id: None,
        }
    }

    pub fn new_h2o(
        request_id: String,
        sequence_hashes: Vec<u64>,
        host_block_ids: Vec<BlockId>,
        operation_id: uuid::Uuid,
        block_size: usize,
        pin_id: uuid::Uuid,
    ) -> Self {
        debug_assert!(sequence_hashes.len() == host_block_ids.len());
        Self {
            request_id,
            sequence_hashes,
            device_block_ids: vec![],
            host_block_ids: Some(host_block_ids),
            operation_id,
            block_size,
            is_onboard: false,
            pin_id: Some(pin_id),
        }
    }

    pub fn is_h2o(&self) -> bool {
        self.host_block_ids.is_some() && !self.is_onboard
    }
}

struct LocalTransferEngine {
    block_manager: VllmBlockManager,
    leader: Arc<KvbmLeader>,
    xfer_rx: mpsc::UnboundedReceiver<LocalTransferRequest>,
}

impl LocalTransferEngine {
    pub fn new(
        block_manager: VllmBlockManager,
        leader: Arc<KvbmLeader>,
        xfer_rx: mpsc::UnboundedReceiver<LocalTransferRequest>,
    ) -> Self {
        Self {
            block_manager,
            leader,
            xfer_rx,
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
        let (remote_tx, mut remote_rx) = mpsc::unbounded_channel::<RemoteTransferRequest>();

        // Pin registry for preventing host block eviction during H2O transfers.
        // Shared between offload task (creates pins) and remote task (releases pins).
        let pin_registry = PinRegistry::new();
        let pin_registry_offload = pin_registry.clone();
        let pin_registry_remote = pin_registry.clone();

        // Clone resources needed for tasks
        let block_manager_offload = self.block_manager.clone();
        let block_manager_remote = self.block_manager.clone();
        let leader_offload = Arc::clone(&self.leader);
        let leader_onboard = Arc::clone(&self.leader);
        let leader_remote = Arc::clone(&self.leader);

        // Clone remote_tx for the offload task to trigger H2O after D2H
        let remote_tx_for_offload = remote_tx.clone();

        let kvbm_metrics_onboard = kvbm_metrics.clone();
        let kvbm_metrics_offload = kvbm_metrics.clone();
        let kvbm_metrics_remote = kvbm_metrics.clone();

        let onboard_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token_onboard| async move {
                while let Some(req) = onboard_rx.recv().await {
                    if cancellation_token_onboard.is_cancelled() {
                        tracing::debug!("LocalOnboardTask: received cancellation signal");
                        break;
                    }
                    if let Err(e) =
                        process_onboard_request(req, &leader_onboard, kvbm_metrics_onboard.clone())
                            .await
                    {
                        tracing::error!("LocalOnboardTask: error processing request: {:?}", e);
                    }
                }
                Ok(())
            },
            task_token.clone(),
            "LocalOnboardTask",
            &task_handle,
        )
        .unwrap();
        let offload_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token_offload| async move {
                while let Some(req) = offload_rx.recv().await {
                    if cancellation_token_offload.is_cancelled() {
                        tracing::debug!("LocalOffloadTask: received cancellation signal");
                        break;
                    }

                    let request_id = req.request_id.clone();
                    let operation_id = req.operation_id;

                    if let Err(e) = process_offload_request(
                        req,
                        &block_manager_offload,
                        &leader_offload,
                        kvbm_metrics_offload.clone(),
                        &remote_tx_for_offload,
                        &pin_registry_offload,
                    )
                    .await
                    {
                        tracing::error!("LocalOffloadTask: error processing request: {:?}", e);

                        // Create a fake/immediate transfer request that completes instantly.
                        // Otherwise, worker side might stuck and cause memory leak.
                        let fake_xfer = BlockTransferRequest {
                            from_pool: BlockTransferPool::Device, // Use valid Device->Host transfer type
                            to_pool: BlockTransferPool::Host,     // (offload path, but no blocks)
                            blocks: vec![],                       // Empty - nothing to transfer
                            connector_req: Some(LeaderTransferRequest {
                                request_id: request_id.clone(),
                                uuid: operation_id,
                                requirement: None,
                                request_type: RequestType::Immediate, // Immediate = completes instantly
                                chained: false,
                            }),
                            sequence_hashes: None,
                        };

                        match leader_offload.transfer_blocks_request(fake_xfer).await {
                            Ok(notify_receiver) => {
                                // Wait for the fake transfer to "complete" (should be instant)
                                let _ = notify_receiver.await;
                            }
                            Err(_xfer_err) => {
                                // Failed to create completion notification - error already logged above
                            }
                        }
                    }
                }
                Ok(())
            },
            task_token.clone(),
            "LocalOffloadTask",
            &task_handle,
        )
        .unwrap();

        let remote_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token_remote| async move {
                while let Some(req) = remote_rx.recv().await {
                    if cancellation_token_remote.is_cancelled() {
                        tracing::debug!("RemoteTransferTask: received cancellation signal");
                        break;
                    }
                    if let Err(e) = process_remote_transfer_request(
                        req,
                        &block_manager_remote,
                        &leader_remote,
                        kvbm_metrics_remote.clone(),
                        &pin_registry_remote,
                    )
                    .await
                    {
                        tracing::error!("RemoteTransferTask: error processing request: {:?}", e);
                    }
                }
                Ok(())
            },
            task_token,
            "RemoteTransferTask",
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
                                LocalTransferRequest::Remote(remote_req) => {
                                    if let Err(e) = remote_tx.send(remote_req) {
                                        tracing::error!("LocalTransferEngine: error sending remote transfer request: {:?}", e);
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
        drop(remote_tx);

        onboard_task.cancel();
        offload_task.cancel();
        remote_task.cancel();

        if let Err(e) = onboard_task.join().await {
            tracing::error!("LocalOnboardTask failed: {:?}", e);
        }
        if let Err(e) = offload_task.join().await {
            tracing::error!("LocalOffloadTask failed: {:?}", e);
        }
        if let Err(e) = remote_task.join().await {
            tracing::error!("RemoteTransferTask failed: {:?}", e);
        }

        tracing::debug!("LocalTransferEngine: shutdown complete");
        Ok(())
    }
}

async fn process_offload_request(
    offload_req: LocalOffloadRequest,
    block_manager: &VllmBlockManager,
    leader: &Arc<KvbmLeader>,
    kvbm_metrics: KvbmMetrics,
    remote_tx: &mpsc::UnboundedSender<RemoteTransferRequest>,
    pin_registry: &PinRegistry,
) -> anyhow::Result<()> {
    let request_id = offload_req.request_id.clone();
    let operation_id = offload_req.operation_id;

    tracing::debug!(
        "Processing offload request for {} blocks",
        offload_req.block_ids.len()
    );

    // Determine if we should bypass CPU memory (G2) and offload directly from GPU (G1) to Disk (G3)
    let bypass_cpu_mem = should_bypass_cpu_cache();

    if bypass_cpu_mem {
        // Direct G1 -> G3 path (Device to Disk, bypassing Host)
        kvbm_metrics
            .offload_blocks_d2d
            .inc_by(offload_req.block_ids.len() as u64);

        tracing::debug!(
            request_id = %request_id,
            operation_id = %operation_id,
            "offloading directly to disk (bypassing host)"
        );

        process_offload_to_storage(
            offload_req,
            block_manager.disk().unwrap(),
            BlockTransferPool::Disk,
            leader,
            &request_id,
            &operation_id,
            "disk",
            None, // No H2O for disk path
            pin_registry,
        )
        .await?;
    } else {
        kvbm_metrics
            .offload_blocks_d2h
            .inc_by(offload_req.block_ids.len() as u64);

        process_offload_to_storage(
            offload_req,
            block_manager.host().unwrap(),
            BlockTransferPool::Host,
            leader,
            &request_id,
            &operation_id,
            "host",
            Some(remote_tx), // Enable H2O after D2H
            pin_registry,
        )
        .await?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn process_offload_to_storage<S, L, M>(
    offload_req: LocalOffloadRequest,
    storage_pool: &dyn BlockPool<S, L, M>,
    transfer_pool: BlockTransferPool,
    leader: &Arc<KvbmLeader>,
    request_id: &str,
    operation_id: &uuid::Uuid,
    storage_name: &str,
    remote_tx: Option<&mpsc::UnboundedSender<RemoteTransferRequest>>,
    pin_registry: &PinRegistry,
) -> anyhow::Result<()>
where
    S: Storage + NixlRegisterableStorage + 'static,
    L: LocalityProvider + 'static,
    M: BlockMetadata + 'static,
{
    // 1. Acquire mutable blocks
    let blocks = storage_pool
        .allocate_blocks(offload_req.block_ids.len())
        .await?;
    let token_blocks = offload_req.token_blocks;

    let allocated_block_ids: Vec<usize> = blocks.iter().map(|b| b.block_id()).collect();
    let block_pairs: Vec<(usize, usize)> = offload_req
        .block_ids
        .into_iter()
        .zip(allocated_block_ids.into_iter())
        .collect();

    tracing::debug!(
        request_id = request_id,
        operation_id = %operation_id,
        "offload to {}",
        storage_name
    );

    let mut blocks_to_register = Vec::new();
    let zipped_blocks = blocks.into_iter().zip(token_blocks.into_iter());

    for (mut mutable_block, token_block) in zipped_blocks {
        mutable_block
            .apply_token_block(token_block.clone())
            .map_err(|e| anyhow::anyhow!("failed to apply token block: {:?}", e))?;

        blocks_to_register.push(mutable_block);
    }
    tracing::debug!(
        request_id = request_id,
        operation_id = %operation_id,
        "offload to {}",
        storage_name
    );

    let sequence_hashes =
        if transfer_pool == BlockTransferPool::Host && leader.remote_registry_enabled() {
            Some(offload_req.sequence_hashes.clone())
        } else {
            None
        };

    let block_xfer_req = BlockTransferRequest {
        from_pool: BlockTransferPool::Device,
        to_pool: transfer_pool,
        blocks: block_pairs,
        connector_req: Some(LeaderTransferRequest {
            request_id: offload_req.request_id.clone(),
            uuid: offload_req.operation_id,
            requirement: None,
            request_type: RequestType::Scheduled,
            chained: false,
        }),
        sequence_hashes,
    };
    let notify_receiver = leader.transfer_blocks_request(block_xfer_req).await?;

    match notify_receiver.await {
        Ok(_) => {
            tracing::debug!(
                "Offloading transfer to {} completed successfully",
                storage_name
            );
        }
        Err(_) => {
            return Err(anyhow::anyhow!(
                "Offloading transfer completion notification failed"
            ));
        }
    }

    let immutable_blocks = storage_pool.register_blocks(blocks_to_register).await?;

    tracing::debug!(
        request_id = request_id,
        operation_id = %operation_id,
        "registered {} blocks to {}",
        immutable_blocks.len(),
        storage_name
    );

    // Decide if H2O should be triggered (uses lib/llm decision logic)
    let is_host_transfer = transfer_pool == BlockTransferPool::Host;
    let should_h2o = vllm_int::should_trigger_h2o(
        is_host_transfer,
        leader.remote_registry_enabled(),
        pin_registry.len(),
        *MAX_CONCURRENT_H2O,
    );

    if is_host_transfer && leader.remote_registry_enabled() && !should_h2o {
        tracing::warn!(
            request_id = request_id,
            current_h2o = pin_registry.len(),
            max_h2o = *MAX_CONCURRENT_H2O,
            num_blocks = offload_req.sequence_hashes.len(),
            "Skipping H2O transfer due to backpressure"
        );
    }

    if let Some(remote_tx) = remote_tx.filter(|_| should_h2o) {
        let host_block_ids: Vec<BlockId> = immutable_blocks.iter().map(|b| b.block_id()).collect();
        let h2o_operation_id = uuid::Uuid::new_v4();

        // Pin the host blocks to prevent eviction during H2O transfer.
        // The PinGuard holds references to the immutable blocks, keeping
        // their Arc reference count > 0, which prevents them from being
        // returned to the inactive pool and evicted.
        let pin_guard = PinGuard::new(immutable_blocks);
        pin_registry.insert(h2o_operation_id, pin_guard);

        tracing::debug!(
            target: "kvbm-g4",
            request_id = request_id,
            pin_id = %h2o_operation_id,
            num_blocks = host_block_ids.len(),
            "pinned host blocks for h2o transfer"
        );

        let h2o_req = RemoteTransferRequest::new_h2o(
            offload_req.request_id.clone(),
            offload_req.sequence_hashes.clone(),
            host_block_ids,
            h2o_operation_id,
            offload_req.block_size,
            h2o_operation_id,
        );

        tracing::debug!(
            request_id = request_id,
            operation_id = %h2o_operation_id,
            num_blocks = offload_req.sequence_hashes.len(),
            "Triggering H2O transfer after D2H offload"
        );

        if let Err(e) = remote_tx.send(h2o_req) {
            tracing::error!(
                request_id = request_id,
                "Failed to send H2O request: {:?}",
                e
            );
            // Remove the pin since H2O won't happen
            pin_registry.remove(&h2o_operation_id);
        }

        // Note: immutable_blocks ownership moved to pin_guard, which is now in the registry.
        // The guard will be released by process_remote_transfer_request after H2O completes.
        return Ok(());
    }

    // If we didn't trigger H2O, the immutable_blocks are dropped here,
    // allowing them to be returned to the inactive pool normally.
    drop(immutable_blocks);

    Ok(())
}

async fn process_onboard_request(
    onboard_req: LocalOnboardRequest,
    leader: &Arc<KvbmLeader>,
    kvbm_metrics: KvbmMetrics,
) -> anyhow::Result<()> {
    if onboard_req.src_blocks.storage_pool() == BlockTransferPool::Host {
        kvbm_metrics
            .onboard_blocks_h2d
            .inc_by(onboard_req.src_blocks.len() as u64);
    } else if onboard_req.src_blocks.storage_pool() == BlockTransferPool::Disk {
        kvbm_metrics
            .onboard_blocks_d2d
            .inc_by(onboard_req.src_blocks.len() as u64);
    }

    let request_id = &onboard_req.request_id;
    let operation_id = &onboard_req.operation_id;

    let src_block_ids = onboard_req.src_blocks.block_ids();

    let block_pairs = src_block_ids
        .iter()
        .zip(onboard_req.dst_block_ids.iter())
        .map(|(src, dst)| (*src, *dst))
        .collect::<Vec<_>>();

    let block_xfer_req = BlockTransferRequest {
        from_pool: onboard_req.src_blocks.storage_pool(),
        to_pool: BlockTransferPool::Device,
        blocks: block_pairs,
        connector_req: Some(LeaderTransferRequest {
            request_id: request_id.clone(),
            uuid: *operation_id,
            requirement: None,
            request_type: RequestType::Immediate,
            chained: false,
        }),
        sequence_hashes: None,
    };

    let notify_receiver = leader.transfer_blocks_request(block_xfer_req).await?;

    match notify_receiver.await {
        Ok(_) => {
            tracing::debug!("Onboarding transfer completed successfully");
        }
        Err(_) => {
            return Err(anyhow::anyhow!(
                "Onboarding transfer completion notification failed"
            ));
        }
    }

    Ok(())
}

/// Process a remote transfer request (onboard/offload) by sending it to the worker.
async fn process_remote_transfer_request(
    req: RemoteTransferRequest,
    block_manager: &VllmBlockManager,
    leader: &Arc<KvbmLeader>,
    kvbm_metrics: KvbmMetrics,
    pin_registry: &PinRegistry,
) -> anyhow::Result<()> {
    let request_id = &req.request_id;
    let operation_id = &req.operation_id;
    let pin_id = req.pin_id;

    // Helper to release pin guard (called on all exit paths)
    let release_pin = |pin_registry: &PinRegistry, pin_id: Option<uuid::Uuid>| {
        if let Some(guard) = pin_id.and_then(|id| pin_registry.remove(&id)) {
            tracing::debug!(
                pin_id = ?pin_id,
                num_blocks = guard.count(),
                "Released pin guard after H2O transfer"
            );
        }
    };

    // Filter out blocks already in object storage
    // Returns (hash, original_position) pairs to preserve correct positions for registry
    let (hashes_with_positions, filtered_host_ids) = if let Some(handle) = leader.remote_handle() {
        match vllm_int::filter_for_offload(
            &handle,
            &req.sequence_hashes,
            req.host_block_ids.as_deref(),
            leader.worker_id(),
            req.is_onboard,
        )
        .await
        {
            Some(filtered) => filtered,
            None => {
                release_pin(pin_registry, pin_id);
                return Ok(());
            }
        }
    } else {
        // No registry - enumerate all hashes with positions
        let hashes_with_positions: Vec<(u64, u32)> = req
            .sequence_hashes
            .iter()
            .enumerate()
            .map(|(pos, &hash)| (hash, pos as u32))
            .collect();
        (hashes_with_positions, req.host_block_ids.clone())
    };

    let num_blocks = hashes_with_positions.len();

    let direction = if req.is_onboard {
        "onboard"
    } else if req.is_h2o() {
        "h2o"
    } else {
        "offload"
    };
    // Get storage configuration from leader
    let storage_config = leader.remote_storage_config().unwrap_or_else(|| {
        // Fallback to object storage with default bucket if not configured
        tracing::warn!("No remote storage configured, falling back to default object storage");
        RemoteStorageConfig::Object {
            default_bucket: std::env::var("AWS_DEFAULT_BUCKET").ok(),
            endpoint: None,
            region: None,
        }
    });

    let storage_type_str = match &storage_config {
        RemoteStorageConfig::Object { .. } => "object",
        RemoteStorageConfig::Disk { .. } => "disk",
    };

    tracing::debug!(
        target = "kvbm-g4",
        request_id = %request_id,
        operation_id = %operation_id,
        direction = direction,
        storage_type = storage_type_str,
        num_blocks = num_blocks,
        "processing remote {} transfer to {} storage",
        direction,
        storage_type_str
    );

    // Build transfer pipeline
    let hashes: Vec<u64> = hashes_with_positions.iter().map(|&(h, _)| h).collect();
    let (bounce, device) = if req.is_h2o() {
        // H2O: use existing host blocks as bounce buffers
        let bounce = filtered_host_ids
            .ok_or_else(|| anyhow::anyhow!("H2O transfer requires host_block_ids"))?;
        (bounce, vec![])
    } else {
        // Allocate bounce buffers from host pool
        let host_pool = block_manager
            .host()
            .ok_or_else(|| anyhow::anyhow!("Host pool not available for bounce buffers"))?;
        let host_blocks = host_pool.allocate_blocks(num_blocks).await?;
        let bounce = host_blocks.iter().map(|b| b.block_id()).collect();
        let device = req.device_block_ids.iter().copied().collect();
        (bounce, device)
    };
    let pipeline = vllm_int::create_transfer_pipeline(
        &hashes,
        &storage_config,
        req.block_size,
        req.is_onboard,
        req.is_h2o(),
        bounce,
        device,
    );

    // Create wire-format request for ZMQ transport
    //
    // Chained flag determines if this operation increments the completion counter:
    // - H2O (offload to object): chained=true, follows D2H which is already tracked
    // - O2D (onboard from object): chained=false, standalone operation that must be tracked
    let is_chained = !req.is_onboard;

    let wire_req =
        dynamo_llm::block_manager::distributed::RemoteTransferRequest::new_with_connector_req(
            req.request_id.clone(),
            req.operation_id,
            &pipeline,
            LeaderTransferRequest {
                request_id: request_id.clone(),
                uuid: *operation_id,
                requirement: None,
                request_type: RequestType::Immediate,
                chained: is_chained,
            },
        );

    let notify_receiver = leader.remote_transfer_request(wire_req).await?;

    let result = match tokio::time::timeout(*G4_TRANSFER_TIMEOUT, notify_receiver).await {
        Ok(Ok(_)) => {
            tracing::debug!(
                target = "kvbm-g4",
                request_id = %request_id,
                operation_id = %operation_id,
                num_blocks = num_blocks,
                direction = if req.is_onboard { "onboard" } else { "offload" },
                "transfer completed"
            );

            // Track success metrics
            if req.is_onboard {
                kvbm_metrics.onboard_blocks_o2d.inc_by(num_blocks as u64);
            } else {
                kvbm_metrics.offload_blocks_d2o.inc_by(num_blocks as u64);
            }

            // Register with remote registry after successful offload (all TP ranks)
            if !req.is_onboard {
                if let Some(handle) = leader.remote_handle() {
                    vllm_int::register_tp(
                        &handle,
                        &hashes_with_positions,
                        &storage_config,
                        leader.world_size(),
                    )
                    .await;
                }
            }

            Ok(())
        }
        Ok(Err(_)) => {
            tracing::error!(
                request_id = %request_id,
                operation_id = %operation_id,
                num_blocks = num_blocks,
                "Remote transfer completion notification failed"
            );
            // Track failure metrics
            if req.is_onboard {
                kvbm_metrics.record_object_read_failure(num_blocks as u64);
            } else {
                kvbm_metrics.record_object_write_failure(num_blocks as u64);
            }
            Err(anyhow::anyhow!(
                "Remote transfer completion notification failed"
            ))
        }
        Err(_elapsed) => {
            tracing::warn!(
                request_id = %request_id,
                operation_id = %operation_id,
                num_blocks = num_blocks,
                timeout_secs = G4_TRANSFER_TIMEOUT.as_secs(),
                "Remote transfer timed out (adjust with DYN_KVBM_G4_TRANSFER_TIMEOUT_SECS)"
            );
            // Track failure metrics
            if req.is_onboard {
                kvbm_metrics.record_object_read_failure(num_blocks as u64);
            } else {
                kvbm_metrics.record_object_write_failure(num_blocks as u64);
            }
            Err(anyhow::anyhow!(
                "Remote transfer timed out after {} seconds",
                G4_TRANSFER_TIMEOUT.as_secs()
            ))
        }
    };

    // Release pin guard after transfer completes (success or failure).
    // This allows the host blocks to be returned to the inactive pool.
    release_pin(pin_registry, pin_id);

    result
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

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_llm::block_manager::block::transfer::remote::{
        RemoteBlockDescriptor, RemoteTransferPipeline,
    };

    /// Test that RemoteTransferRequest::new_h2o creates the correct request structure.
    #[test]
    fn test_h2o_request_creation() {
        let request_id = "test-request-001".to_string();
        let sequence_hashes = vec![0x1234, 0x5678, 0x9ABC];
        let host_block_ids = vec![42, 17, 88];
        let operation_id = uuid::Uuid::new_v4();
        let block_size = 16;

        let pin_id = uuid::Uuid::new_v4();
        let req = RemoteTransferRequest::new_h2o(
            request_id.clone(),
            sequence_hashes.clone(),
            host_block_ids.clone(),
            operation_id,
            block_size,
            pin_id,
        );

        assert!(!req.is_onboard);
        assert!(req.is_h2o());
        assert_eq!(req.sequence_hashes, sequence_hashes);
        assert_eq!(req.host_block_ids, Some(host_block_ids));
        assert!(req.device_block_ids.is_empty()); // H2O doesn't use device blocks
        assert_eq!(req.block_size, block_size);
        assert_eq!(req.request_id, request_id);
        assert_eq!(req.pin_id, Some(pin_id));
    }

    /// Test that H2O pipeline uses offload_with_bounce correctly.
    #[test]
    fn test_h2o_pipeline_uses_host_block_ids() {
        let host_block_ids = vec![42, 17, 88];
        let descriptors = vec![
            RemoteBlockDescriptor::object_from_hash("test-bucket", 0x1234, 4096),
            RemoteBlockDescriptor::object_from_hash("test-bucket", 0x5678, 4096),
            RemoteBlockDescriptor::object_from_hash("test-bucket", 0x9ABC, 4096),
        ];

        // This is what process_remote_transfer_request does for H2O
        let pipeline = RemoteTransferPipeline::offload_with_bounce(
            descriptors,
            host_block_ids.clone(),
            vec![], // Empty device_block_ids for H2O
        );

        assert!(pipeline.has_bounce());
        assert_eq!(pipeline.bounce_block_ids(), Some(host_block_ids.as_slice()));
        assert_eq!(pipeline.device_block_ids(), Some([].as_slice()));
        assert_eq!(pipeline.num_blocks(), 3);
    }

    /// Test that offload_blocks creates LocalOffloadRequest with block_size.
    #[test]
    fn test_local_offload_request_has_block_size() {
        let request_id = "test-request".to_string();
        let block_ids = vec![0, 1, 2];

        // Create mock token blocks (we can't easily create real ones without the full infrastructure)
        // So we just test the struct directly
        let operation_id = uuid::Uuid::new_v4();
        let block_size = 16;

        // Test that LocalOffloadRequest stores block_size
        let req = LocalOffloadRequest {
            request_id: request_id.clone(),
            block_ids: block_ids.clone(),
            token_blocks: vec![], // Empty for this unit test
            operation_id,
            sequence_hashes: vec![0x1234, 0x5678, 0x9ABC],
            block_size,
        };

        assert_eq!(req.block_size, block_size);
        assert_eq!(req.block_ids.len(), 3);
    }

    /// Test H2O filtering logic: already-stored hashes are removed.
    #[test]
    fn test_h2o_filtering_removes_already_stored() {
        // Simulate g4_can_offload response
        let all_hashes: Vec<u64> = vec![0x1111, 0x2222, 0x3333, 0x4444];
        let already_stored: Vec<u64> = vec![0x2222, 0x4444]; // These are already in object storage

        // Build stored set for O(1) lookup
        let stored_set: std::collections::HashSet<u64> = already_stored.into_iter().collect();

        // Filter - keep only hashes NOT in stored_set
        let can_offload_hashes: Vec<u64> = all_hashes
            .iter()
            .filter(|h| !stored_set.contains(h))
            .copied()
            .collect();

        assert_eq!(can_offload_hashes, vec![0x1111, 0x3333]);
    }

    /// Test H2O host block ID filtering matches hash filtering.
    #[test]
    fn test_h2o_host_block_id_filtering() {
        let sequence_hashes: Vec<u64> = vec![0x1111, 0x2222, 0x3333, 0x4444];
        let host_block_ids: Vec<usize> = vec![10, 20, 30, 40];
        let already_stored: Vec<u64> = vec![0x2222, 0x4444];

        let stored_set: std::collections::HashSet<u64> = already_stored.into_iter().collect();

        // Filter both hashes and host block IDs together
        let (filtered_hashes, filtered_host_ids): (Vec<u64>, Vec<usize>) = sequence_hashes
            .iter()
            .zip(host_block_ids.iter())
            .filter(|(hash, _)| !stored_set.contains(hash))
            .map(|(&hash, &id)| (hash, id))
            .unzip();

        assert_eq!(filtered_hashes, vec![0x1111, 0x3333]);
        assert_eq!(filtered_host_ids, vec![10, 30]);
    }

    /// Test that empty can_offload result means skip H2O entirely.
    #[test]
    fn test_h2o_skip_when_all_stored() {
        let all_hashes: Vec<u64> = vec![0x1111, 0x2222];
        let already_stored: Vec<u64> = vec![0x1111, 0x2222]; // ALL are stored

        let stored_set: std::collections::HashSet<u64> = already_stored.into_iter().collect();

        let can_offload_hashes: Vec<u64> = all_hashes
            .iter()
            .filter(|h| !stored_set.contains(h))
            .copied()
            .collect();

        // When can_offload_hashes is empty, H2O should be skipped
        assert!(can_offload_hashes.is_empty());

        // This matches the logic in process_remote_transfer_request:
        // if can_offload_hashes.is_empty() { return Ok(()); }
    }
}
