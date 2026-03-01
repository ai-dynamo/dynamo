// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    any::Any,
    cmp::max,
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::{
    block_manager::{
        BasicMetadata, DiskStorage, ImmutableBlock, PinnedStorage,
        block::{
            BlockId,
            data::logical::distributed_leader_worker::DistributedLeaderWorkerResources,
            locality::Logical,
            transfer::remote::RemoteKey,
        },
        config::should_disable_cpu_cache_lookup,
        connector::{
            RequestKey,
            cache_stats::CacheStatsTracker,
            protocol::{RequestType, TransferType, WorkerTransferRequest},
            tier::{G4State, TierState},
        },
        distributed::registry::{NoMetadata, PositionalKey},
        distributed::{KvbmLeader, RemoteHashOperationsSync, vllm as vllm_int},
        metrics_kvbm::KvbmMetrics,
        KvBlockManager,
    },
    tokens::{SaltHash, TokenBlock, TokenBlockSequence, Tokens},
};
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use tokio::{runtime::Handle, sync::mpsc};
use tokio_util::sync::CancellationToken;

use super::{
    AnyBlocks, AnyImmutableBlocks, LocalOffloadRequest, LocalOnboardRequest, LocalTransferEngine,
    LocalTransferRequest, PendingG4Lookup,
    RemoteTransferRequest, OperationTracker, compute_tp_consensus_hashes, flush_batch_size,
    g4_min_candidate_blocks, g4_transfer_timeout,
    ExternallyManagedDeviceSlot, Slot, SlotError, SlotManager, SlotState,
};

type VllmBlockManager = KvBlockManager<Logical<DistributedLeaderWorkerResources>, BasicMetadata>;
type VllmLocality = Logical<DistributedLeaderWorkerResources>;

pub struct ConnectorSlotManager<R: RequestKey> {
    slots: Mutex<HashMap<R, Arc<Mutex<VllmConnectorSlot>>>>,
    block_manager: VllmBlockManager,
    /// use this to issue [`LocalTransferRequest`]s to the transfer engine
    xfer_tx: mpsc::UnboundedSender<LocalTransferRequest>,
    _transfer_engine_handle: Option<CriticalTaskExecutionHandle>,
    /// Cache statistics tracker
    cache_stats: Arc<CacheStatsTracker>,
    /// KVBM metrics for exposing cache hit rates
    #[allow(dead_code)]
    kvbm_metrics: KvbmMetrics,
    /// Minimum priority threshold for host offload filtering (read once at init)
    offload_min_priority: u32,
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
        let offload_min_priority = std::env::var("DYN_KVBM_HOST_OFFLOAD_PREFIX_MIN_PRIORITY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let kvbm_metrics_clone = kvbm_metrics.clone();
        let cache_stats_clone = cache_stats.clone();

        // Spawn a background task to periodically update metrics and log cache hit rates
        let handle = Handle::current();
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
        let primary_token = CancellationToken::new();
        let primary_token_clone = primary_token.clone();
        let runtime_primary = Handle::current();
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
            offload_min_priority,
            leader,
        }
    }
}

impl<R: RequestKey> ConnectorSlotManager<R> {
    /// Clear (wipe) all KV cache entries from a specific pool.
    ///
    /// This drops **all** tracked slots (releasing block references) and then
    /// resets the target pool, returning every block to the empty state.
    ///
    /// `pool` must be one of: `"gpu"` / `"device"`, `"cpu"` / `"host"`, or `"disk"`.
    pub fn clear_pool(&self, pool: &str) -> Result<(), SlotError> {
        // Step 1: Drop all slots so block references are released back to the pool.
        {
            let mut slots = self.slots.lock().unwrap();
            let count = slots.len();
            if count > 0 {
                tracing::warn!(
                    "clear_pool({pool}): dropping {count} active connector slots to release block references"
                );
                slots.clear();
            }
        }

        // Step 2: Reset the target pool.
        match pool.to_lowercase().as_str() {
            "gpu" | "device" => {
                if let Some(device) = self.block_manager.device() {
                    device.reset_blocking()?;
                    tracing::info!("clear_pool: device (GPU) pool wiped");
                } else {
                    return Err(SlotError::InvalidOperation(
                        "device pool is not configured".into(),
                    ));
                }
            }
            "cpu" | "host" => {
                if let Some(host) = self.block_manager.host() {
                    host.reset_blocking()?;
                    tracing::info!("clear_pool: host (CPU) pool wiped");
                } else {
                    return Err(SlotError::InvalidOperation(
                        "host pool is not configured".into(),
                    ));
                }
            }
            "disk" => {
                if let Some(disk) = self.block_manager.disk() {
                    disk.reset_blocking()?;
                    tracing::info!("clear_pool: disk pool wiped");
                } else {
                    return Err(SlotError::InvalidOperation(
                        "disk pool is not configured".into(),
                    ));
                }
            }
            other => {
                return Err(SlotError::InvalidOperation(format!(
                    "unknown pool '{other}': expected one of 'gpu', 'device', 'cpu', 'host', 'disk'"
                )));
            }
        }

        Ok(())
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
            self.offload_min_priority,
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

type PendingLookup = PendingG4Lookup<
    Vec<ImmutableBlock<PinnedStorage, VllmLocality, BasicMetadata>>,
    Vec<ImmutableBlock<DiskStorage, VllmLocality, BasicMetadata>>,
>;

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

    /// The number of blocks cached from the device
    tokens_cached_from_device: usize,

    /// Host tier state (CPU cache)
    host: TierState<Vec<ImmutableBlock<PinnedStorage, VllmLocality, BasicMetadata>>>,

    /// Disk tier state
    disk: TierState<Vec<ImmutableBlock<DiskStorage, VllmLocality, BasicMetadata>>>,

    /// G4/object tier state
    g4: G4State<PendingLookup>,

    /// Phantom data to ensure the storage type is correct.
    block_manager: VllmBlockManager,

    block_size: usize,

    iteration_first_scheduled: Option<u64>,

    /// Tracks pending and dispatched worker transfer operations for finish-state decisions.
    operation_tracker: OperationTracker,

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

    /// Flag indicating the slot just recovered from a failed transfer.
    /// When true, `apply_scheduler_output` should ignore vLLM's `num_computed_tokens`
    /// since it reflects pre-failure state, not our reset state.
    recovered_from_failed_transfer: bool,

    /// Cache statistics tracker for this KVBM instance
    cache_stats: Arc<CacheStatsTracker>,

    /// Minimum priority threshold for offload filtering.
    /// All blocks after the first occurance of block priority < threshold are not offloaded.
    offload_min_priority: u32,

    /// Block index where offload was terminated due to priority filtering.
    /// When Some, no further blocks will be offloaded to ensure global contiguity.
    offload_terminated_at_block: Option<usize>,

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
        offload_min_priority: u32,
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
            operation_tracker: OperationTracker::new(),
            tokens_cached_from_device: 0,
            host: TierState::new(),
            disk: TierState::new(),
            g4: G4State::new(),
            performed_cache_lookup: false,
            total_blocks_queried: 0,
            recovered_from_failed_transfer: false,
            cache_stats,
            offload_min_priority,
            offload_terminated_at_block: None,
            leader,
        }
    }

    pub fn has_pending_g4_lookup(&self) -> bool {
        self.g4.has_pending_lookup()
    }

    fn prepare_onboard_dst(&self, n: usize) -> Vec<BlockId> {
        let dst: Vec<BlockId> = self
            .device_blocks
            .iter()
            .skip(self.evaluated_blocks)
            .take(n)
            .copied()
            .collect();
        debug_assert_eq!(dst.len(), n);
        dst
    }

    fn evaluate_and_offload_candidates(
        &mut self,
        block_ids: &[BlockId],
        num_candidate_blocks: usize,
        priorities: Option<&[u32]>,
    ) -> Result<(), SlotError> {
        if num_candidate_blocks == 0 {
            return Ok(());
        }

        let candidate_block_ids: Vec<usize> = self
            .device_blocks
            .iter()
            .skip(self.evaluated_blocks)
            .take(num_candidate_blocks)
            .copied()
            .collect();

        let candidate_priorities: Vec<u32> = if let Some(prios) = priorities {
            let new_blocks_start = self.device_blocks.len() - block_ids.len();
            let candidate_start = self.evaluated_blocks;

            if candidate_start >= new_blocks_start {
                let prio_offset = candidate_start - new_blocks_start;
                debug_assert!(
                    prio_offset + num_candidate_blocks <= prios.len(),
                    "prio_offset ({}) + num_candidate_blocks ({}) > prios.len() ({}); \
                     candidate_start={}, new_blocks_start={}, device_blocks.len()={}, block_ids.len()={}",
                    prio_offset,
                    num_candidate_blocks,
                    prios.len(),
                    candidate_start,
                    new_blocks_start,
                    self.device_blocks.len(),
                    block_ids.len()
                );
                prios
                    .iter()
                    .skip(prio_offset)
                    .take(num_candidate_blocks)
                    .copied()
                    .collect()
            } else {
                vec![0; num_candidate_blocks]
            }
        } else {
            vec![0; num_candidate_blocks]
        };

        assert_eq!(
            candidate_block_ids.len(),
            num_candidate_blocks,
            "device block overflow - candidate blocks exceed block count at offset {}",
            self.evaluated_blocks
        );

        let num_blocks_to_offload = if self.offload_min_priority > 0 {
            candidate_priorities
                .iter()
                .take_while(|&&priority| priority >= self.offload_min_priority)
                .count()
        } else {
            num_candidate_blocks
        };

        if num_blocks_to_offload > 0 {
            if self.offload_min_priority > 0 {
                tracing::debug!(
                    "priority filtering: offloading {}/{} blocks (threshold={})",
                    num_blocks_to_offload,
                    num_candidate_blocks,
                    self.offload_min_priority
                );
            }

            let offload_block_ids: Vec<usize> = candidate_block_ids
                .into_iter()
                .take(num_blocks_to_offload)
                .collect();

            let offload_token_blocks: Vec<TokenBlock> = self
                .sequence
                .blocks()
                .iter()
                .skip(self.evaluated_blocks)
                .take(num_blocks_to_offload)
                .cloned()
                .collect();

            let offload_priorities: Vec<u32> = candidate_priorities
                .iter()
                .take(num_blocks_to_offload)
                .copied()
                .collect();

            self.offload_blocks(
                &offload_block_ids,
                &offload_token_blocks,
                &offload_priorities,
            )
            .expect("failed to offload blocks");
        } else if self.offload_min_priority > 0 {
            tracing::debug!(
                "priority filtering: skipping all {} candidate blocks (threshold={})",
                num_candidate_blocks,
                self.offload_min_priority
            );
        }

        if num_blocks_to_offload < num_candidate_blocks {
            let termination_index = self.evaluated_blocks + num_blocks_to_offload;
            self.offload_terminated_at_block = Some(termination_index);

            tracing::info!(
                request_id = %self.request_id,
                "offload terminated at block {}: priority {} < threshold {}; \
                 no further blocks will be offloaded",
                termination_index,
                candidate_priorities.get(num_blocks_to_offload).copied().unwrap_or(0),
                self.offload_min_priority
            );
        }

        self.evaluated_blocks += num_candidate_blocks;
        Ok(())
    }

    fn reset_core_state(&mut self) {
        self.state = SlotState::Preempted;
        self.iteration_first_scheduled = None;
        self.current_position = 0;
        self.evaluated_blocks = 0;
        self.device_blocks.clear();
        self.tokens_cached_from_device = 0;
        crate::all_tiers!(reset self);
        self.performed_cache_lookup = false;
        self.total_blocks_queried = 0;
    }

    fn start_async_g4_lookup(
        &mut self,
        num_computed_tokens: usize,
        host_blocks: Vec<ImmutableBlock<PinnedStorage, VllmLocality, BasicMetadata>>,
        disk_blocks: Vec<ImmutableBlock<DiskStorage, VllmLocality, BasicMetadata>>,
        g4_candidates: Vec<u64>,
    ) -> Result<(), SlotError> {
        if self.g4.pending_lookup.is_some() {
            return Err(SlotError::InvalidOperation(format!(
                "async G4 lookup already pending for request {}",
                self.request_id
            )));
        }

        if g4_candidates.is_empty() {
            return self.stage_local_matches(num_computed_tokens, host_blocks, disk_blocks, vec![]);
        }

        if self.leader.remote_handle().is_none() {
            return self.stage_local_matches(num_computed_tokens, host_blocks, disk_blocks, vec![]);
        }

        let world_size = self.leader.world_size();
        let num_candidates = g4_candidates.len();
        let all_keys: Vec<_> = (0..world_size)
            .flat_map(|wid| {
                g4_candidates
                    .iter()
                    .enumerate()
                    .map(move |(pos, &hash)| PositionalKey {
                        worker_id: wid as u64,
                        sequence_hash: hash,
                        position: pos as u32,
                    })
            })
            .collect();

        let rx = self.leader.schedule_match_prefix(all_keys);

        tracing::debug!(
            target: "kvbm-g4",
            request_id = %self.request_id,
            num_candidates,
            "started async G4 lookup"
        );

        self.g4.pending_lookup = Some(PendingLookup {
            num_computed_tokens,
            host_blocks,
            disk_blocks,
            world_size,
            receiver: rx,
        });
        Ok(())
    }

    /// Returns `Ok(true)` when onboarding is still progressing and caller should skip
    /// match acquisition for this iteration.
    fn maybe_recover_onboarding_timeout(&mut self) -> Result<bool, SlotError> {
        if !matches!(self.state(), SlotState::Onboarding(_)) {
            return Ok(false);
        }

        let elapsed = self
            .g4
            .onboarding_started_at
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO);
        let timeout = g4_transfer_timeout();
        if elapsed < timeout {
            tracing::debug!(
                target: "kvbm-g4",
                request_id = %self.request_id,
                elapsed_ms = elapsed.as_millis(),
                timeout_secs = timeout.as_secs(),
                "Onboarding still in progress, skipping acquire_local_matches"
            );
            return Ok(true);
        }

        if let Some(stale_hash_positions) = self.g4.attempted_hashes.take() {
            tracing::warn!(
                target: "kvbm-g4",
                request_id = %self.request_id,
                num_stale_hashes = stale_hash_positions.len(),
                stale_hash_positions = ?stale_hash_positions,
                elapsed_ms = elapsed.as_millis(),
                "onboard timed out - removing stale hashes from registry"
            );

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
            elapsed_ms = elapsed.as_millis(),
            "onboard timed out in acquire_local_matches; recovering - will skip G4 on retry"
        );

        let _ = self.operation_tracker.discard_pending();
        self.g4.onboarding_started_at = None;
        self.state = SlotState::Preempted;
        self.iteration_first_scheduled = None;
        self.current_position = 0;
        self.evaluated_blocks = 0;
        self.device_blocks.clear();
        self.tokens_cached_from_device = 0;
        self.host.reset();
        self.disk.reset();
        self.g4.tier.reset();
        self.performed_cache_lookup = false;
        self.total_blocks_queried = 0;

        const MAX_G4_RETRIES: u32 = 3;
        self.g4.retry_count += 1;
        if self.g4.retry_count >= MAX_G4_RETRIES {
            self.g4.skip_on_retry = true;
        }
        self.recovered_from_failed_transfer = true;
        Ok(false)
    }

    fn poll_pending_g4_lookup(&mut self) -> Result<bool, SlotError> {
        enum PollStatus {
            Pending,
            Ready(Vec<(PositionalKey, RemoteKey, NoMetadata)>),
            Closed,
        }

        let Some(pending) = self.g4.pending_lookup.as_mut() else {
            return Ok(false);
        };

        let status = match pending.receiver.try_recv() {
            Ok(matches) => PollStatus::Ready(matches),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => PollStatus::Pending,
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => PollStatus::Closed,
        };

        match status {
            PollStatus::Pending => Ok(false),
            PollStatus::Ready(matches) => {
                let pending = self.g4.pending_lookup.take().ok_or_else(|| {
                    SlotError::InvalidOperation("pending g4 lookup unexpectedly missing".into())
                })?;
                let g4_hashes = compute_tp_consensus_hashes(matches, pending.world_size);
                self.stage_local_matches(
                    pending.num_computed_tokens,
                    pending.host_blocks,
                    pending.disk_blocks,
                    g4_hashes,
                )?;
                Ok(true)
            }
            PollStatus::Closed => {
                tracing::warn!(
                    target: "kvbm-g4",
                    request_id = %self.request_id,
                    "async G4 lookup channel closed; proceeding without G4 matches"
                );
                let pending = self.g4.pending_lookup.take().ok_or_else(|| {
                    SlotError::InvalidOperation("pending g4 lookup unexpectedly missing".into())
                })?;
                self.stage_local_matches(
                    pending.num_computed_tokens,
                    pending.host_blocks,
                    pending.disk_blocks,
                    vec![],
                )?;
                Ok(true)
            }
        }
    }

    fn stage_local_matches(
        &mut self,
        num_computed_tokens: usize,
        mut host_blocks: Vec<ImmutableBlock<PinnedStorage, VllmLocality, BasicMetadata>>,
        mut disk_blocks: Vec<ImmutableBlock<DiskStorage, VllmLocality, BasicMetadata>>,
        mut g4_hashes: Vec<u64>,
    ) -> Result<(), SlotError> {
        let block_size = self.block_size;
        let num_matched_host_blocks = host_blocks.len();
        let num_matched_disk_blocks = disk_blocks.len();
        let num_matched_g4_blocks = g4_hashes.len();
        self.g4
            .tier
            .record_cached_tokens(num_matched_g4_blocks * block_size);

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
        if (num_computed_tokens + num_new_matched_tokens) == self.sequence.total_tokens() {
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

        self.host.stage_non_empty(host_blocks);
        self.disk.stage_non_empty(disk_blocks);
        self.g4.tier.stage_non_empty(g4_hashes);

        self.state = SlotState::OnboardStaged(num_new_matched_tokens);
        Ok(())
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
        crate::all_tiers!(clear_staging self);
        if self.operation_tracker.has_any() {
            tracing::warn!(
                request_id = %self.request_id,
                pending_ops = self.operation_tracker.pending_count(),
                dispatched_ops = self.operation_tracker.dispatched_count(),
                "Preemption while operations pending/in-flight"
            );
            self.operation_tracker.clear_all();
        }

        self.reset_core_state();
        self.offload_terminated_at_block = None;
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
        self.host.record_cached_tokens(num_tokens);
        tracing::debug!("recording {} cached host tokens", num_tokens);
    }

    fn record_cached_disk_tokens(&mut self, num_tokens: usize) {
        self.disk.record_cached_tokens(num_tokens);
        tracing::debug!("recording {} cached disk tokens", num_tokens);
    }

    #[tracing::instrument(level = "debug", skip_all, fields(request_id = self.request_id.as_str()))]
    fn apply_scheduler_output(
        &mut self,
        tokens: &[u32],
        block_ids: &[BlockId],
        num_computed_tokens: usize,
        num_scheduled_tokens: usize,
        priorities: Option<&[u32]>,
    ) -> Result<(), SlotError> {
        // Validate contract: priorities must match block_ids length when provided
        if let Some(prios) = priorities {
            assert_eq!(
                prios.len(),
                block_ids.len(),
                "priorities length ({}) must match block_ids length ({})",
                prios.len(),
                block_ids.len()
            );
        }

        // Onboarding state in apply_scheduler_output is NORMAL, not an error.
        // vLLM schedules the request for prefill immediately after get_num_new_matched_tokens
        // returns async=true. The async KV loading happens on the worker during the forward
        // pass. The slot naturally transitions from Onboarding → Prefilling/Decoding via
        // the state assignment below.
        //
        // Genuine onboarding failures are handled by acquire_local_matches, which is called
        // when vLLM re-evaluates the request for KV matching after a failure/preemption.
        if matches!(self.state, SlotState::Onboarding(_)) {
            tracing::debug!(
                request_id = %self.request_id,
                current_position = self.current_position,
                num_computed_tokens = num_computed_tokens,
                "Onboarding state in apply_scheduler_output - transitioning to normal execution"
            );
            self.g4.onboarding_started_at = None;
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

        // Early exit if offload has been permanently terminated.
        // This ensures global contiguity: once a gap is created by priority filtering,
        // no subsequent blocks will be offloaded for this request.
        if let Some(terminated_at) = self.offload_terminated_at_block {
            tracing::debug!(
                "offload terminated at block {}; skipping offload evaluation",
                terminated_at
            );
            self.current_position += num_scheduled_tokens;
            return Ok(());
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

        self.evaluate_and_offload_candidates(block_ids, num_candidate_blocks, priorities)?;

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
        if self.g4.pending_lookup.is_some() {
            tracing::debug!(
                target: "kvbm-g4",
                request_id = %self.request_id,
                "dropping pending async G4 lookup while finishing request"
            );
            self.g4.pending_lookup.take();
        }

        // Report cache statistics if we performed a cache lookup
        if self.performed_cache_lookup {
            let block_size = self.block_size;

            let (host_blocks, disk_blocks, object_blocks) =
                crate::all_tiers!(cache_stats self, block_size);

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

        // Check if there are any pending operations (not yet dispatched to worker)
        let pending_count = self.operation_tracker.pending_count();

        // Check if there are any dispatched operations (sent to worker, not yet confirmed complete).
        // `pending_operations` is drained by `build_connector_metadata` via `take_pending_operations()`
        // well before `request_finished` fires, so without this check the slot would always
        // transition to `Finished` even when the worker is still processing transfers.
        let has_inflight_ops = self.operation_tracker.has_any();

        if has_inflight_ops {
            // There are pending or in-flight operations - need to wait for them to complete
            self.state = SlotState::Finishing;
            tracing::debug!(
                request_id = %self.request_id,
                pending_operations = pending_count,
                dispatched_operations = self.operation_tracker.dispatched_count(),
                "request set to finish (with in-flight operations): cached_gpu_tokens: {}; cached_host_tokens: {}; cached_disk_tokens: {}",
                self.tokens_cached_from_device,
                self.host.tokens_cached,
                self.disk.tokens_cached
            );
        } else {
            // No pending or in-flight operations - can immediately mark as finished
            self.state = SlotState::Finished;
            tracing::debug!(
                request_id = %self.request_id,
                "request set to finished (no in-flight operations): cached_gpu_tokens: {}; cached_host_tokens: {}; cached_disk_tokens: {}",
                self.tokens_cached_from_device,
                self.host.tokens_cached,
                self.disk.tokens_cached
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
        self.operation_tracker.take_pending_for_dispatch()
    }

    #[tracing::instrument(level = "debug", skip_all)]
    fn acquire_local_matches(&mut self, num_computed_tokens: usize) -> Result<(), SlotError> {
        if matches!(self.state(), SlotState::OnboardStaged(_)) {
            tracing::debug!("slot is already in the OnboardStaged state; skipping lookup");
            return Ok(());
        }

        if self.has_pending_g4_lookup() {
            if self.poll_pending_g4_lookup()? {
                tracing::debug!(
                    target: "kvbm-g4",
                    request_id = %self.request_id,
                    state = ?self.state,
                    "async G4 lookup completed"
                );
            } else {
                tracing::debug!(
                    target: "kvbm-g4",
                    request_id = %self.request_id,
                    "async G4 lookup still pending"
                );
            }
            return Ok(());
        }

        if self.maybe_recover_onboarding_timeout()? {
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
        debug_assert!(num_computed_tokens.is_multiple_of(block_size));

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

        let disable_cpu_lookup = should_disable_cpu_cache_lookup();
        if disable_cpu_lookup {
            tracing::info!(
                request_id = %self.request_id,
                "cpu cache lookup disabled via dev flag; skipping host pool match"
            );
        }

        let host_blocks = if disable_cpu_lookup {
            Vec::new()
        } else {
            self.block_manager
                .host()
                .map(|host| host.match_sequence_hashes_blocking(blocks_to_lookup))
                .transpose()?
                .unwrap_or_default()
        };

        let num_matched_host_blocks = host_blocks.len();
        self.record_cached_host_tokens(num_matched_host_blocks * block_size);

        // advance the search offset by the number of matched host blocks
        let search_offset = search_offset + num_matched_host_blocks;

        // start at host offset
        let disk_blocks = self
            .block_manager
            .disk()
            .map(|disk| disk.match_sequence_hashes_blocking(&sequence_hashes[search_offset..]))
            .transpose()?
            .unwrap_or_default();

        let num_matched_disk_blocks = disk_blocks.len();
        self.record_cached_disk_tokens(num_matched_disk_blocks * block_size);

        // Remote registry lookup with TP consensus (G4/object storage)
        let search_offset_g4 = search_offset + num_matched_disk_blocks;
        let g4_candidates = sequence_hashes[search_offset_g4..].to_vec();

        if self.g4.skip_on_retry {
            tracing::info!(
                target: "kvbm-g4",
                request_id = %self.request_id,
                "skipping - previous failure"
            );
            return self.stage_local_matches(num_computed_tokens, host_blocks, disk_blocks, vec![]);
        }

        let min_candidates = g4_min_candidate_blocks();
        if !g4_candidates.is_empty() && g4_candidates.len() < min_candidates {
            tracing::debug!(
                target: "kvbm-g4",
                request_id = %self.request_id,
                g4_candidates = g4_candidates.len(),
                min_candidates,
                "skipping G4 lookup due to minimum-candidate threshold"
            );
            return self.stage_local_matches(num_computed_tokens, host_blocks, disk_blocks, vec![]);
        }

        if !g4_candidates.is_empty() && self.leader.remote_handle().is_some() {
            self.start_async_g4_lookup(
                num_computed_tokens,
                host_blocks,
                disk_blocks,
                g4_candidates,
            )?;
            return Ok(());
        }

        self.stage_local_matches(num_computed_tokens, host_blocks, disk_blocks, vec![])
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

        self.evaluated_blocks = self.current_position / self.block_size;

        crate::onboard_local_tier!(self, host, PinnedStorage);
        crate::onboard_local_tier!(self, disk, DiskStorage);
        crate::onboard_remote_tier!(self);

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

    /// Append device blocks to the slot.
    ///
    /// vLLM's `get_blocks()` returns the FULL block table each time (not just newly allocated
    /// blocks). This method handles that by only appending the new suffix — blocks beyond what
    /// we already track. Without this deduplication, the second `update_state_after_alloc` call
    /// would duplicate the entire table, causing `apply_scheduler_output` to offload from wrong
    /// positions (e.g., system prompt GPU blocks instead of newly computed blocks).
    #[tracing::instrument(level = "debug", skip_all, fields(request_id = self.request_id))]
    fn append_mutable_device_blocks(&mut self, block_ids: &[BlockId]) -> Result<(), SlotError> {
        let existing = self.device_blocks.len();

        if block_ids.len() > existing {
            // Only append the truly new blocks (the suffix beyond what we already have).
            debug_assert_eq!(
                &self.device_blocks[..],
                &block_ids[..existing],
                "existing device blocks don't match prefix of incoming block table"
            );
            let new_blocks = &block_ids[existing..];
            self.device_blocks.extend(new_blocks);
            tracing::debug!(
                "appended {} new device blocks (skipped {} existing); total device blocks: {}",
                new_blocks.len(),
                existing,
                self.num_device_blocks_allocated()
            );
        } else if block_ids.len() == existing {
            tracing::debug!(
                "no new device blocks to append; total device blocks: {}",
                self.num_device_blocks_allocated()
            );
        } else {
            tracing::warn!(
                "received fewer blocks ({}) than already tracked ({}); ignoring",
                block_ids.len(),
                existing
            );
        }

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
        priorities: &[u32],
    ) -> Result<(), SlotError> {
        // Check if slot is in Finishing state before creating operations
        // If we're finishing, don't create new operations
        if matches!(self.state, SlotState::Finishing | SlotState::Finished) {
            return Ok(());
        }

        assert!(block_ids.len() == token_blocks.len());
        assert!(block_ids.len() == priorities.len());

        let operation_id = uuid::Uuid::new_v4();

        let xfer_req = LocalTransferRequest::Offload(LocalOffloadRequest::new(
            self.request_id.clone(),
            block_ids.to_vec(),
            token_blocks.to_vec(),
            priorities.to_vec(),
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

    /// Discard all pending operations WITHOUT counting them as dispatched.
    /// Used when a request is cancelled mid-transfer (e.g., during onboarding) and the
    /// operations should not prevent the slot from transitioning to `Finished`.
    /// Unlike `take_pending_operations()` (which increments `dispatched_operations_count`),
    /// this method simply drops the pending operations.
    pub fn discard_pending_operations(&mut self) {
        let discarded = self.operation_tracker.discard_pending();
        if discarded > 0 {
            tracing::debug!(
                request_id = %self.request_id,
                discarded_ops = discarded,
                "Discarding pending operations (cancelled request)"
            );
        }
    }

    /// Flush blocks that were never offloaded during chunked prefill.
    ///
    /// vLLM v1 chunked prefill only calls `apply_scheduler_output` for the first chunk.
    /// The remaining chunks are processed internally by vLLM without going through the
    /// connector's scheduler interface. This method is called from `request_finished`
    /// with ALL block_ids vLLM allocated, and offloads any blocks that were missed.
    ///
    /// Blocks are flushed in batches (FLUSH_BATCH_SIZE) to allow D2H and H2R to pipeline.
    /// GPU blocks are held until all D2H transfers complete (via pending_operations),
    /// then freed by vLLM. H2R to remote storage continues from CPU blocks in the background.
    pub fn flush_remaining_blocks(&mut self, all_block_ids: &[BlockId]) -> Result<(), SlotError> {
        let already_offloaded = self.evaluated_blocks;
        let total_sequence_blocks = self.sequence.blocks().len();

        // Don't flush past what the sequence covers
        let flushable = std::cmp::min(all_block_ids.len(), total_sequence_blocks);

        if already_offloaded >= flushable {
            return Ok(());
        }

        // Skip the last block if it covers the exact end of the sequence
        // (same boundary logic as apply_scheduler_output)
        let flush_end = if flushable == total_sequence_blocks
            && (total_sequence_blocks * self.block_size) == self.sequence.total_tokens()
        {
            flushable.saturating_sub(1)
        } else {
            flushable
        };

        if already_offloaded >= flush_end {
            return Ok(());
        }

        let total_remaining = flush_end - already_offloaded;
        let batch_size = flush_batch_size();

        tracing::info!(
            request_id = %self.request_id,
            already_offloaded = already_offloaded,
            flushing = total_remaining,
            total_sequence_blocks = total_sequence_blocks,
            batch_size = batch_size,
            num_batches = (total_remaining + batch_size - 1) / batch_size,
            "Flushing remaining blocks on request finish"
        );

        // Temporarily allow offload_blocks to work even though we're about to
        // transition to Finishing. We set state to Prefilling so the
        // Finishing/Finished check in offload_blocks doesn't reject us.
        let saved_state = self.state;
        self.state = SlotState::Prefilling;

        // Split into batches for D2H/H2R pipelining
        let mut offset = already_offloaded;
        while offset < flush_end {
            let batch_end = std::cmp::min(offset + batch_size, flush_end);
            let batch_block_ids = &all_block_ids[offset..batch_end];
            let batch_token_blocks: Vec<TokenBlock> =
                self.sequence.blocks()[offset..batch_end].to_vec();

            // Flushed blocks don't have priority info; use default priority 0
            let batch_priorities = vec![0u32; batch_block_ids.len()];
            self.offload_blocks(batch_block_ids, &batch_token_blocks, &batch_priorities)?;
            offset = batch_end;
        }

        self.evaluated_blocks = flush_end;

        // Restore state (mark_as_finished will set it to Finishing/Finished)
        self.state = saved_state;

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
    /// Token blocks are threaded through so bounce buffers can be persisted
    /// in the host cache after the transfer completes.
    fn onboard_from_g4(
        &mut self,
        sequence_hashes: Vec<u64>,
        device_block_ids: Vec<BlockId>,
        token_blocks: Vec<TokenBlock>,
    ) -> Result<(), SlotError> {
        debug_assert_eq!(sequence_hashes.len(), device_block_ids.len());

        let (params, worker_req) = vllm_int::onboard_from_g4(
            self.request_id.clone(),
            sequence_hashes,
            device_block_ids,
            self.block_size,
            token_blocks,
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
        self.operation_tracker.append_pending(operation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::block::transfer::remote::{
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
        assert!(req.device_block_ids.is_empty()); // H2R doesn't use device blocks
        assert_eq!(req.block_size, block_size);
        assert_eq!(req.request_id, request_id);
        assert_eq!(req.pin_id, Some(pin_id));
    }

    /// Test that H2R pipeline uses offload_with_bounce correctly.
    #[test]
    fn test_h2o_pipeline_uses_host_block_ids() {
        let host_block_ids = vec![42, 17, 88];
        let descriptors = vec![
            RemoteBlockDescriptor::object_from_hash("test-bucket", 0x1234, 4096),
            RemoteBlockDescriptor::object_from_hash("test-bucket", 0x5678, 4096),
            RemoteBlockDescriptor::object_from_hash("test-bucket", 0x9ABC, 4096),
        ];

        // This is what process_remote_transfer_request does for H2R
        let pipeline = RemoteTransferPipeline::offload_with_bounce(
            descriptors,
            host_block_ids.clone(),
            vec![], // Empty device_block_ids for H2R
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

    /// Test H2R filtering logic: already-stored hashes are removed.
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

    /// Test H2R host block ID filtering matches hash filtering.
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

    /// Test that empty can_offload result means skip H2R entirely.
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

        // When can_offload_hashes is empty, H2R should be skipped
        assert!(can_offload_hashes.is_empty());

        // This matches the logic in process_remote_transfer_request:
        // if can_offload_hashes.is_empty() { return Ok(()); }
    }
}
