// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod audit;
mod control;
pub mod control_api;
pub mod disagg;
mod request;
mod slot;
pub mod velo_control;

use super::worker::ConnectorWorkerClient;
use crate::{BlockId, G2, InstanceId, KvbmRuntime};
use kvbm_config::OnboardMode;
use kvbm_engine::leader::InstanceLeader;
use kvbm_engine::leader::{
    FindMatchesOptions, FindMatchesResult, Leader, OnboardingStatus, StagingMode,
};
use kvbm_engine::offload::OffloadEngine;
use kvbm_engine::worker::SerializedLayout;
use kvbm_engine::worker::VeloWorkerClient;
use kvbm_hub::ConditionalDisaggClient;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_observability::CacheStatsTracker;

use anyhow::{Result, anyhow, bail};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::ops::Deref;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use scheduler::{DefaultOracle, ForwardPassSample, Oracle};
use slot::{MatchCheckOutcome, RequestSlot, TransactionState};

pub use request::Request;
pub use scheduler::{CachedRequestData, NewRequestData, SchedulerOutput};
pub use slot::{CdOnboardingPayload, FinishedStatus};
// SlotMatchSplit is defined further down in this file; re-exported here so
// downstream code (e.g., the conditional-disagg `InnerLeaderShim` trait)
// can name it.

pub trait ConnectorLeaderInterface: Send + Sync {}

pub struct ConnectorLeader {
    pub(crate) runtime: Arc<KvbmRuntime>,
    block_size: usize,
    init: Arc<Mutex<WorkerClients>>,
    workers: OnceLock<Arc<WorkerClients>>,
    instance_leader: OnceLock<InstanceLeader>,
    slots: DashMap<String, Arc<Mutex<RequestSlot>>>,
    #[allow(dead_code)] // Will be used for scheduling decisions
    oracle: Arc<dyn Oracle>,
    /// Offload engine for G1→G2→G3 transfers (initialized in initialize_async)
    offload_engine: OnceLock<OffloadEngine>,
    /// Control server shutdown handle (initialized in initialize_async)
    #[allow(dead_code)] // Kept for RAII-based shutdown on drop
    control_server_shutdown: OnceLock<oneshot::Sender<()>>,
    /// Accumulated G2 blocks for intra-pass onboarding.
    ///
    /// These blocks are collected from each request's find session during
    /// `prepare_intra_pass_onboarding` and held until the forward pass completes.
    /// A cleanup task (spawned in `process_scheduler_output`) waits on the
    /// forward pass completion event and then drops these blocks.
    pending_intra_pass_g2_blocks: Mutex<Vec<ImmutableBlock<G2>>>,

    forward_pass_samples: Mutex<Option<ForwardPassSample>>,
    cache_stats: CacheStatsTracker,
    /// Conditional-disagg hub client. Populated in `initialize_async` when
    /// `config.disagg` is present. Holds the hub registration alive for the
    /// life of this leader.
    disagg_client: OnceLock<Arc<ConditionalDisaggClient>>,
    /// CD-role dispatcher (`Arc<ConditionalDisaggLeader>`) installed by
    /// `initialize_async` when `config.disagg` is present. Bindings route
    /// `ConnectorLeaderApi` methods through this when set so role-specific
    /// wrappers (decode/prefill) can intercept GNMT/USAA/etc. without
    /// changing the binding-side handle type.
    cd_api: OnceLock<Arc<dyn disagg::ConnectorLeaderApi>>,
}

#[derive(Default, Clone)]
pub(crate) struct WorkerClients {
    worker_instance_ids: Vec<InstanceId>,
    worker_connector_clients: Vec<ConnectorWorkerClient>,
    worker_transfer_clients: Vec<VeloWorkerClient>,
    worker_metadata: Vec<SerializedLayout>,
}

// Connector leader implementation extensions
mod init;

/// Implementation of the request_finished function.
mod finish;

/// Calls to coordinator workers.
mod clients;

// Implementation of search tools for the get_num_new_matched_tokens function.
mod search;

// Implementation of onboarding tools for the update_state_after_alloc function.
mod onboard;

// Implementation of offloading engine which will be triggered by the build_connector_metadata function.
mod offload;

// Implementation of the [`scheduler::SchedulerOutput`] struct.
pub mod scheduler;

impl ConnectorLeader {
    pub fn new(runtime: Arc<KvbmRuntime>, block_size: usize) -> Self {
        // Pull onboard mode from runtime config
        let onboard_mode = runtime.config().onboard.mode;
        tracing::info!(
            ?onboard_mode,
            "ConnectorLeader initialized with onboard mode"
        );

        Self {
            cache_stats: CacheStatsTracker::new(
                runtime.config().metrics.cache_stats_max_requests,
                Duration::from_secs(runtime.config().metrics.cache_stats_log_interval_secs),
                Some(runtime.messenger().instance_id().to_string()),
            ),
            runtime,
            block_size,
            init: Arc::new(Mutex::new(WorkerClients::default())),
            workers: OnceLock::new(),
            instance_leader: OnceLock::new(),
            slots: DashMap::new(),
            oracle: Arc::new(DefaultOracle::default()),
            offload_engine: OnceLock::new(),
            control_server_shutdown: OnceLock::new(),
            pending_intra_pass_g2_blocks: Mutex::new(Vec::new()),
            forward_pass_samples: Mutex::new(None),
            disagg_client: OnceLock::new(),
            cd_api: OnceLock::new(),
        }
    }

    /// Conditional-disagg dispatcher, populated in `initialize_async` when
    /// `config.disagg` is present. Bindings consult this to route the
    /// `ConnectorLeaderApi` method set through the CD wrapper.
    pub fn cd_api(&self) -> Option<&Arc<dyn disagg::ConnectorLeaderApi>> {
        self.cd_api.get()
    }

    /// Set the CD dispatcher. Called once from `initialize_async`.
    pub(crate) fn set_cd_api(&self, api: Arc<dyn disagg::ConnectorLeaderApi>) -> Result<()> {
        self.cd_api
            .set(api)
            .map_err(|_| anyhow!("cd_api already set"))
    }

    /// Conditional-disagg hub client, if this leader was configured with
    /// `disagg` and `initialize_async` has completed.
    pub fn disagg_client(&self) -> Option<&Arc<ConditionalDisaggClient>> {
        self.disagg_client.get()
    }

    /// Get the current onboard mode.
    pub fn onboard_mode(&self) -> OnboardMode {
        self.runtime.config().onboard.mode
    }

    /// Get the block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Access the InstanceLeader (available after initialize_workers()).
    pub(crate) fn instance_leader(&self) -> Option<&InstanceLeader> {
        self.instance_leader.get()
    }

    /// Access the worker clients, if `initialize_async` has completed.
    ///
    /// Used by the conditional-disagg wrapper to drive
    /// `mark_onboarding_complete` after a remote-prefill request finishes its
    /// G2 → G1 onboarding. Currently dead in this crate; the call site
    /// lands with the `on_block_sets_added` output-pull wiring.
    #[allow(dead_code)]
    pub(crate) fn worker_clients(&self) -> Option<&Arc<WorkerClients>> {
        self.workers.get()
    }

    /// Tokio runtime handle for spawning async work. See `worker_clients`
    /// — the call site lands with the CD output-pull integration.
    #[allow(dead_code)]
    pub(crate) fn tokio_handle(&self) -> tokio::runtime::Handle {
        self.runtime.tokio().clone()
    }

    /// Set the InstanceLeader (called by test infrastructure after worker initialization).
    ///
    /// This is typically called by ConnectorInstance after workers are initialized
    /// and we have access to their DirectWorker instances.
    pub(crate) fn set_instance_leader(&self, leader: InstanceLeader) -> Result<()> {
        self.instance_leader
            .set(leader)
            .map_err(|_| anyhow!("InstanceLeader already set"))
    }

    /// Get the offload engine.
    ///
    /// Returns `None` if `initialize_async()` has not been called.
    pub fn offload_engine(&self) -> Option<&OffloadEngine> {
        self.offload_engine.get()
    }

    /// Check if a slot exists for the given request ID.
    pub fn has_slot(&self, request_id: &str) -> bool {
        self.slots.contains_key(request_id)
    }

    /// Get a slot for the given request ID.
    pub fn get_slot(&self, request_id: &str) -> Result<Arc<Mutex<RequestSlot>>> {
        self.slots
            .get(request_id)
            .map(|slot| slot.clone())
            .ok_or_else(|| anyhow!("Slot not found for request ID: {}", request_id))
    }

    /// Create a new slot for the given request ID, tokens and salt hash.
    pub fn create_slot(&self, request: Request) -> Result<()> {
        let request_id = request.request_id.clone();
        if self.has_slot(&request_id) {
            bail!("Slot already exists for request ID: {}", request_id);
        }
        let slot = RequestSlot::new(request, self.block_size)?;
        self.slots.insert(request_id, Arc::new(Mutex::new(slot)));
        Ok(())
    }

    fn record_cache_metrics(
        &self,
        breakdown: &kvbm_engine::leader::MatchBreakdown,
        blocks_queried: usize,
    ) {
        self.cache_stats.record(
            breakdown.host_blocks,
            breakdown.disk_blocks,
            breakdown.object_blocks,
            blocks_queried,
        );
        let (host_rate, disk_rate, object_rate) = self.cache_stats.rates();
        self.runtime
            .observability()
            .compat_metrics()
            .set_cache_hit_rates(host_rate, disk_rate, object_rate);
        self.runtime
            .observability()
            .transfer_metrics()
            .record_cache_hits("host", breakdown.host_blocks as u64);
        self.runtime
            .observability()
            .transfer_metrics()
            .record_cache_hits("disk", breakdown.disk_blocks as u64);
        self.runtime
            .observability()
            .transfer_metrics()
            .record_cache_hits("object", breakdown.object_blocks as u64);
        self.cache_stats.maybe_log();
    }

    /// Get the total number of tokens in a slot's sequence.
    ///
    /// This is used to compare with the vLLM Request's token count to detect
    /// when new tokens have been generated during decoding.
    pub fn get_slot_total_tokens(&self, request_id: &str) -> Result<usize> {
        let slot = self.get_slot(request_id)?;
        Ok(slot.lock().total_tokens())
    }

    /// Apply G1 block_id assignments to a slot's token sequence without
    /// triggering the inner USAA's G2→G1 transfer.
    ///
    /// Conditional-disagg uses this on USAA-1 to align the vLLM-allocated
    /// G1 block_ids with the slot's sequence hashes; the wrapper then
    /// drives transfers for the local-match and remote-prefill slices
    /// separately via `CdBlockTransport` (step 2 onward).
    #[allow(dead_code)]
    pub(crate) fn apply_block_assignments(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
    ) -> Result<()> {
        let slot = self.get_slot(request_id)?;
        let mut slot = slot.lock();
        slot.set_match_requires_reset(true);
        slot.apply_new_blocks(block_ids);
        Ok(())
    }

    /// Read the local-match split for a slot mid-onboard.
    ///
    /// Non-destructive — does not consume G2 blocks. The wrapper uses this
    /// to decide how many of the vLLM-allocated G1 blocks point at
    /// already-resident local data versus remote-prefill output.
    ///
    /// Slots whose inner GNMT yielded zero local matches don't have an
    /// onboarding state installed — for those the split is degenerate
    /// (`local_match_blocks=0`, `computed_blocks=0`) and the wrapper can
    /// still proceed with a fully-remote prefill.
    #[allow(dead_code)]
    pub(crate) fn slot_match_split(&self, request_id: &str) -> Result<SlotMatchSplit> {
        let slot = self.get_slot(request_id)?;
        let slot = slot.lock();
        let block_size = slot.block_size();
        let total_blocks = slot.sequence.blocks().len();
        let all_sequence_hashes = slot.all_sequence_hashes();

        let onboarding = slot.onboarding_state().map(|o| {
            // CD-only onboarding has no find_session; treat as no local match.
            let breakdown = o
                .find_session
                .as_ref()
                .map(|fs| fs.match_breakdown())
                .unwrap_or_default();
            let local = breakdown.host_blocks + breakdown.disk_blocks + breakdown.object_blocks;
            (local, o.num_computed_tokens)
        });

        Ok(compute_slot_match_split(
            block_size,
            total_blocks,
            all_sequence_hashes,
            onboarding,
        ))
    }

    /// Take ownership of the local-match G2 blocks staged by the slot's
    /// find session.
    ///
    /// Destructive: after this returns the slot's onboarding state has no
    /// G2 blocks left to give. The wrapper hands these to
    /// [`CdBlockTransport::local_g2_to_g1`] for the matched portion of the
    /// G1 destination slice.
    ///
    /// Returns an empty vec when the slot has no onboarding state — that
    /// case corresponds to a 0-local-match request (fully remote prefill);
    /// there's no G2 to take. Mirrors `slot_match_split`'s degenerate case.
    #[allow(dead_code)]
    pub(crate) fn take_local_match_g2_blocks(
        &self,
        request_id: &str,
    ) -> Result<Vec<ImmutableBlock<G2>>> {
        let slot = self.get_slot(request_id)?;
        let mut slot = slot.lock();
        let Some(onboarding) = slot.onboarding_state_mut() else {
            return Ok(Vec::new());
        };
        // CD-only OnboardingState has no find_session — there's nothing to take.
        let Some(fs) = onboarding.find_session.as_mut() else {
            return Ok(Vec::new());
        };
        fs.take_g2_blocks()
            .ok_or_else(|| anyhow!("slot {} G2 blocks not yet ready to take", request_id))
    }

    /// Clone a contiguous slice of the slot's `TokenBlock`s.
    ///
    /// Used by the wrapper to feed `MutableBlock<G2>::complete(token_block)`
    /// once an RDMA pull lands and we need to register the new G2 blocks.
    #[allow(dead_code)]
    pub(crate) fn token_blocks_for_range(
        &self,
        request_id: &str,
        range: std::ops::Range<usize>,
    ) -> Result<Vec<dynamo_tokens::TokenBlock>> {
        let slot = self.get_slot(request_id)?;
        let slot = slot.lock();
        let blocks = slot.sequence.blocks();
        if range.end > blocks.len() {
            bail!(
                "token_blocks_for_range out of bounds: range {:?}, len {}",
                range,
                blocks.len()
            );
        }
        Ok(blocks[range].to_vec())
    }

    /// Drive `mark_onboarding_complete` across all worker clients for
    /// CD-bound requests once the wrapper has finished the G2→G1 transfer.
    #[allow(dead_code)]
    pub(crate) fn mark_workers_onboarding_complete(
        &self,
        request_id: String,
    ) -> futures::future::BoxFuture<'static, Result<()>> {
        use futures::FutureExt;
        let workers = self.workers.get().cloned();
        async move {
            let workers = workers.ok_or_else(|| {
                anyhow!("WorkerClients not initialized for mark_workers_onboarding_complete")
            })?;
            workers.mark_onboarding_complete(request_id).await
        }
        .boxed()
    }

    /// Read the full token sequence for a slot.
    ///
    /// Used by the conditional-disagg wrapper to populate
    /// `RemotePrefillRequest.token_ids` so the prefill peer knows which
    /// tokens to compute.
    #[allow(dead_code)]
    pub(crate) fn slot_token_ids(&self, request_id: &str) -> Result<Vec<u32>> {
        let slot = self.get_slot(request_id)?;
        let slot = slot.lock();
        let total = slot.sequence.total_tokens();
        let tokens = slot.sequence.tokens_at(0..total);
        Ok(Vec::<u32>::from(tokens))
    }

    /// Local velo instance_id — used as the `initiator_instance_id` on
    /// CD remote-prefill requests so the prefill peer can attach back.
    #[allow(dead_code)]
    pub(crate) fn local_instance_id(&self) -> InstanceId {
        self.runtime.messenger().instance_id()
    }

    /// Read the slot's parsed CD transfer params, if any. Returns `None`
    /// when the slot has no metadata or no `kv_transfer_params`.
    ///
    /// Used by the prefill-side conditional-disagg wrapper to detect
    /// CD-bound requests at GNMT time.
    #[allow(dead_code)]
    pub(crate) fn slot_transfer_params(
        &self,
        request_id: &str,
    ) -> Result<Option<kvbm_disagg_protocol::TransferParams>> {
        let slot = self.get_slot(request_id)?;
        let slot = slot.lock();
        slot.transfer_params()
            .map_err(|err| anyhow!("decode kv_transfer_params for {}: {}", request_id, err))
    }

    /// Drive `mark_failed_onboarding` across all worker clients for the
    /// listed G1 block_ids when a CD pipeline cannot complete.
    #[allow(dead_code)]
    pub(crate) fn mark_workers_failed_onboarding(
        &self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> futures::future::BoxFuture<'static, Result<()>> {
        use futures::FutureExt;
        let workers = self.workers.get().cloned();
        async move {
            let workers = workers.ok_or_else(|| {
                anyhow!("WorkerClients not initialized for mark_workers_failed_onboarding")
            })?;
            workers.mark_failed_onboarding(request_id, block_ids).await
        }
        .boxed()
    }
}

/// Snapshot of a slot's match-side state taken mid-onboard, used by the
/// conditional-disagg wrapper to split USAA's G1 block_ids into
/// `[computed | local_match | remote_prefill]` ranges.
///
/// Methods/fields are `#[allow(dead_code)]` until `decode_usaa` rewrite
/// (step 6) wires them in.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SlotMatchSplit {
    pub block_size: usize,
    pub computed_blocks: usize,
    pub local_match_blocks: usize,
    pub total_blocks: usize,
    pub all_sequence_hashes: Vec<crate::SequenceHash>,
}

/// Pure split-computation. Extracted for unit-testing the
/// no-onboarding-state degenerate case (cold-cache prefill request)
/// without standing up a full `ConnectorLeader` fixture.
///
/// `onboarding` carries `(local_match_blocks, num_computed_tokens)`
/// when the slot has an onboarding state attached, `None` otherwise.
fn compute_slot_match_split(
    block_size: usize,
    total_blocks: usize,
    all_sequence_hashes: Vec<crate::SequenceHash>,
    onboarding: Option<(usize, usize)>,
) -> SlotMatchSplit {
    let (local_match_blocks, computed_blocks) = match onboarding {
        Some((local, num_computed)) => (local, num_computed / block_size),
        None => (0, 0),
    };
    SlotMatchSplit {
        block_size,
        computed_blocks,
        local_match_blocks,
        total_blocks,
        all_sequence_hashes,
    }
}

#[allow(dead_code)]
impl SlotMatchSplit {
    pub fn remote_blocks(&self) -> usize {
        self.total_blocks
            .saturating_sub(self.computed_blocks)
            .saturating_sub(self.local_match_blocks)
    }

    pub fn local_match_range(&self) -> std::ops::Range<usize> {
        self.computed_blocks..self.computed_blocks + self.local_match_blocks
    }

    pub fn remote_range(&self) -> std::ops::Range<usize> {
        let start = self.computed_blocks + self.local_match_blocks;
        start..self.total_blocks
    }

    pub fn expected_remote_hashes(&self) -> Vec<crate::SequenceHash> {
        self.all_sequence_hashes[self.remote_range()].to_vec()
    }
}

impl ConnectorLeader {
    /// Extend a slot's token sequence with new tokens.
    ///
    /// This is called during decoding when new tokens have been generated
    /// and need to be synchronized to the slot.
    pub fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> Result<()> {
        let slot = self.get_slot(request_id)?;
        let mut slot = slot.lock();
        slot.extend_tokens(tokens)
    }

    /// Get the number of new tokens that can be loaded from external KV cache.
    ///
    /// This implements the vLLM KVConnector interface for `get_num_new_matched_tokens`:
    /// - Returns `(None, false)` while the find operation is still in progress
    /// - Returns `(Some(0), false)` if no external blocks are found
    /// - Returns `(Some(n), true)` if n tokens can be loaded asynchronously (inter-pass mode)
    /// - Returns `(Some(n), false)` if n tokens will be loaded synchronously (intra-pass mode)
    ///
    /// The first call for a request starts the find operation. Subsequent calls check
    /// the status of the operation and return results when complete.
    ///
    /// The second boolean in the return tuple indicates whether async loading is in progress:
    /// - `true` for inter-pass mode (async out-of-band via Velo messages)
    /// - `false` for intra-pass mode (sync layer-wise during forward pass)
    #[tracing::instrument(level = "info", skip(self), ret)]
    pub fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        let shared_slot = self
            .slots
            .get(request_id)
            .map(|slot| slot.clone())
            .expect("slot should exist");

        let mut slot = shared_slot.lock();

        if slot.match_requires_reset() {
            // Check for inflight offloads - potential race with block freeing
            // vLLM preemption frees G1 blocks immediately, but we may still have
            // offload transfers reading from those blocks.
            // if slot.has_inflight_offloads() {
            //     tracing::error!(
            //         request_id,
            //         "Preemption detected while offloads inflight - deferring reset until transfers complete"
            //     );
            //     // Return (None, false) to signal "not ready" - vLLM will retry next cycle
            //     return Ok((None, false));
            // }

            // Safe to reset - no inflight offloads
            tracing::debug!(request_id, "Resetting slot state after preemption");
            slot.reset_for_preemption();
            // Fall through to normal matching flow
        }

        // Determine the match outcome
        let outcome = self.process_match(&mut slot, num_computed_tokens);
        let match_breakdown = slot
            .onboarding_state()
            .and_then(|state| state.find_session.as_ref())
            .map(|fs| fs.match_breakdown())
            .unwrap_or_default();
        let blocks_queried = slot.match_query_blocks();

        // Single point for state transition
        match slot.finalize_match_check(outcome) {
            Ok((count, uses_async)) => {
                if let Some(count) = count {
                    self.record_cache_metrics(&match_breakdown, blocks_queried);

                    if count > 0 && slot.mark_matched_tokens_reported() {
                        self.runtime
                            .observability()
                            .compat_metrics()
                            .matched_tokens
                            .inc_by(count as u64);
                    }
                }

                // For intra-pass mode, we always return false for the async flag
                // since loading happens synchronously during the forward pass.
                // For inter-pass mode, we preserve the async flag from finalize_match_check.
                let actual_async = match self.onboard_mode() {
                    OnboardMode::Intra => false,
                    OnboardMode::Inter => uses_async,
                };
                Ok((count, actual_async))
            }
            Err(e) => {
                self.recover_from_match_error(&mut slot);
                if cfg!(debug_assertions) {
                    // If we are in debug mode, we want to fail the request so we can find and diagnose errors.
                    // Often times, errors will result in a misalignment in the understanding of the frameworks policy
                    // and how it calls the connnector api. Notably, these policies can change subtly across versions.
                    Err(e)
                } else {
                    // If we are in release mode, we want to ensure the request can still be processed normally,
                    // albeit without the benefits of getting an external kv cache match.
                    Ok((None, false))
                }
            }
        }
    }

    /// If this is called with `num_external_tokens` > 0, it will be called with all the blocks upto the block(s)
    /// that need to be onboarded, this included any matched G1 blocks.
    ///
    /// In this case, we compute the start block to onboard by scanning back from the end of the block lists by
    /// the `num_external_tokens/block_size`.
    ///
    /// If this is called with `num_external_tokens` == 0, we will be given the remainder of the blocks destined
    /// for prefill.
    ///
    /// The behavior depends on the configured onboard mode:
    /// - **Inter-pass mode**: Spawns an async task to transfer blocks from G2 to G1 via Velo messages.
    /// - **Intra-pass mode**: Stores G2/G1 block pairs for later aggregation in `process_scheduler_output`,
    ///   which will pass them to workers via `KvConnectorMetadata.intra_pass_load`.
    #[tracing::instrument(level = "debug", skip(self), fields(?request_id))]
    pub fn update_state_after_alloc(
        self: &Arc<Self>,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        let block_size = self.block_size;
        self.get_slot(request_id)?
            .lock()
            .set_match_requires_reset(true);

        if num_external_tokens == 0 {
            return Ok(());
        }

        if !num_external_tokens.is_multiple_of(block_size) {
            bail!(
                "num_external_tokens {} is not a multiple of block size {}",
                num_external_tokens,
                block_size
            );
        }

        let result = match self.onboard_mode() {
            OnboardMode::Inter => {
                // Async out-of-band onboarding via Velo messages
                self.start_onboarding(request_id, block_ids, num_external_tokens)
            }
            OnboardMode::Intra => {
                // Sync layer-wise onboarding - store G2/G1 pairs for later
                self.prepare_intra_pass_onboarding(request_id, block_ids, num_external_tokens)
            }
        };

        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                tracing::error!("Failed to start onboarding: {}", e);
                todo!("clean up session and free resources")
            }
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(iteration = output.iteration))]
    pub fn build_connector_meta(
        &self,
        output: scheduler::SchedulerOutput,
    ) -> Result<scheduler::KvConnectorMetadata> {
        self.process_scheduler_output(output)
    }

    // ========================================================================
    // Eviction Query Methods (Scheduler Integration)
    // ========================================================================
    //
    // These methods extend the vLLM KVConnector API to support intelligent
    // eviction decisions. The scheduler calls these during preemption to:
    //
    // 1. `can_evict()` - Check if a request can be safely evicted (no inflight offloads)
    // 2. `get_eviction_score()` - Get G2 availability for ranking eviction candidates
    // 3. `get_block_boundary_info()` - Get alignment info for block-boundary eviction
    //
    // These methods are designed to be called from the scheduler's `try_preempt()`
    // method, typically in a loop over candidate requests.

    /// Check if a request can be safely evicted.
    ///
    /// A request **cannot** be evicted if it has inflight G1→G2 offload transfers
    /// in progress. Evicting would free G1 blocks that are being read by RDMA,
    /// causing data corruption or undefined behavior.
    ///
    /// # Returns
    ///
    /// - `true` if the request can be safely evicted (no inflight offloads)
    /// - `false` if the request has inflight offloads and must not be evicted
    ///
    /// # Onboarding Protection
    ///
    /// Requests actively loading KV data from G2 (onboarding) are automatically
    /// protected because they are in the `Waiting` queue (status `WAITING_FOR_REMOTE_KVS`),
    /// not the `Running` queue. Only running requests are candidates for eviction,
    /// so onboarding requests are never considered.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In scheduler's try_preempt():
    /// for candidate in &running_requests {
    ///     if let Some(connector) = &self.connector {
    ///         if !connector.can_evict(&candidate.request_id()) {
    ///             continue; // Skip - has inflight offloads
    ///         }
    ///     }
    ///     // Candidate is safe to evict
    /// }
    /// ```
    pub fn can_evict(&self, request_id: &str) -> bool {
        // If no slot exists, the request has no connector state - safe to evict
        let Some(slot_ref) = self.slots.get(request_id) else {
            return true;
        };

        let slot = slot_ref.lock();

        // Check for inflight offloads
        !slot.has_inflight_offloads()
    }

    /// Get eviction score for a request based on G2 block coverage.
    ///
    /// Requests with more blocks already offloaded to G2 are preferred for eviction
    /// because:
    ///
    /// - They can be resumed with minimal prefill (onboarding from G2 is fast)
    /// - The work invested in offloading is preserved
    /// - Memory is freed without losing computation
    ///
    /// # Returns
    ///
    /// An [`EvictionScore`] containing:
    /// - `g2_block_count`: Number of blocks available in G2 for this request
    /// - `total_block_count`: Total number of blocks assigned to this request
    /// - `coverage_ratio`: Fraction of blocks in G2 (g2_block_count / total_block_count)
    ///
    /// Higher `coverage_ratio` = better eviction candidate.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In scheduler's victim selection:
    /// let scores: Vec<_> = candidates.iter()
    ///     .filter_map(|c| {
    ///         connector.get_eviction_score(&c.request_id())
    ///             .ok()
    ///             .map(|s| (c, s))
    ///     })
    ///     .collect();
    ///
    /// // Select candidate with highest G2 coverage
    /// let best_victim = scores.iter()
    ///     .max_by(|a, b| a.1.coverage_ratio.partial_cmp(&b.1.coverage_ratio).unwrap());
    /// ```
    pub fn get_eviction_score(&self, request_id: &str) -> Result<EvictionScore> {
        let slot_ref = self.get_slot(request_id)?;
        let slot = slot_ref.lock();

        // TODO: Implement actual G2 block counting by querying InstanceLeader
        // For now, return a stub that indicates no G2 coverage
        //
        // The real implementation would:
        // 1. Get the block IDs assigned to this request from the slot
        // 2. Query the InstanceLeader to see which blocks exist in G2
        // 3. Return the count and ratio

        let total_block_count = slot.assigned_block_count();

        // Stub: No G2 coverage information available yet
        Ok(EvictionScore {
            g2_block_count: 0,
            total_block_count,
            coverage_ratio: 0.0,
        })
    }

    /// Get block boundary alignment information for a request.
    ///
    /// For efficient eviction, we want to evict at block boundaries:
    ///
    /// - Continuing generation until a block is full costs zero extra resources
    ///   (the block is already allocated and partially filled)
    /// - Evicting at a boundary preserves complete blocks that can be resumed
    /// - On resume, we prefill just the known next token for the new block
    ///
    /// # Returns
    ///
    /// A [`BlockBoundaryInfo`] containing:
    /// - `is_at_boundary`: True if the request is at a block boundary (last block is full)
    /// - `tokens_until_boundary`: Tokens remaining until the next block boundary
    /// - `current_block_fill`: Tokens in the current (partial) block
    ///
    /// Prefer evicting requests where `is_at_boundary == true` or `tokens_until_boundary`
    /// is small (close to completing a block).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In scheduler's victim selection:
    /// for candidate in &candidates {
    ///     let boundary_info = connector.get_block_boundary_info(&candidate.request_id())?;
    ///     if boundary_info.is_at_boundary {
    ///         // Ideal candidate - evict immediately
    ///         return Some(candidate);
    ///     }
    /// }
    /// ```
    pub fn get_block_boundary_info(&self, request_id: &str) -> Result<BlockBoundaryInfo> {
        let slot_ref = self.get_slot(request_id)?;
        let slot = slot_ref.lock();

        let total_tokens = slot.total_tokens();
        let block_size = self.block_size;

        // Calculate position within current block
        let current_block_fill = total_tokens % block_size;
        let is_at_boundary = current_block_fill == 0 && total_tokens > 0;
        let tokens_until_boundary = if is_at_boundary {
            0
        } else {
            block_size - current_block_fill
        };

        Ok(BlockBoundaryInfo {
            is_at_boundary,
            tokens_until_boundary,
            current_block_fill,
        })
    }

    // =========================================================================
    // Projection System Integration - Priority Offload
    // =========================================================================

    /// Request priority offload for blocks planned for eviction.
    ///
    /// When the projection system identifies requests that need to be evicted,
    /// this method requests priority G2 offload for their blocks. This ensures
    /// that blocks are safely in G2 before eviction, enabling faster resume.
    ///
    /// # Arguments
    ///
    /// * `request_id` - The request whose blocks should be offloaded
    /// * `block_ids` - Specific blocks to prioritize (if empty, all blocks)
    ///
    /// # Returns
    ///
    /// The number of blocks that were queued for priority offload.
    ///
    /// # Note
    ///
    /// This is currently a stub. The real implementation will:
    /// 1. Add blocks to a priority queue in the offload engine
    /// 2. Bump their priority above normal background offload
    /// 3. Track completion via the existing inflight offload mechanism
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In scheduler's planned eviction processing:
    /// if let Some(connector) = &self.connector {
    ///     let block_ids = request.block_ids();
    ///     let queued = connector.request_priority_offload(
    ///         request.request_id(),
    ///         &block_ids,
    ///     )?;
    ///     tracing::debug!(
    ///         request_id = %request.request_id(),
    ///         blocks_queued = queued,
    ///         "Requested priority offload for planned eviction"
    ///     );
    /// }
    /// ```
    pub fn request_priority_offload(
        &self,
        request_id: &str,
        block_ids: &[crate::BlockId],
    ) -> Result<usize> {
        let _slot_ref = self.get_slot(request_id)?;

        // TODO: Implement priority queue in offload engine
        // For now, return 0 to indicate no blocks were queued
        //
        // The real implementation would:
        // 1. Get the OffloadEngine reference from the slot
        // 2. Add blocks to the priority offload queue
        // 3. Track the request in the slot's inflight offloads
        //
        // This requires changes to:
        // - OffloadEngine to support priority queue
        // - RequestSlot to track priority offload state

        tracing::debug!(
            request_id = request_id,
            block_count = block_ids.len(),
            "Priority offload requested (stub - not implemented)"
        );

        Ok(0)
    }

    /// Get per-block G2 status for a request.
    ///
    /// Returns a map of block IDs to their G2 presence status. This is used
    /// by the projection system to determine which blocks are safe to release
    /// from paused requests without losing computed state.
    ///
    /// # Returns
    ///
    /// A HashMap where:
    /// - Key: Block ID
    /// - Value: `true` if the block exists in G2, `false` otherwise
    ///
    /// # Note
    ///
    /// This is currently a stub. The real implementation will query the
    /// InstanceLeader's block registry for G2 presence information.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In scheduler's progressive block release:
    /// if let Some(connector) = &self.connector {
    ///     let g2_status = connector.get_block_g2_status(request_id)?;
    ///     let releasable: Vec<_> = request.block_ids()
    ///         .into_iter()
    ///         .filter(|id| g2_status.get(id).copied().unwrap_or(false))
    ///         .collect();
    ///     // These blocks are in G2 and can be safely released
    /// }
    /// ```
    pub fn get_block_g2_status(
        &self,
        request_id: &str,
    ) -> Result<std::collections::HashMap<crate::BlockId, bool>> {
        // Verify request exists
        let _slot_ref = self.get_slot(request_id)?;

        // TODO: Query InstanceLeader for block-level G2 presence
        // For now, return empty map indicating no G2 status available
        //
        // The real implementation would:
        // 1. Get block IDs from the slot (requires adding a method)
        // 2. Query InstanceLeader's G2 block registry
        // 3. Return presence status for each block
        //
        // Note: The slot currently only tracks block count, not individual IDs.
        // The real implementation will need to either:
        // - Add block ID tracking to RequestSlot, or
        // - Query the scheduler's block state for block IDs

        Ok(std::collections::HashMap::new())
    }
}

/// Eviction score for a request based on G2 block coverage.
///
/// Used by the scheduler to rank eviction candidates. Requests with higher
/// G2 coverage are preferred for eviction because they can resume faster.
#[derive(Debug, Clone, Copy)]
pub struct EvictionScore {
    /// Number of blocks for this request that exist in G2 (host memory).
    pub g2_block_count: usize,

    /// Total number of blocks assigned to this request.
    pub total_block_count: usize,

    /// Coverage ratio: g2_block_count / total_block_count.
    ///
    /// - 1.0 = All blocks in G2, request can resume with zero prefill
    /// - 0.0 = No blocks in G2, request must fully recompute
    pub coverage_ratio: f32,
}

/// Block boundary alignment information for a request.
///
/// Used by the scheduler to prefer evicting at block boundaries, which
/// is more efficient for resume operations.
#[derive(Debug, Clone, Copy)]
pub struct BlockBoundaryInfo {
    /// True if the request is exactly at a block boundary (last block is full).
    pub is_at_boundary: bool,

    /// Number of tokens until the next block boundary.
    ///
    /// - 0 if `is_at_boundary` is true
    /// - `block_size - current_block_fill` otherwise
    pub tokens_until_boundary: usize,

    /// Number of tokens in the current (partial) block.
    ///
    /// - 0 if `is_at_boundary` is true
    /// - 1..block_size otherwise
    pub current_block_fill: usize,
}

impl Deref for ConnectorLeader {
    type Target = dyn Leader;

    fn deref(&self) -> &Self::Target {
        self.instance_leader.get().expect("InstanceLeader not set")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SequenceHash;

    /// Regression for the B.2 disagg-bringup smoke failure:
    /// `commit_gnmt_remote → slot_match_split` panicked with
    /// "slot has no active onboarding state" on cold-cache requests
    /// because the inner GNMT path doesn't install onboarding state
    /// when there's nothing to match locally. The wrapper must still
    /// be able to compute a (degenerate) split and proceed with a
    /// fully-remote prefill.
    #[test]
    fn slot_match_split_handles_no_onboarding_state() {
        let block_size = 16;
        let total_blocks = 8;
        let hashes: Vec<SequenceHash> = (0..total_blocks)
            .map(|i| SequenceHash::new(i as u64, None, i as u64))
            .collect();

        // No onboarding state — the cold-cache path that crashed.
        let split =
            compute_slot_match_split(block_size, total_blocks, hashes.clone(), None);

        assert_eq!(split.local_match_blocks, 0);
        assert_eq!(split.computed_blocks, 0);
        assert_eq!(split.total_blocks, total_blocks);
        assert_eq!(split.block_size, block_size);
        assert_eq!(split.all_sequence_hashes, hashes);
        // The wrapper relies on these range helpers for slicing the
        // prefill request — confirm they're sensible in the
        // degenerate case.
        assert_eq!(split.local_match_range(), 0..0);
        assert_eq!(split.remote_range(), 0..total_blocks);
        assert_eq!(split.remote_blocks(), total_blocks);
    }

    /// Onboarding-state present: split reflects local + computed.
    #[test]
    fn slot_match_split_with_onboarding_state() {
        let block_size = 16;
        let total_blocks = 10;
        let hashes: Vec<SequenceHash> = (0..total_blocks)
            .map(|i| SequenceHash::new(i as u64, None, i as u64))
            .collect();

        // 2 blocks already computed (32 tokens), 3 blocks matched locally.
        let split = compute_slot_match_split(
            block_size,
            total_blocks,
            hashes.clone(),
            Some((3, 32)),
        );

        assert_eq!(split.computed_blocks, 2);
        assert_eq!(split.local_match_blocks, 3);
        assert_eq!(split.local_match_range(), 2..5);
        assert_eq!(split.remote_range(), 5..total_blocks);
        assert_eq!(split.remote_blocks(), 5);
    }
}
