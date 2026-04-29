// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode-side conditional-disaggregation leader wrapper.
//!
//! `DecodeDisaggLeader` wraps a base [`ConnectorLeader`] and intercepts
//! the scheduler-facing API ([`ConnectorLeaderApi`]) to drive the
//! conditional-disagg dataflow on the decode side. See
//! `disagg/mod.rs` for the golden path.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use anyhow::{Result, anyhow};
use dashmap::DashMap;
use kvbm_config::{DisaggConfig, DisaggregationRole};
use kvbm_engine::disagg::{PullComplete, RemoteBlockSet, SessionBlocks};
use kvbm_hub::{ConditionalDisaggClient, HubClient};
use kvbm_logical::blocks::{CompleteBlock, ImmutableBlock, MutableBlock};
use parking_lot::Mutex;
use velo::InstanceId;

use crate::BlockId;
use crate::connector::leader::scheduler::{KvConnectorMetadata, SchedulerOutput};
use crate::connector::leader::{FinishedStatus, Request};
use crate::{G2, SequenceHash};

use super::decode::{CdOutputSink, RemotePrefillCoordinator};
use super::transport::{CdBlockTransport, CdWorkerHook, InnerLeaderShim};
use super::{ConnectorLeaderApi, PolicyInputs, PrefillSelection};

// ============================================================================
// Inflight token budget
// ============================================================================

#[derive(Debug)]
pub(crate) struct InflightBudget {
    available: AtomicUsize,
    capacity: usize,
}

impl InflightBudget {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            available: AtomicUsize::new(capacity),
            capacity,
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn available(&self) -> usize {
        self.available.load(Ordering::Acquire)
    }

    pub(crate) fn try_reserve(&self, n: usize) -> bool {
        if self.capacity == usize::MAX {
            return true;
        }
        let mut current = self.available.load(Ordering::Acquire);
        loop {
            if current < n {
                return false;
            }
            match self.available.compare_exchange_weak(
                current,
                current - n,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }

    pub(crate) fn release(&self, n: usize) {
        if self.capacity == usize::MAX {
            return;
        }
        let prev = self.available.fetch_add(n, Ordering::Release);
        debug_assert!(
            prev.saturating_add(n) <= self.capacity,
            "InflightBudget release overflow: prev={prev} +n={n} capacity={cap}",
            cap = self.capacity
        );
    }
}

// ============================================================================
// Per-request state
// ============================================================================

/// Lifecycle state for a single G2 destination block in the remote-prefill
/// slice of a CD request. See plan §"Block Lifecycle".
#[derive(Default)]
enum RemoteG2State {
    /// `mem::take` sentinel — never observed outside transition windows.
    #[default]
    Empty,
    /// Allocated G2 destination, no data written yet.
    Mutable(MutableBlock<G2>),
    /// RDMA pull from the prefill peer is in flight; data is being written
    /// into this block.
    Staging(MutableBlock<G2>),
    /// Data has landed in G2 and the matching G1 onboard has completed; the
    /// block is staged with its sequence hash but not yet registered.
    ReadyToBeRegistered(CompleteBlock<G2>),
    /// Registered with the leader's G2 manager — data is visible to other
    /// matches. The `ImmutableBlock` is held to keep the G2 entry pinned
    /// for the lifetime of this CD request.
    Registered(#[allow(dead_code)] ImmutableBlock<G2>),
}

impl RemoteG2State {
    fn is_registered(&self) -> bool {
        matches!(self, RemoteG2State::Registered(_))
    }
}

struct RemoteBlockSlot {
    expected_hash: SequenceHash,
    /// Index in the slot's TokenBlockSequence (used for `MutableBlock::complete`).
    sequence_index: usize,
    g1_dst_block_id: BlockId,
    state: RemoteG2State,
}

struct CdRequestState {
    reserved_tokens: usize,

    /// Local-match G2 blocks extracted from the slot during GNMT and
    /// stashed for USAA-1 to use for the local kick. Drained on first
    /// USAA-1 visit; `None` afterward.
    pending_local_g2: Mutex<Option<Vec<ImmutableBlock<G2>>>>,

    /// G1 destinations vLLM allocated for the local-match slice
    /// `[computed, computed + local_match)`.
    local_match_g1_block_ids: Vec<BlockId>,
    local_onboard_complete: AtomicBool,

    /// All remote slots, in order over `[X, N)`. `Mutex` because lifecycle
    /// transitions happen across both the USAA-1 task and the
    /// `on_block_sets_added` async task.
    remote_slots: Mutex<Vec<RemoteBlockSlot>>,
    pending_remote_chunks: AtomicUsize,
    remote_chunks_started: AtomicBool,

    completed: AtomicBool,
}

impl CdRequestState {
    fn unfilled_g1_block_ids(&self) -> Vec<BlockId> {
        let mut out = Vec::new();
        if !self.local_onboard_complete.load(Ordering::Acquire) {
            out.extend(self.local_match_g1_block_ids.iter().copied());
        }
        let slots = self.remote_slots.lock();
        for slot in slots.iter() {
            if !slot.state.is_registered() {
                out.push(slot.g1_dst_block_id);
            }
        }
        out
    }

    fn all_registered(&self) -> bool {
        let slots = self.remote_slots.lock();
        slots.iter().all(|s| s.state.is_registered())
    }
}

// ============================================================================
// Wrapper
// ============================================================================

pub struct DecodeDisaggLeader {
    inner: Arc<dyn InnerLeaderShim>,
    role: DisaggregationRole,
    coordinator: Arc<RemotePrefillCoordinator>,
    transport: Arc<dyn CdBlockTransport>,
    worker_hook: Arc<dyn CdWorkerHook>,
    tokio_handle: tokio::runtime::Handle,

    inflight_budget: InflightBudget,
    cd_request_state: DashMap<String, Arc<CdRequestState>>,

    client: Option<Arc<ConditionalDisaggClient>>,
    hub: Option<Arc<HubClient>>,
    hub_velo_id: Option<InstanceId>,

    /// Weak self-pointer used by `CdOutputSink` impl to spawn async tasks
    /// that need an `Arc<Self>` from a `&self` trait method.
    weak_self: Weak<Self>,
}

impl std::fmt::Debug for DecodeDisaggLeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeDisaggLeader")
            .field("role", &self.role)
            .field("inflight_capacity", &self.inflight_budget.capacity())
            .field("inflight_available", &self.inflight_budget.available())
            .field("active_cd_requests", &self.cd_request_state.len())
            .field("hub_velo_id", &self.hub_velo_id)
            .finish()
    }
}

impl DecodeDisaggLeader {
    /// Build a wrapper from an already-constructed inner leader and
    /// supporting pieces. All injectable for tests.
    pub fn from_parts(
        inner: Arc<dyn InnerLeaderShim>,
        config: &DisaggConfig,
        coordinator: Arc<RemotePrefillCoordinator>,
        transport: Arc<dyn CdBlockTransport>,
        worker_hook: Arc<dyn CdWorkerHook>,
        tokio_handle: tokio::runtime::Handle,
        hub: Option<Arc<HubClient>>,
        client: Option<Arc<ConditionalDisaggClient>>,
        hub_velo_id: Option<InstanceId>,
    ) -> Arc<Self> {
        let inflight_budget = InflightBudget::new(config.max_inflight_remote_prefill_tokens);
        let wrapper = Arc::new_cyclic(|weak_self| Self {
            inner,
            role: config.role,
            coordinator: Arc::clone(&coordinator),
            transport,
            worker_hook,
            tokio_handle,
            inflight_budget,
            cd_request_state: DashMap::new(),
            client,
            hub,
            hub_velo_id,
            weak_self: weak_self.clone(),
        });
        coordinator.set_output_sink(Some(wrapper.clone() as Arc<dyn CdOutputSink>));
        wrapper
    }

    pub fn role(&self) -> DisaggregationRole {
        self.role
    }

    fn current_role(&self) -> DisaggregationRole {
        self.role
    }

    pub fn client(&self) -> Option<&Arc<ConditionalDisaggClient>> {
        self.client.as_ref()
    }

    pub fn hub(&self) -> Option<&Arc<HubClient>> {
        self.hub.as_ref()
    }

    pub fn hub_velo_id(&self) -> Option<InstanceId> {
        self.hub_velo_id
    }

    pub fn coordinator(&self) -> &Arc<RemotePrefillCoordinator> {
        &self.coordinator
    }

    pub fn inflight_available(&self) -> usize {
        self.inflight_budget.available()
    }

    /// Returns true if the wrapper has accepted a request as CD-bound and
    /// has not yet completed or failed it.
    pub fn has_active_cd_request(&self, request_id: &str) -> bool {
        self.cd_request_state.contains_key(request_id)
    }

    fn release_request(&self, request_id: &str) {
        if let Some((_, state)) = self.cd_request_state.remove(request_id) {
            self.inflight_budget.release(state.reserved_tokens);
        }
    }

    // ------------------------------------------------------------------
    // GNMT
    // ------------------------------------------------------------------

    fn decode_gnmt(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        let inner_result = self
            .inner
            .get_num_new_matched_tokens(request_id, num_computed_tokens)?;

        if let Some(state) = self.cd_request_state.get(request_id) {
            return Ok((Some(state.reserved_tokens), true));
        }

        let (count, _async_flag) = inner_result;
        let Some(matched_tokens) = count else {
            return Ok(inner_result);
        };

        let total_tokens = self.inner.get_slot_total_tokens(request_id)?;
        let block_size = self.inner.block_size();
        let inputs = PolicyInputs {
            total_tokens,
            num_computed_tokens,
            num_connector_tokens: matched_tokens,
            transfer_params: None,
        };

        match self.coordinator.evaluate(&inputs) {
            PrefillSelection::Local => Ok(inner_result),
            PrefillSelection::Remote => {
                let prefill_window = total_tokens.saturating_sub(num_computed_tokens);
                let full_block_external_tokens = (prefill_window / block_size) * block_size;
                if full_block_external_tokens == 0 {
                    return Ok(inner_result);
                }

                if !self.inflight_budget.try_reserve(full_block_external_tokens) {
                    tracing::debug!(
                        request_id,
                        full_block_external_tokens,
                        available = self.inflight_available(),
                        "remote prefill rejected: inflight token budget exhausted"
                    );
                    return Ok((None, false));
                }

                // From here on, any failure must release the reservation.
                if let Err(err) =
                    self.commit_gnmt_remote(request_id, full_block_external_tokens, &inputs)
                {
                    self.inflight_budget.release(full_block_external_tokens);
                    return Err(err);
                }

                Ok((Some(full_block_external_tokens), true))
            }
        }
    }

    /// Side-effect-heavy half of `decode_gnmt` for the Remote branch:
    /// extract local-match G2 blocks, build the session-blocks payload,
    /// drive `coordinator.begin_remote_prefill`, and stash the
    /// per-request CD state.
    fn commit_gnmt_remote(
        &self,
        request_id: &str,
        full_block_external_tokens: usize,
        inputs: &PolicyInputs,
    ) -> Result<()> {
        // 1. Read split + sequence hashes.
        let split = self.inner.slot_match_split(request_id)?;

        // 2. Take local-match G2 blocks once. Half goes to the session
        //    (clones — ImmutableBlock is Arc-based), half stashed for
        //    USAA-1's local kick.
        let local_g2 = self.inner.take_local_match_g2_blocks(request_id)?;
        if local_g2.len() != split.local_match_blocks {
            anyhow::bail!(
                "GNMT split says {} local-match blocks but find_session yielded {}",
                split.local_match_blocks,
                local_g2.len()
            );
        }
        let session_g2: Vec<_> = local_g2.iter().cloned().collect();
        let pending_hashes: Vec<SequenceHash> = split.expected_remote_hashes();
        let all_remote_hashes = pending_hashes.clone();

        // 3. Token IDs for the prefill peer to compute over.
        let token_ids = self.inner.slot_token_ids(request_id)?;

        // 4. Build the session-blocks payload.
        let session_blocks = SessionBlocks::new(session_g2, pending_hashes);

        // 5. Stash CD state up-front so a duplicate GNMT call hits the
        //    idempotent retry branch even if begin_remote_prefill spans
        //    multiple polls.
        let initiator = self.inner.local_instance_id();
        let new_state = Arc::new(CdRequestState {
            reserved_tokens: full_block_external_tokens,
            pending_local_g2: Mutex::new(Some(local_g2)),
            local_match_g1_block_ids: Vec::new(),
            local_onboard_complete: AtomicBool::new(false),
            remote_slots: Mutex::new(Vec::new()),
            pending_remote_chunks: AtomicUsize::new(0),
            remote_chunks_started: AtomicBool::new(false),
            completed: AtomicBool::new(false),
        });
        self.cd_request_state
            .insert(request_id.to_string(), Arc::clone(&new_state));

        // 6. Drive coordinator.begin_remote_prefill synchronously —
        //    session creation, state install, and monitor spawn all
        //    complete before we return. The queue.enqueue runs on a
        //    spawned task so a slow queue never blocks GNMT.
        let begin_result = self.coordinator.begin_remote_prefill(
            request_id,
            inputs,
            initiator,
            session_blocks,
            all_remote_hashes,
            token_ids,
        );

        match begin_result {
            Ok(_) => Ok(()),
            Err(err) => {
                // Roll back CD state — caller will release the budget.
                self.cd_request_state.remove(request_id);
                Err(err)
            }
        }
    }

    // ------------------------------------------------------------------
    // USAA — both visits
    // ------------------------------------------------------------------

    fn decode_usaa(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        // Drop our own DashMap guard before any inner call to avoid
        // recursive locking surprises.
        let is_active = self.cd_request_state.contains_key(request_id);
        if !is_active {
            // USAA-2 path or a non-CD request. inner USAA with
            // num_external_tokens == 0 is a clean no-op (mod.rs:381).
            if num_external_tokens != 0 {
                tracing::warn!(
                    request_id,
                    num_external_tokens,
                    "USAA-2 reached wrapper with non-zero num_external_tokens; \
                     falling through to inner — slot may not be in PreparingToOnboard"
                );
                debug_assert!(
                    false,
                    "CD USAA-2 unexpectedly carried non-zero num_external_tokens"
                );
            }
            return self
                .inner
                .update_state_after_alloc(request_id, block_ids, num_external_tokens);
        }

        // USAA-1 commitment.
        self.commit_usaa1(request_id, block_ids, num_external_tokens)
    }

    fn commit_usaa1(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        let block_size = self.inner.block_size();
        if !num_external_tokens.is_multiple_of(block_size) {
            anyhow::bail!(
                "CD USAA-1 num_external_tokens {} is not a multiple of block size {}",
                num_external_tokens,
                block_size,
            );
        }

        // 1. Read the slot's match split (host+disk+object hits) before
        //    we touch state. The find_session no longer holds G2 blocks
        //    (GNMT took them), but match_breakdown remains readable.
        let split = self.inner.slot_match_split(request_id)?;

        // 2. Sanity: external = local_match + remote.
        let actual_external_blocks = num_external_tokens / block_size;
        let expected_external_blocks = split.local_match_blocks + split.remote_blocks();
        if expected_external_blocks != actual_external_blocks {
            anyhow::bail!(
                "CD USAA-1 block count mismatch for {}: \
                 split says local={} + remote={} = {} blocks, \
                 vLLM reports num_external_tokens / block_size = {}",
                request_id,
                split.local_match_blocks,
                split.remote_blocks(),
                expected_external_blocks,
                actual_external_blocks,
            );
        }

        // 3. Slice block_ids.
        if block_ids.len() < split.total_blocks {
            anyhow::bail!(
                "CD USAA-1: block_ids len {} < total_blocks {}",
                block_ids.len(),
                split.total_blocks,
            );
        }
        let local_match_g1: Vec<BlockId> = block_ids[split.local_match_range()].to_vec();
        let remote_g1: Vec<BlockId> = block_ids[split.remote_range()].to_vec();
        let expected_remote_hashes = split.expected_remote_hashes();
        let remote_range = split.remote_range();

        // 4. Apply assignments to the slot — this records all G1 block_ids
        //    against the sequence hashes, but does NOT trigger the inner
        //    G2→G1 transfer.
        self.inner.apply_block_assignments(request_id, block_ids)?;

        // 5. Drain the local-match G2 blocks GNMT stashed.
        let existing_state = self
            .cd_request_state
            .get(request_id)
            .map(|e| Arc::clone(e.value()))
            .ok_or_else(|| anyhow!("CD request state missing for {} at USAA-1", request_id))?;
        let local_g2_blocks = existing_state
            .pending_local_g2
            .lock()
            .take()
            .ok_or_else(|| {
                anyhow!(
                    "CD USAA-1: pending_local_g2 already drained for {} (USAA called twice?)",
                    request_id
                )
            })?;
        if local_g2_blocks.len() != split.local_match_blocks {
            anyhow::bail!(
                "CD USAA-1: pending_local_g2 has {} blocks but split says {}",
                local_g2_blocks.len(),
                split.local_match_blocks,
            );
        }
        let local_g2_block_ids: Vec<BlockId> =
            local_g2_blocks.iter().map(|b| b.block_id()).collect();

        // 6. Allocate G2 mutables for the remote slice.
        let remote_count = split.remote_blocks();
        let mutables = if remote_count > 0 {
            self.inner.allocate_g2_blocks(remote_count)?
        } else {
            Vec::new()
        };

        let mut remote_slots: Vec<RemoteBlockSlot> = Vec::with_capacity(remote_count);
        for ((i, hash), (g1, mutable)) in expected_remote_hashes
            .iter()
            .copied()
            .enumerate()
            .zip(remote_g1.iter().copied().zip(mutables.into_iter()))
        {
            let sequence_index = remote_range.start + i;
            remote_slots.push(RemoteBlockSlot {
                expected_hash: hash,
                sequence_index,
                g1_dst_block_id: g1,
                state: RemoteG2State::Mutable(mutable),
            });
        }

        // 7. Mutate the existing CD state with USAA-1-derived fields.
        //    The state was created during GNMT and already holds
        //    reserved_tokens, and was just drained of pending_local_g2
        //    above.
        let new_state = existing_state;
        // local_match_g1_block_ids and remote_slots are committed once
        // here at USAA-1 — `Mutex` ensures we don't read stale values
        // mid-transition. We rebuild a fresh CdRequestState in-place.
        let updated = Arc::new(CdRequestState {
            reserved_tokens: new_state.reserved_tokens,
            pending_local_g2: Mutex::new(None),
            local_match_g1_block_ids: local_match_g1.clone(),
            local_onboard_complete: AtomicBool::new(false),
            remote_slots: Mutex::new(remote_slots),
            pending_remote_chunks: AtomicUsize::new(0),
            remote_chunks_started: AtomicBool::new(false),
            completed: AtomicBool::new(false),
        });
        self.cd_request_state
            .insert(request_id.to_string(), Arc::clone(&updated));
        let new_state = updated;

        // 8. Spawn the local kick (G2 -> G1 for the local-match slice).
        let local_count = split.local_match_blocks;
        if local_count > 0 {
            let transport = Arc::clone(&self.transport);
            let wrapper = self
                .arc_self()
                .ok_or_else(|| anyhow!("wrapper Arc unavailable in commit_usaa1"))?;
            let request_id_owned = request_id.to_string();
            let state_clone = Arc::clone(&new_state);
            self.tokio_handle.spawn(async move {
                // Hold local_g2_blocks until the transfer resolves so the
                // G2 entries stay pinned for the duration of the copy.
                let _hold = local_g2_blocks;
                match transport
                    .local_g2_to_g1(local_g2_block_ids, local_match_g1)
                    .await
                {
                    Ok(()) => {
                        state_clone
                            .local_onboard_complete
                            .store(true, Ordering::Release);
                        wrapper.maybe_complete(&request_id_owned).await;
                    }
                    Err(err) => {
                        wrapper
                            .cleanup_failed_request(
                                &request_id_owned,
                                format!("local G2→G1 transfer failed: {err}"),
                            )
                            .await;
                    }
                }
            });
        } else {
            new_state
                .local_onboard_complete
                .store(true, Ordering::Release);
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // BlockSetsAdded pipeline
    // ------------------------------------------------------------------

    fn drive_block_sets_added(self: &Arc<Self>, request_id: &str, block_sets: Vec<RemoteBlockSet>) {
        let Some(state) = self
            .cd_request_state
            .get(request_id)
            .map(|e| Arc::clone(e.value()))
        else {
            tracing::warn!(
                request_id,
                "BlockSetsAdded for unknown CD request; ignoring"
            );
            return;
        };

        // Match incoming hashes to expected slots and transition Mutable→Staging.
        let mut chunk_hashes: Vec<SequenceHash> = Vec::new();
        let mut chunk_g2_block_ids: Vec<BlockId> = Vec::new();
        let mut chunk_g1_block_ids: Vec<BlockId> = Vec::new();
        let mut chunk_sequence_indices: Vec<usize> = Vec::new();
        for set in &block_sets {
            for block in &set.blocks {
                chunk_hashes.push(block.sequence_hash);
            }
        }

        {
            let mut slots = state.remote_slots.lock();
            for hash in &chunk_hashes {
                let slot = slots
                    .iter_mut()
                    .find(|s| s.expected_hash == *hash)
                    .unwrap_or_else(|| {
                        panic!(
                            "BlockSetsAdded carried hash {:?} not in expected set for {}",
                            hash, request_id
                        )
                    });
                let prev = std::mem::take(&mut slot.state);
                match prev {
                    RemoteG2State::Mutable(m) => {
                        chunk_g2_block_ids.push(m.block_id());
                        chunk_g1_block_ids.push(slot.g1_dst_block_id);
                        chunk_sequence_indices.push(slot.sequence_index);
                        slot.state = RemoteG2State::Staging(m);
                    }
                    other => {
                        panic!(
                            "BlockSetsAdded for {} hit slot in non-Mutable state: {:?}",
                            request_id,
                            std::mem::discriminant(&other)
                        );
                    }
                }
            }
        }

        state.remote_chunks_started.store(true, Ordering::Release);
        state.pending_remote_chunks.fetch_add(1, Ordering::AcqRel);

        // The peer instance id is set by the coordinator's monitor when
        // SessionEvent::Attached lands. BlockSetsAdded only fires after
        // Attached, so this read should always succeed by here.
        let peer_instance_id = match self
            .coordinator
            .state_for(request_id)
            .and_then(|s| s.lock().peer_instance_id)
        {
            Some(id) => id,
            None => {
                tracing::error!(
                    request_id,
                    "BlockSetsAdded with no peer_instance_id on session — Attached missing?"
                );
                let request_id_owned = request_id.to_string();
                let wrapper = Arc::clone(self);
                self.tokio_handle.spawn(async move {
                    wrapper
                        .cleanup_failed_request(
                            &request_id_owned,
                            "peer_instance_id missing at BlockSetsAdded".to_string(),
                        )
                        .await;
                });
                return;
            }
        };

        let request_id = request_id.to_string();
        let wrapper = Arc::clone(self);
        let state = Arc::clone(&state);
        let transport = Arc::clone(&self.transport);
        self.tokio_handle.spawn(async move {
            // 1. RDMA pull from peer P-G2 into our pre-allocated D-G2 mutables.
            if let Err(err) = transport
                .pull_remote(peer_instance_id, block_sets, chunk_g2_block_ids.clone())
                .await
            {
                wrapper
                    .cleanup_failed_request(&request_id, format!("remote pull failed: {err}"))
                    .await;
                return;
            }

            // 2. Local G2 → G1 onboard for those blocks.
            if let Err(err) = transport
                .local_g2_to_g1(chunk_g2_block_ids.clone(), chunk_g1_block_ids)
                .await
            {
                wrapper
                    .cleanup_failed_request(&request_id, format!("CD G2→G1 onboard failed: {err}"))
                    .await;
                return;
            }

            // 3. Pull the matching token blocks for `complete()`.
            let token_blocks = match wrapper.inner.token_blocks_for_range(
                &request_id,
                chunk_sequence_indices.first().copied().unwrap_or(0)
                    ..chunk_sequence_indices.last().copied().unwrap_or(0) + 1,
            ) {
                Ok(tb) => tb,
                Err(err) => {
                    wrapper
                        .cleanup_failed_request(
                            &request_id,
                            format!("token_blocks_for_range failed: {err}"),
                        )
                        .await;
                    return;
                }
            };
            // chunk_sequence_indices may not be contiguous (in theory); we
            // require contiguity for now and panic loudly if violated. The
            // happy path (one BlockSetsAdded = one contiguous chunk) holds
            // for the current test surface.
            let start = chunk_sequence_indices[0];
            for (i, idx) in chunk_sequence_indices.iter().enumerate() {
                debug_assert_eq!(
                    *idx,
                    start + i,
                    "non-contiguous chunk sequence indices: {:?}",
                    chunk_sequence_indices
                );
            }

            // 4. Transition slots Staging → ReadyToBeRegistered (CompleteBlock)
            //    then register them and replace with Registered(ImmutableBlock).
            let mut completes: Vec<CompleteBlock<G2>> = Vec::with_capacity(token_blocks.len());
            {
                let mut slots = state.remote_slots.lock();
                for (idx, token_block) in chunk_sequence_indices.iter().zip(token_blocks.iter()) {
                    let slot = slots
                        .iter_mut()
                        .find(|s| s.sequence_index == *idx)
                        .expect("slot disappeared between Staging and complete()");
                    let prev = std::mem::take(&mut slot.state);
                    match prev {
                        RemoteG2State::Staging(m) => match m.complete(token_block) {
                            Ok(complete) => {
                                slot.state = RemoteG2State::ReadyToBeRegistered(
                                    // We move complete out below; placeholder put back temporarily.
                                    // This is awkward: we want to take CompleteBlock out for register_blocks.
                                    // Use a sentinel and stash later.
                                    complete,
                                );
                            }
                            Err(err) => {
                                panic!(
                                    "MutableBlock::complete failed for {}: {:?}",
                                    request_id, err
                                );
                            }
                        },
                        other => {
                            panic!(
                                "slot transition out of Staging hit non-Staging state: {:?}",
                                std::mem::discriminant(&other)
                            );
                        }
                    }
                }
                // Pull the CompleteBlocks out for batch register, leaving Empty
                // in their place; we'll put Registered back next.
                for idx in &chunk_sequence_indices {
                    let slot = slots
                        .iter_mut()
                        .find(|s| s.sequence_index == *idx)
                        .expect("slot disappeared mid-register");
                    let prev = std::mem::take(&mut slot.state);
                    match prev {
                        RemoteG2State::ReadyToBeRegistered(c) => completes.push(c),
                        other => panic!(
                            "expected ReadyToBeRegistered, got: {:?}",
                            std::mem::discriminant(&other)
                        ),
                    }
                }
            }

            // 5. Register with the leader's G2 manager.
            let registered = match wrapper.inner.register_g2_blocks(completes) {
                Ok(r) => r,
                Err(err) => {
                    wrapper
                        .cleanup_failed_request(
                            &request_id,
                            format!("register_g2_blocks failed: {err}"),
                        )
                        .await;
                    return;
                }
            };

            // 6. Write the ImmutableBlocks back into the slots.
            {
                let mut slots = state.remote_slots.lock();
                for (idx, immutable) in chunk_sequence_indices.iter().zip(registered.into_iter()) {
                    let slot = slots
                        .iter_mut()
                        .find(|s| s.sequence_index == *idx)
                        .expect("slot disappeared mid-register-writeback");
                    slot.state = RemoteG2State::Registered(immutable);
                }
            }

            state.pending_remote_chunks.fetch_sub(1, Ordering::AcqRel);
            wrapper.maybe_complete(&request_id).await;
        });
    }

    // ------------------------------------------------------------------
    // Completion + failure
    // ------------------------------------------------------------------

    async fn maybe_complete(self: &Arc<Self>, request_id: &str) {
        let Some(state) = self
            .cd_request_state
            .get(request_id)
            .map(|e| Arc::clone(e.value()))
        else {
            return;
        };

        if state.completed.load(Ordering::Acquire) {
            return;
        }
        if !state.local_onboard_complete.load(Ordering::Acquire) {
            return;
        }
        if !state.remote_chunks_started.load(Ordering::Acquire) {
            return;
        }
        if state.pending_remote_chunks.load(Ordering::Acquire) != 0 {
            return;
        }
        if !state.all_registered() {
            return;
        }

        // CAS — only one task gets to run completion.
        if state
            .completed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        // Build PullComplete payload from the registered slots.
        let hashes: Vec<SequenceHash> = state
            .remote_slots
            .lock()
            .iter()
            .map(|s| s.expected_hash)
            .collect();

        // Send PullComplete on the session.
        let session = match self.coordinator.state_for(request_id) {
            Some(s) => s.lock().session.clone(),
            None => {
                tracing::warn!(
                    request_id,
                    "maybe_complete: coordinator state missing; cannot send PullComplete"
                );
                return;
            }
        };

        // pull_id derived deterministically from the session_id and the
        // request_id so the prefill peer can correlate. We don't need
        // cryptographic randomness here.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        request_id.hash(&mut hasher);
        let pull = PullComplete {
            pull_id: hasher.finish(),
            hashes,
        };
        if let Err(err) = session.pull_complete_from_decode(pull).await {
            self.cleanup_failed_request(
                request_id,
                format!("pull_complete_from_decode failed: {err}"),
            )
            .await;
            return;
        }

        // Notify workers — the scheduler can now run the request.
        if let Err(err) = self
            .worker_hook
            .mark_onboarding_complete(request_id.to_string())
            .await
        {
            tracing::error!(request_id, error = %err, "mark_onboarding_complete failed");
        }

        // Drop CD state + refund budget.
        self.release_request(request_id);
        // Release the coordinator's session state — it has done its job.
        self.coordinator.release(request_id);
    }

    async fn cleanup_failed_request(self: &Arc<Self>, request_id: &str, reason: String) {
        tracing::warn!(request_id, reason, "CD request failed; cleaning up");
        let unfilled_ids = self
            .cd_request_state
            .get(request_id)
            .map(|e| e.unfilled_g1_block_ids())
            .unwrap_or_default();

        if !unfilled_ids.is_empty() {
            if let Err(err) = self
                .worker_hook
                .mark_failed_onboarding(request_id.to_string(), unfilled_ids)
                .await
            {
                tracing::error!(
                    request_id,
                    error = %err,
                    "mark_failed_onboarding failed during cleanup"
                );
            }
        }

        // Drop CD state — drops MutableBlock<G2>s, which return to reset pool.
        self.release_request(request_id);
        self.coordinator.release(request_id);
    }

    fn prefill_unimplemented(&self, op: &str, request_id: Option<&str>) -> ! {
        tracing::error!(
            op,
            ?request_id,
            "prefill-side conditional-disagg API hit an unimplemented path"
        );
        todo!("prefill-side conditional-disagg API: {op}")
    }
}

impl DecodeDisaggLeader {
    fn arc_self(&self) -> Option<Arc<Self>> {
        self.weak_self.upgrade()
    }
}

impl CdOutputSink for DecodeDisaggLeader {
    fn on_block_sets_added(&self, request_id: &str, block_sets: Vec<RemoteBlockSet>) {
        let Some(arc) = self.arc_self() else {
            tracing::error!(
                request_id,
                "on_block_sets_added: weak_self upgrade failed (wrapper dropped?)"
            );
            return;
        };
        arc.drive_block_sets_added(request_id, block_sets);
    }

    fn on_request_failed(&self, request_id: &str, reason: String) {
        let Some(arc) = self.arc_self() else {
            tracing::error!(
                request_id,
                "on_request_failed: weak_self upgrade failed (wrapper dropped?)"
            );
            return;
        };
        let request_id = request_id.to_string();
        let handle = self.tokio_handle.clone();
        handle.spawn(async move {
            arc.cleanup_failed_request(&request_id, reason).await;
        });
    }
}

impl ConnectorLeaderApi for DecodeDisaggLeader {
    fn create_slot(&self, request: Request) -> Result<()> {
        self.inner.create_slot(request)
    }

    fn has_slot(&self, request_id: &str) -> bool {
        self.inner.has_slot(request_id)
    }

    fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> Result<()> {
        self.inner.extend_slot_tokens(request_id, tokens)
    }

    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        match self.current_role() {
            DisaggregationRole::Decode => self.decode_gnmt(request_id, num_computed_tokens),
            DisaggregationRole::Prefill => {
                self.prefill_unimplemented("get_num_new_matched_tokens", Some(request_id))
            }
        }
    }

    fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        match self.current_role() {
            DisaggregationRole::Decode => {
                self.decode_usaa(request_id, block_ids, num_external_tokens)
            }
            DisaggregationRole::Prefill => {
                self.prefill_unimplemented("update_state_after_alloc", Some(request_id))
            }
        }
    }

    fn build_connector_meta(&self, output: SchedulerOutput) -> Result<KvConnectorMetadata> {
        self.inner.build_connector_meta(output)
    }

    fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()> {
        self.inner
            .update_connector_output(finished_sending, finished_recving)
    }

    fn request_finished(&self, request_id: &str) -> FinishedStatus {
        if self.cd_request_state.contains_key(request_id) {
            self.release_request(request_id);
        }
        self.inner.request_finished(request_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inflight_budget_unlimited_capacity_skips_atomic() {
        let budget = InflightBudget::new(usize::MAX);
        assert!(budget.try_reserve(1_000_000));
        assert_eq!(budget.available(), usize::MAX);
        budget.release(1_000_000);
        assert_eq!(budget.available(), usize::MAX);
    }

    #[test]
    fn inflight_budget_reserve_then_release_balances() {
        let budget = InflightBudget::new(256);
        assert!(budget.try_reserve(64));
        assert_eq!(budget.available(), 192);
        budget.release(64);
        assert_eq!(budget.available(), 256);
    }

    #[test]
    fn inflight_budget_exhausted_reservation_returns_false() {
        let budget = InflightBudget::new(100);
        assert!(budget.try_reserve(64));
        assert!(!budget.try_reserve(64));
        assert_eq!(budget.available(), 36);
    }

    #[test]
    fn inflight_budget_partial_reservation_succeeds_when_fits() {
        let budget = InflightBudget::new(100);
        assert!(budget.try_reserve(64));
        assert!(budget.try_reserve(32));
        assert_eq!(budget.available(), 4);
        assert!(!budget.try_reserve(8));
        assert!(budget.try_reserve(4));
        assert_eq!(budget.available(), 0);
    }

    #[test]
    fn inflight_budget_zero_reservation_is_a_noop() {
        let budget = InflightBudget::new(64);
        assert!(budget.try_reserve(0));
        assert_eq!(budget.available(), 64);
    }

    // ----------------------------------------------------------------
    // RemoteG2State / CdRequestState pure-data tests.
    // ----------------------------------------------------------------

    fn cd_state_for_test(
        local_g1: Vec<BlockId>,
        remote_g1: Vec<BlockId>,
        reserved: usize,
    ) -> CdRequestState {
        let remote_slots = remote_g1
            .into_iter()
            .enumerate()
            .map(|(i, g1)| RemoteBlockSlot {
                expected_hash: dynamo_tokens::PositionalLineageHash::new(i as u64, None, i as u64),
                sequence_index: i,
                g1_dst_block_id: g1,
                state: RemoteG2State::Empty,
            })
            .collect();
        CdRequestState {
            reserved_tokens: reserved,
            pending_local_g2: Mutex::new(None),
            local_match_g1_block_ids: local_g1,
            local_onboard_complete: AtomicBool::new(false),
            remote_slots: Mutex::new(remote_slots),
            pending_remote_chunks: AtomicUsize::new(0),
            remote_chunks_started: AtomicBool::new(false),
            completed: AtomicBool::new(false),
        }
    }

    #[test]
    fn unfilled_includes_local_when_local_onboard_not_complete() {
        let state = cd_state_for_test(vec![10, 11], vec![20, 21, 22], 80);
        let unfilled = state.unfilled_g1_block_ids();
        assert_eq!(unfilled, vec![10, 11, 20, 21, 22]);
    }

    #[test]
    fn unfilled_excludes_local_after_local_onboard_complete() {
        let state = cd_state_for_test(vec![10, 11], vec![20, 21, 22], 80);
        state.local_onboard_complete.store(true, Ordering::Release);
        let unfilled = state.unfilled_g1_block_ids();
        assert_eq!(unfilled, vec![20, 21, 22]);
    }

    #[test]
    fn all_registered_false_when_any_slot_not_registered() {
        let state = cd_state_for_test(vec![], vec![20, 21], 32);
        assert!(!state.all_registered());
    }

    #[test]
    fn remote_g2_state_is_registered_only_for_registered_variant() {
        // We can't easily build an ImmutableBlock<G2> without the full
        // BlockManager dance; cover the negative cases.
        assert!(!RemoteG2State::Empty.is_registered());
    }
}
