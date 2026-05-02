// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode-side conditional-disaggregation leader wrapper.
//!
//! `DecodeDisaggLeader` wraps a base [`super::ConnectorLeaderApi`]
//! and intercepts the scheduler-facing API to drive the
//! conditional-disagg dataflow on the decode side, against the
//! symmetric [`Session`](kvbm_engine::disagg::session::Session)
//! API.
//!
//! ### Pipelines
//!
//! Decode runs two parallel pipelines per CD-bound request:
//!
//! 1. **Local kick** — G2→G1 transfer for decode's local-match
//!    slice, kicked at USAA-1 from the cached G2 pins.
//! 2. **Remote pull** — subscribe `session.commits()` /
//!    `availability()`, drain, and per-chunk: alloc G2 mutables,
//!    `session.pull`, complete, register, transport.local_g2_to_g1
//!    onboard for that chunk's slice.
//!
//! `mark_workers_onboarding_complete` fires only when **both**
//! pipelines have reported completion (CAS-gated). Either-side
//! failure tears down via `cleanup_failed_request`.
//!
//! Phase A error-path semantics (peer detach, attach timeout)
//! remain deferred — see the canonical plan §"Phase A
//! error-path design (deferred)".

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use anyhow::{Result, anyhow};
use dashmap::DashMap;
use futures::StreamExt;
use kvbm_config::{DisaggConfig, DisaggregationRole};
use kvbm_engine::disagg::session::{AvailabilityDelta, CommitDelta, Session};
use kvbm_hub::{ConditionalDisaggClient, HubClient};
use kvbm_logical::blocks::{CompleteBlock, ImmutableBlock};
use parking_lot::Mutex;
use velo::InstanceId;

use crate::BlockId;
use crate::connector::leader::scheduler::{KvConnectorMetadata, SchedulerOutput};
use crate::connector::leader::{FinishedStatus, Request};
use crate::{G2, SequenceHash};

use super::decode::{CdFailureSink, DecodeCoordinator};
use super::transport::{CdBlockTransport, CdWorkerHook, InnerLeaderShim};
use super::{ConnectorLeaderApi, PolicyInputs, PrefillSelection};
use futures::FutureExt;
use futures::future::BoxFuture;

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

/// Position-indexed metadata for one block in the remote-prefill
/// slice. We do not hold a G2 mutable here — those are allocated
/// in the pull task on availability and consumed by `session.pull`.
struct RemoteSlotMeta {
    sequence_index: usize,
    g1_dst_block_id: BlockId,
}

/// RAII payload installed on the slot's `OnboardingState` when a
/// CD-decode request enters the wrapper.
///
/// Its `Drop` impl runs the canonical CD cleanup — releasing the
/// inflight-budget reservation and dropping the wrapper's
/// `cd_request_state` entry + the coordinator's session state. The
/// payload is dropped exactly once: when
/// [`crate::connector::leader::ConnectorLeader::process_finished_onboarding`]
/// takes the slot's `OnboardingState`. Since cleanup is idempotent
/// (DashMap entries simply remove no-ops when missing), explicit
/// fallback paths (`request_finished` on untracked slots, error
/// recovery) can also drop it without coordination.
struct CdRequestStatePayload {
    request_id: String,
    reserved_tokens: usize,
    wrapper: Weak<DecodeDisaggLeader>,
    coordinator: Weak<dyn DecodeCoordinator>,
}

impl std::fmt::Debug for CdRequestStatePayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CdRequestStatePayload")
            .field("request_id", &self.request_id)
            .field("reserved_tokens", &self.reserved_tokens)
            .finish()
    }
}

impl crate::connector::leader::slot::CdOnboardingPayload for CdRequestStatePayload {}

impl Drop for CdRequestStatePayload {
    fn drop(&mut self) {
        crate::audit!(
            "cd_payload_drop",
            role = "decode",
            request_id = %self.request_id,
            reserved_tokens = self.reserved_tokens
        );
        if let Some(w) = self.wrapper.upgrade() {
            // Removes from cd_request_state + releases inflight budget.
            // Idempotent against earlier release calls.
            w.release_request(&self.request_id);
        }
        if let Some(c) = self.coordinator.upgrade() {
            // Drops the holder-side session and per-request state.
            c.release(&self.request_id);
        }
    }
}


struct CdRequestState {
    reserved_tokens: usize,

    /// Pinned local-match G2 (Arc clones).  Held until the local
    /// G2→G1 kick resolves so the G2 entries stay live for the
    /// duration of the copy. `session.make_available` holds its
    /// own clones; this is the wrapper's independent pin set.
    local_match_g2_pins: Mutex<Option<Vec<ImmutableBlock<G2>>>>,
    local_match_g2_block_ids: Vec<BlockId>,

    /// G1 destinations vLLM allocated for the local-match slice
    /// `[computed, computed + local_match)`.
    local_match_g1_block_ids: Vec<BlockId>,
    local_onboard_complete: AtomicBool,

    /// Per-position remote-slice metadata, in expected order.
    /// Built at USAA-1 and read-only afterward.
    remote_slots: Vec<RemoteSlotMeta>,
    /// `expected_hash → index in remote_slots` lookup; built once.
    remote_slot_index: HashMap<SequenceHash, usize>,

    remote_pipeline_complete: AtomicBool,
    completed: AtomicBool,
}

impl CdRequestState {
    fn unfilled_g1_block_ids(&self) -> Vec<BlockId> {
        let mut out = Vec::new();
        if !self.local_onboard_complete.load(Ordering::Acquire) {
            out.extend(self.local_match_g1_block_ids.iter().copied());
        }
        if !self.remote_pipeline_complete.load(Ordering::Acquire) {
            out.extend(self.remote_slots.iter().map(|s| s.g1_dst_block_id));
        }
        out
    }
}

// ============================================================================
// Wrapper
// ============================================================================

pub struct DecodeDisaggLeader {
    inner: Arc<dyn InnerLeaderShim>,
    role: DisaggregationRole,
    coordinator: Arc<dyn DecodeCoordinator>,
    transport: Arc<dyn CdBlockTransport>,
    worker_hook: Arc<dyn CdWorkerHook>,
    tokio_handle: tokio::runtime::Handle,

    inflight_budget: InflightBudget,
    cd_request_state: DashMap<String, Arc<CdRequestState>>,

    client: Option<Arc<ConditionalDisaggClient>>,
    hub: Option<Arc<HubClient>>,
    hub_velo_id: Option<InstanceId>,

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
    /// Construct a `DecodeDisaggLeader` with any type that implements
    /// [`DecodeCoordinator`].  Accepts both [`RemotePrefillCoordinator`]
    /// (legacy path) and [`super::coordinator::ConditionalDisaggCoordinator`]
    /// (R-B Slice 4) without requiring an explicit coercion at the call
    /// site.
    pub fn from_parts<C: DecodeCoordinator + 'static>(
        inner: Arc<dyn InnerLeaderShim>,
        config: &DisaggConfig,
        coordinator: Arc<C>,
        transport: Arc<dyn CdBlockTransport>,
        worker_hook: Arc<dyn CdWorkerHook>,
        tokio_handle: tokio::runtime::Handle,
        hub: Option<Arc<HubClient>>,
        client: Option<Arc<ConditionalDisaggClient>>,
        hub_velo_id: Option<InstanceId>,
    ) -> Arc<Self> {
        let coordinator: Arc<dyn DecodeCoordinator> = coordinator;
        let inflight_budget = InflightBudget::new(config.max_inflight_remote_prefill_tokens);
        let leader = Arc::new_cyclic(|weak_self| Self {
            inner,
            role: config.role,
            coordinator,
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
        // Install the failure sink AFTER the leader Arc exists so
        // the watchdog/Failed path can call back into the wrapper's
        // `cleanup_failed_request` (which fires
        // `worker_hook.mark_failed_onboarding`).  Without this,
        // a failed peer would leave vLLM hanging in `Onboarding`
        // until something else aborts the request.
        let sink: Weak<dyn CdFailureSink> =
            Arc::downgrade(&leader) as Weak<dyn CdFailureSink>;
        leader.coordinator.install_failure_sink(sink);
        leader
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

    pub fn coordinator(&self) -> &Arc<dyn DecodeCoordinator> {
        &self.coordinator
    }

    pub fn inflight_available(&self) -> usize {
        self.inflight_budget.available()
    }

    pub fn has_active_cd_request(&self, request_id: &str) -> bool {
        self.cd_request_state.contains_key(request_id)
    }

    fn release_request(&self, request_id: &str) {
        if let Some((_, state)) = self.cd_request_state.remove(request_id) {
            self.inflight_budget.release(state.reserved_tokens);
        }
    }

    fn arc_self(&self) -> Option<Arc<Self>> {
        self.weak_self.upgrade()
    }

    // ------------------------------------------------------------------
    // GNMT
    // ------------------------------------------------------------------

    #[tracing::instrument(level = "info", skip(self), fields(role = ?self.role))]
    fn decode_gnmt(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        let inner_result = self
            .inner
            .get_num_new_matched_tokens(request_id, num_computed_tokens)?;
        tracing::info!(?inner_result, "decode_gnmt: inner returned");

        if let Some(state) = self.cd_request_state.get(request_id) {
            tracing::info!(
                reserved_tokens = state.reserved_tokens,
                "decode_gnmt: idempotent — already CD-tracked"
            );
            crate::audit!(
                "gnmt_idempotent",
                role = "decode",
                request_id,
                reserved_tokens = state.reserved_tokens
            );
            return Ok((Some(state.reserved_tokens), true));
        }

        let (count, _async_flag) = inner_result;
        let Some(matched_tokens) = count else {
            tracing::info!("decode_gnmt: inner returned None — passthrough");
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

        let selection = self.coordinator.evaluate(&inputs);
        tracing::info!(
            ?selection,
            total_tokens,
            block_size,
            matched_tokens,
            "decode_gnmt: policy decision"
        );
        crate::audit!(
            "policy_decision",
            role = "decode",
            request_id,
            selection = ?selection,
            total_tokens,
            block_size,
            matched_tokens
        );
        match selection {
            PrefillSelection::Local => {
                tracing::info!(?inner_result, "decode_gnmt: Local — passthrough");
                Ok(inner_result)
            }
            PrefillSelection::Remote => {
                let prefill_window = total_tokens.saturating_sub(num_computed_tokens);
                let full_block_external_tokens = (prefill_window / block_size) * block_size;
                tracing::info!(
                    prefill_window,
                    full_block_external_tokens,
                    "decode_gnmt: Remote — sized window"
                );
                if full_block_external_tokens == 0 {
                    tracing::info!(
                        "decode_gnmt: Remote but no full block to send — passthrough"
                    );
                    crate::audit!(
                        "policy_remote_passthrough_zero_block",
                        role = "decode",
                        request_id,
                        prefill_window
                    );
                    return Ok(inner_result);
                }

                if !self.inflight_budget.try_reserve(full_block_external_tokens) {
                    tracing::warn!(
                        full_block_external_tokens,
                        available = self.inflight_available(),
                        "decode_gnmt: remote prefill rejected — inflight budget exhausted"
                    );
                    return Ok((None, false));
                }

                if let Err(err) =
                    self.commit_gnmt_remote(request_id, full_block_external_tokens, &inputs)
                {
                    tracing::error!(error = %err, "decode_gnmt: commit_gnmt_remote failed");
                    self.inflight_budget.release(full_block_external_tokens);
                    return Err(err);
                }

                tracing::info!(
                    full_block_external_tokens,
                    "decode_gnmt: queued remote prefill — returning (Some(N), true)"
                );
                crate::audit!(
                    "remote_prefill_queued",
                    role = "decode",
                    request_id,
                    full_block_external_tokens
                );
                Ok((Some(full_block_external_tokens), true))
            }
        }
    }

    /// GNMT-Remote side-effects: extract local-match G2, drive
    /// `coordinator.begin_remote_prefill` (which opens the session,
    /// commits + makes-available the local-match, and queues the
    /// request), and stash the wrapper's per-request CD state.
    #[tracing::instrument(level = "info", skip(self, inputs), fields(role = ?self.role))]
    fn commit_gnmt_remote(
        &self,
        request_id: &str,
        full_block_external_tokens: usize,
        inputs: &PolicyInputs,
    ) -> Result<()> {
        let split = self.inner.slot_match_split(request_id)?;
        tracing::info!(
            local_match_blocks = split.local_match_blocks,
            computed_blocks = split.computed_blocks,
            total_blocks = split.total_blocks,
            "commit_gnmt_remote: slot_match_split"
        );

        let local_g2 = self.inner.take_local_match_g2_blocks(request_id)?;
        tracing::info!(
            local_g2_len = local_g2.len(),
            "commit_gnmt_remote: took local-match G2 blocks"
        );
        if local_g2.len() != split.local_match_blocks {
            anyhow::bail!(
                "GNMT split says {} local-match blocks but find_session yielded {}",
                split.local_match_blocks,
                local_g2.len()
            );
        }
        let num_local_match_hashes = local_g2.len();
        let local_match_g2_block_ids: Vec<BlockId> =
            local_g2.iter().map(|b| b.block_id()).collect();
        // Independent pin set: the wrapper holds its own clones to
        // keep the local-match G2 entries pinned for the local-kick.
        // `session.make_available` will get its own clones inside
        // `coordinator.begin_remote_prefill`.
        let session_local_g2: Vec<ImmutableBlock<G2>> = local_g2.to_vec();

        let all_token_ids = self.inner.slot_token_ids(request_id)?;
        // Slice tokens to the prefill window: only the full-block
        // prefix `[num_computed_tokens, num_computed_tokens +
        // full_block_external_tokens)`. The partial tail block (if
        // any) and decode-already-computed prefix stay on decode.
        let base_offset = inputs.num_computed_tokens;
        let prefill_window_end = base_offset + full_block_external_tokens;
        if prefill_window_end > all_token_ids.len() {
            anyhow::bail!(
                "prefill window [{}..{}] out of bounds for {} tokens",
                base_offset,
                prefill_window_end,
                all_token_ids.len(),
            );
        }
        let prefill_token_ids: Vec<u32> =
            all_token_ids[base_offset..prefill_window_end].to_vec();

        let new_state = Arc::new(CdRequestState {
            reserved_tokens: full_block_external_tokens,
            local_match_g2_pins: Mutex::new(Some(local_g2)),
            local_match_g2_block_ids,
            local_match_g1_block_ids: Vec::new(),
            local_onboard_complete: AtomicBool::new(false),
            remote_slots: Vec::new(),
            remote_slot_index: HashMap::new(),
            remote_pipeline_complete: AtomicBool::new(false),
            completed: AtomicBool::new(false),
        });
        self.cd_request_state
            .insert(request_id.to_string(), Arc::clone(&new_state));

        let initiator = self.inner.local_instance_id();
        tracing::info!(
            %initiator,
            num_prefill_token_ids = prefill_token_ids.len(),
            num_session_local_g2 = session_local_g2.len(),
            "commit_gnmt_remote: begin_remote_prefill"
        );
        crate::audit!(
            "begin_remote_prefill_call",
            role = "decode",
            request_id,
            num_prefill_tokens = prefill_token_ids.len(),
            num_session_local_g2 = session_local_g2.len(),
            num_local_match_hashes,
            full_block_external_tokens,
            base_offset
        );
        match self.coordinator.begin_remote_prefill(
            request_id,
            inputs,
            initiator,
            session_local_g2,
            prefill_token_ids,
        ) {
            Ok(outcome) => {
                tracing::info!(
                    session_id = %outcome.session_id,
                    "commit_gnmt_remote: begin_remote_prefill ok"
                );
                // Install the RAII payload on the slot's
                // OnboardingState. This is what brings the slot's
                // canonical transaction state machine in sync with
                // the (Some(N), true) async-load promise we're
                // about to return to vLLM. When
                // process_finished_onboarding takes the
                // OnboardingState, the payload's Drop runs the
                // canonical cleanup chain.
                let coordinator_weak: Weak<dyn DecodeCoordinator> =
                    Arc::downgrade(&self.coordinator);
                let payload = Box::new(CdRequestStatePayload {
                    request_id: request_id.to_string(),
                    reserved_tokens: full_block_external_tokens,
                    wrapper: self.weak_self.clone(),
                    coordinator: coordinator_weak,
                });
                if let Err(err) =
                    self.inner.install_cd_onboarding_payload(request_id, payload)
                {
                    tracing::error!(
                        error = %err,
                        "commit_gnmt_remote: install_cd_onboarding_payload failed; \
                         rolling back cd_request_state"
                    );
                    crate::audit!(
                        "cd_payload_install_failed",
                        role = "decode",
                        request_id,
                        error = %err
                    );
                    // Roll back cd_request_state and coordinator; the
                    // RAII Drop never gets to run for this request.
                    self.cd_request_state.remove(request_id);
                    self.coordinator.release(request_id);
                    return Err(err);
                }
                crate::audit!(
                    "cd_payload_installed",
                    role = "decode",
                    request_id,
                    reserved_tokens = full_block_external_tokens
                );
                Ok(())
            }
            Err(err) => {
                tracing::error!(error = %err, "commit_gnmt_remote: begin_remote_prefill failed");
                self.cd_request_state.remove(request_id);
                Err(err)
            }
        }
    }

    // ------------------------------------------------------------------
    // USAA
    // ------------------------------------------------------------------

    fn decode_usaa(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        let is_active = self.cd_request_state.contains_key(request_id);
        if !is_active {
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

        let split = self.inner.slot_match_split(request_id)?;

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

        self.inner.apply_block_assignments(request_id, block_ids)?;

        // Drain stashed local-match pins + ids and rebuild per-
        // request state with the USAA-1 derived fields.
        let existing = self
            .cd_request_state
            .get(request_id)
            .map(|e| Arc::clone(e.value()))
            .ok_or_else(|| anyhow!("CD request state missing for {} at USAA-1", request_id))?;
        let local_match_g2_pins = existing.local_match_g2_pins.lock().take().ok_or_else(|| {
            anyhow!(
                "CD USAA-1: local_match_g2_pins already drained for {} (USAA called twice?)",
                request_id
            )
        })?;
        if local_match_g2_pins.len() != split.local_match_blocks {
            anyhow::bail!(
                "CD USAA-1: local_match_g2_pins has {} blocks but split says {}",
                local_match_g2_pins.len(),
                split.local_match_blocks,
            );
        }

        let mut remote_slots: Vec<RemoteSlotMeta> =
            Vec::with_capacity(expected_remote_hashes.len());
        let mut remote_slot_index: HashMap<SequenceHash, usize> = HashMap::new();
        for ((i, hash), g1) in expected_remote_hashes
            .iter()
            .copied()
            .enumerate()
            .zip(remote_g1.iter().copied())
        {
            let sequence_index = remote_range.start + i;
            remote_slot_index.insert(hash, remote_slots.len());
            remote_slots.push(RemoteSlotMeta {
                sequence_index,
                g1_dst_block_id: g1,
            });
        }

        let updated = Arc::new(CdRequestState {
            reserved_tokens: existing.reserved_tokens,
            local_match_g2_pins: Mutex::new(None),
            local_match_g2_block_ids: existing.local_match_g2_block_ids.clone(),
            local_match_g1_block_ids: local_match_g1.clone(),
            local_onboard_complete: AtomicBool::new(false),
            remote_slots,
            remote_slot_index,
            remote_pipeline_complete: AtomicBool::new(false),
            completed: AtomicBool::new(false),
        });
        self.cd_request_state
            .insert(request_id.to_string(), Arc::clone(&updated));

        crate::audit!(
            "commit_usaa1_state_built",
            role = "decode",
            request_id,
            local_match_blocks = split.local_match_blocks,
            remote_slots_len = updated.remote_slots.len(),
            local_match_g1_len = updated.local_match_g1_block_ids.len(),
            expected_remote_hashes_len = expected_remote_hashes.len()
        );

        // Spawn the local kick.
        let local_count = split.local_match_blocks;
        if local_count > 0 {
            let transport = Arc::clone(&self.transport);
            let wrapper = self
                .arc_self()
                .ok_or_else(|| anyhow!("wrapper Arc unavailable in commit_usaa1"))?;
            let request_id_owned = request_id.to_string();
            let state_clone = Arc::clone(&updated);
            let local_g2_block_ids = updated.local_match_g2_block_ids.clone();
            self.tokio_handle.spawn(async move {
                // Hold pins for the duration of the copy.
                let _hold = local_match_g2_pins;
                match transport
                    .local_g2_to_g1(local_g2_block_ids, local_match_g1)
                    .await
                {
                    Ok(()) => {
                        state_clone
                            .local_onboard_complete
                            .store(true, Ordering::Release);
                        crate::audit!(
                            "local_onboard_complete_set",
                            role = "decode",
                            request_id = %request_id_owned,
                            reason = "local_g2_to_g1_done"
                        );
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
            updated
                .local_onboard_complete
                .store(true, Ordering::Release);
            crate::audit!(
                "local_onboard_complete_set",
                role = "decode",
                request_id,
                reason = "no_local_match_blocks"
            );
        }

        // Spawn the remote pull pipeline. If there's no remote
        // slice, mark it complete immediately and let
        // maybe_complete fire on local-kick completion alone.
        if updated.remote_slots.is_empty() {
            updated
                .remote_pipeline_complete
                .store(true, Ordering::Release);
            crate::audit!(
                "remote_pipeline_complete_set",
                role = "decode",
                request_id,
                reason = "remote_slots_empty"
            );
        } else {
            let session = match self.coordinator.session_for(request_id) {
                Some(s) => s,
                None => {
                    anyhow::bail!("CD USAA-1: coordinator has no session for {}", request_id);
                }
            };
            let wrapper = self
                .arc_self()
                .ok_or_else(|| anyhow!("wrapper Arc unavailable in commit_usaa1"))?;
            let request_id_owned = request_id.to_string();
            let state_clone = Arc::clone(&updated);
            self.tokio_handle.spawn(async move {
                if let Err(err) = wrapper
                    .run_remote_pipeline(&request_id_owned, state_clone, session)
                    .await
                {
                    wrapper
                        .cleanup_failed_request(&request_id_owned, err.to_string())
                        .await;
                }
            });
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // Remote pull pipeline
    // ------------------------------------------------------------------

    /// Subscribe `session.commits()` + `session.availability()`,
    /// drain commits (informational), and per-availability-chunk:
    /// validate hashes ⊆ expected_remote_hashes (panic if not),
    /// alloc G2 mutables, `session.pull(...)`, complete, register,
    /// transport.local_g2_to_g1 onboard.  When all expected
    /// remote hashes are filled, set `remote_pipeline_complete`
    /// and call `maybe_complete`.
    async fn run_remote_pipeline(
        self: &Arc<Self>,
        request_id: &str,
        state: Arc<CdRequestState>,
        session: Arc<dyn Session>,
    ) -> Result<()> {
        // 1. Drain commits opportunistically. Break as soon as
        //    we've seen all expected remote hashes; if peer signals
        //    Closed before that, treat it as a protocol-level
        //    failure — prefill said "no more commits coming" but
        //    didn't deliver everything decode expected, so the
        //    request is unrecoverable. Caller's spawn handler
        //    routes the Err to `cleanup_failed_request` (#8 Slice C).
        let expected_count = state.remote_slots.len();
        let mut commit_seen: HashSet<SequenceHash> = HashSet::new();
        let mut commits = session.commits();
        while let Some(d) = commits.next().await {
            match d {
                CommitDelta::Added(hashes) => {
                    for h in hashes {
                        commit_seen.insert(h);
                    }
                    if commit_seen.len() >= expected_count {
                        break;
                    }
                }
                CommitDelta::Closed => {
                    if commit_seen.len() < expected_count {
                        crate::audit!(
                            "decode_commits_closed_short",
                            role = "decode",
                            request_id,
                            seen = commit_seen.len(),
                            expected = expected_count
                        );
                        anyhow::bail!(
                            "commits Closed before all expected hashes arrived for {} \
                             (got {} of {})",
                            request_id,
                            commit_seen.len(),
                            expected_count
                        );
                    }
                    break;
                }
            }
        }
        drop(commits);

        // 2. Drain availability and pull each chunk.
        let mut filled: HashSet<SequenceHash> = HashSet::new();
        let mut avail = session.availability();
        while let Some(d) = avail.next().await {
            match d {
                AvailabilityDelta::Available(blocks) => {
                    // Validate every incoming hash is expected.
                    // Any unexpected hash is a protocol violation
                    // — panic loud (mirrors prefill_coordinator).
                    for b in &blocks {
                        if !state.remote_slot_index.contains_key(&b.hash) {
                            panic!(
                                "availability carried hash {:?} not in expected_remote_hashes for {}",
                                b.hash, request_id
                            );
                        }
                    }
                    let chunk: Vec<_> = blocks
                        .into_iter()
                        .filter(|b| !filled.contains(&b.hash))
                        .collect();
                    if chunk.is_empty() {
                        continue;
                    }
                    let chunk_hashes: Vec<SequenceHash> = chunk.iter().map(|b| b.hash).collect();
                    self.pull_register_onboard_chunk(
                        request_id,
                        &state,
                        chunk_hashes.clone(),
                        Arc::clone(&session),
                    )
                    .await?;
                    filled.extend(chunk_hashes);
                    if filled.len() == state.remote_slots.len() {
                        break;
                    }
                }
                AvailabilityDelta::Drained => break,
            }
        }
        drop(avail);

        if filled.len() != state.remote_slots.len() {
            anyhow::bail!(
                "availability drained with {} of {} remote hashes filled for {}",
                filled.len(),
                state.remote_slots.len(),
                request_id
            );
        }

        state
            .remote_pipeline_complete
            .store(true, Ordering::Release);
        crate::audit!(
            "remote_pipeline_complete_set",
            role = "decode",
            request_id,
            reason = "all_remote_pulls_filled",
            filled = filled.len()
        );
        self.maybe_complete(request_id).await;
        Ok(())
    }

    /// Pull a chunk into freshly-allocated G2 mutables, complete
    /// + register, and run the G2→G1 onboard for the chunk's
    /// slice in one shot.  Maintains positional ordering against
    /// `remote_slots` via `sequence_index`.
    async fn pull_register_onboard_chunk(
        self: &Arc<Self>,
        request_id: &str,
        state: &Arc<CdRequestState>,
        hashes: Vec<SequenceHash>,
        session: Arc<dyn Session>,
    ) -> Result<()> {
        crate::audit!(
            "worker_pull_chunk_start",
            role = "decode",
            request_id,
            num_hashes = hashes.len()
        );
        // Reorder hashes by their slot position so chunks pair
        // with the correct G1 destination + token block.  Chunks
        // may arrive in any order on the wire, but positional
        // order is required for `MutableBlock::complete`.
        let mut indexed: Vec<(usize, SequenceHash)> = hashes
            .into_iter()
            .map(|h| {
                let slot_idx = *state
                    .remote_slot_index
                    .get(&h)
                    .expect("validated in run_remote_pipeline");
                (slot_idx, h)
            })
            .collect();
        indexed.sort_by_key(|(slot_idx, _)| *slot_idx);

        // Require contiguous chunk (slot-index space).
        let first_slot = indexed[0].0;
        for (i, (slot_idx, _)) in indexed.iter().enumerate() {
            if *slot_idx != first_slot + i {
                anyhow::bail!(
                    "non-contiguous chunk: expected slot {}, got {} (chunk: {:?})",
                    first_slot + i,
                    slot_idx,
                    indexed.iter().map(|(i, _)| *i).collect::<Vec<_>>()
                );
            }
        }

        let chunk_size = indexed.len();
        let ordered_hashes: Vec<SequenceHash> = indexed.iter().map(|(_, h)| *h).collect();
        let chunk_g1_block_ids: Vec<BlockId> = indexed
            .iter()
            .map(|(slot_idx, _)| state.remote_slots[*slot_idx].g1_dst_block_id)
            .collect();
        let chunk_sequence_indices: Vec<usize> = indexed
            .iter()
            .map(|(slot_idx, _)| state.remote_slots[*slot_idx].sequence_index)
            .collect();

        // 1. Alloc G2 mutables + RDMA pull.
        let dst = self.inner.allocate_g2_blocks(chunk_size)?;
        crate::audit!(
            "worker_session_pull_call",
            role = "decode",
            request_id,
            num_hashes = ordered_hashes.len(),
            num_dst = dst.len()
        );
        let filled = session.pull(ordered_hashes, dst).await?;
        crate::audit!(
            "worker_session_pull_returned",
            role = "decode",
            request_id,
            num_filled = filled.len()
        );

        // 2. Pull token blocks for the chunk's positions to
        //    drive `MutableBlock::complete`.
        let token_range_start = chunk_sequence_indices[0];
        let token_range_end = *chunk_sequence_indices.last().unwrap() + 1;
        let token_blocks = self
            .inner
            .token_blocks_for_range(request_id, token_range_start..token_range_end)?;

        let mut completes: Vec<CompleteBlock<G2>> = Vec::with_capacity(chunk_size);
        for (mutable, token_block) in filled.into_iter().zip(token_blocks.iter()) {
            completes.push(
                mutable
                    .complete(token_block)
                    .map_err(|err| anyhow!("MutableBlock::complete failed: {:?}", err))?,
            );
        }

        // 3. Register with the leader's G2 manager.
        let registered = self.inner.register_g2_blocks(completes)?;
        let chunk_g2_block_ids: Vec<BlockId> = registered.iter().map(|b| b.block_id()).collect();

        // 4. Local G2→G1 onboard for this chunk's slice.
        self.transport
            .local_g2_to_g1(chunk_g2_block_ids.clone(), chunk_g1_block_ids.clone())
            .await?;
        crate::audit!(
            "worker_g2_to_g1_done",
            role = "decode",
            request_id,
            num_blocks = chunk_g1_block_ids.len()
        );

        // Hold registered pins for the lifetime of the request —
        // drop happens via release_request → state drop.
        // (We don't need to thread them anywhere; dropping them
        // here is fine because `register_g2_blocks` already keeps
        // the entries in the manager's hot map.)
        drop(registered);
        Ok(())
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

        let local_done = state.local_onboard_complete.load(Ordering::Acquire);
        let remote_done = state.remote_pipeline_complete.load(Ordering::Acquire);
        crate::audit!(
            "maybe_complete_check",
            role = "decode",
            request_id,
            local_onboard_complete = local_done,
            remote_pipeline_complete = remote_done,
            already_completed = state.completed.load(Ordering::Acquire)
        );

        if state.completed.load(Ordering::Acquire) {
            return;
        }
        if !local_done {
            return;
        }
        if !remote_done {
            return;
        }

        if state
            .completed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        crate::audit!(
            "mark_onboarding_complete",
            role = "decode",
            request_id
        );
        if let Err(err) = self
            .worker_hook
            .mark_onboarding_complete(request_id.to_string())
            .await
        {
            tracing::error!(request_id, error = %err, "mark_onboarding_complete failed");
            crate::audit!(
                "mark_onboarding_complete_error",
                role = "decode",
                request_id,
                error = %err
            );
        }

        // Eager release so the inflight budget reopens immediately
        // and the next `cd_request_state` lookup sees an empty slot.
        // This is a hot-path optimization, not the canonical cleanup
        // — that runs via the `CdRequestStatePayload`'s `Drop` when
        // `process_finished_onboarding` takes the slot's
        // `OnboardingState` on `finished_recving`. Both
        // `release_request` and `coordinator.release` are idempotent
        // (DashMap `remove` returns `None` if absent), so the
        // duplicate Drop is a no-op.
        self.release_request(request_id);
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

        self.release_request(request_id);
        self.coordinator.release(request_id);
    }

}

// ============================================================================
// CdFailureSink — coordinator → wrapper failure callback
// ============================================================================

impl CdFailureSink for DecodeDisaggLeader {
    fn on_session_failure(&self, request_id: String, reason: String) -> BoxFuture<'static, ()> {
        // Mirror the synchronous failure paths in
        // `commit_usaa1`'s spawned local-kick task and the
        // `run_remote_pipeline` task: route to
        // `cleanup_failed_request`, which gathers the unfilled G1
        // ids and calls `worker_hook.mark_failed_onboarding` so
        // vLLM unblocks the slot.
        //
        // Idempotent: cleanup is safe to call after the slot has
        // already drained (DashMap removes are no-ops, and
        // `mark_failed_onboarding` is only invoked when the
        // wrapper actually has unfilled G1 ids).
        let arc_self = match self.arc_self() {
            Some(a) => a,
            None => return async {}.boxed(),
        };
        async move {
            arc_self.cleanup_failed_request(&request_id, reason).await;
        }
        .boxed()
    }
}

impl DecodeDisaggLeader {
    fn prefill_unimplemented(&self, op: &str, request_id: Option<&str>) -> ! {
        tracing::error!(
            op,
            ?request_id,
            "prefill-side conditional-disagg API hit an unimplemented path"
        );
        todo!("prefill-side conditional-disagg API: {op}")
    }
}

impl ConnectorLeaderApi for DecodeDisaggLeader {
    fn create_slot(&self, request: Request) -> Result<()> {
        let request_id = request.request_id.clone();
        let num_tokens = request.tokens.len();
        let has_kv_transfer = request
            .metadata
            .as_ref()
            .and_then(|m| m.kv_transfer_params.as_ref())
            .is_some();
        crate::audit!(
            "create_slot",
            role = "decode",
            request_id = %request_id,
            num_tokens,
            has_kv_transfer
        );
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
        crate::audit!(
            "gnmt_entry",
            role = "decode",
            request_id,
            num_computed_tokens
        );
        let result = match self.current_role() {
            DisaggregationRole::Decode => self.decode_gnmt(request_id, num_computed_tokens),
            DisaggregationRole::Prefill => {
                self.prefill_unimplemented("get_num_new_matched_tokens", Some(request_id))
            }
        };
        match &result {
            Ok((count, async_load)) => crate::audit!(
                "gnmt_exit",
                role = "decode",
                request_id,
                count = ?count,
                async_load
            ),
            Err(err) => crate::audit!(
                "gnmt_error",
                role = "decode",
                request_id,
                error = %err
            ),
        }
        result
    }

    fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        let cd_tracked = self.cd_request_state.contains_key(request_id);
        crate::audit!(
            "usaa_entry",
            role = "decode",
            request_id,
            num_block_ids = block_ids.len(),
            num_external_tokens,
            cd_tracked
        );
        let result = match self.current_role() {
            DisaggregationRole::Decode => {
                self.decode_usaa(request_id, block_ids, num_external_tokens)
            }
            DisaggregationRole::Prefill => {
                self.prefill_unimplemented("update_state_after_alloc", Some(request_id))
            }
        };
        match &result {
            Ok(()) => crate::audit!("usaa_exit", role = "decode", request_id, ok = true),
            Err(err) => crate::audit!(
                "usaa_error",
                role = "decode",
                request_id,
                error = %err
            ),
        }
        result
    }

    fn build_connector_meta(&self, output: SchedulerOutput) -> Result<KvConnectorMetadata> {
        crate::connector::leader::audit::audit_build_meta("decode", &output);
        self.inner.build_connector_meta(output)
    }

    fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()> {
        for rid in &finished_sending {
            crate::audit!(
                "uco_finished_sending",
                role = "decode",
                request_id = %rid,
                cd_tracked = self.cd_request_state.contains_key(rid)
            );
        }
        for rid in &finished_recving {
            crate::audit!(
                "uco_finished_recving",
                role = "decode",
                request_id = %rid,
                cd_tracked = self.cd_request_state.contains_key(rid)
            );
        }
        self.inner
            .update_connector_output(finished_sending, finished_recving)
    }

    fn request_finished(&self, request_id: &str) -> FinishedStatus {
        let cd_tracked = self.cd_request_state.contains_key(request_id);
        crate::audit!(
            "request_finished_entry",
            role = "decode",
            request_id,
            cd_tracked
        );
        if cd_tracked {
            self.release_request(request_id);
            self.coordinator.release(request_id);
        }
        let status = self.inner.request_finished(request_id);
        crate::audit!(
            "request_finished_exit",
            role = "decode",
            request_id,
            status = ?status
        );
        status
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
}
