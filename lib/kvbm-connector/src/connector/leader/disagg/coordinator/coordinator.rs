// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `ConditionalDisaggCoordinator` — unified per-request coordinator for the R-B
//! collapse (Slice 3: prefill flow).
//!
//! Holds a `DashMap<String, Arc<CdRequest>>` keyed by `request_id`.  All
//! requests are prefill-role in Slice 3; decode role will be added in Slice 4.
//!
//! ### Observer strategy (Slice 3)
//!
//! A fresh [`ConditionalDecodeG2Observer`] is created at construction time.
//! Its internal `coordinator` field is left unwired (`Weak::new()`) because
//! the observer's dispatch path calls `PrefillCoordinatorImpl::commit_output_blocks`,
//! which is concretely typed.  Slice 3 tests call `commit_output_blocks` on
//! the coordinator directly — the observer dispatch path is not exercised.
//! Slice 5 will wire the observer to the new coordinator once the old
//! coordinator type is retired.
//!
//! ### Decode role (future — Slice 4)
//!
//! Methods that branch on decode state currently `unreachable!()` — they are
//! unreachable in Slice 3 because only prefill-classified requests are
//! registered.  Do NOT call these paths with a decode-role `CdRequest`.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::{Result, anyhow};
use dashmap::DashMap;
use futures::StreamExt;
use kvbm_disagg_protocol::RemotePrefillParams;
use kvbm_engine::disagg::session::{
    AvailabilityDelta, CommitDelta, CommittedBlock, SessionFactory,
};
use kvbm_logical::blocks::{CompleteBlock, ImmutableBlock};
use parking_lot::Mutex;
use tokio::runtime::Handle;

use crate::connector::leader::scheduler::KvConnectorMetadata;
use crate::{BlockId, G2, InstanceId, SequenceHash};

use super::super::lifecycle::{LIFECYCLE_WATCHDOG, spawn_lifecycle_watcher};
use super::super::peer_resolver::PeerResolver;
use super::super::prefill_coordinator::{
    ConditionalDecodeG2Observer, ObserverHandle, PrefillCoordinator, PrefillStatus,
};
use super::super::transport::{CdBlockTransport, CdWorkerHook, InnerLeaderShim};
use super::{CdRequest, PrefillBits};

// ============================================================================
// ConditionalDisaggCoordinator
// ============================================================================

/// Unified per-request coordinator for conditional disaggregation (R-B).
///
/// In Slice 3 only the **prefill role** is implemented.  Decode-role
/// state will be added in Slice 4.  The struct layout and public API
/// mirror `PrefillCoordinatorImpl` exactly so the test harness can swap
/// `Arc<PrefillCoordinatorImpl>` → `Arc<ConditionalDisaggCoordinator>`.
pub struct ConditionalDisaggCoordinator {
    inner: Arc<dyn InnerLeaderShim>,
    transport: Arc<dyn CdBlockTransport>,
    worker_hook: Arc<dyn CdWorkerHook>,
    session_factory: Arc<dyn SessionFactory>,
    peer_resolver: Arc<dyn PeerResolver>,
    known_peers: dashmap::DashSet<InstanceId>,
    runtime: Handle,
    states: DashMap<String, Arc<CdRequest>>,
    /// Single observer instance registered ONCE with the offload pipeline.
    /// Internal coordinator dispatch is unwired in Slice 3 (see module doc).
    observer: Arc<ConditionalDecodeG2Observer>,
    lifecycle_watchdog: Duration,
    weak_self: Weak<Self>,
}

impl ConditionalDisaggCoordinator {
    /// Construct with the default production lifecycle watchdog (60s).
    pub fn new(
        inner: Arc<dyn InnerLeaderShim>,
        transport: Arc<dyn CdBlockTransport>,
        worker_hook: Arc<dyn CdWorkerHook>,
        session_factory: Arc<dyn SessionFactory>,
        peer_resolver: Arc<dyn PeerResolver>,
        runtime: Handle,
    ) -> Arc<Self> {
        Self::new_with_watchdog(
            inner,
            transport,
            worker_hook,
            session_factory,
            peer_resolver,
            runtime,
            LIFECYCLE_WATCHDOG,
        )
    }

    /// Test-friendly constructor with an injectable lifecycle watchdog.
    /// Production callers use [`new`](Self::new) which fixes the watchdog
    /// at [`LIFECYCLE_WATCHDOG`] (60s).
    pub fn new_with_watchdog(
        inner: Arc<dyn InnerLeaderShim>,
        transport: Arc<dyn CdBlockTransport>,
        worker_hook: Arc<dyn CdWorkerHook>,
        session_factory: Arc<dyn SessionFactory>,
        peer_resolver: Arc<dyn PeerResolver>,
        runtime: Handle,
        lifecycle_watchdog: Duration,
    ) -> Arc<Self> {
        // Observer wired at construction.  Internal coordinator dispatch stays
        // unwired (Weak::new) for Slice 3 — tests call commit_output_blocks
        // directly.  Slice 5 will wire when the old coordinator retires.
        let observer = ConditionalDecodeG2Observer::new();
        Arc::new_cyclic(|weak_self| Self {
            inner,
            transport,
            worker_hook,
            session_factory,
            peer_resolver,
            known_peers: dashmap::DashSet::new(),
            runtime,
            states: DashMap::new(),
            observer,
            lifecycle_watchdog,
            weak_self: weak_self.clone(),
        })
    }

    /// Observer for registration with the offload pipeline.
    pub fn observer(&self) -> &Arc<ConditionalDecodeG2Observer> {
        &self.observer
    }

    /// Returns a closure suitable for `Pipeline::add_register_observer`.
    pub fn observer_callback(
        &self,
    ) -> Arc<dyn Fn(&[ImmutableBlock<G2>]) + Send + Sync + 'static> {
        let observer = Arc::clone(&self.observer);
        Arc::new(move |blocks: &[ImmutableBlock<G2>]| observer.observe(blocks))
    }

    pub fn active_count(&self) -> usize {
        self.states.len()
    }

    /// Returns the granular per-side prefill status for `request_id`.
    /// Returns `None` if the request is not tracked or is not prefill-role.
    pub fn status_for(&self, request_id: &str) -> Option<PrefillStatus> {
        self.states.get(request_id).and_then(|entry| {
            entry
                .value()
                .as_prefill()
                .map(|bits| *bits.status.lock())
        })
    }

    pub fn has_active_request(&self, request_id: &str) -> bool {
        self.states.contains_key(request_id)
    }

    fn state_for(&self, request_id: &str) -> Option<Arc<CdRequest>> {
        self.states.get(request_id).map(|e| Arc::clone(e.value()))
    }

    fn arc_self(&self) -> Option<Arc<Self>> {
        self.weak_self.upgrade()
    }

    /// Commit forward-pass output blocks to the peer via the session's
    /// commit + availability streams.  Semantics mirror
    /// `PrefillCoordinatorImpl::commit_output_blocks` exactly.
    pub fn commit_output_blocks(
        &self,
        request_id: &str,
        blocks: Vec<ImmutableBlock<G2>>,
    ) -> Result<()> {
        let state = self
            .state_for(request_id)
            .ok_or_else(|| anyhow!("commit_output_blocks: unknown request {}", request_id))?;

        // Top-level session is the shared field for prefill (lazily attached).
        let session = state.session.lock().clone().ok_or_else(|| {
            anyhow!(
                "commit_output_blocks: session not attached for {}",
                request_id
            )
        })?;

        let hashes: Vec<SequenceHash> = blocks.iter().map(|b| b.sequence_hash()).collect();
        session.commit(hashes)?;
        session.make_available(blocks)?;
        Ok(())
    }

    /// Surface a failed CD-bound prefill request to vLLM and tear down local
    /// state.  Idempotent: returns early if state already evicted.
    ///
    /// Order:
    /// 1. `mark_failed_onboarding(rid, g1_or_empty)`
    /// 2. `session.close(reason)` if attached
    /// 3. `states.remove(request_id)`
    pub async fn cleanup_failed_request(
        self: &Arc<Self>,
        request_id: &str,
        reason: String,
    ) {
        let state = match self.state_for(request_id) {
            Some(s) => s,
            None => {
                tracing::debug!(
                    request_id,
                    "cleanup_failed_request: state already evicted; no-op"
                );
                return;
            }
        };

        let g1_ids = state.failed_g1_block_ids();
        let num_g1 = g1_ids.len();
        let pre_usaa = num_g1 == 0;

        crate::audit!(
            "prefill_cleanup_failed_request",
            role = "prefill",
            request_id = %request_id,
            num_g1 = num_g1,
            pre_usaa = pre_usaa,
            reason = %reason
        );
        tracing::warn!(
            request_id,
            num_g1,
            pre_usaa,
            reason = %reason,
            "prefill cleanup_failed_request: surfacing to vLLM"
        );

        if let Err(err) = self
            .worker_hook
            .mark_failed_onboarding(request_id.to_string(), g1_ids)
            .await
        {
            tracing::error!(
                request_id,
                error = %err,
                "cleanup_failed_request: mark_failed_onboarding RPC failed; \
                 vLLM will time out the request"
            );
        }

        // Use the top-level shared session field.
        if let Some(session) = state.session.lock().clone() {
            session.close(Some(reason));
        }

        self.states.remove(request_id);
    }

    // =========================================================================
    // Internal async pipeline methods (prefill role only in Slice 3)
    // =========================================================================

    #[tracing::instrument(
        level = "info",
        skip(self, params, state),
        fields(initiator = %params.initiator_instance_id, session_id = %params.session_id)
    )]
    async fn run_setup(
        self: Arc<Self>,
        request_id: String,
        params: RemotePrefillParams,
        state: Arc<CdRequest>,
    ) -> Result<()> {
        let bits = state
            .as_prefill()
            .expect("run_setup: CdRequest must be prefill-role");

        tracing::info!(
            num_expected_hashes = bits.expected_hashes.len(),
            num_external_tokens = bits.num_external_tokens,
            "prefill run_setup: start"
        );

        let endpoint = params.decode_endpoint.clone().ok_or_else(|| {
            anyhow!(
                "RemotePrefillParams.decode_endpoint missing for {}",
                request_id
            )
        })?;

        // 1a. Resolve + register decode's velo peer info.
        if !self.known_peers.contains(&params.initiator_instance_id) {
            tracing::info!("prefill run_setup: resolving decode peer via hub");
            self.peer_resolver
                .resolve_and_register(params.initiator_instance_id)
                .await
                .map_err(|e| {
                    anyhow!(
                        "resolve+register decode peer {} for request {}: {}",
                        params.initiator_instance_id,
                        request_id,
                        e
                    )
                })?;
            self.known_peers.insert(params.initiator_instance_id);
            tracing::info!("prefill run_setup: peer registered");
        }

        // 1b. Attach.
        tracing::info!("prefill run_setup: factory.attach");
        let session = self
            .session_factory
            .attach(params.session_id, params.initiator_instance_id, endpoint)
            .await?;
        // Store in the top-level shared session field.
        *state.session.lock() = Some(Arc::clone(&session));

        // Spawn lifecycle watcher.
        let watcher_coord = self.weak_self.clone();
        let watcher_request_id = request_id.clone();
        spawn_lifecycle_watcher(
            &self.runtime,
            Arc::clone(&session),
            "prefill",
            request_id.clone(),
            params.session_id.to_string(),
            self.lifecycle_watchdog,
            move |_outcome| async move {
                if let Some(coord) = watcher_coord.upgrade() {
                    coord.states.remove(&watcher_request_id);
                }
            },
        );

        tracing::info!("prefill run_setup: attached, draining commits");

        // 2. Drain commit stream.
        let mut commits = session.commits();
        let mut commit_seen = HashSet::new();
        let expected_count = bits.expected_hashes.len();
        while let Some(d) = commits.next().await {
            match d {
                CommitDelta::Added(hashes) => {
                    let n = hashes.len();
                    for h in hashes {
                        commit_seen.insert(h);
                    }
                    tracing::info!(
                        added = n,
                        seen = commit_seen.len(),
                        expected = expected_count,
                        "prefill run_setup: commits Added"
                    );
                    if commit_seen.len() >= expected_count {
                        break;
                    }
                }
                CommitDelta::Closed => {
                    tracing::info!(
                        seen = commit_seen.len(),
                        expected = expected_count,
                        "prefill run_setup: commits Closed"
                    );
                    break;
                }
            }
        }
        tracing::info!("prefill run_setup: commits drained, draining availability");

        // 3. Drain availability and pull chunks.
        *bits.status.lock() = PrefillStatus::Pulling;
        let mut avail = session.availability();
        let expected_set: HashSet<SequenceHash> = bits.expected_hashes.iter().copied().collect();
        let mut remaining: HashSet<SequenceHash> = expected_set.clone();
        let mut filled_index: usize = 0;

        while let Some(d) = avail.next().await {
            match d {
                AvailabilityDelta::Available(blocks) => {
                    let chunk: Vec<CommittedBlock> = blocks
                        .into_iter()
                        .filter(|b| remaining.contains(&b.hash))
                        .collect();
                    tracing::info!(
                        chunk_size = chunk.len(),
                        remaining_before = remaining.len(),
                        "prefill run_setup: availability Available chunk"
                    );
                    if chunk.is_empty() {
                        continue;
                    }
                    self.pull_and_register_chunk(
                        &request_id,
                        &state,
                        chunk,
                        &mut filled_index,
                    )
                    .await?;
                    let registered = bits.registered_g2.lock();
                    remaining = expected_set.clone();
                    for b in registered.iter() {
                        remaining.remove(&b.sequence_hash());
                    }
                    tracing::info!(
                        registered = registered.len(),
                        remaining_after = remaining.len(),
                        "prefill run_setup: chunk pulled+registered"
                    );
                    if remaining.is_empty() {
                        break;
                    }
                }
                AvailabilityDelta::Drained => {
                    tracing::info!(
                        remaining = remaining.len(),
                        "prefill run_setup: availability Drained"
                    );
                    break;
                }
            }
        }

        if !remaining.is_empty() {
            tracing::error!(
                remaining = remaining.len(),
                "prefill run_setup: availability drained with hashes still missing"
            );
            anyhow::bail!(
                "availability drained with {} hashes still missing for {}",
                remaining.len(),
                request_id
            );
        }

        *bits.status.lock() = PrefillStatus::Registered;
        bits.pulls_complete.store(true, Ordering::Release);
        tracing::info!("prefill run_setup: all pulls complete, marking registered");

        // 4. If USAA already arrived, kick onboard now.
        let g1 = bits.pending_g1.lock().take();
        if let Some(g1_ids) = g1 {
            tracing::info!(
                num_g1 = g1_ids.len(),
                "prefill run_setup: USAA already arrived, kicking onboard"
            );
            self.kick_onboard(&request_id, &state, g1_ids).await?;
        } else {
            tracing::info!("prefill run_setup: awaiting USAA for onboard kick");
        }

        Ok(())
    }

    /// Pull a chunk of available blocks into P's G2, register them, and
    /// append to `bits.registered_g2`.  Maintains positional order matching
    /// `bits.expected_hashes`.
    async fn pull_and_register_chunk(
        self: &Arc<Self>,
        request_id: &str,
        state: &Arc<CdRequest>,
        chunk: Vec<CommittedBlock>,
        filled_index: &mut usize,
    ) -> Result<()> {
        let bits = state
            .as_prefill()
            .expect("pull_and_register_chunk: prefill role required");

        let positions: HashMap<SequenceHash, usize> = bits
            .expected_hashes
            .iter()
            .enumerate()
            .map(|(i, h)| (*h, i))
            .collect();
        let mut indexed: Vec<(usize, CommittedBlock)> = chunk
            .into_iter()
            .map(|b| {
                let idx = *positions.get(&b.hash).expect(
                    "chunk hash not in expected_hashes — \
                     filtered before calling pull_and_register_chunk",
                );
                (idx, b)
            })
            .collect();
        indexed.sort_by_key(|(idx, _)| *idx);

        let hashes: Vec<SequenceHash> = indexed.iter().map(|(_, b)| b.hash).collect();
        let chunk_size = hashes.len();
        let dst = self.inner.allocate_g2_blocks(chunk_size)?;

        let session = state
            .session
            .lock()
            .clone()
            .ok_or_else(|| anyhow!("pull: session missing for {}", request_id))?;
        let filled = session.pull(hashes.clone(), dst).await?;

        let token_range_start = indexed.first().map(|(i, _)| *i).unwrap_or(0);
        let token_range_end = indexed.last().map(|(i, _)| *i + 1).unwrap_or(0);
        for (i, (idx, _)) in indexed.iter().enumerate() {
            if *idx != token_range_start + i {
                anyhow::bail!(
                    "non-contiguous chunk: expected idx {}, got {} (chunk: {:?})",
                    token_range_start + i,
                    idx,
                    indexed.iter().map(|(i, _)| *i).collect::<Vec<_>>()
                );
            }
        }
        let abs_start = bits.computed_blocks_offset + token_range_start;
        let abs_end = bits.computed_blocks_offset + token_range_end;
        let token_blocks = self
            .inner
            .token_blocks_for_range(request_id, abs_start..abs_end)?;
        let mut completes: Vec<CompleteBlock<G2>> = Vec::with_capacity(chunk_size);
        for (mutable, token_block) in filled.into_iter().zip(token_blocks.iter()) {
            completes.push(
                mutable
                    .complete(token_block)
                    .map_err(|err| anyhow!("MutableBlock::complete failed: {:?}", err))?,
            );
        }
        let registered = self.inner.register_g2_blocks(completes)?;
        bits.registered_g2.lock().extend(registered);
        *filled_index += chunk_size;
        Ok(())
    }

    /// Drive G2→G1 onboard for the local-match prefix.
    async fn kick_onboard(
        self: &Arc<Self>,
        request_id: &str,
        state: &Arc<CdRequest>,
        g1_dst_local_match_prefix: Vec<BlockId>,
    ) -> Result<()> {
        let bits = state
            .as_prefill()
            .expect("kick_onboard: prefill role required");

        if bits.onboarding_scheduled.swap(true, Ordering::AcqRel) {
            return Ok(()); // already scheduled
        }
        *bits.status.lock() = PrefillStatus::OnboardingScheduled;

        if bits.num_external_tokens == 0 {
            tracing::info!(
                "kick_onboard: num_external_tokens=0 — \
                 no G2→G1 onboard needed (cold-cache CD); \
                 marking complete (no worker_hook signal)"
            );
            *bits.status.lock() = PrefillStatus::OnboardingComplete;
            return Ok(());
        }

        let g2_src_block_ids: Vec<BlockId> = bits
            .registered_g2
            .lock()
            .iter()
            .map(|b| b.block_id())
            .collect();

        if g2_src_block_ids.len() != g1_dst_local_match_prefix.len() {
            anyhow::bail!(
                "kick_onboard: G2 src count {} != local-match G1 dst prefix count {} \
                 (num_external_tokens={}, expected_hashes.len()={})",
                g2_src_block_ids.len(),
                g1_dst_local_match_prefix.len(),
                bits.num_external_tokens,
                bits.expected_hashes.len(),
            );
        }

        self.transport
            .local_g2_to_g1(g2_src_block_ids, g1_dst_local_match_prefix)
            .await?;

        *bits.status.lock() = PrefillStatus::OnboardingComplete;

        self.worker_hook
            .mark_onboarding_complete(request_id.to_string())
            .await?;

        Ok(())
    }
}

// ============================================================================
// PrefillCoordinator trait impl
// ============================================================================

impl PrefillCoordinator for ConditionalDisaggCoordinator {
    fn has_active_request(&self, request_id: &str) -> bool {
        self.states.contains_key(request_id)
    }

    fn ensure_started(&self, request_id: &str, params: &RemotePrefillParams) -> Result<usize> {
        if let Some(state) = self.state_for(request_id) {
            let bits = state
                .as_prefill()
                .expect("ensure_started: existing request must be prefill-role");
            return Ok(bits.num_external_tokens);
        }

        let block_size = self.inner.block_size();
        let num_external_tokens = params.sequence_hashes.len() * block_size;
        let computed_blocks_offset = params.num_computed_tokens / block_size;

        // Compute expected output hashes and install the observer entry.
        let split = self.inner.slot_match_split(request_id)?;
        let pulled: HashSet<SequenceHash> = params.sequence_hashes.iter().copied().collect();
        let expected_outputs: HashSet<SequenceHash> = split
            .all_sequence_hashes
            .iter()
            .filter(|h| !pulled.contains(h))
            .copied()
            .collect();
        let observer_handle: ObserverHandle =
            self.observer.track(request_id.to_string(), expected_outputs);

        let bits = PrefillBits {
            num_external_tokens,
            expected_hashes: params.sequence_hashes.clone(),
            computed_blocks_offset,
            pending_g1: Mutex::new(None),
            g1_block_ids: Mutex::new(None),
            registered_g2: Mutex::new(Vec::new()),
            request_finished_seen: std::sync::atomic::AtomicBool::new(false),
            pulls_complete: std::sync::atomic::AtomicBool::new(false),
            onboarding_scheduled: std::sync::atomic::AtomicBool::new(false),
            status: Mutex::new(PrefillStatus::Attaching),
            observer_handle,
        };

        // Use the top-level CdRequest constructor.  `session_id` is
        // decoded from params; `peer_instance_id` is `initiator_instance_id`.
        let state = CdRequest::new_prefill(
            request_id.to_string(),
            params.session_id,
            params.initiator_instance_id,
            bits,
        );
        self.states
            .insert(request_id.to_string(), Arc::clone(&state));

        let coord = self
            .arc_self()
            .ok_or_else(|| anyhow!("coordinator weak_self upgrade failed"))?;
        let request_id_owned = request_id.to_string();
        let params_owned = params.clone();

        crate::audit!(
            "session_setup_spawned",
            role = "prefill",
            request_id = %request_id_owned,
            session_id = %params.session_id,
            initiator = %params.initiator_instance_id,
            num_sequence_hashes = params.sequence_hashes.len()
        );

        self.runtime.spawn(async move {
            let setup_result = coord
                .clone()
                .run_setup(request_id_owned.clone(), params_owned, state)
                .await;
            if let Err(err) = setup_result {
                tracing::error!(
                    request_id = request_id_owned,
                    error = %err,
                    "prefill setup failed; surfacing to vLLM via cleanup_failed_request"
                );
                coord
                    .cleanup_failed_request(
                        &request_id_owned,
                        format!("setup failed: {err}"),
                    )
                    .await;
            }
        });

        Ok(num_external_tokens)
    }

    fn on_usaa(
        &self,
        request_id: &str,
        block_ids: &[BlockId],
        num_external_tokens: usize,
    ) -> Result<()> {
        let Some(state) = self.state_for(request_id) else {
            return Ok(());
        };
        let bits = state
            .as_prefill()
            .expect("on_usaa: CdRequest must be prefill-role");

        // Post-onboard follow-up USAAs are no-ops (mirrors PrefillCoordinatorImpl).
        if num_external_tokens == 0 && bits.num_external_tokens > 0 {
            return Ok(());
        }

        if num_external_tokens != bits.num_external_tokens {
            anyhow::bail!(
                "on_usaa: num_external_tokens mismatch (got {}, expected {})",
                num_external_tokens,
                bits.num_external_tokens
            );
        }

        // Stash the FULL G1 window for failure paths.
        {
            let mut stash = bits.g1_block_ids.lock();
            if stash.is_none() {
                *stash = Some(block_ids.to_vec());
            }
        }

        let block_size = self.inner.block_size();
        let local_match_blocks = bits.num_external_tokens / block_size;
        if block_ids.len() < local_match_blocks {
            anyhow::bail!(
                "on_usaa: block_ids len {} < local_match_blocks {} (num_external_tokens={})",
                block_ids.len(),
                local_match_blocks,
                bits.num_external_tokens
            );
        }
        let g1_ids: Vec<BlockId> = block_ids[..local_match_blocks].to_vec();
        let pulls_complete = bits.pulls_complete.load(Ordering::Acquire);

        if pulls_complete {
            let coord = self
                .arc_self()
                .ok_or_else(|| anyhow!("coordinator weak_self upgrade failed"))?;
            let request_id_owned = request_id.to_string();
            let state_clone = Arc::clone(&state);
            self.runtime.spawn(async move {
                if let Err(err) = coord
                    .clone()
                    .kick_onboard(&request_id_owned, &state_clone, g1_ids)
                    .await
                {
                    tracing::error!(
                        request_id = request_id_owned,
                        error = %err,
                        "prefill onboard failed; surfacing to vLLM via cleanup_failed_request"
                    );
                    coord
                        .cleanup_failed_request(
                            &request_id_owned,
                            format!("kick_onboard failed: {err}"),
                        )
                        .await;
                }
            });
        } else {
            *bits.pending_g1.lock() = Some(g1_ids);
        }

        Ok(())
    }

    fn observe_forward(&self, _request_id: &str, _meta: &KvConnectorMetadata) -> Result<()> {
        // No-op: tracking established at ensure_started time (matches
        // PrefillCoordinatorImpl::observe_forward semantics).
        Ok(())
    }

    fn on_request_finished(&self, request_id: &str) {
        let Some(state) = self.state_for(request_id) else {
            return;
        };
        let bits = state
            .as_prefill()
            .expect("on_request_finished: CdRequest must be prefill-role");

        bits.request_finished_seen.store(true, Ordering::Release);

        // Defer finalize until observer residual drains (invariant #16).
        let session_opt = state.session.lock().clone();
        let observer = Arc::clone(&self.observer);
        let request_id_owned = request_id.to_string();
        let runtime = self.runtime.clone();
        runtime.spawn(async move {
            const FINALIZE_WAIT: Duration = Duration::from_secs(10);
            const POLL_INTERVAL: Duration = Duration::from_millis(2);

            let deadline = tokio::time::Instant::now() + FINALIZE_WAIT;
            let mut waited_for_observer = false;
            while observer.has_pending(&request_id_owned) {
                waited_for_observer = true;
                if tokio::time::Instant::now() >= deadline {
                    tracing::warn!(
                        request_id = %request_id_owned,
                        wait_secs = FINALIZE_WAIT.as_secs(),
                        "prefill on_request_finished: observer residual not drained \
                         within watchdog; forcing session.finalize"
                    );
                    crate::audit!(
                        "prefill_finalize_observer_watchdog",
                        role = "prefill",
                        request_id = %request_id_owned,
                        wait_secs = FINALIZE_WAIT.as_secs()
                    );
                    break;
                }
                tokio::time::sleep(POLL_INTERVAL).await;
            }
            if waited_for_observer {
                crate::audit!(
                    "prefill_finalize_observer_drained",
                    role = "prefill",
                    request_id = %request_id_owned
                );
            }
            if let Some(session) = session_opt {
                session.finalize(Some("request_finished".to_string()));
            }
        });

        // Mark the coarse status Released (mirrors PrefillCoordinatorImpl).
        // Per-side PrefillStatus is also set to Released for consistency.
        *bits.status.lock() = PrefillStatus::Released;
        // Note: do NOT remove RequestState here — the lifecycle watcher
        // removes it on Detach or watchdog.
    }
}

#[cfg(test)]
mod tests {
    /// `ConditionalDisaggCoordinator` must be constructible via both ctors.
    /// Real construction requires a tokio runtime; covered by cd_prefill_e2e.
    #[test]
    fn size_of_is_nonzero() {
        use super::ConditionalDisaggCoordinator;
        assert!(std::mem::size_of::<ConditionalDisaggCoordinator>() > 0);
    }
}
