// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill-side per-request coordinator trait + production impl.
//!
//! The coordinator owns the async state machine for a single
//! CD-bound request on the prefill participant: attach to D's
//! session, drain D's commit + availability streams chunk-by-
//! chunk, pull each chunk into P's G2 as availability lands,
//! drive the G2→G1 onboard once USAA arrives with the G1
//! destinations, and tear down once the request finishes.
//!
//! See `/home/ryan/.claude/plans/cd-session-refactor.md` for
//! the symmetric `Session` API this is built against.
//!
//! The wrapper ([`super::prefill_leader::PrefillDisaggLeader`])
//! interacts with the coordinator at the `ConnectorLeaderApi`
//! boundary; the coordinator does not see vLLM directly.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Weak;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{Context, Result, anyhow};
use dashmap::DashMap;
use futures::StreamExt;
use kvbm_disagg_protocol::RemotePrefillParams;
use kvbm_engine::disagg::session::{
    AvailabilityDelta, CommitDelta, CommittedBlock, Session, SessionFactory,
};
use kvbm_logical::blocks::{CompleteBlock, ImmutableBlock};
use parking_lot::Mutex;
use tokio::runtime::Handle;

use crate::connector::leader::scheduler::KvConnectorMetadata;
use crate::{BlockId, G2, SequenceHash};

use super::peer_resolver::PeerResolver;
use super::transport::{CdBlockTransport, CdWorkerHook, InnerLeaderShim};
use crate::InstanceId;

/// Per-request prefill-side coordinator.
///
/// All methods are sync; async work is spawned internally.
/// Methods are addressed by `request_id` (the slot id) so the
/// coordinator can be a singleton owning a `DashMap` of
/// per-request state.
pub trait PrefillCoordinator: Send + Sync {
    /// Idempotent per-request init.
    ///
    /// First call for a `request_id` installs state and spawns
    /// attach + commit/availability drain + chunked-pull
    /// pipeline asynchronously. Subsequent calls return the
    /// cached external-token count without side effects.
    fn ensure_started(&self, request_id: &str, params: &RemotePrefillParams) -> Result<usize>;

    /// USAA hand-off.
    ///
    /// Wrapper calls inner USAA first, then hands the freshly
    /// allocated G1 ids and the external-token count to the
    /// coordinator so it can wire G2→G1 onboard completion to
    /// `mark_workers_onboarding_complete`.
    fn on_usaa(
        &self,
        request_id: &str,
        block_ids: &[BlockId],
        num_external_tokens: usize,
    ) -> Result<()>;

    /// Forward-pass-time hook.
    ///
    /// Wrapper's `build_connector_meta` calls this after the
    /// inner build returns. Production wiring (Phase B) attaches
    /// the offload pipeline's register-observer here so output
    /// blocks flow into [`commit_output_blocks`] automatically.
    /// For Phase A this is a no-op; tests call
    /// [`commit_output_blocks`] directly.
    fn observe_forward(&self, request_id: &str, meta: &KvConnectorMetadata) -> Result<()>;

    /// Detach the slot side.
    ///
    /// The wrapper calls this after `inner.request_finished`.
    /// The coordinator finishes its commits/availability and
    /// closes the session. Pins on output blocks are managed
    /// internally by the session (drop on PullAck).
    fn on_request_finished(&self, request_id: &str);
}

// ============================================================================
// PrefillCoordinatorImpl — golden-path implementation
// ============================================================================

/// Per-request state machine state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillStatus {
    Attaching,
    Pulling,
    Registered,
    OnboardingScheduled,
    OnboardingComplete,
    SlotDone,
    Released,
}

struct RequestState {
    num_external_tokens: usize,
    expected_hashes: Vec<SequenceHash>,
    /// Decode-side `computed_blocks` — offset to add when
    /// translating an `expected_hashes` index back to the
    /// absolute token-block index in the original sequence.
    /// `expected_hashes[i]` corresponds to absolute position
    /// `computed_blocks_offset + i`.
    computed_blocks_offset: usize,
    /// Attached session — populated once `factory.attach` resolves.
    session: Mutex<Option<Arc<dyn Session>>>,
    /// G1 destinations from USAA, stashed until pull/register
    /// pipeline completes so we can kick the G2→G1 onboard.
    pending_g1: Mutex<Option<Vec<BlockId>>>,
    /// All registered G2 blocks across chunks.
    registered_g2: Mutex<Vec<ImmutableBlock<G2>>>,
    /// Wrapper has called `on_request_finished`.
    request_finished_seen: AtomicBool,
    /// All chunks pulled+registered (= we saw `Drained` or all
    /// expected hashes were filled).
    pulls_complete: AtomicBool,
    /// Onboard kick was scheduled (idempotent).
    onboarding_scheduled: AtomicBool,
    status: Mutex<PrefillStatus>,
    /// RAII observer-eviction handle. Dropping `RequestState`
    /// (last Arc) drops this, which evicts the residual entry
    /// from `ConditionalDecodeG2Observer::pending`. No explicit
    /// untrack call is needed in any failure path.
    _observer_handle: ObserverHandle,
}

impl RequestState {
    fn new(
        num_external_tokens: usize,
        expected_hashes: Vec<SequenceHash>,
        computed_blocks_offset: usize,
        observer_handle: ObserverHandle,
    ) -> Self {
        Self {
            num_external_tokens,
            expected_hashes,
            computed_blocks_offset,
            session: Mutex::new(None),
            pending_g1: Mutex::new(None),
            registered_g2: Mutex::new(Vec::new()),
            request_finished_seen: AtomicBool::new(false),
            pulls_complete: AtomicBool::new(false),
            onboarding_scheduled: AtomicBool::new(false),
            status: Mutex::new(PrefillStatus::Attaching),
            _observer_handle: observer_handle,
        }
    }
}

// ============================================================================
// ConditionalDecodeG2Observer
// ============================================================================
//
// One observer instance, registered ONCE with the offload pipeline at
// coordinator construction. Holds per-request residual hash sets; as the
// pipeline registers G2 blocks, the observer pops matched hashes from the
// matching request's set and forwards the matched blocks to
// `commit_output_blocks`. When a request's residual goes empty, its entry
// is auto-dropped.
//
// `break;` on first hash match: if two simultaneous requests share an
// expected output hash (rare; would mean identical continuation blocks),
// the first iterator-ordered request claims. Document and revisit only
// if production hits the case.

pub struct ConditionalDecodeG2Observer {
    /// Per-request residual hashset: hashes we still expect to
    /// see registered for this request. Entries removed as
    /// matches land. Empty entry → dropped.
    pending: Mutex<HashMap<String, HashSet<SequenceHash>>>,
    /// Weak so observer outlives drops cleanly without holding
    /// the coordinator alive.
    coordinator: Mutex<Weak<PrefillCoordinatorImpl>>,
}

impl ConditionalDecodeG2Observer {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            pending: Mutex::new(HashMap::new()),
            coordinator: Mutex::new(Weak::new()),
        })
    }

    fn install_coordinator(&self, coord: Weak<PrefillCoordinatorImpl>) {
        *self.coordinator.lock() = coord;
    }

    /// Track a request's expected output hashes; returns an RAII
    /// [`ObserverHandle`] whose drop evicts the residual entry.
    /// Production calls this at `ensure_started` time and stashes
    /// the handle in `RequestState`. Re-tracking with a different
    /// set replaces the prior entry.
    pub fn track(
        self: &Arc<Self>,
        request_id: String,
        expected: HashSet<SequenceHash>,
    ) -> ObserverHandle {
        self.pending.lock().insert(request_id.clone(), expected);
        ObserverHandle {
            request_id,
            observer: Arc::downgrade(self),
        }
    }

    /// Drop residual entry for a request. Production no longer
    /// calls this directly — the [`ObserverHandle`] returned by
    /// [`track`](Self::track) handles it on drop. Kept public for
    /// test scenarios that exercise manual eviction; idempotent
    /// against auto-drop-on-full-match.
    pub fn untrack_request(&self, request_id: &str) {
        self.pending.lock().remove(request_id);
    }

    /// Test accessor: snapshot of remaining hashes for `request_id`.
    #[cfg(any(test, feature = "testing"))]
    pub fn pending_for(&self, request_id: &str) -> Option<HashSet<SequenceHash>> {
        self.pending.lock().get(request_id).cloned()
    }

    /// Test accessor: count of tracked requests.
    #[cfg(any(test, feature = "testing"))]
    pub fn tracked_count(&self) -> usize {
        self.pending.lock().len()
    }

    /// Called by the offload pipeline's register-observer path
    /// after each batch of G2 blocks is registered.
    ///
    /// Lock split: matches are computed under the `pending`
    /// guard, then the guard is dropped before
    /// `commit_output_blocks` is called (which acquires session
    /// locks). Avoids cross-lock hazards.
    pub fn observe(&self, blocks: &[ImmutableBlock<G2>]) {
        let mut by_request: HashMap<String, Vec<ImmutableBlock<G2>>> = HashMap::new();
        let mut empty_after: Vec<String> = Vec::new();
        {
            let mut pending = self.pending.lock();
            for block in blocks {
                let hash = block.sequence_hash();
                // Linear scan over active CD requests. N expected
                // small; if profiling shows hot, swap for a
                // SequenceHash → request_id reverse index updated
                // alongside `pending` mutations.
                for (request_id, hashes) in pending.iter_mut() {
                    if hashes.remove(&hash) {
                        by_request
                            .entry(request_id.clone())
                            .or_default()
                            .push(block.clone());
                        break;
                    }
                }
            }
            for (request_id, hashes) in pending.iter() {
                if hashes.is_empty() {
                    empty_after.push(request_id.clone());
                }
            }
            for request_id in &empty_after {
                pending.remove(request_id);
            }
        }

        let coord = self.coordinator.lock().upgrade();
        if let Some(coord) = coord {
            for (request_id, blocks) in by_request {
                if let Err(err) = coord.commit_output_blocks(&request_id, blocks) {
                    tracing::error!(
                        request_id,
                        error = %err,
                        "commit_output_blocks failed from observer"
                    );
                }
            }
        }
    }
}

/// RAII handle returned by [`ConditionalDecodeG2Observer::track`].
/// On drop, evicts the per-request residual entry. Held by
/// `RequestState`, so any path that drops the request's state
/// (success, failure, panic mid-pipeline) cleans up automatically.
pub struct ObserverHandle {
    request_id: String,
    observer: Weak<ConditionalDecodeG2Observer>,
}

impl Drop for ObserverHandle {
    fn drop(&mut self) {
        if let Some(observer) = self.observer.upgrade() {
            observer.untrack_request(&self.request_id);
        }
    }
}

/// Production coordinator.
pub struct PrefillCoordinatorImpl {
    inner: Arc<dyn InnerLeaderShim>,
    transport: Arc<dyn CdBlockTransport>,
    worker_hook: Arc<dyn CdWorkerHook>,
    session_factory: Arc<dyn SessionFactory>,
    /// Resolves a remote `InstanceId` to a `PeerInfo` and registers it
    /// on the local messenger. Production: hub-backed; tests: no-op.
    /// Called once per (peer, request) before `factory.attach`; an
    /// internal cache makes repeat resolves cheap.
    peer_resolver: Arc<dyn PeerResolver>,
    /// `InstanceId`s already resolved + registered locally. Skips the
    /// resolver round-trip for subsequent requests targeting the same
    /// peer.
    known_peers: dashmap::DashSet<InstanceId>,
    runtime: Handle,
    states: DashMap<String, Arc<RequestState>>,
    /// Single observer instance, registered ONCE with the
    /// offload pipeline. Holds per-request residual state.
    observer: Arc<ConditionalDecodeG2Observer>,
    weak_self: std::sync::Weak<Self>,
}

impl PrefillCoordinatorImpl {
    pub fn new(
        inner: Arc<dyn InnerLeaderShim>,
        transport: Arc<dyn CdBlockTransport>,
        worker_hook: Arc<dyn CdWorkerHook>,
        session_factory: Arc<dyn SessionFactory>,
        peer_resolver: Arc<dyn PeerResolver>,
        runtime: Handle,
    ) -> Arc<Self> {
        let observer = ConditionalDecodeG2Observer::new();
        let coord = Arc::new_cyclic(|weak_self| Self {
            inner,
            transport,
            worker_hook,
            session_factory,
            peer_resolver,
            known_peers: dashmap::DashSet::new(),
            runtime,
            states: DashMap::new(),
            observer: Arc::clone(&observer),
            weak_self: weak_self.clone(),
        });
        // Wire the weak coordinator back into the observer so
        // observe() can call commit_output_blocks.
        observer.install_coordinator(Arc::downgrade(&coord));
        coord
    }

    /// Accessor for the offload-pipeline observer. Phase B
    /// `init.rs` wiring registers this with the G1→G2 pipeline:
    /// `pipeline.add_register_observer(coord.observer_callback())`.
    pub fn observer(&self) -> &Arc<ConditionalDecodeG2Observer> {
        &self.observer
    }

    /// Returns a closure suitable for
    /// `Pipeline::add_register_observer`. Captures
    /// `Arc<ConditionalDecodeG2Observer>` so the closure stays
    /// valid for the pipeline's lifetime.
    pub fn observer_callback(
        &self,
    ) -> Arc<dyn Fn(&[ImmutableBlock<G2>]) + Send + Sync + 'static> {
        let observer = Arc::clone(&self.observer);
        Arc::new(move |blocks: &[ImmutableBlock<G2>]| observer.observe(blocks))
    }

    pub fn active_count(&self) -> usize {
        self.states.len()
    }

    pub fn status_for(&self, request_id: &str) -> Option<PrefillStatus> {
        self.states
            .get(request_id)
            .map(|entry| *entry.value().status.lock())
    }

    fn state_for(&self, request_id: &str) -> Option<Arc<RequestState>> {
        self.states.get(request_id).map(|e| Arc::clone(e.value()))
    }

    fn arc_self(&self) -> Option<Arc<Self>> {
        self.weak_self.upgrade()
    }

    /// Test/production helper: commit + make_available for a
    /// chunk of forward-pass output blocks. Wires output back
    /// to the peer via the session's commit + availability
    /// streams; pins are managed internally by the session.
    pub fn commit_output_blocks(
        &self,
        request_id: &str,
        blocks: Vec<ImmutableBlock<G2>>,
    ) -> Result<()> {
        let state = self
            .state_for(request_id)
            .ok_or_else(|| anyhow!("commit_output_blocks: unknown request {}", request_id))?;
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

    /// Drive the per-request setup pipeline: attach, drain
    /// commits, drain availability (pull each chunk as it lands),
    /// register pulled G2 blocks, kick onboard if USAA already
    /// arrived.
    #[tracing::instrument(level = "info", skip(self, params, state), fields(initiator = %params.initiator_instance_id, session_id = %params.session_id))]
    async fn run_setup(
        self: Arc<Self>,
        request_id: String,
        params: RemotePrefillParams,
        state: Arc<RequestState>,
    ) -> Result<()> {
        tracing::info!(
            num_expected_hashes = state.expected_hashes.len(),
            num_external_tokens = state.num_external_tokens,
            "prefill run_setup: start"
        );
        let endpoint = params.decode_endpoint.clone().ok_or_else(|| {
            anyhow!(
                "RemotePrefillParams.decode_endpoint missing for {}",
                request_id
            )
        })?;

        // 1a. Resolve + register decode's velo peer info on our
        //     messenger so `factory.attach` (which issues both unary
        //     metadata RPC and a streaming-anchor open) can reach it.
        //     Skip if we've already resolved this peer for an earlier
        //     request — `register_peer` semantics across transports
        //     vary, the local cache makes the call site idempotent
        //     regardless.
        if !self.known_peers.contains(&params.initiator_instance_id) {
            tracing::info!("prefill run_setup: resolving decode peer via hub");
            self.peer_resolver
                .resolve_and_register(params.initiator_instance_id)
                .await
                .with_context(|| {
                    format!(
                        "resolve+register decode peer {} for request {}",
                        params.initiator_instance_id, request_id
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
        *state.session.lock() = Some(Arc::clone(&session));
        tracing::info!("prefill run_setup: attached, draining commits");

        // 2. Drain commit stream until Closed (or until we have
        //    the expected set). Informational; the coordinator
        //    trusts `params.sequence_hashes` for planning.
        let mut commits = session.commits();
        let mut commit_seen = HashSet::new();
        let expected_count = state.expected_hashes.len();
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
                        // Don't break — peer may still send Closed,
                        // but we have what we need to plan against.
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

        // 3. Drain availability and pull chunks as they land.
        *state.status.lock() = PrefillStatus::Pulling;
        let mut avail = session.availability();
        let expected_set: HashSet<SequenceHash> = state.expected_hashes.iter().copied().collect();
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
                    self.pull_and_register_chunk(&request_id, &state, chunk, &mut filled_index)
                        .await?;
                    for h in &state.expected_hashes[filled_index - 1..filled_index] {
                        let _ = h;
                    }
                    // Recompute remaining from registered_g2.
                    let registered = state.registered_g2.lock();
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

        *state.status.lock() = PrefillStatus::Registered;
        state.pulls_complete.store(true, Ordering::Release);
        tracing::info!("prefill run_setup: all pulls complete, marking registered");

        // 4. If USAA already happened, kick onboard now.
        let g1 = state.pending_g1.lock().take();
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

    /// Pull a chunk of available blocks into P's G2,
    /// register them, and append to `state.registered_g2`.
    /// Maintains positional order matching `state.expected_hashes`
    /// via the running `filled_index` cursor.
    async fn pull_and_register_chunk(
        self: &Arc<Self>,
        request_id: &str,
        state: &Arc<RequestState>,
        chunk: Vec<CommittedBlock>,
        filled_index: &mut usize,
    ) -> Result<()> {
        // The chunk's hashes may arrive in any order. Reorder
        // the chunk to match `expected_hashes`'s positional
        // order so the token-block sequence is correct on
        // `MutableBlock::complete`.
        let positions: std::collections::HashMap<SequenceHash, usize> = state
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
                     filtered out before calling pull_and_register_chunk",
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

        // Register pulled blocks.
        let token_range_start = indexed.first().map(|(i, _)| *i).unwrap_or(0);
        let token_range_end = indexed.last().map(|(i, _)| *i + 1).unwrap_or(0);
        // For positional correctness we require the chunk to
        // be a contiguous range of expected positions. If the
        // peer sends out-of-order chunks, error.
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
        // Translate the chunk's `expected_hashes`-relative range back
        // to absolute token-block indices in the original sequence.
        // `expected_hashes[i]` is at absolute position
        // `computed_blocks_offset + i` (carry-forward #1).
        let abs_start = state.computed_blocks_offset + token_range_start;
        let abs_end = state.computed_blocks_offset + token_range_end;
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
        state.registered_g2.lock().extend(registered);
        *filled_index += chunk_size;
        Ok(())
    }

    /// Onboard the blocks decode forwarded into the prefill side's
    /// G1 destinations.
    ///
    /// `g1_dst_local_match_prefix` is the **local-match prefix** of
    /// the G1 destinations vLLM allocated — its length must equal
    /// `state.registered_g2.len()` (= the count decode published via
    /// its session = `params.sequence_hashes.len()`). Callers
    /// (`on_usaa`, `run_setup`'s post-USAA kick) are responsible
    /// for slicing the full G1 list down to that prefix. The
    /// trailing G1 slots — `prefill_window - local_match` — are
    /// filled by the prefill model's forward pass; the worker
    /// observer publishes those blocks back to decode via
    /// `commit_output_blocks` and the session.
    ///
    /// When `state.num_external_tokens == 0` (decode had no cache
    /// to forward — the cold-cache common case), there is nothing
    /// to onboard from G2; this returns after marking the request
    /// complete so the worker side knows the CD-decode flow is
    /// closed and forward-pass-only continues normally.
    async fn kick_onboard(
        self: &Arc<Self>,
        request_id: &str,
        state: &Arc<RequestState>,
        g1_dst_local_match_prefix: Vec<BlockId>,
    ) -> Result<()> {
        if state.onboarding_scheduled.swap(true, Ordering::AcqRel) {
            return Ok(()); // already scheduled
        }
        *state.status.lock() = PrefillStatus::OnboardingScheduled;

        // Cold-cache CD: decode forwarded zero blocks. Nothing to
        // onboard from G2; the prefill model's forward pass is the
        // entire data flow. We do NOT call
        // `worker_hook.mark_onboarding_complete` here: the prefill
        // leader's gnmt short-circuited to inner passthrough
        // (`(Some(0), false)`), so vLLM never registered this
        // request as having an "external receive" pending. The
        // worker holds no kv-recv state for it; signaling
        // `finished_recving` would panic vLLM at
        // `scheduler.py::_update_from_kv_xfer_finished` with
        // `assert req_id in self.requests`.
        if state.num_external_tokens == 0 {
            tracing::info!(
                "kick_onboard: num_external_tokens=0 — \
                 no G2→G1 onboard needed (cold-cache CD); \
                 marking complete (no worker_hook signal)"
            );
            *state.status.lock() = PrefillStatus::OnboardingComplete;
            return Ok(());
        }

        let g2_src_block_ids: Vec<BlockId> = state
            .registered_g2
            .lock()
            .iter()
            .map(|b| b.block_id())
            .collect();

        if g2_src_block_ids.len() != g1_dst_local_match_prefix.len() {
            anyhow::bail!(
                "kick_onboard: G2 src count {} != local-match G1 dst prefix count {} \
                 (state.num_external_tokens={}, state.expected_hashes.len()={})",
                g2_src_block_ids.len(),
                g1_dst_local_match_prefix.len(),
                state.num_external_tokens,
                g2_src_block_ids.len(),
            );
        }

        self.transport
            .local_g2_to_g1(g2_src_block_ids, g1_dst_local_match_prefix)
            .await?;

        *state.status.lock() = PrefillStatus::OnboardingComplete;

        self.worker_hook
            .mark_onboarding_complete(request_id.to_string())
            .await?;

        Ok(())
    }
}

impl PrefillCoordinator for PrefillCoordinatorImpl {
    fn ensure_started(&self, request_id: &str, params: &RemotePrefillParams) -> Result<usize> {
        if let Some(state) = self.state_for(request_id) {
            return Ok(state.num_external_tokens);
        }

        let block_size = self.inner.block_size();
        let num_external_tokens = params.sequence_hashes.len() * block_size;
        let computed_blocks_offset = params.num_computed_tokens / block_size;

        // Compute prefill's expected output hashes and track them
        // in the offload observer. Output hashes = full sequence
        // hashes ⊖ what we pulled from decode (params.sequence_hashes).
        // The observer pops matches as the offload pipeline
        // registers G2 blocks; the returned RAII handle is stashed
        // in `RequestState` so any drop path evicts the residual.
        let split = self.inner.slot_match_split(request_id)?;
        let pulled: HashSet<SequenceHash> = params.sequence_hashes.iter().copied().collect();
        let expected_outputs: HashSet<SequenceHash> = split
            .all_sequence_hashes
            .iter()
            .filter(|h| !pulled.contains(h))
            .copied()
            .collect();
        let observer_handle = self.observer.track(request_id.to_string(), expected_outputs);

        let state = Arc::new(RequestState::new(
            num_external_tokens,
            params.sequence_hashes.clone(),
            computed_blocks_offset,
            observer_handle,
        ));
        self.states
            .insert(request_id.to_string(), Arc::clone(&state));

        let coord = self
            .arc_self()
            .ok_or_else(|| anyhow!("coordinator weak_self upgrade failed"))?;
        let request_id_owned = request_id.to_string();
        let params_owned = params.clone();
        self.runtime.spawn(async move {
            if let Err(err) = coord
                .run_setup(request_id_owned.clone(), params_owned, state)
                .await
            {
                tracing::error!(
                    request_id = request_id_owned,
                    error = %err,
                    "prefill coordinator setup failed (error path is Phase A deferred work)"
                );
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
            // Non-CD or already cleaned up: nothing to do.
            return Ok(());
        };
        // num_external_tokens here is what vLLM passes through from
        // the prefill leader's gnmt return. With the leader's n=0
        // short-circuit (`prefill_leader::ensure_started_zero_passthrough`),
        // cold-cache CD reaches USAA with `num_external_tokens=0`
        // even though `state.num_external_tokens=0` from
        // `params.sequence_hashes.len() * block_size`. Both are 0
        // and agree.
        //
        // For non-zero local-match (decode forwarded N blocks),
        // vLLM allocates ALL G1 slots for the full prefill window
        // (local-match + remote). The leader returns
        // `(Some(local_match_tokens), true)` to vLLM; vLLM passes
        // that back here as `num_external_tokens`, which equals
        // `state.num_external_tokens`. block_ids covers the full
        // window though, so we slice the local-match prefix below
        // before kick_onboard.
        if num_external_tokens != state.num_external_tokens {
            anyhow::bail!(
                "on_usaa: num_external_tokens mismatch (got {}, expected {})",
                num_external_tokens,
                state.num_external_tokens
            );
        }

        // Slice the G1 destinations to the local-match prefix —
        // that's the subset kick_onboard fills via G2→G1 from
        // decode's forwarded blocks. The trailing slots are
        // forward-pass-filled by the prefill model.
        let block_size = self.inner.block_size();
        let local_match_blocks = state.num_external_tokens / block_size;
        if block_ids.len() < local_match_blocks {
            anyhow::bail!(
                "on_usaa: block_ids len {} < local_match_blocks {} (num_external_tokens={})",
                block_ids.len(),
                local_match_blocks,
                state.num_external_tokens
            );
        }
        let g1_ids: Vec<BlockId> = block_ids[..local_match_blocks].to_vec();
        let pulls_complete = state.pulls_complete.load(Ordering::Acquire);

        if pulls_complete {
            // Setup task already finished pull+register; spawn
            // the G2→G1 onboard so this call returns promptly.
            let coord = self
                .arc_self()
                .ok_or_else(|| anyhow!("coordinator weak_self upgrade failed"))?;
            let request_id_owned = request_id.to_string();
            self.runtime.spawn(async move {
                if let Err(err) = coord.kick_onboard(&request_id_owned, &state, g1_ids).await {
                    tracing::error!(
                        request_id = request_id_owned,
                        error = %err,
                        "prefill onboard failed"
                    );
                }
            });
        } else {
            // Stash for the setup task to pick up post-register.
            *state.pending_g1.lock() = Some(g1_ids);
        }

        Ok(())
    }

    fn observe_forward(&self, _request_id: &str, _meta: &KvConnectorMetadata) -> Result<()> {
        // Tracking is now established at `ensure_started` time
        // (where we know the request is CD-bound and can compute
        // expected output hashes). The offload pipeline observer
        // is registered ONCE at coordinator construction; this
        // per-tick hook is no longer needed for output capture.
        // Kept on the trait for now in case the wrapper has
        // other forward-pass-time needs; remove in Phase B.
        Ok(())
    }

    fn on_request_finished(&self, request_id: &str) {
        let Some(state) = self.state_for(request_id) else {
            return;
        };
        state.request_finished_seen.store(true, Ordering::Release);

        // Finish our own commit/availability streams and close.
        if let Some(session) = state.session.lock().clone() {
            let _ = session.finish_commits();
            let _ = session.finish_availability();
            session.close(Some("request_finished".to_string()));
        }

        let mut status = state.status.lock();
        *status = PrefillStatus::Released;
        drop(status);
        // Drops the last `Arc<RequestState>` (assuming no
        // in-flight task still holds one); RAII observer-handle
        // drop evicts the residual entry. Any in-flight task that
        // still holds a clone keeps the entry alive until it
        // finishes — that's intentional, we don't want to evict
        // mid-pipeline.
        self.states.remove(request_id);

        // Suppress unused warning on the unused field — it's
        // checked in `on_usaa` via `pulls_complete` already.
        let _ = AtomicUsize::new(0);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::G2;
    use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
    use kvbm_engine::testing::token_blocks::create_token_sequence;
    use kvbm_logical::manager::BlockManager;

    const BLOCK_SIZE: usize = 16;

    fn make_g2_manager(capacity: usize) -> Arc<BlockManager<G2>> {
        let registry = TestRegistryBuilder::new().build();
        Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(capacity)
                .block_size(BLOCK_SIZE)
                .registry(registry)
                .build(),
        )
    }

    fn make_blocks(
        g2: &Arc<BlockManager<G2>>,
        count: usize,
        start_token: u32,
    ) -> Vec<ImmutableBlock<G2>> {
        let token_sequence = create_token_sequence(count, BLOCK_SIZE, start_token);
        let mutables = g2.allocate_blocks(count).expect("alloc");
        let completes: Vec<_> = mutables
            .into_iter()
            .zip(token_sequence.blocks().iter())
            .map(|(m, tb)| m.complete(tb).expect("complete"))
            .collect();
        g2.register_blocks(completes)
    }

    /// Mixed-batch dispatch: two requests tracked with disjoint
    /// expected-hash sets; one batch contains blocks for both
    /// plus an unrelated block. Each request's residual should
    /// be drained only by its own matches; the unrelated block
    /// is silently ignored.
    #[test]
    fn mixed_batch_routes_per_request_and_ignores_unrelated() {
        let observer = ConditionalDecodeG2Observer::new();
        // No coordinator installed — observe()'s commit_output_blocks
        // dispatch is a silent no-op. We test the pending-state
        // transitions, not the downstream dispatch.

        let mgr = make_g2_manager(8);
        let a_blocks = make_blocks(&mgr, 2, 0); // hashes for "a"
        let b_blocks = make_blocks(&mgr, 1, 100); // hashes for "b"
        let unrelated = make_blocks(&mgr, 1, 9000); // not tracked

        let a_hashes: HashSet<_> = a_blocks.iter().map(|b| b.sequence_hash()).collect();
        let b_hashes: HashSet<_> = b_blocks.iter().map(|b| b.sequence_hash()).collect();

        let _h_a = observer.track("a".into(), a_hashes.clone());
        let _h_b = observer.track("b".into(), b_hashes.clone());
        assert_eq!(observer.tracked_count(), 2);

        // Mixed batch: 1 of "a"'s blocks, "b"'s block, unrelated.
        let batch: Vec<_> = vec![
            a_blocks[0].clone(),
            b_blocks[0].clone(),
            unrelated[0].clone(),
        ];
        observer.observe(&batch);

        // "a" still has 1 hash residual (a_blocks[1] not yet seen).
        let pending_a = observer.pending_for("a").expect("a still tracked");
        assert_eq!(pending_a.len(), 1);
        assert!(pending_a.contains(&a_blocks[1].sequence_hash()));

        // "b" auto-dropped (its only hash matched).
        assert!(
            observer.pending_for("b").is_none(),
            "b should be auto-dropped after full match"
        );
        assert_eq!(observer.tracked_count(), 1);

        // Second batch with "a"'s remaining block — "a" should auto-drop.
        observer.observe(&[a_blocks[1].clone()]);
        assert!(observer.pending_for("a").is_none());
        assert_eq!(observer.tracked_count(), 0);
    }

    /// untrack_request evicts a residual entry that didn't fully
    /// drain (failure path).
    #[test]
    fn untrack_request_evicts_partial_residual() {
        let observer = ConditionalDecodeG2Observer::new();
        let mgr = make_g2_manager(4);
        let blocks = make_blocks(&mgr, 3, 200);
        let hashes: HashSet<_> = blocks.iter().map(|b| b.sequence_hash()).collect();

        // Hold the handle alive so RAII drop doesn't pre-empt
        // the manual untrack we're testing.
        let _h = observer.track("req-fail".into(), hashes);
        // Observe only 1 of 3 — residual non-empty.
        observer.observe(&[blocks[0].clone()]);
        assert_eq!(observer.pending_for("req-fail").unwrap().len(), 2);

        observer.untrack_request("req-fail");
        assert!(observer.pending_for("req-fail").is_none());
    }

    /// RAII observer handle: dropping the handle evicts the
    /// residual entry from `pending`. Models the production drop
    /// path where `RequestState::_observer_handle` is dropped
    /// alongside the per-request state.
    #[test]
    fn observer_handle_drop_evicts_pending_entry() {
        let observer = ConditionalDecodeG2Observer::new();
        let mgr = make_g2_manager(4);
        let blocks = make_blocks(&mgr, 2, 400);
        let hashes: HashSet<_> = blocks.iter().map(|b| b.sequence_hash()).collect();

        let handle = observer.track("req-raii".into(), hashes);
        assert_eq!(observer.tracked_count(), 1);
        assert!(observer.pending_for("req-raii").is_some());

        drop(handle);

        assert_eq!(observer.tracked_count(), 0);
        assert!(observer.pending_for("req-raii").is_none());
    }

    /// Dropping a handle whose entry was already auto-removed
    /// (full match) is a safe no-op — the underlying
    /// `untrack_request` is idempotent.
    #[test]
    fn observer_handle_drop_after_auto_remove_is_noop() {
        let observer = ConditionalDecodeG2Observer::new();
        let mgr = make_g2_manager(2);
        let blocks = make_blocks(&mgr, 1, 500);
        let hash = blocks[0].sequence_hash();

        let handle = observer.track("a".into(), [hash].into_iter().collect());
        // Full match — entry auto-drops from `pending`.
        observer.observe(&[blocks[0].clone()]);
        assert_eq!(observer.tracked_count(), 0);

        // Handle drop should not panic, observer state unchanged.
        drop(handle);
        assert_eq!(observer.tracked_count(), 0);
    }

    /// re-observing an already-popped block is a no-op
    /// (idempotent against offload P-rule re-emission).
    #[test]
    fn re_observe_is_idempotent() {
        let observer = ConditionalDecodeG2Observer::new();
        let mgr = make_g2_manager(2);
        let blocks = make_blocks(&mgr, 1, 300);
        let hash = blocks[0].sequence_hash();

        let _h = observer.track("a".into(), [hash].into_iter().collect());
        observer.observe(&[blocks[0].clone()]);
        // Auto-dropped after full match.
        assert_eq!(observer.tracked_count(), 0);

        // Re-observing the same block: no panic, no resurrection
        // of the dropped entry.
        observer.observe(&[blocks[0].clone()]);
        assert_eq!(observer.tracked_count(), 0);
    }
}
