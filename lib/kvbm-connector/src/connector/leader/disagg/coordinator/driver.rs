// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `ConditionalDisaggCoordinator` — unified per-request coordinator for the R-B
//! collapse (Slice 3: prefill flow).
//!
//! Holds a `DashMap<String, Arc<CdRequest>>` keyed by `request_id`.  All
//! requests are prefill-role in Slice 3; decode role will be added in Slice 4.
//!
//! ### Observer strategy
//!
//! A single [`ConditionalDecodeG2Observer`] is created at construction
//! time and registered ONCE with the offload pipeline. The coordinator
//! installs a type-erased commit closure that captures `Weak<Self>` and
//! re-upgrades per call — decoupled from any concrete coordinator type
//! so future coordinator variants can drive the same observer.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use dashmap::DashMap;
use futures::StreamExt;
use kvbm_engine::p2p::session::{
    AvailabilityDelta, CommitDelta, CommittedBlock, Session, SessionFactory,
};
use kvbm_logical::blocks::{CompleteBlock, ImmutableBlock};
use kvbm_protocols::disagg::{
    DISAGG_PROTOCOL_VERSION, KvHashingRequestEnvelope, RemotePrefillParams, RemotePrefillRequest,
    digest_provided_hashes,
};
use parking_lot::Mutex;
use tokio::runtime::Handle;

use crate::connector::leader::scheduler::KvConnectorMetadata;
use crate::{BlockId, G2, InstanceId, SequenceHash};

use super::super::decode::{BeginOutcome, CdFailureSink};
use super::super::lifecycle::{LIFECYCLE_WATCHDOG, LifecycleOutcome, spawn_lifecycle_watcher};
use super::super::prefill_coordinator::{
    ConditionalDecodeG2Observer, ObserverHandle, PrefillStatus,
};
use super::super::queue::RemotePrefillQueue;
use super::super::{ConditionalDisaggPolicy, PolicyInputs, PrefillSelection};
use super::{CdRequest, CdRequestStatus, DecodeBits, PrefillBits};
use crate::connector::leader::p2p::peer_resolver::PeerResolver;
use crate::connector::leader::p2p::transport::{InnerLeaderShim, P2pBlockTransport, P2pWorkerHook};

/// Type-erased register-observer callback handed to
/// `Pipeline::add_register_observer`.
pub type CdRegisterObserverFn = Arc<dyn Fn(&[ImmutableBlock<G2>]) + Send + Sync + 'static>;

/// Inputs to [`ConditionalDisaggCoordinator::begin_remote_prefill`]. Bundled
/// so the method stays below the argument-count threshold; the RAII install
/// closure stays a separate generic argument to avoid boxing.
pub struct RemotePrefillStart<'a> {
    pub request_id: &'a str,
    pub inputs: &'a PolicyInputs,
    pub initiator_instance_id: InstanceId,
    pub prefix_g2: Vec<ImmutableBlock<G2>>,
    pub local_match_g2: Vec<ImmutableBlock<G2>>,
    pub prefill_token_ids: Vec<u32>,
    /// LoRA adapter name from the slot's request (the same value decode
    /// passed to `kv_hashing::Request`). Forwarded onto the CD wire so
    /// prefill can rebuild an identical `kv_hashing::Request` and
    /// recompute matching PLHs locally.
    pub lora_name: Option<String>,
    /// Raw salt string from the slot's request. Same purpose as
    /// `lora_name` — feeds the prefill-side hasher.
    pub salt: Option<String>,
}

/// Shared dependencies every coordinator constructor needs. Bundled so the
/// ctors stay below the argument-count threshold; these six always travel
/// together.
pub struct CoordinatorParts {
    pub inner: Arc<dyn InnerLeaderShim>,
    pub transport: Arc<dyn P2pBlockTransport>,
    pub worker_hook: Arc<dyn P2pWorkerHook>,
    pub session_factory: Arc<dyn SessionFactory>,
    pub peer_resolver: Arc<dyn PeerResolver>,
    pub runtime: Handle,
}

// ============================================================================
// ConditionalDisaggCoordinator
// ============================================================================

/// Unified per-request coordinator for conditional disaggregation.
///
/// Holds per-request state for both decode-side and prefill-side flows
/// keyed by `request_id`. Each role's per-side state is carried by the
/// `CdRequestRole` enum on each `CdRequest`. Construct via [`Self::new`]
/// (prefill-only) or [`Self::new_with_decode`] (full dual-role).
pub struct ConditionalDisaggCoordinator {
    inner: Arc<dyn InnerLeaderShim>,
    transport: Arc<dyn P2pBlockTransport>,
    worker_hook: Arc<dyn P2pWorkerHook>,
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

    // ---- Decode-role additions (Slice 4) ----
    /// Policy for decode's `evaluate()` call in GNMT.  `None` when the
    /// coordinator is constructed without decode-role wiring (legacy
    /// prefill-only path).
    policy: Option<Arc<dyn ConditionalDisaggPolicy>>,
    /// Queue for publishing `RemotePrefillRequest` to the prefill peer.
    /// `None` when decode-role is not wired.
    queue: Option<Arc<dyn RemotePrefillQueue>>,
    /// Failure sink installed by the wrapper after construction.
    failure_sink: Mutex<Option<Weak<dyn CdFailureSink>>>,
}

impl ConditionalDisaggCoordinator {
    /// Construct with the default production lifecycle watchdog (60s).
    ///
    /// This is the prefill-only constructor (no decode-role policy or queue).
    /// Use [`new_with_decode`] to enable the decode role.
    pub fn new(parts: CoordinatorParts) -> Arc<Self> {
        Self::new_with_watchdog(parts, LIFECYCLE_WATCHDOG)
    }

    /// Test-friendly constructor with an injectable lifecycle watchdog.
    /// Production callers use [`new`](Self::new) which fixes the watchdog
    /// at [`LIFECYCLE_WATCHDOG`] (60s).
    pub fn new_with_watchdog(parts: CoordinatorParts, lifecycle_watchdog: Duration) -> Arc<Self> {
        Self::new_with_watchdog_and_decode(parts, lifecycle_watchdog, None, None)
    }

    /// Construct with both prefill and decode roles enabled (Slice 4).
    pub fn new_with_decode(
        parts: CoordinatorParts,
        policy: Arc<dyn ConditionalDisaggPolicy>,
        queue: Arc<dyn RemotePrefillQueue>,
    ) -> Arc<Self> {
        Self::new_with_watchdog_and_decode(parts, LIFECYCLE_WATCHDOG, Some(policy), Some(queue))
    }

    /// Full constructor used by all public ctors.
    fn new_with_watchdog_and_decode(
        parts: CoordinatorParts,
        lifecycle_watchdog: Duration,
        policy: Option<Arc<dyn ConditionalDisaggPolicy>>,
        queue: Option<Arc<dyn RemotePrefillQueue>>,
    ) -> Arc<Self> {
        let CoordinatorParts {
            inner,
            transport,
            worker_hook,
            session_factory,
            peer_resolver,
            runtime,
        } = parts;
        // Observer wired at construction.  Closure-based commit
        // dispatch (set after `Arc::new_cyclic` produces the coord)
        // forwards observer-matched blocks to
        // `commit_output_blocks` — without this the offload pipeline
        // observes G2 blocks but the prefill session never sees the
        // commit, so decode bails on `commits Closed` short.
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
            lifecycle_watchdog,
            weak_self: weak_self.clone(),
            policy,
            queue,
            failure_sink: Mutex::new(None),
        });
        let weak_coord = Arc::downgrade(&coord);
        observer.install_commit_fn(Arc::new(
            move |request_id: &str, blocks: Vec<ImmutableBlock<G2>>| {
                if let Some(coord) = weak_coord.upgrade()
                    && let Err(err) = coord.commit_output_blocks(request_id, blocks)
                {
                    tracing::error!(
                        request_id,
                        error = %err,
                        "commit_output_blocks failed from observer"
                    );
                }
            },
        ));
        coord
    }

    /// Observer for registration with the offload pipeline.
    pub fn observer(&self) -> &Arc<ConditionalDecodeG2Observer> {
        &self.observer
    }

    /// Returns a closure suitable for `Pipeline::add_register_observer`.
    pub fn observer_callback(&self) -> CdRegisterObserverFn {
        let observer = Arc::clone(&self.observer);
        Arc::new(move |blocks: &[ImmutableBlock<G2>]| observer.observe(blocks))
    }

    pub fn active_count(&self) -> usize {
        self.states.len()
    }

    /// Returns the granular per-side prefill status for `request_id`.
    /// Returns `None` if the request is not tracked or is not prefill-role.
    pub fn status_for(&self, request_id: &str) -> Option<PrefillStatus> {
        self.states
            .get(request_id)
            .and_then(|entry| entry.value().as_prefill().map(|bits| *bits.status.lock()))
    }

    pub fn has_active_request(&self, request_id: &str) -> bool {
        self.states.contains_key(request_id)
    }

    /// Whether the coordinator owns USAA for this prefill request.
    ///
    /// True iff a prefill-role request is tracked AND it has non-zero
    /// external tokens to onboard. The zero-hash case (decode forwarded
    /// no local-match hashes) returns false: the coordinator's state
    /// exists only to receive output via the observer; vLLM's USAA
    /// must flow through the inner connector so an inner cache hit's
    /// `num_external_tokens` is honored without colliding with the
    /// coordinator's `bits.num_external_tokens == 0`.
    pub fn prefill_owns_usaa(&self, request_id: &str) -> bool {
        self.states
            .get(request_id)
            .and_then(|e| {
                e.value().as_prefill().map(|b| {
                    b.num_external_tokens
                        .load(std::sync::atomic::Ordering::Acquire)
                        > 0
                })
            })
            .unwrap_or(false)
    }

    fn state_for(&self, request_id: &str) -> Option<Arc<CdRequest>> {
        self.states.get(request_id).map(|e| Arc::clone(e.value()))
    }

    fn arc_self(&self) -> Option<Arc<Self>> {
        self.weak_self.upgrade()
    }

    /// Commit forward-pass output blocks to the peer via the session's
    /// commit + availability streams.
    ///
    /// Session attachment is asynchronous in `run_setup`. The G1→G2
    /// register observer (vLLM forward-pass + offload-pipeline lift)
    /// can fire BEFORE `run_setup` has set `state.session` — common on
    /// cold-cache R1 in asymmetric topologies (e.g. TP=2 decode +
    /// TP=1 prefill, where prefill's lift wins the race against
    /// velo's `factory.attach`). Rather than dropping those blocks
    /// (which leaves decode stuck at `decode_commits_closed_short
    /// seen=0`), buffer them in `bits.pending_output_commits`;
    /// `run_setup` drains the buffer atomically when it sets
    /// `state.session = Some(...)`.
    ///
    /// Lock order: acquire `state.session` first, then nest
    /// `bits.pending_output_commits` inside it. Same order used in
    /// `run_setup` so observer + attach can't interleave with the
    /// buffer in an inconsistent state.
    pub fn commit_output_blocks(
        &self,
        request_id: &str,
        blocks: Vec<ImmutableBlock<G2>>,
    ) -> Result<()> {
        let state = self
            .state_for(request_id)
            .ok_or_else(|| anyhow!("commit_output_blocks: unknown request {}", request_id))?;

        let bits = state.as_prefill().ok_or_else(|| {
            anyhow!(
                "commit_output_blocks: request {} is not prefill-role",
                request_id
            )
        })?;

        // Lock the session FIRST; buffer-or-commit decision happens
        // inside this critical section so a concurrent `run_setup`
        // attach can't transition session=None → Some between our
        // check and the buffer append.
        let session_opt = state.session.lock().clone();
        match session_opt {
            None => {
                let buffered = blocks.len();
                bits.pending_output_commits.lock().extend(blocks);
                crate::audit!(
                    "commit_output_blocks_buffered",
                    role = "prefill",
                    request_id,
                    buffered
                );
                Ok(())
            }
            Some(session) => {
                let hashes: Vec<SequenceHash> = blocks.iter().map(|b| b.sequence_hash()).collect();
                session.commit(hashes)?;
                session.make_available(blocks)?;
                Ok(())
            }
        }
    }

    /// Surface a failed CD-bound prefill request to vLLM and tear down local
    /// state.  Idempotent: returns early if state already evicted.
    ///
    /// Order:
    /// 1. `mark_failed_onboarding(rid, g1_or_empty)`
    /// 2. `session.close(reason)` if attached
    /// 3. `states.remove(request_id)`
    pub async fn cleanup_failed_request(self: &Arc<Self>, request_id: &str, reason: String) {
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

        if pre_usaa {
            // Pre-USAA failure: no G1 destinations to report yet.
            // vLLM's connector contract treats `mark_failed_onboarding(
            // rid, [])` as a successful async load. Stash the failure
            // on `PrefillBits.pending_failure` so `on_usaa` can replay
            // it at the next USAA call with the just-arrived G1 ids.
            // If USAA never arrives (e.g. vLLM cancels), the stash is
            // dropped via the slot's RAII payload — no notification is
            // emitted (vLLM owns cancellation).
            if let Some(bits) = state.as_prefill() {
                let mut slot = bits.pending_failure.lock();
                if slot.is_none() {
                    *slot = Some(reason.clone());
                }
            }
            crate::audit!(
                "prefill_cleanup_pending_usaa",
                role = "prefill",
                request_id = %request_id,
                reason = %reason
            );
            // Close the session so the peer learns of the failure.
            // Do NOT remove `self.states[request_id]` — `on_usaa` needs
            // it to find the stashed failure.
            if let Some(session) = state.session.lock().clone() {
                session.close(Some(reason));
            }
            return;
        }

        // Post-USAA failure: emit the failed external G1 ids and tear
        // down. `failed_g1_block_ids` already returns the external
        // window only (g1_block_ids stashed at first non-no-op USAA).
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
            num_external_tokens = bits
                .num_external_tokens
                .load(std::sync::atomic::Ordering::Acquire),
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
        // Store in the top-level shared session field AND atomically
        // drain any output blocks the observer buffered while session
        // was None. The two operations must happen in the same
        // critical section so a concurrent `commit_output_blocks`
        // can't see session=Some and still try to append to the
        // buffer (which would silently lose blocks).
        let buffered_outputs: Vec<ImmutableBlock<G2>> = {
            let mut session_slot = state.session.lock();
            *session_slot = Some(Arc::clone(&session));
            let mut pending = bits.pending_output_commits.lock();
            std::mem::take(&mut *pending)
        };
        if !buffered_outputs.is_empty() {
            crate::audit!(
                "commit_output_blocks_drained",
                role = "prefill",
                request_id = %request_id,
                drained = buffered_outputs.len()
            );
            let hashes: Vec<SequenceHash> =
                buffered_outputs.iter().map(|b| b.sequence_hash()).collect();
            session
                .commit(hashes)
                .context("draining buffered output commits after attach")?;
            session
                .make_available(buffered_outputs)
                .context("draining buffered output availability after attach")?;
        }

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
                    self.pull_and_register_chunk(&request_id, &state, chunk, &mut filled_index)
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

    /// Pull, register, and append blocks for a possibly non-contiguous
    /// availability delta. Splits the input into maximal contiguous runs
    /// (by `expected_hashes` index) and processes each run via
    /// [`Self::pull_and_register_contiguous_chunk`]. The session API does
    /// not guarantee that a single `Available(blocks)` covers a
    /// contiguous slot range — sparse or coalesced deltas are valid.
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

        // Group into maximal contiguous runs; each run is one
        // pull/register transaction (positional order required by
        // `MutableBlock::complete`).
        let mut runs: Vec<Vec<(usize, CommittedBlock)>> = Vec::new();
        for entry in indexed {
            match runs.last_mut() {
                Some(run) if run.last().unwrap().0 + 1 == entry.0 => run.push(entry),
                _ => runs.push(vec![entry]),
            }
        }

        for run in runs {
            self.pull_and_register_contiguous_chunk(request_id, state, bits, run, filled_index)
                .await?;
        }
        Ok(())
    }

    /// Inner helper: ONE contiguous run of expected-hash positions.
    async fn pull_and_register_contiguous_chunk(
        self: &Arc<Self>,
        request_id: &str,
        state: &Arc<CdRequest>,
        bits: &PrefillBits,
        indexed: Vec<(usize, CommittedBlock)>,
        filled_index: &mut usize,
    ) -> Result<()> {
        debug_assert!(!indexed.is_empty());
        debug_assert!(
            indexed.windows(2).all(|w| w[1].0 == w[0].0 + 1),
            "pull_and_register_contiguous_chunk requires contiguous indices"
        );

        let hashes: Vec<SequenceHash> = indexed.iter().map(|(_, b)| b.hash).collect();
        let chunk_size = hashes.len();
        let dst = self.inner.allocate_g2_blocks(chunk_size)?;

        let session = state
            .session
            .lock()
            .clone()
            .ok_or_else(|| anyhow!("pull: session missing for {}", request_id))?;
        let filled = session.pull(hashes.clone(), dst).await?;
        // Transport must return exactly `chunk_size` blocks; a short or
        // long result would silently truncate the zip below.
        if filled.len() != chunk_size {
            anyhow::bail!(
                "session.pull returned {} blocks, expected {} (request_id={})",
                filled.len(),
                chunk_size,
                request_id
            );
        }

        let token_range_start = indexed.first().map(|(i, _)| *i).unwrap_or(0);
        let token_range_end = indexed.last().map(|(i, _)| *i + 1).unwrap_or(0);
        let abs_start = bits.computed_blocks_offset + token_range_start;
        let abs_end = bits.computed_blocks_offset + token_range_end;
        let token_blocks = self
            .inner
            .token_blocks_for_range(request_id, abs_start..abs_end)?;
        if token_blocks.len() != chunk_size {
            anyhow::bail!(
                "token_blocks_for_range returned {} blocks, expected {} (request_id={})",
                token_blocks.len(),
                chunk_size,
                request_id
            );
        }
        let mut completes: Vec<CompleteBlock<G2>> = Vec::with_capacity(chunk_size);
        for (mutable, token_block) in filled.into_iter().zip(token_blocks.iter()) {
            completes.push(
                mutable
                    .complete(token_block)
                    .map_err(|err| anyhow!("MutableBlock::complete failed: {:?}", err))?,
            );
        }
        let registered = self.inner.register_g2_blocks(completes)?;
        if registered.len() != chunk_size {
            anyhow::bail!(
                "register_g2_blocks returned {} blocks, expected {} (request_id={})",
                registered.len(),
                chunk_size,
                request_id
            );
        }
        bits.registered_g2.lock().extend(registered);
        *filled_index += chunk_size;
        Ok(())
    }

    /// Drive G2→G1 onboard for the local-match prefix.
    async fn kick_onboard(
        self: &Arc<Self>,
        request_id: &str,
        state: &Arc<CdRequest>,
        g1_dst_external: Vec<BlockId>,
    ) -> Result<()> {
        let bits = state
            .as_prefill()
            .expect("kick_onboard: prefill role required");

        if bits.onboarding_scheduled.swap(true, Ordering::AcqRel) {
            return Ok(()); // already scheduled
        }
        *bits.status.lock() = PrefillStatus::OnboardingScheduled;

        let bits_external_tokens = bits
            .num_external_tokens
            .load(std::sync::atomic::Ordering::Acquire);
        if bits_external_tokens == 0 {
            tracing::info!(
                "kick_onboard: num_external_tokens=0 — \
                 no G2→G1 onboard needed (cold-cache CD); \
                 marking complete (no worker_hook signal)"
            );
            *bits.status.lock() = PrefillStatus::OnboardingComplete;
            return Ok(());
        }

        // We pull all `expected_hashes.len()` blocks into G2; vLLM
        // gave us only `external_blocks` G1 destinations (the
        // external suffix beyond the local-prefix blocks vLLM
        // already has cached). Onboard the SUFFIX of registered_g2
        // matching the external_blocks count.
        let registered_count = bits.registered_g2.lock().len();
        let external_blocks = g1_dst_external.len();
        if external_blocks > registered_count {
            anyhow::bail!(
                "kick_onboard: external_blocks {} > registered_g2 count {} \
                 (num_external_tokens={}, expected_hashes.len()={})",
                external_blocks,
                registered_count,
                bits_external_tokens,
                bits.expected_hashes.len(),
            );
        }
        let suffix_start = registered_count - external_blocks;
        let g2_src_block_ids: Vec<BlockId> = bits
            .registered_g2
            .lock()
            .iter()
            .skip(suffix_start)
            .map(|b| b.block_id())
            .collect();

        self.transport
            .local_g2_to_g1(g2_src_block_ids, g1_dst_external)
            .await?;

        *bits.status.lock() = PrefillStatus::OnboardingComplete;

        self.worker_hook
            .mark_onboarding_complete(request_id.to_string())
            .await?;

        Ok(())
    }
}

// ============================================================================
// Decode-role public API (Slice 4)
// ============================================================================

impl ConditionalDisaggCoordinator {
    /// Return the decode-role state for `request_id`, or `None` if absent or
    /// not decode-role.  Used by decode-only tests that want concrete access.
    pub fn state_for_decode(&self, request_id: &str) -> Option<Arc<CdRequest>> {
        self.states.get(request_id).and_then(|e| {
            let req = Arc::clone(e.value());
            if req.as_decode().is_some() {
                Some(req)
            } else {
                None
            }
        })
    }

    /// Coarse decode status, mapped from `CdRequestStatus` + `failure_reason`
    /// to the `RemotePrefillStatus` shape tests use.
    pub fn status_for_decode(
        &self,
        request_id: &str,
    ) -> Option<super::super::decode::RemotePrefillStatus> {
        use super::super::decode::RemotePrefillStatus;
        let state = self.state_for_decode(request_id)?;
        let bits = state
            .as_decode()
            .expect("state_for_decode filters to decode-role");
        match *state.status.lock() {
            CdRequestStatus::Released => {
                if bits.failure_reason.lock().is_some() {
                    Some(RemotePrefillStatus::Failed)
                } else {
                    Some(RemotePrefillStatus::Released)
                }
            }
            CdRequestStatus::Active => Some(RemotePrefillStatus::Active),
        }
    }

    /// Convenience wrapper for tests: the unfilled G1 block-ids for a
    /// decode-role request (union of unfinished local+remote slices).
    pub fn unfilled_g1_block_ids_for(&self, request_id: &str) -> Vec<BlockId> {
        self.state_for_decode(request_id)
            .map(|s| s.failed_g1_block_ids())
            .unwrap_or_default()
    }

    /// Invoke the failure sink (if installed) with `reason`, then evict
    /// the state entry.  Idempotent if state already evicted.
    async fn invoke_decode_failure_sink_and_evict(
        self: &Arc<Self>,
        request_id: &str,
        reason: String,
    ) {
        let sink_handle = self.failure_sink.lock().clone();
        if let Some(sink_weak) = sink_handle {
            if let Some(sink) = sink_weak.upgrade() {
                crate::audit!(
                    "decode_failure_sink_invoked",
                    role = "decode",
                    request_id = %request_id,
                    reason = %reason
                );
                sink.on_session_failure(request_id.to_string(), reason)
                    .await;
            } else {
                crate::audit!(
                    "decode_failure_sink_unavailable",
                    role = "decode",
                    request_id = %request_id,
                    reason = "wrapper Arc dropped"
                );
            }
        } else {
            crate::audit!(
                "decode_failure_sink_unavailable",
                role = "decode",
                request_id = %request_id,
                reason = "no failure sink installed"
            );
        }
        self.states.remove(request_id);
    }

    /// Map a [`LifecycleOutcome`] to a failure reason for the decode-side watcher,
    /// or `None` for cooperative paths (Detached / StreamEnded).
    fn decode_failure_reason(outcome: &LifecycleOutcome) -> Option<String> {
        match outcome {
            LifecycleOutcome::Detached { .. } | LifecycleOutcome::StreamEnded => None,
            LifecycleOutcome::Failed { reason } => Some(reason.clone()),
            LifecycleOutcome::WatchdogTimeout { watchdog } => Some(format!(
                "decode lifecycle watchdog fired ({}s) without Detached or Failed",
                watchdog.as_secs()
            )),
        }
    }
}

// ============================================================================
// Decode-role inherent methods
// ============================================================================

impl ConditionalDisaggCoordinator {
    pub fn evaluate(&self, inputs: &PolicyInputs) -> PrefillSelection {
        self.policy
            .as_ref()
            .expect("evaluate: ConditionalDisaggCoordinator not wired for decode role")
            .evaluate(inputs)
    }

    pub fn install_failure_sink(&self, sink: Weak<dyn CdFailureSink>) {
        *self.failure_sink.lock() = Some(sink);
    }

    /// Begin a remote-prefill session.
    ///
    /// `install_payload` is invoked synchronously after coordinator state
    /// is inserted but BEFORE the enqueue task is spawned. This ordering
    /// closes a race where a fast queue-enqueue failure could fire
    /// `mark_failed` (and via the failure sink, cleanup) before the
    /// caller had a chance to install the slot's RAII payload — leaving
    /// the wrapper holding a payload referencing a state that had
    /// already been cleaned up. With the install in-line, by the time
    /// the spawned task can run, the payload is already on the slot.
    ///
    /// # `prefix_g2` vs. `local_match_g2`
    ///
    /// Two G2 sets feed the holder side. They route differently:
    ///
    /// * `local_match_g2` — the inner-search hits for the prefill window
    ///   `[num_computed_tokens / bs, num_computed_tokens / bs + matched)`.
    ///   Hashes ride into BOTH `session.commit + make_available` AND
    ///   `RemotePrefillRequest.sequence_hashes`. The prefill peer pulls
    ///   them via `pull_and_register_*`, placing each at absolute block
    ///   index `decode_offset_blocks + i` (see `ensure_started` —
    ///   `expected_hashes[i] lands at D/BS + i`).
    ///
    /// * `prefix_g2` — the prefix-range backfill hits for
    ///   `[0, num_computed_tokens / bs)`. Hashes go ONLY into
    ///   `session.commit + make_available`; they are **deliberately
    ///   excluded** from `RemotePrefillRequest.sequence_hashes` because
    ///   the request's positional contract would mis-place them
    ///   (prefix_0 covers absolute block 0, not block `D/BS`).
    ///
    /// # Prefix is published but currently has no prefill consumer
    ///
    /// The current prefill code path pulls only `expected_hashes` (=
    /// `params.sequence_hashes` = local-match only). Prefix hashes
    /// pinned via `make_available` sit in decode's `available_pins`
    /// for the session's lifetime and drop when the session closes
    /// (i.e., on request completion).
    ///
    /// This is **intentional**, not a leak:
    ///
    /// * The session model is asymmetric: decode publishes everything
    ///   that's pullable; prefill pulls what it needs. Prefix follows
    ///   the same publisher contract as local-match — only the consumer
    ///   side hasn't been wired yet.
    /// * Future consumers (speculative prefill prefetch, recompute
    ///   fallback, multi-turn cache fill) get prefix-pullable-by-hash
    ///   for free once they exist.
    ///
    /// # Cost — be honest about the upper bound
    ///
    /// Per CD-remote request, the prefix pin holds
    /// `num_computed_blocks × bytes_per_block` of G2 host RAM for the
    /// session's lifetime (decode side). `num_computed_blocks` scales
    /// with the prefix length reported by vLLM, which can grow to
    /// nearly the slot's full context (PC enabled, long multi-turn). It
    /// is **NOT** bounded by the CD inflight token budget — that budget
    /// caps `full_block_external_tokens` (the remote-prefill window), a
    /// different quantity.
    ///
    /// Aggregate ceiling across all concurrent CD-remote sessions:
    ///
    /// ```text
    ///     concurrent_cd_sessions × max_prefix_blocks × bytes_per_block
    /// ```
    ///
    /// where `concurrent_cd_sessions ≤ inflight_cap` (in concurrent
    /// requests, not tokens) and `max_prefix_blocks ≈ slot_context_len /
    /// block_size`. At typical KV sizes (tens of KB per block) and
    /// long-context configs this can reach GBs of pinned host RAM —
    /// non-trivial, monitor at scale.
    ///
    /// Pin lifetime is the full request: from `begin_remote_prefill`
    /// through prefill completion + decode's reverse-direction
    /// generation streaming back through the session. Bandwidth cost is
    /// negligible (one extra `Frame::Commit + Frame::Available` per
    /// request carrying P hash + (hash, peer_block_id) pairs ≈ 32B per
    /// hash).
    ///
    /// If memory pressure becomes observable, a `Frame::DropAvailable`
    /// signal from prefill (or time-based GC of unconsumed entries in
    /// `available_pins`) is the natural release path — it applies
    /// symmetrically to prefix and local-match. The asymmetric session
    /// lifetime (decode keeps pulling prefill's generated outputs after
    /// prefill is "done" pulling decode's prefix) makes a drop-signal
    /// preferable to bilateral session close.
    ///
    /// Today `prefix_g2` is empty unless vLLM prefix-caching is enabled
    /// (and the decode wrapper's `find_prefix_g2_blocks` returns a
    /// non-empty result). With PC disabled (current default)
    /// `num_computed_tokens == 0` and this is a no-op — the cost
    /// discussion above only applies once PC is turned on.
    pub fn begin_remote_prefill<F>(
        &self,
        start: RemotePrefillStart<'_>,
        install_payload: F,
    ) -> Result<BeginOutcome>
    where
        F: FnOnce(&str) -> Result<()>,
    {
        let RemotePrefillStart {
            request_id,
            inputs,
            initiator_instance_id,
            prefix_g2,
            local_match_g2,
            prefill_token_ids,
            lora_name,
            salt,
        } = start;

        // Loud-fail guards for not-yet-wired CD inputs at the decode
        // enqueue site. The envelope mirrors `kv_hashing::Request` so
        // future enablement of LoRA / salt / mm_info is a value change,
        // not a wire-format change — but until each is exercised end-
        // to-end, fail loud here so the first user with such a request
        // gets a clear error instead of an RDMA-pull stall or silent
        // hash divergence. Multimodal is blocked upstream in
        // `kvbm/v2/vllm/schedulers/leader.py` (raises on
        // `mm_features`/`mm_positions`), so `mm_info` is empty by
        // construction here.
        //
        // PC-on (`inputs.num_computed_tokens > 0`) is intentionally NOT
        // guarded here: decode-side enqueue with a non-zero
        // already-computed prefix is well-formed. The mismatch surfaces
        // on the PREFILL side when it tries to recompute absolute-coord
        // PLHs from its windowed slot — see the matching bail in
        // `ensure_started`.
        if lora_name.is_some() {
            anyhow::bail!(
                "CD wire: LoRA not yet wired end-to-end for conditional disagg \
                 (request_id={request_id}); decode-side request carried a LoRA adapter \
                 but the prefill-side recompute path is not validated"
            );
        }
        if salt.is_some() {
            anyhow::bail!(
                "CD wire: cache_salt not yet wired end-to-end for conditional disagg \
                 (request_id={request_id}); decode-side request carried a cache_salt \
                 but the prefill-side recompute path is not validated"
            );
        }

        if self.states.contains_key(request_id) {
            anyhow::bail!(
                "begin_remote_prefill called twice for request_id={}",
                request_id
            );
        }

        let queue = self
            .queue
            .as_ref()
            .expect("begin_remote_prefill: ConditionalDisaggCoordinator not wired for decode role")
            .clone();

        // Local-match hashes ride into BOTH the session and the
        // RemotePrefillRequest. The prefill side uses
        // `RemotePrefillRequest.sequence_hashes[i]` to place blocks at
        // absolute index `decode_offset_blocks + i` (see
        // `ensure_started`'s `expected_hashes[i] lands at D/BS + i`
        // comment), so the request must contain ONLY the local-match
        // range — `[num_computed_tokens / bs, num_computed_tokens / bs +
        // local_match_blocks)`.
        let local_match_hashes: Vec<SequenceHash> =
            local_match_g2.iter().map(|b| b.sequence_hash()).collect();

        // Session-side hashes/blocks include the prefix range
        // `[0, num_computed_tokens / bs)` ahead of the local-match range.
        // The session's commit + make_available expose the full
        // `[0, num_computed_tokens + matched)` window so the prefill peer
        // can pull any block (including prefix) by hash. Prefix blocks
        // are NOT positionally indexed by the prefill side — they're
        // located via hash lookup against the session's available_pins.
        let prefix_hashes: Vec<SequenceHash> =
            prefix_g2.iter().map(|b| b.sequence_hash()).collect();
        let mut session_hashes: Vec<SequenceHash> =
            Vec::with_capacity(prefix_hashes.len() + local_match_hashes.len());
        session_hashes.extend_from_slice(&prefix_hashes);
        session_hashes.extend_from_slice(&local_match_hashes);
        let session_g2: Vec<ImmutableBlock<G2>> =
            prefix_g2.into_iter().chain(local_match_g2).collect();

        let session_id = uuid::Uuid::new_v4();
        let session = self.session_factory.open(session_id)?;

        // Holder side: publish commit + availability, then close terminators.
        session.commit(session_hashes)?;
        session.make_available(session_g2)?;
        session.finish_commits()?;
        session.finish_availability()?;

        // Build DecodeBits — remote_slots / remote_slot_index are empty at
        // this point; they get populated at USAA-1 in decode_leader.rs.
        let bits = DecodeBits {
            reserved_tokens: 0, // wrapper owns the budget; 0 here is a sentinel
            failure_reason: Mutex::new(None),
            local_match_g2_pins: Mutex::new(None),
            local_match_g2_block_ids: Vec::new(),
            local_match_g1_block_ids: Vec::new(),
            local_onboard_complete: std::sync::atomic::AtomicBool::new(false),
            remote_slots: Vec::new(),
            remote_slot_index: HashMap::new(),
            remote_pipeline_complete: std::sync::atomic::AtomicBool::new(false),
            completed: std::sync::atomic::AtomicBool::new(false),
        };

        let state = CdRequest::new_decode(
            request_id.to_string(),
            session_id,
            initiator_instance_id,
            bits,
        );
        *state.session.lock() = Some(Arc::clone(&session));
        self.states
            .insert(request_id.to_string(), Arc::clone(&state));

        // Install the slot RAII payload BEFORE spawning the enqueue task.
        // If install fails, roll back coordinator state and close the
        // session — the wrapper has not yet committed to returning
        // (Some(N), true) for this request, so we abort cleanly.
        //
        // Order: close session, then remove from `states`. Matches the
        // canonical `cleanup_failed_request` teardown order
        // (close → remove); keeps observer semantics uniform across all
        // rollback paths so any consumer of `states[request_id]` sees
        // either {live state + live session} or {no state} but never a
        // {live state + already-removed Arc<Session>} transient.
        if let Err(err) = install_payload(request_id) {
            session.close(Some(format!("payload install failed: {err}")));
            self.states.remove(request_id);
            return Err(err);
        }

        // Spawn lifecycle watcher.
        let watcher_coord = self.weak_self.clone();
        let watcher_request_id = request_id.to_string();
        spawn_lifecycle_watcher(
            &self.runtime,
            Arc::clone(&session),
            "decode",
            request_id.to_string(),
            session_id.to_string(),
            self.lifecycle_watchdog,
            move |outcome| async move {
                let failure_reason = Self::decode_failure_reason(&outcome);
                if let Some(coord) = watcher_coord.upgrade() {
                    if let Some(reason) = failure_reason {
                        coord
                            .invoke_decode_failure_sink_and_evict(&watcher_request_id, reason)
                            .await;
                    } else {
                        coord.states.remove(&watcher_request_id);
                    }
                }
            },
        );

        let endpoint = session.endpoint();

        // DNPT = decode's total committed prefix length (in tokens) from
        // absolute position 0. Folds in the vLLM-decode G1 prefix
        // (`inputs.num_computed_tokens`) AND the G2 local-match window
        // decode just resolved (`local_match_hashes.len() * block_size`).
        // Decode's slot's PLH chain at indices `[0, DNPT/BS)` covers
        // exactly this range; the digest below pins that slice for the
        // prefill-side verifier.
        let block_size = self.inner.block_size();
        let dnpt_tokens = inputs.num_computed_tokens + local_match_hashes.len() * block_size;
        let dnpt_blocks = dnpt_tokens / block_size;
        let provided_hashes: Vec<SequenceHash> = self
            .inner
            .slot_match_split(request_id)?
            .all_sequence_hashes
            .iter()
            .take(dnpt_blocks)
            .copied()
            .collect();
        // Defense-in-depth: digest covers the FULL `[0, DNPT/BS)` slice
        // decode commits to serve. Prefill recomputes the same slice from
        // its own slot's absolute-coord PLH chain and asserts the digest
        // matches before accepting the dispatch.
        let expected_hash_digest = Some(digest_provided_hashes(&provided_hashes));
        let envelope = KvHashingRequestEnvelope {
            lora_name,
            salt,
            mm_info: Vec::new(),
        };

        let request = RemotePrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: request_id.to_string(),
            session_id,
            initiator_instance_id,
            decode_endpoint: endpoint,
            token_ids: prefill_token_ids,
            num_provided_tokens: dnpt_tokens,
            request: envelope,
            expected_hash_digest,
        };

        // Enqueue asynchronously so a slow queue doesn't block the sync caller.
        let coord = self.weak_self.clone();
        let request_id_owned = request_id.to_string();
        self.runtime.spawn(async move {
            match queue.enqueue(request).await {
                Ok(()) => {
                    tracing::info!(
                        request_id = request_id_owned,
                        "decode begin_remote_prefill: enqueue ok"
                    );
                }
                Err(err) => {
                    let reason = format!("remote prefill enqueue failed: {err}");
                    tracing::error!(
                        request_id = request_id_owned,
                        error = %err,
                        "decode begin_remote_prefill: enqueue failed"
                    );
                    if let Some(c) = coord.upgrade() {
                        c.mark_failed(&request_id_owned, reason);
                    }
                }
            }
        });

        Ok(BeginOutcome {
            session_id,
            session,
        })
    }

    pub fn session_for(&self, request_id: &str) -> Option<Arc<dyn Session>> {
        self.states
            .get(request_id)
            .and_then(|e| e.value().session.lock().clone())
    }

    pub fn release(&self, request_id: &str) {
        // Atomic remove() closes the get()-then-remove() race observed under
        // kv_load_failure_policy=recompute (see commit message for details).
        if let Some((_, state)) = self.states.remove(request_id) {
            let session_opt = state.session.lock().clone();
            let mut status = state.status.lock();
            if *status != CdRequestStatus::Released {
                *status = CdRequestStatus::Released;
            }
            drop(status);
            if let Some(session) = session_opt {
                session.finalize(Some("released".to_string()));
            }
        }
    }

    pub fn mark_failed(&self, request_id: &str, reason: String) {
        let mut found = false;
        if let Some(state) = self.states.get(request_id) {
            found = true;
            let session_opt = state.value().session.lock().clone();
            let mut status = state.value().status.lock();
            *status = CdRequestStatus::Released;
            drop(status);
            // Stash failure reason in DecodeBits if available.
            if let Some(bits) = state.value().as_decode() {
                *bits.failure_reason.lock() = Some(reason.clone());
            }
            drop(state);
            if let Some(session) = session_opt {
                session.close(Some(reason.clone()));
            }
        }

        // Drive the failure sink directly so vLLM unblocks promptly,
        // not after the lifecycle watcher's Failed-event roundtrip
        // (which can take up to the watchdog deadline if the session
        // emits no terminal event for any reason). The sink and the
        // watcher path are both idempotent (HashSet inserts and
        // DashMap removes), so a duplicate invocation is safe.
        if found {
            let sink_handle = self.failure_sink.lock().clone();
            if let Some(sink_weak) = sink_handle
                && let Some(sink) = sink_weak.upgrade()
            {
                let request_id_owned = request_id.to_string();
                self.runtime.spawn(async move {
                    sink.on_session_failure(request_id_owned, reason).await;
                });
            }
        }
    }
}

// ============================================================================
// Prefill-role inherent methods
// ============================================================================

impl ConditionalDisaggCoordinator {
    /// Idempotent per-request init for the prefill side.
    ///
    /// `install_payload` is invoked synchronously after coordinator state
    /// is inserted but BEFORE the run_setup task is spawned, and ONLY
    /// when this is the first call (idempotent retries return cached
    /// `num_external_tokens` without re-installing). It is also skipped
    /// when `num_external_tokens == 0` (zero-hash CD case where the
    /// wrapper falls through to the inner connector and inner owns the
    /// async-load slot transaction).
    ///
    /// This ordering closes a race where a fast `run_setup` failure
    /// could fire `cleanup_failed_request` before the wrapper had a
    /// chance to install the slot's RAII payload. It also makes the
    /// install non-idempotent at the wrapper layer (vLLM may call gnmt
    /// multiple times without an intervening USAA; the production slot
    /// rejects a second install).
    pub fn ensure_started<F>(
        &self,
        request_id: &str,
        params: &RemotePrefillParams,
        prefill_num_computed_tokens: usize,
        install_payload: F,
    ) -> Result<usize>
    where
        F: FnOnce(&str) -> Result<()>,
    {
        if let Some(state) = self.state_for(request_id) {
            let bits = state
                .as_prefill()
                .expect("ensure_started: existing request must be prefill-role");
            // Idempotent retry: vLLM may call gnmt multiple times with
            // a different `num_computed_tokens` (e.g., chunked prefill
            // continuation, prefix-cache hit on retry, allocation
            // failure between gnmt + USAA). Recompute and update the
            // reported external_tokens; the underlying
            // `expected_hashes` and pulled blocks are unchanged — the
            // caller's USAA path slices the external SUFFIX of the
            // pulled blocks based on the freshly-computed value.
            let new_external = bits
                .total_position_end_tokens
                .saturating_sub(prefill_num_computed_tokens);
            bits.num_external_tokens
                .store(new_external, std::sync::atomic::Ordering::Release);
            crate::audit!(
                "ensure_started_idempotent_recompute",
                role = "prefill",
                request_id = %request_id,
                prefill_num_computed_tokens = prefill_num_computed_tokens,
                total_position_end_tokens = bits.total_position_end_tokens,
                num_external_tokens = new_external
            );
            return Ok(new_external);
        }

        let block_size = self.inner.block_size();
        // DNPT = decode's total committed prefix in tokens (folds in
        // both vLLM-decode's G1 prefix and the G2 local-match window).
        // Decode sends the FULL prompt `token_ids` on the wire, so
        // prefill's slot is built from the same full sequence and its
        // PLH chain is in ABSOLUTE coordinates — bit-identical to
        // decode's by construction. Decode's slot's PLH chain at
        // `[0, DNPT/BS)` covers exactly what decode commits to serve.
        //
        // Today prefill pulls all of `[0, DNPT/BS)` (conservative —
        // over-pull instead of under-pull). The future PNCT
        // optimization will narrow this to `[PNCT/BS, DNPT/BS)` so
        // prefill skips hashes it already has locally; that requires
        // computing PNCT from prefill-vLLM's GNMT hint + prefill's own
        // G2/G3 local-match search.
        let dnpt_tokens = params.num_provided_tokens;
        let dnpt_blocks = dnpt_tokens / block_size;
        let total_position_end_tokens = dnpt_tokens;
        let num_external_tokens =
            total_position_end_tokens.saturating_sub(prefill_num_computed_tokens);
        // `computed_blocks_offset` is the absolute-coordinate offset
        // where `expected_hashes` starts. Currently 0 because we pull
        // the full `[0, DNPT/BS)` range; when PNCT-aware pulls land,
        // this becomes `PNCT/BS`. `pull_and_register_*` uses it to
        // translate the local position-i element of `expected_hashes`
        // to the peer's absolute token-block index `offset + i`.
        let computed_blocks_offset = 0;
        if prefill_num_computed_tokens > dnpt_tokens {
            crate::audit!(
                "prefill_local_prefix_exceeds_decode_commitment",
                role = "prefill",
                request_id = %request_id,
                dnpt_tokens = dnpt_tokens,
                prefill_num_computed_tokens = prefill_num_computed_tokens,
                num_external_tokens = num_external_tokens
            );
        }

        // Slice prefill's locally-computed PLH chain `[0, DNPT/BS)` —
        // these are the hashes prefill will RDMA-pull from decode.
        let split = self.inner.slot_match_split(request_id)?;
        if dnpt_blocks > split.all_sequence_hashes.len() {
            anyhow::bail!(
                "CD prefill: DNPT window [0, {dnpt_blocks}) exceeds prefill slot's \
                 hashed block count {} (request_id={request_id}); prefill slot's \
                 token_ids must cover at least decode's DNPT range",
                split.all_sequence_hashes.len()
            );
        }
        let expected_hashes: Vec<SequenceHash> = split.all_sequence_hashes[..dnpt_blocks].to_vec();

        // Defense-in-depth: if decode shipped a digest, assert our
        // recomputed `[0, DNPT/BS)` slice matches bit-for-bit.
        // Mismatch means decode and prefill disagree on the hashing
        // inputs (missed salt/LoRA propagation, hasher version skew,
        // etc.) — fail loud here so the operator sees the real cause
        // instead of an RDMA-pull stall.
        if let Some(expected_digest) = params.expected_hash_digest {
            let local_digest = digest_provided_hashes(&expected_hashes);
            if local_digest != expected_digest {
                anyhow::bail!(
                    "CD hash divergence (request_id={request_id}): \
                     decode_digest=0x{expected_digest:016x} \
                     prefill_digest=0x{local_digest:016x}; \
                     KvHashingRequestEnvelope inputs do not match across decode/prefill"
                );
            }
        }

        let pulled: HashSet<SequenceHash> = expected_hashes.iter().copied().collect();
        let expected_outputs: HashSet<SequenceHash> = split
            .all_sequence_hashes
            .iter()
            .filter(|h| !pulled.contains(h))
            .copied()
            .collect();
        let observer_handle: ObserverHandle = self
            .observer
            .track(request_id.to_string(), expected_outputs);

        let bits = PrefillBits {
            num_external_tokens: std::sync::atomic::AtomicUsize::new(num_external_tokens),
            expected_hashes,
            computed_blocks_offset,
            pending_g1: Mutex::new(None),
            g1_block_ids: Mutex::new(None),
            registered_g2: Mutex::new(Vec::new()),
            request_finished_seen: std::sync::atomic::AtomicBool::new(false),
            pulls_complete: std::sync::atomic::AtomicBool::new(false),
            onboarding_scheduled: std::sync::atomic::AtomicBool::new(false),
            status: Mutex::new(PrefillStatus::Attaching),
            observer_handle,
            pending_failure: Mutex::new(None),
            total_position_end_tokens,
            pending_output_commits: Mutex::new(Vec::new()),
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

        // Install the slot RAII payload BEFORE spawning run_setup,
        // and only when there's CD-bound work to onboard. Zero-hash
        // requests (decode forwarded no local-match hashes) leave
        // payload install to the wrapper's inner-passthrough path; the
        // coordinator state still tracks for observer-side output flow.
        if num_external_tokens > 0
            && let Err(err) = install_payload(request_id)
        {
            self.states.remove(request_id);
            return Err(err);
        }

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
            num_provided_tokens = params.num_provided_tokens
        );

        self.runtime.spawn(async move {
            let setup_result = coord
                .clone()
                .run_setup(request_id_owned.clone(), params_owned, state)
                .await;
            if let Err(err) = setup_result {
                tracing::error!(
                    request_id = request_id_owned,
                    error = ?err,
                    "prefill setup failed; surfacing to vLLM via cleanup_failed_request"
                );
                coord
                    .cleanup_failed_request(&request_id_owned, format!("setup failed: {err}"))
                    .await;
            }
        });

        Ok(num_external_tokens)
    }

    pub fn on_usaa(
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

        let bits_external_tokens = bits
            .num_external_tokens
            .load(std::sync::atomic::Ordering::Acquire);
        let block_size = self.inner.block_size();
        let external_blocks = if block_size > 0 {
            bits_external_tokens / block_size
        } else {
            0
        };

        // Replay any pre-USAA failure stash: if `cleanup_failed_request`
        // ran before USAA installed G1 ids, it stashed the reason
        // instead of emitting `mark_failed_onboarding(rid, [])` (which
        // vLLM treats as success). USAA is the first point we have
        // real G1 destinations to report — emit them as failed (the
        // EXTERNAL suffix only, not the local-prefix blocks vLLM
        // already has cached) and tear down without continuing the
        // USAA bookkeeping.
        let pending = bits.pending_failure.lock().clone();
        if let Some(reason) = pending {
            let external_slice_start = block_ids.len().saturating_sub(external_blocks);
            let external_g1_ids: Vec<BlockId> = block_ids[external_slice_start..].to_vec();
            crate::audit!(
                "prefill_usaa_replay_pending_failure",
                role = "prefill",
                request_id = %request_id,
                reason = %reason,
                num_external_g1_ids = external_g1_ids.len(),
                num_total_g1_ids = block_ids.len()
            );
            let coord = self
                .arc_self()
                .ok_or_else(|| anyhow!("coordinator weak_self upgrade failed"))?;
            let request_id_owned = request_id.to_string();
            self.runtime.spawn(async move {
                if let Err(err) = coord
                    .worker_hook
                    .mark_failed_onboarding(request_id_owned.clone(), external_g1_ids)
                    .await
                {
                    tracing::error!(
                        request_id = request_id_owned,
                        error = %err,
                        "prefill on_usaa: mark_failed_onboarding RPC failed"
                    );
                }
                coord.states.remove(&request_id_owned);
            });
            return Ok(());
        }

        // Post-onboard follow-up USAAs are no-ops.
        if num_external_tokens == 0 && bits_external_tokens > 0 {
            return Ok(());
        }

        if num_external_tokens != bits_external_tokens {
            anyhow::bail!(
                "on_usaa: num_external_tokens mismatch (got {}, expected {})",
                num_external_tokens,
                bits_external_tokens
            );
        }

        if block_ids.len() < external_blocks {
            anyhow::bail!(
                "on_usaa: block_ids len {} < external_blocks {} (num_external_tokens={})",
                block_ids.len(),
                external_blocks,
                bits_external_tokens
            );
        }

        // vLLM lays out USAA's `block_ids` as `[local_computed_prefix |
        // external]`. The connector loads the EXTERNAL suffix only;
        // taking `block_ids[..external_blocks]` would copy remote KV
        // into vLLM's already-computed prefix block AND miss the last
        // external block. Correct slice is the suffix.
        let external_slice_start = block_ids.len().saturating_sub(external_blocks);
        let g1_ids: Vec<BlockId> = block_ids[external_slice_start..].to_vec();

        // Stash ONLY the external suffix for failure paths so
        // post-USAA cleanup reports just the external blocks (not the
        // local-prefix blocks vLLM already has).
        {
            let mut stash = bits.g1_block_ids.lock();
            if stash.is_none() {
                *stash = Some(g1_ids.clone());
            }
        }

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

    pub fn observe_forward(&self, _request_id: &str, _meta: &KvConnectorMetadata) -> Result<()> {
        // No-op: tracking is established at ensure_started time.
        Ok(())
    }

    pub fn on_request_finished(&self, request_id: &str) {
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

        // Mark the coarse status Released; per-side PrefillStatus is
        // also set to Released for consistency.
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
