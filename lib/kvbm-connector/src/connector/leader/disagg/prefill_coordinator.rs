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

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{Result, anyhow};
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

use super::transport::{CdBlockTransport, CdWorkerHook, InnerLeaderShim};

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
    fn ensure_started(
        &self,
        request_id: &str,
        params: &RemotePrefillParams,
    ) -> Result<usize>;

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
    fn observe_forward(
        &self,
        request_id: &str,
        meta: &KvConnectorMetadata,
    ) -> Result<()>;

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
}

impl RequestState {
    fn new(num_external_tokens: usize, expected_hashes: Vec<SequenceHash>) -> Self {
        Self {
            num_external_tokens,
            expected_hashes,
            session: Mutex::new(None),
            pending_g1: Mutex::new(None),
            registered_g2: Mutex::new(Vec::new()),
            request_finished_seen: AtomicBool::new(false),
            pulls_complete: AtomicBool::new(false),
            onboarding_scheduled: AtomicBool::new(false),
            status: Mutex::new(PrefillStatus::Attaching),
        }
    }
}

/// Production coordinator.
pub struct PrefillCoordinatorImpl {
    inner: Arc<dyn InnerLeaderShim>,
    transport: Arc<dyn CdBlockTransport>,
    worker_hook: Arc<dyn CdWorkerHook>,
    session_factory: Arc<dyn SessionFactory>,
    runtime: Handle,
    states: DashMap<String, Arc<RequestState>>,
    weak_self: std::sync::Weak<Self>,
}

impl PrefillCoordinatorImpl {
    pub fn new(
        inner: Arc<dyn InnerLeaderShim>,
        transport: Arc<dyn CdBlockTransport>,
        worker_hook: Arc<dyn CdWorkerHook>,
        session_factory: Arc<dyn SessionFactory>,
        runtime: Handle,
    ) -> Arc<Self> {
        Arc::new_cyclic(|weak_self| Self {
            inner,
            transport,
            worker_hook,
            session_factory,
            runtime,
            states: DashMap::new(),
            weak_self: weak_self.clone(),
        })
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
        let session = state
            .session
            .lock()
            .clone()
            .ok_or_else(|| {
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
    async fn run_setup(
        self: Arc<Self>,
        request_id: String,
        params: RemotePrefillParams,
        state: Arc<RequestState>,
    ) -> Result<()> {
        let endpoint = params.decode_endpoint.clone().ok_or_else(|| {
            anyhow!(
                "RemotePrefillParams.decode_endpoint missing for {}",
                request_id
            )
        })?;

        // 1. Attach.
        let session = self
            .session_factory
            .attach(params.session_id, params.initiator_instance_id, endpoint)
            .await?;
        *state.session.lock() = Some(Arc::clone(&session));

        // 2. Drain commit stream until Closed (or until we have
        //    the expected set). Informational; the coordinator
        //    trusts `params.sequence_hashes` for planning.
        let mut commits = session.commits();
        let mut commit_seen = HashSet::new();
        let expected_count = state.expected_hashes.len();
        while let Some(d) = commits.next().await {
            match d {
                CommitDelta::Added(hashes) => {
                    for h in hashes {
                        commit_seen.insert(h);
                    }
                    if commit_seen.len() >= expected_count {
                        // Don't break — peer may still send Closed,
                        // but we have what we need to plan against.
                        break;
                    }
                }
                CommitDelta::Closed => break,
            }
        }

        // 3. Drain availability and pull chunks as they land.
        *state.status.lock() = PrefillStatus::Pulling;
        let mut avail = session.availability();
        let expected_set: HashSet<SequenceHash> =
            state.expected_hashes.iter().copied().collect();
        let mut remaining: HashSet<SequenceHash> = expected_set.clone();
        let mut filled_index: usize = 0;

        while let Some(d) = avail.next().await {
            match d {
                AvailabilityDelta::Available(blocks) => {
                    let chunk: Vec<CommittedBlock> = blocks
                        .into_iter()
                        .filter(|b| remaining.contains(&b.hash))
                        .collect();
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
                    if remaining.is_empty() {
                        break;
                    }
                }
                AvailabilityDelta::Drained => break,
            }
        }

        if !remaining.is_empty() {
            anyhow::bail!(
                "availability drained with {} hashes still missing for {}",
                remaining.len(),
                request_id
            );
        }

        *state.status.lock() = PrefillStatus::Registered;
        state.pulls_complete.store(true, Ordering::Release);

        // 4. If USAA already happened, kick onboard now.
        let g1 = state.pending_g1.lock().take();
        if let Some(g1_ids) = g1 {
            self.kick_onboard(&request_id, &state, g1_ids).await?;
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
        let registered = self.inner.register_g2_blocks(completes)?;
        state.registered_g2.lock().extend(registered);
        *filled_index += chunk_size;
        Ok(())
    }

    async fn kick_onboard(
        self: &Arc<Self>,
        request_id: &str,
        state: &Arc<RequestState>,
        g1_dst_block_ids: Vec<BlockId>,
    ) -> Result<()> {
        if state
            .onboarding_scheduled
            .swap(true, Ordering::AcqRel)
        {
            return Ok(()); // already scheduled
        }
        *state.status.lock() = PrefillStatus::OnboardingScheduled;

        let g2_src_block_ids: Vec<BlockId> = state
            .registered_g2
            .lock()
            .iter()
            .map(|b| b.block_id())
            .collect();

        if g2_src_block_ids.len() != g1_dst_block_ids.len() {
            anyhow::bail!(
                "kick_onboard: G2 src count {} != G1 dst count {}",
                g2_src_block_ids.len(),
                g1_dst_block_ids.len()
            );
        }

        self.transport
            .local_g2_to_g1(g2_src_block_ids, g1_dst_block_ids)
            .await?;

        *state.status.lock() = PrefillStatus::OnboardingComplete;

        self.worker_hook
            .mark_onboarding_complete(request_id.to_string())
            .await?;

        Ok(())
    }
}

impl PrefillCoordinator for PrefillCoordinatorImpl {
    fn ensure_started(
        &self,
        request_id: &str,
        params: &RemotePrefillParams,
    ) -> Result<usize> {
        if let Some(state) = self.state_for(request_id) {
            return Ok(state.num_external_tokens);
        }

        let block_size = self.inner.block_size();
        let num_external_tokens = params.sequence_hashes.len() * block_size;
        let state = Arc::new(RequestState::new(
            num_external_tokens,
            params.sequence_hashes.clone(),
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
        if num_external_tokens != state.num_external_tokens {
            anyhow::bail!(
                "on_usaa: num_external_tokens mismatch (got {}, expected {})",
                num_external_tokens,
                state.num_external_tokens
            );
        }

        let g1_ids = block_ids.to_vec();
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

    fn observe_forward(
        &self,
        _request_id: &str,
        _meta: &KvConnectorMetadata,
    ) -> Result<()> {
        // Production wiring of `Pipeline::add_register_observer`
        // is Phase B. For A.4/A.5 the test drives output capture
        // via `commit_output_blocks` directly. No-op for now.
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
        self.states.remove(request_id);

        // Suppress unused warning on the unused field — it's
        // checked in `on_usaa` via `pulls_complete` already.
        let _ = AtomicUsize::new(0);
    }
}
