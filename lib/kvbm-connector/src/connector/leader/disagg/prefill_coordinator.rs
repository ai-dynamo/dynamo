// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill-side per-request coordinator trait + production impl.
//!
//! The coordinator owns the async state machine for a single
//! CD-bound request on the prefill participant: attach to D's
//! session, request D's `sequence_hashes` block sets, RDMA-pull
//! them, register them in P's G2, drive the G2→G1 onboard once
//! USAA arrives with the G1 destinations, and tear down once
//! D has acknowledged its `PullComplete`.
//!
//! The wrapper ([`super::prefill_leader::PrefillDisaggLeader`])
//! interacts with the coordinator at the `ConnectorLeaderApi`
//! boundary; the coordinator does not see vLLM directly.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, anyhow};
use dashmap::DashMap;
use futures::StreamExt;
use kvbm_common::LogicalLayoutHandle;
use kvbm_disagg_protocol::RemotePrefillParams;
use kvbm_engine::disagg::{
    BlockSetRequest, HashSelection, PrefillSession, PullAck, PullComplete, RemoteBlockRef,
    RemoteBlockSet, SessionEvent,
};
use kvbm_logical::blocks::{CompleteBlock, ImmutableBlock, MutableBlock};
use parking_lot::Mutex;
use tokio::runtime::Handle;

use crate::connector::leader::scheduler::KvConnectorMetadata;
use crate::{BlockId, G2, SequenceHash};

use super::transport::{CdBlockTransport, CdWorkerHook, InnerLeaderShim, PrefillSessionAttacher};

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
    /// attach + diff + remote-pull asynchronously. Subsequent
    /// calls are no-ops. Returns the number of external tokens
    /// the wrapper should report — `params.sequence_hashes.len()
    /// * block_size` — computed once and cached.
    fn ensure_started(
        &self,
        request_id: &str,
        params: &RemotePrefillParams,
    ) -> Result<usize>;

    /// USAA-1 hand-off.
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
    /// inner build returns so the coordinator can install the
    /// offload-completion observer that captures G1→G2 outputs
    /// for this request and publishes them on the session.
    fn observe_forward(
        &self,
        request_id: &str,
        meta: &KvConnectorMetadata,
    ) -> Result<()>;

    /// Detach the slot side.
    ///
    /// The wrapper calls this after `inner.request_finished`.
    /// The session continues to live, holding output
    /// `ImmutableBlock<G2>` pins, until D's `PullComplete`
    /// arrives — at which point the coordinator sends `PullAck`
    /// and drops state.
    fn on_request_finished(&self, request_id: &str);
}

// ============================================================================
// PrefillCoordinatorImpl — golden-path implementation
// ============================================================================

/// Per-request state machine state.
///
/// `Attaching` → `Pulling` → `Registered` → `OnboardingScheduled`
/// → `OnboardingComplete` → `SlotDone` → `Released`.
/// Error states are deferred (see plan §"Phase A error-path
/// design (deferred)").
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
    /// Attached session — populated once `attach` resolves.
    session: Mutex<Option<Arc<dyn PrefillSession>>>,
    /// Peer instance id, taken from
    /// `RemotePrefillParams.initiator_instance_id`.
    /// (P never receives a session-level Attached event from D
    /// — D sends Attached on its own side after P's Attach
    /// frame arrives.)
    peer_instance_id: crate::InstanceId,
    /// Mutables allocated for the remote pull. Drained into
    /// `registered_g2` after pull + register.
    pulled_mutables: Mutex<Option<Vec<MutableBlock<G2>>>>,
    /// Registered G2 blocks pulled from D, held to keep them
    /// pinned through the G2→G1 onboard.
    registered_g2: Mutex<Vec<ImmutableBlock<G2>>>,
    /// G1 destinations from USAA, stashed until pull/register
    /// completes so we can kick the G2→G1 onboard.
    pending_g1: Mutex<Option<Vec<BlockId>>>,
    /// Forward-pass output captures (held to keep pinned until
    /// D's `PullComplete`).
    output_pins: Mutex<Vec<ImmutableBlock<G2>>>,
    /// Wrapper has called `on_request_finished` — the slot is
    /// gone but the session keeps living until PullComplete.
    request_finished_seen: AtomicBool,
    /// D has sent `PullComplete` (and we acked it); output
    /// pins are dropped. Final cleanup waits until
    /// `request_finished_seen` is also true.
    pull_complete_seen: AtomicBool,
    status: Mutex<PrefillStatus>,
}

impl RequestState {
    fn new(
        num_external_tokens: usize,
        expected_hashes: Vec<SequenceHash>,
        peer_instance_id: crate::InstanceId,
    ) -> Self {
        Self {
            num_external_tokens,
            expected_hashes,
            session: Mutex::new(None),
            peer_instance_id,
            pulled_mutables: Mutex::new(None),
            registered_g2: Mutex::new(Vec::new()),
            pending_g1: Mutex::new(None),
            output_pins: Mutex::new(Vec::new()),
            request_finished_seen: AtomicBool::new(false),
            pull_complete_seen: AtomicBool::new(false),
            status: Mutex::new(PrefillStatus::Attaching),
        }
    }
}

/// Production coordinator.
pub struct PrefillCoordinatorImpl {
    inner: Arc<dyn InnerLeaderShim>,
    transport: Arc<dyn CdBlockTransport>,
    worker_hook: Arc<dyn CdWorkerHook>,
    attacher: Arc<dyn PrefillSessionAttacher>,
    runtime: Handle,
    states: DashMap<String, Arc<RequestState>>,
    weak_self: std::sync::Weak<Self>,
}

impl PrefillCoordinatorImpl {
    pub fn new(
        inner: Arc<dyn InnerLeaderShim>,
        transport: Arc<dyn CdBlockTransport>,
        worker_hook: Arc<dyn CdWorkerHook>,
        attacher: Arc<dyn PrefillSessionAttacher>,
        runtime: Handle,
    ) -> Arc<Self> {
        Arc::new_cyclic(|weak_self| Self {
            inner,
            transport,
            worker_hook,
            attacher,
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

    /// Test hook: simulate G1→G2 outputs landing for a CD-bound
    /// request. Production wires this through
    /// `Pipeline::add_register_observer`; for A.4/A.5 the
    /// wrapper test calls this directly.
    pub fn simulate_offload_complete(
        &self,
        request_id: &str,
        blocks: Vec<ImmutableBlock<G2>>,
    ) -> Result<()> {
        let Some(state) = self.state_for(request_id) else {
            anyhow::bail!("simulate_offload_complete: unknown request {}", request_id);
        };
        let session = state
            .session
            .lock()
            .clone()
            .ok_or_else(|| {
                anyhow!(
                    "simulate_offload_complete: session not attached for {}",
                    request_id
                )
            })?;

        let block_sets = vec![RemoteBlockSet {
            source_layout: LogicalLayoutHandle::G2,
            blocks: blocks
                .iter()
                .map(|b| RemoteBlockRef {
                    block_id: b.block_id(),
                    sequence_hash: b.sequence_hash(),
                })
                .collect(),
        }];

        // Pin captured outputs until D acks PullComplete.
        state.output_pins.lock().extend(blocks);

        let publish = session.publish_output_block_sets(block_sets);
        self.runtime.spawn(async move {
            if let Err(err) = publish.await {
                tracing::error!(error = %err, "publish_output_block_sets failed");
            }
        });

        Ok(())
    }

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
        let session = self.attacher.attach(params.session_id, endpoint).await?;
        *state.session.lock() = Some(Arc::clone(&session));

        // 2. Spawn the session monitor.
        let monitor_self = Arc::clone(&self);
        let monitor_request_id = request_id.clone();
        let mut event_stream = session.subscribe();
        self.runtime.spawn(async move {
            while let Some(event) = event_stream.next().await {
                match event {
                    SessionEvent::PullComplete(complete) => {
                        monitor_self
                            .handle_pull_complete(&monitor_request_id, complete)
                            .await;
                    }
                    SessionEvent::Detached { reason: _ } | SessionEvent::Failed { reason: _ } => {
                        // Error-path teardown deferred; just stop the monitor.
                        break;
                    }
                    _ => {}
                }
            }
        });

        // 3. Request D's blocks for all hashes.
        *state.status.lock() = PrefillStatus::Pulling;
        let response = session
            .request_block_sets(BlockSetRequest {
                request_id: request_id.clone(),
                hashes: HashSelection::All,
            })
            .await?;

        // D may advertise pending hashes for the remote slice
        // (the blocks P will produce). P doesn't pull those —
        // it computes them. Consume only `response.ready`.
        let total_blocks: usize = response.ready.iter().map(|s| s.blocks.len()).sum();
        if total_blocks != state.expected_hashes.len() {
            anyhow::bail!(
                "BlockSetResponse ready block count {} != expected {} (D advertised \
                 {} pending hashes)",
                total_blocks,
                state.expected_hashes.len(),
                response.pending_hashes.len()
            );
        }

        // 4. Allocate P-side G2 destinations and pull.
        let mutables = self.inner.allocate_g2_blocks(total_blocks)?;
        let dst_block_ids: Vec<BlockId> = mutables.iter().map(|m| m.block_id()).collect();
        *state.pulled_mutables.lock() = Some(mutables);

        self.transport
            .pull_remote(
                state.peer_instance_id,
                response.ready.clone(),
                dst_block_ids.clone(),
            )
            .await?;

        // 5. Register pulled blocks.
        let token_blocks = self
            .inner
            .token_blocks_for_range(&request_id, 0..total_blocks)?;

        let mutables = state
            .pulled_mutables
            .lock()
            .take()
            .ok_or_else(|| anyhow!("pulled_mutables disappeared mid-register"))?;
        let mut completes: Vec<CompleteBlock<G2>> = Vec::with_capacity(total_blocks);
        for (mutable, token_block) in mutables.into_iter().zip(token_blocks.iter()) {
            completes.push(
                mutable
                    .complete(token_block)
                    .map_err(|err| anyhow!("MutableBlock::complete failed: {:?}", err))?,
            );
        }
        let registered = self.inner.register_g2_blocks(completes)?;
        *state.registered_g2.lock() = registered;
        *state.status.lock() = PrefillStatus::Registered;

        // 6. If USAA already happened, kick onboard now.
        let g1 = state.pending_g1.lock().take();
        if let Some(g1_ids) = g1 {
            self.kick_onboard(&request_id, &state, g1_ids).await?;
        }

        Ok(())
    }

    async fn kick_onboard(
        self: &Arc<Self>,
        request_id: &str,
        state: &Arc<RequestState>,
        g1_dst_block_ids: Vec<BlockId>,
    ) -> Result<()> {
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

    async fn handle_pull_complete(
        self: &Arc<Self>,
        request_id: &str,
        complete: PullComplete,
    ) {
        let Some(state) = self.state_for(request_id) else {
            return;
        };
        let session = match state.session.lock().clone() {
            Some(s) => s,
            None => return,
        };

        let pull_id = complete.pull_id;
        if let Err(err) = session.ack_pull_from_prefill(PullAck { pull_id }).await {
            tracing::error!(request_id, error = %err, "ack_pull_from_prefill failed");
            return;
        }

        // Drop output pins — D has confirmed it's done with them.
        state.output_pins.lock().clear();
        state.pull_complete_seen.store(true, Ordering::Release);

        if state.request_finished_seen.load(Ordering::Acquire) {
            *state.status.lock() = PrefillStatus::Released;
            self.states.remove(request_id);
            session.close(Some("released".to_string()));
        }
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
            params.initiator_instance_id,
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
        let already_registered = matches!(*state.status.lock(), PrefillStatus::Registered);

        if already_registered {
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
                        "prefill onboard failed (error path is Phase A deferred work)"
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
        // via `simulate_offload_complete`. No-op for now.
        Ok(())
    }

    fn on_request_finished(&self, request_id: &str) {
        let Some(state) = self.state_for(request_id) else {
            return;
        };
        state.request_finished_seen.store(true, Ordering::Release);

        // If `PullComplete` already arrived, do final cleanup
        // now (the PullComplete handler skipped removal because
        // request_finished_seen wasn't true at the time).
        if state.pull_complete_seen.load(Ordering::Acquire) {
            let session = state.session.lock().clone();
            *state.status.lock() = PrefillStatus::Released;
            self.states.remove(request_id);
            if let Some(session) = session {
                session.close(Some("released".to_string()));
            }
            return;
        }

        let mut status = state.status.lock();
        if matches!(*status, PrefillStatus::Released) {
            return;
        }
        *status = PrefillStatus::SlotDone;
    }
}
