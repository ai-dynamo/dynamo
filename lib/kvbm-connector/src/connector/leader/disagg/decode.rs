// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode-side conditional-disaggregation coordinator.
//!
//! Owns per-request session lifecycle: opens a holder-side
//! [`Session`], commits + makes-available decode's local-match
//! G2 blocks (plus stream terminators), and queues the
//! [`RemotePrefillRequest`] for the prefill peer to attach.
//!
//! The wrapper ([`super::DecodeDisaggLeader`]) drives the puller
//! half — subscribing to `session.commits()` / `availability()`
//! to consume the prefill peer's outputs, and calling
//! `session.pull(...)` to RDMA-pull each chunk.  This coordinator
//! does not own the pull pipeline; it owns session lifecycle +
//! queue dispatch only.
//!
//! Phase A error-path semantics (attach timeout, peer detach,
//! enqueue failure recovery) remain deferred — see the canonical
//! plan §"Phase A error-path design (deferred)".

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dashmap::DashMap;
use futures::StreamExt;
use kvbm_disagg_protocol::{DISAGG_PROTOCOL_VERSION, RemotePrefillRequest, SessionId};
use kvbm_engine::disagg::session::{LifecycleEvent, Session, SessionFactory};
use kvbm_logical::blocks::ImmutableBlock;
use parking_lot::Mutex;
use tokio::runtime::Handle;

/// Defensive watchdog for the decode-side lifecycle watcher.
/// In normal operation, the rendezvous-driven Detached
/// arrives within a few ms of `process_finished_onboarding`
/// (decode calls `session.finalize()`, prefill is already
/// finalized or imminent). 60s is a well-padded fallback so
/// session leaks are visible as evictions rather than as
/// unbounded gauge drift.
const LIFECYCLE_WATCHDOG: Duration = Duration::from_secs(60);

use super::queue::RemotePrefillQueue;
use super::{ConditionalDisaggPolicy, PolicyInputs, PrefillSelection};
use crate::{G2, InstanceId, SequenceHash};

#[derive(Clone)]
pub struct BeginOutcome {
    pub session_id: SessionId,
    pub session: Arc<dyn Session>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemotePrefillStatus {
    Active,
    Failed,
    Released,
}

pub struct RemotePrefillState {
    pub session: Arc<dyn Session>,
    pub initiator_instance_id: InstanceId,
    pub status: RemotePrefillStatus,
    pub failure_reason: Option<String>,
}

pub struct RemotePrefillCoordinator {
    policy: Arc<dyn ConditionalDisaggPolicy>,
    session_factory: Arc<dyn SessionFactory>,
    queue: Arc<dyn RemotePrefillQueue>,
    states: DashMap<String, Arc<Mutex<RemotePrefillState>>>,
    tokio_handle: Handle,
}

impl RemotePrefillCoordinator {
    pub fn new(
        policy: Arc<dyn ConditionalDisaggPolicy>,
        session_factory: Arc<dyn SessionFactory>,
        queue: Arc<dyn RemotePrefillQueue>,
        tokio_handle: Handle,
    ) -> Arc<Self> {
        Arc::new(Self {
            policy,
            session_factory,
            queue,
            states: DashMap::new(),
            tokio_handle,
        })
    }

    pub fn evaluate(&self, inputs: &PolicyInputs) -> PrefillSelection {
        self.policy.evaluate(inputs)
    }

    /// Open a holder-side session, publish decode's local-match
    /// (commit + make_available + finish_*), install per-request
    /// state, and asynchronously queue the prefill request.
    ///
    /// `local_match_g2`'s hashes are committed and the blocks
    /// made available; the session retains pins until the peer's
    /// pull resolves.
    ///
    /// `RemotePrefillRequest.sequence_hashes` carries decode's
    /// **local-match hashes** (what the prefill peer will pull
    /// from us) — empty when decode has no cached prefix to
    /// offer. `token_ids` is the prefill-window slice only: the
    /// full-block prefix `(num_computed_tokens, full_block_external_tokens)`
    /// that prefill must compute KV for. The partial tail block
    /// stays on decode.
    #[tracing::instrument(
        level = "info",
        skip(self, inputs, local_match_g2, prefill_token_ids)
    )]
    pub fn begin_remote_prefill(
        self: &Arc<Self>,
        request_id: &str,
        inputs: &PolicyInputs,
        initiator_instance_id: InstanceId,
        local_match_g2: Vec<ImmutableBlock<G2>>,
        prefill_token_ids: Vec<u32>,
    ) -> Result<BeginOutcome> {
        tracing::info!(
            num_local_match = local_match_g2.len(),
            num_prefill_token_ids = prefill_token_ids.len(),
            num_computed_tokens = inputs.num_computed_tokens,
            %initiator_instance_id,
            "begin_remote_prefill"
        );
        if self.states.contains_key(request_id) {
            anyhow::bail!(
                "begin_remote_prefill called twice for request_id={}",
                request_id
            );
        }

        let local_match_hashes: Vec<SequenceHash> =
            local_match_g2.iter().map(|b| b.sequence_hash()).collect();

        let session_id = uuid::Uuid::new_v4();
        let session = self.session_factory.open(session_id)?;
        tracing::info!(%session_id, "session opened (holder side)");

        // Holder side: publish the committed set + make blocks
        // available, then close both terminator streams.  Decode
        // has nothing more to publish on this session — the only
        // payload is the local-match slice.
        session.commit(local_match_hashes.clone())?;
        session.make_available(local_match_g2)?;
        session.finish_commits()?;
        session.finish_availability()?;

        let state = RemotePrefillState {
            session: Arc::clone(&session),
            initiator_instance_id,
            status: RemotePrefillStatus::Active,
            failure_reason: None,
        };
        self.states
            .insert(request_id.to_string(), Arc::new(Mutex::new(state)));

        // Spawn the lifecycle watcher: when the rendezvous fires
        // (peer also called `finalize` → both sides triggered
        // velo `StreamSender::finalize` → peer monitor sees
        // `Finalized` → `LifecycleEvent::Detached`) OR velo
        // heartbeat detects peer detach OR the watchdog times
        // out, evict the per-request state. The session Arc
        // held by the state map is the last strong ref; dropping
        // it lets the per-session sender task drain and exit.
        let watcher_session = Arc::clone(&session);
        let watcher_request_id = request_id.to_string();
        let watcher_session_id = session_id;
        let watcher_coord = Arc::clone(self);
        let watchdog = LIFECYCLE_WATCHDOG;
        self.tokio_handle.spawn(async move {
            let outcome = tokio::time::timeout(watchdog, async {
                let mut lifecycle = watcher_session.lifecycle();
                while let Some(event) = lifecycle.next().await {
                    match event {
                        LifecycleEvent::Detached { reason } => {
                            crate::audit!(
                                "decode_lifecycle_detached",
                                role = "decode",
                                request_id = %watcher_request_id,
                                session_id = %watcher_session_id,
                                reason = ?reason
                            );
                            return;
                        }
                        LifecycleEvent::Failed { reason } => {
                            crate::audit!(
                                "decode_lifecycle_failed",
                                role = "decode",
                                request_id = %watcher_request_id,
                                session_id = %watcher_session_id,
                                reason = %reason
                            );
                            return;
                        }
                        LifecycleEvent::Attached { .. } => {}
                    }
                }
            })
            .await;
            if outcome.is_err() {
                crate::audit!(
                    "decode_lifecycle_watchdog_fired",
                    role = "decode",
                    request_id = %watcher_request_id,
                    session_id = %watcher_session_id,
                    watchdog_secs = watchdog.as_secs()
                );
                tracing::warn!(
                    request_id = %watcher_request_id,
                    session_id = %watcher_session_id,
                    watchdog_secs = watchdog.as_secs(),
                    "decode lifecycle watchdog fired without Detached; \
                     evicting RemotePrefillState (potential session leak signal)"
                );
            }
            watcher_coord.states.remove(&watcher_request_id);
        });

        let endpoint = session.endpoint();
        tracing::info!(?endpoint, "decode endpoint published; enqueueing remote prefill request");

        let request = RemotePrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: request_id.to_string(),
            session_id,
            initiator_instance_id,
            decode_endpoint: endpoint,
            sequence_hashes: local_match_hashes,
            token_ids: prefill_token_ids,
            num_computed_tokens: inputs.num_computed_tokens,
        };

        // Spawn enqueue separately so a slow queue doesn't block
        // the sync caller.  On failure, mark the request failed
        // — the wrapper's pull pipeline will surface this via
        // the session being closed.
        let coord = Arc::clone(self);
        let request_id_owned = request_id.to_string();
        let queue = Arc::clone(&self.queue);
        self.tokio_handle.spawn(async move {
            tracing::info!(request_id = request_id_owned, "enqueue remote prefill on hub");
            match queue.enqueue(request).await {
                Ok(()) => {
                    tracing::info!(
                        request_id = request_id_owned,
                        "enqueue ok — awaiting prefill peer pull"
                    );
                }
                Err(err) => {
                    let reason = format!("remote prefill enqueue failed: {err}");
                    tracing::error!(
                        request_id = request_id_owned,
                        error = %err,
                        "enqueue failed"
                    );
                    coord.mark_failed(&request_id_owned, reason);
                }
            }
        });

        Ok(BeginOutcome {
            session_id,
            session,
        })
    }

    pub fn state_for(&self, request_id: &str) -> Option<Arc<Mutex<RemotePrefillState>>> {
        self.states.get(request_id).map(|e| Arc::clone(e.value()))
    }

    pub fn status_for(&self, request_id: &str) -> Option<RemotePrefillStatus> {
        self.state_for(request_id).map(|s| s.lock().status)
    }

    pub fn active_count(&self) -> usize {
        self.states.len()
    }

    /// Cooperative end-of-request: declare decode is finished
    /// (worker pulls confirmed via process_finished_onboarding)
    /// and remove the per-request state. Session stays alive
    /// until the rendezvous fires (peer also finalizes) or velo
    /// heartbeat detects the peer is gone — at which point the
    /// lifecycle watcher drops the session Arc. Idempotent.
    pub fn release(&self, request_id: &str) {
        if let Some(state) = self.state_for(request_id) {
            let session = {
                let mut s = state.lock();
                if s.status != RemotePrefillStatus::Failed {
                    s.status = RemotePrefillStatus::Released;
                }
                Arc::clone(&s.session)
            };
            session.finalize(Some("released".to_string()));
        }
    }

    /// Abort path: forcibly tear down the session. Used when
    /// the request has failed and we don't want to wait for
    /// the cooperative rendezvous.
    pub(crate) fn mark_failed(&self, request_id: &str, reason: String) {
        if let Some(state) = self.state_for(request_id) {
            let session = {
                let mut s = state.lock();
                s.status = RemotePrefillStatus::Failed;
                s.failure_reason = Some(reason.clone());
                Arc::clone(&s.session)
            };
            session.close(Some(reason));
        }
    }
}
