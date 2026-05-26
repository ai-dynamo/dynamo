// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode-side disaggregation leader wrapper.
//!
//! `DecodeDisaggLeader` wraps a base [`super::ConnectorLeaderApi`]
//! and intercepts the scheduler-facing API to drive the
//! disagg dataflow on the decode side, against the
//! symmetric [`Session`](kvbm_engine::p2p::session::Session)
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
use kvbm_engine::p2p::session::{AvailabilityDelta, CommitDelta, Session};
use kvbm_hub::{ConditionalDisaggClient, HubClient};
use kvbm_logical::blocks::{CompleteBlock, ImmutableBlock};
use parking_lot::Mutex;
use velo::InstanceId;

use crate::BlockId;
use crate::connector::leader::scheduler::{KvConnectorMetadata, SchedulerOutput};
use crate::connector::leader::{FinishedStatus, Request};
use crate::{G2, SequenceHash};

use super::coordinator::{ConditionalDisaggCoordinator, RemotePrefillStart};
use super::decode::CdFailureSink;
use super::{ConnectorLeaderApi, PolicyInputs, PrefillSelection};
use crate::connector::leader::p2p::transport::{InnerLeaderShim, P2pBlockTransport, P2pWorkerHook};
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
    coordinator: Weak<ConditionalDisaggCoordinator>,
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

/// Source tier for a pending prefix promotion. Selected at decode
/// GNMT based on which lower tier (G1 or G3) actually backs the
/// requested prefix window. Drives dispatch in `commit_usaa1`'s
/// promotion-spawn block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceTier {
    /// Prefix lives in vLLM's G1 (HBM). Source blocks are built
    /// at USAA from the `block_ids` vLLM hands the wrapper, paired
    /// with the GNMT-time `prefix_hashes`. Promotion runs through
    /// the offload pipeline's `enqueue_g1_to_g2`.
    G1,
    /// Prefix lives in decode's G3 (NVMe/SSD). Source blocks are
    /// resolved by the production shim by querying the G3 manager
    /// for `prefix_hashes` at promotion time. Promotion runs
    /// through `kvbm_engine::leader::staging::stage_g3_to_g2`.
    ///
    /// Producer wiring lands in Stage 2c (extend
    /// `commit_gnmt_remote`'s empty-prefix arm to try G3 first);
    /// dispatch wiring lands in Stage 2d (match on `source_tier`
    /// in `commit_usaa1`'s promotion-spawn block). Until then,
    /// `commit_usaa1` bails on any G3 plan that reaches it.
    #[allow(dead_code)] // wired in Stage 2c/2d
    G3,
}

/// Decision recorded at decode GNMT when the slot's prefix range is
/// known to vLLM (G1) but not yet backed by decode's G2 cache, OR
/// when decode's G3 holds the prefix and a G3→G2 stage is required.
/// The promotion fires at USAA — when vLLM has handed the wrapper
/// the actual G1 `block_ids` (G1 source) or simply at the admission
/// boundary (G3 source) — and runs as an uncancellable task that
/// completes the alloc + transfer + register flow then publishes the
/// resulting G2 blocks on the open CD session.
///
/// `prefix_hashes` are the canonical hashes from the slot's full
/// sequence; they are committed up-front in `begin_remote_prefill`
/// alongside the local-match hashes (so the prefill peer observes
/// the full planned set when `finish_commits` seals the commit set)
/// and re-asserted when the promotion task calls `session.commit`
/// with the same hashes (which deduplicates against the up-front
/// commit — see `lib/kvbm-engine/src/p2p/session/CONTRACT.md` §3.2
/// "Monotonic-add sets"). `session.finish_availability` is
/// **deferred** at GNMT — it only fires once the promotion task
/// reaches `session.make_available(promoted_g2)`.
#[derive(Debug, Clone)]
struct PendingTierPromotion {
    /// Which lower tier the source blocks live in.
    source_tier: SourceTier,
    /// Width of the prefix slice — `num_computed_tokens / block_size`.
    prefix_block_count: usize,
    /// Canonical sequence hashes for blocks `[0..prefix_block_count)`,
    /// in absolute-position order.
    prefix_hashes: Vec<SequenceHash>,
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

    /// Holder-side session captured at gnmt time from
    /// `coordinator.begin_remote_prefill`'s outcome. Carrying the
    /// `Arc<dyn Session>` on the wrapper's per-request state closes
    /// the CD USAA-1 race where a sibling `cleanup_failed_request`
    /// from a previous run of the same request id could call
    /// `coordinator.release` (Bug B's atomic-remove site) between
    /// USAA installing wrapper state and `commit_usaa1` reaching the
    /// remote-pipeline spawn — leaving `coordinator.session_for(rid)`
    /// returning None and the wrapper bailing fatally to vLLM.
    /// With the session held here, the spawn site uses the same
    /// session that was opened for this request; if the coordinator
    /// has finalized it externally, the remote pipeline observes
    /// `CommitDelta::Closed` and routes to `cleanup_failed_request`
    /// — the documented async-failure path.
    ///
    /// Mirrors the lazily-attached shape on the coordinator side
    /// (`CdRequest.session: Mutex<Option<Arc<dyn Session>>>`,
    /// coordinator.rs:284). Set to `Some` synchronously inside
    /// `commit_gnmt_remote` after `begin_remote_prefill` returns
    /// `Ok(outcome)`. By the time `commit_usaa1` reads it, the
    /// invariant is "always Some" — the None branch is unreachable
    /// unless `commit_gnmt_remote` is reordered.
    session: Mutex<Option<Arc<dyn Session>>>,

    /// Pre-USAA failure stash. Set by `cleanup_failed_request` when
    /// the request fails before USAA had a chance to install G1
    /// destinations (no `local_match_g1_block_ids` and no
    /// `remote_slots`). vLLM's connector contract treats an empty
    /// `failed_block_ids` plus `finished_recving` as a successful
    /// async load, so we cannot emit `mark_failed_onboarding(rid, [])`
    /// to surface a pre-USAA failure. Instead the failure is stashed
    /// here and replayed at USAA time with the now-known G1 ids; if
    /// the request is torn down before USAA arrives, no notification
    /// is emitted (vLLM owns the cancellation path in that case).
    pending_failure: Mutex<Option<String>>,

    /// CAS-guard for `cleanup_failed_request`. Five connector-spawned
    /// paths can call cleanup for the same request_id concurrently:
    /// the lifecycle watcher's failure-sink route, commit_usaa1's
    /// local-kick spawn Err, commit_usaa1's remote-pipeline spawn
    /// Err, commit_usaa1's session-missing branch spawn, and the
    /// enqueue spawn's `mark_failed` → failure-sink route. The
    /// winner of the CAS (false → true) runs the cleanup (gathers
    /// unfilled_ids, calls `mark_failed_onboarding`, releases state);
    /// losers early-return so vLLM is not double-notified. Mirrors
    /// the prefill-side guard on `CdRequest.cleanup_claimed`.
    cleanup_claimed: std::sync::atomic::AtomicBool,

    /// Stage 1/2 prefix promotion plan, captured at GNMT when
    /// `num_computed_tokens > 0` but decode's G2 has no record of
    /// the prefix. The plan's `source_tier` indicates whether the
    /// backing comes from vLLM's G1 (Stage 1) or decode's G3 (Stage
    /// 2). Consumed at USAA, where the promotion task is spawned;
    /// the G1 path needs the actual G1 `block_ids` from vLLM, the
    /// G3 path resolves source blocks internally from the hashes.
    /// `None` outside this path (the common case).
    ///
    /// The actual `JoinHandle` for the spawned task lives on the
    /// COORDINATOR-side `DecodeBits.pending_promotion_task`, not
    /// here — `coordinator.release` awaits the handle before
    /// calling `session.finalize` (finalize would otherwise emit
    /// `CommitsClosed + Drained` per session CONTRACT §2.13 and
    /// pre-empt the task's `make_available`).
    pending_promotion: Option<PendingTierPromotion>,
}

/// Opaque test handle wrapping an `Arc<CdRequestState>` snapshot,
/// so that tests/ integration code can capture the per-lifecycle
/// Arc identity without exposing the private `CdRequestState`
/// type. Produced by
/// [`DecodeDisaggLeader::snapshot_cd_state_for_test`] and consumed
/// by [`DecodeDisaggLeader::release_request_if_matches_for_test`].
#[cfg(any(test, feature = "testing"))]
pub struct CdRequestStateHandle(Arc<CdRequestState>);

#[cfg(any(test, feature = "testing"))]
impl CdRequestStateHandle {
    /// Two handles refer to the same lifecycle's Arc.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
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
    coordinator: Arc<ConditionalDisaggCoordinator>,
    transport: Arc<dyn P2pBlockTransport>,
    worker_hook: Arc<dyn P2pWorkerHook>,
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

/// Optional hub-side wiring for a [`DecodeDisaggLeader`]. Grouped so
/// `from_parts` stays below the argument-count threshold; these three are a
/// single conceptual unit (all `None` when the hub path is disabled).
pub struct HubWiring {
    pub hub: Option<Arc<HubClient>>,
    pub client: Option<Arc<ConditionalDisaggClient>>,
    pub hub_velo_id: Option<InstanceId>,
}

impl DecodeDisaggLeader {
    /// Construct a `DecodeDisaggLeader`.
    pub fn from_parts(
        inner: Arc<dyn InnerLeaderShim>,
        config: &DisaggConfig,
        coordinator: Arc<ConditionalDisaggCoordinator>,
        transport: Arc<dyn P2pBlockTransport>,
        worker_hook: Arc<dyn P2pWorkerHook>,
        tokio_handle: tokio::runtime::Handle,
        hub_wiring: HubWiring,
    ) -> Arc<Self> {
        let HubWiring {
            hub,
            client,
            hub_velo_id,
        } = hub_wiring;
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
        let sink: Weak<dyn CdFailureSink> = Arc::downgrade(&leader) as Weak<dyn CdFailureSink>;
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

    pub fn coordinator(&self) -> &Arc<ConditionalDisaggCoordinator> {
        &self.coordinator
    }

    pub fn inflight_available(&self) -> usize {
        self.inflight_budget.available()
    }

    pub fn has_active_cd_request(&self, request_id: &str) -> bool {
        self.cd_request_state.contains_key(request_id)
    }

    /// Test-only: read the wrapper's per-request `cleanup_claimed`
    /// CAS flag. Returns `None` if no `cd_request_state` entry exists.
    #[cfg(any(test, feature = "testing"))]
    pub fn cleanup_claimed_for_test(&self, request_id: &str) -> Option<bool> {
        self.cd_request_state.get(request_id).map(|e| {
            e.value()
                .cleanup_claimed
                .load(std::sync::atomic::Ordering::Acquire)
        })
    }

    /// Test-only: force-set the wrapper's per-request
    /// `cleanup_claimed` flag. Returns `false` if no entry exists.
    /// Used to simulate the racy "existing state had a CAS claimed
    /// before commit_usaa1's rebuild" scenario without timing
    /// fragility — see the rebuild-loses-stash test in
    /// `tests/cd_decode_e2e.rs`.
    #[cfg(any(test, feature = "testing"))]
    pub fn force_cleanup_claimed_for_test(&self, request_id: &str, value: bool) -> bool {
        if let Some(e) = self.cd_request_state.get(request_id) {
            e.value()
                .cleanup_claimed
                .store(value, std::sync::atomic::Ordering::Release);
            true
        } else {
            false
        }
    }

    /// Test-only: force-set the wrapper's per-request
    /// `pending_failure` stash. Returns `false` if no entry exists.
    /// Used to simulate the racy "cleanup_failed_request landed
    /// between decode_usaa's pending check and commit_usaa1's read"
    /// scenario without timing fragility — see the
    /// commit_usaa1-race-replay test in `tests/cd_decode_e2e.rs`.
    #[cfg(any(test, feature = "testing"))]
    pub fn force_pending_failure_for_test(&self, request_id: &str, reason: Option<String>) -> bool {
        if let Some(e) = self.cd_request_state.get(request_id) {
            *e.value().pending_failure.lock() = reason;
            true
        } else {
            false
        }
    }

    /// Test-only: enter `commit_usaa1` directly, bypassing
    /// `decode_usaa`'s outer pending_failure check. Lets tests
    /// simulate the race where a `cleanup_failed_request` stashes
    /// `pending_failure` AFTER `decode_usaa` already observed
    /// `None` — exercising the inner re-check inside commit_usaa1.
    #[cfg(any(test, feature = "testing"))]
    pub fn commit_usaa1_for_test(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        self.commit_usaa1(request_id, block_ids, num_external_tokens)
    }

    /// Test-only: snapshot the current `Arc<CdRequestState>` for
    /// `request_id`. Opaque to keep the inner type crate-private;
    /// the handle is consumed by
    /// [`Self::release_request_if_matches_for_test`] to drive the
    /// identity-checked release path.
    #[cfg(any(test, feature = "testing"))]
    pub fn snapshot_cd_state_for_test(&self, request_id: &str) -> Option<CdRequestStateHandle> {
        self.cd_request_state
            .get(request_id)
            .map(|e| CdRequestStateHandle(Arc::clone(e.value())))
    }

    /// Test-only: drive [`Self::release_request_if_matches`] with a
    /// captured handle. Returns the bool the inner method returns
    /// (`true` if the captured Arc still matches the DashMap entry
    /// and the release fired; `false` otherwise).
    #[cfg(any(test, feature = "testing"))]
    pub fn release_request_if_matches_for_test(
        &self,
        request_id: &str,
        handle: &CdRequestStateHandle,
    ) -> bool {
        self.release_request_if_matches(request_id, &handle.0)
    }

    /// Test-only: peek at the `pending_promotion.source_tier`
    /// recorded for `request_id` by `commit_gnmt_remote`. Returns
    /// `None` if no promotion is planned; `Some("G1")` or
    /// `Some("G3")` for the planned tier. Used by Stage 2
    /// reproducer tests to verify the G3 path won the GNMT-time
    /// finder race.
    #[cfg(any(test, feature = "testing"))]
    pub fn pending_promotion_tier_for_test(&self, request_id: &str) -> Option<&'static str> {
        self.cd_request_state.get(request_id).and_then(|e| {
            e.value()
                .pending_promotion
                .as_ref()
                .map(|p| match p.source_tier {
                    SourceTier::G1 => "G1",
                    SourceTier::G3 => "G3",
                })
        })
    }

    fn release_request(&self, request_id: &str) {
        if let Some((_, state)) = self.cd_request_state.remove(request_id) {
            self.inflight_budget.release(state.reserved_tokens);
        }
    }

    /// Identity-checked variant of [`Self::release_request`] for
    /// callers that captured a specific `Arc<CdRequestState>`
    /// snapshot and want to release ONLY if the DashMap still
    /// holds that same Arc.
    ///
    /// Pairs with [`ConditionalDisaggCoordinator::release_if_matches`]
    /// to close the cross-lifecycle stale-release window under
    /// `kv_load_failure_policy=recompute`: a spawn-replay task
    /// parked in `mark_failed_onboarding.await` from a prior
    /// lifecycle MUST NOT wipe the budget reservation or evict the
    /// freshly-installed state of a subsequent lifecycle for the
    /// same rid.
    ///
    /// Returns `true` if the matching entry was removed and the
    /// budget was released; `false` if the captured Arc no longer
    /// matches what the DashMap holds.
    fn release_request_if_matches(&self, request_id: &str, expected: &Arc<CdRequestState>) -> bool {
        match self
            .cd_request_state
            .remove_if(request_id, |_, v| Arc::ptr_eq(expected, v))
        {
            Some((_, state)) => {
                self.inflight_budget.release(state.reserved_tokens);
                true
            }
            None => false,
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
        // Idempotent retry: vLLM may call gnmt multiple times for the
        // same request without an intervening USAA (e.g. allocation
        // failed and the scheduler re-runs gnmt). Check our CD state
        // BEFORE calling inner so we don't double-invoke inner's gnmt
        // on retries — inner is contractually idempotent but skipping
        // the call avoids unnecessary work and tightens the surface.
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

        let inner_result = self
            .inner
            .get_num_new_matched_tokens(request_id, num_computed_tokens)?;
        tracing::info!(?inner_result, "decode_gnmt: inner returned");

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
                    tracing::info!("decode_gnmt: Remote but no full block to send — passthrough");
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
                    // `commit_gnmt_remote` owns the budget lifecycle for
                    // every internal failure path: pre-insert paths
                    // release directly; post-insert paths route through
                    // `release_request` (idempotent against the
                    // payload-Drop release). No additional cleanup
                    // needed here — propagating the Err is correct.
                    tracing::error!(error = %err, "decode_gnmt: commit_gnmt_remote failed");
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
        // Budget lifecycle for this call:
        //   - decode_gnmt reserved `full_block_external_tokens` before
        //     invoking us. The reservation is OURS to release on any
        //     failure path until ownership transfers to per-request
        //     state at the `cd_request_state.insert` below.
        //   - Pre-insert failures (everything between here and the
        //     insert at the bottom of this section) MUST release the
        //     reservation explicitly via `release_pre_insert_budget`.
        //     No `cd_request_state` exists yet, so `release_request`
        //     would be a no-op and the reservation would leak.
        //   - Post-insert failures (begin_remote_prefill Err arm) route
        //     through `release_request`, which is state-guarded and
        //     idempotent against the payload-Drop path.
        let release_pre_insert_budget = |e: anyhow::Error| -> anyhow::Error {
            self.inflight_budget.release(full_block_external_tokens);
            e
        };

        let split = self
            .inner
            .slot_match_split(request_id)
            .map_err(release_pre_insert_budget)?;
        tracing::info!(
            local_match_blocks = split.local_match_blocks,
            computed_blocks = split.computed_blocks,
            total_blocks = split.total_blocks,
            "commit_gnmt_remote: slot_match_split"
        );

        let local_g2 = self
            .inner
            .take_local_match_g2_blocks(request_id)
            .map_err(release_pre_insert_budget)?;
        tracing::info!(
            local_g2_len = local_g2.len(),
            "commit_gnmt_remote: took local-match G2 blocks"
        );
        if local_g2.len() != split.local_match_blocks {
            self.inflight_budget.release(full_block_external_tokens);
            anyhow::bail!(
                "GNMT split says {} local-match blocks but find_session yielded {}",
                split.local_match_blocks,
                local_g2.len()
            );
        }

        // Prefix-range G2 backfill.
        //
        // The inner search at `process_match` covers
        // `[num_computed_tokens / bs, total / bs)` — vLLM is implicitly
        // trusted to "have the prefix in G1." The CD session contract
        // requires every block in `[0, total)` to be reachable by the
        // prefill peer via decode's G2/G3, so the prefix range needs to
        // be backfilled here so decode can publish it on the session.
        //
        // With vLLM prefix-caching disabled (current default)
        // `num_computed_tokens == 0` and `num_prefix_blocks == 0` — this
        // is a no-op. With PC enabled the prefix range is non-empty.
        //
        // Two outcomes (all-or-nothing):
        //   1. Full G2 hit — every prefix block was in G2; pin them
        //      (the session's `available_pins` owns the pin for the
        //      session's lifetime once we hand it to
        //      `begin_remote_prefill`).
        //   2. Any miss — vLLM has the full prefix in G1, but G2
        //      doesn't back all of it. `find_prefix_g2_blocks` drops
        //      any partial hits and returns empty; the empty arm
        //      below plans a G1→G2 promotion task that fires at
        //      USAA, when vLLM has handed us the actual G1
        //      `block_ids`. The session's commit set includes the
        //      planned-promoted hashes from the start (see
        //      `begin_remote_prefill` — `pending_promotion_hashes`),
        //      so `finish_commits` seals the full planned window up
        //      front and the prefill peer never observes a partial
        //      commit set. `session.finish_availability` is deferred
        //      until the promotion task calls
        //      `session.make_available` on the landed G2 blocks.
        //
        // Why prefix and local-match are passed to `begin_remote_prefill`
        // as SEPARATE Vecs (not concatenated) — two invariants:
        //
        // 1. CD USAA-1 local-match contract: `local_match_g2_pins` /
        //    `local_match_g2_block_ids` must contain exactly
        //    `split.local_match_blocks` entries, paired 1:1 with the
        //    G1 destinations sliced by `split.local_match_range()` for
        //    the local-kick `transport.local_g2_to_g1` call. We hand
        //    `local_g2` (the original N) into `local_match_g2_pins`
        //    and do NOT mix prefix blocks in.
        //
        // 2. `RemotePrefillRequest` positional contract: prefill places
        //    `params.sequence_hashes[i]` at absolute block index
        //    `decode_offset_blocks + i` (see `ensure_started`'s
        //    `expected_hashes[i] lands at D/BS + i` comment). The
        //    request must therefore include ONLY local-match hashes;
        //    prefix hashes cover `[0, num_computed_tokens / bs)` and
        //    would mis-place by P positions if folded in.
        //
        // The coordinator (`begin_remote_prefill`) routes them
        // accordingly: prefix + local → `session.commit` /
        // `session.make_available`; local-match only →
        // `RemotePrefillRequest.sequence_hashes`. See the doc-comment
        // on `begin_remote_prefill` for the publish-without-consumer
        // discussion (prefix hashes are pullable by the prefill peer
        // but no current prefill code path queries them; they drop on
        // session close).
        let block_size = self.inner.block_size();
        let num_prefix_blocks = inputs.num_computed_tokens / block_size;
        let prefix_g2 = self
            .inner
            .find_prefix_g2_blocks(request_id, num_prefix_blocks)
            .map_err(release_pre_insert_budget)?;
        if !prefix_g2.is_empty() {
            tracing::info!(
                prefix_g2_len = prefix_g2.len(),
                num_prefix_blocks,
                "commit_gnmt_remote: backfilled prefix G2 blocks (session-only)"
            );
            crate::audit!(
                "prefix_g2_backfill",
                role = "decode",
                request_id,
                prefix_g2_len = prefix_g2.len(),
                num_prefix_blocks
            );
        }

        // Stage 1/2: when G2 didn't cover the prefix but vLLM is
        // claiming one (`num_computed_tokens > 0`), plan a
        // promotion to fire at USAA. Try G3 first (decode's NVMe
        // tier) since promoting from G3 doesn't need vLLM's G1
        // `block_ids` and avoids contending for the G1 offload
        // pipeline; fall back to G1 (Stage 1 path) if G3 doesn't
        // hold the prefix.
        //
        // The slot's full sequence is the source of truth for the
        // prefix hashes; the G3 finder is contract-bound to return
        // the same canonical chain on hit (verified by debug
        // assert).
        let pending_promotion = if prefix_g2.is_empty() && num_prefix_blocks > 0 {
            let prefix_hashes_from_slot: Vec<SequenceHash> =
                split.all_sequence_hashes[..num_prefix_blocks].to_vec();
            let g3_hashes = self
                .inner
                .find_prefix_g3_hashes(request_id, num_prefix_blocks)
                .map_err(release_pre_insert_budget)?;
            let (source_tier, prefix_hashes) = if !g3_hashes.is_empty() {
                debug_assert_eq!(
                    g3_hashes, prefix_hashes_from_slot,
                    "find_prefix_g3_hashes returned hashes inconsistent with slot PLH chain"
                );
                crate::audit!(
                    "prefix_g3_to_g2_promotion_planned",
                    role = "decode",
                    request_id,
                    prefix_block_count = num_prefix_blocks
                );
                (SourceTier::G3, g3_hashes)
            } else {
                crate::audit!(
                    "prefix_g1_to_g2_promotion_planned",
                    role = "decode",
                    request_id,
                    prefix_block_count = num_prefix_blocks
                );
                (SourceTier::G1, prefix_hashes_from_slot)
            };
            Some(PendingTierPromotion {
                source_tier,
                prefix_block_count: num_prefix_blocks,
                prefix_hashes,
            })
        } else {
            None
        };

        let num_local_match_hashes = local_g2.len();
        let local_match_g2_block_ids: Vec<BlockId> =
            local_g2.iter().map(|b| b.block_id()).collect();
        // Independent pin set for the SESSION (not the wrapper). The wrapper's
        // separate `local_match_g2_pins` (filled below from `local_g2`) keeps
        // the local-match subset pinned for the local-kick. Prefix blocks have
        // no wrapper-side pin — `session.make_available` moves them into
        // `available_pins` and the session owns the pin for its lifetime.
        //
        // `session_local_g2` here carries ONLY local-match; the coordinator
        // composes prefix + local-match internally for session.commit +
        // session.make_available, but builds RemotePrefillRequest.sequence_hashes
        // from local-match only.
        let session_local_g2: Vec<ImmutableBlock<G2>> = local_g2.to_vec();
        let session_prefix_g2: Vec<ImmutableBlock<G2>> = prefix_g2;

        // Send the FULL prompt tokens on the wire (not the suffix
        // starting at `num_computed_tokens`). Prefill's slot must
        // hash the same TokenBlockSequence decode hashed so its
        // PositionalLineageHash chain matches decode's at absolute
        // positions — windowing the tokens would reset prefill's
        // chain to relative position 0 and silently break
        // cross-instance pull keying for PC-on requests.
        // `num_computed_tokens` rides separately on the wire so
        // prefill's vLLM knows to skip recomputing the prefix portion;
        // the connector pulls the prefix-G2 hashes decode published.
        let all_token_ids = self
            .inner
            .slot_token_ids(request_id)
            .map_err(release_pre_insert_budget)?;
        let base_offset = inputs.num_computed_tokens;
        let prefill_window_end = base_offset + full_block_external_tokens;
        if prefill_window_end > all_token_ids.len() {
            self.inflight_budget.release(full_block_external_tokens);
            anyhow::bail!(
                "prefill window [{}..{}] out of bounds for {} tokens",
                base_offset,
                prefill_window_end,
                all_token_ids.len(),
            );
        }
        // Reserve only the full-block tail still in `all_token_ids`
        // (everything before `prefill_window_end`). The partial tail
        // block, if any, stays on decode.
        let prefill_token_ids: Vec<u32> = all_token_ids[..prefill_window_end].to_vec();

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
            // Attached after `begin_remote_prefill` returns Ok below
            // (the session doesn't exist yet at this point). The
            // Ok-arm assignment closes the CD USAA-1 race; see the
            // doc-comment on `CdRequestState::session`.
            session: Mutex::new(None),
            pending_failure: Mutex::new(None),
            cleanup_claimed: std::sync::atomic::AtomicBool::new(false),
            pending_promotion: pending_promotion.clone(),
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
        // Closure invoked synchronously inside begin_remote_prefill,
        // BEFORE the enqueue task is spawned — closes the race where
        // a fast queue failure could clean up coordinator state before
        // the wrapper installed its slot RAII payload.
        let coordinator_weak: Weak<ConditionalDisaggCoordinator> =
            Arc::downgrade(&self.coordinator);
        let inner_for_install = Arc::clone(&self.inner);
        let weak_self_for_install = self.weak_self.clone();
        let install_payload = move |rid: &str| -> Result<()> {
            let payload = Box::new(CdRequestStatePayload {
                request_id: rid.to_string(),
                reserved_tokens: full_block_external_tokens,
                wrapper: weak_self_for_install,
                coordinator: coordinator_weak,
            });
            inner_for_install.install_cd_onboarding_payload(rid, payload)
        };
        // Past the cd_request_state.insert: failures here clean up via
        // release_request (removes state + releases budget; idempotent
        // against the payload-Drop path that runs inside
        // begin_remote_prefill on install failure).
        let lora_name = self.inner.slot_lora_name(request_id).inspect_err(|_| {
            self.release_request(request_id);
        })?;
        let salt = self.inner.slot_salt(request_id).inspect_err(|_| {
            self.release_request(request_id);
        })?;
        let pending_promotion_hashes_for_session = pending_promotion
            .as_ref()
            .map(|p| p.prefix_hashes.clone())
            .unwrap_or_default();
        match self.coordinator.begin_remote_prefill(
            RemotePrefillStart {
                request_id,
                inputs,
                initiator_instance_id: initiator,
                prefix_g2: session_prefix_g2,
                local_match_g2: session_local_g2,
                prefill_token_ids,
                lora_name,
                salt,
                pending_promotion_hashes: pending_promotion_hashes_for_session,
            },
            install_payload,
        ) {
            Ok(outcome) => {
                // Attach the session to the wrapper's per-request
                // state so `commit_usaa1` can hand it to the remote
                // pipeline spawn without re-querying the coordinator.
                // Closes the CD USAA-1 race against
                // `coordinator.release` from a sibling cleanup. See
                // the doc-comment on `CdRequestState::session`.
                *new_state.session.lock() = Some(Arc::clone(&outcome.session));
                tracing::info!(
                    session_id = %outcome.session_id,
                    "commit_gnmt_remote: begin_remote_prefill ok"
                );
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
                // Two cases collapse into one cleanup:
                //   - Payload-install failed inside begin_remote_prefill:
                //     the payload's Drop ran wrapper.release_request
                //     (removed cd_request_state + released budget).
                //     release_request here is a state-guarded no-op.
                //   - Pre-payload bailout (e.g. "called twice" at
                //     driver.rs:1085): cd_request_state still has the
                //     entry, no Drop fired. release_request removes
                //     it and releases budget.
                self.release_request(request_id);
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

        // Replay any pre-USAA failure stash: if `cleanup_failed_request`
        // ran before USAA installed G1 ids, it stashed the reason
        // instead of emitting `mark_failed_onboarding(rid, [])` (which
        // vLLM treats as success). USAA is the first point we have
        // real G1 destinations to report — emit them as failed and
        // tear down without continuing the USAA bookkeeping.
        let existing_wrapper = self
            .cd_request_state
            .get(request_id)
            .map(|e| Arc::clone(e.value()));
        let pending = existing_wrapper
            .as_ref()
            .and_then(|s| s.pending_failure.lock().clone());
        if let Some(reason) = pending {
            // Only the EXTERNAL slice should be reported failed.
            // vLLM truncates `request.num_computed_tokens` at the
            // first invalid block; reporting the entire `block_ids`
            // (including the already-computed prefix) would force
            // recomputation from token 0. The external load covers
            // exactly `num_external_tokens / block_size` blocks at
            // the tail of `block_ids`.
            let block_size = self.inner.block_size();
            let external_blocks = if block_size > 0 {
                num_external_tokens / block_size
            } else {
                0
            };
            let external_slice_start = block_ids.len().saturating_sub(external_blocks);
            let external_g1_ids: Vec<BlockId> = block_ids[external_slice_start..].to_vec();
            crate::audit!(
                "usaa_replay_pending_failure",
                role = "decode",
                request_id,
                reason = %reason,
                num_external_g1_ids = external_g1_ids.len(),
                num_total_g1_ids = block_ids.len()
            );
            tracing::warn!(
                request_id,
                reason = %reason,
                num_external_g1_ids = external_g1_ids.len(),
                num_total_g1_ids = block_ids.len(),
                "decode_usaa: replaying pre-USAA failure with external G1 slice"
            );
            // Spawn the worker notification (async) and the wrapper-
            // side cleanup. Returning Ok lets vLLM's USAA bookkeeping
            // complete; the failure surfaces via finished_recving with
            // the failed_block_ids in the same forward pass.
            //
            // Capture both per-lifecycle Arcs BEFORE the spawn so the
            // identity-checked release inside the task no-ops if a
            // subsequent lifecycle (recompute reschedule of the same
            // rid) has already replaced the DashMap entries. See
            // `release_request_if_matches` /
            // `ConditionalDisaggCoordinator::release_if_matches`.
            let captured_wrapper =
                existing_wrapper.expect("pending was Some, so the wrapper Arc must exist");
            let captured_coord = self.coordinator.state_for_decode(request_id);
            let weak_self = self.weak_self.clone();
            let request_id_owned = request_id.to_string();
            self.tokio_handle.spawn(async move {
                if let Some(this) = weak_self.upgrade() {
                    if let Err(err) = this
                        .worker_hook
                        .mark_failed_onboarding(request_id_owned.clone(), external_g1_ids)
                        .await
                    {
                        tracing::error!(
                            request_id = request_id_owned,
                            error = %err,
                            "mark_failed_onboarding failed during USAA replay"
                        );
                    }
                    this.release_request_if_matches(&request_id_owned, &captured_wrapper);
                    if let Some(coord) = captured_coord {
                        this.coordinator
                            .release_if_matches(&request_id_owned, &coord);
                    }
                }
            });
            return Ok(());
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
        // Stage 1: capture the prefix slice now (before
        // `apply_block_assignments` consumes `block_ids`). The
        // promotion task built at the bottom of `commit_usaa1`
        // pairs these G1 ids with the GNMT-time `prefix_hashes`.
        let prefix_g1_block_ids: Vec<BlockId> = block_ids[..split.computed_blocks].to_vec();
        let expected_remote_hashes = split.expected_remote_hashes();
        let remote_range = split.remote_range();

        // Resolve the wrapper's gnmt-time state BEFORE any state
        // mutation (block assignments, pin drain, remote_slots
        // rebuild). The pending_failure re-check below short-circuits
        // out without mutating anything if a concurrent cleanup
        // stashed a failure between decode_usaa's check and ours.
        //
        // CD USAA-1 race (read side): under
        // `kv_load_failure_policy=recompute`, a sibling
        // `cleanup_failed_request` on a previous run of this same
        // request id can call `coordinator.release` (Bug B's atomic-
        // remove site, coordinator.rs:1045) between the `is_active`
        // check at the top of `decode_usaa` and the lookup here,
        // dropping the wrapper-side state. The cleanup chain has
        // already run — there's nothing for us to do, and bailing
        // would propagate worker-fatal up to vLLM's EngineCore for
        // a race vLLM already owns recovery for.
        //
        // Mirrors the soft-skip pattern at coordinator.rs:1045 and
        // coordinator.rs:309-313. See `feedback-bail-vs-softskip-callbacks`
        // for the Tier-2 ranking under Graham King's review skill.
        let existing = match self
            .cd_request_state
            .get(request_id)
            .map(|e| Arc::clone(e.value()))
        {
            Some(s) => s,
            None => {
                crate::audit!(
                    "commit_usaa1_state_gone",
                    role = "decode",
                    request_id,
                    reason = "cd_state_removed_between_is_active_and_commit_usaa1"
                );
                return Ok(());
            }
        };

        // Race re-check: `decode_usaa`'s pending_failure check happens
        // BEFORE entering `commit_usaa1`. A concurrent
        // `cleanup_failed_request` firing between that check and here
        // stashes pending_failure=Some on `existing` (pre-USAA branch
        // does NOT release). Without this re-check, commit_usaa1
        // would call apply_block_assignments, drain
        // local_match_g2_pins, build remote_slots, and spawn the
        // local-kick + remote-pipeline. Those tasks could complete
        // successfully, fire mark_onboarding_complete, and report
        // the failed request to vLLM as a SUCCESS — the stash would
        // never reach mark_failed_onboarding. Take the same replay
        // path `decode_usaa` uses for pre-USAA stashes, BEFORE any
        // mutation.
        let pending = existing.pending_failure.lock().clone();
        if let Some(reason) = pending {
            let external_blocks = if block_size > 0 {
                num_external_tokens / block_size
            } else {
                0
            };
            let external_slice_start = block_ids.len().saturating_sub(external_blocks);
            let external_g1_ids: Vec<BlockId> = block_ids[external_slice_start..].to_vec();
            crate::audit!(
                "commit_usaa1_replay_pending_failure",
                role = "decode",
                request_id,
                reason = %reason,
                num_external_g1_ids = external_g1_ids.len(),
                num_total_g1_ids = block_ids.len()
            );
            tracing::warn!(
                request_id,
                reason = %reason,
                num_external_g1_ids = external_g1_ids.len(),
                num_total_g1_ids = block_ids.len(),
                "commit_usaa1: replaying pre-USAA failure stashed after decode_usaa check"
            );
            // Capture per-lifecycle Arcs so the spawn's release no-ops
            // if a recompute reschedule replaces the entries before
            // `mark_failed_onboarding.await` returns. See
            // `release_request_if_matches`.
            let captured_wrapper = Arc::clone(&existing);
            let captured_coord = self.coordinator.state_for_decode(request_id);
            let weak_self = self.weak_self.clone();
            let request_id_owned = request_id.to_string();
            self.tokio_handle.spawn(async move {
                if let Some(this) = weak_self.upgrade() {
                    if let Err(err) = this
                        .worker_hook
                        .mark_failed_onboarding(request_id_owned.clone(), external_g1_ids)
                        .await
                    {
                        tracing::error!(
                            request_id = request_id_owned,
                            error = %err,
                            "mark_failed_onboarding failed during commit_usaa1 replay"
                        );
                    }
                    this.release_request_if_matches(&request_id_owned, &captured_wrapper);
                    if let Some(coord) = captured_coord {
                        this.coordinator
                            .release_if_matches(&request_id_owned, &coord);
                    }
                }
            });
            return Ok(());
        }

        self.inner.apply_block_assignments(request_id, block_ids)?;

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
            // Carry over the gnmt-time session so the remote-pipeline
            // spawn below uses the held Arc instead of querying the
            // coordinator. Closes the CD USAA-1 race; see the
            // doc-comment on `CdRequestState::session`.
            session: Mutex::new(existing.session.lock().clone()),
            // The new state ALWAYS starts with `pending_failure=None`
            // and `cleanup_claimed=false`. Threading either field
            // forward from `existing` re-opens the rebuild race:
            //
            // - If the new state inherits `pending_failure=Some` from
            //   a stash captured during rebuild, no downstream code
            //   surfaces it. The outer pending_failure re-check has
            //   already passed (it ran against `existing` before the
            //   rebuild started); any stash that lands during rebuild
            //   is missed unless the post-insert check below catches
            //   it on `existing`.
            // - If the new state inherits `cleanup_claimed=true`, no
            //   future `cleanup_failed_request` can pass the CAS to
            //   emit `mark_failed_onboarding` with the now-known G1
            //   ids — vLLM is never notified.
            //
            // The post-insert re-check (below `cd_request_state.insert`)
            // is the canonical surface for late-arriving stashes; it
            // queries `existing.pending_failure` (the OLD Arc which
            // commit_usaa1 still holds) so any cleanup that stashed
            // between the outer re-check and the insert is caught.
            //
            // Starting fresh is safe: a fully-completed cleanup would
            // have called `release_request`, removing the
            // `cd_request_state` entry; then commit_usaa1's `get` of
            // `existing` would fail and we wouldn't reach here.
            pending_failure: Mutex::new(None),
            cleanup_claimed: std::sync::atomic::AtomicBool::new(false),
            // Carry the GNMT-time promotion plan forward — the task
            // is spawned BELOW (after the post-insert pending_failure
            // re-check). Threading it through `updated` keeps cleanup
            // paths observing a single source of truth via the latest
            // per-request Arc.
            pending_promotion: existing.pending_promotion.clone(),
        });
        self.cd_request_state
            .insert(request_id.to_string(), Arc::clone(&updated));

        // Post-insert pending_failure re-check. A
        // `cleanup_failed_request` that fired between the outer
        // pending_failure re-check (above the rebuild) and here would
        // have:
        //   - Seen `existing` (OLD state) in `cd_request_state.get`
        //     because the insert above had not yet replaced the entry.
        //   - CAS-claimed `existing.cleanup_claimed` (no contention
        //     since we don't claim it ourselves).
        //   - Computed `unfilled_g1_block_ids` from OLD state — empty,
        //     since OLD has empty `local_match_g1_block_ids` and
        //     empty `remote_slots`.
        //   - Taken the pre-USAA branch: stashed `pending_failure` on
        //     OLD state and returned without releasing.
        //
        // OLD state is now unreachable via `cd_request_state.get`
        // (replaced by `updated`), but commit_usaa1 still holds the
        // OLD Arc via `existing`. Reading `existing.pending_failure`
        // here observes the stash and lets us replay before spawning
        // the local-kick + remote-pipeline, which would otherwise be
        // able to complete successfully and report SUCCESS to vLLM
        // for a request that was supposed to be failed.
        //
        // Cleanups that fire AFTER the insert see `updated` (NEW
        // state) which has populated `remote_slots`, so they take
        // the post-USAA branch and emit `mark_failed_onboarding`
        // directly — no stash, no race.
        let late_stash = existing.pending_failure.lock().clone();
        if let Some(reason) = late_stash {
            let external_g1_ids = updated.unfilled_g1_block_ids();
            crate::audit!(
                "commit_usaa1_post_insert_replay",
                role = "decode",
                request_id,
                reason = %reason,
                num_external_g1_ids = external_g1_ids.len()
            );
            tracing::warn!(
                request_id,
                reason = %reason,
                num_external_g1_ids = external_g1_ids.len(),
                "commit_usaa1: replaying stash that landed between outer re-check and insert"
            );
            // Capture the NEW (`updated`) wrapper Arc — that's the
            // one the DashMap currently holds at this point in
            // `commit_usaa1`. OLD (`existing`) is no longer
            // reachable via the map (the insert above replaced it).
            // Capturing OLD would make the spawn's release no-op
            // immediately on a single-lifecycle run. Capturing NEW
            // makes it no-op only if a SUBSEQUENT lifecycle
            // (recompute reschedule) further replaces the entry —
            // the cross-lifecycle stale-release semantics we want.
            let captured_wrapper = Arc::clone(&updated);
            let captured_coord = self.coordinator.state_for_decode(request_id);
            let weak_self = self.weak_self.clone();
            let request_id_owned = request_id.to_string();
            self.tokio_handle.spawn(async move {
                if let Some(this) = weak_self.upgrade() {
                    if let Err(err) = this
                        .worker_hook
                        .mark_failed_onboarding(request_id_owned.clone(), external_g1_ids)
                        .await
                    {
                        tracing::error!(
                            request_id = request_id_owned,
                            error = %err,
                            "mark_failed_onboarding failed during post-insert replay"
                        );
                    }
                    this.release_request_if_matches(&request_id_owned, &captured_wrapper);
                    if let Some(coord) = captured_coord {
                        this.coordinator
                            .release_if_matches(&request_id_owned, &coord);
                    }
                }
            });
            return Ok(());
        }

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
            // Read the session from the wrapper's per-request state
            // (set synchronously in `commit_gnmt_remote`'s Ok arm).
            // Closes the CD USAA-1 race against `coordinator.release`
            // from a sibling cleanup, which Bug B's atomic-remove
            // (coordinator.rs:1045) closed on the write side.
            //
            // Invariant: by the time `commit_usaa1` runs, the session
            // was set under `commit_gnmt_remote`'s Ok arm. The None
            // branch is unreachable on the documented orderings, but
            // we route to `cleanup_failed_request` (the documented
            // async-failure path) rather than `bail!` so a future
            // unusual ordering surfaces as a graceful per-request
            // failure to vLLM instead of a worker-fatal up the
            // EngineCore stack. Matches Graham's Rule 13 ("graceful
            // spawned-task error handling"). See
            // `feedback-bail-vs-softskip-callbacks`.
            let wrapper = self
                .arc_self()
                .ok_or_else(|| anyhow!("wrapper Arc unavailable in commit_usaa1"))?;
            let session = match updated.session.lock().clone() {
                Some(s) => s,
                None => {
                    crate::audit!(
                        "commit_usaa1_session_missing",
                        role = "decode",
                        request_id,
                        reason = "wrapper_state_session_unset_post_gnmt"
                    );
                    let request_id_owned = request_id.to_string();
                    let wrapper_clone = Arc::clone(&wrapper);
                    self.tokio_handle.spawn(async move {
                        wrapper_clone
                            .cleanup_failed_request(
                                &request_id_owned,
                                "CD USAA-1: wrapper-side session missing post-gnmt".to_string(),
                            )
                            .await;
                    });
                    return Ok(());
                }
            };
            let request_id_owned = request_id.to_string();
            let state_clone = Arc::clone(&updated);
            self.tokio_handle.spawn(async move {
                if let Err(err) = wrapper
                    .run_remote_pipeline(&request_id_owned, state_clone, session)
                    .await
                {
                    wrapper
                        .cleanup_failed_request(&request_id_owned, format!("{:#}", err))
                        .await;
                }
            });
        }

        // Stage 1: spawn the uncancellable G1→G2 prefix promotion
        // task. Independent of the local-kick and remote-pipeline
        // spawns above — it only drives the session's deferred
        // availability close (`begin_remote_prefill` skipped
        // `session.finish_availability` because we promised more
        // blocks would land). The task survives request teardown:
        // dropping its `JoinHandle` detaches but does not abort.
        // Even if the session has been closed by the time it
        // completes, the resulting G2 blocks remain registered in
        // the cache and benefit future requests.
        if let Some(plan) = updated.pending_promotion.clone() {
            // Lookup the coordinator-side state FIRST so we can
            // skip the spawn entirely when cleanup has already
            // evicted state — a non-gated promotion task would
            // race `coordinator.release`'s finalize (no handle to
            // await means release would proceed unconditionally).
            // If state is gone, the session is already closing
            // (cleanup_failed_request → session.close); the
            // promotion is moot.
            let coord_state = match self.coordinator.state_for_decode(request_id) {
                Some(s) => s,
                None => {
                    tracing::warn!(
                        request_id,
                        "coord state missing at promotion-spawn; cleanup ran first — \
                         skipping Stage 1 promotion"
                    );
                    return Ok(());
                }
            };
            let bits = match coord_state.as_decode() {
                Some(b) => b,
                None => {
                    tracing::error!(
                        request_id,
                        "promotion-spawn: coordinator state is not decode-role; \
                         skipping Stage 1 promotion"
                    );
                    return Ok(());
                }
            };

            // Acquire the slot's Mutex IMMEDIATELY after the lookup
            // and hold it across all subsequent work (build source
            // blocks, call `promote_g1_to_g2`, build task closure,
            // spawn, store handle, release lock). This closes any
            // window between "coordinator state is alive with no
            // pending task" and "pending task is stored" — a
            // concurrent observer of `DecodeBits.pending_promotion_task`
            // (e.g., `coordinator.release` via
            // `CdRequestStatePayload::Drop` on another thread) sees
            // either None-before-promotion or Some(handle), never an
            // in-between None state where the gate is missed but a
            // task is actually in flight.
            //
            // All operations under the lock are sync (no awaits):
            // `promote_g1_to_g2` returns a BoxFuture without
            // side-effecting the session, the closure is built
            // without polling it, `tokio_handle.spawn` is sync.
            // Holding parking_lot across them is safe — the task
            // body never touches this lock.
            let mut slot_guard = bits.pending_promotion_task.lock();

            let inner = Arc::clone(&self.inner);
            let session = updated.session.lock().clone();
            let request_id_owned = request_id.to_string();
            let prefix_block_count = plan.prefix_block_count;
            let prefix_hashes = plan.prefix_hashes.clone();
            // Tier-dispatched promotion future construction. Both
            // arms build the same `BoxFuture<Result<Vec<ImmutableBlock<G2>>>>`
            // shape; the task body below is identical and operates
            // on the resulting G2 blocks regardless of source tier.
            //
            // - G1: pair vLLM's prefix block_ids with the GNMT-time
            //   prefix hashes into `ExternalBlock<G1>` entries. The
            //   offload pipeline registers each resulting G2 block
            //   with the carried sequence_hash.
            // - G3: pass the prefix hashes directly. The shim
            //   re-matches G3 internally and stages via
            //   `kvbm_engine::leader::stage_g3_to_g2`; no vLLM
            //   block_ids are needed (the source blocks live in
            //   the connector's own G3 manager).
            let (tier_label, promotion_fut) = match plan.source_tier {
                SourceTier::G1 => {
                    debug_assert_eq!(prefix_g1_block_ids.len(), prefix_block_count);
                    let source_blocks: Vec<kvbm_engine::offload::ExternalBlock<crate::G1>> =
                        prefix_g1_block_ids
                            .iter()
                            .copied()
                            .zip(prefix_hashes.iter().copied())
                            .map(|(bid, h)| {
                                kvbm_engine::offload::ExternalBlock::<crate::G1>::new(bid, h)
                            })
                            .collect();
                    crate::audit!(
                        "prefix_g1_to_g2_promotion_enqueued",
                        role = "decode",
                        request_id,
                        prefix_block_count
                    );
                    ("g1->g2", inner.promote_g1_to_g2(source_blocks))
                }
                SourceTier::G3 => {
                    crate::audit!(
                        "prefix_g3_to_g2_promotion_enqueued",
                        role = "decode",
                        request_id,
                        prefix_block_count
                    );
                    ("g3->g2", inner.promote_g3_to_g2(prefix_hashes.clone()))
                }
            };
            // Task returns `bool` consumed by `coordinator.release`'s
            // deferred-finalize path:
            //
            //   `true`  — promotion landed cleanly; release MUST call
            //             `session.finalize` to close streams.
            //   `false` — task already called `session.close(reason)`;
            //             release MUST skip finalize (`finalize` on
            //             a closed session is undefined protocol
            //             behavior — emits Frame::Finished after the
            //             close terminators).
            //
            // See `DecodeBits.pending_promotion_task` doc.
            let promotion_task = async move {
                let Some(session) = session else {
                    tracing::warn!(
                        request_id = %request_id_owned,
                        "promotion task: session unset on per-request state; dropping"
                    );
                    // No session to finalize; return true is harmless
                    // (the deferred-release path has no session either).
                    return true;
                };
                match promotion_fut.await {
                    Ok(g2_blocks) => {
                        crate::audit!(
                            "prefix_g2_promotion_landed",
                            role = "decode",
                            request_id = %request_id_owned,
                            promoted = g2_blocks.len()
                        );
                        // `prefix_hashes` were committed up-front in
                        // `begin_remote_prefill`, and `finish_commits`
                        // sealed the committed set there too — calling
                        // `session.commit` here would error per
                        // CONTRACT §2.3 (commit after finish_commits).
                        // Drop straight to `make_available`.
                        let _ = &prefix_hashes;
                        if let Err(err) = session.make_available(g2_blocks) {
                            tracing::warn!(
                                request_id = %request_id_owned,
                                error = %err,
                                "promotion task: session.make_available failed; closing session"
                            );
                            session.close(Some(format!("make_available: {err}")));
                            return false;
                        }
                        if let Err(err) = session.finish_availability() {
                            tracing::warn!(
                                request_id = %request_id_owned,
                                error = %err,
                                "promotion task: session.finish_availability failed; closing"
                            );
                            session.close(Some(format!("finish_availability: {err}")));
                            return false;
                        }
                        // Success — release should finalize cooperatively.
                        true
                    }
                    Err(err) => {
                        crate::audit!(
                            "prefix_g2_promotion_failed",
                            role = "decode",
                            request_id = %request_id_owned,
                            source_tier = tier_label,
                            error = %err
                        );
                        // session.close implies finish_commits +
                        // finish_availability with `closed_reason`
                        // set, so the prefill peer observes a
                        // terminal lifecycle event and bails.
                        session.close(Some(format!("{tier_label} promotion: {err}")));
                        false
                    }
                }
            };
            // `slot_guard` was acquired at the top of this block,
            // before any of the lookup/build/spawn work — see the
            // comment there. Store and drop now.
            let handle = self.tokio_handle.spawn(promotion_task);
            *slot_guard = Some(handle);
            drop(slot_guard);
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
        // Per-request lifecycle-driven bail. The coordinator's
        // CdRequest holds the canonical token; we grab a clone here
        // so the session awaits below race against velo's
        // Detached/Failed via the watcher's `cancel.cancel()`. If
        // the coordinator already evicted state (watcher fired
        // before this task started running), bail immediately —
        // there's nothing left to pipeline against.
        let cancel = self.coordinator.cancel_for(request_id).ok_or_else(|| {
            anyhow::anyhow!(
                "run_remote_pipeline: coordinator state already evicted for {}",
                request_id
            )
        })?;

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
        loop {
            let next = tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    // Tie tolerance — biased select would spuriously
                    // fail when cancel + a normal terminator
                    // (CommitsClosed, or sufficient Added) land in
                    // the same poll cycle. Synchronously poll once
                    // more; bail only if nothing is buffered.
                    match commits.next().now_or_never() {
                        Some(d) => d,
                        None => {
                            anyhow::bail!(
                                "run_remote_pipeline: session cancelled mid-commits-drain \
                                 (seen {}/{}) for {}",
                                commit_seen.len(),
                                expected_count,
                                request_id
                            );
                        }
                    }
                }
                d = commits.next() => d,
            };
            let Some(d) = next else { break };
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

        // 2. Drain availability and pull each chunk. Same cancel
        //    discipline as the commits drain plus a select around
        //    the per-chunk pull — velo's `pull` awaits a
        //    `Frame::PullComplete` from the peer and would
        //    otherwise pin this task on a hung peer even though
        //    velo already surfaced Detached/Failed.
        let mut filled: HashSet<SequenceHash> = HashSet::new();
        let mut avail = session.availability();
        loop {
            let next = tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    // Tie tolerance — see commits-drain comment above.
                    match avail.next().now_or_never() {
                        Some(d) => d,
                        None => {
                            anyhow::bail!(
                                "run_remote_pipeline: session cancelled mid-availability-drain \
                                 ({} of {} filled) for {}",
                                filled.len(),
                                state.remote_slots.len(),
                                request_id
                            );
                        }
                    }
                }
                d = avail.next() => d,
            };
            let Some(d) = next else { break };
            match d {
                AvailabilityDelta::Available(blocks) => {
                    // Validate every incoming hash is expected. Any
                    // unexpected hash is a protocol violation — return
                    // an error so the caller's `?` routes to
                    // `cleanup_failed_request`, surfacing the failure to
                    // vLLM instead of hanging the request via a panic in
                    // a spawned task.
                    for b in &blocks {
                        if !state.remote_slot_index.contains_key(&b.hash) {
                            anyhow::bail!(
                                "availability carried hash {:?} not in expected_remote_hashes for {}",
                                b.hash,
                                request_id
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
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => {
                            anyhow::bail!(
                                "run_remote_pipeline: session cancelled mid-pull \
                                 ({} of {} filled) for {}",
                                filled.len(),
                                state.remote_slots.len(),
                                request_id
                            );
                        }
                        r = self.pull_register_onboard_chunk(
                            request_id,
                            &state,
                            chunk_hashes.clone(),
                            Arc::clone(&session),
                        ) => {
                            r?;
                        }
                    }
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

    /// Pull, register, and onboard a set of remote hashes — possibly
    /// non-contiguous in slot-index space. Splits the input into
    /// maximal contiguous runs (by slot index) and processes each run
    /// via [`Self::pull_register_onboard_contiguous_chunk`]. The
    /// session API does not guarantee that a single
    /// `AvailabilityDelta::Available(blocks)` covers a contiguous slot
    /// range — sparse or coalesced deltas are valid shapes.
    async fn pull_register_onboard_chunk(
        self: &Arc<Self>,
        request_id: &str,
        state: &Arc<CdRequestState>,
        hashes: Vec<SequenceHash>,
        session: Arc<dyn Session>,
    ) -> Result<()> {
        // Reorder hashes by their slot position; slot order is required
        // by `MutableBlock::complete` per contiguous run.
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

        // Group into maximal contiguous runs (slot_idx[i+1] ==
        // slot_idx[i] + 1). Each run is one onboard transaction.
        let mut runs: Vec<Vec<(usize, SequenceHash)>> = Vec::new();
        for entry in indexed {
            match runs.last_mut() {
                Some(run) if run.last().unwrap().0 + 1 == entry.0 => run.push(entry),
                _ => runs.push(vec![entry]),
            }
        }

        for run in runs {
            self.pull_register_onboard_contiguous_chunk(
                request_id,
                state,
                run,
                Arc::clone(&session),
            )
            .await?;
        }
        Ok(())
    }

    /// Inner helper that handles ONE contiguous run of slot indices.
    /// Splits + dispatch is the caller's job; see
    /// [`Self::pull_register_onboard_chunk`].
    async fn pull_register_onboard_contiguous_chunk(
        self: &Arc<Self>,
        request_id: &str,
        state: &Arc<CdRequestState>,
        indexed: Vec<(usize, SequenceHash)>,
        session: Arc<dyn Session>,
    ) -> Result<()> {
        debug_assert!(!indexed.is_empty());
        debug_assert!(
            indexed.windows(2).all(|w| w[1].0 == w[0].0 + 1),
            "pull_register_onboard_contiguous_chunk requires contiguous slot indices"
        );

        crate::audit!(
            "worker_pull_chunk_start",
            role = "decode",
            request_id,
            num_hashes = indexed.len()
        );

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
        // Transport must return exactly `chunk_size` blocks; a short or
        // long result would silently truncate the zip below and copy
        // mismatched G2/G1 blocks.
        if filled.len() != chunk_size {
            anyhow::bail!(
                "session.pull returned {} blocks, expected {} (request_id={})",
                filled.len(),
                chunk_size,
                request_id
            );
        }

        // 2. Pull token blocks for the chunk's positions to
        //    drive `MutableBlock::complete`.
        let token_range_start = chunk_sequence_indices[0];
        let token_range_end = *chunk_sequence_indices.last().unwrap() + 1;
        let token_blocks = self
            .inner
            .token_blocks_for_range(request_id, token_range_start..token_range_end)?;
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

        // 3. Register with the leader's G2 manager.
        let registered = self.inner.register_g2_blocks(completes)?;
        if registered.len() != chunk_size {
            anyhow::bail!(
                "register_g2_blocks returned {} blocks, expected {} (request_id={})",
                registered.len(),
                chunk_size,
                request_id
            );
        }
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
        // Capture the coordinator-side Arc BEFORE the
        // `mark_onboarding_complete` await below so the
        // bottom-of-method `coordinator.release` is identity-
        // checked. Under `kv_load_failure_policy=recompute` the
        // await can park unbounded while a recompute reschedule
        // installs a new lifecycle for the same rid — by-name
        // release would wipe the new lifecycle. See
        // `lib/kvbm-connector/CONTRACT.md` §"Cross-lifecycle
        // stale-release race".
        let captured_coord = self.coordinator.state_for_decode(request_id);

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

        crate::audit!("mark_onboarding_complete", role = "decode", request_id);
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
        // `release_request_if_matches` and
        // `coordinator.release_if_matches` are no-ops if the
        // captured Arc no longer matches the DashMap entry (the
        // recompute cross-lifecycle case), and they're also no-ops
        // if the entry is already absent — so duplicate Drop is
        // safe.
        self.release_request_if_matches(request_id, &state);
        if let Some(coord) = captured_coord.as_ref() {
            self.coordinator.release_if_matches(request_id, coord);
        }
    }

    async fn cleanup_failed_request(self: &Arc<Self>, request_id: &str, reason: String) {
        tracing::warn!(request_id, reason = %reason, "CD request failed; cleaning up");

        // CAS-guard against parallel cleanup: five connector-spawned
        // paths can race here for the same request_id — the lifecycle
        // watcher's failure-sink route, commit_usaa1's local-kick
        // spawn Err, commit_usaa1's remote-pipeline spawn Err,
        // commit_usaa1's session-missing branch spawn, and the
        // enqueue spawn's `mark_failed`. Whichever wins false → true
        // gathers unfilled_ids and calls `mark_failed_onboarding`;
        // losers early-return so vLLM is not double-notified for the
        // same failed G1 window. Mirrors the prefill-side guard on
        // `CdRequest.cleanup_claimed`.
        //
        // If `cd_request_state` is absent (state already evicted by a
        // prior cleanup's `release_request`), no CAS is needed —
        // there is nothing to notify or release.
        //
        // Also capture the per-lifecycle Arc once at the top so the
        // release calls at the bottom use identity-checked variants
        // (`release_request_if_matches` /
        // `coordinator.release_if_matches`). Under
        // `kv_load_failure_policy=recompute`, the `mark_failed_onboarding.await`
        // below can park unbounded while a recompute reschedule
        // replaces the DashMap entry with a new lifecycle's Arc —
        // by-name release would wipe the new lifecycle's state. See
        // `lib/kvbm-connector/CONTRACT.md` §"Cross-lifecycle
        // stale-release race".
        let captured_wrapper = match self.cd_request_state.get(request_id) {
            Some(entry) => {
                let arc = Arc::clone(entry.value());
                if arc
                    .cleanup_claimed
                    .compare_exchange(
                        false,
                        true,
                        std::sync::atomic::Ordering::AcqRel,
                        std::sync::atomic::Ordering::Relaxed,
                    )
                    .is_err()
                {
                    tracing::debug!(
                        request_id,
                        "cleanup_failed_request: cleanup already claimed; no-op"
                    );
                    return;
                }
                Some(arc)
            }
            None => None,
        };
        let captured_coord = self.coordinator.state_for_decode(request_id);

        let unfilled_ids = captured_wrapper
            .as_ref()
            .map(|s| s.unfilled_g1_block_ids())
            .unwrap_or_default();

        if unfilled_ids.is_empty() {
            // Pre-USAA failure: no G1 destinations to report yet.
            // vLLM's connector contract treats `mark_failed_onboarding(
            // rid, [])` as a successful async load (it surfaces the
            // request_id in `finished_recving` with no failed blocks),
            // which is the wrong signal here. Stash the failure on
            // cd_request_state so `decode_usaa` can replay it once
            // USAA arrives with the G1 ids. If USAA never arrives
            // (e.g. vLLM cancels the request first), the stash is
            // dropped via the slot's RAII payload — no notification
            // is needed.
            //
            // Stash on the captured per-lifecycle Arc (not a fresh
            // `cd_request_state.get`) — if a recompute reschedule
            // replaced the DashMap entry while this method was
            // running, we still want the stash to land on THIS
            // lifecycle's state (USAA for this lifecycle is the
            // consumer; the new lifecycle gets its own fresh
            // pending_failure=None).
            if let Some(state) = captured_wrapper.as_ref() {
                let mut slot = state.pending_failure.lock();
                if slot.is_none() {
                    *slot = Some(reason.clone());
                }
                // Do NOT release cd_request_state: USAA needs it.
                // Do NOT call coordinator.release here either; the
                // coordinator's session has already been closed by
                // mark_failed (caller path), and the wrapper's
                // pending_failure is the source of truth for vLLM
                // notification.
            } else {
                tracing::warn!(
                    request_id,
                    "cleanup_failed_request: no cd_request_state and no G1 ids — \
                     request unknown to wrapper, dropping"
                );
            }
            crate::audit!(
                "cleanup_pending_usaa",
                role = "decode",
                request_id,
                reason = %reason
            );
            return;
        }

        // Post-USAA failure (or partial completion): emit failed
        // block_ids so vLLM can surface the failure to the request.
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

        // Identity-checked release: if a recompute reschedule
        // replaced the DashMap entry while `mark_failed_onboarding`
        // was awaiting, these no-op against the new lifecycle's
        // state. See `release_request_if_matches` /
        // `coordinator.release_if_matches` doc-comments.
        if let Some(wrapper_arc) = captured_wrapper.as_ref() {
            self.release_request_if_matches(request_id, wrapper_arc);
        } else {
            // No prior wrapper-side state — nothing to release.
            // (Original code unconditionally called
            // `self.release_request(rid)` here; under the new
            // identity-checked variant we skip when we never
            // captured an Arc, which is a no-op equivalent.)
        }
        if let Some(coord_arc) = captured_coord.as_ref() {
            self.coordinator.release_if_matches(request_id, coord_arc);
        }
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
            "prefill-side disagg API hit an unimplemented path"
        );
        todo!("prefill-side disagg API: {op}")
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
