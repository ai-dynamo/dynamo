// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::config::RouterConfigOverride;
use super::filter::RoutingEligibility;
use super::overlap::{OverlapSignals, SelectedWorkerTierSnapshot};
use super::prefill_load::effective_prefill_tokens;
pub use crate::protocols::PotentialLoad;
use crate::protocols::{
    LocalBlockHash, RoutingConstraints, SharedCacheHits, WorkerConfigLike, WorkerId,
    WorkerWithDpRank,
};
use crate::router_hint::RouterHintRootCandidates;
use crate::scheduling::policy_queue::QueueRejection;
use crate::sequences::WorkerLoadProjection;

pub type OverloadedWorkerProvider =
    Arc<dyn Fn() -> Option<HashSet<WorkerId>> + Send + Sync + 'static>;

/// Supplies the authoritative set of workers currently available for selection.
///
/// This is an inclusion set, unlike [`OverloadedWorkerProvider`]'s exclusion
/// set. `None` means no hard-availability source is attached; `Some` is
/// authoritative, so an empty set rejects every candidate.
pub type WorkerAvailabilityProvider =
    Arc<dyn Fn() -> Option<Arc<HashSet<WorkerId>>> + Send + Sync + 'static>;

#[derive(Debug, thiserror::Error)]
pub enum WorkerSelectionPolicyError {
    #[error("worker selection policy failed: {0}")]
    Failed(String),

    #[error("scorer {scorer_index} produced a non-finite cost for candidate row {row}")]
    NonFiniteCost { scorer_index: usize, row: usize },

    #[error(
        "picker returned candidate row {row}, but the candidate table contains {candidate_count} rows"
    )]
    InvalidPickerRow { row: usize, candidate_count: usize },
}

impl WorkerSelectionPolicyError {
    pub fn failed(message: impl Into<String>) -> Self {
        Self::Failed(message.into())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierOverlapBlocks {
    #[serde(default)]
    pub device: FxHashMap<WorkerWithDpRank, usize>,
    #[serde(default)]
    pub host_pinned: FxHashMap<WorkerWithDpRank, usize>,
    #[serde(default)]
    pub disk: FxHashMap<WorkerWithDpRank, usize>,
}

/// Downstream matches must include a wildcard arm because this enum is non-exhaustive.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum KvSchedulerError {
    #[error("no endpoints available to route work")]
    NoEndpoints,

    #[error(transparent)]
    QueueRejected(#[from] QueueRejection),

    #[error("all eligible workers are overloaded")]
    AllEligibleWorkersOverloaded,

    #[error("all eligible workers were rejected by policy filters")]
    AllEligibleWorkersFiltered,

    #[error("pinned worker {worker_id} is overloaded")]
    PinnedWorkerOverloaded { worker_id: WorkerId },

    #[error("pinned worker {worker_id} is not in allowed worker set")]
    PinnedWorkerNotAllowed { worker_id: WorkerId },

    #[error("endpoint subscriber shutdown")]
    SubscriberShutdown,

    #[error("failed to book scheduler state: {0}")]
    BookingFailed(String),

    #[error("failed to initialize event publisher: {0}")]
    InitFailed(String),

    #[error(transparent)]
    WorkerSelectionPolicy(#[from] WorkerSelectionPolicyError),
}

impl KvSchedulerError {
    pub fn is_overload(&self) -> bool {
        matches!(
            self,
            Self::AllEligibleWorkersOverloaded | Self::PinnedWorkerOverloaded { .. }
        )
    }
}

#[derive(Debug)]
pub struct SchedulingResponse {
    pub best_worker: WorkerWithDpRank,
    pub effective_overlap_blocks: f64,
    pub cached_tokens: usize,
    pub selected_worker_tiers: SelectedWorkerTierSnapshot,
    pub target_cached_prefix_blocks: u32,
    pub router_hint_candidates: Option<RouterHintRootCandidates>,
    pub potential_decode_blocks: usize,
}

/// A routing decision that selected less KV overlap than another eligible worker.
///
/// This records the observable locality tradeoff without attributing its cause. The
/// final selection may differ because of worker load, routing preferences, or
/// temperature-based sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonMaxOverlapSelection {
    /// Worker selected by the scheduler's complete scoring policy.
    pub selected_worker: WorkerWithDpRank,
    /// Eligible worker with the greatest effective KV overlap.
    pub highest_overlap_worker: WorkerWithDpRank,
    /// Effective overlap available on `highest_overlap_worker`.
    pub highest_overlap_blocks: f64,
    /// Effective overlap available on `selected_worker`.
    pub selected_overlap_blocks: f64,
}

/// Callback dispatched off the scheduler actor after an admitted non-max-overlap selection is
/// delivered.
pub type NonMaxOverlapSelectionObserver =
    Arc<dyn Fn(&str, NonMaxOverlapSelection) + Send + Sync + 'static>;

impl NonMaxOverlapSelection {
    /// Return the effective overlap blocks sacrificed by the final selection.
    pub fn overlap_blocks_lost(self) -> f64 {
        self.highest_overlap_blocks - self.selected_overlap_blocks
    }
}

#[derive(Debug)]
pub struct AdvisorySchedulingResponse {
    pub response: SchedulingResponse,
    pub selected_worker_load: AdvisoryWorkerLoad,
}

#[derive(Debug, Clone, Copy)]
pub struct AdvisoryWorkerLoad {
    pub active_prefill_tokens: usize,
    pub prefill_token_capacity: usize,
    pub total_kv_blocks: Option<usize>,
}

impl AdvisoryWorkerLoad {
    pub fn prefill_load_exceeds(&self, threshold: f64) -> bool {
        self.active_prefill_tokens as f64 > threshold * self.prefill_token_capacity as f64
    }

    pub fn decode_load_exceeds(
        &self,
        potential_decode_blocks: u64,
        threshold: f64,
    ) -> Option<bool> {
        let total_kv_blocks = self.total_kv_blocks?;
        Some(potential_decode_blocks as f64 > threshold * total_kv_blocks as f64)
    }
}

#[derive(Debug, Clone)]
pub enum ScheduleMode {
    QueryOnly {
        request_id: Option<String>,
    },
    /// Tracks worker state; the caller releases it with `free`.
    Tracked {
        request_id: String,
    },
    TrackedWithLifecycle {
        request_id: String,
    },
}

impl ScheduleMode {
    pub fn from_legacy(
        request_id: Option<String>,
        update_states: bool,
    ) -> Result<Self, KvSchedulerError> {
        if !update_states {
            return Ok(Self::QueryOnly { request_id });
        }

        let Some(request_id) = request_id else {
            return Err(KvSchedulerError::BookingFailed(
                "tracked scheduling request requires a request_id".to_string(),
            ));
        };
        Ok(Self::Tracked { request_id })
    }

    pub fn request_id(&self) -> Option<&str> {
        match self {
            Self::QueryOnly { request_id } => request_id.as_deref(),
            Self::Tracked { request_id } | Self::TrackedWithLifecycle { request_id } => {
                Some(request_id)
            }
        }
    }

    pub fn is_tracked(&self) -> bool {
        matches!(
            self,
            Self::Tracked { .. } | Self::TrackedWithLifecycle { .. }
        )
    }

    pub(crate) fn lifecycle_request_id(&self) -> Option<&str> {
        match self {
            Self::TrackedWithLifecycle { request_id } => Some(request_id),
            Self::QueryOnly { .. } | Self::Tracked { .. } => None,
        }
    }

    pub fn tracked_request_id(&self) -> Option<&str> {
        match self {
            Self::QueryOnly { .. } => None,
            Self::Tracked { request_id } | Self::TrackedWithLifecycle { request_id } => {
                Some(request_id)
            }
        }
    }
}

/// The event that caused an agent request to enter worker selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerSelectionInputTrigger {
    /// A user message started the turn.
    UserMessage,
    /// A tool result continued the turn.
    ToolResult,
    /// Another event caused the request.
    Other,
}

/// KV lifecycle hints supplied with an agent request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkerSelectionKvHints {
    evict_session: bool,
}

impl WorkerSelectionKvHints {
    /// Create the KV hints passed to worker selection.
    pub fn new(evict_session: bool) -> Self {
        Self { evict_session }
    }

    /// Return whether the caller asked consumers to evict the session state.
    pub fn evict_session(&self) -> bool {
        self.evict_session
    }
}

/// Session metadata supplied to a custom worker-selection policy.
///
/// The internal request protocol supplies these values. Optional values remain
/// absent when the request does not include them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionContext {
    session_id: String,
    parent_session_id: Option<String>,
    session_final: Option<bool>,
    kv_hints: Option<WorkerSelectionKvHints>,
    input_trigger: Option<WorkerSelectionInputTrigger>,
}

impl SessionContext {
    /// Create the session metadata available to worker selection.
    pub fn new(
        session_id: String,
        parent_session_id: Option<String>,
        session_final: Option<bool>,
        kv_hints: Option<WorkerSelectionKvHints>,
        input_trigger: Option<WorkerSelectionInputTrigger>,
    ) -> Self {
        Self {
            session_id,
            parent_session_id,
            session_final,
            kv_hints,
            input_trigger,
        }
    }

    /// Return the stable reasoning or tool-session identifier.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Return the parent session identifier for a subagent request.
    pub fn parent_session_id(&self) -> Option<&str> {
        self.parent_session_id.as_deref()
    }

    /// Return the optional terminal marker for this session.
    ///
    /// `Some(true)` marks the session as final, `Some(false)` explicitly marks
    /// it as continuing, and `None` means the caller supplied no marker.
    pub fn session_final(&self) -> Option<bool> {
        self.session_final
    }

    /// Return optional KV lifecycle hints from the request.
    pub fn kv_hints(&self) -> Option<&WorkerSelectionKvHints> {
        self.kv_hints.as_ref()
    }

    /// Return the event that caused this request, when supplied.
    pub fn input_trigger(&self) -> Option<WorkerSelectionInputTrigger> {
        self.input_trigger
    }
}

/// Validated request accepted by [`LocalScheduler`](super::LocalScheduler).
pub struct ScheduleRequest {
    pub mode: ScheduleMode,
    pub token_seq: Option<Vec<SequenceHash>>,
    pub block_hashes: Option<Vec<LocalBlockHash>>,
    pub isl_tokens: usize,
    pub lora_name: Option<String>,
    pub expected_output_tokens: Option<u32>,
    /// A session-affinity preference resolved by the request host.
    ///
    /// Unlike `pinned_worker`, this target does not constrain eligibility. The
    /// built-in policy uses it when it remains eligible, while custom policies
    /// can retain or replace it.
    pub affinity_target: Option<WorkerWithDpRank>,
    pub pinned_worker: Option<WorkerWithDpRank>,
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    pub routing_constraints: RoutingConstraints,
    pub router_config_override: Option<RouterConfigOverride>,
    pub priority_jump: f64,
    pub strict_priority: u32,
    pub policy_class: Option<String>,
    pub session_context: Option<SessionContext>,
    pub overlap: OverlapSignals,
    pub router_hint_candidates: Option<RouterHintRootCandidates>,
    pub retain_router_hint_chain: bool,
    pub shared_cache_hits: Option<SharedCacheHits>,
}

/// Actor-owned admission request.
///
/// After enqueue, the caller retains only the response receiver while the
/// scheduler owns this request and its sender. Dropping the caller's selection
/// future closes that receiver, but cannot retract the request from the actor.
pub struct SchedulingRequest {
    // Request identity and payload.
    pub mode: ScheduleMode,
    pub token_seq: Option<Vec<SequenceHash>>,
    pub isl_tokens: usize,
    pub lora_name: Option<String>,
    pub expected_output_tokens: Option<u32>,

    // Routing constraints and request-level config.
    pub affinity_target: Option<WorkerWithDpRank>,
    pub pinned_worker: Option<WorkerWithDpRank>,
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    pub routing_constraints: RoutingConstraints,
    pub router_config_override: Option<RouterConfigOverride>,
    pub track_prefill_tokens: bool,
    pub priority_jump: f64,
    pub strict_priority: u32,
    pub policy_class: Option<String>,
    pub session_context: Option<SessionContext>,

    // Overlap and cache signals.
    pub overlap: OverlapSignals,
    pub router_hint_candidates: Option<RouterHintRootCandidates>,
    pub retain_router_hint_chain: bool,
    pub shared_cache_hits: Option<SharedCacheHits>,

    // Load state computed during admission.
    pub worker_loads: FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,

    /// Sender half of the admission ownership handoff. For tracked requests,
    /// the actor must book before sending and undo the booking if delivery fails.
    pub resp_tx: Option<tokio::sync::oneshot::Sender<Result<SchedulingResponse, KvSchedulerError>>>,
}

#[derive(Clone, Copy)]
pub struct SchedulingContext<'a, C> {
    request: &'a SchedulingRequest,
    eligibility: RoutingEligibility<'a>,
    workers: &'a HashMap<WorkerId, C>,
}

impl<'a, C: WorkerConfigLike> SchedulingContext<'a, C> {
    pub fn new(request: &'a SchedulingRequest, workers: &'a HashMap<WorkerId, C>) -> Self {
        Self {
            request,
            eligibility: request.eligibility(),
            workers,
        }
    }

    pub fn request(&self) -> &'a SchedulingRequest {
        self.request
    }

    pub fn best_effective_prefill_tokens(&self) -> usize {
        effective_prefill_tokens(self.request.isl_tokens, self.best_cached_tokens())
    }

    pub fn best_cached_tokens(&self) -> usize {
        match self.eligibility.pinned_worker() {
            Some(worker) => self.request.effective_cached_tokens_for(worker),
            None => self
                .request
                .overlap
                .effective_cached_tokens
                .iter()
                .filter(|(worker, _)| {
                    self.workers.get(&worker.worker_id).is_some_and(|config| {
                        self.eligibility.allows_worker(worker.worker_id, config)
                    })
                })
                .map(|(_, cached_tokens)| *cached_tokens)
                .max()
                .unwrap_or(0),
        }
    }
}

impl SchedulingRequest {
    #[inline]
    pub fn eligibility(&self) -> RoutingEligibility<'_> {
        self.eligibility_with_overloaded(None)
    }

    #[inline]
    pub fn eligibility_with_overloaded<'a>(
        &'a self,
        overloaded_worker_ids: Option<&'a HashSet<WorkerId>>,
    ) -> RoutingEligibility<'a> {
        RoutingEligibility::new(
            self.allowed_worker_ids.as_ref(),
            overloaded_worker_ids,
            self.pinned_worker,
            &self.routing_constraints,
        )
    }

    pub(crate) fn effective_cached_tokens_for(&self, worker: WorkerWithDpRank) -> usize {
        self.overlap
            .effective_cached_tokens
            .get(&worker)
            .copied()
            .unwrap_or(0)
    }

    pub(crate) fn effective_overlap_blocks_for(&self, worker: WorkerWithDpRank) -> f64 {
        self.overlap
            .effective_overlap_blocks
            .get(&worker)
            .copied()
            .unwrap_or(0.0)
    }

    pub fn worker_load_for(&self, worker: WorkerWithDpRank) -> WorkerLoadProjection {
        self.worker_loads.get(&worker).copied().unwrap_or_default()
    }

    pub(crate) fn request_blocks(&self, block_size: u32) -> u64 {
        self.isl_tokens.div_ceil(block_size as usize) as u64
    }

    pub(crate) fn potential_decode_blocks_after_admission(
        &self,
        worker: WorkerWithDpRank,
        block_size: u32,
    ) -> usize {
        let request_blocks = self.request_blocks(block_size) as usize;
        let tracked_request_blocks = self.token_seq.as_ref().map_or(0, Vec::len);
        let untracked_request_blocks = request_blocks.saturating_sub(tracked_request_blocks);

        self.worker_load_for(worker)
            .potential_decode_blocks()
            .saturating_add(untracked_request_blocks)
    }

    pub(crate) fn response_is_closed(&self) -> bool {
        self.resp_tx.as_ref().is_none_or(|tx| tx.is_closed())
    }

    pub fn respond(&mut self, result: Result<SchedulingResponse, KvSchedulerError>) -> bool {
        let Some(tx) = self.resp_tx.take() else {
            tracing::error!("respond called multiple times on same request");
            return false;
        };
        if tx.send(result).is_err() {
            tracing::debug!("requestor dropped scheduling response");
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_with_decode_load(
        isl_tokens: usize,
        token_seq_blocks: Option<usize>,
        active_decode_blocks: usize,
        additional_active_blocks: usize,
        worker: WorkerWithDpRank,
    ) -> SchedulingRequest {
        let token_seq = token_seq_blocks.map(|blocks| (0..blocks as SequenceHash).collect());
        let mut worker_loads = FxHashMap::default();
        worker_loads.insert(
            worker,
            WorkerLoadProjection {
                active_decode_blocks,
                additional_active_blocks,
                ..Default::default()
            },
        );

        SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq,
            isl_tokens,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: false,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            shared_cache_hits: None,
            worker_loads,
            resp_tx: None,
        }
    }

    #[test]
    fn potential_decode_blocks_after_admission_counts_only_untracked_request_blocks() {
        let worker = WorkerWithDpRank::new(0, 0);

        for (name, isl_tokens, token_seq_blocks, additional_active_blocks, expected) in [
            ("untracked", 32, None, 0, 102),
            ("tracked_no_overlap", 32, Some(2), 2, 102),
            ("tracked_full_active_overlap", 32, Some(2), 0, 100),
            ("tracked_partial_tail", 33, Some(2), 2, 103),
        ] {
            let request = request_with_decode_load(
                isl_tokens,
                token_seq_blocks,
                100,
                additional_active_blocks,
                worker,
            );

            assert_eq!(
                request.potential_decode_blocks_after_admission(worker, 16),
                expected,
                "{name}"
            );
        }
    }
}
