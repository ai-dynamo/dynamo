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
use crate::scheduling::policy_queue::QueueRejection;
use crate::sequences::WorkerLoadProjection;

pub type OverloadedWorkerProvider =
    Arc<dyn Fn() -> Option<HashSet<WorkerId>> + Send + Sync + 'static>;

/// DIS-2404: supplies the set of fenced (dead/deregistered) workers at
/// selection time. Same shape as [`OverloadedWorkerProvider`], but a fenced
/// worker is rejected on every eligibility path (availability overrides
/// cache affinity), whereas overload is ignored by the affinity path.
pub type FencedWorkerProvider = Arc<dyn Fn() -> Option<HashSet<WorkerId>> + Send + Sync + 'static>;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierOverlapBlocks {
    #[serde(default)]
    pub device: FxHashMap<WorkerWithDpRank, usize>,
    #[serde(default)]
    pub host_pinned: FxHashMap<WorkerWithDpRank, usize>,
    #[serde(default)]
    pub disk: FxHashMap<WorkerWithDpRank, usize>,
}

#[derive(Debug, thiserror::Error)]
pub enum KvSchedulerError {
    #[error("no endpoints available to route work")]
    NoEndpoints,

    #[error(transparent)]
    QueueRejected(#[from] QueueRejection),

    #[error("all eligible workers are overloaded")]
    AllEligibleWorkersOverloaded,

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
}

#[derive(Debug, Clone)]
pub enum ScheduleMode {
    QueryOnly { request_id: Option<String> },
    Tracked { request_id: String },
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
            Self::Tracked { request_id } => Some(request_id),
        }
    }

    pub fn is_tracked(&self) -> bool {
        matches!(self, Self::Tracked { .. })
    }

    pub fn tracked_request_id(&self) -> Option<&str> {
        match self {
            Self::QueryOnly { .. } => None,
            Self::Tracked { request_id } => Some(request_id),
        }
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
    pub pinned_worker: Option<WorkerWithDpRank>,
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    pub routing_constraints: RoutingConstraints,
    pub router_config_override: Option<RouterConfigOverride>,
    pub priority_jump: f64,
    pub strict_priority: u32,
    pub policy_class: Option<String>,
    pub session_id: Option<String>,
    pub overlap: OverlapSignals,
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
    pub pinned_worker: Option<WorkerWithDpRank>,
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    pub routing_constraints: RoutingConstraints,
    pub router_config_override: Option<RouterConfigOverride>,
    pub track_prefill_tokens: bool,
    pub priority_jump: f64,
    pub strict_priority: u32,
    pub policy_class: Option<String>,
    pub session_id: Option<String>,

    // Overlap and cache signals.
    pub overlap: OverlapSignals,
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
