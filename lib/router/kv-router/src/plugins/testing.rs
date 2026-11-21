// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A harness for unit-testing a worker-selection policy.
//!
//! [`SelectionCase`] describes one selection as a policy sees it (a request and the workers it
//! may pick) and runs a [`WorkerSelectionPolicy`] over it through the same path the router
//! uses, so a plugin crate tests its filters, scorers, and picker without naming a router
//! type outside the plugin API. Enable the `plugin-testing` feature of `dynamo-kv-router` in
//! the plugin crate's `[dev-dependencies]`.
//!
//! ```
//! use dynamo_kv_router::plugins::testing::{CaseWorker, SelectionCase};
//! use dynamo_kv_router::plugins::v1alpha1::*;
//!
//! struct FewestRequests;
//!
//! impl WorkerPicker for FewestRequests {
//!     fn required_worker_inputs(&self) -> WorkerInputs {
//!         WorkerInputs::LOAD
//!     }
//!
//!     fn pick(
//!         &self,
//!         _context: &WorkerSelectionContext<'_>,
//!         input: WorkerInputView<'_>,
//!     ) -> Result<usize, WorkerSelectionPolicyError> {
//!         let loads = input.load().expect("declared LOAD");
//!         (0..input.len())
//!             .min_by_key(|&row| loads[row].active_requests())
//!             .ok_or_else(|| WorkerSelectionPolicyError::failed("no candidates"))
//!     }
//! }
//!
//! let policy = WorkerSelectionPolicy::new(
//!     "test",
//!     Vec::new(),
//!     Box::new(FewestRequests),
//! );
//! let selected = SelectionCase::new(160, 16)
//!     .worker(29, CaseWorker::new().active_requests(4))
//!     .worker(41, CaseWorker::new().active_requests(1))
//!     .select(&policy)
//!     .unwrap();
//! assert_eq!(selected, WorkerWithDpRank::from_worker_id(41));
//! ```
//!
//! [`classify_request`] and [`request_progress`] build the inputs a request classifier's
//! tests need.
//!
//! This module follows the stability of [`super::v1alpha1`].

use std::collections::HashMap;

use dynamo_admission::host::ClassifyRequestHostExt;
use dynamo_admission::{ClassifyRequest, RequestProgress};
use dynamo_router_protocol::session::SessionContext;
use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};
use dynamo_scheduler::host::{
    AccountingAttribution, KvSchedulerError, OverlapSignals, RoutingConstraints, ScheduleMode,
    SchedulingRequest, WorkerConfigLike, WorkerSelectionInput, WorkerSelector,
};
use dynamo_session_affinity::WorkerAffinityTarget;
use dynamo_slot_tracker::WorkerLoadProjection;
use dynamo_worker_selection::{WorkerSelectionPolicy, WorkerSelectionPolicyError};

/// The signals one worker reports in a [`SelectionCase`]. Every value starts at zero or
/// unreported; set the ones the policy under test reads.
#[derive(Clone, Copy, Debug, Default)]
pub struct CaseWorker {
    device_overlap_blocks: usize,
    host_overlap_blocks: usize,
    disk_overlap_blocks: usize,
    load: WorkerLoadProjection,
    max_num_batched_tokens: Option<u64>,
    total_kv_blocks: Option<u64>,
}

impl CaseWorker {
    /// A worker with no cached prefix, no load, and no advertised capacity.
    pub fn new() -> Self {
        Self::default()
    }

    /// Prefix blocks of the request this worker holds on its device tier
    /// (`WorkerInputs::CACHE`).
    pub fn device_overlap_blocks(mut self, blocks: usize) -> Self {
        self.device_overlap_blocks = blocks;
        self
    }

    /// Prefix blocks of the request this worker holds on its host tier
    /// (`WorkerInputs::CACHE`).
    pub fn host_overlap_blocks(mut self, blocks: usize) -> Self {
        self.host_overlap_blocks = blocks;
        self
    }

    /// Prefix blocks of the request this worker holds on its disk tier
    /// (`WorkerInputs::CACHE`).
    pub fn disk_overlap_blocks(mut self, blocks: usize) -> Self {
        self.disk_overlap_blocks = blocks;
        self
    }

    /// Requests in flight on this worker (`WorkerInputs::LOAD` and
    /// `WorkerInputs::ACTIVE_REQUESTS`).
    pub fn active_requests(mut self, requests: usize) -> Self {
        self.load.active_requests = requests;
        self
    }

    /// Prompt tokens this worker has yet to prefill (`WorkerInputs::LOAD`).
    pub fn active_prefill_tokens(mut self, tokens: usize) -> Self {
        self.load.active_prefill_tokens = tokens;
        self
    }

    /// KV blocks this worker's in-flight requests hold (`WorkerInputs::LOAD`).
    pub fn active_decode_blocks(mut self, blocks: usize) -> Self {
        self.load.active_decode_blocks = blocks;
        self
    }

    /// Prefill tokens the worker batches per engine step, as `WorkerInfo` reports it.
    pub fn max_num_batched_tokens(mut self, tokens: u64) -> Self {
        self.max_num_batched_tokens = Some(tokens);
        self
    }

    /// Total KV blocks the worker advertises, as `WorkerInfo` reports it.
    pub fn total_kv_blocks(mut self, blocks: u64) -> Self {
        self.total_kv_blocks = Some(blocks);
        self
    }
}

/// The worker registration the router reads capacity from: one data-parallel rank.
struct CaseWorkerConfig {
    max_num_batched_tokens: Option<u64>,
    total_kv_blocks: Option<u64>,
}

impl WorkerConfigLike for CaseWorkerConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }

    fn data_parallel_size(&self) -> u32 {
        1
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        self.max_num_batched_tokens
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        self.total_kv_blocks
    }
}

/// Why a [`SelectionCase`] selected no worker.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SelectionCaseError {
    /// A filter, scorer, or picker of the policy returned an error, or the router rejected
    /// what it returned (a non-finite cost, a row outside the candidate table).
    #[error(transparent)]
    Policy(WorkerSelectionPolicyError),
    /// The router had no worker to offer or the policy's filters removed every one.
    #[error("no worker selected: {0}")]
    NoWorker(String),
}

/// One selection: a request and the workers a policy may pick from.
///
/// Each worker is one data-parallel rank. Every worker added is eligible; the case models no
/// overload, taint, or LoRA narrowing, which the router applies before a policy runs.
pub struct SelectionCase {
    request: SchedulingRequest,
    workers: HashMap<WorkerId, CaseWorkerConfig>,
    block_size: u32,
}

impl SelectionCase {
    /// A request of `isl_tokens` prompt tokens against workers whose KV block holds
    /// `block_size` tokens.
    pub fn new(isl_tokens: usize, block_size: u32) -> Self {
        Self {
            request: SchedulingRequest {
                mode: ScheduleMode::QueryOnly { request_id: None },
                token_seq: None,
                isl_tokens,
                lora_name: None,
                accounting_attribution: AccountingAttribution::default(),
                expected_output_tokens: None,
                affinity_target: None,
                pinned_worker: None,
                allowed_worker_ids: None,
                routing_constraints: RoutingConstraints::default(),
                router_config_override: None,
                track_prefill_tokens: true,
                priority_jump: 0.0,
                strict_priority: 0,
                policy_class: None,
                session_context: None,
                overlap: OverlapSignals::default(),
                kv_transfer_candidates: None,
                retain_kv_transfer_chain: false,
                shared_cache_hits: None,
                worker_loads: Default::default(),
                resp_tx: None,
            },
            workers: HashMap::new(),
            block_size,
        }
    }

    /// The request ID `WorkerSelectionContext::request_id` returns.
    pub fn request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request.mode = ScheduleMode::QueryOnly {
            request_id: Some(request_id.into()),
        };
        self
    }

    /// The session `WorkerSelectionContext::session_context` returns.
    pub fn session_context(mut self, session_context: SessionContext) -> Self {
        self.request.session_context = Some(session_context);
        self
    }

    /// The worker a session binding or a classifier asked for, as
    /// `WorkerSelectionContext::affinity_target` returns it.
    pub fn affinity_target(mut self, target: impl Into<WorkerAffinityTarget>) -> Self {
        self.request.affinity_target = Some(target.into());
        self
    }

    /// The worker the request is pinned to, as `WorkerSelectionContext::pinned_worker`
    /// returns it. The router offers the policy only this worker.
    pub fn pinned_worker(mut self, worker: WorkerWithDpRank) -> Self {
        self.request.pinned_worker = Some(worker);
        self
    }

    /// The output length `WorkerSelectionContext::expected_output_tokens` returns.
    pub fn expected_output_tokens(mut self, tokens: u32) -> Self {
        self.request.expected_output_tokens = Some(tokens);
        self
    }

    /// Add worker `worker_id`, or replace its signals when the case already has it.
    pub fn worker(mut self, worker_id: WorkerId, worker: CaseWorker) -> Self {
        self.set_worker(worker_id, worker);
        self
    }

    /// Add worker `worker_id` in place, or replace its signals when the case already has it,
    /// for a test that changes one worker between selections.
    pub fn set_worker(&mut self, worker_id: WorkerId, worker: CaseWorker) {
        let rank = WorkerWithDpRank::from_worker_id(worker_id);
        let tiers = &mut self.request.overlap.tier_overlap_blocks;
        for (tier, blocks) in [
            (&mut tiers.device, worker.device_overlap_blocks),
            (&mut tiers.host_pinned, worker.host_overlap_blocks),
            (&mut tiers.disk, worker.disk_overlap_blocks),
        ] {
            if blocks == 0 {
                tier.remove(&rank);
            } else {
                tier.insert(rank, blocks);
            }
        }
        self.request.worker_loads.insert(rank, worker.load);
        self.workers.insert(
            worker_id,
            CaseWorkerConfig {
                max_num_batched_tokens: worker.max_num_batched_tokens,
                total_kv_blocks: worker.total_kv_blocks,
            },
        );
    }

    /// Run `policy` over this case and return the worker it picked.
    pub fn select(
        &self,
        policy: &WorkerSelectionPolicy,
    ) -> Result<WorkerWithDpRank, SelectionCaseError> {
        policy
            .select_worker(WorkerSelectionInput::configured(
                &self.workers,
                &self.request,
                self.request.eligibility(),
                self.block_size,
            ))
            .map(|selected| selected.worker)
            .map_err(|error| match error {
                KvSchedulerError::WorkerSelectionPolicy(error) => SelectionCaseError::Policy(error),
                other => SelectionCaseError::NoWorker(other.to_string()),
            })
    }
}

/// A request to hand to `RequestClassifier::classify`: `isl_tokens` prompt tokens with no
/// cached prefix, arriving now, with the request ID and session a test gives it.
pub fn classify_request(
    isl_tokens: usize,
    request_id: Option<&str>,
    session_context: Option<SessionContext>,
) -> ClassifyRequest {
    let mut request = ClassifyRequest::with_timing(isl_tokens, 0, tokio::time::Instant::now());
    if let Some(request_id) = request_id {
        request = request.with_request_id(request_id);
    }
    if let Some(session_context) = session_context {
        request = request.with_session_context(session_context);
    }
    request
}

/// A progress reading fixed at `context_tokens`, as `ClassifyRequest::progress` returns one
/// for a request the router has not updated.
pub fn request_progress(context_tokens: usize) -> RequestProgress {
    dynamo_admission::host::request_progress(context_tokens).0
}
