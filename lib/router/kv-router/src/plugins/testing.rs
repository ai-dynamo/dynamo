// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A harness for unit-testing a worker-selection policy.
//!
//! [`SelectionCase`] describes one selection as a policy sees it (a request and the workers it
//! may pick) and runs a [`WorkerSelectionPolicy`] over it through the same path the router
//! uses, so a plugin crate tests its filters, scorers, and picker without naming a router
//! type outside the plugin API.
//!
//! Import it only from tests and benches. It builds requests, resolves registered types, and
//! runs a selection, which in shipped code only a routing host does (see [`super::host`]).
//! The module is public in every build, so that is a rule for plugin crates, not a check.
//!
//! ```
//! use dynamo_kv_router::plugins::testing::{CaseWorker, SelectionCase};
//! use dynamo_kv_router::plugins::worker_selection::*;
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
//!         _request: &WorkerSelectionRequest<'_>,
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
//! [`registered_policy`] builds a policy from a registered type name and parameters, for a
//! registration test, and [`registered_classifier`] does the same for a request classifier.
//!
//! For a request classifier, [`SelectionCase::classify`] runs it on the case: it builds the
//! request and the worker rows from the case, as the router builds both from one request.
//! Without a case, [`classify_request`] builds a request and [`classify`] runs the classifier
//! on it with no worker rows. [`request_progress`] builds a progress reading.
//!
//! This module follows the stability of the plugin API ([`super`]).

use std::collections::HashMap;

use dynamo_admission::host::ClassifyRequestHostExt;
use dynamo_admission::{
    ClassifierError, ClassifyRequest, RequestClassifier, RequestClassifierContext, RequestProgress,
};
use dynamo_kv_events::residency::StorageTier;
use dynamo_kv_indexer::TieredMatchDetails;
use dynamo_router_protocol::WorkerType;
use dynamo_router_protocol::identity::RoutingPartitionRef;
use dynamo_router_protocol::session::SessionContext;
use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};
use dynamo_scheduler::host::{
    AccountingAttribution, CacheHitWeights, KvSchedulerError, OverlapAnalysis, OverlapSignals,
    RoutingConstraints, ScheduleMode, SchedulingRequest, WorkerConfigLike, WorkerSelectionInput,
    WorkerSelector, classify_inputs,
};
use dynamo_session_affinity::WorkerAffinityTarget;
use dynamo_slot_tracker::{BusyThresholds, WorkerLoadProjection};
use dynamo_worker_selection::{
    SelectionConfig, WorkerInputRows, WorkerSelectionPolicy, WorkerSelectionPolicyContext,
    WorkerSelectionPolicyError,
};
use rustc_hash::FxHashMap;

use crate::plugins::RouterPluginRegistry;
use crate::plugins::request_classifier::{
    RequestClassifierProviderError, RequestClassifierRegistryError,
};
use crate::plugins::worker_selection::{
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};

/// The signals one worker reports in a [`SelectionCase`]. Every value starts at zero or
/// unreported; set the ones the policy under test reads.
///
/// The cache estimate a policy reads (`WorkerCacheInput::accounting_cache_estimate`) follows
/// from the three overlap tiers as it does in the router: the device blocks plus each lower
/// tier at its default cache weight, and that many blocks of tokens.
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

    /// The request ID `RequestInfo::request_id` returns.
    pub fn request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request.mode = ScheduleMode::QueryOnly {
            request_id: Some(request_id.into()),
        };
        self
    }

    /// The session `RequestInfo::session_context` returns.
    pub fn session_context(mut self, session_context: SessionContext) -> Self {
        self.request.session_context = Some(session_context);
        self
    }

    /// The worker a session binding or a classifier asked for, as
    /// `WorkerSelectionRequest::affinity_target` returns it.
    pub fn affinity_target(mut self, target: impl Into<WorkerAffinityTarget>) -> Self {
        self.request.affinity_target = Some(target.into());
        self
    }

    /// The worker the request is pinned to, as `RequestInfo::pinned_worker` returns it. The
    /// router offers the policy only this worker.
    pub fn pinned_worker(mut self, worker: WorkerWithDpRank) -> Self {
        self.request.pinned_worker = Some(worker);
        self
    }

    /// The output length `WorkerSelectionRequest::expected_output_tokens` returns.
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
        // The router's overlap analysis turns the worker's tiers into the tier blocks and the
        // cache estimate a row reads, so the two agree here as they do in the router. A tier
        // with no blocks is not reported, as an indexer reports none.
        let mut matches = TieredMatchDetails::default();
        if worker.device_overlap_blocks > 0 {
            let blocks = u32::try_from(worker.device_overlap_blocks).unwrap_or(u32::MAX);
            matches.device.overlap_scores.scores.insert(rank, blocks);
        }
        for (tier, blocks) in [
            (StorageTier::HostPinned, worker.host_overlap_blocks),
            (StorageTier::Disk, worker.disk_overlap_blocks),
        ] {
            if blocks > 0 {
                let tier = matches.lower_tier.entry(tier).or_default();
                tier.hits.insert(rank, blocks);
            }
        }
        let signals =
            OverlapAnalysis::new(&CacheHitWeights::default(), self.block_size, &matches).signals();
        let overlap = &mut self.request.overlap;
        let tiers = &mut overlap.tier_overlap_blocks;
        let reported = signals.tier_overlap_blocks;
        set_rank(&mut tiers.device, rank, reported.device.get(&rank).copied());
        set_rank(
            &mut tiers.host_pinned,
            rank,
            reported.host_pinned.get(&rank).copied(),
        );
        set_rank(&mut tiers.disk, rank, reported.disk.get(&rank).copied());
        set_rank(
            &mut overlap.effective_cached_tokens,
            rank,
            signals.effective_cached_tokens.get(&rank).copied(),
        );
        set_rank(
            &mut overlap.effective_overlap_blocks,
            rank,
            signals.effective_overlap_blocks.get(&rank).copied(),
        );
        self.request.worker_loads.insert(rank, worker.load);
        self.workers.insert(
            worker_id,
            CaseWorkerConfig {
                max_num_batched_tokens: worker.max_num_batched_tokens,
                total_kv_blocks: worker.total_kv_blocks,
            },
        );
    }

    /// Run `classifier` on this case. The request it classifies and the worker rows it reads
    /// are both built from the case, as the router builds both from one request: the case's
    /// request ID, session, and pinned worker are the request's, the rows hold the workers
    /// that pin admits with the input groups the classifier declares, and the request's cached
    /// tokens are the best cache estimate among those workers.
    pub fn classify<C: RequestClassifier + ?Sized>(
        &self,
        classifier: &mut C,
    ) -> Result<ClassifyRequest, Box<ClassifierError>> {
        let eligibility = self.request.eligibility();
        let (request, rows) = classify_inputs(
            &self.workers,
            &self.request,
            eligibility,
            eligibility,
            tokio::time::Instant::now(),
            BusyThresholds::default(),
            classifier.required_worker_inputs(),
        );
        rows.read(|workers| classifier.classify(request, workers))
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

/// Set `rank`'s entry in one of the request's per-rank overlap maps, or clear it.
fn set_rank<V>(map: &mut FxHashMap<WorkerWithDpRank, V>, rank: WorkerWithDpRank, value: Option<V>) {
    match value {
        Some(value) => {
            map.insert(rank, value);
        }
        None => {
            map.remove(&rank);
        }
    }
}

/// Build the policy that `registry` registers under `policy_type`, for workers of
/// `worker_type`, from `parameters`: the YAML a policy file would give that type, such as
/// `"max_idle_secs: 600"`, or `""` for none. Use it to test that a plugin registers under its
/// name and that its provider accepts or rejects parameters, without resolving a policy file.
pub fn registered_policy(
    registry: &RouterPluginRegistry,
    policy_type: &str,
    parameters: &str,
    worker_type: WorkerType,
) -> Result<WorkerSelectionPolicy, WorkerSelectionPolicyRegistryError> {
    let parameters = policy_file_parameters(parameters).map_err(|error| {
        WorkerSelectionPolicyRegistryError::Provider {
            policy_type: policy_type.to_owned(),
            source: WorkerSelectionPolicyProviderError::new(error.to_string()),
        }
    })?;
    let factory = registry.resolve_type_with(policy_type, parameters)?;
    Ok(factory(WorkerSelectionPolicyContext::new(
        &SelectionConfig::default(),
        worker_type,
        RoutingPartitionRef::new("model", "default"),
    )))
}

/// Build the classifier that `registry` registers under `classifier_type` from `parameters`,
/// the YAML a policy file would give that type or `""` for none, for the router `context`
/// describes. Use it to test that a classifier registers under its name and that its provider
/// accepts or rejects parameters, without resolving a policy file.
pub fn registered_classifier(
    registry: &RouterPluginRegistry,
    classifier_type: &str,
    parameters: &str,
    context: RequestClassifierContext,
) -> Result<Box<dyn RequestClassifier>, RequestClassifierRegistryError> {
    let parameters = policy_file_parameters(parameters).map_err(|error| {
        RequestClassifierRegistryError::Provider {
            classifier_type: classifier_type.to_owned(),
            source: RequestClassifierProviderError::new(error.to_string()),
        }
    })?;
    let factory = registry.resolve_classifier_type_with(classifier_type, parameters)?;
    Ok(factory(context))
}

/// Parse `parameters` as the mapping a policy file hands a provider. Empty text is no
/// parameters, which a provider receives as an empty mapping.
fn policy_file_parameters(parameters: &str) -> Result<serde_yaml::Value, serde_yaml::Error> {
    match serde_yaml::from_str(parameters)? {
        serde_yaml::Value::Null => Ok(serde_yaml::Value::Mapping(Default::default())),
        parameters => Ok(parameters),
    }
}

/// A request to hand to `RequestClassifier::classify`: `isl_tokens` prompt tokens with no
/// cached prefix, arriving now, with the request ID, session, and pinned worker a test gives
/// it. Together with `isl_tokens` those are the facts `ClassifyRequest::info` returns.
pub fn classify_request(
    isl_tokens: usize,
    request_id: Option<&str>,
    session_context: Option<SessionContext>,
    pinned_worker: Option<WorkerWithDpRank>,
) -> ClassifyRequest {
    let mut request = ClassifyRequest::with_timing(isl_tokens, 0, tokio::time::Instant::now())
        .with_pinned_worker(pinned_worker);
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

/// Run `classifier` on `request` with no worker rows, as the router does for a classifier that
/// declares no worker inputs.
pub fn classify<C: RequestClassifier + ?Sized>(
    classifier: &mut C,
    request: ClassifyRequest,
) -> Result<ClassifyRequest, Box<ClassifierError>> {
    WorkerInputRows::empty().read(|workers| classifier.classify(request, workers))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    // Only the harness names: every other name comes from the plugin API, as in a plugin's
    // own tests, and not from this module's kernel and host imports.
    use super::{
        CaseWorker, SelectionCase, classify, classify_request, registered_classifier,
        registered_policy,
    };
    use crate::plugins::request_classifier::*;
    use crate::plugins::worker_selection::*;

    /// Picks the first row.
    struct PickFirst;

    impl WorkerPicker for PickFirst {
        fn pick(
            &self,
            _request: &WorkerSelectionRequest<'_>,
            _input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            Ok(0)
        }
    }

    /// What a scorer reads from one worker's `CACHE` row: the three tiers and the estimate.
    #[derive(Debug, PartialEq)]
    struct CacheRow {
        device_blocks: f64,
        host_blocks: f64,
        disk_blocks: f64,
        estimate: (f64, usize),
    }

    /// Records each candidate's cache row by worker.
    struct RecordCache(Arc<Mutex<HashMap<WorkerId, CacheRow>>>);

    impl WorkerScorer for RecordCache {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::CACHE
        }

        fn score(
            &self,
            _request: &WorkerSelectionRequest<'_>,
            candidates: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            let mut seen = self.0.lock().unwrap();
            for (candidate, cost) in candidates.iter().zip(costs) {
                let cache = candidate.cache().expect("declared CACHE");
                seen.insert(
                    candidate.worker().worker_id,
                    CacheRow {
                        device_blocks: cache.device_overlap_blocks(),
                        host_blocks: cache.host_overlap_blocks(),
                        disk_blocks: cache.disk_overlap_blocks(),
                        estimate: cache.accounting_cache_estimate(),
                    },
                );
                *cost = 0.0;
            }
            Ok(())
        }
    }

    #[test]
    fn a_classifier_reads_the_request_facts_a_test_gives() {
        const PINNED: WorkerWithDpRank = WorkerWithDpRank {
            worker_id: 41,
            dp_rank: 0,
        };

        /// Checks the four facts `info()` returns.
        struct ReadFacts;

        impl RequestClassifier for ReadFacts {
            fn classify(
                &mut self,
                request: ClassifyRequest,
                _workers: WorkerCandidates<'_>,
            ) -> Result<ClassifyRequest, Box<ClassifierError>> {
                let info = request.info();
                assert_eq!(info.request_id(), Some("r1"));
                assert_eq!(info.isl_tokens(), 160);
                let session = info.session_context().expect("the test's session");
                assert_eq!(session.session_id(), "s1");
                assert_eq!(info.pinned_worker(), Some(PINNED));
                Ok(request)
            }
        }

        let session = SessionContext::new("s1".to_owned(), None, None, None);
        let request = classify_request(160, Some("r1"), Some(session), Some(PINNED));
        classify(&mut ReadFacts, request).unwrap();

        let unpinned = classify_request(160, None, None, None);
        assert_eq!(unpinned.info().pinned_worker(), None);
    }

    #[test]
    fn a_workers_cache_estimate_follows_from_its_tiers() {
        const BLOCK_SIZE: u32 = 16;
        let seen = Arc::new(Mutex::new(HashMap::new()));
        let policy = WorkerSelectionPolicy::new(
            "test",
            vec![Box::new(RecordCache(Arc::clone(&seen)))],
            Box::new(PickFirst),
        );
        // The estimate counts a device block as one and a lower tier at its default weight.
        let weights = SelectionConfig::default();
        let row = |device_blocks: f64, host_blocks: f64, disk_blocks: f64| {
            let blocks = device_blocks
                + host_blocks * weights.host_cache_hit_weight()
                + disk_blocks * weights.disk_cache_hit_weight();
            CacheRow {
                device_blocks,
                host_blocks,
                disk_blocks,
                estimate: (blocks, (blocks * f64::from(BLOCK_SIZE)).round() as usize),
            }
        };

        let mut case = SelectionCase::new(320, BLOCK_SIZE)
            .worker(1, CaseWorker::new().device_overlap_blocks(8))
            .worker(2, CaseWorker::new().host_overlap_blocks(4))
            .worker(3, CaseWorker::new().disk_overlap_blocks(8))
            // One lower tier on top of the device blocks. The router adds lower tiers in an
            // unspecified order, so with both tiers the sum would match this one bit for bit
            // only for weights like the defaults.
            .worker(
                4,
                CaseWorker::new()
                    .device_overlap_blocks(2)
                    .host_overlap_blocks(4),
            )
            .worker(5, CaseWorker::new());
        case.select(&policy).unwrap();
        assert_eq!(
            *seen.lock().unwrap(),
            HashMap::from([
                (1, row(8.0, 0.0, 0.0)),
                (2, row(0.0, 4.0, 0.0)),
                (3, row(0.0, 0.0, 8.0)),
                (4, row(2.0, 4.0, 0.0)),
                (5, row(0.0, 0.0, 0.0)),
            ])
        );
        assert_eq!(seen.lock().unwrap()[&1].estimate, (8.0, 128));

        // Replacing a worker's signals replaces its tiers and its estimate.
        case.set_worker(4, CaseWorker::new().disk_overlap_blocks(8));
        case.select(&policy).unwrap();
        assert_eq!(seen.lock().unwrap()[&4], row(0.0, 0.0, 8.0));
    }

    /// Marks a request `batch` when every observed worker is at or past a request count.
    struct LoadAwareClassifier {
        max_active_requests: usize,
        rows_seen: usize,
    }

    impl RequestClassifier for LoadAwareClassifier {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::LOAD
        }

        fn classify(
            &mut self,
            mut request: ClassifyRequest,
            workers: WorkerCandidates<'_>,
        ) -> Result<ClassifyRequest, Box<ClassifierError>> {
            self.rows_seen = workers.len();
            let mut loads = workers
                .iter()
                .filter_map(|worker| worker.load())
                .filter(|load| load.is_available())
                .peekable();
            if loads.peek().is_some()
                && loads.all(|load| load.active_requests() >= self.max_active_requests)
            {
                request.set_policy_class("batch");
            }
            Ok(request)
        }
    }

    #[test]
    fn a_classifier_reads_the_case_workers_through_the_selection_rows() {
        let mut classifier = LoadAwareClassifier {
            max_active_requests: 4,
            rows_seen: 0,
        };
        let busy = SelectionCase::new(160, 16)
            .request_id("r1")
            .worker(29, CaseWorker::new().active_requests(4))
            .worker(41, CaseWorker::new().active_requests(9));
        let classified = busy.classify(&mut classifier).unwrap();
        assert_eq!(classifier.rows_seen, 2);
        assert_eq!(classified.policy_class(), Some("batch"));
        assert_eq!(classified.info().request_id(), Some("r1"));

        let idle = SelectionCase::new(160, 16)
            .request_id("r2")
            .worker(29, CaseWorker::new().active_requests(4))
            .worker(41, CaseWorker::new().active_requests(1));
        let classified = idle.classify(&mut classifier).unwrap();
        assert_eq!(classified.policy_class(), None);

        // Without a case the classifier is lent no rows.
        let classified =
            classify(&mut classifier, classify_request(160, None, None, None)).unwrap();
        assert_eq!(classifier.rows_seen, 0);
        assert_eq!(classified.policy_class(), None);
    }

    #[test]
    fn a_case_builds_the_classifier_request_and_its_rows_from_one_request() {
        const PINNED: WorkerWithDpRank = WorkerWithDpRank {
            worker_id: 41,
            dp_rank: 0,
        };
        let mut classifier = LoadAwareClassifier {
            max_active_requests: 4,
            rows_seen: 0,
        };
        // Worker 29 holds half of the prompt's ten blocks; worker 41 holds none.
        let case = SelectionCase::new(160, 16)
            .worker(29, CaseWorker::new().device_overlap_blocks(5))
            .worker(41, CaseWorker::new());

        // No pin: both workers are rows, and the cost leaves out the best cached prefix.
        let classified = case.classify(&mut classifier).unwrap();
        assert_eq!(classifier.rows_seen, 2);
        assert_eq!(classified.info().pinned_worker(), None);
        assert!(classified.scheduling_cost_tokens() < 160);

        // Pinned to the worker with no cache: the pin narrows the rows and is the request's
        // pin, and the cost is the whole prompt.
        let classified = case
            .pinned_worker(PINNED)
            .classify(&mut classifier)
            .unwrap();
        assert_eq!(classifier.rows_seen, 1);
        assert_eq!(classified.info().pinned_worker(), Some(PINNED));
        assert_eq!(classified.scheduling_cost_tokens(), 160);
    }

    #[test]
    fn a_policy_reads_back_the_request_and_capacity_a_case_describes() {
        const PINNED: WorkerWithDpRank = WorkerWithDpRank {
            worker_id: 41,
            dp_rank: 0,
        };

        /// Checks what the request and its one row show, then picks that row.
        struct ReadBack;

        impl WorkerPicker for ReadBack {
            fn pick(
                &self,
                request: &WorkerSelectionRequest<'_>,
                input: WorkerInputView<'_>,
            ) -> Result<usize, WorkerSelectionPolicyError> {
                let info = request.info();
                assert_eq!(info.request_id(), Some("r1"));
                assert_eq!(info.isl_tokens(), 320);
                let session = info.session_context().expect("the case's session");
                assert_eq!(session.session_id(), "s1");
                assert_eq!(info.pinned_worker(), Some(PINNED));
                assert_eq!(request.affinity_target(), Some(PINNED.into()));
                assert_eq!(request.expected_output_tokens(), Some(64));

                // A pinned request is offered its pin alone.
                assert_eq!(input.len(), 1);
                let worker = input.info(0).expect("row 0");
                assert_eq!(worker.worker(), PINNED);
                assert_eq!(worker.max_num_batched_tokens(), Some(4096));
                assert_eq!(worker.total_kv_blocks(), Some(1000));
                Ok(0)
            }
        }

        let policy = WorkerSelectionPolicy::new("test", Vec::new(), Box::new(ReadBack));
        let selected = SelectionCase::new(320, 16)
            .request_id("r1")
            .session_context(SessionContext::new("s1".to_owned(), None, None, None))
            .pinned_worker(PINNED)
            .affinity_target(PINNED)
            .expected_output_tokens(64)
            .worker(29, CaseWorker::new())
            .worker(
                41,
                CaseWorker::new()
                    .max_num_batched_tokens(4096)
                    .total_kv_blocks(1000),
            )
            .select(&policy)
            .unwrap();
        assert_eq!(selected, PINNED);
    }

    #[test]
    fn a_registered_policy_is_built_from_its_parameters() {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Parameters {
            #[serde(default)]
            prefer: u64,
        }

        struct Prefer(u64);
        impl WorkerPicker for Prefer {
            fn pick(
                &self,
                _request: &WorkerSelectionRequest<'_>,
                input: WorkerInputView<'_>,
            ) -> Result<usize, WorkerSelectionPolicyError> {
                input
                    .candidates()
                    .iter()
                    .position(|candidate| candidate.worker().worker_id == self.0)
                    .ok_or_else(|| WorkerSelectionPolicyError::failed("preferred worker absent"))
            }
        }

        let mut registry = RouterPluginRegistry::default();
        registry
            .register_worker_selection(
                "prefer",
                Arc::new(|parameters: &WorkerSelectionPolicyParameters| {
                    let parameters: Parameters = parameters.deserialize()?;
                    let factory: WorkerSelectionPolicyFactory = Arc::new(move |context| {
                        WorkerSelectionPolicy::new(
                            context.worker_type().as_str(),
                            Vec::new(),
                            Box::new(Prefer(parameters.prefer)),
                        )
                    });
                    Ok(factory)
                }),
            )
            .unwrap();

        let policy =
            registered_policy(&registry, "prefer", "prefer: 41", WorkerType::Aggregated).unwrap();
        let selected = SelectionCase::new(16, 16)
            .worker(29, CaseWorker::new())
            .worker(41, CaseWorker::new())
            .select(&policy)
            .unwrap();
        assert_eq!(selected, WorkerWithDpRank::from_worker_id(41));

        assert!(matches!(
            registered_policy(&registry, "prefer", "unknown: 1", WorkerType::Aggregated,),
            Err(WorkerSelectionPolicyRegistryError::Provider { .. })
        ));
        // Text that is not YAML is reported as bad parameters, not a panic.
        assert!(matches!(
            registered_policy(&registry, "prefer", "prefer: [", WorkerType::Aggregated),
            Err(WorkerSelectionPolicyRegistryError::Provider { .. })
        ));
        assert!(matches!(
            registered_policy(&registry, "absent", "", WorkerType::Aggregated,),
            Err(WorkerSelectionPolicyRegistryError::UnknownType { .. })
        ));
    }

    #[test]
    fn a_registered_classifier_is_built_from_its_parameters() {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Parameters {
            class: String,
        }

        /// Puts every request in one policy class.
        struct OneClass(String);
        impl RequestClassifier for OneClass {
            fn classify(
                &mut self,
                mut request: ClassifyRequest,
                _workers: WorkerCandidates<'_>,
            ) -> Result<ClassifyRequest, Box<ClassifierError>> {
                request.set_policy_class(&self.0);
                Ok(request)
            }
        }

        let mut registry = RouterPluginRegistry::default();
        registry
            .register_request_classifier(
                "one-class",
                Arc::new(|parameters: &RequestClassifierParameters| {
                    let parameters: Parameters = parameters.deserialize()?;
                    let factory: RequestClassifierFactory =
                        Arc::new(move |_context| Box::new(OneClass(parameters.class.clone())));
                    Ok(factory)
                }),
            )
            .unwrap();
        let context = || RequestClassifierContext::new(16, Vec::new);

        let mut classifier =
            registered_classifier(&registry, "one-class", "class: batch", context()).unwrap();
        let classified = classify(
            classifier.as_mut(),
            classify_request(160, Some("r1"), None, None),
        )
        .unwrap();
        assert_eq!(classified.policy_class(), Some("batch"));

        assert!(matches!(
            registered_classifier(&registry, "one-class", "unknown: 1", context(),),
            Err(RequestClassifierRegistryError::Provider { .. })
        ));
        assert!(matches!(
            registered_classifier(&registry, "absent", "", context()),
            Err(RequestClassifierRegistryError::UnknownType { .. })
        ));
    }
}
