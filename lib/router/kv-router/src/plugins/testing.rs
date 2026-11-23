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
//! registration test.
//!
//! For a request classifier, [`classify_request`] builds the request, [`classify`] runs the
//! classifier with no worker rows, and [`SelectionCase::classify`] runs it over the case's
//! workers when it declares worker inputs. [`request_progress`] builds a progress reading.
//!
//! This module follows the stability of the plugin API ([`super`]).

use std::collections::HashMap;

use dynamo_admission::host::ClassifyRequestHostExt;
use dynamo_admission::{ClassifierError, ClassifyRequest, RequestClassifier, RequestProgress};
use dynamo_router_protocol::WorkerType;
use dynamo_router_protocol::columns::WorkerInputs;
use dynamo_router_protocol::identity::RoutingPartitionRef;
use dynamo_router_protocol::session::SessionContext;
use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};
use dynamo_scheduler::host::{
    AccountingAttribution, KvSchedulerError, OverlapSignals, RoutingConstraints, ScheduleMode,
    SchedulingRequest, WorkerConfigLike, WorkerSelectionInput, WorkerSelector, worker_input_rows,
};
use dynamo_session_affinity::WorkerAffinityTarget;
use dynamo_slot_tracker::WorkerLoadProjection;
use dynamo_worker_selection::{
    SelectionConfig, WorkerInputRows, WorkerSelectionPolicy, WorkerSelectionPolicyContext,
    WorkerSelectionPolicyError,
};

use crate::plugins::RouterPluginRegistry;
use crate::plugins::worker_selection::WorkerSelectionPolicyRegistryError;

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

    /// Run `classifier` on `request` with this case's workers as its worker view, built for
    /// the input groups the classifier declares. The view holds every worker in the case.
    pub fn classify<C: RequestClassifier + ?Sized>(
        &self,
        classifier: &mut C,
        request: ClassifyRequest,
    ) -> Result<ClassifyRequest, Box<ClassifierError>> {
        let inputs = classifier.required_worker_inputs();
        let rows = if inputs == WorkerInputs::NONE {
            WorkerInputRows::empty()
        } else {
            worker_input_rows(
                &self.workers,
                &self.request,
                self.request.eligibility(),
                inputs,
            )
        };
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

/// Build the policy that `registry` registers under `policy_type`, for workers of
/// `worker_type`, from `parameters`: the mapping a policy file would give that type. Use it to
/// test that a plugin registers under its name and that its provider accepts or rejects
/// parameters, without resolving a policy file.
pub fn registered_policy(
    registry: &RouterPluginRegistry,
    policy_type: &str,
    parameters: serde_json::Value,
    worker_type: WorkerType,
) -> Result<WorkerSelectionPolicy, WorkerSelectionPolicyRegistryError> {
    let parameters = match parameters {
        serde_json::Value::Null => serde_yaml::Value::Mapping(Default::default()),
        parameters => serde_yaml::to_value(parameters).expect("JSON converts to YAML"),
    };
    let factory = registry.resolve_type_with(policy_type, parameters)?;
    Ok(factory(WorkerSelectionPolicyContext::new(
        &SelectionConfig::default(),
        worker_type,
        RoutingPartitionRef::new("model", "default"),
    )))
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
    classify_request(context_tokens, None, None)
        .progress()
        .clone()
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
    use std::sync::Arc;

    // Only the harness names: every other name comes from the plugin API, as in a plugin's
    // own tests, and not from this module's kernel and host imports.
    use super::{CaseWorker, SelectionCase, classify, classify_request, registered_policy};
    use crate::plugins::request_classifier::*;
    use crate::plugins::worker_selection::*;

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
            .worker(29, CaseWorker::new().active_requests(4))
            .worker(41, CaseWorker::new().active_requests(9));
        let classified = busy
            .classify(&mut classifier, classify_request(160, Some("r1"), None))
            .unwrap();
        assert_eq!(classifier.rows_seen, 2);
        assert_eq!(classified.policy_class(), Some("batch"));

        let idle = SelectionCase::new(160, 16)
            .worker(29, CaseWorker::new().active_requests(4))
            .worker(41, CaseWorker::new().active_requests(1));
        let classified = idle
            .classify(&mut classifier, classify_request(160, Some("r2"), None))
            .unwrap();
        assert_eq!(classified.policy_class(), None);

        // Without a case the classifier is lent no rows.
        let classified = classify(&mut classifier, classify_request(160, None, None)).unwrap();
        assert_eq!(classifier.rows_seen, 0);
        assert_eq!(classified.policy_class(), None);
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

        let policy = registered_policy(
            &registry,
            "prefer",
            serde_json::json!({ "prefer": 41 }),
            WorkerType::Aggregated,
        )
        .unwrap();
        let selected = SelectionCase::new(16, 16)
            .worker(29, CaseWorker::new())
            .worker(41, CaseWorker::new())
            .select(&policy)
            .unwrap();
        assert_eq!(selected, WorkerWithDpRank::from_worker_id(41));

        assert!(matches!(
            registered_policy(
                &registry,
                "prefer",
                serde_json::json!({ "unknown": 1 }),
                WorkerType::Aggregated,
            ),
            Err(WorkerSelectionPolicyRegistryError::Provider { .. })
        ));
        assert!(matches!(
            registered_policy(
                &registry,
                "absent",
                serde_json::Value::Null,
                WorkerType::Aggregated,
            ),
            Err(WorkerSelectionPolicyRegistryError::UnknownType { .. })
        ));
    }
}
