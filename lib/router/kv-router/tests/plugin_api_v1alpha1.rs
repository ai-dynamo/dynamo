// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A worker-selection policy and a request classifier written against
//! `dynamo_kv_router::plugins::v1alpha1` alone, with the experimental dispatch gate imported
//! by name: no other path from this crate, no other router crate. This file is the surface
//! new plugins target. The tests below play the routing host, so they also name
//! `plugins::host` and the router configuration.

use std::sync::Arc;

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::plugins::experimental::dispatch_gate::{
    DispatchDecision, DispatchGate, DispatchHead,
};
use dynamo_kv_router::plugins::host::{
    HostedSelectionRequest, HostedWorkerPicker, RouterPluginRegistryHostExt, RouterPlugins,
    SelectionFailure, WorkerSelectionPolicyHostExt,
};
use dynamo_kv_router::plugins::v1alpha1::request_classifier::*;
use dynamo_kv_router::plugins::v1alpha1::worker_selection::*;

struct ContractPolicy;

impl WorkerFilter for ContractPolicy {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE
    }

    fn keep(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidate: WorkerCandidate<'_>,
    ) -> Result<bool, WorkerSelectionPolicyError> {
        let _: Option<&SessionContext> = context.session_context();
        let _: Option<u32> = context.expected_output_tokens();
        let _: WorkerWithDpRank = candidate.worker();
        let cache = candidate.cache().expect("requested cache inputs");
        Ok(cache.device_overlap_blocks() + cache.host_overlap_blocks() >= 0.0)
    }
}

impl WorkerScorer for ContractPolicy {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        for (candidate, cost) in candidates.iter().zip(costs) {
            let _: u64 = context.request_blocks();
            let _: u32 = context.block_size();
            let load = candidate.load().expect("requested load inputs");
            *cost = load.active_requests() as f64 + load.decode_cost_blocks();
        }
        Ok(())
    }
}

impl WorkerPicker for ContractPolicy {
    fn pick(
        &self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        input
            .candidates()
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.cost().total_cmp(&b.cost()))
            .map(|(row, _)| row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("empty candidate table"))
    }
}

struct ContractClassifier;

impl RequestClassifier for ContractClassifier {
    fn classify(
        &mut self,
        request: ClassifyRequest,
        _workers: WorkerCandidates<'_>,
    ) -> Result<ClassifyRequest, Box<ClassifierError>> {
        let _: Option<&str> = request.request_id();
        let _: usize = request.isl_tokens();
        let _: Option<&[u64]> = request.sequence_hashes();
        let _: &RequestProgress = request.progress();
        let _: Option<tokio::time::Instant> = request.due_at();
        Ok(request)
    }

    fn dispatch_gate(&self) -> Option<Arc<dyn DispatchGate>> {
        struct Open;
        impl DispatchGate for Open {
            fn check(&self, head: &DispatchHead<'_>) -> DispatchDecision {
                let _: Option<&str> = head.request_id();
                DispatchDecision::Go { target: None }
            }
        }
        Some(Arc::new(Open))
    }
}

fn contract_provider(
    _parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    Ok(Arc::new(|_config, worker_type, _partition| {
        WorkerSelectionPolicy::new_with_filters(
            worker_type.as_str(),
            vec![Box::new(ContractPolicy)],
            vec![Box::new(ContractPolicy)],
            Box::new(ContractPolicy),
        )
    }))
}

#[test]
fn v1alpha1_plugins_resolve_through_the_registry() {
    let _: Box<dyn WorkerFilter> = Box::new(ContractPolicy);
    let _: Box<dyn WorkerScorer> = Box::new(ContractPolicy);
    let _: Box<dyn WorkerPicker> = Box::new(ContractPolicy);
    let _: Box<dyn RequestClassifier> = Box::new(ContractClassifier);

    let mut registry = RouterPluginRegistry::default();
    registry
        .register_worker_selection("contract", Arc::new(contract_provider))
        .unwrap();
    let registry: RouterPluginRegistry = registry;
    let policy = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(policy.path(), "worker_selection:\n  aggregated: contract\n  prefill: contract\n  decode: contract\n  encode: contract\n").unwrap();
    let config = KvRouterConfig {
        router_policy_config: Some(policy.path().display().to_string()),
        ..Default::default()
    };
    let plugins: RouterPlugins = registry.resolve_plugins(&config).unwrap();
    let factory = plugins.worker_selection().unwrap();
    let _policy: WorkerSelectionPolicy = factory(
        &SelectionConfig::from(&config),
        WorkerType::Aggregated,
        RoutingPartitionRef::new("model", "default"),
    );
    let _: RoutingPartitionId = RoutingPartitionId::new("model", "default");
    let _: Result<RouterPlugins, RouterPluginRegistryError> = Ok(RouterPlugins::default());
}

/// Reads the host's in-flight counts lazily, the way a least-loaded picker does.
struct ActiveRequestsPicker;

impl WorkerPicker for ActiveRequestsPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::ACTIVE_REQUESTS
    }

    fn pick(
        &self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        (0..input.len())
            .min_by_key(|&row| input.active_requests(row).unwrap_or(u64::MAX))
            .ok_or_else(|| WorkerSelectionPolicyError::failed("empty candidate table"))
    }
}

#[test]
fn active_requests_pickers_select_through_the_hosted_api() {
    #[allow(deprecated)]
    let alias = WorkerInputs::OCCUPANCY;
    assert_eq!(alias, WorkerInputs::ACTIVE_REQUESTS);

    let Ok(picker) =
        WorkerSelectionPolicy::new("decode", Vec::new(), Box::new(ActiveRequestsPicker))
            .into_hosted_picker()
    else {
        panic!("a picker-only ACTIVE_REQUESTS policy runs hosted");
    };
    let picker: HostedWorkerPicker = picker;
    assert_eq!(picker.inputs(), WorkerInputs::ACTIVE_REQUESTS);
    let counts = |worker_id: u64| if worker_id == 4 { 0 } else { 3 };
    let selected = picker
        .select(&[2, 4, 6], Some(&counts), &HostedSelectionRequest::new(32))
        .unwrap();
    assert_eq!(selected, 4);
    let _: Result<WorkerId, SelectionFailure> =
        picker.select(&[], None, &HostedSelectionRequest::new(0));
}

/// The test harness a plugin crate uses, exercised the way a plugin's own tests would.
#[cfg(feature = "plugin-testing")]
mod harness {
    use std::sync::Arc;

    use dynamo_kv_router::plugins::testing::{
        CaseWorker, SelectionCase, classify, classify_request, registered_policy,
    };
    use dynamo_kv_router::plugins::v1alpha1::request_classifier::*;
    use dynamo_kv_router::plugins::v1alpha1::worker_selection::*;

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
                _context: &WorkerSelectionContext<'_>,
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
                    let factory: WorkerSelectionPolicyFactory =
                        Arc::new(move |_config, worker_type, _partition| {
                            WorkerSelectionPolicy::new(
                                worker_type.as_str(),
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
