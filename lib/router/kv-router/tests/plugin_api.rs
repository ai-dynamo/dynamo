// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A worker-selection policy and a request classifier written against
//! `dynamo_kv_router::plugins::{worker_selection, request_classifier}` alone, with the
//! experimental dispatch gate imported by name: no other path from this crate, no other router
//! crate. This file is the surface
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
use dynamo_kv_router::plugins::request_classifier::*;
use dynamo_kv_router::plugins::worker_selection::*;

struct ContractPolicy;

impl WorkerFilter for ContractPolicy {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE
    }

    fn keep(
        &self,
        request: &WorkerSelectionRequest<'_>,
        candidate: WorkerCandidate<'_>,
    ) -> Result<bool, WorkerSelectionPolicyError> {
        let _: Option<&SessionContext> = request.info().session_context();
        let _: Option<u32> = request.expected_output_tokens();
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
        request: &WorkerSelectionRequest<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        for (candidate, cost) in candidates.iter().zip(costs) {
            let _: u64 = request.request_blocks();
            let _: u32 = request.block_size();
            let load = candidate.load().expect("requested load inputs");
            *cost = load.active_requests() as f64 + load.decode_cost_blocks();
        }
        Ok(())
    }
}

impl WorkerPicker for ContractPolicy {
    fn pick(
        &self,
        _request: &WorkerSelectionRequest<'_>,
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
        let info: RequestInfo<'_> = request.info();
        let _: Option<&str> = info.request_id();
        let _: usize = info.isl_tokens();
        let _: Option<&[u64]> = request.sequence_hashes();
        let _: &RequestProgress = request.progress();
        let _: Option<tokio::time::Instant> = request.due_at();
        Ok(request)
    }

    fn dispatch_gate(&self) -> Option<Arc<dyn DispatchGate>> {
        struct Open;
        impl DispatchGate for Open {
            fn check(&self, head: &DispatchHead<'_>) -> DispatchDecision {
                let _: Option<&str> = head.info().request_id();
                DispatchDecision::Go { target: None }
            }
        }
        Some(Arc::new(Open))
    }
}

fn contract_provider(
    _parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    Ok(Arc::new(|context| {
        WorkerSelectionPolicy::new_with_filters(
            context.worker_type().as_str(),
            vec![Box::new(ContractPolicy)],
            vec![Box::new(ContractPolicy)],
            Box::new(ContractPolicy),
        )
    }))
}

#[test]
fn plugins_resolve_through_the_registry() {
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
    let _policy: WorkerSelectionPolicy = factory(WorkerSelectionPolicyContext::new(
        &SelectionConfig::from(&config),
        WorkerType::Aggregated,
        RoutingPartitionRef::new("model", "default"),
    ));
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
        _request: &WorkerSelectionRequest<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        (0..input.len())
            .min_by_key(|&row| input.active_requests(row).unwrap_or(u64::MAX))
            .ok_or_else(|| WorkerSelectionPolicyError::failed("empty candidate table"))
    }
}

#[test]
fn active_requests_pickers_select_through_the_hosted_api() {
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
