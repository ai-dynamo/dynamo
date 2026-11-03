// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile external plugins against the public `plugins` contracts and the common registry.
//! Scorer implementations use the current batch signature; old scalar signatures must migrate.

use std::sync::Arc;

use dynamo_kv_router::plugins::request_classifier::{
    self, ClassifyFuture, ClassifyRequest, RequestClassifier,
};
use dynamo_kv_router::plugins::worker_selection::{
    self, WorkerCandidate, WorkerCandidates, WorkerFilter, WorkerInputView, WorkerInputs,
    WorkerPicker, WorkerScorer, WorkerSelectionContext, WorkerSelectionPolicy,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
};
use dynamo_kv_router::selector::{HostedSelectionRequest, HostedWorkerPicker};
use dynamo_kv_router::{
    KvRouterConfig, RoutingPartitionRef, WorkerSelectionPolicyError, WorkerType,
};

struct ExamplePolicy;

impl WorkerFilter for ExamplePolicy {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE
    }

    fn keep(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidate: WorkerCandidate<'_>,
    ) -> Result<bool, WorkerSelectionPolicyError> {
        let _: Option<&dynamo_kv_router::SessionContext> = context.session_context();
        let _: Option<u32> = context.expected_output_tokens();
        let cache = candidate.cache().expect("requested cache inputs");
        Ok(cache.device_overlap_blocks() + cache.host_overlap_blocks() >= 0.0)
    }
}

impl WorkerScorer for ExamplePolicy {
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

impl WorkerPicker for ExamplePolicy {
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

struct ExampleClassifier;

impl RequestClassifier for ExampleClassifier {
    fn classify(
        &mut self,
        request: ClassifyRequest,
        _workers: WorkerCandidates<'_>,
    ) -> ClassifyFuture {
        let _: Option<&str> = request.request_id();
        let _: usize = request.isl_tokens();
        let _: Option<&[u64]> = request.sequence_hashes();
        let _: &request_classifier::RequestProgress = request.progress();
        let _: Option<tokio::time::Instant> = request.due_at();
        Box::pin(async move { Ok(request) })
    }
}

fn example_provider(
    _parameters: &WorkerSelectionPolicyParameters,
) -> Result<worker_selection::WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    Ok(Arc::new(|config, worker_type, _partition| {
        WorkerSelectionPolicy::new_with_filters(
            config.clone(),
            worker_type.as_str(),
            vec![Box::new(ExamplePolicy)],
            vec![Box::new(ExamplePolicy)],
            Box::new(ExamplePolicy),
        )
    }))
}

#[test]
fn plugins_resolve_through_the_common_registry() {
    let _: Box<dyn RequestClassifier> = Box::new(ExampleClassifier);

    let mut registry = RouterPluginRegistry::default();
    registry
        .register_worker_selection("example", Arc::new(example_provider))
        .unwrap();
    let policy = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(policy.path(), "worker_selection:\n  aggregated: example\n  prefill: example\n  decode: example\n  encode: example\n  instances:\n    - name: example\n      type: example\n").unwrap();
    let config = KvRouterConfig {
        router_policy_config: Some(policy.path().display().to_string()),
        ..Default::default()
    };
    let plugins = registry.resolve_plugins(&config).unwrap();
    let factory = plugins.worker_selection().unwrap();
    let _policy: WorkerSelectionPolicy = factory(
        &config,
        WorkerType::Aggregated,
        RoutingPartitionRef::new("model", "default"),
    );
}

#[test]
fn active_requests_pickers_select_through_the_public_hosted_api() {
    #[allow(deprecated)]
    let alias = WorkerInputs::OCCUPANCY;
    assert_eq!(alias, WorkerInputs::ACTIVE_REQUESTS);

    let Ok(picker) = WorkerSelectionPolicy::new(
        KvRouterConfig::default(),
        "decode",
        Vec::new(),
        Box::new(ActiveRequestsPicker),
    )
    .into_hosted_picker() else {
        panic!("a picker-only ACTIVE_REQUESTS policy runs hosted");
    };
    let picker: HostedWorkerPicker = picker;
    assert_eq!(picker.inputs(), WorkerInputs::ACTIVE_REQUESTS);
    let counts = |worker_id: u64| if worker_id == 4 { 0 } else { 3 };
    let selected = picker
        .select(
            &[2, 4, 6],
            Some(&counts),
            &HostedSelectionRequest::new(32, 16),
        )
        .unwrap();
    assert_eq!(selected.worker.worker_id, 4);
}
