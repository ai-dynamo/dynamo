// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A worker-selection policy and a request classifier written against
//! `dynamo_kv_router::plugins::prelude` alone: no other path from this crate, no other
//! router crate. This file is the surface new plugins target.

use std::sync::Arc;

use dynamo_kv_router::plugins::prelude::*;

struct PreludePolicy;

impl WorkerFilter for PreludePolicy {
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

impl WorkerScorer for PreludePolicy {
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

impl WorkerPicker for PreludePolicy {
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

struct PreludeClassifier;

impl RequestClassifier for PreludeClassifier {
    fn classify(&mut self, request: ClassifyRequest, _workers: &dyn WorkerView) -> ClassifyFuture {
        let _: Option<&str> = request.request_id();
        let _: usize = request.isl_tokens();
        let _: Option<&[u64]> = request.sequence_hashes();
        let _: &RequestProgress = request.progress();
        let _: Option<tokio::time::Instant> = request.due_at();
        Box::pin(async move { Ok(request) })
    }
}

fn prelude_provider(
    _parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    Ok(Arc::new(|config, worker_type, _partition| {
        WorkerSelectionPolicy::new_with_filters(
            config.clone(),
            worker_type.as_str(),
            vec![Box::new(PreludePolicy)],
            vec![Box::new(PreludePolicy)],
            Box::new(PreludePolicy),
        )
    }))
}

#[test]
fn prelude_plugins_resolve_through_the_registry() {
    let _: Box<dyn WorkerFilter> = Box::new(PreludePolicy);
    let _: Box<dyn WorkerScorer> = Box::new(PreludePolicy);
    let _: Box<dyn WorkerPicker> = Box::new(PreludePolicy);
    let _: Box<dyn RequestClassifier> = Box::new(PreludeClassifier);

    let mut registry = RouterPluginRegistry::default();
    registry
        .register_worker_selection("prelude", Arc::new(prelude_provider))
        .unwrap();
    let registry: RouterPluginRegistry = registry;
    let policy = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(policy.path(), "worker_selection:\n  aggregated: prelude\n  prefill: prelude\n  decode: prelude\n  encode: prelude\n").unwrap();
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
fn active_requests_pickers_select_through_the_prelude_hosted_api() {
    #[allow(deprecated)]
    let alias = WorkerInputs::OCCUPANCY;
    assert_eq!(alias, WorkerInputs::ACTIVE_REQUESTS);

    let Ok(picker) = WorkerSelectionPolicy::new(
        SelectionConfig::default(),
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
        .select(&[2, 4, 6], Some(&counts), &HostedSelectionRequest::new(32))
        .unwrap();
    assert_eq!(selected, 4);
    let _: Result<WorkerId, SelectionFailure> =
        picker.select(&[], None, &HostedSelectionRequest::new(0));
}
