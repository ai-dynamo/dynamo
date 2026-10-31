// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A worker-selection policy and a request classifier written against
//! `dynamo_kv_router::plugins::prelude` alone: no other path from this crate, no other
//! router crate. `plugin_api_compat.rs` keeps the pre-prelude paths compiling; this file is
//! the surface new plugins target.

use std::sync::Arc;

use dynamo_kv_router::plugins::prelude::*;

struct PreludePolicy;

impl WorkerFilter for PreludePolicy {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE
    }

    fn keep(
        &mut self,
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
        &mut self,
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
        &mut self,
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

    let mut registry = WorkerSelectionPolicyRegistry::default();
    registry
        .register("prelude", Arc::new(prelude_provider))
        .unwrap();
    let registry: RouterPluginRegistry = registry;
    let policy = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(policy.path(), "worker_selection:\n  aggregated: prelude\n  prefill: prelude\n  decode: prelude\n  encode: prelude\n  instances:\n    - name: prelude\n      type: prelude\n").unwrap();
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
