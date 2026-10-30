// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile legacy plugin import paths against the canonical registry and contracts.
//! Scorer implementations use the current batch signature; old scalar signatures must migrate.
//! TODO(v1.7): Remove the legacy import coverage when the compatibility exports are removed.

use std::sync::Arc;

use dynamo_kv_router::plugins::{RouterPluginRegistry, request_classifier, worker_selection};
use dynamo_kv_router::scheduling::{ClassifyFuture, ClassifyRequest, RequestClassifier};
use dynamo_kv_router::selector::{HostedSelectionRequest, HostedWorkerPicker};
use dynamo_kv_router::{
    KvRouterConfig, RoutingPartitionRef, WorkerCandidate, WorkerCandidates, WorkerFilter,
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerType,
};

struct LegacyPolicy;

impl WorkerFilter for LegacyPolicy {
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

impl WorkerScorer for LegacyPolicy {
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

impl WorkerPicker for LegacyPolicy {
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

struct LegacyClassifier;

impl RequestClassifier for LegacyClassifier {
    fn classify(
        &mut self,
        request: ClassifyRequest,
        _workers: WorkerCandidates<'_>,
    ) -> ClassifyFuture {
        let _: Option<&str> = request.request_id();
        let _: usize = request.isl_tokens();
        let _: Option<&[u64]> = request.sequence_hashes();
        let _: &dynamo_kv_router::scheduling::RequestProgress = request.progress();
        let _: &request_classifier::RequestProgress = request.progress();
        let _: Option<tokio::time::Instant> = request.due_at();
        Box::pin(async move { Ok(request) })
    }
}

fn legacy_provider(
    _parameters: &dynamo_kv_router::plugins::WorkerSelectionPolicyParameters,
) -> Result<
    dynamo_kv_router::WorkerSelectionPolicyFactory,
    dynamo_kv_router::plugins::WorkerSelectionPolicyProviderError,
> {
    Ok(Arc::new(|config, worker_type, _partition| {
        WorkerSelectionPolicy::new_with_filters(
            config.clone(),
            worker_type.as_str(),
            vec![Box::new(LegacyPolicy)],
            vec![Box::new(LegacyPolicy)],
            Box::new(LegacyPolicy),
        )
    }))
}

#[test]
fn legacy_plugins_resolve_through_the_common_registry() {
    // These assignments require identity, not wrappers around the original traits.
    let _: Box<dyn worker_selection::WorkerFilter> = Box::new(LegacyPolicy);
    let _: Box<dyn worker_selection::WorkerScorer> = Box::new(LegacyPolicy);
    let _: Box<dyn worker_selection::WorkerPicker> = Box::new(LegacyPolicy);
    let _: Box<dyn request_classifier::RequestClassifier> = Box::new(LegacyClassifier);

    let mut registry = dynamo_kv_router::plugins::WorkerSelectionPolicyRegistry::default();
    registry
        .register("legacy", Arc::new(legacy_provider))
        .unwrap();
    let registry: RouterPluginRegistry = registry;
    let policy = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(policy.path(), "worker_selection:\n  aggregated: legacy\n  prefill: legacy\n  decode: legacy\n  encode: legacy\n  instances:\n    - name: legacy\n      type: legacy\n").unwrap();
    let config = KvRouterConfig {
        router_policy_config: Some(policy.path().display().to_string()),
        ..Default::default()
    };
    let plugins = registry.resolve_plugins(&config).unwrap();
    let factory = plugins.worker_selection().unwrap();
    let _policy: worker_selection::WorkerSelectionPolicy = factory(
        &config,
        WorkerType::Aggregated,
        RoutingPartitionRef::new("model", "default"),
    );
}

#[cfg(feature = "standalone-selection")]
#[test]
fn legacy_service_catalog_registers_into_the_common_registry() {
    use dynamo_kv_router::services::selection::{
        WorkerSelectionPolicyProvider, WorkerSelectionPolicyRegistry,
    };

    let mut registry = RouterPluginRegistry::default();
    let legacy_registry: &mut WorkerSelectionPolicyRegistry = &mut registry;
    let provider: WorkerSelectionPolicyProvider = Arc::new(legacy_provider);
    legacy_registry.register("legacy", provider).unwrap();
    assert!(!registry.is_empty());
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
