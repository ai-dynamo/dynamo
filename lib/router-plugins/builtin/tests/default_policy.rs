// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod support;
use dynamo_custom_policy_builtin::{DefaultWorkerSelector, default_policy, default_registry};
use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::{
    KvRouterConfig, RoutingPartitionRef, WorkerInputView, WorkerInputs, WorkerPicker,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelector,
    WorkerType,
};
use support::*;

#[test]
fn seeded_selection_matches_reference_across_cache_and_load_shapes() {
    for temperature in [0.0, 0.7] {
        for prompt in [1, 17, 127, 2048] {
            for mode in 0..32 {
                let (workers, mut request) = fixture(16, prompt);
                let config = KvRouterConfig {
                    router_temperature: temperature,
                    overlap_score_credit_decay: 0.6,
                    host_cache_hit_weight: 0.25,
                    disk_cache_hit_weight: 0.1,
                    decode_active_request_weight: if mode & 8 == 0 { 0.0 } else { 0.7 },
                    shared_cache_multiplier: if mode & 16 == 0 { 0.0 } else { 0.6 },
                    ..Default::default()
                };
                if mode % 8 >= 4 {
                    request.shared_cache_hits =
                        Some(dynamo_kv_router::SharedCacheHits::from_ranges(vec![
                            1..3,
                            5..12,
                        ]));
                }
                match mode % 4 {
                    1 => request.overlap.tier_overlap_blocks = Default::default(),
                    2 => request.worker_loads.clear(),
                    3 => request.track_prefill_tokens = false,
                    _ => {}
                }
                let reference = dynamo_kv_router::DefaultWorkerSelector::new_seeded(
                    Some(config.clone()),
                    "test",
                    42,
                );
                let plugin = DefaultWorkerSelector::new_seeded(Some(config), "test", 42);
                for _ in 0..64 {
                    let input = support::selection_input(&workers, &request, 16);
                    let expected = reference.select_worker(input).unwrap();
                    let actual = plugin.select_worker(input).unwrap();
                    assert_eq!(
                        actual.worker, expected.worker,
                        "temperature={temperature} prompt={prompt} mode={mode}"
                    );
                    assert_eq!(actual.cached_tokens, expected.cached_tokens);
                    assert_eq!(
                        actual.potential_decode_blocks,
                        expected.potential_decode_blocks
                    );
                }
            }
        }
    }
}

#[test]
fn unseeded_sampling_matches_reference_with_the_same_random_draw() {
    for count in [1, 8, 64] {
        for temperature in [0.1, 0.7, 1.0, 2.0] {
            for equal_costs in [false, true] {
                let (workers, mut request) = fixture(count, 2048);
                if equal_costs {
                    request.worker_loads.clear();
                    request.overlap = Default::default();
                }
                let config = KvRouterConfig {
                    router_temperature: temperature,
                    ..Default::default()
                };
                let reference =
                    dynamo_kv_router::DefaultWorkerSelector::new(Some(config.clone()), "prefill");
                let plugin = default_policy(config, "prefill");
                for seed in 0..64 {
                    let input = selection_input(&workers, &request, 16);
                    fastrand::seed(seed);
                    let expected = reference.select_worker(input).unwrap();
                    let next_random = fastrand::u64(..);
                    fastrand::seed(seed);
                    let actual = plugin.select_worker(input).unwrap();
                    assert_eq!(actual.worker, expected.worker);
                    assert_eq!(actual.cached_tokens, expected.cached_tokens);
                    assert_eq!(fastrand::u64(..), next_random);
                }
            }
        }
    }
}

#[test]
fn unseeded_minimum_picker_only_selects_workers_tied_for_lowest_cost() {
    let (workers, mut request) = fixture(8, 17);
    let policy = default_policy(
        KvRouterConfig {
            overlap_score_credit: 0.0,
            prefill_load_scale: 0.0,
            decode_active_request_weight: 1.0,
            ..Default::default()
        },
        "prefill",
    );
    for best in [[0, 1], [2, 5], [6, 7]] {
        for (worker, load) in &mut request.worker_loads {
            load.active_requests = if best.contains(&worker.worker_id) {
                0
            } else {
                10
            };
            load.active_decode_blocks = 0;
            load.additional_active_blocks = 0;
        }
        for _ in 0..32 {
            let selected = policy
                .select_worker(support::selection_input(&workers, &request, 16))
                .unwrap();
            assert!(best.contains(&selected.worker.worker_id));
        }
    }
}

#[test]
fn prepared_values_follow_each_request_when_reusing_a_policy() {
    for label in ["prefill", "decode"] {
        for temperature in [0.0, 0.7] {
            let config = KvRouterConfig {
                router_temperature: temperature,
                overlap_score_credit_decay: 0.6,
                host_cache_hit_weight: 0.25,
                disk_cache_hit_weight: 0.1,
                ..Default::default()
            };
            let reference = dynamo_kv_router::DefaultWorkerSelector::new_seeded(
                Some(config.clone()),
                label,
                42,
            );
            let plugin = DefaultWorkerSelector::new_seeded(Some(config), label, 42);
            for round in 0..64 {
                let prompt = [1, 17, 127, 2048][round % 4];
                let block_size = [8, 16, 32, 17, 3][round % 5];
                let (workers, mut request) = fixture(16, prompt);
                request.track_prefill_tokens = round % 2 == 0;
                if round % 3 == 0 {
                    request.overlap.tier_overlap_blocks = Default::default();
                }
                if round % 5 == 0 {
                    request.worker_loads.clear();
                } else {
                    for load in request.worker_loads.values_mut() {
                        load.active_prefill_tokens += round * 19;
                    }
                }
                let input = support::selection_input(&workers, &request, block_size);
                let expected = reference.select_worker(input).unwrap();
                let actual = plugin.select_worker(input).unwrap();
                assert_eq!(
                    actual.worker, expected.worker,
                    "label={label} temperature={temperature} round={round}"
                );
                assert_eq!(actual.cached_tokens, expected.cached_tokens);
                assert_eq!(
                    actual.potential_decode_blocks,
                    expected.potential_decode_blocks
                );
            }
        }
    }
}

#[test]
fn exact_prompt_and_accounting_inputs_are_available_to_external_pickers() {
    struct Inspect;
    impl WorkerPicker for Inspect {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::CACHE | WorkerInputs::LOAD
        }
        fn pick(
            &mut self,
            context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            assert_eq!(context.prompt_tokens(), 17);
            assert_eq!(context.request_blocks(), 2);
            assert!(input.load().unwrap().iter().all(|load| load.is_available()));
            for cache in input.cache().unwrap().iter() {
                assert!(cache.has_tier_matches());
                let (blocks, tokens) = cache.accounting_cache_estimate();
                assert_eq!((blocks * 16.0) as usize, tokens);
            }
            Ok(0)
        }
    }
    let (workers, mut request) = fixture(2, 17);
    // Tier availability belongs to the lookup, even when its only entry is ineligible.
    request.overlap.tier_overlap_blocks = Default::default();
    request
        .overlap
        .tier_overlap_blocks
        .disk
        .insert(WorkerWithDpRank::from_worker_id(999), 1);
    WorkerSelectionPolicy::new(KvRouterConfig::default(), "test", vec![], Box::new(Inspect))
        .select_worker(support::selection_input(&workers, &request, 16))
        .unwrap();
}

#[test]
fn configured_default_resolves_for_every_role_and_uses_exclusive_affinity() {
    let registry = default_registry();
    for conditional_disagg_enabled in [false, true] {
        for role in [
            WorkerType::Aggregated,
            WorkerType::Prefill,
            WorkerType::Decode,
            WorkerType::Encode,
        ] {
            let config = KvRouterConfig {
                conditional_disagg_enabled,
                ..Default::default()
            };
            let factory = registry
                .resolve_for_worker_type(&config, role)
                .unwrap()
                .unwrap();
            let policy = factory(&config, role, RoutingPartitionRef::new("model", "default"));
            let inputs =
                <WorkerSelectionPolicy as WorkerSelector<TestWorker>>::required_worker_inputs(
                    &policy,
                );
            assert_eq!(
                inputs.contains(WorkerInputs::CACHE),
                role != WorkerType::Decode || conditional_disagg_enabled
            );
            assert!(
                <WorkerSelectionPolicy as WorkerSelector<TestWorker>>::uses_exclusive_affinity_target(
                    &policy
                )
            );
        }
    }
}

#[test]
fn mandatory_pin_is_preserved() {
    let (workers, mut request) = fixture(4, 17);
    request.pinned_worker = Some(WorkerWithDpRank::new(3, 1));
    let selected = default_policy(KvRouterConfig::default(), "test")
        .select_worker(support::selection_input(&workers, &request, 16))
        .unwrap();
    assert_eq!(Some(selected.worker), request.pinned_worker);
}

#[test]
fn configured_parameters_replace_request_score_overrides() {
    use dynamo_kv_router::RouterConfigOverride;
    let (workers, mut request) = fixture(2, 160);
    for (worker, load) in &mut request.worker_loads {
        load.active_prefill_tokens = 0;
        load.active_requests = 0;
        load.active_decode_blocks = if worker.worker_id == 0 { 8 } else { 0 };
        load.additional_active_blocks = 0;
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(*worker, if worker.worker_id == 0 { 10 } else { 0 });
    }
    request.router_config_override = Some(RouterConfigOverride {
        overlap_score_credit: Some(0.0),
        router_temperature: Some(10.0),
        ..Default::default()
    });
    let config = KvRouterConfig {
        overlap_score_credit: 2.0,
        host_cache_hit_weight: 0.0,
        disk_cache_hit_weight: 0.0,
        router_temperature: 0.0,
        ..Default::default()
    };
    let policy = default_policy(config, "prefill");
    for _ in 0..32 {
        let result = policy
            .select_worker(support::selection_input(&workers, &request, 16))
            .unwrap();
        assert_eq!(result.worker.worker_id, 0);
    }
}

#[test]
fn named_default_parameters_resolve_and_reject_invalid_values() {
    use std::io::Write;
    for (parameter, succeeds) in [
        ("overlap_score_credit: 2.0", true),
        ("overlap_score_credit: -1.0", false),
        ("unknown: 1", false),
    ] {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        write!(file, "worker_selection:\n  aggregated: tuned\n  instances:\n    - name: tuned\n      type: dynamo-default-cost-fn\n      parameters:\n        {parameter}\n").unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(file.path().display().to_string()),
            ..Default::default()
        };
        let mut registry = default_registry();
        dynamo_custom_policy_builtin::register(&mut registry).unwrap();
        assert_eq!(registry.resolve(&config).is_ok(), succeeds);
    }
}

#[test]
fn pin_does_not_advance_seeded_random_stream() {
    let (workers, mut request) = fixture(8, 127);
    let config = KvRouterConfig {
        router_temperature: 0.7,
        ..Default::default()
    };
    let reference =
        dynamo_kv_router::DefaultWorkerSelector::new_seeded(Some(config.clone()), "prefill", 42);
    let plugin = DefaultWorkerSelector::new_seeded(Some(config), "prefill", 42);
    for pinned in [true, false, true, false, false] {
        request.pinned_worker = pinned.then_some(WorkerWithDpRank::new(3, 1));
        let input = support::selection_input(&workers, &request, 16);
        assert_eq!(
            reference.select_worker(input).unwrap().worker,
            plugin.select_worker(input).unwrap().worker
        );
    }
}
