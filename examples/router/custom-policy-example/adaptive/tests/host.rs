// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_custom_policy_example_adaptive::register;
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyRegistry,
};
use dynamo_kv_router::{
    KvRouterConfig, RoutingPartitionRef, SchedulingRequest, WorkerLoadProjection,
    WorkerSelectionInput, WorkerSelector, WorkerType,
};
use std::collections::{HashMap, HashSet};

struct TestWorker;
impl WorkerConfigLike for TestWorker {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }
    fn data_parallel_size(&self) -> u32 {
        2
    }
    fn max_num_batched_tokens(&self) -> Option<u64> {
        None
    }
    fn total_kv_blocks(&self) -> Option<u64> {
        Some(4096)
    }
}

fn resolve(parameters: &str) -> Result<(KvRouterConfig, WorkerSelectionPolicyFactory), String> {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), format!("worker_selection:\n  aggregated: test\n  prefill: test\n  decode: test\n  instances:\n    - name: test\n      type: adaptive\n      parameters: {parameters}\n")).unwrap();
    let config = KvRouterConfig {
        router_policy_config: Some(file.path().display().to_string()),
        ..Default::default()
    };
    let mut registry = WorkerSelectionPolicyRegistry::default();
    register(&mut registry).unwrap();
    let factory = registry
        .resolve(&config)
        .map_err(|e| e.to_string())?
        .unwrap();
    Ok((config, factory))
}

fn request() -> SchedulingRequest {
    let mut request = SchedulingRequest {
        mode: ScheduleMode::QueryOnly { request_id: None },
        token_seq: None,
        isl_tokens: 160,
        lora_name: None,
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
    };
    for id in [10, 20] {
        for rank in 0..2 {
            let worker = WorkerWithDpRank {
                worker_id: id,
                dp_rank: rank,
            };
            request
                .overlap
                .tier_overlap_blocks
                .device
                .insert(worker, if id == 10 { 10 } else { 0 });
            request.worker_loads.insert(
                worker,
                WorkerLoadProjection {
                    active_requests: usize::from(id == 10),
                    ..Default::default()
                },
            );
        }
    }
    request
}

#[test]
fn yaml_factory_and_real_host_preserve_cache_preference() {
    let (config, factory) = resolve("{seed: 42}").unwrap();
    for role in [WorkerType::Aggregated, WorkerType::Prefill] {
        let policy = factory(&config, role, RoutingPartitionRef::new("model", "group"));
        let request = request();
        let workers = HashMap::from([(10, TestWorker), (20, TestWorker)]);
        let selected = policy
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(selected.worker.worker_id, 10);
    }
}

#[test]
fn decode_does_not_trade_load_for_prefill_cache() {
    let (config, factory) = resolve("{}").unwrap();
    let policy = factory(
        &config,
        WorkerType::Decode,
        RoutingPartitionRef::new("model", "group"),
    );
    let mut request = request();
    request.track_prefill_tokens = false;
    let workers = HashMap::from([(10, TestWorker), (20, TestWorker)]);
    let selected = policy
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap();
    assert_eq!(selected.worker.worker_id, 20);
}

#[test]
fn host_enforces_allowlist_and_exact_dp_pin_before_policy() {
    let (config, factory) = resolve("{}").unwrap();
    let policy = factory(
        &config,
        WorkerType::Aggregated,
        RoutingPartitionRef::new("model", "group"),
    );
    let mut request = request();
    let workers = HashMap::from([(10, TestWorker), (20, TestWorker)]);
    request.allowed_worker_ids = Some(HashSet::from([20]));
    let selected = policy
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap();
    assert_eq!(selected.worker.worker_id, 20);
    request.pinned_worker = Some(WorkerWithDpRank {
        worker_id: 20,
        dp_rank: 1,
    });
    let selected = policy
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap();
    assert_eq!(selected.worker, request.pinned_worker.unwrap());
    request.allowed_worker_ids = Some(HashSet::new());
    assert!(
        policy
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16
            ))
            .is_err()
    );
}

#[test]
fn each_partition_gets_fresh_state() {
    let (config, factory) = resolve("{seed: 42, smoothing: 1.0, max_step: 1.0}").unwrap();
    let a = factory(
        &config,
        WorkerType::Aggregated,
        RoutingPartitionRef::new("a", "group"),
    );
    let b = factory(
        &config,
        WorkerType::Aggregated,
        RoutingPartitionRef::new("b", "group"),
    );
    let mut hot = request();
    for (worker, load) in &mut hot.worker_loads {
        if worker.worker_id == 10 {
            load.active_requests = 1000;
        }
    }
    let workers = HashMap::from([(10, TestWorker), (20, TestWorker)]);
    let selected = a
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &hot,
            hot.eligibility(),
            16,
        ))
        .unwrap();
    assert_eq!(selected.worker.worker_id, 20);
    let cold = request();
    let selected = b
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &cold,
            cold.eligibility(),
            16,
        ))
        .unwrap();
    assert_eq!(selected.worker.worker_id, 10);
}

#[test]
fn invalid_yaml_fails_at_startup() {
    for params in [
        "{algorithm: bandit}",
        "{smoothing: .nan}",
        "{update_interval_ms: 0}",
        "{load_scale: 0}",
        "{unexpected: 1}",
        "{distribution_min: 0.8}",
    ] {
        assert!(resolve(params).is_err(), "{params}");
    }
}
