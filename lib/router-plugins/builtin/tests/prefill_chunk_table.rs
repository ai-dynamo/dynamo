// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod support;
use dynamo_kv_router::plugins::prelude::*;
use dynamo_scheduler::host::WorkerSelector;
use std::time::{Duration, Instant};
use support::{fixture, selection_input};

#[test]
fn partial_prefix_can_change_placement_without_waiting_for_a_profile() {
    let dir = tempfile::tempdir().unwrap();
    let profile = dir.path().join("profile.json");
    let yaml = dir.path().join("policy.yaml");
    std::fs::write(&yaml, format!("worker_selection:\n  prefill:\n    type: dynamo-prefill-chunk-table\n    parameters:\n      profile_file: {}\n",
        serde_json::to_string(&profile).unwrap())).unwrap();
    let config = KvRouterConfig {
        router_policy_config: Some(yaml.display().to_string()),
        router_temperature: 0.0,
        prefill_load_scale: 1.0,
        ..Default::default()
    };
    let mut registry = dynamo_router_policies::default_registry();
    dynamo_router_policies::register(&mut registry).unwrap();
    let factory = registry.resolve(&config).unwrap().unwrap();
    let policy = factory(
        &SelectionConfig::from(&config),
        WorkerType::Prefill,
        RoutingPartitionRef::new("model", "default"),
    );
    let (workers, mut request) = fixture(2, 160);
    for (worker, load) in &mut request.worker_loads {
        load.active_prefill_tokens = 0;
        load.active_requests = 0;
        load.additional_active_blocks = 0;
        load.active_decode_blocks = worker.worker_id as usize;
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(*worker, 2);
        request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .insert(*worker, 0);
        request.overlap.tier_overlap_blocks.disk.insert(*worker, 0);
        request.overlap.effective_cached_tokens.insert(*worker, 24);
        request
            .overlap
            .effective_overlap_blocks
            .insert(*worker, 2.0);
    }
    let choose = || {
        policy
            .select_worker(selection_input(&workers, &request, 16))
            .unwrap()
            .worker
            .worker_id
    };
    assert_eq!(choose(), 0);
    let profiles: Vec<_> = (0..4)
        .map(|rank| {
            serde_json::json!({
                "worker_id":rank / 2, "dp_rank":rank % 2, "step_cost_blocks":8,
                "paths":if rank / 2 == 0 { vec![vec![32,64,96,160]] } else { vec![vec![32,160]] }
            })
        })
        .collect();
    std::fs::write(
        profile,
        serde_json::to_vec(&serde_json::json!({"profiles":profiles,"model_name":"model","routing_group":"default","block_size":16, "valid_until_unix_ms":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64 + 30000})).unwrap(),
    )
    .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while choose() == 0 {
        assert!(
            Instant::now() < deadline,
            "the published path should change the chosen rank"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    // 48 tokens is not a recorded boundary on either rank.
    for prefix in request.overlap.tier_overlap_blocks.device.values_mut() {
        *prefix = 3;
    }
    assert_eq!(
        policy
            .select_worker(selection_input(&workers, &request, 16))
            .unwrap()
            .worker
            .worker_id,
        0
    );
}
