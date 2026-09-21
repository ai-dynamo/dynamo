// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Includes load projection, selection, booking, and release in the shared core.
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dynamo_custom_policy_builtin::default_factory;
use dynamo_kv_router::services::selection::{
    PromptRequest, SelectAndReserveRequest, SelectionCacheConfig, SelectionCore, WorkerRequest,
};
use dynamo_kv_router::{KvRouterConfig, WorkerSelectionPolicy, WorkerSelectionPolicyFactory};
use std::sync::Arc;
use std::time::Duration;

fn bench(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let _guard = runtime.enter();
    let mut group = c.benchmark_group("default_lifecycle");
    group
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(1))
        .sample_size(30)
        .nresamples(1_000);
    for count in [8, 64, 512] {
        let reference: WorkerSelectionPolicyFactory = Arc::new(|config, role, _| {
            WorkerSelectionPolicy::reference(config.clone(), role.default_selector_label())
        });
        for (label, factory) in [("reference", reference), ("plugin", default_factory())] {
            let core = SelectionCore::try_new_local(
                KvRouterConfig {
                    use_kv_events: false,
                    router_queue_threshold: None,
                    ..Default::default()
                },
                1,
                Default::default(),
                SelectionCacheConfig::default(),
                factory,
            )
            .unwrap();
            runtime.block_on(async {
                for worker_id in 0..count {
                    core.upsert_worker(WorkerRequest {
                        worker_id,
                        endpoint: Some(format!("http://worker-{worker_id}:8000")),
                        block_size: Some(16),
                        max_num_batched_tokens: Some(8192),
                        ..Default::default()
                    })
                    .await
                    .unwrap();
                }
            });
            group.bench_function(BenchmarkId::new(label, count), |b| {
                b.iter(|| {
                    runtime.block_on(async {
                        let response = core
                            .select_and_reserve(SelectAndReserveRequest {
                                selection_id: Some("request".into()),
                                prompt: PromptRequest {
                                    token_ids: Some((0..2048).collect()),
                                    ..Default::default()
                                },
                                model_name: "default".into(),
                                routing_group: "default".into(),
                                router_config_override: None,
                                expected_output_tokens: None,
                                priority_jump: None,
                                strict_priority: None,
                                session_id: None,
                                session_context: None,
                                affinity_target: None,
                                pinned_worker: None,
                                allowed_worker_ids: None,
                                routing_constraints: Default::default(),
                            })
                            .await
                            .unwrap();
                        criterion::black_box(response);
                        core.free_reservation("request").await.unwrap();
                    })
                });
            });
            core.shutdown();
        }
    }
    group.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);
