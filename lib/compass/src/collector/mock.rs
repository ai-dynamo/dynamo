// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use chrono::{Duration, Utc};
use rand::Rng;

use crate::engine::{EngineInput, FloorInput, QueueSnapshot, SpanNode};
use crate::engine::floor::FloorFormula;
use crate::types::*;

pub fn generate_mock_input(deployment_name: &str, window_minutes: i64) -> EngineInput {
    let now = Utc::now();
    let start = now - Duration::minutes(window_minutes);
    let mut rng = rand::rng();

    let traces = generate_mock_traces(&mut rng, 200);
    let queue_snapshots = generate_mock_queue_snapshots(&mut rng, window_minutes);
    let floor_inputs = generate_mock_floor_inputs();

    EngineInput {
        window: TimeWindow {
            start,
            end: now,
        },
        deployment: DeploymentInfo {
            name: deployment_name.to_string(),
            engine: "vllm".to_string(),
            topology: "disaggregated".to_string(),
        },
        workload: WorkloadSummary {
            qps: 12.3 + rng.random_range(-2.0..2.0),
            isl_p50: 2048,
            osl_p50: 256,
        },
        end_to_end: EndToEndMetrics {
            ttft_ms: PercentileSet {
                p50: 142.0 + rng.random_range(-10.0..10.0),
                p95: 380.0 + rng.random_range(-20.0..20.0),
                p99: 620.0 + rng.random_range(-30.0..30.0),
            },
            itl_ms: PercentileSet {
                p50: 18.0 + rng.random_range(-2.0..2.0),
                p95: 31.0 + rng.random_range(-3.0..3.0),
                p99: 47.0 + rng.random_range(-5.0..5.0),
            },
        },
        traces,
        queue_snapshots,
        floor_inputs,
        sub_component_breakdowns: generate_mock_breakdowns(&mut rng),
    }
}

fn generate_mock_traces(rng: &mut impl Rng, count: usize) -> Vec<Vec<SpanNode>> {
    (0..count)
        .map(|_| {
            vec![SpanNode {
                component: "request".to_string(),
                sub_component: None,
                duration_ms: 0.5 + rng.random_range(0.0..0.5),
                children: vec![
                    SpanNode {
                        component: "router".to_string(),
                        sub_component: Some("dispatch".to_string()),
                        duration_ms: 0.3 + rng.random_range(0.0..0.8),
                        children: vec![],
                    },
                    SpanNode {
                        component: "kvbm".to_string(),
                        sub_component: Some("allocate".to_string()),
                        duration_ms: 2.0 + rng.random_range(0.0..4.0),
                        children: vec![],
                    },
                    SpanNode {
                        component: "nixl".to_string(),
                        sub_component: Some("transfer.h2d".to_string()),
                        duration_ms: 1.5 + rng.random_range(0.0..2.0),
                        children: vec![],
                    },
                    SpanNode {
                        component: "prefill".to_string(),
                        sub_component: Some("compute".to_string()),
                        duration_ms: 10.0 + rng.random_range(0.0..4.0),
                        children: vec![],
                    },
                ],
            }]
        })
        .collect()
}

fn generate_mock_queue_snapshots(rng: &mut impl Rng, window_minutes: i64) -> Vec<QueueSnapshot> {
    let points = (window_minutes * 4) as usize;
    let timestamps: Vec<f64> = (0..points).map(|i| i as f64 * 15.0).collect();

    vec![
        QueueSnapshot {
            component: "kvbm.radix_tree_lock".to_string(),
            timestamps: timestamps.clone(),
            queue_lengths: (0..points)
                .map(|i| 7.0 + (i as f64 * 0.08) + rng.random_range(-0.5..0.5))
                .collect(),
            arrival_rates: vec![10.0 + rng.random_range(-1.0..1.0); points],
            capacity: 10.0,
        },
        QueueSnapshot {
            component: "nixl.h2d_channel".to_string(),
            timestamps: timestamps.clone(),
            queue_lengths: (0..points)
                .map(|_| 4.0 + rng.random_range(-1.0..1.0))
                .collect(),
            arrival_rates: vec![8.0; points],
            capacity: 10.0,
        },
        QueueSnapshot {
            component: "prefill.gpu".to_string(),
            timestamps,
            queue_lengths: (0..points)
                .map(|_| 5.5 + rng.random_range(-0.5..0.5))
                .collect(),
            arrival_rates: vec![10.0; points],
            capacity: 8.0,
        },
    ]
}

fn generate_mock_floor_inputs() -> Vec<FloorInput> {
    vec![
        FloorInput {
            component: "kvbm.allocate".to_string(),
            observed_p99_ms: 3.51,
            formula: FloorFormula::KvbmAllocate {
                hash_time_ms: 0.08,
                radix_depth: 12,
            },
        },
        FloorInput {
            component: "prefill.compute".to_string(),
            observed_p99_ms: 12.10,
            formula: FloorFormula::Custom { floor_ms: 11.90 },
        },
        FloorInput {
            component: "nixl.transfer.h2d".to_string(),
            observed_p99_ms: 2.40,
            formula: FloorFormula::NixlTransfer {
                block_size_bytes: 2 * 1024 * 1024,
                bandwidth_gbps: 200.0,
            },
        },
        FloorInput {
            component: "router.dispatch".to_string(),
            observed_p99_ms: 0.84,
            formula: FloorFormula::Custom { floor_ms: 0.30 },
        },
    ]
}

fn generate_mock_breakdowns(rng: &mut impl Rng) -> Vec<SubComponentBreakdown> {
    vec![
        SubComponentBreakdown {
            component: "kvbm".to_string(),
            phases: vec![
                PhaseMetrics {
                    name: "hash".to_string(),
                    p99_ms: 0.12 + rng.random_range(-0.02..0.02),
                    on_cpu_ms: None,
                    lock_wait_ms: None,
                },
                PhaseMetrics {
                    name: "lookup".to_string(),
                    p99_ms: 0.84 + rng.random_range(-0.1..0.1),
                    on_cpu_ms: None,
                    lock_wait_ms: None,
                },
                PhaseMetrics {
                    name: "allocate".to_string(),
                    p99_ms: 3.51 + rng.random_range(-0.3..0.3),
                    on_cpu_ms: Some(1.20 + rng.random_range(-0.1..0.1)),
                    lock_wait_ms: Some(2.31 + rng.random_range(-0.2..0.2)),
                },
                PhaseMetrics {
                    name: "evict".to_string(),
                    p99_ms: 0.43 + rng.random_range(-0.05..0.05),
                    on_cpu_ms: None,
                    lock_wait_ms: None,
                },
                PhaseMetrics {
                    name: "return".to_string(),
                    p99_ms: 0.09 + rng.random_range(-0.01..0.01),
                    on_cpu_ms: None,
                    lock_wait_ms: None,
                },
            ],
            theoretical_floor_ms: Some(0.80),
        },
        SubComponentBreakdown {
            component: "router".to_string(),
            phases: vec![
                PhaseMetrics {
                    name: "cost_compute".to_string(),
                    p99_ms: 0.15,
                    on_cpu_ms: None,
                    lock_wait_ms: None,
                },
                PhaseMetrics {
                    name: "kv_overlap_score".to_string(),
                    p99_ms: 0.42,
                    on_cpu_ms: None,
                    lock_wait_ms: None,
                },
                PhaseMetrics {
                    name: "worker_select".to_string(),
                    p99_ms: 0.18,
                    on_cpu_ms: None,
                    lock_wait_ms: None,
                },
                PhaseMetrics {
                    name: "dispatch".to_string(),
                    p99_ms: 0.09,
                    on_cpu_ms: None,
                    lock_wait_ms: None,
                },
            ],
            theoretical_floor_ms: Some(0.30),
        },
    ]
}
