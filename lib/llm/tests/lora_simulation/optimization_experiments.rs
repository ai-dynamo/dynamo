// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::{LazyLock, Mutex};

/// The controller's LoRA gauges are process-global and intentionally support one controller per
/// process. Rust runs integration tests concurrently, so serialize the experiments that read the
/// overflow gauge or they would observe each other's controller ticks.
static EXPERIMENT_GAUGE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[derive(Debug, Clone, PartialEq)]
struct ControllerExperimentMetrics {
    total_churn: usize,
    target_additions: usize,
    target_removals: usize,
    peak_targets: usize,
    mean_targets: f64,
    ticks_over_resident_capacity: usize,
    overflow_total: usize,
}

fn run_controller_experiment(
    sim_config: &SimConfig,
    schedules: &[LoraLoadSchedule],
    alloc_config: LoraAllocationConfig,
) -> ControllerExperimentMetrics {
    let is_mcf = matches!(alloc_config.algorithm, AllocationAlgorithmType::MinCostFlow);
    let routing_table = LoraRoutingTable::new();
    let state_tracker = LoraStateTracker::new();
    let load_estimator = Arc::new(LoadEstimator::new());
    let mut controller = LoraController::new(
        alloc_config,
        routing_table.clone(),
        state_tracker.clone(),
        load_estimator.clone(),
    );

    for worker_id in 0..sim_config.num_backends {
        state_tracker.set_worker_capacity(
            WorkerWithDpRank::new(worker_id as u64, 0),
            sim_config.slots_per_backend as u32,
        );
    }

    let resident_capacity = sim_config.num_backends * sim_config.slots_per_backend;
    let mut previous = AllocationSnapshot::new();
    let mut target_additions = 0;
    let mut target_removals = 0;
    let mut target_sum = 0;
    let mut peak_targets = 0;
    let mut ticks_over_resident_capacity = 0;
    let mut overflow_total = 0;

    for tick in 0..sim_config.total_ticks {
        for name in load_estimator
            .get_current_load()
            .keys()
            .cloned()
            .collect::<Vec<_>>()
        {
            load_estimator.remove_lora(&name);
        }
        for schedule in schedules {
            for _ in 0..schedule.load_at_tick(tick) {
                load_estimator.increment_load(&schedule.lora_name);
            }
        }

        controller.recompute_now();

        let current: AllocationSnapshot = routing_table
            .snapshot_configs()
            .into_iter()
            .filter_map(|(name, config)| {
                (!config.replica_set.is_empty()).then_some((name, config.replica_set))
            })
            .collect();
        let (additions, removals) = compute_churn(&previous, &current);
        target_additions += additions;
        target_removals += removals;

        let target_count: usize = current.values().map(Vec::len).sum();
        target_sum += target_count;
        peak_targets = peak_targets.max(target_count);
        ticks_over_resident_capacity += usize::from(target_count > resident_capacity);
        if is_mcf {
            let overflow = dynamo_llm::http::service::metrics::LORA_OVERFLOW_COUNT_GAUGE
                .get()
                .max(0) as usize;
            overflow_total += overflow;
        }

        previous = current;
    }

    ControllerExperimentMetrics {
        total_churn: target_additions + target_removals,
        target_additions,
        target_removals,
        peak_targets,
        mean_targets: target_sum as f64 / sim_config.total_ticks.max(1) as f64,
        ticks_over_resident_capacity,
        overflow_total,
    }
}

fn mcf_config(gamma_load: i64, beta_keep: i64) -> LoraAllocationConfig {
    let mut config = LoraAllocationConfig {
        algorithm: AllocationAlgorithmType::MinCostFlow,
        timestep_secs: 1,
        scale_down_cooldown_ticks: 0,
        ..Default::default()
    };
    config.mcf.gamma_load = gamma_load;
    config.mcf.beta_keep = beta_keep;
    config
}

fn oscillating_schedules(total_ticks: usize) -> Vec<LoraLoadSchedule> {
    let alternating = |even, odd| {
        (0..total_ticks)
            .map(|tick| if tick % 2 == 0 { even } else { odd })
            .collect::<Vec<_>>()
    };
    [
        ("lora-a", alternating(30, 1)),
        ("lora-b", alternating(1, 30)),
        ("lora-c", vec![5; total_ticks]),
        ("lora-d", vec![5; total_ticks]),
    ]
    .into_iter()
    .map(|(name, per_tick_loads)| LoraLoadSchedule {
        lora_name: name.to_string(),
        active_window: (0, total_ticks),
        peak_load: 30,
        ramp_up: 0,
        steady: total_ticks,
        ramp_down: 0,
        per_tick_loads: Some(per_tick_loads),
    })
    .collect()
}

#[test]
#[ignore = "deterministic parameter sweep; run explicitly with --ignored --nocapture"]
fn test_mcf_churn_incentive_sweep() {
    let _gauge_guard = EXPERIMENT_GAUGE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // With N workers, the greatest possible HRW rank improvement is alpha * (N - 1).
    // Once gamma_load + beta_keep exceeds that bound, a feasible prior placement should never
    // move solely for HRW preference. This sweep checks the threshold and whether the much larger
    // production default buys any further churn reduction in a bounded L <= N*K workload.
    let config = SimConfig {
        num_backends: 8,
        slots_per_backend: 4,
        total_loras: 32,
        total_ticks: 100,
        seed: 42,
        ..Default::default()
    };
    assert!(config.total_loras <= config.num_backends * config.slots_per_backend);
    let schedules = generate_zipf_poisson_schedules(32, 100, 1.0, 40.0, 42);
    let settings = [
        ("none", 0, 0),
        ("below-bound", 3, 2),
        ("above-bound", 8, 0),
        ("default", 1000, 250),
    ];
    let mut results = Vec::new();

    println!(
        "label,gamma_load,beta_keep,total_churn,adds,removes,mean_targets,peak_targets,overflow_total"
    );
    for (label, gamma_load, beta_keep) in settings {
        let metrics =
            run_controller_experiment(&config, &schedules, mcf_config(gamma_load, beta_keep));
        println!(
            "{label},{gamma_load},{beta_keep},{},{},{},{:.3},{},{}",
            metrics.total_churn,
            metrics.target_additions,
            metrics.target_removals,
            metrics.mean_targets,
            metrics.peak_targets,
            metrics.overflow_total,
        );
        assert_eq!(metrics.overflow_total, 0, "{label} unexpectedly overflowed");
        results.push((label, metrics));
    }

    let none = &results[0].1;
    let above_bound = &results[2].1;
    let default = &results[3].1;
    assert!(
        above_bound.total_churn <= none.total_churn,
        "a sufficient keep/load incentive should not increase churn"
    );
    assert_eq!(
        above_bound, default,
        "weights above the maximum HRW-rank advantage should be behaviorally equivalent"
    );
}

#[test]
#[ignore = "deterministic parameter sweep; run explicitly with --ignored --nocapture"]
fn test_scale_down_cooldown_churn_pressure_sweep() {
    let _gauge_guard = EXPERIMENT_GAUGE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // Alternating hot adapters repeatedly ask the controller to scale replicas down and back up.
    // Cooldown should suppress that oscillation, at the cost of retaining more route targets and
    // potentially introducing soft pressure or MCF overflow when retained demand exceeds N*K.
    let total_ticks = 40;
    let config = SimConfig {
        num_backends: 4,
        slots_per_backend: 2,
        total_loras: 4,
        total_ticks,
        seed: 42,
        ..Default::default()
    };
    let schedules = oscillating_schedules(total_ticks);

    println!(
        "algorithm,cooldown,total_churn,adds,removes,mean_targets,peak_targets,ticks_over_capacity,overflow_total"
    );
    for algorithm in [
        AllocationAlgorithmType::Hrw,
        AllocationAlgorithmType::MinCostFlow,
    ] {
        let mut results = Vec::new();
        for cooldown in [0, 1, 3, 5] {
            let alloc_config = LoraAllocationConfig {
                algorithm,
                timestep_secs: 1,
                scale_down_cooldown_ticks: cooldown,
                ..Default::default()
            };
            let metrics = run_controller_experiment(&config, &schedules, alloc_config);
            println!(
                "{algorithm:?},{cooldown},{},{},{},{:.3},{},{},{}",
                metrics.total_churn,
                metrics.target_additions,
                metrics.target_removals,
                metrics.mean_targets,
                metrics.peak_targets,
                metrics.ticks_over_resident_capacity,
                metrics.overflow_total,
            );
            results.push((cooldown, metrics));
        }

        // Re-run the endpoints to make seed/config determinism an explicit experimental gate.
        for index in [0, results.len() - 1] {
            let (cooldown, expected) = &results[index];
            let actual = run_controller_experiment(
                &config,
                &schedules,
                LoraAllocationConfig {
                    algorithm,
                    timestep_secs: 1,
                    scale_down_cooldown_ticks: *cooldown,
                    ..Default::default()
                },
            );
            assert_eq!(&actual, expected);
        }

        let immediate = &results[0].1;
        let hysteretic = &results[1].1;
        assert!(
            hysteretic.total_churn < immediate.total_churn,
            "cooldown should suppress alternating scale-down/scale-up churn"
        );
        assert!(
            hysteretic.mean_targets > immediate.mean_targets,
            "lower churn should come with retained route-target pressure"
        );
        assert_eq!(
            results[1].1, results[2].1,
            "this every-other-tick workload re-arms any positive cooldown identically"
        );
        assert_eq!(results[2].1, results[3].1);
        match algorithm {
            AllocationAlgorithmType::Hrw => assert!(
                hysteretic.ticks_over_resident_capacity > 0,
                "HRW should expose the soft route-target pressure tradeoff"
            ),
            AllocationAlgorithmType::MinCostFlow => assert!(
                hysteretic.overflow_total > 0,
                "MCF should expose retained-demand overflow instead of hiding it"
            ),
            AllocationAlgorithmType::Random => unreachable!("not part of this sweep"),
        }
    }
}
