use super::*;

// ============================================================================
// Test Cases
// ============================================================================

#[test]
fn test_sample_poisson_high_lambda_has_poisson_variance() {
    let mut rng = StdRng::seed_from_u64(1234);
    let samples: Vec<f64> = (0..10_000)
        .map(|_| sample_poisson(&mut rng, 100.0) as f64)
        .collect();
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;

    assert!((mean - 100.0).abs() < 2.0, "mean was {mean}");
    assert!((50.0..150.0).contains(&variance), "variance was {variance}");
}

#[test]
fn test_random_allocator_does_not_overbook_full_workers() {
    let workers = [WorkerWithDpRank::new(0, 0), WorkerWithDpRank::new(1, 0)];
    let worker_slot_usage = workers
        .iter()
        .copied()
        .map(|worker| (worker, (2, 2)))
        .collect();

    let replicas =
        RandomAllocator::new(1234).compute_replica_set("lora-0", &workers, 1, &worker_slot_usage);

    assert!(replicas.is_empty());
}

#[test]
fn test_simulation_small_cluster() {
    // Small cluster: 4 backends, 2 slots each, 6 LoRAs, 3 concurrent
    let config = SimConfig {
        num_backends: 4,
        slots_per_backend: 2,
        total_loras: 6,
        concurrent_loras: 3,
        total_ticks: 40,
        ramp_ticks: 3,
        steady_ticks: 8,
        ramp_down_ticks: 3,
        max_load_per_lora: 10,
        scale_down_cooldown_ticks: 2,
        ..Default::default()
    };

    let schedules = generate_load_schedules(&config);
    print_simulation_header(&config);

    let hrw_metrics = run_hrw_simulation(&config, &schedules);
    let random_metrics = run_random_simulation(&config, &schedules);
    let mcf_metrics = run_mcf_simulation(&config, &schedules);

    print_comparison(&hrw_metrics, &random_metrics, &mcf_metrics);
    print_per_tick_churn(
        &hrw_metrics,
        &random_metrics,
        &mcf_metrics,
        config.total_ticks,
    );

    assert!(
        hrw_metrics.total_churn <= random_metrics.total_churn,
        "HRW ({}) should have <= churn than Random ({})",
        hrw_metrics.total_churn,
        random_metrics.total_churn
    );
    assert!(
        mcf_metrics.total_churn <= random_metrics.total_churn,
        "MCF ({}) should have <= churn than Random ({})",
        mcf_metrics.total_churn,
        random_metrics.total_churn
    );
}

#[test]
fn test_simulation_medium_cluster() {
    // Medium cluster: 8 backends, 4 slots each, 20 LoRAs, 6 concurrent
    let config = SimConfig::default();

    let schedules = generate_load_schedules(&config);
    print_simulation_header(&config);

    let hrw_metrics = run_hrw_simulation(&config, &schedules);
    let random_metrics = run_random_simulation(&config, &schedules);
    let mcf_metrics = run_mcf_simulation(&config, &schedules);

    print_comparison(&hrw_metrics, &random_metrics, &mcf_metrics);
    print_per_tick_churn(
        &hrw_metrics,
        &random_metrics,
        &mcf_metrics,
        config.total_ticks,
    );

    assert!(
        hrw_metrics.total_churn <= random_metrics.total_churn,
        "HRW ({}) should have <= churn than Random ({})",
        hrw_metrics.total_churn,
        random_metrics.total_churn
    );
    assert!(
        mcf_metrics.total_churn <= random_metrics.total_churn,
        "MCF ({}) should have <= churn than Random ({})",
        mcf_metrics.total_churn,
        random_metrics.total_churn
    );
}

#[test]
fn test_simulation_large_cluster() {
    // Large cluster: 16 backends, 8 slots each, 50 LoRAs, 12 concurrent
    let config = SimConfig {
        num_backends: 16,
        slots_per_backend: 8,
        total_loras: 50,
        concurrent_loras: 12,
        total_ticks: 100,
        ramp_ticks: 5,
        steady_ticks: 15,
        ramp_down_ticks: 5,
        max_load_per_lora: 30,
        scale_down_cooldown_ticks: 2,
        ..Default::default()
    };

    let schedules = generate_load_schedules(&config);
    print_simulation_header(&config);

    let hrw_metrics = run_hrw_simulation(&config, &schedules);
    let random_metrics = run_random_simulation(&config, &schedules);
    let mcf_metrics = run_mcf_simulation(&config, &schedules);

    print_comparison(&hrw_metrics, &random_metrics, &mcf_metrics);

    assert!(
        hrw_metrics.total_churn <= random_metrics.total_churn,
        "HRW ({}) should have <= churn than Random ({})",
        hrw_metrics.total_churn,
        random_metrics.total_churn
    );
    assert!(
        mcf_metrics.total_churn <= random_metrics.total_churn,
        "MCF ({}) should have <= churn than Random ({})",
        mcf_metrics.total_churn,
        random_metrics.total_churn
    );
}

#[test]
fn test_simulation_steady_state_zero_churn() {
    // Verify that HRW has zero churn during steady state (no load changes)
    let config = SimConfig {
        num_backends: 4,
        slots_per_backend: 4,
        total_loras: 4,
        concurrent_loras: 4,
        total_ticks: 30,
        ramp_ticks: 2,
        steady_ticks: 20,
        ramp_down_ticks: 2,
        max_load_per_lora: 10,
        scale_down_cooldown_ticks: 2,
        ..Default::default()
    };

    let active_ticks = config.ramp_ticks + config.steady_ticks + config.ramp_down_ticks;
    let schedules: Vec<LoraLoadSchedule> = (0..config.total_loras)
        .map(|index| LoraLoadSchedule {
            lora_name: format!("lora-{index:03}"),
            active_window: (0, active_ticks),
            peak_load: config.max_load_per_lora,
            ramp_up: config.ramp_ticks,
            steady: config.steady_ticks,
            ramp_down: config.ramp_down_ticks,
            per_tick_loads: None,
        })
        .collect();
    print_simulation_header(&config);

    let hrw_metrics = run_hrw_simulation(&config, &schedules);
    let random_metrics = run_random_simulation(&config, &schedules);
    let mcf_metrics = run_mcf_simulation(&config, &schedules);

    // During the steady phase (ticks ramp_ticks .. ramp_ticks+steady_ticks),
    // there should be zero churn for HRW/MCF since nothing changes
    let steady_start = config.ramp_ticks + 1; // Allow 1 tick after ramp for settling
    let steady_end = config.ramp_ticks + config.steady_ticks;

    let steady_churn = |m: &ChurnMetrics| -> usize {
        m.per_tick_churn
            .iter()
            .enumerate()
            .filter(|(tick, _)| *tick >= steady_start && *tick < steady_end)
            .map(|(_, &churn)| churn)
            .sum()
    };

    let hrw_steady_churn = steady_churn(&hrw_metrics);
    let random_steady_churn = steady_churn(&random_metrics);
    let mcf_steady_churn = steady_churn(&mcf_metrics);

    let num_steady_ticks = steady_end - steady_start;
    let avg = |c: usize| -> f64 {
        if num_steady_ticks > 0 {
            c as f64 / num_steady_ticks as f64
        } else {
            0.0
        }
    };

    println!(
        "\nSteady-State Churn (ticks {}-{}):",
        steady_start, steady_end
    );
    println!(
        "  {:>12}  {:>12}  {:>12}  {:>12}",
        "Metric", "HRW", "Random", "MCF"
    );
    println!("  {:->12}  {:->12}  {:->12}  {:->12}", "", "", "", "");
    println!(
        "  {:>12}  {:>12}  {:>12}  {:>12}",
        "Total", hrw_steady_churn, random_steady_churn, mcf_steady_churn
    );
    println!(
        "  {:>12}  {:>12.2}  {:>12.2}  {:>12.2}",
        "Avg/Tick",
        avg(hrw_steady_churn),
        avg(random_steady_churn),
        avg(mcf_steady_churn)
    );

    print_comparison(&hrw_metrics, &random_metrics, &mcf_metrics);

    assert!(
        avg(hrw_steady_churn) < 1.0,
        "HRW should have near-zero average churn during steady state, got {:.2}",
        avg(hrw_steady_churn)
    );
    assert!(
        avg(mcf_steady_churn) < 1.0,
        "MCF should have near-zero average churn during steady state, got {:.2}",
        avg(mcf_steady_churn)
    );
    assert!(
        hrw_steady_churn <= random_steady_churn,
        "HRW steady churn ({}) should be <= Random ({})",
        hrw_steady_churn,
        random_steady_churn
    );
}

#[test]
fn test_simulation_hrw_stability_across_seeds() {
    // Run simulation with multiple seeds to verify HRW/MCF consistently beat random
    let mut hrw_wins = 0;
    let mut mcf_wins = 0;
    let mut hrw_ties = 0;
    let mut mcf_ties = 0;
    let num_runs = 10;

    for seed in 0..num_runs {
        let config = SimConfig {
            num_backends: 6,
            slots_per_backend: 4,
            total_loras: 15,
            concurrent_loras: 5,
            total_ticks: 50,
            ramp_ticks: 3,
            steady_ticks: 10,
            ramp_down_ticks: 3,
            max_load_per_lora: 15,
            scale_down_cooldown_ticks: 2,
            seed: seed as u64 * 100 + 7,
            ..Default::default()
        };

        let schedules = generate_load_schedules(&config);
        let hrw = run_hrw_simulation(&config, &schedules);
        let random = run_random_simulation(&config, &schedules);
        let mcf = run_mcf_simulation(&config, &schedules);

        if hrw.total_churn < random.total_churn {
            hrw_wins += 1;
        } else if hrw.total_churn == random.total_churn {
            hrw_ties += 1;
        }
        if mcf.total_churn < random.total_churn {
            mcf_wins += 1;
        } else if mcf.total_churn == random.total_churn {
            mcf_ties += 1;
        }

        println!(
            "Seed {:>3}: HRW={:<5} Random={:<5} MCF={:<5} HRW{} MCF{}",
            config.seed,
            hrw.total_churn,
            random.total_churn,
            mcf.total_churn,
            if hrw.total_churn <= random.total_churn {
                "✓"
            } else {
                "✗"
            },
            if mcf.total_churn <= random.total_churn {
                "✓"
            } else {
                "✗"
            }
        );
    }

    println!(
        "\nHRW vs Random: wins={}/{}, ties={}/{}",
        hrw_wins, num_runs, hrw_ties, num_runs
    );
    println!(
        "MCF vs Random: wins={}/{}, ties={}/{}",
        mcf_wins, num_runs, mcf_ties, num_runs
    );

    assert!(
        hrw_wins + hrw_ties >= num_runs * 7 / 10,
        "HRW should win or tie in at least 70% of runs, got {}/{}",
        hrw_wins + hrw_ties,
        num_runs
    );
    assert!(
        mcf_wins + mcf_ties >= num_runs * 7 / 10,
        "MCF should win or tie in at least 70% of runs, got {}/{}",
        mcf_wins + mcf_ties,
        num_runs
    );
}

#[test]
fn test_simulation_load_pattern_visualization() {
    // Visual test: print load patterns for debugging
    let config = SimConfig {
        num_backends: 4,
        slots_per_backend: 4,
        total_loras: 6,
        concurrent_loras: 3,
        total_ticks: 30,
        ramp_ticks: 3,
        steady_ticks: 6,
        ramp_down_ticks: 3,
        max_load_per_lora: 10,
        scale_down_cooldown_ticks: 2,
        ..Default::default()
    };

    let schedules = generate_load_schedules(&config);

    println!("\nLoad Pattern Visualization:");
    println!(
        "  {:>4}  {}",
        "Tick",
        schedules
            .iter()
            .map(|s| format!("{:>8}", s.lora_name))
            .collect::<Vec<_>>()
            .join("")
    );
    println!(
        "  {:->4}  {}",
        "",
        schedules
            .iter()
            .map(|_| format!("{:->8}", ""))
            .collect::<Vec<_>>()
            .join("")
    );

    let mut max_concurrent = 0;

    for tick in 0..config.total_ticks {
        let loads: Vec<usize> = schedules.iter().map(|s| s.load_at_tick(tick)).collect();
        let active_count = loads.iter().filter(|&&l| l > 0).count();
        max_concurrent = max_concurrent.max(active_count);

        let total_load: usize = loads.iter().sum();
        if total_load > 0 || tick < 5 || tick >= config.total_ticks - 5 {
            println!(
                "  {:>4}  {} (active: {}, total: {})",
                tick,
                loads
                    .iter()
                    .map(|l| if *l > 0 {
                        format!("{:>8}", l)
                    } else {
                        format!("{:>8}", "·")
                    })
                    .collect::<Vec<_>>()
                    .join(""),
                active_count,
                total_load
            );
        }
    }

    println!("\nMax concurrent LoRAs observed: {}", max_concurrent);
    assert!(
        max_concurrent <= config.concurrent_loras + 2,
        "Max concurrent LoRAs ({}) should be close to configured ({})",
        max_concurrent,
        config.concurrent_loras
    );
}
