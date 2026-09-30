// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Modeled cost of one physical adapter load in the swap-time estimate.
const ADAPTER_LOAD_COST_MS: usize = 50;

fn percentile_99(values: &[usize]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[((sorted.len() - 1) * 99) / 100]
}

fn percentile_50_ms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) / 2]
}

fn percentile_99_ms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) * 99) / 100]
}

// ============================================================================
// CSV Export for Visualization
// ============================================================================

/// Runs load-driven simulations and writes per-tick churn, load, lifecycle,
/// and summary data to CSV files under
/// `target/lora_sim_csv/`. Use the companion `plot_lora_churn.py` script
/// to visualize.
///
/// Run with:
///   cargo test --test lora_simulation -- test_export_csv --nocapture --ignored
///   python lib/llm/tests/lora_simulation/plot_lora_churn.py
#[test]
#[ignore] // Run explicitly: cargo test --test lora_simulation -- test_export_csv --ignored --nocapture
fn test_export_csv() {
    use std::fs;
    use std::io::Write;

    let out_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/lora_sim_csv");
    fs::create_dir_all(&out_dir).expect("create output dir");
    let v2_out_dir = out_dir.join("v2");
    fs::create_dir_all(&v2_out_dir).expect("create v2 output dir");
    let mut timeseries =
        fs::File::create(v2_out_dir.join("timeseries.csv")).expect("create v2 timeseries csv");
    writeln!(
        timeseries,
        "scenario,algorithm,seed,tick,adapter_loads,adapter_unloads,adapter_churn,route_churn,requests,hits,hit_rate,active_loras,routing_entries,cold_start_entries,mean_occupancy,max_occupancy,worker_load_cov,solve_ms,overflow_count"
    )
    .unwrap();
    let mut summary =
        fs::File::create(v2_out_dir.join("summary.csv")).expect("create v2 summary csv");
    writeln!(
        summary,
        "scenario,algorithm,seed,total_adapter_churn,total_route_churn,adapter_churn_free_ticks,adapter_churn_free_pct,peak_adapter_churn_per_tick,p99_adapter_churn_per_tick,overall_hit_rate,mean_worker_load_cov,mean_occupancy_pct,solve_ms_p50,solve_ms_p99,total_overflow,est_swap_ms"
    )
    .unwrap();

    // Fixed cluster: N=8 backends, K=4 resident LoRA slots → 32 total slots. Replica sets are
    // controller route targets; under pressure they can exceed resident capacity and trigger lazy
    // backend cache swaps. The exported churn is therefore routing-target churn, not observed GPU
    // cache load/unload operations.
    let num_backends: usize = 8;
    let slots_per_backend: usize = 4;
    let total_slots = num_backends * slots_per_backend; // 32

    let mut all_runs: Vec<(&str, SimConfig, Vec<LoraLoadSchedule>)> = Vec::new();

    // ── 1. Zipf + Poisson scenario: 100 LoRAs, power-law popularity, Poisson arrivals.
    // Zipf s=1.0 means top LoRA gets ~19% of traffic, top-5 get ~45%.
    let zipf_total_loras: usize = 100;
    let zipf_s: f64 = 1.0;
    let zipf_avg_load: f64 = 40.0;
    let zipf_ticks: usize = 200; // longer to show steady-state stochastic behavior
    let zipf_schedules =
        generate_zipf_poisson_schedules(zipf_total_loras, zipf_ticks, zipf_s, zipf_avg_load, 42);
    let zipf_config = SimConfig {
        num_backends,
        slots_per_backend,
        total_loras: zipf_total_loras,
        concurrent_loras: 0, // not meaningful for Zipf (driven by Poisson draws)
        total_ticks: zipf_ticks,
        ramp_ticks: 0,
        steady_ticks: 0,
        ramp_down_ticks: 0,
        max_load_per_lora: 0,
        lifetime_mean: 0,
        lifetime_stddev: 0.0,
        seed: 42,
        ..Default::default()
    };
    all_runs.push(("hot_lora_poisson", zipf_config.clone(), zipf_schedules));

    // Diurnal (time-of-day) scenario: Zipf + Poisson with day/night cycle.
    // ticks_per_day=100, 200 ticks = 2 full days.
    // Peak (noon) load=50, trough (midnight) load=10.
    let diurnal_total_loras: usize = 100;
    let diurnal_s: f64 = 1.0;
    let diurnal_peak: f64 = 50.0;
    let diurnal_trough: f64 = 10.0;
    let diurnal_ticks: usize = 200;
    let diurnal_ticks_per_day: usize = 100;
    let diurnal_schedules = generate_diurnal_schedules(
        diurnal_total_loras,
        diurnal_ticks,
        diurnal_ticks_per_day,
        diurnal_s,
        diurnal_peak,
        diurnal_trough,
        42,
    );
    let diurnal_config = SimConfig {
        num_backends,
        slots_per_backend,
        total_loras: diurnal_total_loras,
        concurrent_loras: 0,
        total_ticks: diurnal_ticks,
        ramp_ticks: 0,
        steady_ticks: 0,
        ramp_down_ticks: 0,
        max_load_per_lora: 0,
        lifetime_mean: 0,
        lifetime_stddev: 0.0,
        seed: 42,
        ..Default::default()
    };
    all_runs.push(("daily", diurnal_config.clone(), diurnal_schedules));

    // ── 3. Flash Crowd: baseline Zipf+Poisson with two viral spikes.
    // Spike at tick 50 and 130, multiplier=5x, half-life=8 ticks.
    let flash_total_loras: usize = 100;
    let flash_s: f64 = 1.0;
    let flash_base_load: f64 = 25.0;
    let flash_spike: f64 = 5.0;
    let flash_half_life: f64 = 8.0;
    let flash_ticks: usize = 200;
    let flash_events: Vec<usize> = vec![50, 130];
    let flash_schedules = generate_flash_crowd_schedules(
        flash_total_loras,
        flash_ticks,
        flash_s,
        flash_base_load,
        flash_spike,
        flash_half_life,
        &flash_events,
        42,
    );
    let flash_config = SimConfig {
        num_backends,
        slots_per_backend,
        total_loras: flash_total_loras,
        concurrent_loras: 0,
        total_ticks: flash_ticks,
        ramp_ticks: 0,
        steady_ticks: 0,
        ramp_down_ticks: 0,
        max_load_per_lora: 0,
        lifetime_mean: 0,
        lifetime_stddev: 0.0,
        seed: 42,
        ..Default::default()
    };
    all_runs.push(("spike", flash_config.clone(), flash_schedules));

    // ── 4. MMPP: Markov-modulated Poisson process with 3 states.
    //   calm  (rate=15): 90% stay, 10% → busy
    //   busy  (rate=40): 15% → calm, 80% stay, 5% → surge
    //   surge (rate=70): 10% → calm, 20% → busy, 70% stay
    let mmpp_total_loras: usize = 100;
    let mmpp_s: f64 = 1.0;
    let mmpp_ticks: usize = 200;
    let mmpp_state_rates = vec![15.0, 40.0, 70.0];
    let mmpp_transitions = vec![
        vec![0.90, 0.10, 0.00], // calm  → {calm, busy, surge}
        vec![0.15, 0.80, 0.05], // busy  → {calm, busy, surge}
        vec![0.10, 0.20, 0.70], // surge → {calm, busy, surge}
    ];
    let (mmpp_schedules, mmpp_state_seq) = generate_mmpp_schedules(
        mmpp_total_loras,
        mmpp_ticks,
        mmpp_s,
        &mmpp_state_rates,
        &mmpp_transitions,
        42,
    );
    let mmpp_config = SimConfig {
        num_backends,
        slots_per_backend,
        total_loras: mmpp_total_loras,
        concurrent_loras: 0,
        total_ticks: mmpp_ticks,
        ramp_ticks: 0,
        steady_ticks: 0,
        ramp_down_ticks: 0,
        max_load_per_lora: 0,
        lifetime_mean: 0,
        lifetime_stddev: 0.0,
        seed: 42,
        ..Default::default()
    };
    all_runs.push(("mmpp", mmpp_config.clone(), mmpp_schedules));

    for (name, config, schedules) in &all_runs {
        let hrw = run_hrw_simulation(config, schedules);
        let random = run_random_simulation(config, schedules);
        let mcf = run_mcf_simulation(config, schedules);

        println!(
            "\n── {} (load-driven, N={} backends, K={} slots/backend, L={}) ──",
            name, config.num_backends, config.slots_per_backend, config.total_loras
        );
        print_comparison(&hrw, &random, &mcf);

        for metrics in [&hrw, &random, &mcf] {
            for tick in 0..config.total_ticks {
                let adapter_loads = metrics.per_tick_adapter_loads[tick];
                let adapter_unloads = metrics.per_tick_adapter_unloads[tick];
                let requests = metrics.per_tick_requests[tick];
                let hits = metrics.per_tick_hits[tick];
                let hit_rate = if requests == 0 {
                    0.0
                } else {
                    hits as f64 / requests as f64
                };
                writeln!(
                    timeseries,
                    "{},{},{},{},{},{},{},{},{},{},{:.6},{},{},{},{:.6},{},{:.6},{:.6},{}",
                    name,
                    metrics.algorithm,
                    config.seed,
                    tick,
                    adapter_loads,
                    adapter_unloads,
                    adapter_loads + adapter_unloads,
                    metrics.per_tick_churn[tick],
                    requests,
                    hits,
                    hit_rate,
                    metrics.per_tick_active_loras[tick],
                    metrics.per_tick_routing_entries[tick],
                    metrics.per_tick_cold_start_entries[tick],
                    metrics.per_tick_mean_occupancy[tick],
                    metrics.per_tick_max_occupancy[tick],
                    metrics.per_tick_worker_load_cov[tick],
                    metrics.per_tick_solve_ms[tick],
                    metrics.per_tick_overflow_count[tick],
                )
                .unwrap();
            }

            let total_loads: usize = metrics.per_tick_adapter_loads.iter().sum();
            let total_unloads: usize = metrics.per_tick_adapter_unloads.iter().sum();
            let total_requests: usize = metrics.per_tick_requests.iter().sum();
            let total_hits: usize = metrics.per_tick_hits.iter().sum();
            let mean_cov = metrics.per_tick_worker_load_cov.iter().sum::<f64>()
                / metrics.per_tick_worker_load_cov.len() as f64;
            let mean_occupancy_pct = metrics.per_tick_mean_occupancy.iter().sum::<f64>()
                / metrics.per_tick_mean_occupancy.len() as f64
                / config.slots_per_backend as f64
                * 100.0;
            let total_overflow: usize = metrics.per_tick_overflow_count.iter().sum();
            let adapter_churn_per_tick: Vec<usize> = metrics
                .per_tick_adapter_loads
                .iter()
                .zip(&metrics.per_tick_adapter_unloads)
                .map(|(loads, unloads)| loads + unloads)
                .collect();
            let churn_free_ticks = adapter_churn_per_tick
                .iter()
                .filter(|&&churn| churn == 0)
                .count();
            writeln!(
                summary,
                "{},{},{},{},{},{},{:.6},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
                name,
                metrics.algorithm,
                config.seed,
                total_loads + total_unloads,
                metrics.total_churn,
                churn_free_ticks,
                churn_free_ticks as f64 / config.total_ticks as f64 * 100.0,
                adapter_churn_per_tick.iter().copied().max().unwrap_or(0),
                percentile_99(&adapter_churn_per_tick),
                if total_requests == 0 {
                    0.0
                } else {
                    total_hits as f64 / total_requests as f64
                },
                mean_cov,
                mean_occupancy_pct,
                percentile_50_ms(&metrics.per_tick_solve_ms),
                percentile_99_ms(&metrics.per_tick_solve_ms),
                total_overflow,
                total_loads * ADAPTER_LOAD_COST_MS,
            )
            .unwrap();
        }

        // ── Write churn CSV (with LoRA adds/removes) ────────────────────────
        let churn_path = out_dir.join(format!("{}_churn.csv", name));
        let mut f = fs::File::create(&churn_path).expect("create churn csv");
        writeln!(
            f,
            "tick,hrw_churn,random_churn,mcf_churn,hrw_cumulative,random_cumulative,mcf_cumulative,\
             hrw_lora_adds,random_lora_adds,mcf_lora_adds,hrw_lora_removes,random_lora_removes,mcf_lora_removes"
        )
        .unwrap();

        let mut hrw_cum: usize = 0;
        let mut rand_cum: usize = 0;
        let mut mcf_cum: usize = 0;
        for tick in 0..config.total_ticks {
            let h = hrw.per_tick_churn.get(tick).copied().unwrap_or(0);
            let r = random.per_tick_churn.get(tick).copied().unwrap_or(0);
            let m = mcf.per_tick_churn.get(tick).copied().unwrap_or(0);
            hrw_cum += h;
            rand_cum += r;
            mcf_cum += m;
            let ha = hrw.per_tick_lora_additions.get(tick).copied().unwrap_or(0);
            let ra = random
                .per_tick_lora_additions
                .get(tick)
                .copied()
                .unwrap_or(0);
            let ma = mcf.per_tick_lora_additions.get(tick).copied().unwrap_or(0);
            let hr = hrw.per_tick_lora_removals.get(tick).copied().unwrap_or(0);
            let rr = random
                .per_tick_lora_removals
                .get(tick)
                .copied()
                .unwrap_or(0);
            let mr = mcf.per_tick_lora_removals.get(tick).copied().unwrap_or(0);
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{},{},{},{},{}",
                tick, h, r, m, hrw_cum, rand_cum, mcf_cum, ha, ra, ma, hr, rr, mr
            )
            .unwrap();
        }

        // ── Write load pattern CSV ──────────────────────────────────────────
        let load_path = out_dir.join(format!("{}_load.csv", name));
        let mut f = fs::File::create(&load_path).expect("create load csv");

        // Header
        let lora_names: Vec<String> = schedules.iter().map(|s| s.lora_name.clone()).collect();
        write!(f, "tick,total_load,active_loras").unwrap();
        for ln in &lora_names {
            write!(f, ",{}", ln).unwrap();
        }
        writeln!(f).unwrap();

        // Rows
        for tick in 0..config.total_ticks {
            let loads: Vec<usize> = schedules.iter().map(|s| s.load_at_tick(tick)).collect();
            let total: usize = loads.iter().sum();
            let active = loads.iter().filter(|&&l| l > 0).count();
            write!(f, "{},{},{}", tick, total, active).unwrap();
            for l in &loads {
                write!(f, ",{}", l).unwrap();
            }
            writeln!(f).unwrap();
        }

        // ── Write lifecycle CSV (per-LoRA start/end/peak for timeline) ──────
        let lifecycle_path = out_dir.join(format!("{}_lifecycle.csv", name));
        let mut f = fs::File::create(&lifecycle_path).expect("create lifecycle csv");
        writeln!(f, "lora_name,start_tick,end_tick,peak_load,lora_index").unwrap();
        for (idx, schedule) in schedules.iter().enumerate() {
            writeln!(
                f,
                "{},{},{},{},{}",
                schedule.lora_name,
                schedule.active_window.0,
                schedule.active_window.1,
                schedule.peak_load,
                idx
            )
            .unwrap();
        }

        // ── Write replica distribution CSV ───────────────────────────────────
        // Find the maximum replica count across all ticks for all algorithms
        let max_replicas = {
            let max_of = |m: &ChurnMetrics| -> usize {
                m.per_tick_replica_dist
                    .iter()
                    .flat_map(|d| d.keys())
                    .copied()
                    .max()
                    .unwrap_or(0)
            };
            max_of(&hrw).max(max_of(&random)).max(max_of(&mcf))
        };

        let replica_path = out_dir.join(format!("{}_replicas.csv", name));
        let mut f = fs::File::create(&replica_path).expect("create replicas csv");

        // Header: tick,hrw_r1,...,random_r1,...,mcf_r1,...
        write!(f, "tick").unwrap();
        for algo in &["hrw", "random", "mcf"] {
            for r in 1..=max_replicas {
                write!(f, ",{}_r{}", algo, r).unwrap();
            }
        }
        writeln!(f).unwrap();

        for tick in 0..config.total_ticks {
            write!(f, "{}", tick).unwrap();
            for metrics in [&hrw, &random, &mcf] {
                let dist = metrics.per_tick_replica_dist.get(tick);
                for r in 1..=max_replicas {
                    let count = dist.and_then(|d| d.get(&r)).copied().unwrap_or(0);
                    write!(f, ",{}", count).unwrap();
                }
            }
            writeln!(f).unwrap();
        }

        // ── Write summary CSV ───────────────────────────────────────────────
        let summary_path = out_dir.join(format!("{}_summary.csv", name));
        let mut f = fs::File::create(&summary_path).expect("create summary csv");
        writeln!(f, "metric,hrw,random,mcf").unwrap();
        writeln!(
            f,
            "total_churn,{},{},{}",
            hrw.total_churn, random.total_churn, mcf.total_churn
        )
        .unwrap();
        writeln!(
            f,
            "route_target_additions,{},{},{}",
            hrw.total_target_additions, random.total_target_additions, mcf.total_target_additions
        )
        .unwrap();
        writeln!(
            f,
            "route_target_removals,{},{},{}",
            hrw.total_target_removals, random.total_target_removals, mcf.total_target_removals
        )
        .unwrap();
        writeln!(
            f,
            "peak_churn_per_tick,{},{},{}",
            hrw.peak_churn_per_tick, random.peak_churn_per_tick, mcf.peak_churn_per_tick
        )
        .unwrap();
        writeln!(
            f,
            "avg_churn_per_active_tick,{:.2},{:.2},{:.2}",
            hrw.avg_churn_per_active_tick,
            random.avg_churn_per_active_tick,
            mcf.avg_churn_per_active_tick
        )
        .unwrap();
        writeln!(
            f,
            "lora_additions,{},{},{}",
            hrw.total_lora_additions, random.total_lora_additions, mcf.total_lora_additions
        )
        .unwrap();
        writeln!(
            f,
            "lora_removals,{},{},{}",
            hrw.total_lora_removals, random.total_lora_removals, mcf.total_lora_removals
        )
        .unwrap();

        // ── Write config metadata ───────────────────────────────────────────
        let meta_path = out_dir.join(format!("{}_meta.csv", name));
        let mut f = fs::File::create(&meta_path).expect("create meta csv");
        writeln!(f, "key,value").unwrap();
        writeln!(f, "scenario,{}", name).unwrap();
        writeln!(f, "num_backends,{}", config.num_backends).unwrap();
        writeln!(f, "slots_per_backend,{}", config.slots_per_backend).unwrap();
        writeln!(f, "total_slots,{}", total_slots).unwrap();
        writeln!(f, "total_loras,{}", config.total_loras).unwrap();
        writeln!(f, "total_ticks,{}", config.total_ticks).unwrap();
        writeln!(f, "loras_used,{}", schedules.len()).unwrap();
        writeln!(f, "lifetime_mean,{}", config.lifetime_mean).unwrap();
        writeln!(f, "lifetime_stddev,{:.1}", config.lifetime_stddev).unwrap();
        writeln!(
            f,
            "scale_down_cooldown_ticks,{}",
            config.scale_down_cooldown_ticks
        )
        .unwrap();
        writeln!(f, "seed,{}", config.seed).unwrap();
        if *name == "hot_lora_poisson" {
            writeln!(f, "load_model,zipf_poisson").unwrap();
            writeln!(f, "zipf_s,{:.1}", zipf_s).unwrap();
            writeln!(f, "avg_total_load,{:.0}", zipf_avg_load).unwrap();
        }
        if *name == "daily" {
            writeln!(f, "load_model,diurnal").unwrap();
            writeln!(f, "zipf_s,{:.1}", diurnal_s).unwrap();
            writeln!(f, "peak_total_load,{:.0}", diurnal_peak).unwrap();
            writeln!(f, "trough_total_load,{:.0}", diurnal_trough).unwrap();
            writeln!(f, "ticks_per_day,{}", diurnal_ticks_per_day).unwrap();
        }
        if *name == "spike" {
            writeln!(f, "load_model,flash_crowd").unwrap();
            writeln!(f, "zipf_s,{:.1}", flash_s).unwrap();
            writeln!(f, "base_total_load,{:.0}", flash_base_load).unwrap();
            writeln!(f, "spike_multiplier,{:.0}", flash_spike).unwrap();
            writeln!(f, "decay_half_life,{:.0}", flash_half_life).unwrap();
            writeln!(
                f,
                "flash_ticks,{}",
                flash_events
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(";")
            )
            .unwrap();
        }
        if *name == "mmpp" {
            writeln!(f, "load_model,mmpp").unwrap();
            writeln!(f, "zipf_s,{:.1}", mmpp_s).unwrap();
            writeln!(
                f,
                "state_rates,{}",
                mmpp_state_rates
                    .iter()
                    .map(|r| format!("{:.0}", r))
                    .collect::<Vec<_>>()
                    .join(";")
            )
            .unwrap();
            writeln!(f, "state_names,calm;busy;surge").unwrap();
            // Write MMPP state sequence as a separate CSV for plotting
            let state_path = out_dir.join("mmpp_states.csv");
            let mut sf = fs::File::create(&state_path).expect("create mmpp states csv");
            writeln!(sf, "tick,state,state_name,rate").unwrap();
            let state_names = ["calm", "busy", "surge"];
            for (t, &s) in mmpp_state_seq.iter().enumerate() {
                writeln!(
                    sf,
                    "{},{},{},{:.0}",
                    t, s, state_names[s], mmpp_state_rates[s]
                )
                .unwrap();
            }
        }

        println!("Exported CSVs for '{}' → {}", name, out_dir.display());
    }

    println!("\nAll CSVs written to: {}", out_dir.display());
    let meta_config = &all_runs[0].1;
    assert!(
        all_runs.iter().all(|(_, config, _)| {
            config.timestep_secs == meta_config.timestep_secs
                && config.rate_window_multiplier == meta_config.rate_window_multiplier
                && config.scale_down_cooldown_ticks == meta_config.scale_down_cooldown_ticks
        }),
        "v2 meta.csv describes a single timing configuration for all scenarios"
    );
    let mut meta = fs::File::create(v2_out_dir.join("meta.csv")).expect("create v2 meta csv");
    writeln!(meta, "key,value").unwrap();
    writeln!(meta, "timestep_secs,{}", meta_config.timestep_secs).unwrap();
    writeln!(
        meta,
        "rate_window_secs,{}",
        meta_config
            .timestep_secs
            .saturating_mul(meta_config.rate_window_multiplier)
            .max(5)
    )
    .unwrap();
    writeln!(
        meta,
        "scale_down_cooldown_ticks,{}",
        meta_config.scale_down_cooldown_ticks
    )
    .unwrap();
    writeln!(meta, "adapter_load_cost_ms,{}", ADAPTER_LOAD_COST_MS).unwrap();
    writeln!(meta, "seed_count,1").unwrap();
    println!("Run: python lib/llm/tests/lora_simulation/plot_lora_churn.py");
}
