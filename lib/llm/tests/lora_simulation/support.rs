// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA Allocation Simulation / Churn Measurement
//!
//! A CPU-only integration test that simulates a cluster environment with
//! N backends (each having K LoRA slots), L distinct LoRAs, and measures
//! routing-target churn (worker additions/removals in target replica sets) across
//! various load patterns.
//!
//! Compares HRW, Random, and Min-Cost Flow (MCF) allocation algorithms to
//! demonstrate churn characteristics across various load patterns.
//!
//! Run with: `cargo test --test lora_simulation -- --nocapture`

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dynamo_llm::kv_router::protocols::WorkerWithDpRank;
use dynamo_llm::lora::config::LoraAllocationConfig;
use dynamo_llm::lora::controller::LoraController;
use dynamo_llm::lora::load_estimator::LoadEstimator;
use dynamo_llm::lora::routing::AllocationAlgorithmType;
use dynamo_llm::lora::routing::table::LoraRoutingTable;
use dynamo_llm::lora::state_tracker::LoraStateTracker;

use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;

// Workload configuration and generators are kept separate from allocation runners.
include!("workloads.rs");

/// Keep allocator comparisons free of the controller's separate scale-down policy.
const COMPARISON_SCALE_DOWN_COOLDOWN_TICKS: u32 = 0;

// ============================================================================
// Random Allocator (for baseline comparison)
// ============================================================================

/// A random allocator that picks `replica_factor` workers randomly each tick.
/// This simulates the worst-case scenario for churn: no consistency across ticks.
struct RandomAllocator {
    rng: StdRng,
}

impl RandomAllocator {
    fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn compute_replica_set(
        &mut self,
        _lora_name: &str,
        workers: &[WorkerWithDpRank],
        replica_factor: usize,
        worker_slot_usage: &HashMap<WorkerWithDpRank, (usize, usize)>,
    ) -> Vec<WorkerWithDpRank> {
        if workers.is_empty() || replica_factor == 0 {
            return Vec::new();
        }

        let mut available: Vec<WorkerWithDpRank> = workers
            .iter()
            .filter_map(|worker| {
                let (used, capacity) = worker_slot_usage
                    .get(worker)
                    .copied()
                    .unwrap_or((0, usize::MAX));
                (used < capacity).then_some(*worker)
            })
            .collect();

        if available.is_empty() {
            return Vec::new();
        }

        available.shuffle(&mut self.rng);
        available.into_iter().take(replica_factor).collect()
    }

    /// Pick route targets without applying resident-slot capacity. Controllers use this kind of
    /// fallback to keep an over-budget LoRA routable; it is a route hint, not proof of residency.
    fn compute_fallback_replica_set(
        &mut self,
        workers: &[WorkerWithDpRank],
        replica_factor: usize,
    ) -> Vec<WorkerWithDpRank> {
        let mut shuffled = workers.to_vec();
        shuffled.shuffle(&mut self.rng);
        shuffled.into_iter().take(replica_factor).collect()
    }
}

// ============================================================================
// Simulation Runner
// ============================================================================

/// Snapshot of the allocation state at a given tick
type AllocationSnapshot = HashMap<String, Vec<WorkerWithDpRank>>;

/// Compute churn between two allocation snapshots
fn compute_churn(prev: &AllocationSnapshot, curr: &AllocationSnapshot) -> (usize, usize) {
    let mut loads = 0;
    let mut unloads = 0;

    // Check all LoRAs in current allocation
    for (lora_name, curr_workers) in curr {
        let prev_workers = prev.get(lora_name).cloned().unwrap_or_default();
        let prev_set: HashSet<_> = prev_workers.iter().collect();
        let curr_set: HashSet<_> = curr_workers.iter().collect();

        // Workers added to this LoRA = loads
        loads += curr_set.difference(&prev_set).count();
        // Workers removed from this LoRA = unloads
        unloads += prev_set.difference(&curr_set).count();
    }

    // Check LoRAs that were in prev but not in curr (fully unloaded)
    for (lora_name, prev_workers) in prev {
        if !curr.contains_key(lora_name) {
            unloads += prev_workers.len();
        }
    }

    (loads, unloads)
}

/// Compute the controller's proportional, capacity-normalized active replica budget.
///
/// Keep this in lockstep with `LoraController::compute_active_replica_counts`: the random
/// baseline must differ only in placement, not in how many replicas each LoRA requests.
fn compute_normalized_replica_counts(
    active_loras: &[(String, usize)],
    total_slots: usize,
    num_workers: usize,
) -> HashMap<String, usize> {
    if active_loras.is_empty() || total_slots == 0 || num_workers == 0 {
        return HashMap::new();
    }

    let mut ranked = active_loras.to_vec();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(total_slots);

    let effective_total_load: usize = ranked.iter().map(|(_, load)| *load).sum();
    if effective_total_load == 0 {
        return HashMap::new();
    }

    let mut raw_counts: Vec<(String, f64)> = ranked
        .into_iter()
        .map(|(name, load)| {
            let count = ((load as f64 / effective_total_load as f64) * total_slots as f64)
                .ceil()
                .max(1.0)
                .min(num_workers as f64);
            (name, count)
        })
        .collect();

    let raw_sum: f64 = raw_counts.iter().map(|(_, count)| *count).sum();
    if raw_sum <= total_slots as f64 {
        return raw_counts
            .into_iter()
            .map(|(name, count)| (name, count as usize))
            .collect();
    }

    let scale = total_slots as f64 / raw_sum;
    for (_, count) in &mut raw_counts {
        *count = (*count * scale).max(1.0);
    }

    let mut normalized: Vec<(String, usize, f64)> = raw_counts
        .into_iter()
        .map(|(name, count)| {
            let floor = count.floor().max(1.0) as usize;
            (name, floor, count - floor as f64)
        })
        .collect();
    let floor_sum: usize = normalized.iter().map(|(_, count, _)| *count).sum();
    let mut leftover = total_slots.saturating_sub(floor_sum);

    normalized.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    for (_, count, _) in &mut normalized {
        if leftover == 0 {
            break;
        }
        if *count < num_workers {
            *count += 1;
            leftover -= 1;
        }
    }

    normalized
        .into_iter()
        .map(|(name, count, _)| (name, count))
        .collect()
}

/// Build one independently randomized routing snapshot using the controller's replica budget and
/// overflow-routability rules.
fn compute_random_snapshot(
    random_allocator: &mut RandomAllocator,
    active_loads: &[(String, usize)],
    workers: &[WorkerWithDpRank],
    slots_per_backend: usize,
) -> AllocationSnapshot {
    let total_slots = workers.len() * slots_per_backend;
    let replica_counts =
        compute_normalized_replica_counts(active_loads, total_slots, workers.len());
    let mut snapshot = AllocationSnapshot::new();
    let mut worker_slot_usage: HashMap<WorkerWithDpRank, (usize, usize)> = workers
        .iter()
        .map(|&worker| (worker, (0, slots_per_backend)))
        .collect();

    // Match the controller's deterministic per-LoRA processing order. Randomness affects only
    // worker selection, while the normalized demand and capacity inputs stay identical.
    let mut ordered_counts: Vec<_> = replica_counts.iter().collect();
    ordered_counts.sort_by(|a, b| a.0.cmp(b.0));
    for (lora_name, &desired) in ordered_counts {
        let capacity_aware =
            random_allocator.compute_replica_set(lora_name, workers, desired, &worker_slot_usage);
        let replica_set = if capacity_aware.is_empty() {
            random_allocator.compute_fallback_replica_set(workers, desired)
        } else {
            capacity_aware
        };

        for worker in &replica_set {
            if let Some((used, _)) = worker_slot_usage.get_mut(worker) {
                *used += 1;
            }
        }
        snapshot.insert(lora_name.clone(), replica_set);
    }

    // The controller keeps a single route target for active LoRAs truncated from the resident
    // slot budget. Mirror that routability behavior in the random baseline. These fallback
    // targets are deliberately not interpreted as resident slots or successful load events.
    let mut dropped: Vec<_> = active_loads
        .iter()
        .filter(|(name, _)| !replica_counts.contains_key(name))
        .collect();
    dropped.sort_by(|a, b| a.0.cmp(&b.0));
    for (lora_name, _) in dropped {
        snapshot.insert(
            lora_name.clone(),
            random_allocator.compute_fallback_replica_set(workers, 1),
        );
    }

    snapshot
}

/// Run the simulation with HRW using `LoraController` and measure routing-target churn.
///
/// Replica sets are desired routes, not observed backend residency. Under capacity pressure the
/// controller deliberately retains fallback routes that can exceed the resident-slot budget and
/// cause lazy cache swaps when requests arrive.
fn run_hrw_simulation(config: &SimConfig, schedules: &[LoraLoadSchedule]) -> ChurnMetrics {
    let alloc_config = LoraAllocationConfig {
        enabled: true,
        algorithm: AllocationAlgorithmType::Hrw,
        timestep_secs: 1, // Not used in sync mode
        // The benchmark isolates routing-target selection. Hysteresis is a separate policy
        // that the random baseline does not implement and would confound the comparison.
        scale_down_cooldown_ticks: COMPARISON_SCALE_DOWN_COOLDOWN_TICKS,
        rate_window_multiplier: 5,
        ..Default::default()
    };

    let routing_table = LoraRoutingTable::new();
    let state_tracker = LoraStateTracker::new();
    let load_estimator = Arc::new(LoadEstimator::new());
    let mut controller = LoraController::new(
        alloc_config,
        routing_table.clone(),
        state_tracker.clone(),
        load_estimator.clone(),
    );

    // Register all workers
    let workers: Vec<WorkerWithDpRank> = (0..config.num_backends)
        .map(|i| WorkerWithDpRank::new(i as u64, 0))
        .collect();

    // Register per-worker capacity without loading a phantom adapter (a dummy LoRA would consume
    // a real slot and add a phantom inactive pin).
    for &worker in &workers {
        state_tracker.set_worker_capacity(worker, config.slots_per_backend as u32);
    }

    let mut metrics = ChurnMetrics::new("HRW");
    let mut prev_snapshot: AllocationSnapshot = HashMap::new();

    for tick in 0..config.total_ticks {
        // Fully clear the previous tick's load signal. `decrement_load` only touches the in-flight
        // counter, not the windowed rate counter the controller actually reads via
        // `get_current_load()`; since the whole simulation runs in one real-time instant, without
        // `remove_lora` the rate counter would accumulate across ticks and the controller would see
        // ever-growing load instead of this tick's pattern.
        for name in load_estimator
            .get_current_load()
            .keys()
            .cloned()
            .collect::<Vec<_>>()
        {
            load_estimator.remove_lora(&name);
        }

        // The controller unions adapter names from the load estimator with discovery state, so
        // active adapters do not need synthetic MDC registrations. Registering them on an
        // arbitrary worker would consume that worker's simulated slots for HRW but not MCF.
        // Set only this tick's demand to keep the compared allocators capacity-equivalent.
        for schedule in schedules {
            let load = schedule.load_at_tick(tick);
            for _ in 0..load {
                load_estimator.increment_load(&schedule.lora_name);
            }
        }

        // Recompute allocations
        controller.recompute_now();

        // Snapshot current allocation
        let mut curr_snapshot: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name)
                && !config.replica_set.is_empty()
            {
                curr_snapshot.insert(lora_name, config.replica_set);
            }
        }
        // Compute churn
        let (loads, unloads) = compute_churn(&prev_snapshot, &curr_snapshot);
        let tick_churn = loads + unloads;
        metrics.total_target_additions += loads;
        metrics.total_target_removals += unloads;
        metrics.per_tick_churn.push(tick_churn);

        // Track LoRA additions/removals
        let prev_loras: HashSet<&String> = prev_snapshot.keys().collect();
        let curr_loras: HashSet<&String> = curr_snapshot.keys().collect();
        let additions = curr_loras.difference(&prev_loras).count();
        let removals = prev_loras.difference(&curr_loras).count();
        metrics.per_tick_lora_additions.push(additions);
        metrics.per_tick_lora_removals.push(removals);

        // Track replica distribution
        let mut replica_dist: HashMap<usize, usize> = HashMap::new();
        for replica_set in curr_snapshot.values() {
            *replica_dist.entry(replica_set.len()).or_insert(0) += 1;
        }
        metrics.per_tick_replica_dist.push(replica_dist);

        prev_snapshot = curr_snapshot;
    }

    metrics.finalize();
    metrics
}

/// Run the simulation with random allocation (baseline)
fn run_random_simulation(config: &SimConfig, schedules: &[LoraLoadSchedule]) -> ChurnMetrics {
    let mut random_allocator = RandomAllocator::new(config.seed + 1000);

    // Create workers
    let workers: Vec<WorkerWithDpRank> = (0..config.num_backends)
        .map(|i| WorkerWithDpRank::new(i as u64, 0))
        .collect();

    let mut metrics = ChurnMetrics::new("Random");
    let mut prev_snapshot: AllocationSnapshot = HashMap::new();

    for tick in 0..config.total_ticks {
        // Compute loads for this tick
        let mut active_loads: Vec<(String, usize)> = Vec::new();
        // Match the controller paths: adapters with zero current load have no load-estimator
        // entry, so the random baseline excludes them rather than adding unmatched cold pins.
        for schedule in schedules {
            let load = schedule.load_at_tick(tick);
            if load > 0 {
                active_loads.push((schedule.lora_name.clone(), load));
            }
        }

        let curr_snapshot = compute_random_snapshot(
            &mut random_allocator,
            &active_loads,
            &workers,
            config.slots_per_backend,
        );
        // Compute churn
        let (loads, unloads) = compute_churn(&prev_snapshot, &curr_snapshot);
        let tick_churn = loads + unloads;
        metrics.total_target_additions += loads;
        metrics.total_target_removals += unloads;
        metrics.per_tick_churn.push(tick_churn);

        // Track LoRA additions/removals
        let prev_loras: HashSet<&String> = prev_snapshot.keys().collect();
        let curr_loras: HashSet<&String> = curr_snapshot.keys().collect();
        let additions = curr_loras.difference(&prev_loras).count();
        let removals = prev_loras.difference(&curr_loras).count();
        metrics.per_tick_lora_additions.push(additions);
        metrics.per_tick_lora_removals.push(removals);

        // Track replica distribution
        let mut replica_dist: HashMap<usize, usize> = HashMap::new();
        for replica_set in curr_snapshot.values() {
            *replica_dist.entry(replica_set.len()).or_insert(0) += 1;
        }
        metrics.per_tick_replica_dist.push(replica_dist);

        prev_snapshot = curr_snapshot;
    }

    metrics.finalize();
    metrics
}

/// Run the simulation with MCF algorithm using LoraController
fn run_mcf_simulation(config: &SimConfig, schedules: &[LoraLoadSchedule]) -> ChurnMetrics {
    let alloc_config = LoraAllocationConfig {
        enabled: true,
        algorithm: AllocationAlgorithmType::MinCostFlow,
        timestep_secs: 1,
        // See run_hrw_simulation: keep the comparison focused on routing-target selection.
        scale_down_cooldown_ticks: COMPARISON_SCALE_DOWN_COOLDOWN_TICKS,
        rate_window_multiplier: 5,
        ..Default::default()
    };

    let routing_table = LoraRoutingTable::new();
    let state_tracker = LoraStateTracker::new();
    let load_estimator = Arc::new(LoadEstimator::new());
    let mut controller = LoraController::new(
        alloc_config,
        routing_table.clone(),
        state_tracker.clone(),
        load_estimator.clone(),
    );

    // Register all workers
    let workers: Vec<WorkerWithDpRank> = (0..config.num_backends)
        .map(|i| WorkerWithDpRank::new(i as u64, 0))
        .collect();

    // Capacity only — no phantom adapter (see run_hrw_simulation).
    for &worker in &workers {
        state_tracker.set_worker_capacity(worker, config.slots_per_backend as u32);
    }

    let mut metrics = ChurnMetrics::new("MCF");
    let mut prev_snapshot: AllocationSnapshot = HashMap::new();
    for tick in 0..config.total_ticks {
        // Fully clear the previous tick's load signal (see run_hrw_simulation: `decrement_load`
        // leaves the windowed rate counter, which would otherwise accumulate across ticks).
        for name in load_estimator
            .get_current_load()
            .keys()
            .cloned()
            .collect::<Vec<_>>()
        {
            load_estimator.remove_lora(&name);
        }

        // See run_hrw_simulation: use the load estimator as the active-adapter source instead of
        // synthetic loaded locations, which would give HRW and MCF different capacity inputs.
        for schedule in schedules {
            let load = schedule.load_at_tick(tick);
            for _ in 0..load {
                load_estimator.increment_load(&schedule.lora_name);
            }
        }

        controller.recompute_now();

        let mut curr_snapshot: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name)
                && !config.replica_set.is_empty()
            {
                curr_snapshot.insert(lora_name, config.replica_set);
            }
        }
        let (loads, unloads) = compute_churn(&prev_snapshot, &curr_snapshot);
        let tick_churn = loads + unloads;
        metrics.total_target_additions += loads;
        metrics.total_target_removals += unloads;
        metrics.per_tick_churn.push(tick_churn);

        let prev_loras: HashSet<&String> = prev_snapshot.keys().collect();
        let curr_loras: HashSet<&String> = curr_snapshot.keys().collect();
        let additions = curr_loras.difference(&prev_loras).count();
        let removals = prev_loras.difference(&curr_loras).count();
        metrics.per_tick_lora_additions.push(additions);
        metrics.per_tick_lora_removals.push(removals);

        let mut replica_dist: HashMap<usize, usize> = HashMap::new();
        for replica_set in curr_snapshot.values() {
            *replica_dist.entry(replica_set.len()).or_insert(0) += 1;
        }
        metrics.per_tick_replica_dist.push(replica_dist);

        prev_snapshot = curr_snapshot;
    }

    metrics.finalize();
    metrics
}

// ============================================================================
// Print Helpers
// ============================================================================

fn print_simulation_header(config: &SimConfig) {
    println!("\n{}", "=".repeat(60));
    println!("LoRA Allocation Simulation");
    println!("{}", "=".repeat(60));
    println!("Configuration:");
    println!("  Backends (N):        {}", config.num_backends);
    println!("  Slots/Backend (K):   {}", config.slots_per_backend);
    println!(
        "  Total Slots (N×K):   {}",
        config.num_backends * config.slots_per_backend
    );
    println!("  Total LoRAs (L):     {}", config.total_loras);
    println!("  Concurrent LoRAs (C): {}", config.concurrent_loras);
    println!("  Total Ticks (T):     {}", config.total_ticks);
    if config.lifetime_mean > 0 {
        println!(
            "  Lifetime:            mean={} stddev={:.1} (uniform)",
            config.lifetime_mean, config.lifetime_stddev
        );
    } else {
        println!(
            "  Load Pattern:        ramp-up({}t) → steady({}t) → ramp-down({}t)",
            config.ramp_ticks, config.steady_ticks, config.ramp_down_ticks
        );
    }
    println!("  Max Load/LoRA:       {}", config.max_load_per_lora);
    println!(
        "  Scale-Down Cooldown: {} (comparison mode)",
        COMPARISON_SCALE_DOWN_COOLDOWN_TICKS
    );
    println!("  Seed:                {}", config.seed);
    println!();
}

fn print_comparison(hrw: &ChurnMetrics, random: &ChurnMetrics, mcf: &ChurnMetrics) {
    println!("{}", "=".repeat(72));
    println!("Comparison: HRW vs Random vs MCF");
    println!("{}", "=".repeat(72));

    println!("\nHRW Algorithm:");
    print!("{}", hrw);

    println!("\nRandom Algorithm:");
    print!("{}", random);

    println!("\nMCF Algorithm:");
    print!("{}", mcf);

    println!("\n{}", "-".repeat(72));
    println!("Churn Reduction:");

    fn pct_reduction(a: usize, b: usize) -> String {
        if b > 0 {
            format!("{:.1}%", (1.0 - a as f64 / b as f64) * 100.0)
        } else {
            "N/A".to_string()
        }
    }

    println!(
        "  {:>18}  {:>8}  {:>8}  {:>8}  {:>12}  {:>12}",
        "Metric", "HRW", "Random", "MCF", "MCF vs HRW", "MCF vs Rand"
    );
    println!(
        "  {:->18}  {:->8}  {:->8}  {:->8}  {:->12}  {:->12}",
        "", "", "", "", "", ""
    );
    println!(
        "  {:>18}  {:>8}  {:>8}  {:>8}  {:>12}  {:>12}",
        "Total Churn",
        hrw.total_churn,
        random.total_churn,
        mcf.total_churn,
        pct_reduction(mcf.total_churn, hrw.total_churn),
        pct_reduction(mcf.total_churn, random.total_churn),
    );
    println!(
        "  {:>18}  {:>8}  {:>8}  {:>8}  {:>12}  {:>12}",
        "Target Adds",
        hrw.total_target_additions,
        random.total_target_additions,
        mcf.total_target_additions,
        pct_reduction(mcf.total_target_additions, hrw.total_target_additions),
        pct_reduction(mcf.total_target_additions, random.total_target_additions),
    );
    println!(
        "  {:>18}  {:>8}  {:>8}  {:>8}",
        "Peak/Tick", hrw.peak_churn_per_tick, random.peak_churn_per_tick, mcf.peak_churn_per_tick,
    );
    println!(
        "  {:>18}  {:>8.2}  {:>8.2}  {:>8.2}",
        "Avg/Active Tick",
        hrw.avg_churn_per_active_tick,
        random.avg_churn_per_active_tick,
        mcf.avg_churn_per_active_tick,
    );
    println!();
}

fn print_per_tick_churn(
    hrw: &ChurnMetrics,
    random: &ChurnMetrics,
    mcf: &ChurnMetrics,
    total_ticks: usize,
) {
    println!("Per-Tick Churn Timeline:");
    println!(
        "  {:>4}  {:>8}  {:>8}  {:>8}",
        "Tick", "HRW", "Random", "MCF"
    );
    println!("  {:->4}  {:->8}  {:->8}  {:->8}", "", "", "", "");
    for tick in 0..total_ticks {
        let h = hrw.per_tick_churn.get(tick).copied().unwrap_or(0);
        let r = random.per_tick_churn.get(tick).copied().unwrap_or(0);
        let m = mcf.per_tick_churn.get(tick).copied().unwrap_or(0);
        if h > 0 || r > 0 || m > 0 {
            println!("  {:>4}  {:>8}  {:>8}  {:>8}", tick, h, r, m);
        }
    }
    println!();
}

#[path = "csv_export.rs"]
mod csv_export;
#[path = "scenario_tests.rs"]
mod scenario_tests;
#[path = "topology_tests.rs"]
mod topology_tests;
