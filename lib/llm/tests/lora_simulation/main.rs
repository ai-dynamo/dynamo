// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA Allocation Simulation / Churn Measurement
//!
//! A CPU-only integration test that simulates a cluster environment with
//! N backends (each having K LoRA slots), L distinct LoRAs, and measures
//! churn (load/unload operations) across various load patterns.
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
use dynamo_llm::model_card::LoraInfo;

use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;

// ============================================================================
// Simulation Configuration
// ============================================================================

/// Configuration for a simulation run
#[derive(Debug, Clone)]
struct SimConfig {
    /// Number of backend workers
    num_backends: usize,
    /// LoRA slots per backend
    slots_per_backend: usize,
    /// Total distinct LoRAs in the system
    total_loras: usize,
    /// Maximum concurrent active LoRAs at any point
    concurrent_loras: usize,
    /// Number of simulation ticks
    total_ticks: usize,
    /// Ticks for ramp-up phase per LoRA (used when lifetime_mean == 0)
    ramp_ticks: usize,
    /// Ticks for steady-state phase per LoRA (used when lifetime_mean == 0)
    steady_ticks: usize,
    /// Ticks for ramp-down phase per LoRA (used when lifetime_mean == 0)
    ramp_down_ticks: usize,
    /// Maximum load (active requests) per LoRA at peak
    max_load_per_lora: usize,
    /// Scale-down cooldown ticks (hysteresis)
    scale_down_cooldown_ticks: u32,
    /// Mean LoRA lifetime in ticks (0 = use ramp+steady+ramp_down directly).
    /// When > 0, each LoRA's lifetime is sampled from a uniform distribution
    /// with this mean and `lifetime_stddev` standard deviation.
    /// Phases are split proportionally: 20% ramp-up, 60% steady, 20% ramp-down.
    lifetime_mean: usize,
    /// Standard deviation of LoRA lifetime (uniform distribution).
    /// 0.0 = all LoRAs have exactly `lifetime_mean` ticks.
    /// For uniform U(a,b): stddev = (b-a) / sqrt(12), so half_range = stddev * sqrt(3).
    lifetime_stddev: f64,
    /// Random seed for reproducibility
    seed: u64,
}

impl SimConfig {
    /// Effective average lifetime in ticks.
    fn effective_lifetime(&self) -> usize {
        if self.lifetime_mean > 0 {
            self.lifetime_mean
        } else {
            self.ramp_ticks + self.steady_ticks + self.ramp_down_ticks
        }
    }
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            num_backends: 8,
            slots_per_backend: 4,
            total_loras: 20,
            concurrent_loras: 6,
            total_ticks: 60,
            ramp_ticks: 5,
            steady_ticks: 10,
            ramp_down_ticks: 5,
            max_load_per_lora: 20,
            scale_down_cooldown_ticks: 2,
            lifetime_mean: 0,
            lifetime_stddev: 0.0,
            seed: 42,
        }
    }
}

// ============================================================================
// Churn Metrics
// ============================================================================

/// Metrics collected during a simulation run
#[derive(Debug, Clone)]
struct ChurnMetrics {
    /// Algorithm name
    algorithm: String,
    /// Total LoRA load operations (LoRA added to a worker)
    total_loads: usize,
    /// Total LoRA unload operations (LoRA removed from a worker)
    total_unloads: usize,
    /// Total churn = loads + unloads
    total_churn: usize,
    /// Peak churn in a single tick
    peak_churn_per_tick: usize,
    /// Churn per tick (for analysis)
    per_tick_churn: Vec<usize>,
    /// Number of ticks where churn occurred
    ticks_with_churn: usize,
    /// Average churn per tick (only counting ticks with churn)
    avg_churn_per_active_tick: f64,
    /// Per-tick LoRA additions (new LoRA appeared in routing table)
    per_tick_lora_additions: Vec<usize>,
    /// Per-tick LoRA removals (LoRA disappeared from routing table)
    per_tick_lora_removals: Vec<usize>,
    /// Total distinct LoRAs that were added during the simulation
    total_lora_additions: usize,
    /// Total distinct LoRAs that were removed during the simulation
    total_lora_removals: usize,
    /// Per-tick replica distribution: `per_tick_replica_dist[tick]` is a map from
    /// replica_count → number of LoRAs with that replica count at that tick.
    per_tick_replica_dist: Vec<HashMap<usize, usize>>,
}

impl ChurnMetrics {
    fn new(algorithm: &str) -> Self {
        Self {
            algorithm: algorithm.to_string(),
            total_loads: 0,
            total_unloads: 0,
            total_churn: 0,
            peak_churn_per_tick: 0,
            per_tick_replica_dist: Vec::new(),
            per_tick_churn: Vec::new(),
            ticks_with_churn: 0,
            avg_churn_per_active_tick: 0.0,
            per_tick_lora_additions: Vec::new(),
            per_tick_lora_removals: Vec::new(),
            total_lora_additions: 0,
            total_lora_removals: 0,
        }
    }

    fn finalize(&mut self) {
        self.total_churn = self.total_loads + self.total_unloads;
        self.peak_churn_per_tick = self.per_tick_churn.iter().max().copied().unwrap_or(0);
        self.ticks_with_churn = self.per_tick_churn.iter().filter(|&&c| c > 0).count();
        self.avg_churn_per_active_tick = if self.ticks_with_churn > 0 {
            self.total_churn as f64 / self.ticks_with_churn as f64
        } else {
            0.0
        };
        self.total_lora_additions = self.per_tick_lora_additions.iter().sum();
        self.total_lora_removals = self.per_tick_lora_removals.iter().sum();
    }
}

impl std::fmt::Display for ChurnMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Algorithm:           {}", self.algorithm)?;
        writeln!(f, "  Total Loads:         {}", self.total_loads)?;
        writeln!(f, "  Total Unloads:       {}", self.total_unloads)?;
        writeln!(f, "  Total Churn:         {}", self.total_churn)?;
        writeln!(f, "  Peak Churn/Tick:     {}", self.peak_churn_per_tick)?;
        writeln!(f, "  Ticks with Churn:    {}", self.ticks_with_churn)?;
        writeln!(
            f,
            "  Avg Churn/Active Tick: {:.2}",
            self.avg_churn_per_active_tick
        )?;
        writeln!(f, "  LoRA Additions:      {}", self.total_lora_additions)?;
        writeln!(f, "  LoRA Removals:       {}", self.total_lora_removals)?;
        Ok(())
    }
}

// ============================================================================
// Load Pattern Generator
// ============================================================================

/// Represents the load for a single LoRA at a given tick
#[derive(Debug, Clone)]
struct LoraLoadSchedule {
    lora_name: String,
    /// (start_tick, end_tick) for this LoRA's active period
    active_window: (usize, usize),
    /// Peak load during steady state
    peak_load: usize,
    /// Ramp-up ticks
    ramp_up: usize,
    /// Steady ticks
    steady: usize,
    /// Ramp-down ticks
    ramp_down: usize,
    /// Optional per-tick load override. When set, `load_at_tick` returns
    /// `per_tick_loads[tick]` instead of computing from the window model.
    per_tick_loads: Option<Vec<usize>>,
}

impl LoraLoadSchedule {
    /// Get the load for this LoRA at a given tick
    fn load_at_tick(&self, tick: usize) -> usize {
        // Per-tick override takes precedence
        if let Some(ref loads) = self.per_tick_loads {
            return loads.get(tick).copied().unwrap_or(0);
        }

        if tick < self.active_window.0 || tick >= self.active_window.1 {
            return 0;
        }

        let relative_tick = tick - self.active_window.0;
        let total_active = self.ramp_up + self.steady + self.ramp_down;

        if relative_tick >= total_active {
            return 0;
        }

        if relative_tick < self.ramp_up {
            // Ramp up: linearly increase from 1 to peak_load
            let progress = (relative_tick + 1) as f64 / self.ramp_up as f64;
            (progress * self.peak_load as f64).ceil() as usize
        } else if relative_tick < self.ramp_up + self.steady {
            // Steady state
            self.peak_load
        } else {
            // Ramp down: linearly decrease from peak_load to 0
            let ramp_down_tick = relative_tick - self.ramp_up - self.steady;
            let progress = 1.0 - ((ramp_down_tick + 1) as f64 / self.ramp_down as f64);
            ((progress * self.peak_load as f64).ceil() as usize).max(0)
        }
    }
}

/// Sample from a Poisson distribution using Knuth's algorithm.
///
/// For lambda <= 30 uses the classic inverse-transform method;
/// for larger lambda falls back to the normal approximation.
fn sample_poisson(rng: &mut StdRng, lambda: f64) -> usize {
    if lambda <= 0.0 {
        return 0;
    }
    if lambda > 30.0 {
        // Normal approximation: Poisson(λ) ≈ N(λ, λ)
        let normal = lambda + lambda.sqrt() * rng.random_range(-3.0f64..3.0f64);
        return normal.round().max(0.0) as usize;
    }
    // Knuth's algorithm
    let l = (-lambda).exp();
    let mut k: usize = 0;
    let mut p: f64 = 1.0;
    loop {
        k += 1;
        p *= rng.random::<f64>();
        if p < l {
            break;
        }
    }
    k - 1
}

/// Compute the generalized harmonic number H(n, s) = sum_{k=1}^{n} 1/k^s.
fn harmonic_number(n: usize, s: f64) -> f64 {
    (1..=n).map(|k| 1.0 / (k as f64).powf(s)).sum()
}

/// Sample a LoRA lifetime and split it into (ramp_up, steady, ramp_down).
///
/// Proportions: 20% ramp-up, 60% steady, 20% ramp-down (minimum 1 tick each).
fn sample_lifetime(config: &SimConfig, rng: &mut StdRng) -> (usize, usize, usize) {
    let lifetime = if config.lifetime_mean > 0 {
        if config.lifetime_stddev > 0.0 {
            // Uniform distribution: half_range = stddev * sqrt(3)
            let half_range = config.lifetime_stddev * 3.0_f64.sqrt();
            let lo = (config.lifetime_mean as f64 - half_range).max(3.0);
            let hi = config.lifetime_mean as f64 + half_range;
            rng.random_range(lo as usize..=hi as usize).max(3)
        } else {
            config.lifetime_mean
        }
    } else {
        config.ramp_ticks + config.steady_ticks + config.ramp_down_ticks
    };

    if config.lifetime_mean > 0 {
        // Proportional split: 20% ramp, 60% steady, 20% ramp_down
        let ramp_up = (lifetime as f64 * 0.20).round().max(1.0) as usize;
        let ramp_down = (lifetime as f64 * 0.20).round().max(1.0) as usize;
        let steady = lifetime.saturating_sub(ramp_up + ramp_down).max(1);
        (ramp_up, steady, ramp_down)
    } else {
        (
            config.ramp_ticks,
            config.steady_ticks,
            config.ramp_down_ticks,
        )
    }
}

/// Generate load schedules for all LoRAs with staggered activation.
///
/// Every LoRA has the same request shape (ramp-up → steady → ramp-down)
/// and the same peak load. Activations are staggered so that at most
/// `concurrent_loras` are active simultaneously.
///
/// When `lifetime_mean > 0`, each LoRA's lifetime is sampled from a
/// uniform distribution with the specified mean and stddev.
///
/// Uses fractional spacing so that when C > lifetime, multiple LoRAs
/// can start in the same tick (e.g. lifetime=10, C=20 → ~2 starts/tick).
///
/// LoRAs whose start tick would fall beyond `total_ticks` are simply
/// not generated (they wouldn't contribute any load).
fn generate_load_schedules(config: &SimConfig) -> Vec<LoraLoadSchedule> {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut schedules = Vec::new();

    let avg_lifetime = config.effective_lifetime() as f64;
    let c = config.concurrent_loras.max(1) as f64;

    // Fractional spacing: lifetime / C
    // When C < lifetime → spacing > 1 (spread out)
    // When C > lifetime → spacing < 1 (multiple starts per tick)
    let spacing = avg_lifetime / c;

    for i in 0..config.total_loras {
        let start_tick = (i as f64 * spacing) as usize;

        // Skip LoRAs that would start after the simulation ends
        if start_tick >= config.total_ticks {
            break;
        }

        let (ramp_up, steady, ramp_down) = sample_lifetime(config, &mut rng);
        let active_duration = ramp_up + steady + ramp_down;

        // The active window may be truncated at the end of the simulation
        let end_tick = (start_tick + active_duration).min(config.total_ticks);

        schedules.push(LoraLoadSchedule {
            lora_name: format!("lora-{:03}", i),
            active_window: (start_tick, end_tick),
            peak_load: config.max_load_per_lora,
            ramp_up,
            steady,
            ramp_down,
            per_tick_loads: None,
        });
    }

    schedules
}

/// Generate load schedules using Zipf popularity + Poisson arrivals.
///
/// Each of the `total_loras` adapters has a Zipf-distributed popularity
/// weight:  w_k = 1/k^s  (rank k, exponent s).  At every tick the load
/// for LoRA k is drawn independently from Poisson(λ_k) where
/// λ_k = avg_total_load × w_k / H(L,s).
///
/// This produces a realistic skewed workload: a few "hot" adapters carry
/// most traffic while a long tail of cold adapters flicker in and out.
///
/// The `active_window` for each LoRA covers the full simulation so that
/// the stochastic appearance / disappearance is driven entirely by
/// whether the Poisson draw is > 0.
fn generate_zipf_poisson_schedules(
    total_loras: usize,
    total_ticks: usize,
    zipf_s: f64,
    avg_total_load: f64,
    seed: u64,
) -> Vec<LoraLoadSchedule> {
    let mut rng = StdRng::seed_from_u64(seed);

    let h = harmonic_number(total_loras, zipf_s);

    // Compute per-LoRA Poisson rate
    let lambdas: Vec<f64> = (1..=total_loras)
        .map(|k| avg_total_load / ((k as f64).powf(zipf_s) * h))
        .collect();

    // Generate per-tick loads
    let mut schedules: Vec<LoraLoadSchedule> = Vec::with_capacity(total_loras);
    for (idx, &lambda) in lambdas.iter().enumerate() {
        let mut per_tick = Vec::with_capacity(total_ticks);
        let mut peak = 0usize;
        let mut first_active = total_ticks; // track first tick with load > 0
        let mut last_active = 0usize;

        for _tick in 0..total_ticks {
            let load = sample_poisson(&mut rng, lambda);
            if load > 0 {
                if _tick < first_active {
                    first_active = _tick;
                }
                last_active = _tick;
                peak = peak.max(load);
            }
            per_tick.push(load);
        }

        // Only include LoRAs that had at least one active tick
        if first_active < total_ticks {
            schedules.push(LoraLoadSchedule {
                lora_name: format!("lora-{:03}", idx),
                active_window: (first_active, last_active + 1),
                peak_load: peak,
                ramp_up: 0,
                steady: 0,
                ramp_down: 0,
                per_tick_loads: Some(per_tick),
            });
        }
    }

    // Sort by rank (index) for stable ordering
    schedules.sort_by_key(|s| s.lora_name.clone());
    schedules
}

/// Generate load schedules with a diurnal (time-of-day) pattern.
///
/// Combines Zipf popularity with a sinusoidal daily load envelope:
///   total_rate(t) = trough + (peak - trough) * (1 - cos(2π·(t mod T)/T)) / 2
///
/// This gives a smooth cycle: trough at midnight (t=0), peak at noon (t=T/2).
/// Individual LoRA loads are Poisson(zipf_weight * total_rate(t)).
///
/// With `ticks_per_day=100` and `total_ticks=200`, the simulation covers
/// two full day/night cycles showing how each algorithm handles the
/// gradual ramp-up and ramp-down of traffic.
fn generate_diurnal_schedules(
    total_loras: usize,
    total_ticks: usize,
    ticks_per_day: usize,
    zipf_s: f64,
    peak_total_load: f64,
    trough_total_load: f64,
    seed: u64,
) -> Vec<LoraLoadSchedule> {
    let mut rng = StdRng::seed_from_u64(seed);
    let h = harmonic_number(total_loras, zipf_s);

    // Zipf weights (unnormalized rates, will be scaled by diurnal envelope)
    let weights: Vec<f64> = (1..=total_loras)
        .map(|k| 1.0 / ((k as f64).powf(zipf_s) * h))
        .collect();

    let amplitude = (peak_total_load - trough_total_load) / 2.0;
    let baseline = (peak_total_load + trough_total_load) / 2.0;

    let mut schedules: Vec<LoraLoadSchedule> = Vec::with_capacity(total_loras);
    for (idx, &w) in weights.iter().enumerate() {
        let mut per_tick = Vec::with_capacity(total_ticks);
        let mut peak = 0usize;
        let mut first_active = total_ticks;
        let mut last_active = 0usize;

        for tick in 0..total_ticks {
            // Diurnal envelope: trough at t=0 (midnight), peak at t=T/2 (noon)
            let phase =
                2.0 * std::f64::consts::PI * (tick % ticks_per_day) as f64 / ticks_per_day as f64;
            let total_rate = baseline - amplitude * phase.cos(); // ranges [trough, peak]
            let lambda = total_rate * w;
            let load = sample_poisson(&mut rng, lambda);

            if load > 0 {
                if tick < first_active {
                    first_active = tick;
                }
                last_active = tick;
                peak = peak.max(load);
            }
            per_tick.push(load);
        }

        if first_active < total_ticks {
            schedules.push(LoraLoadSchedule {
                lora_name: format!("lora-{:03}", idx),
                active_window: (first_active, last_active + 1),
                peak_load: peak,
                ramp_up: 0,
                steady: 0,
                ramp_down: 0,
                per_tick_loads: Some(per_tick),
            });
        }
    }

    schedules.sort_by_key(|s| s.lora_name.clone());
    schedules
}

/// Generate load schedules simulating a flash-crowd (viral spike) event.
///
/// Baseline traffic follows Zipf+Poisson at `base_total_load`.  At each
/// tick in `flash_ticks`, the total rate jumps to `spike_multiplier × base`
/// and then decays exponentially with the given `decay_half_life` (in ticks).
///
/// This models viral events: a sudden surge of requests to previously-cold
/// LoRAs, followed by a rapid taper.  The key stress test: how much
/// unnecessary churn does each algorithm produce during the spike and
/// how quickly does it stabilize during the decay?
fn generate_flash_crowd_schedules(
    total_loras: usize,
    total_ticks: usize,
    zipf_s: f64,
    base_total_load: f64,
    spike_multiplier: f64,
    decay_half_life: f64,
    flash_ticks: &[usize],
    seed: u64,
) -> Vec<LoraLoadSchedule> {
    let mut rng = StdRng::seed_from_u64(seed);
    let h = harmonic_number(total_loras, zipf_s);

    let weights: Vec<f64> = (1..=total_loras)
        .map(|k| 1.0 / ((k as f64).powf(zipf_s) * h))
        .collect();

    // Pre-compute the rate multiplier for each tick.
    // At each tick the multiplier is: 1 + sum over past flashes of
    //   (spike_multiplier - 1) * 2^(-(t - flash_t) / half_life)
    let decay_rate = (0.5_f64).ln() / decay_half_life; // negative
    let mut multipliers = vec![1.0_f64; total_ticks];
    for &ft in flash_ticks {
        for t in ft..total_ticks {
            let elapsed = (t - ft) as f64;
            multipliers[t] += (spike_multiplier - 1.0) * (decay_rate * elapsed).exp();
        }
    }

    let mut schedules: Vec<LoraLoadSchedule> = Vec::with_capacity(total_loras);
    for (idx, &w) in weights.iter().enumerate() {
        let mut per_tick = Vec::with_capacity(total_ticks);
        let mut peak = 0usize;
        let mut first_active = total_ticks;
        let mut last_active = 0usize;

        for (tick, &mult) in multipliers.iter().enumerate() {
            let lambda = base_total_load * mult * w;
            let load = sample_poisson(&mut rng, lambda);

            if load > 0 {
                if tick < first_active {
                    first_active = tick;
                }
                last_active = tick;
                peak = peak.max(load);
            }
            per_tick.push(load);
        }

        if first_active < total_ticks {
            schedules.push(LoraLoadSchedule {
                lora_name: format!("lora-{:03}", idx),
                active_window: (first_active, last_active + 1),
                peak_load: peak,
                ramp_up: 0,
                steady: 0,
                ramp_down: 0,
                per_tick_loads: Some(per_tick),
            });
        }
    }

    schedules.sort_by_key(|s| s.lora_name.clone());
    schedules
}

/// Generate load schedules using a Markov-Modulated Poisson Process (MMPP).
///
/// The system transitions between discrete states (e.g. calm / busy / surge)
/// according to a Markov chain.  Each state has its own total Poisson rate,
/// and individual LoRA loads are Zipf-weighted draws from that rate.
///
/// This produces **bursty, temporally-correlated** traffic: the system
/// dwells in a state for many ticks (geometric duration), then transitions
/// to another, causing step-changes in the load level.  The key stress
/// test: algorithms must handle both the steady periods (should be zero
/// churn) and the sudden state transitions (should be minimal churn).
fn generate_mmpp_schedules(
    total_loras: usize,
    total_ticks: usize,
    zipf_s: f64,
    state_rates: &[f64],            // total Poisson rate for each state
    transition_matrix: &[Vec<f64>], // row-stochastic transition probabilities
    seed: u64,
) -> (Vec<LoraLoadSchedule>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let h = harmonic_number(total_loras, zipf_s);
    let n_states = state_rates.len();

    let weights: Vec<f64> = (1..=total_loras)
        .map(|k| 1.0 / ((k as f64).powf(zipf_s) * h))
        .collect();

    // Simulate Markov chain to get state sequence
    let mut state_seq = Vec::with_capacity(total_ticks);
    let mut current_state = 0usize; // start in first state (calm)
    for _ in 0..total_ticks {
        state_seq.push(current_state);
        // Transition
        let r: f64 = rng.random();
        let mut cumulative = 0.0;
        for next in 0..n_states {
            cumulative += transition_matrix[current_state][next];
            if r < cumulative {
                current_state = next;
                break;
            }
        }
    }

    // Generate per-LoRA loads
    let mut schedules: Vec<LoraLoadSchedule> = Vec::with_capacity(total_loras);
    for (idx, &w) in weights.iter().enumerate() {
        let mut per_tick = Vec::with_capacity(total_ticks);
        let mut peak = 0usize;
        let mut first_active = total_ticks;
        let mut last_active = 0usize;

        for (tick, &state) in state_seq.iter().enumerate() {
            let lambda = state_rates[state] * w;
            let load = sample_poisson(&mut rng, lambda);

            if load > 0 {
                if tick < first_active {
                    first_active = tick;
                }
                last_active = tick;
                peak = peak.max(load);
            }
            per_tick.push(load);
        }

        if first_active < total_ticks {
            schedules.push(LoraLoadSchedule {
                lora_name: format!("lora-{:03}", idx),
                active_window: (first_active, last_active + 1),
                peak_load: peak,
                ramp_up: 0,
                steady: 0,
                ramp_down: 0,
                per_tick_loads: Some(per_tick),
            });
        }
    }

    schedules.sort_by_key(|s| s.lora_name.clone());
    (schedules, state_seq)
}

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

        // Filter workers with available capacity
        let available: Vec<WorkerWithDpRank> = workers
            .iter()
            .filter(|w| {
                if let Some(&(used, cap)) = worker_slot_usage.get(w) {
                    used < cap
                } else {
                    true // New worker, assume has capacity
                }
            })
            .copied()
            .collect();

        if available.is_empty() {
            // Fall back to any workers if all are full
            let mut shuffled = workers.to_vec();
            shuffled.shuffle(&mut self.rng);
            return shuffled.into_iter().take(replica_factor).collect();
        }

        let mut shuffled = available;
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

/// Run the simulation with HRW algorithm using LoraController
fn run_hrw_simulation(config: &SimConfig, schedules: &[LoraLoadSchedule]) -> ChurnMetrics {
    let alloc_config = LoraAllocationConfig {
        enabled: true,
        algorithm: AllocationAlgorithmType::Hrw,
        timestep_secs: 1, // Not used in sync mode
        scale_down_cooldown_ticks: config.scale_down_cooldown_ticks,
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

    // Register capacity for each worker (use a dummy LoRA to set capacity)
    for &worker in &workers {
        state_tracker.handle_mdc_update(
            worker,
            &LoraInfo {
                name: format!("__capacity_init_{}", worker.worker_id),
                max_gpu_lora_count: Some(config.slots_per_backend as u32),
            },
        );
    }

    let mut metrics = ChurnMetrics::new("HRW");
    let mut prev_snapshot: AllocationSnapshot = HashMap::new();

    // Track which LoRAs are currently registered on the state tracker
    let mut registered_loras: HashSet<String> = HashSet::new();

    for tick in 0..config.total_ticks {
        // Clear all loads from previous tick
        // (We use absolute counts, not increments)
        let prev_loads = load_estimator.get_current_load();
        for (name, count) in &prev_loads {
            for _ in 0..*count {
                load_estimator.decrement_load(name);
            }
        }

        // Set loads for this tick
        for schedule in schedules {
            let load = schedule.load_at_tick(tick);
            for _ in 0..load {
                load_estimator.increment_load(&schedule.lora_name);
            }

            // Register the LoRA on state tracker when it first becomes active
            if load > 0 && !registered_loras.contains(&schedule.lora_name) {
                state_tracker.handle_mdc_update(
                    workers[0],
                    &LoraInfo {
                        name: schedule.lora_name.clone(),
                        max_gpu_lora_count: Some(config.slots_per_backend as u32),
                    },
                );
                registered_loras.insert(schedule.lora_name.clone());
            }

            // Unregister the LoRA from state tracker when its active window ends
            if registered_loras.contains(&schedule.lora_name) && tick >= schedule.active_window.1 {
                state_tracker.handle_mdc_removal(workers[0], &schedule.lora_name);
                load_estimator.remove_lora(&schedule.lora_name);
                registered_loras.remove(&schedule.lora_name);
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
        metrics.total_loads += loads;
        metrics.total_unloads += unloads;
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

    let total_slots = config.num_backends * config.slots_per_backend;

    let mut metrics = ChurnMetrics::new("Random");
    let mut prev_snapshot: AllocationSnapshot = HashMap::new();

    for tick in 0..config.total_ticks {
        // Compute loads for this tick
        let mut active_loads: Vec<(String, usize)> = Vec::new();
        let mut inactive_loras: Vec<String> = Vec::new();
        let mut total_load: usize = 0;

        for schedule in schedules {
            let load = schedule.load_at_tick(tick);
            if load > 0 {
                active_loads.push((schedule.lora_name.clone(), load));
                total_load += load;
            } else if tick >= schedule.active_window.0 && tick < schedule.active_window.1 {
                // LoRA exists but has no load (ramp-down completed within window)
                inactive_loras.push(schedule.lora_name.clone());
            }
        }

        // Compute replica counts (same formula as controller)
        let mut curr_snapshot: AllocationSnapshot = HashMap::new();

        // Track slot usage for capacity-aware allocation
        let mut worker_slot_usage: HashMap<WorkerWithDpRank, (usize, usize)> = workers
            .iter()
            .map(|&w| (w, (0, config.slots_per_backend)))
            .collect();

        // Active LoRAs: proportional allocation with random placement
        if total_load > 0 && total_slots > 0 {
            for (lora_name, load) in &active_loads {
                let fraction = *load as f64 / total_load as f64;
                let desired = (fraction * total_slots as f64)
                    .ceil()
                    .max(1.0)
                    .min(workers.len() as f64) as usize;

                let replica_set = random_allocator.compute_replica_set(
                    lora_name,
                    &workers,
                    desired,
                    &worker_slot_usage,
                );

                // Update slot usage
                for w in &replica_set {
                    if let Some((used, _)) = worker_slot_usage.get_mut(w) {
                        *used += 1;
                    }
                }

                curr_snapshot.insert(lora_name.clone(), replica_set);
            }
        }

        // Inactive LoRAs: single random worker
        for lora_name in &inactive_loras {
            let replica_set =
                random_allocator.compute_replica_set(lora_name, &workers, 1, &worker_slot_usage);
            if !replica_set.is_empty() {
                if let Some((used, _)) = worker_slot_usage.get_mut(&replica_set[0]) {
                    *used += 1;
                }
                curr_snapshot.insert(lora_name.clone(), replica_set);
            }
        }

        // Compute churn
        let (loads, unloads) = compute_churn(&prev_snapshot, &curr_snapshot);
        let tick_churn = loads + unloads;
        metrics.total_loads += loads;
        metrics.total_unloads += unloads;
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
        scale_down_cooldown_ticks: config.scale_down_cooldown_ticks,
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

    for &worker in &workers {
        state_tracker.handle_mdc_update(
            worker,
            &LoraInfo {
                name: format!("__capacity_init_{}", worker.worker_id),
                max_gpu_lora_count: Some(config.slots_per_backend as u32),
            },
        );
    }

    let mut metrics = ChurnMetrics::new("MCF");
    let mut prev_snapshot: AllocationSnapshot = HashMap::new();
    let mut registered_loras: HashSet<String> = HashSet::new();

    for tick in 0..config.total_ticks {
        let prev_loads = load_estimator.get_current_load();
        for (name, count) in &prev_loads {
            for _ in 0..*count {
                load_estimator.decrement_load(name);
            }
        }

        for schedule in schedules {
            let load = schedule.load_at_tick(tick);
            for _ in 0..load {
                load_estimator.increment_load(&schedule.lora_name);
            }

            if load > 0 && !registered_loras.contains(&schedule.lora_name) {
                state_tracker.handle_mdc_update(
                    workers[0],
                    &LoraInfo {
                        name: schedule.lora_name.clone(),
                        max_gpu_lora_count: Some(config.slots_per_backend as u32),
                    },
                );
                registered_loras.insert(schedule.lora_name.clone());
            }

            if registered_loras.contains(&schedule.lora_name) && tick >= schedule.active_window.1 {
                state_tracker.handle_mdc_removal(workers[0], &schedule.lora_name);
                load_estimator.remove_lora(&schedule.lora_name);
                registered_loras.remove(&schedule.lora_name);
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
        metrics.total_loads += loads;
        metrics.total_unloads += unloads;
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
        "  Scale-Down Cooldown: {} ticks",
        config.scale_down_cooldown_ticks
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
        "Total Loads",
        hrw.total_loads,
        random.total_loads,
        mcf.total_loads,
        pct_reduction(mcf.total_loads, hrw.total_loads),
        pct_reduction(mcf.total_loads, random.total_loads),
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

// ============================================================================
// Test Cases
// ============================================================================

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

    let schedules = generate_load_schedules(&config);
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
fn test_simulation_worker_addition() {
    // Test that adding a worker causes minimal churn with HRW vs Random
    let num_workers: usize = 3;
    let num_loras: usize = 4;
    let slots_per_worker: usize = 4;
    let load_per_lora: usize = 5;

    // --- HRW ---
    let hrw_churn = {
        let alloc_config = LoraAllocationConfig {
            enabled: true,
            algorithm: AllocationAlgorithmType::Hrw,
            timestep_secs: 1,
            scale_down_cooldown_ticks: 0,
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

        for i in 0..num_workers {
            let w = WorkerWithDpRank::new(i as u64, 0);
            state_tracker.handle_mdc_update(
                w,
                &LoraInfo {
                    name: format!("lora-{}", i % num_loras),
                    max_gpu_lora_count: Some(slots_per_worker as u32),
                },
            );
        }
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            for _ in 0..load_per_lora {
                load_estimator.increment_load(&name);
            }
        }
        controller.recompute_now();

        let mut before: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                before.insert(lora_name, config.replica_set);
            }
        }

        // Add a new worker
        state_tracker.handle_mdc_update(
            WorkerWithDpRank::new(num_workers as u64, 0),
            &LoraInfo {
                name: "lora-0".to_string(),
                max_gpu_lora_count: Some(slots_per_worker as u32),
            },
        );
        controller.recompute_now();

        let mut after: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                after.insert(lora_name, config.replica_set);
            }
        }

        let (loads, unloads) = compute_churn(&before, &after);
        (loads, unloads)
    };

    // --- Random ---
    let random_churn = {
        let workers: Vec<WorkerWithDpRank> = (0..num_workers)
            .map(|i| WorkerWithDpRank::new(i as u64, 0))
            .collect();
        let total_slots = num_workers * slots_per_worker;
        let total_load = num_loras * load_per_lora;
        let mut rng = RandomAllocator::new(42);

        let mut worker_slot_usage: HashMap<WorkerWithDpRank, (usize, usize)> = workers
            .iter()
            .map(|&w| (w, (0, slots_per_worker)))
            .collect();

        // Allocate before
        let mut before: AllocationSnapshot = HashMap::new();
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            let fraction = load_per_lora as f64 / total_load as f64;
            let desired = (fraction * total_slots as f64)
                .ceil()
                .max(1.0)
                .min(workers.len() as f64) as usize;
            let set = rng.compute_replica_set(&name, &workers, desired, &worker_slot_usage);
            for w in &set {
                if let Some((used, _)) = worker_slot_usage.get_mut(w) {
                    *used += 1;
                }
            }
            before.insert(name, set);
        }

        // Add new worker
        let mut new_workers = workers.clone();
        new_workers.push(WorkerWithDpRank::new(num_workers as u64, 0));
        let new_total_slots = (num_workers + 1) * slots_per_worker;
        let mut new_slot_usage: HashMap<WorkerWithDpRank, (usize, usize)> = new_workers
            .iter()
            .map(|&w| (w, (0, slots_per_worker)))
            .collect();

        // Reallocate after
        let mut after: AllocationSnapshot = HashMap::new();
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            let fraction = load_per_lora as f64 / total_load as f64;
            let desired = (fraction * new_total_slots as f64)
                .ceil()
                .max(1.0)
                .min(new_workers.len() as f64) as usize;
            let set = rng.compute_replica_set(&name, &new_workers, desired, &new_slot_usage);
            for w in &set {
                if let Some((used, _)) = new_slot_usage.get_mut(w) {
                    *used += 1;
                }
            }
            after.insert(name, set);
        }

        let (loads, unloads) = compute_churn(&before, &after);
        (loads, unloads)
    };

    // --- MCF ---
    let mcf_churn = {
        let alloc_config = LoraAllocationConfig {
            enabled: true,
            algorithm: AllocationAlgorithmType::MinCostFlow,
            timestep_secs: 1,
            scale_down_cooldown_ticks: 0,
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

        for i in 0..num_workers {
            let w = WorkerWithDpRank::new(i as u64, 0);
            state_tracker.handle_mdc_update(
                w,
                &LoraInfo {
                    name: format!("lora-{}", i % num_loras),
                    max_gpu_lora_count: Some(slots_per_worker as u32),
                },
            );
        }
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            for _ in 0..load_per_lora {
                load_estimator.increment_load(&name);
            }
        }
        controller.recompute_now();

        let mut before: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                before.insert(lora_name, config.replica_set);
            }
        }

        // Add a new worker
        state_tracker.handle_mdc_update(
            WorkerWithDpRank::new(num_workers as u64, 0),
            &LoraInfo {
                name: "lora-0".to_string(),
                max_gpu_lora_count: Some(slots_per_worker as u32),
            },
        );
        controller.recompute_now();

        let mut after: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                after.insert(lora_name, config.replica_set);
            }
        }

        let (loads, unloads) = compute_churn(&before, &after);
        (loads, unloads)
    };

    let hrw_total = hrw_churn.0 + hrw_churn.1;
    let random_total = random_churn.0 + random_churn.1;
    let mcf_total = mcf_churn.0 + mcf_churn.1;

    println!("\n{}", "=".repeat(72));
    println!(
        "Worker Addition Churn (3 → 4 workers, {} LoRAs, {} slots/worker)",
        num_loras, slots_per_worker
    );
    println!("{}", "=".repeat(72));
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Metric", "HRW", "Random", "MCF"
    );
    println!("  {:->12}  {:->8}  {:->8}  {:->8}", "", "", "", "");
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Loads", hrw_churn.0, random_churn.0, mcf_churn.0
    );
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Unloads", hrw_churn.1, random_churn.1, mcf_churn.1
    );
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Total", hrw_total, random_total, mcf_total
    );
    println!();

    // HRW should have bounded churn
    assert!(
        hrw_total <= num_loras * 2,
        "HRW worker addition churn ({}) should be bounded by 2x num_loras ({})",
        hrw_total,
        num_loras * 2
    );
    // MCF should have bounded churn
    assert!(
        mcf_total <= num_loras * 2,
        "MCF worker addition churn ({}) should be bounded by 2x num_loras ({})",
        mcf_total,
        num_loras * 2
    );
}

#[test]
fn test_simulation_worker_removal() {
    // Test that removing a worker causes minimal churn with HRW vs Random vs MCF
    let num_workers: usize = 5;
    let num_loras: usize = 6;
    let slots_per_worker: usize = 4;
    let load_per_lora: usize = 3;
    let removed_worker_id: u64 = 2;

    // --- HRW ---
    let hrw_churn = {
        let alloc_config = LoraAllocationConfig {
            enabled: true,
            algorithm: AllocationAlgorithmType::Hrw,
            timestep_secs: 1,
            scale_down_cooldown_ticks: 0,
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

        for i in 0..num_workers {
            let w = WorkerWithDpRank::new(i as u64, 0);
            state_tracker.handle_mdc_update(
                w,
                &LoraInfo {
                    name: format!("lora-{}", i % num_loras),
                    max_gpu_lora_count: Some(slots_per_worker as u32),
                },
            );
        }
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            for _ in 0..load_per_lora {
                load_estimator.increment_load(&name);
            }
        }
        controller.recompute_now();

        let mut before: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                before.insert(lora_name, config.replica_set);
            }
        }

        // Remove worker
        state_tracker.handle_worker_removal(WorkerWithDpRank::new(removed_worker_id, 0));
        controller.recompute_now();

        let mut after: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                after.insert(lora_name, config.replica_set);
            }
        }

        // Verify removed worker is not in any replica set
        for (lora_name, workers) in &after {
            for w in workers {
                assert!(
                    w.worker_id != removed_worker_id,
                    "Removed worker {} should not be in replica set for {}",
                    removed_worker_id,
                    lora_name
                );
            }
        }

        compute_churn(&before, &after)
    };

    // --- Random ---
    let random_churn = {
        let workers: Vec<WorkerWithDpRank> = (0..num_workers)
            .map(|i| WorkerWithDpRank::new(i as u64, 0))
            .collect();
        let total_slots = num_workers * slots_per_worker;
        let total_load = num_loras * load_per_lora;
        let mut rng = RandomAllocator::new(42);

        let mut worker_slot_usage: HashMap<WorkerWithDpRank, (usize, usize)> = workers
            .iter()
            .map(|&w| (w, (0, slots_per_worker)))
            .collect();

        // Allocate before
        let mut before: AllocationSnapshot = HashMap::new();
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            let fraction = load_per_lora as f64 / total_load as f64;
            let desired = (fraction * total_slots as f64)
                .ceil()
                .max(1.0)
                .min(workers.len() as f64) as usize;
            let set = rng.compute_replica_set(&name, &workers, desired, &worker_slot_usage);
            for w in &set {
                if let Some((used, _)) = worker_slot_usage.get_mut(w) {
                    *used += 1;
                }
            }
            before.insert(name, set);
        }

        // Remove worker — allocate on reduced set
        let remaining_workers: Vec<WorkerWithDpRank> = workers
            .iter()
            .filter(|w| w.worker_id != removed_worker_id)
            .copied()
            .collect();
        let new_total_slots = remaining_workers.len() * slots_per_worker;
        let mut new_slot_usage: HashMap<WorkerWithDpRank, (usize, usize)> = remaining_workers
            .iter()
            .map(|&w| (w, (0, slots_per_worker)))
            .collect();

        let mut after: AllocationSnapshot = HashMap::new();
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            let fraction = load_per_lora as f64 / total_load as f64;
            let desired = (fraction * new_total_slots as f64)
                .ceil()
                .max(1.0)
                .min(remaining_workers.len() as f64) as usize;
            let set = rng.compute_replica_set(&name, &remaining_workers, desired, &new_slot_usage);
            for w in &set {
                if let Some((used, _)) = new_slot_usage.get_mut(w) {
                    *used += 1;
                }
            }
            after.insert(name, set);
        }

        compute_churn(&before, &after)
    };

    // --- MCF ---
    let mcf_churn = {
        let alloc_config = LoraAllocationConfig {
            enabled: true,
            algorithm: AllocationAlgorithmType::MinCostFlow,
            timestep_secs: 1,
            scale_down_cooldown_ticks: 0,
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

        for i in 0..num_workers {
            let w = WorkerWithDpRank::new(i as u64, 0);
            state_tracker.handle_mdc_update(
                w,
                &LoraInfo {
                    name: format!("lora-{}", i % num_loras),
                    max_gpu_lora_count: Some(slots_per_worker as u32),
                },
            );
        }
        for i in 0..num_loras {
            let name = format!("lora-{}", i);
            for _ in 0..load_per_lora {
                load_estimator.increment_load(&name);
            }
        }
        controller.recompute_now();

        let mut before: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                before.insert(lora_name, config.replica_set);
            }
        }

        // Remove worker
        state_tracker.handle_worker_removal(WorkerWithDpRank::new(removed_worker_id, 0));
        controller.recompute_now();

        let mut after: AllocationSnapshot = HashMap::new();
        for lora_name in routing_table.list_loras() {
            if let Some(config) = routing_table.get_config(&lora_name) {
                after.insert(lora_name, config.replica_set);
            }
        }

        // Verify removed worker is not in any replica set
        for (lora_name, workers) in &after {
            for w in workers {
                assert!(
                    w.worker_id != removed_worker_id,
                    "MCF: Removed worker {} should not be in replica set for {}",
                    removed_worker_id,
                    lora_name
                );
            }
        }

        compute_churn(&before, &after)
    };

    let hrw_total = hrw_churn.0 + hrw_churn.1;
    let random_total = random_churn.0 + random_churn.1;
    let mcf_total = mcf_churn.0 + mcf_churn.1;

    println!("\n{}", "=".repeat(72));
    println!(
        "Worker Removal Churn ({} → {} workers, {} LoRAs, {} slots/worker)",
        num_workers,
        num_workers - 1,
        num_loras,
        slots_per_worker
    );
    println!("{}", "=".repeat(72));
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Metric", "HRW", "Random", "MCF"
    );
    println!("  {:->12}  {:->8}  {:->8}  {:->8}", "", "", "", "");
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Loads", hrw_churn.0, random_churn.0, mcf_churn.0
    );
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Unloads", hrw_churn.1, random_churn.1, mcf_churn.1
    );
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Total", hrw_total, random_total, mcf_total
    );
    println!();

    // HRW should have bounded churn
    assert!(
        hrw_total <= num_loras * 2,
        "HRW worker removal churn ({}) should be bounded by 2x num_loras ({})",
        hrw_total,
        num_loras * 2
    );
    // MCF should have bounded churn
    assert!(
        mcf_total <= num_loras * 2,
        "MCF worker removal churn ({}) should be bounded by 2x num_loras ({})",
        mcf_total,
        num_loras * 2
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

// ============================================================================
// CSV Export for Visualization
// ============================================================================

/// Runs simulations at C=20%, 50%, 90% concurrent slot usage and writes
/// per-tick churn, load, lifecycle, and summary data to CSV files under
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

    // Fixed cluster: N=8 backends, K=4 slots → 32 total slots
    let num_backends: usize = 8;
    let slots_per_backend: usize = 4;
    let total_slots = num_backends * slots_per_backend; // 32

    let mut all_runs: Vec<(&str, SimConfig, Vec<LoraLoadSchedule>)> = Vec::new();

    // ── 1. Zipf + Poisson scenario: 100 LoRAs, power-law popularity, Poisson arrivals.
    // avg_total_load = 40 → keeps slot utilization well under 100% on 32 slots.
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
        scale_down_cooldown_ticks: 2,
        lifetime_mean: 0,
        lifetime_stddev: 0.0,
        seed: 42,
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
        scale_down_cooldown_ticks: 2,
        lifetime_mean: 0,
        lifetime_stddev: 0.0,
        seed: 42,
    };
    all_runs.push(("daily", diurnal_config.clone(), diurnal_schedules));

    // ── 4. Flash Crowd: baseline Zipf+Poisson with two viral spikes.
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
        scale_down_cooldown_ticks: 2,
        lifetime_mean: 0,
        lifetime_stddev: 0.0,
        seed: 42,
    };
    all_runs.push(("spike", flash_config.clone(), flash_schedules));

    // ── 5. MMPP: Markov-modulated Poisson process with 3 states.
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
        scale_down_cooldown_ticks: 2,
        lifetime_mean: 0,
        lifetime_stddev: 0.0,
        seed: 42,
    };
    all_runs.push(("mmpp", mmpp_config.clone(), mmpp_schedules));

    for (name, config, schedules) in &all_runs {
        let hrw = run_hrw_simulation(config, schedules);
        let random = run_random_simulation(config, schedules);
        let mcf = run_mcf_simulation(config, schedules);

        let c_pct = (config.concurrent_loras as f64 / total_slots as f64 * 100.0).round();
        println!(
            "\n── {} (C={:.0}%, {}/{} slots, L={}) ──",
            name, c_pct, config.concurrent_loras, total_slots, config.total_loras
        );
        print_comparison(&hrw, &random, &mcf);

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
            "total_loads,{},{},{}",
            hrw.total_loads, random.total_loads, mcf.total_loads
        )
        .unwrap();
        writeln!(
            f,
            "total_unloads,{},{},{}",
            hrw.total_unloads, random.total_unloads, mcf.total_unloads
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
        writeln!(f, "concurrent_loras,{}", config.concurrent_loras).unwrap();
        writeln!(f, "c_pct,{:.0}", c_pct).unwrap();
        writeln!(f, "total_ticks,{}", config.total_ticks).unwrap();
        writeln!(f, "loras_used,{}", schedules.len()).unwrap();
        writeln!(f, "lifetime_mean,{}", config.lifetime_mean).unwrap();
        writeln!(f, "lifetime_stddev,{:.1}", config.lifetime_stddev).unwrap();
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
    println!("Run: python lib/llm/tests/lora_simulation/plot_lora_churn.py");
}
