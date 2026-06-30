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
        let uniform_1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let uniform_2 = rng.random::<f64>();
        let standard_normal =
            (-2.0 * uniform_1.ln()).sqrt() * (std::f64::consts::TAU * uniform_2).cos();
        let normal = lambda + lambda.sqrt() * standard_normal;
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
#[allow(clippy::too_many_arguments)]
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
        for (t, m) in multipliers.iter_mut().enumerate().skip(ft) {
            let elapsed = (t - ft) as f64;
            *m += (spike_multiplier - 1.0) * (decay_rate * elapsed).exp();
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
        for (next, &prob) in transition_matrix[current_state].iter().enumerate() {
            cumulative += prob;
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
            return Vec::new();
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
        let mut total_load: usize = 0;

        // Match the controller paths: adapters with zero current load have no load-estimator
        // entry, so the random baseline excludes them rather than adding unmatched cold pins.
        for schedule in schedules {
            let load = schedule.load_at_tick(tick);
            if load > 0 {
                active_loads.push((schedule.lora_name.clone(), load));
                total_load += load;
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

#[path = "csv_export.rs"]
mod csv_export;
#[path = "scenario_tests.rs"]
mod scenario_tests;
#[path = "topology_tests.rs"]
mod topology_tests;
