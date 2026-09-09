// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-injected control law and picker, shared unchanged by the plugin and simulation.

use std::time::Duration;

/// Signal-driven algorithms; neither learns latency rewards.
#[derive(Debug, Clone, Copy, Default, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Algorithm {
    Sigmoid,
    #[default]
    Aimd,
}

/// Per-instance settings. Bounds and time constants remain deployment choices.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Parameters {
    pub algorithm: Algorithm,
    pub update_interval_ms: u64,
    pub smoothing: f64,
    pub max_step: f64,
    pub distribution_min: f64,
    pub distribution_max: f64,
    pub pressure_max: f64,
    pub midpoint: f64,
    pub slope: f64,
    pub load_scale: f64,
    /// Optional deterministic tie breaking for replay; production uses independent seeds.
    pub seed: Option<u64>,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::Aimd,
            update_interval_ms: 100,
            smoothing: 0.2,
            max_step: 0.1,
            distribution_min: 0.1,
            distribution_max: 0.9,
            pressure_max: 0.5,
            midpoint: 0.35,
            slope: 8.0,
            load_scale: 8.0,
            seed: None,
        }
    }
}

impl Parameters {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.update_interval_ms == 0 {
            return Err("update_interval_ms must be positive");
        }
        for (value, error) in [
            (self.smoothing, "smoothing must be finite and in (0, 1]"),
            (self.max_step, "max_step must be finite and in (0, 1]"),
        ] {
            if !value.is_finite() || value <= 0.0 || value > 1.0 {
                return Err(error);
            }
        }
        if !(self.distribution_min.is_finite()
            && self.distribution_max.is_finite()
            && self.pressure_max.is_finite()
            && 0.0 < self.distribution_min
            && self.distribution_min <= self.pressure_max
            && self.pressure_max <= self.distribution_max
            && self.distribution_max < 1.0)
        {
            return Err("require 0 < distribution_min <= pressure_max <= distribution_max < 1");
        }
        if !self.midpoint.is_finite() || self.midpoint <= 0.0 || self.midpoint >= 1.0 {
            return Err("midpoint must be finite and in (0, 1)");
        }
        // Bounded steepness keeps endpoint normalization numerically well-conditioned.
        if !self.slope.is_finite() || !(0.1..=100.0).contains(&self.slope) {
            return Err("slope must be finite and in [0.1, 100]");
        }
        if !self.load_scale.is_finite() || self.load_scale < 1.0 {
            return Err("load_scale must be finite and >= 1");
        }
        Ok(())
    }
}

/// Only the two scorer inputs this policy consumes, borrowed through iterators in production.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub affinity: f64,
    pub active_requests: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Snapshot {
    pub distribution_weight: f64,
    pub imbalance: f64,
    /// Active-count proxy; this is not engine queue depth or measured utilization.
    pub pressure: f64,
    pub updates: u64,
}

/// O(1) actor-local state: no worker identity map, mutex, background task, or I/O.
pub struct AdaptiveRouter {
    parameters: Parameters,
    snapshot: Snapshot,
    last_update: Option<Duration>,
    rng: fastrand::Rng,
}

impl AdaptiveRouter {
    pub fn new(parameters: Parameters) -> Result<Self, &'static str> {
        parameters.validate()?;
        Ok(Self::from_validated(parameters))
    }

    pub(crate) fn from_validated(parameters: Parameters) -> Self {
        Self {
            snapshot: Snapshot {
                distribution_weight: parameters.distribution_min,
                ..Snapshot::default()
            },
            last_update: None,
            rng: parameters
                .seed
                .map_or_else(fastrand::Rng::new, fastrand::Rng::with_seed),
            parameters,
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        self.snapshot
    }

    /// `now` is elapsed monotonic time. Missing ticks do not replay an unobserved workload.
    /// Iterator cloning borrows the host's columns without allocating candidate copies.
    pub fn select(
        &mut self,
        now: Duration,
        candidates: impl Iterator<Item = Candidate> + Clone,
    ) -> Option<usize> {
        let min = candidates.clone().map(|c| c.active_requests).min()?;
        let max = candidates.clone().map(|c| c.active_requests).max()?;
        let interval = Duration::from_millis(self.parameters.update_interval_ms);
        if self
            .last_update
            .is_none_or(|last| now.saturating_sub(last) >= interval)
        {
            let mut count = 0;
            let mut mean = 0.0;
            let mut m2 = 0.0;
            for candidate in candidates.clone() {
                count += 1;
                let delta = candidate.active_requests as f64 - mean;
                mean += delta / count as f64;
                m2 += delta * (candidate.active_requests as f64 - mean);
            }
            // CV / sqrt(N - 1) is bounded by one for nonnegative observations.
            // The absolute-load gate prevents one active request from looking saturated.
            let imbalance = if count > 1 && mean > 0.0 {
                ((m2.max(0.0) / count as f64).sqrt() / mean / ((count - 1) as f64).sqrt())
                    .clamp(0.0, 1.0)
                    * (max as f64 / (max as f64 + self.parameters.load_scale))
            } else {
                0.0
            };
            // All candidates must be busy to activate the conservative pressure cap.
            let pressure = min as f64 / (min as f64 + self.parameters.load_scale);
            self.update(imbalance, pressure);
            self.last_update = Some(now);
        }
        pick_weighted(
            candidates,
            min,
            max,
            self.snapshot.distribution_weight,
            self.parameters.load_scale,
            &mut self.rng,
        )
    }

    fn update(&mut self, imbalance: f64, pressure: f64) {
        let p = &self.parameters;
        let s = &mut self.snapshot;
        s.imbalance += p.smoothing * (imbalance - s.imbalance);
        s.pressure += p.smoothing * (pressure - s.pressure);
        let cap = p.distribution_max - s.pressure * (p.distribution_max - p.pressure_max);
        let target = match p.algorithm {
            Algorithm::Sigmoid => {
                let sigmoid = |v: f64| 1.0 / (1.0 + (-p.slope * (v - p.midpoint)).exp());
                // Normalize endpoints: exactly balanced pools restore the configured baseline.
                let fraction =
                    (sigmoid(s.imbalance) - sigmoid(0.0)) / (sigmoid(1.0) - sigmoid(0.0));
                p.distribution_min + fraction * (cap - p.distribution_min)
            }
            Algorithm::Aimd => {
                if s.imbalance > p.midpoint {
                    s.distribution_weight + p.max_step
                } else if s.imbalance < p.midpoint * 0.5 {
                    // Multiplicative decrease of the extra budget, with a hysteresis deadband.
                    p.distribution_min + 0.9 * (s.distribution_weight - p.distribution_min)
                } else {
                    s.distribution_weight
                }
            }
        }
        .clamp(p.distribution_min, cap);
        s.distribution_weight += (target - s.distribution_weight).clamp(-p.max_step, p.max_step);
        s.updates = s.updates.saturating_add(1);
    }
}

/// Common scorer arithmetic, also used by the fixed-weight and bandit experiment controls.
/// Callers supply valid bounds and a weight in [0, 1]. Non-finite cache signals receive no credit.
pub fn pick_weighted(
    candidates: impl Iterator<Item = Candidate>,
    min: usize,
    max: usize,
    distribution_weight: f64,
    load_scale: f64,
    rng: &mut fastrand::Rng,
) -> Option<usize> {
    let denominator = load_scale + max.saturating_sub(min) as f64;
    let mut best = None;
    let mut best_cost = f64::INFINITY;
    let mut ties = 0;
    for (row, candidate) in candidates.enumerate() {
        let affinity = if candidate.affinity.is_finite() {
            candidate.affinity.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let distribution = candidate.active_requests.saturating_sub(min) as f64 / denominator;
        let cost =
            (1.0 - distribution_weight) * (1.0 - affinity) + distribution_weight * distribution;
        if cost < best_cost {
            best_cost = cost;
            best = Some(row);
            ties = 1;
        } else if cost == best_cost {
            ties += 1;
            if rng.usize(..ties) == 0 {
                best = Some(row);
            }
        }
    }
    best
}
