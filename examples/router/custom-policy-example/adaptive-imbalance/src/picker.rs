// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Picks the lowest cost at temperature 0, at random among exact ties as the default picker does;
//! otherwise samples with probability falling as cost rises.
//! The loop moves only the cost weights, so the temperature is the baseline one unless the request
//! overrides it with `router_temperature_override`.
//!
//! The softmax copies the default picker's: plugin crates cannot use `DefaultWorkerPicker`, which
//! `dynamo-kv-router` exports only to tests and benchmarks.

use dynamo_kv_router::plugins::prelude::*;
use parking_lot::Mutex;

pub struct AdaptivePicker {
    temperature: f64,
    /// The PRNG and a cost buffer reused across picks. Concurrent picks take turns on the lock,
    /// as the default picker does with its softmax scratch.
    scratch: Mutex<PickScratch>,
}

struct PickScratch {
    rng: XorShift64,
    costs: Vec<f64>,
}

impl AdaptivePicker {
    pub fn new(temperature: f64) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15)
            | 1;
        Self {
            temperature,
            scratch: Mutex::new(PickScratch {
                rng: XorShift64(seed),
                costs: Vec::new(),
            }),
        }
    }

    /// A picker with a fixed PRNG seed, so sampled picks are reproducible in tests.
    #[cfg(test)]
    pub fn with_seed(temperature: f64, seed: u64) -> Self {
        Self {
            temperature,
            scratch: Mutex::new(PickScratch {
                // xorshift has an all-zero fixed point; keep the state odd as `new` does.
                rng: XorShift64(seed | 1),
                costs: Vec::new(),
            }),
        }
    }
}

impl WorkerPicker for AdaptivePicker {
    fn pick(
        &self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let temperature = context
            .router_temperature_override()
            .unwrap_or(self.temperature);
        let mut scratch = self.scratch.lock();
        let PickScratch { rng, costs } = &mut *scratch;
        costs.clear();
        costs.extend(input.candidates().iter().map(|c| c.cost()));
        if costs.is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }
        Ok(choose(costs, temperature, rng.next_f64()))
    }
}

/// Row index for the given costs. `u` is a uniform draw in `0..1`, passed in so the choice is testable.
///
/// Matches the default selector's softmax: costs are normalized by their range before the
/// temperature applies, so the probabilities depend only on the relative gaps between workers.
/// A temperature that is not a finite positive number picks the lowest cost, using `u` to choose
/// among exact ties. Costs that make the softmax undefined (NaN or infinite) fall back to the
/// deterministic argmin.
pub fn choose(costs: &[f64], temperature: f64, u: f64) -> usize {
    if costs.len() < 2 || !(temperature.is_finite() && temperature > 0.0) {
        return argmin_breaking_ties(costs, u);
    }
    let (min, max) = costs
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &c| {
            (lo.min(c), hi.max(c))
        });
    let n = costs.len();
    if !(min.is_finite() && max.is_finite()) {
        return argmin(costs);
    }
    if min == max {
        return (0..n)
            .find(|&i| u < (i + 1) as f64 / n as f64)
            .unwrap_or(n - 1);
    }

    // A range that overflows f64 is rescaled by the largest magnitude first, as the default selector does.
    let range = max - min;
    let magnitude = if range.is_finite() {
        1.0
    } else {
        min.abs().max(max.abs())
    };
    let min_normalized = min / magnitude;
    let scale = -1.0 / ((max / magnitude - min_normalized) * temperature);
    let offset = min_normalized * scale;
    let weight = |c: f64| (c / magnitude * scale - offset).exp();

    let total: f64 = costs.iter().map(|&c| weight(c)).sum();
    if !(total.is_finite() && total > 0.0) {
        return argmin(costs);
    }
    let mut acc = 0.0;
    for (i, &c) in costs.iter().enumerate() {
        acc += weight(c) / total;
        if u < acc {
            return i;
        }
    }
    n - 1
}

/// Lowest-cost row; `u` picks uniformly among rows with exactly that cost. On an idle pool every
/// worker ties, and always taking the first row would send each new request to the same worker.
fn argmin_breaking_ties(costs: &[f64], u: f64) -> usize {
    let best = argmin(costs);
    let Some(&min) = costs.get(best) else {
        return best;
    };
    let ties = costs.iter().filter(|c| c.total_cmp(&min).is_eq()).count();
    let k = ((u * ties as f64) as usize).min(ties.saturating_sub(1));
    costs
        .iter()
        .enumerate()
        .filter(|(_, c)| c.total_cmp(&min).is_eq())
        .nth(k)
        .map_or(best, |(i, _)| i)
}

fn argmin(costs: &[f64]) -> usize {
    costs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Small, dependency-free PRNG; quality is more than enough for load spreading.
struct XorShift64(u64);
impl XorShift64 {
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_zero_is_argmin() {
        assert_eq!(choose(&[19.0, 56.0, 65.0, 46.5], 0.0, 0.99), 0);
        assert_eq!(choose(&[384.0, 324.0, 320.0, 326.0], 0.0, 0.01), 2);
    }

    #[test]
    fn temperature_zero_picks_uniformly_among_exact_ties() {
        let costs = [5.0, 3.0, 3.0, 7.0, 3.0];
        let picks: Vec<usize> = (0..6)
            .map(|i| choose(&costs, 0.0, (i as f64 + 0.5) / 6.0))
            .collect();
        assert_eq!(picks, vec![1, 1, 2, 2, 4, 4]);
    }

    #[test]
    fn positive_temperature_spreads_across_cheap_workers() {
        // Range 55, so the normalized gaps are 0, 1/55, 2/55 and 1. At temperature 0.1 the weights are
        // exp(0)=1, exp(-10/55)≈0.834, exp(-20/55)≈0.695, exp(-10)≈4.5e-5, giving cumulative
        // probabilities ≈ 0.395, 0.725, 0.99998, 1. The midpoint draws 0.05..0.95 hit rows 0, 1, 2 only.
        let costs = [69.0, 70.0, 71.0, 124.0];
        let picks: Vec<usize> = (0..10)
            .map(|i| choose(&costs, 0.1, (i as f64 + 0.5) / 10.0))
            .collect();
        assert_eq!(picks, vec![0, 0, 0, 0, 1, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn picks_depend_on_relative_gaps_not_cost_scale() {
        let small = [69.0, 70.0, 71.0, 124.0];
        let large = [690.0, 700.0, 710.0, 1240.0];
        for i in 0..100 {
            let u = (i as f64 + 0.5) / 100.0;
            for temperature in [0.05, 0.5, 2.0] {
                assert_eq!(
                    choose(&small, temperature, u),
                    choose(&large, temperature, u),
                    "u={u} temperature={temperature}"
                );
            }
        }
    }

    #[test]
    fn equal_costs_are_picked_uniformly() {
        let costs = [5.0, 5.0, 5.0, 5.0];
        let picks: Vec<usize> = (0..8)
            .map(|i| choose(&costs, 1.0, (i as f64 + 0.5) / 8.0))
            .collect();
        assert_eq!(picks, vec![0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn nan_cost_falls_back_to_argmin() {
        let costs = [70.0, f64::NAN, 69.0, 124.0];
        for i in 0..10 {
            assert_eq!(choose(&costs, 0.5, (i as f64 + 0.5) / 10.0), 2);
        }
    }

    #[test]
    fn all_infinite_costs_fall_back_to_a_deterministic_pick() {
        let costs = [f64::INFINITY; 4];
        let first = choose(&costs, 0.5, 0.05);
        assert!(first < costs.len());
        for i in 0..10 {
            assert_eq!(choose(&costs, 0.5, (i as f64 + 0.5) / 10.0), first);
        }
    }

    #[test]
    fn nan_temperature_is_argmin() {
        let costs = [70.0, 71.0, 69.0, 124.0];
        for i in 0..10 {
            assert_eq!(choose(&costs, f64::NAN, (i as f64 + 0.5) / 10.0), 2);
        }
    }

    #[test]
    fn prng_is_in_unit_interval_and_not_constant() {
        let mut r = XorShift64(42);
        let a = r.next_f64();
        let b = r.next_f64();
        assert!((0.0..1.0).contains(&a) && (0.0..1.0).contains(&b));
        assert_ne!(a, b);
    }
}
