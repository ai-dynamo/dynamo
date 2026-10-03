// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Seeded selection over candidate rows in canonical worker order, shared by the campaign
//! policies (`learned-choice`, `sticky-session`).
//!
//! Every method visits rows in worker order, never the host's unspecified row order, so a seeded
//! policy decides identically across processes.
//!
//! Two tie-break disciplines are offered:
//!
//! - `one_draw` (default) consumes exactly one 64-bit draw per decision, whatever path the
//!   decision takes: a tie, a softmax sample, a pin, a sticky hit, or a unique minimum. Two
//!   policies that share a seed therefore share every later draw even after one early decision
//!   differs, which keeps common random numbers aligned across candidates in policy search.
//! - `reservoir` reproduces the seeded `dynamo-default-cost-fn` picker draw for draw: it draws
//!   only on ties (once per tied candidate) and for softmax samples, and not at all on pins. It
//!   exists so replay can check a policy against the default bit for bit.

use dynamo_kv_router::plugins::worker_selection::ScoredWorkerCandidate;
use dynamo_kv_router::protocols::WorkerWithDpRank;

use crate::default::softmax_sample_index;

/// How a seeded chooser breaks ties and draws samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TieBreak {
    #[default]
    OneDraw,
    Reservoir,
}

/// Map a 64-bit draw onto `0..n` by multiply-shift, without rejection, so it never draws again.
fn index_below(draw: u64, n: usize) -> usize {
    ((u128::from(draw) * n as u128) >> 64) as usize
}

/// Map a 64-bit draw onto [0, 1) with 53 bits of precision.
fn unit_interval(draw: u64) -> f64 {
    (draw >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

pub(crate) struct Chooser {
    rng: fastrand::Rng,
    tie_break: TieBreak,
    /// This decision's draw under `one_draw`; always None under `reservoir`.
    draw: Option<u64>,
    order: Vec<usize>,
    tied: Vec<usize>,
    weights: Vec<f64>,
}

impl Chooser {
    /// A chooser seeded with `seed`, or from fresh entropy when None, as production routers want:
    /// router replicas sharing one seed would break ties identically and herd.
    pub(crate) fn new(seed: Option<u64>, tie_break: TieBreak) -> Self {
        Self {
            rng: seed.map_or_else(fastrand::Rng::new, fastrand::Rng::with_seed),
            tie_break,
            draw: None,
            order: Vec::new(),
            tied: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Start one decision over `candidates`, ordering their rows canonically by worker.
    /// Under `one_draw` this consumes the decision's single draw.
    pub(crate) fn begin(&mut self, candidates: &[ScoredWorkerCandidate]) {
        self.draw = match self.tie_break {
            TieBreak::OneDraw => Some(self.rng.u64(..)),
            TieBreak::Reservoir => None,
        };
        self.order.clear();
        self.order.extend(0..candidates.len());
        self.order
            .sort_unstable_by_key(|&row| candidates[row].worker());
    }

    /// Rows of the current decision in canonical worker order.
    pub(crate) fn order(&self) -> &[usize] {
        &self.order
    }

    fn uniform(&mut self) -> f64 {
        match self.draw {
            Some(draw) => unit_interval(draw),
            None => self.rng.f64(),
        }
    }

    /// The row with the lowest `key`, ties broken by the configured discipline.
    pub(crate) fn lowest(&mut self, key: impl Fn(usize) -> f64) -> usize {
        if let Some(draw) = self.draw {
            let minimum = self
                .order
                .iter()
                .map(|&row| key(row))
                .fold(f64::INFINITY, f64::min);
            self.tied.clear();
            self.tied.extend(
                self.order
                    .iter()
                    .copied()
                    .filter(|&row| key(row) == minimum),
            );
            return match self.tied.len() {
                0 => self.order[0],
                ties => self.tied[index_below(draw, ties)],
            };
        }
        // The seeded default picker's reservoir, step for step.
        let mut best = 0;
        let mut best_key = f64::INFINITY;
        let mut ties = 0;
        for (index, &row) in self.order.iter().enumerate() {
            let value = key(row);
            if value < best_key {
                best = index;
                best_key = value;
                ties = 1;
            } else if value == best_key {
                ties += 1;
                if self.rng.usize(0..ties) == 0 {
                    best = index;
                }
            }
        }
        self.order[best]
    }

    /// Sample a row with probability ∝ exp(utility / temperature).
    pub(crate) fn sample_utilities(
        &mut self,
        utility: impl Fn(usize) -> f64,
        temperature: f64,
    ) -> usize {
        debug_assert!(temperature > 0.0);
        let maximum = self
            .order
            .iter()
            .map(|&row| utility(row))
            .fold(f64::NEG_INFINITY, f64::max);
        self.weights.clear();
        self.weights.extend(
            self.order
                .iter()
                .map(|&row| ((utility(row) - maximum) / temperature).exp()),
        );
        let total: f64 = self.weights.iter().sum();
        let target = self.uniform() * total;
        let mut cumulative = 0.0;
        for (index, weight) in self.weights.iter().enumerate() {
            cumulative += weight;
            if target < cumulative {
                return self.order[index];
            }
        }
        self.order[self.order.len() - 1]
    }

    /// Sample a row from costs with the default picker's range-normalized softmax.
    pub(crate) fn sample_default(
        &mut self,
        cost: impl Fn(usize) -> f64,
        temperature: f64,
    ) -> usize {
        let sample = self.uniform();
        let index = softmax_sample_index(
            &self.order,
            |&row| cost(row),
            temperature,
            sample,
            &mut self.weights,
        );
        self.order[index]
    }
}

/// The row holding `worker`, if any.
pub(crate) fn row_of(
    candidates: &[ScoredWorkerCandidate],
    worker: WorkerWithDpRank,
) -> Option<usize> {
    candidates
        .iter()
        .position(|candidate| candidate.worker() == worker)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_mappings_stay_in_range() {
        for draw in [0, 1, u64::MAX / 2, u64::MAX - 1, u64::MAX] {
            assert!(unit_interval(draw) < 1.0 && unit_interval(draw) >= 0.0);
            for n in [1, 2, 3, 7, 32] {
                assert!(index_below(draw, n) < n);
            }
        }
        assert_eq!(index_below(u64::MAX, 4), 3);
        assert_eq!(index_below(0, 4), 0);
    }
}
