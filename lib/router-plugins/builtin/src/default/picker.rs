// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimum-cost selection, tie-breaking, and temperature sampling of scored candidates.

use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicyError,
};
use parking_lot::Mutex;
use std::sync::Arc;

fn softmax_sample_index<T>(
    entries: &[T],
    cost: impl Fn(&T) -> f64,
    temperature: f64,
    sample: f64,
    probabilities: &mut Vec<f64>,
) -> usize {
    assert!(!entries.is_empty(), "Empty entries for softmax sampling");
    debug_assert_ne!(temperature, 0.0);

    let (min_cost, max_cost) = entries
        .iter()
        .map(&cost)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), cost| {
            (lo.min(cost), hi.max(cost))
        });

    probabilities.clear();
    if min_cost == max_cost {
        probabilities.resize(entries.len(), 1.0 / entries.len() as f64);
    } else {
        let range = max_cost - min_cost;
        let magnitude = if range.is_finite() {
            1.0
        } else {
            min_cost.abs().max(max_cost.abs())
        };
        let min_normalized = min_cost / magnitude;
        let scale = -1.0 / ((max_cost / magnitude - min_normalized) * temperature);
        let max_scaled = min_normalized * scale;
        probabilities.extend(
            entries
                .iter()
                .map(|entry| (cost(entry) / magnitude * scale - max_scaled).exp()),
        );
    }

    let sum: f64 = probabilities.iter().sum();
    for probability in probabilities.iter_mut() {
        *probability /= sum;
    }
    let mut cumulative = 0.0;
    for (row, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if sample <= cumulative {
            return row;
        }
    }
    entries.len() - 1
}

pub(super) struct DefaultPicker {
    temperature: f64,
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
    entries: Vec<(usize, f64)>,
    probabilities: Vec<f64>,
}

impl DefaultPicker {
    pub(super) fn new(temperature: f64, rng: Option<Arc<Mutex<fastrand::Rng>>>) -> Self {
        Self {
            temperature,
            rng,
            entries: Vec::new(),
            probabilities: Vec::new(),
        }
    }
}

impl WorkerPicker for DefaultPicker {
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        if candidates.is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }
        if context.pinned_worker().is_some() {
            return Ok(0);
        }
        if self.temperature == 0.0 && self.rng.is_none() {
            let mut best_row = 0;
            let mut best_cost = f64::INFINITY;
            let mut ties = 0;
            for (row, candidate) in candidates.iter().enumerate() {
                let cost = candidate.cost();
                if cost < best_cost {
                    best_row = row;
                    best_cost = cost;
                    ties = 1;
                } else if cost == best_cost {
                    ties += 1;
                    if fastrand::usize(0..ties) == 0 {
                        best_row = row;
                    }
                }
            }
            return Ok(best_row);
        }
        let Some(rng) = &self.rng else {
            return Ok(softmax_sample_index(
                candidates,
                |candidate| candidate.cost(),
                self.temperature,
                fastrand::f64(),
                &mut self.probabilities,
            ));
        };
        self.entries.clear();
        self.entries.extend(
            candidates
                .iter()
                .enumerate()
                .map(|(row, candidate)| (row, candidate.cost())),
        );
        // Canonical order is required only for deterministic replay, never for production ties.
        self.entries.sort_unstable_by_key(|(row, _)| {
            let worker = candidates[*row].worker();
            (worker.worker_id, worker.dp_rank)
        });
        let mut rng = rng.lock();
        let selected = if self.temperature == 0.0 {
            let mut best = 0;
            let mut best_cost = f64::INFINITY;
            let mut ties = 0;
            for (index, (_, cost)) in self.entries.iter().enumerate() {
                if *cost < best_cost {
                    best = index;
                    best_cost = *cost;
                    ties = 1;
                } else if *cost == best_cost {
                    ties += 1;
                    if rng.usize(0..ties) == 0 {
                        best = index;
                    }
                }
            }
            best
        } else {
            softmax_sample_index(
                &self.entries,
                |(_, cost)| *cost,
                self.temperature,
                rng.f64(),
                &mut self.probabilities,
            )
        };
        Ok(self.entries[selected].0)
    }
}
