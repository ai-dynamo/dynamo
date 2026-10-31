// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The reference selector's picker: the lowest cost with uniformly random tie-breaks at
//! zero temperature, otherwise a softmax sample over the costs, drawing from a seeded RNG
//! when parity tests and benchmarks ask for replayable picks.

use std::sync::Arc;

use parking_lot::{Mutex, MutexGuard};

use dynamo_router_protocol::worker::WorkerWithDpRank;

pub struct DefaultWorkerPicker {
    // Preserve the reference selector's Sync contract. Zero-temperature picks never lock it.
    softmax_scratch: Mutex<SoftmaxScratch>,
    deterministic_rng: Option<Arc<Mutex<fastrand::Rng>>>,
}

#[derive(Default)]
struct SoftmaxScratch {
    entries: Vec<(WorkerWithDpRank, f64)>,
    probabilities: Vec<f64>,
}

impl Default for DefaultWorkerPicker {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultWorkerPicker {
    /// A picker drawing from the thread-local RNG.
    pub fn new() -> Self {
        Self::with_rng(None)
    }

    /// A picker drawing from `rng` when given, so ties and samples replay.
    pub fn with_rng(deterministic_rng: Option<Arc<Mutex<fastrand::Rng>>>) -> Self {
        Self {
            softmax_scratch: Mutex::default(),
            deterministic_rng,
        }
    }

    /// The seeded RNG, for a selector that clones it and keeps drawing from the same one.
    pub fn rng(&self) -> Option<&Arc<Mutex<fastrand::Rng>>> {
        self.deterministic_rng.as_ref()
    }

    /// Whether picks draw from a seeded RNG. Callers then offer candidates in a fixed order.
    pub fn is_deterministic(&self) -> bool {
        self.deterministic_rng.is_some()
    }

    /// Start a pick: offer every candidate's cost, then finish. Zero `temperature` takes the
    /// lowest cost, breaking ties uniformly at random as they arrive; any other temperature
    /// samples a softmax over every offered cost.
    pub fn begin(&self, temperature: f64) -> DefaultPick<'_> {
        DefaultPick {
            rng: self.deterministic_rng.as_ref().map(|rng| rng.lock()),
            mode: if temperature == 0.0 {
                PickMode::Lowest {
                    best_worker: None,
                    best_cost: f64::INFINITY,
                    tie_count: 0,
                }
            } else {
                let mut scratch = self.softmax_scratch.lock();
                scratch.entries.clear();
                PickMode::Softmax {
                    temperature,
                    scratch,
                }
            },
        }
    }
}

/// One pick in progress; see [`DefaultWorkerPicker::begin`].
pub struct DefaultPick<'p> {
    rng: Option<MutexGuard<'p, fastrand::Rng>>,
    mode: PickMode<'p>,
}

enum PickMode<'p> {
    Lowest {
        best_worker: Option<WorkerWithDpRank>,
        best_cost: f64,
        tie_count: usize,
    },
    Softmax {
        temperature: f64,
        scratch: MutexGuard<'p, SoftmaxScratch>,
    },
}

impl DefaultPick<'_> {
    pub fn offer(&mut self, worker: WorkerWithDpRank, cost: f64) {
        match &mut self.mode {
            PickMode::Lowest {
                best_worker,
                best_cost,
                tie_count,
            } => {
                if cost < *best_cost {
                    *best_worker = Some(worker);
                    *best_cost = cost;
                    *tie_count = 1;
                } else if cost == *best_cost {
                    *tie_count += 1;
                    let draw = match &mut self.rng {
                        Some(rng) => rng.usize(0..*tie_count),
                        None => fastrand::usize(0..*tie_count),
                    };
                    if draw == 0 {
                        *best_worker = Some(worker);
                    }
                }
            }
            PickMode::Softmax { scratch, .. } => scratch.entries.push((worker, cost)),
        }
    }

    /// The picked worker and its cost; `None` when nothing was offered.
    pub fn finish(self) -> Option<(WorkerWithDpRank, f64)> {
        let DefaultPick { rng, mode } = self;
        match mode {
            PickMode::Lowest {
                best_worker,
                best_cost,
                ..
            } => best_worker.map(|worker| (worker, best_cost)),
            PickMode::Softmax {
                temperature,
                mut scratch,
            } => {
                let SoftmaxScratch {
                    entries,
                    probabilities,
                } = &mut *scratch;
                if entries.is_empty() {
                    return None;
                }
                let sample = match rng {
                    Some(mut rng) => rng.f64(),
                    None => fastrand::f64(),
                };
                let row = softmax_sample_index(
                    entries,
                    |(_, cost)| *cost,
                    temperature,
                    sample,
                    probabilities,
                );
                Some(entries[row])
            }
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sample_orders_extreme_finite_costs() {
        let entries = [(0u64, -f64::MAX), (1u64, f64::MAX)];
        let row = softmax_sample_index(&entries, |(_, cost)| *cost, 1.0, 0.6, &mut Vec::new());
        assert_eq!(entries[row].0, 0);
    }

    #[test]
    fn lowest_cost_wins_and_ties_replay_with_a_seed() {
        let seeded = |seed| {
            DefaultWorkerPicker::with_rng(Some(Arc::new(Mutex::new(fastrand::Rng::with_seed(
                seed,
            )))))
        };
        let offers = |pick: &mut DefaultPick<'_>| {
            pick.offer(WorkerWithDpRank::from_worker_id(0), 2.0);
            pick.offer(WorkerWithDpRank::from_worker_id(1), 1.0);
            pick.offer(WorkerWithDpRank::from_worker_id(2), 1.0);
            pick.offer(WorkerWithDpRank::from_worker_id(3), f64::NAN);
        };
        let first = seeded(7);
        let mut pick = first.begin(0.0);
        offers(&mut pick);
        let (worker, cost) = pick.finish().unwrap();
        assert_eq!(cost, 1.0);
        assert!(worker.worker_id == 1 || worker.worker_id == 2);
        let again = seeded(7);
        let mut pick = again.begin(0.0);
        offers(&mut pick);
        assert_eq!(pick.finish().unwrap().0, worker);
        assert!(first.begin(0.0).finish().is_none());
        assert!(first.begin(0.5).finish().is_none());
    }
}
