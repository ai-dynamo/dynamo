// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Power-of-d-choices selection over the default cost.
//!
//! The default picker takes the exact minimum cost, so router replicas that hold the same synced
//! load table and pick within one sync delay all choose the same worker. This picker draws
//! `choices` distinct candidates uniformly at random and takes the lowest default cost among
//! them, which spreads those picks while still steering away from loaded workers. Candidates,
//! their cost, and everything after the pick are the default's.

use dynamo_kv_router::plugins::prelude::*;
use parking_lot::Mutex;
use std::cell::Cell;
use std::sync::Arc;

use super::{PolicyParameters, role_parameters, scorer};

/// Policy type selected under a `worker_selection` role.
pub const POWER_OF_TWO_POLICY_TYPE: &str = "dynamo-power-of-two-cost-fn";

const DEFAULT_CHOICES: usize = 2;

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Candidates drawn per pick.
    choices: usize,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            choices: DEFAULT_CHOICES,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if self.choices < 2 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "choices must be at least 2",
            ));
        }
        Ok(())
    }
}

thread_local! {
    // Each thread keeps one buffer of drawn rows, shared by every picker on it, so picks reuse
    // its memory without a lock.
    static DRAWN_ROWS: Cell<Vec<usize>> = const { Cell::new(Vec::new()) };
}

/// Draw `choices` distinct indexes below `len`, or all of them when there are no more than that,
/// and return the one with the lowest cost, ties broken uniformly. `index(upper)` is a uniform
/// draw below `upper`. Only the drawn indexes have their cost read; they are left in `drawn`.
fn lowest_cost_of_draw(
    len: usize,
    choices: usize,
    cost: impl Fn(usize) -> f64,
    mut index: impl FnMut(usize) -> usize,
    drawn: &mut Vec<usize>,
) -> usize {
    drawn.clear();
    if len <= choices {
        drawn.extend(0..len);
    } else {
        // Floyd's sampling: `choices` draws, each distinct from the ones before it.
        for upper in len - choices..len {
            let candidate = index(upper + 1);
            drawn.push(if drawn.contains(&candidate) {
                upper
            } else {
                candidate
            });
        }
    }
    let mut best = 0;
    let mut best_cost = f64::INFINITY;
    let mut ties = 0;
    for &candidate in drawn.iter() {
        let cost = cost(candidate);
        if cost < best_cost {
            best = candidate;
            best_cost = cost;
            ties = 1;
        } else if cost == best_cost {
            ties += 1;
            if index(ties) == 0 {
                best = candidate;
            }
        }
    }
    best
}

struct PowerOfTwoPicker {
    choices: usize,
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
}

impl WorkerPicker for PowerOfTwoPicker {
    fn pick(
        &self,
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
        let mut drawn = DRAWN_ROWS.take();
        let row = match &self.rng {
            None => lowest_cost_of_draw(
                candidates.len(),
                self.choices,
                |row| candidates[row].cost(),
                |upper| fastrand::usize(..upper),
                &mut drawn,
            ),
            Some(rng) => {
                // Canonical order is required only for deterministic replay.
                let mut rows: Vec<usize> = (0..candidates.len()).collect();
                rows.sort_unstable_by_key(|row| {
                    let worker = candidates[*row].worker();
                    (worker.worker_id, worker.dp_rank)
                });
                let mut rng = rng.lock();
                rows[lowest_cost_of_draw(
                    rows.len(),
                    self.choices,
                    |index| candidates[rows[index]].cost(),
                    |upper| rng.usize(..upper),
                    &mut drawn,
                )]
            }
        };
        DRAWN_ROWS.set(drawn);
        Ok(row)
    }
}

/// The default cost for `worker_label` with a pick of the lowest among `choices` (at least 2)
/// random candidates. `rng` is as in [`super::default_policy_with_rng`].
pub fn power_of_two_policy_with_rng(
    config: SelectionConfig,
    worker_label: &'static str,
    choices: usize,
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
) -> WorkerSelectionPolicy {
    policy(
        PolicyParameters::from(&config),
        config,
        worker_label,
        choices,
        rng,
    )
}

fn policy(
    parameters: PolicyParameters,
    config: SelectionConfig,
    worker_label: &'static str,
    choices: usize,
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
) -> WorkerSelectionPolicy {
    WorkerSelectionPolicy::new(
        config,
        worker_label,
        vec![scorer::build(&parameters, worker_label)],
        Box::new(PowerOfTwoPicker { choices, rng }),
    )
    .with_exclusive_affinity(true)
}

pub(crate) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(
        POWER_OF_TWO_POLICY_TYPE,
        Arc::new(|parameters| {
            let parameters: Parameters = parameters.deserialize()?;
            parameters.validate()?;
            Ok(Arc::new(
                move |config: &SelectionConfig, role, _partition| {
                    policy(
                        role_parameters(config, role, PolicyParameters::from(config)),
                        config.clone(),
                        role.default_selector_label(),
                        parameters.choices,
                        None,
                    )
                },
            ))
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Returns the scripted values in order, reduced below each bound.
    fn scripted(values: &[usize]) -> impl FnMut(usize) -> usize + '_ {
        let mut values = values.iter();
        move |upper| values.next().expect("an unscripted draw") % upper
    }

    #[test]
    fn two_choices_read_two_costs_and_take_the_lower() {
        let costs = [5.0, 1.0, 9.0, 7.0, 3.0];
        let read = RefCell::new(Vec::new());
        let cost = |row: usize| {
            read.borrow_mut().push(row);
            costs[row]
        };
        let mut drawn = Vec::new();
        // Floyd's sampling draws below 4 and then below 5: rows 3 and 2.
        let picked = lowest_cost_of_draw(costs.len(), 2, cost, scripted(&[3, 2]), &mut drawn);
        assert_eq!(picked, 3, "the lower cost of rows 3 and 2");
        assert_eq!(*read.borrow(), [3, 2], "the global minimum is never read");
    }

    #[test]
    fn a_repeated_draw_takes_the_next_distinct_row() {
        let mut drawn = Vec::new();
        lowest_cost_of_draw(5, 2, |_| 0.0, scripted(&[3, 3, 1]), &mut drawn);
        assert_eq!(drawn, [3, 4]);
    }

    #[test]
    fn fewer_candidates_than_choices_considers_all_of_them() {
        let costs = [4.0, 2.0, 6.0];
        for len in 1..=costs.len() {
            let mut drawn = Vec::new();
            // No draw is scripted: every candidate is considered and none tie.
            let picked = lowest_cost_of_draw(len, 4, |row| costs[row], scripted(&[]), &mut drawn);
            assert_eq!(drawn, (0..len).collect::<Vec<_>>());
            assert_eq!(picked, if len == 1 { 0 } else { 1 });
        }
    }

    #[test]
    fn ties_among_the_draw_are_broken_by_a_draw() {
        let mut drawn = Vec::new();
        for (tie_break, expected) in [(0, 1), (1, 0)] {
            let picked = lowest_cost_of_draw(2, 2, |_| 3.0, scripted(&[tie_break]), &mut drawn);
            assert_eq!(picked, expected);
        }
    }

    /// Exact-minimum selection gives the single cheapest row every pick. Two choices give it only
    /// the picks that draw it, never choose outside the draw, and never choose the costliest row.
    #[test]
    fn picks_spread_over_the_draws_instead_of_the_global_minimum() {
        const ROWS: usize = 16;
        const PICKS: usize = 4000;
        let costs: Vec<f64> = (0..ROWS).map(|row| ((row * 7) % ROWS) as f64).collect();
        let mut rng = fastrand::Rng::with_seed(7);
        let mut counts = [0usize; ROWS];
        let mut drawn = Vec::new();
        for _ in 0..PICKS {
            let picked = lowest_cost_of_draw(
                ROWS,
                2,
                |row| costs[row],
                |upper| rng.usize(..upper),
                &mut drawn,
            );
            assert_eq!(drawn.len(), 2);
            assert_ne!(drawn[0], drawn[1]);
            assert!(drawn.contains(&picked));
            assert_eq!(costs[picked], costs[drawn[0]].min(costs[drawn[1]]));
            counts[picked] += 1;
        }
        let cheapest = costs.iter().position(|cost| *cost == 0.0).unwrap();
        let costliest = costs.iter().position(|cost| *cost == 15.0).unwrap();
        // The cheapest row wins whenever it is drawn: 2 / 16 of picks.
        let share = counts[cheapest] as f64 / PICKS as f64;
        assert!((0.09..0.16).contains(&share), "cheapest share {share}");
        assert_eq!(counts[costliest], 0);
        assert_eq!(counts.iter().filter(|count| **count > 0).count(), ROWS - 1);
    }

    #[test]
    fn choices_default_to_two_and_reject_fewer() {
        assert_eq!(Parameters::default().choices, 2);
        for (choices, valid) in [(0, false), (1, false), (2, true), (5, true)] {
            assert_eq!(
                Parameters { choices }.validate().is_ok(),
                valid,
                "{choices}"
            );
        }
    }
}
