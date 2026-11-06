// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV-capacity-aware decode worker selection.
//!
//! The default decode cost `D + W * N` ignores a rank's KV capacity, so a rank can be booked past
//! what it holds while others have room. Here `D` is `decode_cost_blocks` (blocks booked on the
//! rank plus this request's), `N` is `active_requests`, and a rank's capacity is the
//! `total_kv_blocks` it advertises, else the `capacity_blocks` parameter.
//!
//! - A rank fits when `D <= capacity_fraction * capacity`, compared exactly with the fraction in
//!   parts per million. A rank without a capacity fits, and is warned about once.
//! - Among fitting ranks: fewest requests first (`count_priority`), then the lowest
//!   `(D + W * N) * taint`, where `taint` is the preferred-taint multiplier, as in the default.
//! - When no rank fits: the lowest `D`, ignoring taints.
//! - Ties are broken uniformly at random. Session-affinity targets bind, as in the default.
//!
//! The capacity may be an estimate, so the policy never queues or rejects a request. It is meant
//! for decode pools; on other pools it logs a warning at startup. With `capacity_fraction: null`
//! and `count_priority: false` it ties exactly like the default plain-decode cost.

use std::sync::Arc;

use dynamo_kv_router::plugins::worker_selection::{
    ScoredWorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::{KvRouterConfig, WorkerType};
use parking_lot::Mutex;
use rustc_hash::FxHashSet;

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-kv-capacity-aware";

/// `capacity_fraction` applies in parts per million, so the fit test is exact: with 0.7 and a
/// capacity of 90, 63 blocks fit, although the f64 product is 62.999...
const PPM: f64 = 1_000_000.0;

/// Instance parameters. Every field is optional; unknown keys fail startup.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Capacity in KV blocks for ranks that do not report `total_kv_blocks`.
    capacity_blocks: Option<u64>,
    /// Share of a rank's capacity a booking may fill; `null` disables the filter.
    capacity_fraction: Option<f64>,
    /// Prefer fewer active requests before the block cost.
    count_priority: bool,
    /// `W` in `D + W * N`.
    decode_active_request_weight: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            capacity_blocks: None,
            capacity_fraction: Some(1.0),
            count_priority: true,
            decode_active_request_weight: 0.0,
        }
    }
}

/// One candidate rank as the rule reads it.
#[derive(Debug, Clone, Copy)]
struct Rank {
    decode_cost: f64,
    requests: usize,
    capacity: Option<u64>,
    taint: Option<f64>,
}

/// A rank's place in the order, lower first. Every fitting rank comes before every other; fitting
/// ranks are ordered by `(count, cost)` and the others by decode cost.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
enum Key {
    Fits { count: usize, cost: f64 },
    Over { decode_cost: f64 },
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if let Some(fraction) = self.capacity_fraction
            && !(fraction.is_finite() && (fraction * PPM).round() >= 1.0)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "capacity_fraction must be a finite number of at least 0.000001, or null for no capacity filter",
            ));
        }
        if self.capacity_blocks == Some(0) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "capacity_blocks must be greater than 0",
            ));
        }
        let weight = self.decode_active_request_weight;
        if !(weight.is_finite() && weight >= 0.0) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "decode_active_request_weight must be a finite number greater than or equal to 0",
            ));
        }
        Ok(())
    }

    fn fits(&self, rank: &Rank) -> bool {
        match (rank.capacity, self.capacity_fraction) {
            (Some(capacity), Some(fraction)) => {
                let ppm = (fraction * PPM).round() as u128;
                // `decode_cost` counts whole blocks, so the cast to an integer drops nothing.
                rank.decode_cost as u128 * PPM as u128 <= ppm * u128::from(capacity)
            }
            _ => true,
        }
    }

    /// The key `rank` has if it fits.
    fn fit_key(&self, rank: &Rank) -> Key {
        let count = if self.count_priority {
            rank.requests
        } else {
            0
        };
        let cost = (rank.decode_cost + self.decode_active_request_weight * rank.requests as f64)
            * rank.taint.unwrap_or(1.0);
        Key::Fits { count, cost }
    }

    /// The key of `rank`, given the key it has if it fits.
    fn key_given(&self, rank: &Rank, fit_key: Key) -> Key {
        if self.fits(rank) {
            fit_key
        } else {
            Key::Over {
                decode_cost: rank.decode_cost,
            }
        }
    }

    #[cfg(test)]
    fn key(&self, rank: &Rank) -> Key {
        self.key_given(rank, self.fit_key(rank))
    }
}

/// The row to place the request on, or `None` without rows. Ties are drawn uniformly in the order
/// `rows` yields them; `draw(n)` returns an index below `n`.
///
/// `rank(row)` comes without a capacity, and `capacity(row)` is a lookup the filter needs only for a
/// row that can still win: once a fitting row is found, a row whose fitting key is already worse
/// loses whether it fits or not.
fn select_row(
    parameters: &Parameters,
    rows: impl Iterator<Item = usize>,
    rank: impl Fn(usize) -> Rank,
    capacity: impl Fn(usize) -> Option<u64>,
    mut draw: impl FnMut(usize) -> usize,
) -> Option<usize> {
    let filter = parameters.capacity_fraction.is_some();
    let mut best: Option<(Key, usize)> = None;
    let mut ties = 0;
    for row in rows {
        let mut rank = rank(row);
        let fit_key = parameters.fit_key(&rank);
        if let Some((best_key @ Key::Fits { .. }, _)) = best
            && fit_key > best_key
        {
            continue;
        }
        if filter {
            rank.capacity = capacity(row);
        }
        let key = parameters.key_given(&rank, fit_key);
        match best {
            Some((best_key, _)) if key > best_key => {}
            Some((best_key, _)) if key == best_key => {
                ties += 1;
                if draw(ties) == 0 {
                    best = Some((key, row));
                }
            }
            _ => {
                best = Some((key, row));
                ties = 1;
            }
        }
    }
    best.map(|(_, row)| row)
}

struct KvCapacityAwarePicker {
    parameters: Parameters,
    /// Ranks already warned about a missing capacity. Pruned to the candidates whenever it
    /// outgrows them, so workers that come and go do not accumulate.
    warned: Mutex<FxHashSet<WorkerWithDpRank>>,
    /// Seeded generator for deterministic tests; production picks use the thread's generator.
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
}

impl KvCapacityAwarePicker {
    fn new(parameters: Parameters, rng: Option<Arc<Mutex<fastrand::Rng>>>) -> Self {
        Self {
            parameters,
            warned: Mutex::default(),
            rng,
        }
    }

    /// Warn once about the rank at `row`, which reports no capacity that `capacity_blocks` could
    /// replace.
    fn warn_unknown(&self, candidates: &[ScoredWorkerCandidate], row: usize) {
        let worker = candidates[row].worker();
        let mut warned = self.warned.lock();
        if !warned.insert(worker) {
            return;
        }
        tracing::warn!(
            policy = POLICY_TYPE,
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            "no total_kv_blocks reported and no capacity_blocks set; the rank is not filtered"
        );
        if warned.len() > candidates.len() {
            warned.retain(|worker| {
                candidates
                    .iter()
                    .any(|candidate| candidate.worker() == *worker)
            });
        }
    }
}

impl WorkerPicker for KvCapacityAwarePicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD | WorkerInputs::PREFERRED_TAINT
    }

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
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        if load.len() != candidates.len() {
            return Err(WorkerSelectionPolicyError::failed(
                "load input does not match candidates",
            ));
        }
        let rank = |row: usize| Rank {
            decode_cost: load[row].decode_cost_blocks(),
            requests: load[row].active_requests(),
            capacity: None,
            taint: candidates[row].preferred_taint_multiplier(),
        };
        let capacity = |row: usize| {
            let capacity = input
                .info(row)
                .and_then(|info| info.total_kv_blocks())
                .or(self.parameters.capacity_blocks);
            if capacity.is_none() {
                self.warn_unknown(candidates, row);
            }
            capacity
        };
        let row = match &self.rng {
            None => select_row(&self.parameters, 0..candidates.len(), rank, capacity, |n| {
                fastrand::usize(0..n)
            }),
            Some(rng) => {
                // Canonical order only for deterministic replay, as the default picker does.
                let mut rows: Vec<usize> = (0..candidates.len()).collect();
                rows.sort_unstable_by_key(|&row| {
                    let worker = candidates[row].worker();
                    (worker.worker_id, worker.dp_rank)
                });
                let mut rng = rng.lock();
                select_row(&self.parameters, rows.into_iter(), rank, capacity, |n| {
                    rng.usize(0..n)
                })
            }
        }
        .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?;
        Ok(row)
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;
    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            if worker_type != WorkerType::Decode {
                tracing::warn!(
                    policy = POLICY_TYPE,
                    role = worker_type.as_str(),
                    "selected for a non-decode pool; it ranks by decode blocks and requests only, \
                     ignoring cache overlap and prefill load"
                );
            }
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(KvCapacityAwarePicker::new(parameters, None)),
            )
            .with_exclusive_affinity(true)
        },
    ))
}

pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{
        SchedulingRequest, WorkerLoadProjection, WorkerSelectionInput, WorkerSelector,
    };

    use super::*;

    fn parameters(fraction: f64, count_priority: bool, weight: f64) -> Parameters {
        Parameters {
            capacity_blocks: None,
            capacity_fraction: fraction.is_finite().then_some(fraction),
            count_priority,
            decode_active_request_weight: weight,
        }
    }

    /// Ranks given as `(D, N, capacity, taint)`.
    fn ranks(rows: &[(u64, usize, Option<u64>, Option<f64>)]) -> Vec<Rank> {
        rows.iter()
            .map(|&(decode_cost, requests, capacity, taint)| Rank {
                decode_cost: decode_cost as f64,
                requests,
                capacity,
                taint,
            })
            .collect()
    }

    /// Every row at the minimum key, and whether no rank fits.
    fn tie_set(parameters: &Parameters, ranks: &[Rank]) -> (Vec<usize>, bool) {
        let keys: Vec<Key> = ranks.iter().map(|rank| parameters.key(rank)).collect();
        let min = keys
            .iter()
            .copied()
            .reduce(|a, b| if b < a { b } else { a })
            .expect("non-empty");
        let ties = (0..keys.len()).filter(|&row| keys[row] == min).collect();
        (ties, matches!(min, Key::Over { .. }))
    }

    fn ties(
        parameters: &Parameters,
        rows: &[(u64, usize, Option<u64>, Option<f64>)],
    ) -> Vec<usize> {
        tie_set(parameters, &ranks(rows)).0
    }

    const CAP: Option<u64> = Some(1_000);

    /// A rank wins on the fit boundary only while the request still fits on it. In each case the
    /// rank one block over would win if it fit, and the boundary rank, if it did not fit, would
    /// lose to the rank with more requests.
    #[test]
    fn the_fit_test_decides_at_its_boundary() {
        let p = Parameters::default();
        // Exactly at capacity fits; one block more does not.
        assert_eq!(
            ties(
                &p,
                &[
                    (1_000, 1, CAP, None),
                    (1_001, 0, CAP, None),
                    (100, 2, CAP, None)
                ]
            ),
            [0]
        );
        // 0.95 x 16,001 = 15,200.95: 15,200 fits, 15,201 does not.
        let p = parameters(0.95, true, 0.0);
        let cap = Some(16_001);
        assert_eq!(
            ties(
                &p,
                &[
                    (15_200, 1, cap, None),
                    (15_201, 0, cap, None),
                    (100, 2, cap, None)
                ]
            ),
            [0]
        );
        // 0.7 x 90 = 63 exactly, although the f64 product is 62.999...: 63 fits, 64 does not.
        let p = parameters(0.7, true, 0.0);
        let cap = Some(90);
        assert_eq!(
            ties(
                &p,
                &[(63, 1, cap, None), (64, 0, cap, None), (10, 2, cap, None)]
            ),
            [0]
        );
        // A rank without a capacity is not filtered.
        assert_eq!(
            ties(
                &p,
                &[
                    (5_000, 0, None, None),
                    (10, 1, cap, None),
                    (64, 0, cap, None)
                ]
            ),
            [0]
        );
    }

    /// Among fitting ranks the request count comes first, then `D + W * N`; equal keys tie.
    #[test]
    fn fitting_ranks_are_ordered_by_count_then_cost() {
        let p = Parameters::default();
        assert_eq!(
            ties(
                &p,
                &[
                    (600, 1, CAP, None),
                    (500, 1, CAP, None),
                    (100, 2, CAP, None),
                    (500, 1, CAP, None)
                ]
            ),
            [1, 3]
        );
        // Without count priority, the default cost decides: 2,100 / 2,200 / 1,900.
        let p = parameters(1.0, false, 1_000.0);
        let cap = Some(10_000);
        assert_eq!(
            ties(
                &p,
                &[
                    (100, 2, cap, None),
                    (2_200, 0, cap, None),
                    (900, 1, cap, None)
                ]
            ),
            [2]
        );
        // A fractional W is compared as the default compares it: 11.5 / 11.0 / 11.0.
        let p = parameters(f64::INFINITY, false, 0.5);
        assert_eq!(
            ties(
                &p,
                &[
                    (10, 3, None, None),
                    (11, 0, None, None),
                    (10, 2, None, None)
                ]
            ),
            [1, 2]
        );
    }

    /// When nothing fits, the smallest footprint wins whatever its requests or taint, so a
    /// preferred rank never receives a request it cannot hold.
    #[test]
    fn nothing_fitting_falls_back_to_the_smallest_footprint() {
        let p = Parameters::default();
        let rows = [
            (1_500, 0, CAP, None),
            (1_200, 9, CAP, None),
            (1_300, 1, CAP, Some(0.4)),
        ];
        assert_eq!(tie_set(&p, &ranks(&rows)), (vec![1], true));
        // A preferred rank that does not fit loses; one that fits wins: 900 / 950 x 0.4 = 380.
        assert_eq!(
            ties(
                &p,
                &[
                    (1_100, 0, CAP, Some(0.4)),
                    (900, 1, CAP, None),
                    (950, 1, CAP, Some(0.4))
                ]
            ),
            [2]
        );
    }

    /// The preferred-taint multiplier scales the cost after the count, never the count itself.
    #[test]
    fn taint_scales_the_cost_after_the_count() {
        let p = Parameters::default();
        // Keys: (1, 500), (1, 600 x 0.8 = 480), (2, 100 x 0.4 = 40). A multiplier that also
        // scaled the count would let rank 2 win.
        assert_eq!(
            ties(
                &p,
                &[
                    (500, 1, CAP, None),
                    (600, 1, CAP, Some(0.8)),
                    (100, 2, CAP, Some(0.4))
                ]
            ),
            [1]
        );
    }

    /// With no capacity filter and no count priority, the tie set is that of the default
    /// plain-decode cost `D + W * N`.
    #[test]
    fn degenerate_parameters_tie_exactly_like_the_default_cost() {
        let mut rng = fastrand::Rng::with_seed(96);
        for _ in 0..4000 {
            let n_ranks = rng.usize(1..40);
            let w = [0.0, 500.0, 1000.0, 2000.0][rng.usize(0..4)];
            let rows: Vec<(u64, usize, Option<u64>, Option<f64>)> = (0..n_ranks)
                .map(|_| {
                    let d = rng.u64(0..4) * 1000 + rng.u64(0..3);
                    (d, rng.usize(0..4), Some(16_000), None)
                })
                .collect();
            let (ties, fallback) = tie_set(&parameters(f64::INFINITY, false, w), &ranks(&rows));
            let cost: Vec<f64> = rows.iter().map(|r| r.0 as f64 + w * r.1 as f64).collect();
            let min = cost.iter().cloned().fold(f64::INFINITY, f64::min);
            let expected: Vec<usize> = (0..n_ranks).filter(|&r| cost[r] == min).collect();
            assert_eq!(ties, expected);
            assert!(!fallback);
        }
    }

    /// On random states the tie set equals the one an integer oracle computes, and the picked row
    /// is one of its members.
    #[test]
    fn random_states_match_an_integer_oracle() {
        let mut rng = fastrand::Rng::with_seed(7);
        for _ in 0..4000 {
            let n_ranks = rng.usize(1..33);
            // The fraction as a ratio of integers: 95 / 100 or 1 / 1.
            let (num, den, fraction) = [(95, 100, 0.95), (1, 1, 1.0)][rng.usize(0..2)];
            let rows: Vec<(u64, usize, Option<u64>, Option<f64>)> = (0..n_ranks)
                .map(|_| {
                    let cap = rng.u64(10_000..20_000);
                    (rng.u64(0..25_000), rng.usize(0..10), Some(cap), None)
                })
                .collect();
            let fits = |r: &(u64, usize, Option<u64>, Option<f64>)| r.0 * den <= num * r.2.unwrap();
            let expected: Vec<usize> = if rows.iter().any(fits) {
                let best = rows
                    .iter()
                    .filter(|r| fits(r))
                    .map(|r| (r.1, r.0))
                    .min()
                    .unwrap();
                (0..n_ranks)
                    .filter(|&i| fits(&rows[i]) && (rows[i].1, rows[i].0) == best)
                    .collect()
            } else {
                let best = rows.iter().map(|r| r.0).min().unwrap();
                (0..n_ranks).filter(|&i| rows[i].0 == best).collect()
            };
            let p = parameters(fraction, true, 0.0);
            let rs = ranks(&rows);
            let (ties, fallback) = tie_set(&p, &rs);
            assert_eq!(ties, expected);
            assert_eq!(fallback, !rows.iter().any(fits));
            let picked = select_row(
                &p,
                0..rs.len(),
                |r| Rank {
                    capacity: None,
                    ..rs[r]
                },
                |r| rs[r].capacity,
                |k| rng.usize(0..k),
            );
            assert!(ties.contains(&picked.expect("non-empty")));
        }
    }

    /// The same seeded generator reproduces the same picks.
    #[test]
    fn seeded_picks_repeat() {
        let picks = |seed| {
            let rng = Arc::new(Mutex::new(fastrand::Rng::with_seed(seed)));
            let policy = host_policy(KvCapacityAwarePicker::new(Parameters::default(), Some(rng)));
            (0..50)
                .map(|_| host_select(&policy, None, &[(0, 0, 0); 24]))
                .collect::<Vec<_>>()
        };
        assert_eq!(picks(11), picks(11));
        assert_ne!(picks(11), picks(12));
    }

    /// A worker advertising `total_kv_blocks` (None: not reported) and taints.
    struct TestWorker {
        capacity: Option<u64>,
        taints: HashSet<String>,
    }

    impl WorkerConfigLike for TestWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }
        fn data_parallel_size(&self) -> u32 {
            1
        }
        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }
        fn total_kv_blocks(&self) -> Option<u64> {
            self.capacity
        }
        fn taints(&self) -> &HashSet<String> {
            &self.taints
        }
    }

    /// The host's `WorkerSelectionPolicy` around `picker`, as `provider` builds it for decode.
    fn host_policy(picker: KvCapacityAwarePicker) -> WorkerSelectionPolicy {
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "decode",
            Vec::new(),
            Box::new(picker),
        )
    }

    /// Select through the host's policy among workers 1..=k given as
    /// `(booked decode blocks, this request's blocks, active requests)`, each advertising
    /// `total_kv_blocks = reported`; returns the worker id.
    fn host_select(
        policy: &WorkerSelectionPolicy,
        reported: Option<u64>,
        workers: &[(usize, usize, usize)],
    ) -> u64 {
        host_select_with(policy, workers, RoutingConstraints::default(), |_| {
            TestWorker {
                capacity: reported,
                taints: HashSet::new(),
            }
        })
    }

    fn host_select_with(
        policy: &WorkerSelectionPolicy,
        workers: &[(usize, usize, usize)],
        routing_constraints: RoutingConstraints,
        worker: impl Fn(u64) -> TestWorker,
    ) -> u64 {
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: 160,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints,
            router_config_override: None,
            track_prefill_tokens: false,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        };
        for (i, &(booked, blocks, active)) in workers.iter().enumerate() {
            request.worker_loads.insert(
                WorkerWithDpRank::from_worker_id(i as u64 + 1),
                WorkerLoadProjection {
                    active_decode_blocks: booked,
                    additional_active_blocks: blocks,
                    active_requests: active,
                    ..Default::default()
                },
            );
        }
        let configs: HashMap<u64, TestWorker> = (1..=workers.len() as u64)
            .map(|id| (id, worker(id)))
            .collect();
        policy
            .select_worker(WorkerSelectionInput::configured(
                &configs,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap()
            .worker
            .worker_id
    }

    /// Through the host: the capacity filter on the advertised `total_kv_blocks`, the count priority
    /// and the over-capacity fallback.
    #[test]
    fn selects_through_the_host_policy() {
        let policy = host_policy(KvCapacityAwarePicker::new(Parameters::default(), None));
        let pick = |workers: &[(usize, usize, usize)]| host_select(&policy, Some(1_000), workers);
        // Worker 1 has the fewest requests but 950 + 100 > 1,000; worker 2 fits with one more request.
        assert_eq!(pick(&[(950, 100, 0), (100, 100, 1), (100, 100, 2)]), 2);
        // Exactly at capacity fits (<=).
        assert_eq!(pick(&[(900, 100, 0), (100, 100, 1)]), 1);
        // Nothing fits: the smallest footprint, whatever its request count.
        assert_eq!(
            pick(&[(1_500, 100, 0), (1_200, 100, 9), (1_300, 100, 1)]),
            2
        );
    }

    /// Through the host, the request's preferred taints reach the picker: among workers with equal
    /// request counts the preferred one wins at a higher cost.
    #[test]
    fn preferred_taints_reach_the_picker_through_the_host() {
        let policy = host_policy(KvCapacityAwarePicker::new(Parameters::default(), None));
        // D = 500 / 600 / 200 with 1 / 1 / 2 requests; workers 2 and 3 carry the "fast" taint.
        let workers = [(400, 100, 1), (500, 100, 1), (100, 100, 2)];
        let worker = |id: u64| TestWorker {
            capacity: Some(10_000),
            taints: if id == 1 {
                HashSet::new()
            } else {
                HashSet::from(["fast".to_string()])
            },
        };
        let preferred = RoutingConstraints {
            preferred_taints: HashMap::from([("fast".to_string(), 1.0)]),
            ..Default::default()
        };
        // exp(-tanh(1)) = 0.47 scales worker 2 to 280, below worker 1's 500; worker 3 has more
        // requests, so its lower cost does not matter.
        assert_eq!(host_select_with(&policy, &workers, preferred, worker), 2);
        assert_eq!(
            host_select_with(&policy, &workers, RoutingConstraints::default(), worker),
            1
        );
    }

    /// The advertised `total_kv_blocks` is the capacity; `capacity_blocks` applies only to a rank that
    /// does not report it (a reported zero counts as not reported). With neither, nothing is filtered.
    #[test]
    fn capacity_is_total_kv_blocks_then_the_parameter() {
        let workers = [(950, 100, 0), (100, 100, 1), (100, 100, 2)];
        let with = |capacity_blocks| {
            let parameters = Parameters {
                capacity_blocks,
                ..Parameters::default()
            };
            host_policy(KvCapacityAwarePicker::new(parameters, None))
        };
        // The reported capacity wins in both directions.
        assert_eq!(host_select(&with(Some(100_000)), Some(1_000), &workers), 2);
        assert_eq!(host_select(&with(Some(1_000)), Some(100_000), &workers), 1);
        // Not reported, or reported as zero: the parameter applies.
        assert_eq!(host_select(&with(Some(1_000)), None, &workers), 2);
        assert_eq!(host_select(&with(Some(1_000)), Some(0), &workers), 2);
        // Neither: not filtered, so the fewest requests wins.
        assert_eq!(host_select(&with(None), None, &workers), 1);
    }

    /// A decode-only policy YAML alongside the deprecated decode-weight flag, applied and validated
    /// as the Python `KvRouterConfig` constructor does: it resolves, decode skips the worker that
    /// does not fit (which the default `D + 1000 N` would pick), and prefill still constructs.
    #[test]
    fn resolves_alongside_the_decode_weight_flag() {
        use dynamo_kv_router::{RoutingPartitionRef, WorkerType};

        const POLICY_YAML: &str = r#"
worker_selection:
  decode: kvca
  instances:
    - name: kvca
      type: dynamo-kv-capacity-aware
      parameters:
        capacity_blocks: 20000
        capacity_fraction: 1.0
        count_priority: true
        decode_active_request_weight: 0
"#;
        let configured = |yaml: &str| {
            let file = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(file.path(), yaml).unwrap();
            let mut config = KvRouterConfig {
                router_policy_config: Some(file.path().display().to_string()),
                decode_active_request_weight: 1000.0,
                ..Default::default()
            };
            config.apply_policy_config().unwrap();
            config.validate().unwrap();
            assert_eq!(config.decode_active_request_weight, 1000.0);
            let mut registry = crate::default_registry();
            crate::register(&mut registry).unwrap();
            let factory = registry
                .resolve(&config)
                .unwrap()
                .expect("a configured decode instance resolves");
            (file, config, factory)
        };
        let partition = RoutingPartitionRef::new("model", "default");

        // The YAML resolves and builds both stages.
        let (_file, config, factory) = configured(POLICY_YAML);
        factory(&config, WorkerType::Decode, partition);
        factory(&config, WorkerType::Prefill, partition);

        // Same shape with a capacity of 1,000 blocks for workers that report none, so the two
        // policies disagree on one state.
        let small = POLICY_YAML.replace("20000", "1000");
        let (_file, config, factory) = configured(&small);
        let workers = [(950, 100, 0), (100, 100, 1), (100, 100, 2)];
        let decode = factory(&config, WorkerType::Decode, partition);
        assert_eq!(host_select(&decode, None, &workers), 2);
        // Like the default, the policy binds session-affinity targets.
        assert!(WorkerSelector::<TestWorker>::uses_exclusive_affinity_target(&decode));
        let default_decode = crate::default_policy(config.clone(), "decode");
        assert_eq!(host_select(&default_decode, None, &workers), 1);
    }

    #[test]
    fn rejects_invalid_parameters() {
        let with = |f: fn(&mut Parameters)| {
            let mut p = Parameters::default();
            f(&mut p);
            p.validate()
        };
        assert!(with(|p| p.capacity_fraction = Some(0.0)).is_err());
        assert!(with(|p| p.capacity_fraction = Some(0.000_000_4)).is_err());
        assert!(with(|p| p.capacity_fraction = Some(f64::NAN)).is_err());
        assert!(with(|p| p.capacity_blocks = Some(0)).is_err());
        assert!(with(|p| p.decode_active_request_weight = -1.0).is_err());
        assert!(with(|p| p.capacity_fraction = None).is_ok());
        assert!(Parameters::default().validate().is_ok());
    }
}
