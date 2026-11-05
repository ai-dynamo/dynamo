// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV-capacity-aware decode worker selection.
//!
//! The default decode cost `D + W * N` (booked decode blocks plus a weight per active request)
//! places requests without regard to a rank's KV capacity, so a busy rank can be booked past what
//! it holds while others have room. This policy keeps the default's inputs and changes only the
//! choice. For each candidate rank it reads `D_r = decode_cost_blocks` (blocks already booked on the
//! rank, prefill-stage requests included, plus this request's blocks), `N_r = active_requests`
//! (prefill-stage requests included), and the rank's KV capacity `cap_r` in blocks:
//!
//! 1. `fits = { r : cap_r unknown, or D_r <= g * cap_r }`. `g` is `capacity_fraction`; without it
//!    every rank fits. The product is taken in f64 and compared with `<=`, not rounded first.
//! 2. If `fits` is not empty, pick the minimum of `key_r = (count_priority ? N_r : 0, D_r + W * N_r)`
//!    within it. Integer-valued inputs are compared exactly (u128), so ties are exact.
//! 3. If no rank fits, pick the minimum `D_r` over all candidates. Selection never waits or fails on
//!    a non-empty candidate list, even when every rank is already over capacity.
//! 4. Ties are broken uniformly at random, as the default picker does; with an injected generator
//!    the tied rows are put in `(worker_id, dp_rank)` order first, so a seed reproduces the choice.
//!
//! `cap_r` is the `total_kv_blocks` the rank advertises (`WorkerInfo::total_kv_blocks`); for a rank
//! that does not report it, the `capacity_blocks` parameter. With neither, the rank is not filtered
//! and a warning is logged once per worker rank. Some adapters advertise an aggregate or
//! representative-rank value, so the capacity only steers the choice; it never rejects a request.
//!
//! With `capacity_fraction: null` and `count_priority: false` the policy selects exactly as the
//! default plain-decode cost does (the same tie sets). A pinned worker is returned as is, and an
//! empty candidate list is an error, as in the default picker.
//!
//! `preferred_taint_multiplier` multiplies the scalar cost
//! `((count_priority ? N_r * 2^24 : 0) + D_r + W * N_r) * taint_r` inside `fits` only, as the default
//! multiplies its cost. The over-capacity fallback ignores it, so a preference can never place a
//! request on a rank that cannot hold it. With every multiplier 1 the scalar order equals the key
//! order above.

use std::collections::HashSet;
use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy,
    WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::protocols::WorkerWithDpRank;
use parking_lot::Mutex;

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-kv-capacity-aware";

/// Scale of the request count in the scalar cost used with preferred-taint multipliers; the cost
/// part `D + W * N` must stay below it for the count to dominate.
const COUNT_SCALE: u128 = 1 << 24;
/// Largest integer every f64 holds exactly.
const EXACT_F64: f64 = 9_007_199_254_740_992.0;

/// Instance parameters. Every field is optional; unknown keys fail startup.
#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// KV capacity in blocks for ranks that do not report `total_kv_blocks`. Absent: such ranks
    /// are not filtered.
    capacity_blocks: Option<u64>,
    /// The share `g` of a rank's capacity a booking may fill. `null` removes the filter.
    capacity_fraction: Option<f64>,
    /// Prefer fewer active requests before the block cost.
    count_priority: bool,
    /// `W` in `D + W * N`, as the default policy's `decode_active_request_weight`.
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

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if let Some(g) = self.capacity_fraction
            && !(g.is_finite() && g > 0.0)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "capacity_fraction must be a finite number greater than 0, or null for no capacity filter",
            ));
        }
        if self.capacity_blocks == Some(0) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "capacity_blocks must be greater than 0",
            ));
        }
        let w = self.decode_active_request_weight;
        if !(w.is_finite() && w >= 0.0) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "decode_active_request_weight must be a finite number greater than or equal to 0",
            ));
        }
        Ok(())
    }

    fn rule(&self) -> Rule {
        Rule {
            capacity_fraction: self.capacity_fraction,
            count_priority: self.count_priority,
            decode_active_request_weight: self.decode_active_request_weight,
        }
    }
}

/// The selection rule, separate from where its inputs come from.
#[derive(Debug, Clone, Copy)]
struct Rule {
    capacity_fraction: Option<f64>,
    count_priority: bool,
    decode_active_request_weight: f64,
}

/// One candidate rank as the rule sees it.
#[derive(Debug, Clone, Copy)]
struct Rank {
    decode_cost: f64,
    requests: usize,
    capacity: Option<u64>,
    taint: Option<f64>,
}

/// The rows that tie for the minimum, and whether no rank fit (the over-capacity fallback).
#[derive(Debug, Default, PartialEq)]
struct Decision {
    ties: Vec<usize>,
    fallback: bool,
}

/// `value` as an exact integer, when it is one an f64 holds exactly.
fn exact(value: f64) -> Option<u128> {
    (value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= EXACT_F64)
        .then_some(value as u128)
}

/// Rows whose key equals the minimum (exact equality; the caller supplies exact keys when it can).
fn argmin<K: PartialOrd + Copy>(rows: impl Iterator<Item = (usize, K)>, ties: &mut Vec<usize>) {
    ties.clear();
    let mut best: Option<K> = None;
    for (row, key) in rows {
        match best {
            Some(b) if key > b => {}
            Some(b) if key == b => ties.push(row),
            _ => {
                best = Some(key);
                ties.clear();
                ties.push(row);
            }
        }
    }
}

impl Rule {
    fn fits(&self, rank: &Rank) -> bool {
        match (rank.capacity, self.capacity_fraction) {
            (Some(capacity), Some(g)) => rank.decode_cost <= g * capacity as f64,
            _ => true,
        }
    }

    /// Decide on `ranks`; `None` for an empty list. `unknown` receives the rows without a capacity.
    fn decide(
        &self,
        ranks: &[Rank],
        decision: &mut Decision,
        unknown: &mut Vec<usize>,
    ) -> Option<()> {
        if ranks.is_empty() {
            return None;
        }
        unknown.clear();
        unknown.extend(
            ranks
                .iter()
                .enumerate()
                .filter(|(_, r)| r.capacity.is_none())
                .map(|(row, _)| row),
        );
        let fitting = || ranks.iter().enumerate().filter(|(_, r)| self.fits(r));
        decision.fallback = fitting().next().is_none();
        let w = self.decode_active_request_weight;
        if decision.fallback {
            // No rank fits: least booked blocks (D and the booked blocks differ by this request only).
            match ranks
                .iter()
                .map(|r| exact(r.decode_cost))
                .collect::<Option<Vec<_>>>()
            {
                Some(d) => argmin(d.into_iter().enumerate(), &mut decision.ties),
                None => argmin(
                    ranks.iter().map(|r| r.decode_cost).enumerate(),
                    &mut decision.ties,
                ),
            }
            return Some(());
        }
        let count = |r: &Rank| if self.count_priority { r.requests } else { 0 };
        if fitting().any(|(_, r)| r.taint.is_some_and(|t| t != 1.0)) {
            // Preferred taint: the default's multiplicative semantics on the scalar cost.
            let scalar = |r: &Rank| -> f64 {
                let base = match (exact(r.decode_cost), exact(w)) {
                    (Some(d), Some(w)) => {
                        (count(r) as u128 * COUNT_SCALE + d + w * r.requests as u128) as f64
                    }
                    _ => {
                        count(r) as f64 * COUNT_SCALE as f64 + r.decode_cost + w * r.requests as f64
                    }
                };
                base * r.taint.unwrap_or(1.0)
            };
            argmin(
                fitting().map(|(row, r)| (row, scalar(r))),
                &mut decision.ties,
            );
        } else if let (Some(w), true) = (
            exact(w),
            fitting().all(|(_, r)| exact(r.decode_cost).is_some()),
        ) {
            let key = |r: &Rank| {
                (
                    count(r),
                    exact(r.decode_cost).unwrap() + w * r.requests as u128,
                )
            };
            argmin(fitting().map(|(row, r)| (row, key(r))), &mut decision.ties);
        } else {
            let key = |r: &Rank| (count(r) as f64, r.decode_cost + w * r.requests as f64);
            argmin(fitting().map(|(row, r)| (row, key(r))), &mut decision.ties);
        }
        Some(())
    }
}

struct KvCapacityAwarePicker {
    rule: Rule,
    capacity_blocks: Option<u64>,
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
    warned: HashSet<WorkerWithDpRank>,
    ranks: Vec<Rank>,
    unknown: Vec<usize>,
    decision: Decision,
}

impl KvCapacityAwarePicker {
    fn new(parameters: Parameters, rng: Option<Arc<Mutex<fastrand::Rng>>>) -> Self {
        Self {
            rule: parameters.rule(),
            capacity_blocks: parameters.capacity_blocks,
            rng,
            warned: HashSet::new(),
            ranks: Vec::new(),
            unknown: Vec::new(),
            decision: Decision::default(),
        }
    }

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
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        if load.len() != candidates.len() {
            return Err(WorkerSelectionPolicyError::failed("load input misaligned"));
        }
        let fallback_capacity = self.capacity_blocks;
        self.ranks.clear();
        self.ranks
            .extend(
                candidates
                    .iter()
                    .zip(load)
                    .enumerate()
                    .map(|(row, (candidate, load))| Rank {
                        decode_cost: load.decode_cost_blocks(),
                        requests: load.active_requests(),
                        capacity: input
                            .info(row)
                            .and_then(|info| info.total_kv_blocks())
                            .or(fallback_capacity),
                        taint: candidate.preferred_taint_multiplier(),
                    }),
            );
        self.rule
            .decide(&self.ranks, &mut self.decision, &mut self.unknown)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?;
        for &row in &self.unknown {
            let worker = candidates[row].worker();
            if self.warned.insert(worker) {
                tracing::warn!(
                    worker_id = worker.worker_id,
                    dp_rank = worker.dp_rank,
                    "{POLICY_TYPE}: no total_kv_blocks reported for this rank and no capacity_blocks \
                     set; it is not filtered"
                );
            }
        }
        let ties = &mut self.decision.ties;
        let Some(rng) = &self.rng else {
            return Ok(ties[fastrand::usize(0..ties.len())]);
        };
        // Canonical order only for deterministic replay, as the default picker does.
        ties.sort_unstable_by_key(|row| {
            let worker = candidates[*row].worker();
            (worker.worker_id, worker.dp_rank)
        });
        let index = rng.lock().usize(0..ties.len());
        Ok(ties[index])
    }
}

/// [`WorkerPicker::pick`] takes `&self`, so the per-rank warnings and the reused buffers sit behind
/// a lock.
struct SharedKvCapacityAwarePicker(Mutex<KvCapacityAwarePicker>);

impl WorkerPicker for SharedKvCapacityAwarePicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD | WorkerInputs::PREFERRED_TAINT
    }

    fn pick(
        &self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        self.0.lock().pick(context, input)
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;
    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(SharedKvCapacityAwarePicker(Mutex::new(
                    KvCapacityAwarePicker::new(parameters, None),
                ))),
            )
        },
    ))
}

pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod vectors;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{
        SchedulingRequest, WorkerLoadProjection, WorkerSelectionInput, WorkerSelector,
    };

    use super::vectors::VECTORS;
    use super::*;

    fn rule(g: f64, count_priority: bool, w: f64) -> Rule {
        Rule {
            capacity_fraction: g.is_finite().then_some(g),
            count_priority,
            decode_active_request_weight: w,
        }
    }

    fn ranks(d: &[u64], n: &[usize], cap: &[Option<u64>], taint: Option<&[f64]>) -> Vec<Rank> {
        (0..d.len())
            .map(|r| Rank {
                decode_cost: d[r] as f64,
                requests: n[r],
                capacity: cap[r],
                taint: taint.map(|t| t[r]),
            })
            .collect()
    }

    fn decide(rule: Rule, ranks: &[Rank]) -> (Decision, Vec<usize>) {
        let (mut decision, mut unknown) = (Decision::default(), Vec::new());
        rule.decide(ranks, &mut decision, &mut unknown)
            .expect("non-empty");
        decision.ties.sort_unstable();
        (decision, unknown)
    }

    /// The 37 boundary vectors (I1-I9, D1/D2): the exact tie set, the fallback flag and the rows to
    /// warn about all equal those of an independent reference implementation of the rule.
    #[test]
    fn boundary_vectors_match_the_reference_tie_sets() {
        for v in VECTORS.iter() {
            let r = rule(
                v.capacity_fraction,
                v.count_priority,
                v.decode_active_request_weight,
            );
            let (decision, unknown) =
                decide(r, &ranks(v.decode_cost, v.requests, v.capacity, v.taint));
            assert_eq!(decision.ties, v.expect_ties, "{} {}", v.inv, v.desc);
            assert_eq!(decision.fallback, v.expect_fallback, "{} {}", v.inv, v.desc);
            assert_eq!(unknown, v.expect_warning_for, "{} {}", v.inv, v.desc);
            assert!(
                v.booked
                    .iter()
                    .zip(v.decode_cost)
                    .all(|(a, d)| a + v.request_blocks == *d)
            );
        }
        assert_eq!(VECTORS.len(), 37);
    }

    /// I8: with no capacity filter and no count priority, the tie set is the default plain-decode
    /// cost `D + W * N`'s, on random states.
    #[test]
    fn degenerate_parameters_tie_exactly_like_the_default_cost() {
        let mut rng = fastrand::Rng::with_seed(96);
        for _ in 0..4000 {
            let n_ranks = rng.usize(1..40);
            let w = [0.0, 500.0, 1000.0, 2000.0][rng.usize(0..4)];
            let d: Vec<u64> = (0..n_ranks)
                .map(|_| rng.u64(0..4) * 1000 + rng.u64(0..3))
                .collect();
            let n: Vec<usize> = (0..n_ranks).map(|_| rng.usize(0..4)).collect();
            let cap = vec![Some(16_000); n_ranks];
            let (decision, _) = decide(rule(f64::INFINITY, false, w), &ranks(&d, &n, &cap, None));
            let cost: Vec<f64> = (0..n_ranks)
                .map(|r| d[r] as f64 + w * n[r] as f64)
                .collect();
            let min = cost.iter().cloned().fold(f64::INFINITY, f64::min);
            let expected: Vec<usize> = (0..n_ranks).filter(|&r| cost[r] == min).collect();
            assert_eq!(decision.ties, expected);
            assert!(!decision.fallback);
        }
    }

    /// I1 / I2 / I6 on random states: a non-empty list always decides; a fitting rank always wins
    /// over a non-fitting one; with none fitting the choice is among the least booked.
    #[test]
    fn never_fails_and_prefers_fitting_ranks() {
        let mut rng = fastrand::Rng::with_seed(7);
        for _ in 0..4000 {
            let n_ranks = rng.usize(1..33);
            let g = [0.95, 1.0][rng.usize(0..2)];
            let cap: Vec<Option<u64>> = (0..n_ranks)
                .map(|_| Some(rng.u64(10_000..20_000)))
                .collect();
            let d: Vec<u64> = (0..n_ranks).map(|_| rng.u64(0..25_000)).collect();
            let n: Vec<usize> = (0..n_ranks).map(|_| rng.usize(0..10)).collect();
            let rule = rule(g, true, 0.0);
            let rs = ranks(&d, &n, &cap, None);
            let (decision, _) = decide(rule, &rs);
            let fits: Vec<usize> = (0..n_ranks).filter(|&r| rule.fits(&rs[r])).collect();
            assert_eq!(decision.fallback, fits.is_empty());
            if fits.is_empty() {
                let min = *d.iter().min().unwrap();
                assert!(decision.ties.iter().all(|&r| d[r] == min));
            } else {
                assert!(decision.ties.iter().all(|r| fits.contains(r)));
                let key = |r: usize| (n[r], d[r]);
                let best = fits.iter().map(|&r| key(r)).min().unwrap();
                assert!(decision.ties.iter().all(|&r| key(r) == best));
            }
        }
    }

    /// I4: raising a rank's capacity never takes it out of `fits`.
    #[test]
    fn more_capacity_never_makes_a_rank_not_fit() {
        let r = rule(0.95, true, 0.0);
        for (d, cap) in [
            (15_200u64, 16_001u64),
            (15_201, 16_001),
            (17_000, 20_000),
            (20_000, 12_000),
        ] {
            let fits = |c| r.fits(&ranks(&[d], &[0], &[Some(c)], None)[0]);
            assert!(!fits(cap) || fits(cap + 1) && fits(cap * 2));
        }
        // 0.95 x 16,001 = 15,200.95: 15,200 fits, 15,201 does not (f64 product, compared with <=).
        assert!(r.fits(&ranks(&[15_200], &[0], &[Some(16_001)], None)[0]));
        assert!(!r.fits(&ranks(&[15_201], &[0], &[Some(16_001)], None)[0]));
    }

    /// I9: the same seeded generator reproduces the same picks; ties are drawn in (worker, rank) order.
    #[test]
    fn seeded_picks_repeat() {
        let picks = |seed| {
            let rng = Arc::new(Mutex::new(fastrand::Rng::with_seed(seed)));
            let policy = host_policy(KvCapacityAwarePicker::new(Parameters::default(), Some(rng)));
            (0..50)
                .map(|_| select(&policy, None, &[(0, 0, 0); 24]))
                .collect::<Vec<_>>()
        };
        assert_eq!(picks(11), picks(11));
        assert_ne!(picks(11), picks(12));
    }

    /// A worker advertising `total_kv_blocks` (None: not reported).
    struct TestWorker(Option<u64>);

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
            self.0
        }
    }

    /// The host's `WorkerSelectionPolicy` around `picker`, as `provider` builds it for decode.
    fn host_policy(picker: KvCapacityAwarePicker) -> WorkerSelectionPolicy {
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "decode",
            Vec::new(),
            Box::new(SharedKvCapacityAwarePicker(Mutex::new(picker))),
        )
    }

    /// Select through the host's policy among workers 1..=k given as
    /// `(booked decode blocks, this request's blocks, active requests)`, each advertising
    /// `total_kv_blocks = reported`; returns the worker id.
    fn select(
        policy: &WorkerSelectionPolicy,
        reported: Option<u64>,
        workers: &[(usize, usize, usize)],
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
            routing_constraints: RoutingConstraints::default(),
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
            .map(|id| (id, TestWorker(reported)))
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
        let pick = |workers: &[(usize, usize, usize)]| select(&policy, Some(1_000), workers);
        // Worker 1 has the fewest requests but 950 + 100 > 1,000; worker 2 fits with one more request.
        assert_eq!(pick(&[(950, 100, 0), (100, 100, 1), (100, 100, 2)]), 2);
        // Exactly at capacity fits (<=).
        assert_eq!(pick(&[(900, 100, 0), (100, 100, 1)]), 1);
        // Nothing fits: the least booked, whatever its request count.
        assert_eq!(
            pick(&[(1_500, 100, 0), (1_200, 100, 9), (1_300, 100, 1)]),
            2
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
        assert_eq!(select(&with(Some(100_000)), Some(1_000), &workers), 2);
        assert_eq!(select(&with(Some(1_000)), Some(100_000), &workers), 1);
        // Not reported, or reported as zero: the parameter applies.
        assert_eq!(select(&with(Some(1_000)), None, &workers), 2);
        assert_eq!(select(&with(Some(1_000)), Some(0), &workers), 2);
        // Neither: not filtered, so the fewest requests wins.
        assert_eq!(select(&with(None), None, &workers), 1);
    }

    /// A decode-only policy YAML together with the deprecated decode weight flag
    /// (`decode_active_request_weight = 1000`, the default policy's). Applied and validated in the
    /// order the Python `KvRouterConfig` constructor uses (`apply_policy_config`, then `validate`), it
    /// resolves; decode then selects with this policy (it skips the worker that does not fit, which
    /// the default cost `D + 1000 N` would pick), and prefill still constructs.
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
        assert_eq!(select(&decode, None, &workers), 2);
        let default_decode = crate::default_policy(config.clone(), "decode");
        assert_eq!(select(&default_decode, None, &workers), 1);
    }

    #[test]
    fn rejects_invalid_parameters() {
        let with = |f: fn(&mut Parameters)| {
            let mut p = Parameters::default();
            f(&mut p);
            p.validate()
        };
        assert!(with(|p| p.capacity_fraction = Some(0.0)).is_err());
        assert!(with(|p| p.capacity_fraction = Some(f64::NAN)).is_err());
        assert!(with(|p| p.capacity_blocks = Some(0)).is_err());
        assert!(with(|p| p.decode_active_request_weight = -1.0).is_err());
        assert!(with(|p| p.capacity_fraction = None).is_ok());
        assert!(Parameters::default().validate().is_ok());
    }
}
