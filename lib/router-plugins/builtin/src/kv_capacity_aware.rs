// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV-capacity-aware decode worker selection.
//!
//! The default decode cost `D + W * N` ignores a rank's KV capacity, so a rank can be booked past
//! what it holds while others have room. Here `D` is `decode_cost_blocks` (blocks booked on the
//! rank plus this request's), `N` is `active_requests`, and a rank's capacity is the
//! `total_kv_blocks` it advertises, else the `capacity_blocks` parameter.
//!
//! - A rank fits when `D <= capacity_fraction * capacity`. A rank without a capacity fits.
//! - Among fitting ranks: fewest requests first (`count_priority`), then the lowest `D + W * N`.
//!   With preferred taints, the multiplier scales `N * 2^24 + D + W * N`, as the default scales
//!   its cost.
//! - When no rank fits: the lowest `D`, ignoring taints.
//! - Ties are broken uniformly at random.
//!
//! The capacity may be an estimate, so the policy never queues or rejects a request. With
//! `capacity_fraction: null` and `count_priority: false` it ties exactly like the default
//! plain-decode cost.

use std::cell::Cell;
use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    ScoredWorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::protocols::WorkerWithDpRank;
use parking_lot::Mutex;
use rustc_hash::FxHashSet;

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-kv-capacity-aware";

/// Weight of the request count in the taint-scaled cost; `D + W * N` must stay below it.
const COUNT_SCALE: f64 = 16_777_216.0;

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
            && !(fraction.is_finite() && fraction > 0.0)
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
            (Some(capacity), Some(fraction)) => rank.decode_cost <= fraction * capacity as f64,
            _ => true,
        }
    }

    /// With `scaled`, the taint multiplies `N * 2^24 + D + W * N`; selection sets it when a
    /// fitting rank carries a multiplier other than 1.
    fn key(&self, rank: &Rank, scaled: bool) -> Key {
        if !self.fits(rank) {
            return Key::Over {
                decode_cost: rank.decode_cost,
            };
        }
        let count = if self.count_priority {
            rank.requests
        } else {
            0
        };
        let cost = rank.decode_cost + self.decode_active_request_weight * rank.requests as f64;
        if scaled {
            let cost = (count as f64 * COUNT_SCALE + cost) * rank.taint.unwrap_or(1.0);
            Key::Fits { count: 0, cost }
        } else {
            Key::Fits { count, cost }
        }
    }

    fn scaled(&self, mut ranks: impl Iterator<Item = Rank>) -> bool {
        ranks.any(|rank| rank.taint.is_some_and(|taint| taint != 1.0) && self.fits(&rank))
    }
}

/// The row to place the request on, or `None` without rows. Ties are drawn uniformly in the order
/// `rows` yields them; `draw(n)` returns an index below `n`.
fn select_row(
    parameters: &Parameters,
    rows: impl Iterator<Item = (usize, Rank)> + Clone,
    mut draw: impl FnMut(usize) -> usize,
) -> Option<usize> {
    let scaled = parameters.scaled(rows.clone().map(|(_, rank)| rank));
    let mut best: Option<(Key, usize)> = None;
    let mut ties = 0;
    for (row, rank) in rows {
        let key = parameters.key(&rank, scaled);
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

    fn warn_unknown(&self, candidates: &[ScoredWorkerCandidate], unknown: impl Fn(usize) -> bool) {
        let mut warned = self.warned.lock();
        for (row, candidate) in candidates.iter().enumerate() {
            let worker = candidate.worker();
            if unknown(row) && warned.insert(worker) {
                tracing::warn!(
                    policy = POLICY_TYPE,
                    worker_id = worker.worker_id,
                    dp_rank = worker.dp_rank,
                    "no total_kv_blocks reported and no capacity_blocks set; the rank is not filtered"
                );
            }
        }
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
        // Capacity matters only to the filter, so without one no rank is looked up.
        let filter = self.parameters.capacity_fraction.is_some();
        let unknown = Cell::new(false);
        let rank = |row: usize| {
            let capacity = if filter {
                let capacity = input
                    .info(row)
                    .and_then(|info| info.total_kv_blocks())
                    .or(self.parameters.capacity_blocks);
                unknown.set(unknown.get() || capacity.is_none());
                capacity
            } else {
                None
            };
            Rank {
                decode_cost: load[row].decode_cost_blocks(),
                requests: load[row].active_requests(),
                capacity,
                taint: candidates[row].preferred_taint_multiplier(),
            }
        };
        let row = match &self.rng {
            None => select_row(
                &self.parameters,
                (0..candidates.len()).map(|row| (row, rank(row))),
                |n| fastrand::usize(0..n),
            ),
            Some(rng) => {
                // Canonical order only for deterministic replay, as the default picker does.
                let mut rows: Vec<usize> = (0..candidates.len()).collect();
                rows.sort_unstable_by_key(|&row| {
                    let worker = candidates[row].worker();
                    (worker.worker_id, worker.dp_rank)
                });
                let mut rng = rng.lock();
                select_row(
                    &self.parameters,
                    rows.iter().map(|&row| (row, rank(row))),
                    |n| rng.usize(0..n),
                )
            }
        }
        .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?;
        if unknown.get() {
            self.warn_unknown(candidates, |row| rank(row).capacity.is_none());
        }
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
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(KvCapacityAwarePicker::new(parameters, None)),
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

    fn parameters(fraction: f64, count_priority: bool, weight: f64) -> Parameters {
        Parameters {
            capacity_blocks: None,
            capacity_fraction: fraction.is_finite().then_some(fraction),
            count_priority,
            decode_active_request_weight: weight,
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

    /// Every row at the minimum key, and whether no rank fits.
    fn tie_set(parameters: &Parameters, ranks: &[Rank]) -> (Vec<usize>, bool) {
        let scaled = parameters.scaled(ranks.iter().copied());
        let keys: Vec<Key> = ranks
            .iter()
            .map(|rank| parameters.key(rank, scaled))
            .collect();
        let min = keys
            .iter()
            .copied()
            .reduce(|a, b| if b < a { b } else { a })
            .expect("non-empty");
        let ties = (0..keys.len()).filter(|&row| keys[row] == min).collect();
        (ties, matches!(min, Key::Over { .. }))
    }

    /// The rows a filtering selection warns about: those without a capacity.
    fn unknown_rows(parameters: &Parameters, ranks: &[Rank]) -> Vec<usize> {
        (0..ranks.len())
            .filter(|&row| parameters.capacity_fraction.is_some() && ranks[row].capacity.is_none())
            .collect()
    }

    /// The table's tie sets, fallback flags and warned rows equal those of an independent
    /// reference implementation of the rule.
    #[test]
    fn boundary_vectors_match_the_reference_tie_sets() {
        for v in VECTORS.iter() {
            let p = parameters(
                v.capacity_fraction,
                v.count_priority,
                v.decode_active_request_weight,
            );
            let rs = ranks(v.decode_cost, v.requests, v.capacity, v.taint);
            let (ties, fallback) = tie_set(&p, &rs);
            assert_eq!(ties, v.expect_ties, "{} {}", v.inv, v.desc);
            assert_eq!(fallback, v.expect_fallback, "{} {}", v.inv, v.desc);
            assert_eq!(
                unknown_rows(&p, &rs),
                v.expect_warning_for,
                "{} {}",
                v.inv,
                v.desc
            );
            assert!(
                v.booked
                    .iter()
                    .zip(v.decode_cost)
                    .all(|(a, d)| a + v.request_blocks == *d)
            );
        }
        assert_eq!(VECTORS.len(), 37);
    }

    /// With no capacity filter and no count priority, the tie set is that of the default
    /// plain-decode cost `D + W * N`.
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
            let (ties, fallback) = tie_set(
                &parameters(f64::INFINITY, false, w),
                &ranks(&d, &n, &cap, None),
            );
            let cost: Vec<f64> = (0..n_ranks)
                .map(|r| d[r] as f64 + w * n[r] as f64)
                .collect();
            let min = cost.iter().cloned().fold(f64::INFINITY, f64::min);
            let expected: Vec<usize> = (0..n_ranks).filter(|&r| cost[r] == min).collect();
            assert_eq!(ties, expected);
            assert!(!fallback);
        }
    }

    /// On random states a non-empty list always yields a row from the tie set, a fitting rank
    /// beats every rank that does not fit, and with none fitting the least booked wins.
    #[test]
    fn never_fails_and_prefers_fitting_ranks() {
        let mut rng = fastrand::Rng::with_seed(7);
        for _ in 0..4000 {
            let n_ranks = rng.usize(1..33);
            let fraction = [0.95, 1.0][rng.usize(0..2)];
            let cap: Vec<Option<u64>> = (0..n_ranks)
                .map(|_| Some(rng.u64(10_000..20_000)))
                .collect();
            let d: Vec<u64> = (0..n_ranks).map(|_| rng.u64(0..25_000)).collect();
            let n: Vec<usize> = (0..n_ranks).map(|_| rng.usize(0..10)).collect();
            let p = parameters(fraction, true, 0.0);
            let rs = ranks(&d, &n, &cap, None);
            let (ties, fallback) = tie_set(&p, &rs);
            let fits: Vec<usize> = (0..n_ranks).filter(|&r| p.fits(&rs[r])).collect();
            assert_eq!(fallback, fits.is_empty());
            if fits.is_empty() {
                let min = *d.iter().min().unwrap();
                assert!(ties.iter().all(|&r| d[r] == min));
            } else {
                assert!(ties.iter().all(|r| fits.contains(r)));
                let key = |r: usize| (n[r], d[r]);
                let best = fits.iter().map(|&r| key(r)).min().unwrap();
                assert!(ties.iter().all(|&r| key(r) == best));
            }
            let picked = select_row(&p, rs.iter().copied().enumerate(), |k| rng.usize(0..k));
            assert!(ties.contains(&picked.expect("non-empty")));
        }
    }

    /// Raising a rank's capacity never takes it out of the fitting set.
    #[test]
    fn more_capacity_never_makes_a_rank_not_fit() {
        let p = parameters(0.95, true, 0.0);
        for (d, cap) in [
            (15_200u64, 16_001u64),
            (15_201, 16_001),
            (17_000, 20_000),
            (20_000, 12_000),
        ] {
            let fits = |c| p.fits(&ranks(&[d], &[0], &[Some(c)], None)[0]);
            assert!(!fits(cap) || fits(cap + 1) && fits(cap * 2));
        }
        // 0.95 x 16,001 = 15,200.95: 15,200 fits, 15,201 does not (f64 product, compared with <=).
        assert!(p.fits(&ranks(&[15_200], &[0], &[Some(16_001)], None)[0]));
        assert!(!p.fits(&ranks(&[15_201], &[0], &[Some(16_001)], None)[0]));
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
        assert!(with(|p| p.capacity_fraction = Some(f64::NAN)).is_err());
        assert!(with(|p| p.capacity_blocks = Some(0)).is_err());
        assert!(with(|p| p.decode_active_request_weight = -1.0).is_err());
        assert!(with(|p| p.capacity_fraction = None).is_ok());
        assert!(Parameters::default().validate().is_ok());
    }
}
