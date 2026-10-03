// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use dynamo_kv_router::WorkerType;
use dynamo_kv_router::plugins::worker_selection::WorkerSelectionContext;
use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::scheduling::SchedulingRequest;
use parking_lot::Mutex;

use super::features::*;
use super::*;
use crate::choice::TieBreak;
use crate::test_support::{
    BLOCK_SIZE, Worker, in_session, populate_replay, request, resolve_policy, select_populated,
    select_replay,
};

const V1_DIM: usize = 8;
const V2_DIM: usize = 23;

/// What one decision exposed, by host row.
#[derive(Clone, Default)]
struct Snapshot {
    workers: Vec<u64>,
    dim: usize,
    features: Vec<f64>,
    utilities: Vec<f64>,
    sources: Vec<f64>,
}

impl Snapshot {
    fn row(&self, worker: u64) -> &[f64] {
        let row = self.workers.iter().position(|&id| id == worker).unwrap();
        &self.features[row * self.dim..(row + 1) * self.dim]
    }

    fn rows(&self) -> impl Iterator<Item = &[f64]> {
        self.features.chunks(self.dim)
    }
}

const ALL_SOURCES: [ContextSource; 6] = [
    ContextSource::MeanKvLoadFrac,
    ContextSource::MeanActivePrefillTokensK,
    ContextSource::MeanActiveRequestsS,
    ContextSource::MaxOverlapFrac,
    ContextSource::IslK,
    ContextSource::IsFirstTurn,
];

/// Records the picker's feature table and utilities after every decision.
struct Probe {
    inner: LearnedChoicePicker,
    sink: Arc<Mutex<Snapshot>>,
}

impl WorkerPicker for Probe {
    fn required_worker_inputs(&self) -> WorkerInputs {
        self.inner.required_worker_inputs()
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let row = self.inner.pick(context, input)?;
        if context.pinned_worker().is_some() {
            // Pinned decisions compute no features.
            return Ok(row);
        }
        let order = self.inner.chooser.order();
        *self.sink.lock() = Snapshot {
            workers: input
                .candidates()
                .iter()
                .map(|candidate| candidate.worker().worker_id)
                .collect(),
            dim: self.inner.model.feature_set.dim(),
            features: self.inner.table.values.clone(),
            utilities: self.inner.utilities.clone(),
            sources: ALL_SOURCES
                .iter()
                .map(|&source| self.inner.table.source(source, order))
                .collect(),
        };
        Ok(row)
    }
}

fn model(feature_set: FeatureSet, theta: Vec<f64>) -> Model {
    Model {
        feature_set,
        theta,
        context: ContextTerm::None,
        temperature: 0.0,
        seed: Some(7),
        tie_break: TieBreak::OneDraw,
        max_sessions: 64,
    }
}

fn unit(dim: usize, index: usize, value: f64) -> Vec<f64> {
    let mut theta = vec![0.0; dim];
    theta[index] = value;
    theta
}

fn probe(model: Model) -> (WorkerSelectionPolicy, Arc<Mutex<Snapshot>>) {
    let config = KvRouterConfig::default();
    let sink = Arc::new(Mutex::new(Snapshot::default()));
    let scorer = crate::default::cost_scorer(&config, &config, WorkerType::Aggregated);
    let policy = WorkerSelectionPolicy::new(
        config,
        "decode",
        vec![scorer],
        Box::new(Probe {
            inner: LearnedChoicePicker::new(model),
            sink: sink.clone(),
        }),
    );
    (policy, sink)
}

/// Every candidate's (worker ID, cost) from the last decision.
type CostLog = Arc<Mutex<Vec<(u64, f64)>>>;

/// The default scorer at default weights with a picker that records every row's cost.
fn default_costs() -> (WorkerSelectionPolicy, CostLog) {
    struct Recorder(CostLog);
    impl WorkerPicker for Recorder {
        fn pick(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            let candidates = input.candidates();
            *self.0.lock() = candidates
                .iter()
                .map(|candidate| (candidate.worker().worker_id, candidate.cost()))
                .collect();
            Ok(0)
        }
    }
    let config = KvRouterConfig::default();
    let costs = Arc::new(Mutex::new(Vec::new()));
    let scorer = crate::default::cost_scorer(&config, &config, WorkerType::Aggregated);
    let policy = WorkerSelectionPolicy::new(
        config,
        "decode",
        vec![scorer],
        Box::new(Recorder(costs.clone())),
    );
    (policy, costs)
}

fn unique_argmin(costs: &[(u64, f64)]) -> Option<u64> {
    let minimum = costs
        .iter()
        .map(|(_, cost)| *cost)
        .fold(f64::INFINITY, f64::min);
    let mut lowest = costs.iter().filter(|(_, cost)| *cost == minimum);
    let (worker, _) = lowest.next()?;
    lowest.next().is_none().then_some(*worker)
}

fn theta0_yaml(feature_set: &str, dim: usize, extra: &str) -> String {
    let mut theta = vec!["0".to_owned(); dim];
    theta[0] = "-1".to_owned();
    format!(
        "{{feature_set: {feature_set}, theta: [{}]{extra}}}",
        theta.join(", ")
    )
}

/// A random table: up to 12 workers with random cache and load state, plus a prompt that may end
/// in a partial block.
fn random_table(rng: &mut fastrand::Rng) -> (SchedulingRequest, Vec<Worker>) {
    let prompt_blocks = rng.usize(1..=256);
    let mut request = request(prompt_blocks, rng.u64(..1_000));
    request.isl_tokens -= rng.usize(0..BLOCK_SIZE as usize);
    let workers = (0..rng.u64(2..=12))
        .map(|id| Worker {
            id: id * 7 + 3,
            device_blocks: rng.usize(0..=prompt_blocks.saturating_sub(1)),
            active_requests: rng.usize(0..64),
            active_prefill_tokens: rng.usize(0..20_000),
            decode_blocks: rng.usize(0..15_000),
            total_kv_blocks: Some(18_863),
            modeled_backlog_ms: None,
        })
        .collect();
    (request, workers)
}

/// The four host input shapes the default scorer distinguishes.
fn shape_request(
    request: &mut SchedulingRequest,
    workers: &[Worker],
    shape: usize,
) -> std::collections::HashMap<u64, crate::test_support::TestWorker> {
    let configs = populate_replay(request, workers);
    match shape % 4 {
        1 => request.overlap.tier_overlap_blocks = Default::default(),
        2 => request.worker_loads.clear(),
        3 => request.track_prefill_tokens = false,
        _ => {}
    }
    configs
}

#[test]
fn v1_features_follow_their_definitions() {
    let (policy, sink) = probe(model(FeatureSet::V1, unit(V1_DIM, 0, -1.0)));
    // Ten whole blocks: 160 prompt tokens.
    let workers = [
        Worker::new(0)
            .cached(4)
            .prefill(32)
            .requests(3)
            .decode(100, 1_000),
        Worker {
            decode_blocks: 655,
            ..Worker::new(1)
        },
    ];
    let first = select_replay(&policy, in_session(request(10, 1), "s"), &workers);
    let snapshot = sink.lock().clone();
    let isl_k = 160.0 / 8192.0;

    // Worker 0's default logit: (32 + 96 + 64) / 16 − 4 credited blocks + 100 decode blocks.
    let x = snapshot.row(0);
    assert_eq!(x[DEFAULT_LOGIT_SCALED], 108.0 / 10.0);
    assert_eq!(x[OVERLAP_FRAC], 4.0 / 10.0);
    assert_eq!(x[NEW_PREFILL_TOKENS_K], 96.0 / 8192.0);
    assert_eq!(x[ACTIVE_PREFILL_TOKENS_K], 32.0 / 8192.0);
    assert_eq!(x[KV_LOAD_FRAC], 100.0 / 1_000.0);
    assert_eq!(x[ACTIVE_REQUESTS_S], 3.0 / 32.0);
    assert_eq!(x[SESSION_AFFINITY], 0.0);
    assert_eq!(x[ISL_X_PREFILL_LOAD], isl_k * (32.0 / 8192.0));

    // Worker 1 advertises no capacity, so its KV load uses the 65,536-block fallback.
    let x = snapshot.row(1);
    assert_eq!(x[DEFAULT_LOGIT_SCALED], (10.0 + 655.0) / 10.0);
    assert_eq!(x[OVERLAP_FRAC], 0.0);
    assert_eq!(x[NEW_PREFILL_TOKENS_K], 160.0 / 8192.0);
    assert_eq!(x[KV_LOAD_FRAC], 655.0 / FALLBACK_KV_BLOCKS as f64);
    assert_eq!(x[ISL_X_PREFILL_LOAD], 0.0);
    assert_eq!(first, 0, "θ = −e₀ takes the lower default logit");
    assert_eq!(snapshot.sources[5], 1.0, "a session's first request");

    // The session's next request marks the worker that served it, and only that worker.
    select_replay(&policy, in_session(request(10, 1), "s"), &workers);
    let snapshot = sink.lock().clone();
    assert_eq!(snapshot.row(0)[SESSION_AFFINITY], 1.0);
    assert_eq!(snapshot.row(1)[SESSION_AFFINITY], 0.0);
    assert_eq!(snapshot.sources[5], 0.0, "a session's later request");

    // Another session, and a request without one, see no affinity.
    for request in [in_session(request(10, 1), "t"), request(10, 1)] {
        select_replay(&policy, request, &workers);
        let snapshot = sink.lock().clone();
        assert!(snapshot.rows().all(|x| x[SESSION_AFFINITY] == 0.0));
    }
}

#[test]
fn v2_features_follow_their_definitions() {
    let (policy, sink) = probe(model(FeatureSet::V2, unit(V2_DIM, 0, -1.0)));
    let workers = [
        Worker::new(0)
            .cached(4)
            .prefill(1_000)
            .requests(3)
            .decode(100, 1_000),
        Worker::new(1).prefill(3_000).requests(1).decode(300, 1_000),
        Worker::new(2).cached(9).requests(0).decode(0, 1_000),
        Worker::new(3).prefill(1_000).requests(8).decode(500, 1_000),
    ];
    select_replay(&policy, request(10, 1), &workers);
    let snapshot = sink.lock().clone();

    let x = snapshot.row(0);
    assert_eq!(x[LOG_PTOK], ((1_000.0 + 96.0) / 1024.0_f64).ln());
    assert_eq!(x[LOG_NEW_PREFILL], (96.0 / 1024.0_f64).ln());
    assert_eq!(x[LOG_BS], 4.0_f64.ln());
    assert_eq!(x[NEW_PREFILL_X_REQUESTS], (96.0 / 8192.0) * (3.0 / 32.0));
    assert_eq!(x[PREFILL_ATTN], 96.0 * (160.0 - 48.0) / (8192.0 * 8192.0));
    // Worker 2 holds 9 of 10 blocks: 16 new tokens, and log_new_prefill floors at one token.
    let x = snapshot.row(2);
    assert_eq!(x[LOG_NEW_PREFILL], (16.0 / 1024.0_f64).ln());
    assert_eq!(x[LOG_BS], 0.0);

    // Set-relative transforms of kv_load_frac (loads 0.1, 0.3, 0, 0.5; mean 0.225).
    let mean = (0.1 + 0.3 + 0.0 + 0.5) / 4.0;
    for (worker, load, below) in [(0, 0.1, 1.0), (1, 0.3, 2.0), (2, 0.0, 0.0), (3, 0.5, 3.0)] {
        let x = snapshot.row(worker);
        assert!(
            (x[12] - load / (0.01 + mean)).abs() < 1e-12,
            "kv_load_ratio {worker}"
        );
        assert_eq!(x[15], below / 4.0, "kv_load_below_frac {worker}");
        assert!(
            (x[18] - (load - mean).max(0.0)).abs() < 1e-12,
            "kv_load_excess {worker}"
        );
    }
    // The strict-below fraction counts ties as not below: workers 0 and 3 share a prefill load.
    assert_eq!(snapshot.row(0)[16], 1.0 / 4.0);
    assert_eq!(snapshot.row(3)[16], 1.0 / 4.0);
    assert_eq!(snapshot.row(2)[16], 0.0);
    assert_eq!(snapshot.row(1)[16], 3.0 / 4.0);

    // The prompt's first 256 tokens key exactly two of four workers as home.
    let homes: Vec<u64> = (0..4)
        .filter(|&id| snapshot.row(id)[HASH_HOME] == 1.0)
        .collect();
    assert_eq!(homes.len(), 2);
    // The home set depends on the key and the workers, not on load or row order.
    let reordered = [workers[3], workers[1], Worker::new(2), workers[0]];
    select_replay(&policy, request(10, 1), &reordered);
    let again: Vec<u64> = (0..4)
        .filter(|&id| sink.lock().row(id)[HASH_HOME] == 1.0)
        .collect();
    assert_eq!(homes, again);
    // With two candidates both are home.
    select_replay(&policy, request(10, 1), &workers[..2]);
    assert!(sink.lock().rows().all(|x| x[HASH_HOME] == 1.0));
    // The key is the prompt prefix only: a session ID, which closed-loop replay synthesizes per
    // request, never moves the home (build audit F1).
    for session in ["s", "request_17"] {
        select_replay(&policy, in_session(request(10, 1), session), &workers);
        let with_session: Vec<u64> = (0..4)
            .filter(|&id| sink.lock().row(id)[HASH_HOME] == 1.0)
            .collect();
        assert_eq!(with_session, homes, "session {session}");
    }
    // Without prefix hashes there is no key, so no worker is home.
    let mut unhashed = in_session(request(10, 1), "s");
    unhashed.token_seq = None;
    select_replay(&policy, unhashed, &workers);
    assert!(sink.lock().rows().all(|x| x[HASH_HOME] == 0.0));
}

/// Duplicating every worker (N → 2N identical copies) leaves each worker's features, every
/// context source, and the argmax class unchanged, apart from `hash_home`, which keys on worker
/// identity. Linear set-relative terms, z-scores, sums, and counts would fail this.
#[test]
fn duplicating_the_worker_set_keeps_features_sources_and_choice() {
    let mut rng = fastrand::Rng::with_seed(11);
    for _ in 0..50 {
        let (_, base) = random_table(&mut rng);
        let prompt_blocks = rng.usize(16..=256);
        let base: Vec<Worker> = base
            .into_iter()
            .map(|worker| Worker {
                device_blocks: worker.device_blocks.min(prompt_blocks - 1),
                ..worker
            })
            .collect();
        let doubled: Vec<Worker> = base
            .iter()
            .flat_map(|worker| {
                [
                    *worker,
                    Worker {
                        id: worker.id + 1_000,
                        ..*worker
                    },
                ]
            })
            .collect();
        let mut theta: Vec<f64> = (0..V2_DIM).map(|_| rng.f64() * 2.0 - 1.0).collect();
        theta[HASH_HOME] = 0.0;
        let decide = |workers: &[Worker]| {
            let (policy, sink) = probe(model(FeatureSet::V2, theta.clone()));
            let chosen = select_replay(&policy, request(prompt_blocks, 4), workers);
            (chosen, sink.lock().clone())
        };
        let (chosen, small) = decide(&base);
        let (chosen_doubled, large) = decide(&doubled);
        assert!(chosen_doubled == chosen || chosen_doubled == chosen + 1_000);
        for (left, right) in small.sources.iter().zip(&large.sources) {
            assert!((left - right).abs() <= 1e-12 * left.abs().max(1.0));
        }
        for worker in &base {
            for copy in [worker.id, worker.id + 1_000] {
                for (index, (left, right)) in
                    small.row(worker.id).iter().zip(large.row(copy)).enumerate()
                {
                    if index == HASH_HOME {
                        continue;
                    }
                    assert!(
                        (left - right).abs() <= 1e-12 * left.abs().max(1.0),
                        "feature {index}: {left} vs {right}"
                    );
                }
            }
        }
    }
}

fn mean_row(snapshot: &Snapshot) -> Vec<f64> {
    let rows = snapshot.workers.len() as f64;
    (0..snapshot.dim)
        .map(|index| snapshot.rows().map(|x| x[index]).sum::<f64>() / rows)
        .collect()
}

/// A random table regenerated from `seed`, since requests are not `Clone`.
fn seeded_table(seed: u64) -> (SchedulingRequest, Vec<Worker>) {
    random_table(&mut fastrand::Rng::with_seed(seed))
}

#[test]
fn context_term_matches_the_bilinear_formula() {
    let mut rng = fastrand::Rng::with_seed(5);
    let mut random = |dim: usize| -> Vec<f64> { (0..dim).map(|_| rng.f64() * 2.0 - 1.0).collect() };
    for trial in 0..100 {
        let theta = random(V1_DIM);
        let p = vec![random(V1_DIM), random(V1_DIM)];
        let q = vec![random(V1_DIM), random(V1_DIM)];
        let pooled = ContextTerm::Pooled {
            p: p.clone(),
            q: q.clone(),
        };
        let named = ContextTerm::Sources {
            sources: vec![ContextSource::IslK, ContextSource::MeanKvLoadFrac],
            p: p.clone(),
        };
        for context in [pooled, named] {
            let (policy, sink) = probe(Model {
                context: context.clone(),
                ..model(FeatureSet::V1, theta.clone())
            });
            let (request, workers) = seeded_table(trial);
            select_replay(&policy, request, &workers);
            let snapshot = sink.lock().clone();
            let mean = mean_row(&snapshot);
            let coefficients: Vec<f64> = match &context {
                // u_i = θ·x_i + Σ_k (p_k·x_i)(q_k·x̄_S)
                ContextTerm::Pooled { q, .. } => q.iter().map(|q_k| dot(q_k, &mean)).collect(),
                // u_i = θ·x_i + isl_k (p_0·x_i) + mean_S kv_load_frac (p_1·x_i)
                ContextTerm::Sources { .. } => vec![snapshot.sources[4], mean[KV_LOAD_FRAC]],
                ContextTerm::None => unreachable!(),
            };
            for (row, x) in snapshot.rows().enumerate() {
                let expected = dot(&theta, x)
                    + p.iter()
                        .zip(&coefficients)
                        .map(|(p_k, c_k)| dot(p_k, x) * c_k)
                        .sum::<f64>();
                let actual = snapshot.utilities[row];
                assert!(
                    (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                    "trial {trial} row {row}: {actual} vs {expected}"
                );
            }
        }
    }
}

/// The parity anchor: θ = −e₀, rank 0, temperature 0 picks the default cost function's argmin on
/// randomized tie-free tables, across the host input shapes the default scorer distinguishes.
#[test]
fn theta0_picks_the_default_argmin() {
    let (recorder, costs) = default_costs();
    let default = resolve_policy("dynamo-default-cost-fn", "{seed: 1}").unwrap();
    let learned = [
        resolve_policy(POLICY_TYPE, &theta0_yaml("v1", V1_DIM, ", seed: 1")).unwrap(),
        resolve_policy(POLICY_TYPE, &theta0_yaml("v2", V2_DIM, ", seed: 1")).unwrap(),
    ];
    let mut tie_free = 0;
    for trial in 0..2_000 {
        let (mut request, workers) = seeded_table(trial);
        let configs = shape_request(&mut request, &workers, trial as usize);
        select_populated(&recorder, &request, &configs);
        let Some(expected) = unique_argmin(&costs.lock()) else {
            continue;
        };
        tie_free += 1;
        assert_eq!(select_populated(&default, &request, &configs), expected);
        for policy in &learned {
            assert_eq!(
                select_populated(policy, &request, &configs),
                expected,
                "trial {trial}"
            );
        }
    }
    assert!(tie_free >= 1_500, "only {tie_free} tie-free tables");
}

/// With `tie_break: reservoir`, θ₀ reproduces the seeded default draw for draw, ties included.
#[test]
fn reservoir_tie_break_reproduces_the_seeded_default() {
    let default = resolve_policy("dynamo-default-cost-fn", "{seed: 9}").unwrap();
    let learned = resolve_policy(
        POLICY_TYPE,
        &theta0_yaml("v1", V1_DIM, ", seed: 9, tie_break: reservoir"),
    )
    .unwrap();
    let (recorder, costs) = default_costs();
    let mut rng = fastrand::Rng::with_seed(3);
    let mut tied = 0;
    for _ in 0..500 {
        // Few distinct states, so most tables tie.
        let workers: Vec<Worker> = (0..rng.u64(2..=8))
            .map(|id| {
                Worker::new(id)
                    .requests(rng.usize(0..2))
                    .prefill(16 * rng.usize(0..2))
            })
            .collect();
        let mut request = request(4, 1);
        let configs = populate_replay(&mut request, &workers);
        select_populated(&recorder, &request, &configs);
        tied += usize::from(unique_argmin(&costs.lock()).is_none());
        assert_eq!(
            select_populated(&learned, &request, &configs),
            select_populated(&default, &request, &configs)
        );
    }
    assert!(tied > 250, "only {tied} tied tables");
}

#[test]
fn fixed_seed_reproduces_sampled_decisions() {
    let run = |seed: u64| -> Vec<u64> {
        let policy = resolve_policy(
            POLICY_TYPE,
            &theta0_yaml("v2", V2_DIM, &format!(", temperature: 0.7, seed: {seed}")),
        )
        .unwrap();
        (0..300)
            .map(|trial| {
                let (request, workers) = seeded_table(trial);
                select_replay(&policy, request, &workers)
            })
            .collect()
    };
    assert_eq!(run(5), run(5));
    assert_ne!(run(5), run(6));
}

/// LR-03: every decision consumes exactly one draw, so two policies that differ in one early
/// decision still break every later tie identically. The reservoir discipline does not.
#[test]
fn one_draw_per_decision_keeps_later_draws_aligned() {
    let picks = |theta: Vec<f64>, tie_break: TieBreak| -> Vec<u64> {
        let (policy, _) = probe(Model {
            tie_break,
            ..model(FeatureSet::V1, theta)
        });
        // First decision: equal default logits, distinct request counts. θ = −e₀ sees a
        // three-way tie; adding a request penalty sees a unique best.
        let first = [
            Worker::new(0).requests(2),
            Worker::new(1).requests(1),
            Worker::new(2).requests(3),
        ];
        let mut out = vec![select_replay(&policy, request(4, 1), &first)];
        // Later decisions: identical workers, a full tie for both policies.
        let idle = [
            Worker::new(0),
            Worker::new(1),
            Worker::new(2),
            Worker::new(3),
        ];
        out.extend((0..40).map(|_| select_replay(&policy, request(4, 1), &idle)));
        out
    };
    let mut penalized = unit(V1_DIM, 0, -1.0);
    penalized[ACTIVE_REQUESTS_S] = -10.0;
    for (tie_break, aligned) in [(TieBreak::OneDraw, true), (TieBreak::Reservoir, false)] {
        let plain = picks(unit(V1_DIM, 0, -1.0), tie_break);
        let other = picks(penalized.clone(), tie_break);
        assert_eq!(other[0], 1);
        assert_eq!(plain[1..] == other[1..], aligned, "{tie_break:?}");
    }
}

#[test]
fn temperature_samples_follow_the_softmax() {
    // u = −4 · active_requests_s: 0, −1, −2 at 0, 8, 16 requests; at τ = 0.5, p ∝ 1, e⁻², e⁻⁴.
    let (policy, _) = probe(Model {
        temperature: 0.5,
        ..model(FeatureSet::V1, unit(V1_DIM, ACTIVE_REQUESTS_S, -4.0))
    });
    let workers = [
        Worker::new(0),
        Worker::new(1).requests(8),
        Worker::new(2).requests(16),
    ];
    let draws = 20_000;
    let mut counts = [0usize; 3];
    for _ in 0..draws {
        counts[select_replay(&policy, request(4, 1), &workers) as usize] += 1;
    }
    let weights = [1.0, (-2.0_f64).exp(), (-4.0_f64).exp()];
    let total: f64 = weights.iter().sum();
    for (count, weight) in counts.iter().zip(weights) {
        let observed = *count as f64 / draws as f64;
        assert!((observed - weight / total).abs() < 0.01, "{counts:?}");
    }
}

#[test]
fn sessions_beyond_max_sessions_are_forgotten() {
    let (policy, sink) = probe(Model {
        max_sessions: 1,
        ..model(FeatureSet::V1, unit(V1_DIM, 0, -1.0))
    });
    let workers = [Worker::new(0), Worker::new(1).requests(1)];
    select_replay(&policy, in_session(request(4, 1), "a"), &workers);
    select_replay(&policy, in_session(request(4, 1), "b"), &workers);
    select_replay(&policy, in_session(request(4, 1), "a"), &workers);
    assert!(sink.lock().rows().all(|x| x[SESSION_AFFINITY] == 0.0));
    assert_eq!(sink.lock().sources[5], 1.0);
}

#[test]
fn a_host_pin_wins_and_rebinds_the_session() {
    let (policy, sink) = probe(model(FeatureSet::V1, unit(V1_DIM, 0, -1.0)));
    let workers = [Worker::new(0), Worker::new(1).prefill(8_000)];
    let mut pinned = in_session(request(4, 1), "s");
    pinned.pinned_worker = Some(WorkerWithDpRank::from_worker_id(1));
    assert_eq!(select_replay(&policy, pinned, &workers), 1);
    // The unpinned follow-up still prefers worker 0, but sees worker 1 as the session's last.
    assert_eq!(
        select_replay(&policy, in_session(request(4, 1), "s"), &workers),
        0
    );
    assert_eq!(sink.lock().row(1)[SESSION_AFFINITY], 1.0);
}

#[test]
fn eligibility_restricts_the_choice() {
    let (policy, _) = probe(model(FeatureSet::V1, unit(V1_DIM, 0, -1.0)));
    let workers = [
        Worker::new(0),
        Worker::new(1).prefill(8_000),
        Worker::new(2).prefill(9_000),
    ];
    let mut restricted = request(4, 1);
    restricted.allowed_worker_ids = Some(HashSet::from([1, 2]));
    assert_eq!(select_replay(&policy, restricted, &workers), 1);
}

#[test]
fn parameter_validation_is_strict() {
    let theta = |dim: usize| format!("[{}]", vec!["0"; dim].join(", "));
    let v1 = theta(V1_DIM);
    let short = theta(V1_DIM - 1);
    let invalid = [
        (
            format!("{{feature_set: v1, theta: {short}}}"),
            "theta has 7 entries; feature_set v1 needs 8",
        ),
        (
            format!("{{feature_set: v2, theta: {v1}}}"),
            "theta has 8 entries; feature_set v2 needs 23",
        ),
        (
            format!("{{feature_set: v3, theta: {v1}}}"),
            "unknown variant `v3`",
        ),
        (format!("{{theta: {v1}}}"), "missing field `feature_set`"),
        ("{feature_set: v1}".to_owned(), "missing field `theta`"),
        (
            format!("{{feature_set: v1, theta: {v1}, bogus: 1}}"),
            "unknown field `bogus`",
        ),
        (
            "{feature_set: v1, theta: [.nan, 0, 0, 0, 0, 0, 0, 0]}".to_owned(),
            "theta[0] must be finite",
        ),
        (
            format!("{{feature_set: v1, theta: {v1}, temperature: -0.1}}"),
            "temperature must be finite and non-negative",
        ),
        (
            format!("{{feature_set: v1, theta: {v1}, temperature: .inf}}"),
            "temperature must be finite and non-negative",
        ),
        (
            format!("{{feature_set: v1, theta: {v1}, max_sessions: 0}}"),
            "max_sessions must be positive",
        ),
        (
            format!("{{feature_set: v1, theta: {v1}, seed: -1}}"),
            "expected u64",
        ),
        (
            format!("{{feature_set: v1, theta: {v1}, tie_break: coin}}"),
            "unknown variant `coin`",
        ),
        (
            format!("{{feature_set: v1, theta: {v1}, context: {{p: [{v1}]}}}}"),
            "context.p needs context.q or context.sources",
        ),
        (
            format!("{{feature_set: v1, theta: {v1}, context: {{p: [{v1}], q: []}}}}"),
            "context.q has 0 rows but context.p has 1",
        ),
        (
            format!("{{feature_set: v1, theta: {v1}, context: {{p: [{short}], q: [{v1}]}}}}"),
            "context.p[0] has 7 entries",
        ),
        (
            format!("{{feature_set: v1, theta: {v1}, context: {{p: [{v1}], q: [{short}]}}}}"),
            "context.q[0] has 7 entries",
        ),
        (
            format!(
                "{{feature_set: v1, theta: {v1}, context: {{p: [{v1}], q: [{v1}], sources: [isl_k]}}}}"
            ),
            "context takes q or sources, not both",
        ),
        (
            format!("{{feature_set: v1, theta: {v1}, context: {{p: [{v1}], sources: [nope]}}}}"),
            "unknown context source `nope`; expected one of mean_kv_load_frac",
        ),
        (
            format!(
                "{{feature_set: v1, theta: {v1}, context: {{p: [{v1}, {v1}], sources: [isl_k, isl_k]}}}}"
            ),
            "context source `isl_k` is repeated",
        ),
        (
            format!("{{feature_set: v1, theta: {v1}, context: {{p: [{v1}], sources: []}}}}"),
            "context.sources has 0 entries but context.p has 1 rows",
        ),
        (
            format!("{{feature_set: v1, theta: {v1}, context: {{p: [], r: 1}}}}"),
            "unknown field `r`",
        ),
    ];
    for (parameters, expected) in invalid {
        let Err(error) = resolve_policy(POLICY_TYPE, &parameters) else {
            panic!("{parameters} must fail");
        };
        assert!(error.contains(expected), "{parameters}: {error}");
        assert!(error.contains(POLICY_TYPE), "{parameters}: {error}");
    }

    let v2 = theta(V2_DIM);
    for valid in [
        format!("{{feature_set: v1, theta: {v1}}}"),
        format!("{{feature_set: v1, theta: {v1}, context: {{}}}}"),
        format!("{{feature_set: v1, theta: {v1}, context: {{p: [], q: []}}}}"),
        format!("{{feature_set: v2, theta: {v2}, context: {{p: [{v2}, {v2}], q: [{v2}, {v2}]}}}}"),
        format!(
            "{{feature_set: v2, theta: {v2}, context: {{sources: [is_first_turn, max_overlap_frac], p: [{v2}, {v2}]}}, temperature: 0.5, seed: 3, tie_break: reservoir, max_sessions: 10}}"
        ),
    ] {
        resolve_policy(POLICY_TYPE, &valid).unwrap_or_else(|error| panic!("{valid}: {error}"));
    }
}

/// The scorer composed for feature 0 keeps default weights even when the host configures others.
#[test]
fn feature_zero_ignores_the_hosts_score_weights() {
    let tuned = KvRouterConfig {
        overlap_score_credit: 3.0,
        prefill_load_scale: 0.25,
        ..Default::default()
    };
    let scorer_inputs = |config: &KvRouterConfig| {
        crate::default::cost_scorer(config, &KvRouterConfig::default(), WorkerType::Aggregated)
            .required_worker_inputs()
    };
    assert_eq!(
        scorer_inputs(&tuned),
        scorer_inputs(&KvRouterConfig::default())
    );
    let learned = policy(
        &tuned,
        WorkerType::Aggregated,
        model(FeatureSet::V1, unit(V1_DIM, 0, -1.0)),
    );
    let (default_at_defaults, costs) = default_costs();
    let mut rng = fastrand::Rng::with_seed(17);
    for _ in 0..200 {
        let (mut request, workers) = random_table(&mut rng);
        let configs = populate_replay(&mut request, &workers);
        select_populated(&default_at_defaults, &request, &configs);
        let Some(expected) = unique_argmin(&costs.lock()) else {
            continue;
        };
        assert_eq!(select_populated(&learned, &request, &configs), expected);
    }
}

/// LR-07: v2 holds LMetric's product rules exactly. θ = −(e_log_new_prefill + e_log_bs) takes the
/// `lmetric` port's choice (without its hot-spot filter), and θ = −(e_log_ptok + e_log_bs) the
/// argmin of LMetric's queued-prefill product.
#[test]
fn v2_represents_the_lmetric_products() {
    let port = resolve_policy("lmetric", "{hotspot_detection: false}").unwrap();
    let mut port_theta = vec![0.0; V2_DIM];
    port_theta[LOG_NEW_PREFILL] = -1.0;
    port_theta[LOG_BS] = -1.0;
    let mut queued_theta = vec![0.0; V2_DIM];
    queued_theta[LOG_PTOK] = -1.0;
    queued_theta[LOG_BS] = -1.0;
    let (port_like, _) = probe(model(FeatureSet::V2, port_theta));
    let (queued, _) = probe(model(FeatureSet::V2, queued_theta));
    let mut checked = 0;
    for trial in 0..1_000 {
        let (mut request, workers) = seeded_table(trial);
        let configs = populate_replay(&mut request, &workers);
        // The two products, used here only to skip tables whose minimum is tied.
        let product = |queued_prefill: bool| -> Vec<(u64, f64)> {
            workers
                .iter()
                .map(|worker| {
                    let new_prefill = request
                        .isl_tokens
                        .saturating_sub(worker.device_blocks * BLOCK_SIZE as usize);
                    let prefill = if queued_prefill {
                        new_prefill + worker.active_prefill_tokens
                    } else {
                        new_prefill
                    };
                    let batch = worker.active_requests + 1;
                    (worker.id, (prefill.max(1) * batch) as f64)
                })
                .collect()
        };
        if let Some(expected) = unique_argmin(&product(false)) {
            assert_eq!(select_populated(&port, &request, &configs), expected);
            assert_eq!(select_populated(&port_like, &request, &configs), expected);
            checked += 1;
        }
        if let Some(expected) = unique_argmin(&product(true)) {
            assert_eq!(select_populated(&queued, &request, &configs), expected);
            checked += 1;
        }
    }
    assert!(checked > 1_500, "only {checked} tie-free tables");
}
