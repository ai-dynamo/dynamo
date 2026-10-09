// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Randomized differential test against the set-semantics reference.
//!
//! Each seed generates one operation stream over two to eight ranks (two to four workers,
//! one or two dp ranks each) sharing a prefix pool: stores from the rank's held prefix,
//! some split into two chained events, some re-storing only part of a hole, some duplicate;
//! decode extensions that ranks race to share; removals of tail runs, head runs, single
//! mid-chain blocks and random subsets, sent tail-first, head-first or shuffled and
//! sometimes spanning two sequences; per-rank `Cleared`; `remove_worker_dp_rank`; and
//! `remove_worker`, after which a worker either returns under the same id or is replaced.
//!
//! The same stream runs on a serial lane, checked after every operation, and through a
//! `ThreadPoolIndexer` with pipelined events, checked at random quiescent checkpoints.

use std::fmt::Write as _;

use rustc_hash::FxHashMap;

use super::driver::{Driver, Op, PoolDriver, SerialLane};
use super::reference::{Expect, Reference};
use super::workload::{Pool, mix};
use super::{HarnessBackend, env_u64};
use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheStoreData, LocalBlockHash, OverlapScores,
    WorkerWithDpRank, compute_seq_hash_for_block,
};
use crate::test_utils::{remove_event, router_event, stored_blocks_with_sequence_hashes};

const LIVE_PER_RANK: usize = 6;
const PRINTED_FAILURES: u64 = 5;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Mode {
    Serial,
    Pool { lanes: usize },
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serial => f.write_str("serial"),
            Self::Pool { lanes } => write!(f, "pool{lanes}"),
        }
    }
}

/// One generated operation and the lookups to check after it.
struct Step {
    op: Op,
    queries: Vec<Vec<LocalBlockHash>>,
    /// Pool modes check only here; the serial lane checks after every step.
    checkpoint: bool,
}

struct Generator {
    rng: fastrand::Rng,
    pool: Pool,
    reference: Reference,
    ranks: Vec<WorkerWithDpRank>,
    live: FxHashMap<WorkerWithDpRank, Vec<Vec<LocalBlockHash>>>,
    next_worker_id: u64,
    next_event_id: u64,
}

fn seq_hashes(seq: &[LocalBlockHash]) -> Vec<u64> {
    compute_seq_hash_for_block(seq)
}

impl Generator {
    fn new(seed: u64, doc_len: u64) -> Self {
        let workers = 2 + seed % 3;
        let dp_ranks = 1 + (seed / 3 % 2) as u32;
        let ranks = (0..workers)
            .flat_map(|w| (0..dp_ranks).map(move |r| WorkerWithDpRank::new(w, r)))
            .collect();
        Self {
            rng: fastrand::Rng::with_seed(mix(seed ^ 0xD1FF)),
            pool: Pool {
                salt: seed,
                doc_len,
            },
            reference: Reference::default(),
            ranks,
            live: FxHashMap::default(),
            next_worker_id: workers,
            next_event_id: 0,
        }
    }

    fn event(&mut self, rank: WorkerWithDpRank, data: KvCacheEventData) -> Op {
        self.next_event_id += 1;
        Op::Event(router_event(
            rank.worker_id,
            self.next_event_id,
            rank.dp_rank,
            data,
        ))
    }

    fn store_event(
        &mut self,
        rank: WorkerWithDpRank,
        seq: &[LocalBlockHash],
        seqs: &[u64],
        start: usize,
        end: usize,
    ) -> Op {
        let data = KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: start
                .checked_sub(1)
                .map(|i| ExternalSequenceBlockHash(seqs[i])),
            start_position: Some(start as u32),
            blocks: stored_blocks_with_sequence_hashes(&seq[start..end], &seqs[start..end]),
        });
        self.event(rank, data)
    }

    fn remember(&mut self, rank: WorkerWithDpRank, seq: Vec<LocalBlockHash>) {
        let live = self.live.entry(rank).or_default();
        live.retain(|existing| *existing != seq);
        if live.len() >= LIVE_PER_RANK {
            live.remove(0);
        }
        live.push(seq);
    }

    fn pick_live(&mut self, rank: WorkerWithDpRank) -> Option<Vec<LocalBlockHash>> {
        let live = self.live.get(&rank)?;
        (!live.is_empty()).then(|| live[self.rng.usize(..live.len())].clone())
    }

    fn holds(&self, rank: WorkerWithDpRank, hash: u64) -> bool {
        self.reference.rank(rank).is_some_and(|s| s.holds(hash))
    }

    fn held_prefix(&self, rank: WorkerWithDpRank, seqs: &[u64]) -> usize {
        self.reference.rank(rank).map_or(0, |s| s.prefix(seqs).0)
    }

    /// Stores from the rank's held prefix: a fresh range, a partial re-store of a hole,
    /// or a duplicate of blocks it already holds.
    fn store_ops(&mut self, rank: WorkerWithDpRank) -> Vec<Op> {
        let seq = match self.pick_live(rank) {
            Some(live) if self.rng.u32(..100) < 50 => live,
            _ => self.pool.random(&mut self.rng),
        };
        let seqs = seq_hashes(&seq);
        let held = self.held_prefix(rank, &seqs);
        let (start, end) = if held < seq.len() {
            let hole = seqs[held + 1..].iter().any(|&h| self.holds(rank, h));
            let end = if hole && self.rng.u32(..100) < 40 {
                held + 1
            } else {
                self.rng.usize(held + 1..=seq.len())
            };
            (held, end)
        } else if self.rng.u32(..100) < 25 {
            let end = self.rng.usize(1..=seq.len());
            (self.rng.usize(..end), end)
        } else {
            return Vec::new();
        };

        let ops = if end - start >= 2 && self.rng.u32(..100) < 30 {
            let mid = self.rng.usize(start + 1..end);
            vec![
                self.store_event(rank, &seq, &seqs, start, mid),
                self.store_event(rank, &seq, &seqs, mid, end),
            ]
        } else {
            vec![self.store_event(rank, &seq, &seqs, start, end)]
        };
        self.remember(rank, seq[..end.max(held)].to_vec());
        ops
    }

    /// Decode extension of a fully held live sequence.
    fn extend_ops(&mut self, rank: WorkerWithDpRank) -> Vec<Op> {
        let Some(mut seq) = self.pick_live(rank) else {
            return Vec::new();
        };
        let seqs = seq_hashes(&seq);
        if self.held_prefix(rank, &seqs) != seq.len() {
            return Vec::new();
        }
        let tail = *seqs.last().expect("live sequences are non-empty");
        let variant = match self.rng.u64(..3) {
            0 => 100 + rank.worker_id * 4 + u64::from(rank.dp_rank),
            v => v,
        };
        let start = seq.len();
        let count = self.rng.u64(1..=3);
        seq.extend(self.pool.extension(tail, variant, count));
        let seqs = seq_hashes(&seq);
        let op = self.store_event(rank, &seq, &seqs, start, seq.len());
        self.remember(rank, seq);
        vec![op]
    }

    fn held_positions(&self, rank: WorkerWithDpRank, seqs: &[u64]) -> Vec<usize> {
        (0..seqs.len())
            .filter(|&i| self.holds(rank, seqs[i]))
            .collect()
    }

    /// Removes held blocks of one live sequence (sometimes two) in a random pattern and
    /// order.
    fn remove_ops(&mut self, rank: WorkerWithDpRank) -> Vec<Op> {
        let Some(seq) = self.pick_live(rank) else {
            return Vec::new();
        };
        let seqs = seq_hashes(&seq);
        let held = self.held_positions(rank, &seqs);
        if held.is_empty() {
            return Vec::new();
        }
        let prefix = self.held_prefix(rank, &seqs);
        let mut positions: Vec<usize> = match self.rng.u32(..100) {
            // Tail run of the held prefix, the leaf-first eviction an LRU produces.
            0..35 if prefix > 0 => {
                let k = self.rng.usize(1..=prefix.min(4));
                (prefix - k..prefix).collect()
            }
            35..50 => {
                let k = self.rng.usize(1..=held.len().min(4));
                held[..k].to_vec()
            }
            50..70 => vec![held[self.rng.usize(..held.len())]],
            _ => {
                let k = self.rng.usize(1..=held.len().min(5));
                let mut pick = held.clone();
                self.rng.shuffle(&mut pick);
                pick.truncate(k);
                pick
            }
        };
        positions.sort_unstable();
        let mut hashes: Vec<u64> = positions.iter().map(|&i| seqs[i]).collect();
        match self.rng.u32(..3) {
            0 => hashes.reverse(),
            1 => {}
            _ => self.rng.shuffle(&mut hashes),
        }

        // Engines batch evictions across sequences.
        if self.rng.u32(..100) < 15
            && let Some(other) = self.pick_live(rank)
        {
            let other_seqs = seq_hashes(&other);
            let mut extra: Vec<u64> = self
                .held_positions(rank, &other_seqs)
                .into_iter()
                .rev()
                .take(self.rng.usize(1..=3))
                .map(|i| other_seqs[i])
                .filter(|h| !hashes.contains(h))
                .collect();
            if self.rng.bool() {
                extra.reverse();
            }
            hashes.extend(extra);
        }

        let data = KvCacheEventData::Removed(crate::protocols::KvCacheRemoveData {
            block_hashes: hashes.into_iter().map(ExternalSequenceBlockHash).collect(),
        });
        vec![self.event(rank, data)]
    }

    fn remove_worker(&mut self, worker_id: u64) -> Op {
        // Half the time the worker returns under its id; otherwise fresh ids take its
        // place, so released slots go to new ranks.
        if self.rng.bool() {
            let fresh = self.next_worker_id;
            self.next_worker_id += 1;
            for rank in &mut self.ranks {
                if rank.worker_id == worker_id {
                    self.live.remove(rank);
                    *rank = WorkerWithDpRank::new(fresh, rank.dp_rank);
                }
            }
        } else {
            self.live.retain(|rank, _| rank.worker_id != worker_id);
        }
        Op::RemoveWorker(worker_id)
    }

    fn next_ops(&mut self) -> Vec<Op> {
        let rank = self.ranks[self.rng.usize(..self.ranks.len())];
        match self.rng.u32(..100) {
            0..40 => self.store_ops(rank),
            40..52 => self.extend_ops(rank),
            52..94 => self.remove_ops(rank),
            94..97 => {
                self.live.remove(&rank);
                vec![self.event(rank, KvCacheEventData::Cleared)]
            }
            97..99 => {
                self.live.remove(&rank);
                vec![Op::RemoveRank(rank)]
            }
            _ => vec![self.remove_worker(rank.worker_id)],
        }
    }

    fn queries(&mut self) -> Vec<Vec<LocalBlockHash>> {
        let mut queries: Vec<Vec<LocalBlockHash>> = self
            .ranks
            .iter()
            .filter_map(|rank| self.live.get(rank))
            .flatten()
            .cloned()
            .collect();
        for _ in 0..4 {
            let mut seq = self.pool.random(&mut self.rng);
            seq.truncate(self.rng.usize(1..=seq.len()));
            queries.push(seq);
        }
        let mut stray = self.pool.random(&mut self.rng);
        stray.push(LocalBlockHash(self.rng.u64(..)));
        queries.push(stray);
        queries
    }
}

fn generate(seed: u64, steps: usize, doc_len: u64) -> Vec<Step> {
    let mut generator = Generator::new(seed, doc_len);
    let mut plan: Vec<Step> = Vec::with_capacity(steps + 1);
    while plan.len() < steps {
        for op in generator.next_ops() {
            generator.reference.apply(&op);
            let checkpoint = matches!(op, Op::RemoveWorker(_)) || generator.rng.u32(..6) == 0;
            let queries = generator.queries();
            plan.push(Step {
                op,
                queries,
                checkpoint,
            });
        }
    }
    if let Some(last) = plan.last_mut() {
        last.checkpoint = true;
    }
    plan
}

/// Where an overcount was first seen in a seed, enough to replay it.
#[derive(Clone, Debug)]
struct Failure {
    seed: u64,
    step: usize,
    query: Vec<LocalBlockHash>,
}

#[derive(Default)]
struct Tally {
    seeds: u64,
    ops: u64,
    checks: u64,
    queries: u64,
    rank_scores: u64,
    exact: u64,
    overcounts: u64,
    hole_undercounts: u64,
    unexplained_undercounts: u64,
    hash_mismatches: u64,
    allowed_rejects: u64,
    unexpected_apply_errors: u64,
    seeds_with_overcount: Vec<u64>,
    /// The first overcount of each failing seed.
    overcount_failures: Vec<Failure>,
}

fn rank_label(rank: WorkerWithDpRank) -> String {
    format!("w{}.r{}", rank.worker_id, rank.dp_rank)
}

impl Tally {
    fn check(
        &mut self,
        driver_scores: &OverlapScores,
        details: Option<&crate::indexer::MatchDetails>,
        reference: &Reference,
        query: &[LocalBlockHash],
        seed: u64,
        step: usize,
    ) {
        let seqs = seq_hashes(query);
        self.queries += 1;
        let mut overcount_here = false;
        for (&rank, state) in reference.ranks() {
            let (held, reachable) = state.prefix(&seqs);
            let got = driver_scores.scores.get(&rank).copied().unwrap_or(0) as usize;
            self.rank_scores += 1;
            if got > held {
                self.overcounts += 1;
                overcount_here = true;
                if self.overcounts <= PRINTED_FAILURES {
                    eprintln!(
                        "overcount: seed {seed} step {step}: {} scored {got} on a {}-block query, \
                         holds a {held}-block prefix ({reachable} reachable)",
                        rank_label(rank),
                        seqs.len()
                    );
                }
                if !self.overcount_failures.iter().any(|f| f.seed == seed) {
                    self.overcount_failures.push(Failure {
                        seed,
                        step,
                        query: query.to_vec(),
                    });
                }
            } else if got < reachable {
                self.unexplained_undercounts += 1;
                if self.unexplained_undercounts <= PRINTED_FAILURES {
                    eprintln!(
                        "unexplained undercount: seed {seed} step {step}: {} scored {got} on a \
                         {}-block query, {reachable} reachable",
                        rank_label(rank),
                        seqs.len()
                    );
                }
            } else if got < held {
                self.hole_undercounts += 1;
            } else {
                self.exact += 1;
            }
        }
        for (&rank, &score) in &driver_scores.scores {
            if score > 0 && reference.rank(rank).is_none() {
                self.overcounts += 1;
                overcount_here = true;
                if self.overcounts <= PRINTED_FAILURES {
                    eprintln!(
                        "overcount: seed {seed} step {step}: unknown rank {} scored {score}",
                        rank_label(rank)
                    );
                }
            }
        }
        if overcount_here && !self.seeds_with_overcount.contains(&seed) {
            self.seeds_with_overcount.push(seed);
        }

        let Some(details) = details else {
            return;
        };
        let mismatch = details.overlap_scores.scores != driver_scores.scores
            || details
                .overlap_scores
                .scores
                .iter()
                .filter(|&(_, &score)| score > 0)
                .any(|(rank, &score)| {
                    details.last_matched_hashes.get(rank).map(|h| h.0)
                        != seqs.get(score as usize - 1).copied()
                });
        if mismatch {
            self.hash_mismatches += 1;
            if self.hash_mismatches <= PRINTED_FAILURES {
                eprintln!(
                    "detail mismatch: seed {seed} step {step}: scores {:?} vs details {:?}",
                    driver_scores.scores, details.overlap_scores.scores
                );
            }
        }
    }

    fn record_ack(&mut self, applied: bool, expect: Expect, seed: u64, step: usize) {
        self.ops += 1;
        match (applied, expect) {
            (true, _) => {}
            (false, Expect::MayReject) => self.allowed_rejects += 1,
            (false, Expect::Apply) => {
                self.unexpected_apply_errors += 1;
                if self.unexpected_apply_errors <= PRINTED_FAILURES {
                    eprintln!("unexpected apply error: seed {seed} step {step}");
                }
            }
        }
    }
}

fn new_driver<T: HarnessBackend>(mode: Mode) -> Driver<T> {
    match mode {
        Mode::Serial => Driver::Serial(SerialLane::new()),
        Mode::Pool { lanes } => Driver::Pool(PoolDriver::new(lanes)),
    }
}

fn run_seed<T: HarnessBackend>(mode: Mode, seed: u64, plan: &[Step], tally: &mut Tally) {
    let driver = new_driver::<T>(mode);
    let mut reference = Reference::default();
    let mut batch = Vec::new();
    let mut expects = Vec::new();
    for (step, planned) in plan.iter().enumerate() {
        expects.push((step, reference.apply(&planned.op)));
        batch.push(planned.op.clone());
        if matches!(mode, Mode::Pool { .. }) && !planned.checkpoint {
            continue;
        }
        let acks = driver.apply_all(std::mem::take(&mut batch));
        for (applied, (at, expect)) in acks.into_iter().zip(expects.drain(..)) {
            tally.record_ack(applied, expect, seed, at);
        }
        tally.checks += 1;
        for (i, query) in planned.queries.iter().enumerate() {
            let scores = driver.find(query);
            let details = if i % 4 == 0 {
                driver.details(query)
            } else {
                None
            };
            tally.check(&scores, details.as_ref(), &reference, query, seed, step);
        }
    }
    tally.seeds += 1;
}

struct Settings {
    seeds: Vec<u64>,
    steps: usize,
    doc_len: u64,
    shrink: bool,
    strict: bool,
    trace: bool,
}

impl Settings {
    fn from_env() -> Self {
        let seeds = match std::env::var("HARNESS_SEED") {
            Ok(_) => vec![env_u64("HARNESS_SEED", 1)],
            Err(_) => {
                let base = env_u64("HARNESS_SEED_BASE", 1);
                (base..base + env_u64("HARNESS_SEEDS", 40)).collect()
            }
        };
        Self {
            seeds,
            steps: env_u64("HARNESS_STEPS", 2000) as usize,
            doc_len: env_u64("HARNESS_DOC_LEN", 6),
            shrink: env_u64("HARNESS_SHRINK", 1) != 0,
            strict: env_u64("HARNESS_STRICT", 1) != 0,
            trace: env_u64("HARNESS_TRACE", 0) != 0,
        }
    }
}

pub(crate) fn run_suite<T: HarnessBackend>(backend: &str, mode: Mode) {
    let settings = Settings::from_env();
    let mut tally = Tally::default();
    for &seed in &settings.seeds {
        let plan = generate(seed, settings.steps, settings.doc_len);
        if settings.trace {
            trace_plan(seed, &plan);
        }
        run_seed::<T>(mode, seed, &plan, &mut tally);
    }

    let mut seeds_with_overcount = tally.seeds_with_overcount.clone();
    seeds_with_overcount.sort_unstable();
    eprintln!(
        "DIFF backend={backend} mode={mode} seeds={} steps={} doc_len={} ops={} checks={} \
         queries={} rank_scores={} exact={} overcounts={} hole_undercounts={} \
         unexplained_undercounts={} hash_mismatches={} allowed_rejects={} \
         unexpected_apply_errors={} seeds_with_overcount={}/{} {:?}",
        tally.seeds,
        settings.steps,
        settings.doc_len,
        tally.ops,
        tally.checks,
        tally.queries,
        tally.rank_scores,
        tally.exact,
        tally.overcounts,
        tally.hole_undercounts,
        tally.unexplained_undercounts,
        tally.hash_mismatches,
        tally.allowed_rejects,
        tally.unexpected_apply_errors,
        seeds_with_overcount.len(),
        tally.seeds,
        seeds_with_overcount,
    );

    if settings.shrink {
        let lanes = match mode {
            Mode::Serial => 1,
            Mode::Pool { lanes } => lanes,
        };
        report_smallest_repro::<T>(backend, lanes, &tally.overcount_failures, &settings);
    }

    assert_eq!(
        tally.overcounts, 0,
        "{backend}/{mode}: lookups credited ranks past the blocks they hold"
    );
    assert_eq!(
        tally.hash_mismatches, 0,
        "{backend}/{mode}: detailed lookups disagreed with scores or the scored tail"
    );
    if settings.strict {
        assert_eq!(
            tally.unexplained_undercounts, 0,
            "{backend}/{mode}: undercounts not explained by holes"
        );
        assert_eq!(
            tally.unexpected_apply_errors, 0,
            "{backend}/{mode}: valid operations failed to apply"
        );
    }
}

// ----------------------------------------------------------------------------
// Shrinking
// ----------------------------------------------------------------------------

/// Failing seeds shrunk, earliest failure first; the smallest repro is printed.
const SHRUNK_SEEDS: usize = 2;

#[derive(Clone, Copy, Debug)]
struct Witness {
    rank: WorkerWithDpRank,
    got: usize,
    held: usize,
    reachable: usize,
}

/// How the shrinker replays a candidate stream.
#[derive(Clone, Copy, Debug)]
enum Replay {
    /// One operation at a time on `lanes` event lanes (a serial lane for one), which is
    /// deterministic: ranks take lanes round-robin in order of first event.
    Sequential { lanes: usize },
    /// The run's own checkpoint batches, pipelined on `lanes` lanes, for overcounts that
    /// need another rank's events to run concurrently. Accepted candidates are real
    /// overcounts, but their replay may need a few attempts.
    Batched { lanes: usize },
}

impl std::fmt::Display for Replay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sequential { lanes: 1 } => f.write_str("applied one at a time on a serial lane"),
            Self::Sequential { lanes } => write!(
                f,
                "applied one at a time on {lanes} lanes, ranks assigned round-robin by first event"
            ),
            Self::Batched { lanes } => write!(
                f,
                "pipelined in the listed batches on {lanes} lanes, ranks assigned round-robin by \
                 first event"
            ),
        }
    }
}

/// An operation and the checkpoint batch it ran in.
type Tagged = (usize, Op);

/// Replays `ops` on a fresh backend as `replay` says and returns an overcount on `query`,
/// if any. Streams an engine could not emit are rejected first.
fn overcount_witness<T: HarnessBackend>(
    replay: Replay,
    ops: &[Tagged],
    query: &[LocalBlockHash],
) -> Option<Witness> {
    let mut reference = Reference::default();
    for (_, op) in ops {
        if !reference.is_valid(op) {
            return None;
        }
        reference.apply(op);
    }
    let untagged = || ops.iter().map(|(_, op)| op.clone()).collect::<Vec<_>>();
    let scores = match replay {
        Replay::Sequential { lanes: 1 } => {
            let lane = SerialLane::<T>::new();
            for op in untagged() {
                lane.apply(op);
            }
            lane.backend().find_matches(query, false)
        }
        Replay::Sequential { lanes } => {
            let pool = PoolDriver::<T>::new(lanes);
            pool.apply_sequential(untagged());
            pool.backend().find_matches(query, false)
        }
        Replay::Batched { lanes } => {
            let pool = PoolDriver::<T>::new(lanes);
            for batch in ops.chunk_by(|a, b| a.0 == b.0) {
                pool.apply_batch(batch.iter().map(|(_, op)| op.clone()).collect());
            }
            pool.backend().find_matches(query, false)
        }
    };
    let seqs = seq_hashes(query);
    let mut witnesses: Vec<Witness> = scores
        .scores
        .iter()
        .filter_map(|(&rank, &score)| {
            let (held, reachable) = reference.rank(rank).map_or((0, 0), |s| s.prefix(&seqs));
            (score as usize > held).then_some(Witness {
                rank,
                got: score as usize,
                held,
                reachable,
            })
        })
        .collect();
    witnesses.sort_by_key(|w| (w.rank.worker_id, w.rank.dp_rank));
    witnesses.into_iter().next()
}

struct Repro {
    replay: Replay,
    seed: u64,
    step: usize,
    ops: Vec<Tagged>,
    query: Vec<LocalBlockHash>,
    witness: Witness,
}

/// Blocks named across `ops`, the shrinker's secondary size measure.
fn weight(ops: &[Tagged]) -> usize {
    ops.iter()
        .map(|(_, op)| match op {
            Op::Event(event) => match &event.event.data {
                KvCacheEventData::Stored(store) => store.blocks.len(),
                KvCacheEventData::Removed(remove) => remove.block_hashes.len(),
                KvCacheEventData::Cleared => 1,
            },
            _ => 1,
        })
        .sum()
}

fn op_rank(op: &Op) -> Option<WorkerWithDpRank> {
    match op {
        Op::Event(event) => Some(WorkerWithDpRank::new(event.worker_id, event.event.dp_rank)),
        Op::RemoveRank(rank) => Some(*rank),
        Op::RemoveWorker(_) => None,
    }
}

fn with_rank(op: &Op, from: WorkerWithDpRank, to: WorkerWithDpRank) -> Op {
    match op {
        Op::Event(event)
            if event.worker_id == from.worker_id && event.event.dp_rank == from.dp_rank =>
        {
            let mut event = event.clone();
            event.worker_id = to.worker_id;
            event.event.dp_rank = to.dp_rank;
            Op::Event(event)
        }
        Op::RemoveRank(rank) if *rank == from => Op::RemoveRank(to),
        Op::RemoveWorker(worker_id) if *worker_id == from.worker_id => {
            Op::RemoveWorker(to.worker_id)
        }
        other => other.clone(),
    }
}

fn stored(op: &Op) -> Option<(WorkerWithDpRank, &KvCacheStoreData)> {
    let Op::Event(event) = op else {
        return None;
    };
    let KvCacheEventData::Stored(store) = &event.event.data else {
        return None;
    };
    Some((
        WorkerWithDpRank::new(event.worker_id, event.event.dp_rank),
        store,
    ))
}

fn with_store(op: &Op, store: KvCacheStoreData) -> Op {
    let Op::Event(event) = op else {
        unreachable!("only store events are rewritten as stores")
    };
    let mut event = event.clone();
    event.event.data = KvCacheEventData::Stored(store);
    Op::Event(event)
}

/// Candidate rewrites of `ops`, smallest first: drop chunks, drop a rank's operations,
/// fold one rank into another, merge chained stores, and trim events.
fn candidates(ops: &[Tagged]) -> Vec<Vec<Tagged>> {
    let mut out = Vec::new();
    let mut chunk = ops.len() / 2;
    while chunk >= 1 {
        for start in (0..ops.len()).step_by(chunk) {
            let mut candidate = ops[..start].to_vec();
            candidate.extend_from_slice(&ops[(start + chunk).min(ops.len())..]);
            out.push(candidate);
        }
        chunk /= 2;
    }

    let mut ranks: Vec<WorkerWithDpRank> = Vec::new();
    for rank in ops.iter().filter_map(|(_, op)| op_rank(op)) {
        if !ranks.contains(&rank) {
            ranks.push(rank);
        }
    }
    for &drop in &ranks {
        out.push(
            ops.iter()
                .filter(|(_, op)| op_rank(op) != Some(drop))
                .cloned()
                .collect(),
        );
    }
    for (i, &keep) in ranks.iter().enumerate() {
        for &fold in &ranks[i + 1..] {
            out.push(
                ops.iter()
                    .map(|(tag, op)| (*tag, with_rank(op, fold, keep)))
                    .collect(),
            );
        }
    }

    for i in 0..ops.len().saturating_sub(1) {
        let (Some((rank, first)), Some((next_rank, second))) =
            (stored(&ops[i].1), stored(&ops[i + 1].1))
        else {
            continue;
        };
        let chained = second.parent_hash == first.blocks.last().map(|b| b.block_hash);
        if rank != next_rank || !chained {
            continue;
        }
        let mut merged = first.clone();
        merged.blocks.extend(second.blocks.iter().cloned());
        let mut candidate = ops[..i].to_vec();
        candidate.push((ops[i].0, with_store(&ops[i].1, merged)));
        candidate.extend_from_slice(&ops[i + 2..]);
        out.push(candidate);
    }

    for (i, (tag, op)) in ops.iter().enumerate() {
        let Op::Event(event) = op else {
            continue;
        };
        match &event.event.data {
            KvCacheEventData::Removed(remove) if remove.block_hashes.len() > 1 => {
                for j in 0..remove.block_hashes.len() {
                    let mut candidate = ops.to_vec();
                    let Op::Event(event) = &mut candidate[i].1 else {
                        unreachable!()
                    };
                    let KvCacheEventData::Removed(remove) = &mut event.event.data else {
                        unreachable!()
                    };
                    remove.block_hashes.remove(j);
                    out.push(candidate);
                }
            }
            KvCacheEventData::Stored(store) if store.blocks.len() > 1 => {
                let mut trailing = store.clone();
                trailing.blocks.pop();
                let mut candidate = ops.to_vec();
                candidate[i] = (*tag, with_store(op, trailing));
                out.push(candidate);

                let mut leading = store.clone();
                let first = leading.blocks.remove(0);
                leading.parent_hash = Some(first.block_hash);
                leading.start_position = leading.start_position.map(|p| p + 1);
                let mut candidate = ops.to_vec();
                candidate[i] = (*tag, with_store(op, leading));
                out.push(candidate);
            }
            _ => {}
        }
    }
    out
}

/// Greedily applies the first candidate that still overcounts until none does, then
/// trims the query.
fn shrink<T: HarnessBackend>(
    replay: Replay,
    mut ops: Vec<Tagged>,
    mut query: Vec<LocalBlockHash>,
) -> Option<(Vec<Tagged>, Vec<LocalBlockHash>, Witness)> {
    let measure = |ops: &[Tagged]| {
        let mut ranks: Vec<_> = ops.iter().filter_map(|(_, op)| op_rank(op)).collect();
        ranks.sort_by_key(|r| (r.worker_id, r.dp_rank));
        ranks.dedup();
        (ops.len(), weight(ops), ranks.len())
    };
    'outer: loop {
        for candidate in candidates(&ops) {
            if measure(&candidate) < measure(&ops)
                && overcount_witness::<T>(replay, &candidate, &query).is_some()
            {
                ops = candidate;
                continue 'outer;
            }
        }
        break;
    }
    for len in 1..query.len() {
        if overcount_witness::<T>(replay, &ops, &query[..len]).is_some() {
            query.truncate(len);
            break;
        }
    }
    // A batched candidate was seen to overcount once; retry it a few times for the report.
    let witness = (0..8).find_map(|_| overcount_witness::<T>(replay, &ops, &query))?;
    Some((ops, query, witness))
}

fn report_smallest_repro<T: HarnessBackend>(
    backend: &str,
    lanes: usize,
    failures: &[Failure],
    settings: &Settings,
) {
    let mut best: Option<Repro> = None;
    let mut earliest: Vec<&Failure> = failures.iter().collect();
    earliest.sort_by_key(|f| (f.step, f.seed));
    for failure in earliest.into_iter().take(SHRUNK_SEEDS) {
        let plan = generate(failure.seed, settings.steps, settings.doc_len);
        let mut batch = 0;
        let mut ops: Vec<Tagged> = Vec::with_capacity(failure.step + 1);
        for step in &plan[..=failure.step] {
            ops.push((batch, step.op.clone()));
            batch += usize::from(step.checkpoint);
        }
        let sequential = Replay::Sequential { lanes };
        let replay = if overcount_witness::<T>(sequential, &ops, &failure.query).is_some() {
            sequential
        } else {
            let batched = Replay::Batched { lanes };
            let reproduces =
                (0..4).any(|_| overcount_witness::<T>(batched, &ops, &failure.query).is_some());
            if lanes == 1 || !reproduces {
                eprintln!(
                    "DIFF repro backend={backend}: seed {} step {} does not reproduce on a fresh \
                     backend",
                    failure.seed, failure.step
                );
                continue;
            }
            batched
        };
        let Some((mut ops, mut query, mut witness)) =
            shrink::<T>(replay, ops, failure.query.clone())
        else {
            eprintln!(
                "DIFF repro backend={backend}: seed {} step {} stopped reproducing while \
                 shrinking",
                failure.seed, failure.step
            );
            continue;
        };
        // A batched repro often no longer needs concurrency once shrunk; prefer a
        // deterministic one.
        let mut replay = replay;
        if matches!(replay, Replay::Batched { .. })
            && overcount_witness::<T>(sequential, &ops, &query).is_some()
            && let Some(sequential_repro) = shrink::<T>(sequential, ops.clone(), query.clone())
        {
            (ops, query, witness) = sequential_repro;
            replay = sequential;
        }
        let repro = Repro {
            replay,
            seed: failure.seed,
            step: failure.step,
            ops,
            query,
            witness,
        };
        let size = |r: &Repro| {
            (
                matches!(r.replay, Replay::Batched { .. }),
                r.ops.len(),
                weight(&r.ops),
                r.query.len(),
            )
        };
        if best.as_ref().is_none_or(|b| size(&repro) < size(b)) {
            best = Some(repro);
        }
    }
    if let Some(best) = best {
        eprintln!("{}", best.render(backend));
    }
}

/// Short names for hashes and ranks, in order of first appearance.
#[derive(Default)]
struct Labeler {
    hashes: FxHashMap<u64, String>,
    workers: Vec<u64>,
}

impl Labeler {
    fn hash(&mut self, hash: u64) -> String {
        let next = self.hashes.len() + 1;
        self.hashes
            .entry(hash)
            .or_insert_with(|| format!("b{next}"))
            .clone()
    }

    fn hashes(&mut self, hashes: impl IntoIterator<Item = u64>) -> String {
        let labels: Vec<String> = hashes.into_iter().map(|h| self.hash(h)).collect();
        labels.join(" ")
    }

    fn worker(&mut self, worker_id: u64) -> usize {
        self.workers
            .iter()
            .position(|&w| w == worker_id)
            .unwrap_or_else(|| {
                self.workers.push(worker_id);
                self.workers.len() - 1
            })
    }

    fn rank(&mut self, rank: WorkerWithDpRank) -> String {
        format!("w{}.r{}", self.worker(rank.worker_id), rank.dp_rank)
    }

    fn op(&mut self, op: &Op) -> String {
        match op {
            Op::Event(event) => {
                let rank = self.rank(WorkerWithDpRank::new(event.worker_id, event.event.dp_rank));
                match &event.event.data {
                    KvCacheEventData::Stored(store) => {
                        let parent = store
                            .parent_hash
                            .map_or_else(|| "root".to_string(), |p| self.hash(p.0));
                        let blocks = self.hashes(store.blocks.iter().map(|b| b.block_hash.0));
                        format!("{rank} store  parent={parent} blocks=[{blocks}]")
                    }
                    KvCacheEventData::Removed(remove) => {
                        let hashes = self.hashes(remove.block_hashes.iter().map(|h| h.0));
                        format!("{rank} remove [{hashes}]")
                    }
                    KvCacheEventData::Cleared => format!("{rank} cleared"),
                }
            }
            Op::RemoveRank(rank) => format!("remove_worker_dp_rank {}", self.rank(*rank)),
            Op::RemoveWorker(worker_id) => {
                format!("remove_worker w{}", self.worker(*worker_id))
            }
        }
    }
}

impl Repro {
    fn render(&self, backend: &str) -> String {
        let mut labels = Labeler::default();
        let mut out = String::new();
        let _ = writeln!(
            out,
            "DIFF repro backend={backend} (seed {} step {}, shrunk to {} ops, {}):",
            self.seed,
            self.step,
            self.ops.len(),
            self.replay
        );
        let batched = matches!(self.replay, Replay::Batched { .. });
        for (i, (tag, op)) in self.ops.iter().enumerate() {
            if batched && i > 0 && self.ops[i - 1].0 != *tag {
                let _ = writeln!(out, "     -- flush --");
            }
            let _ = writeln!(out, "  {}. {}", i + 1, labels.op(op));
        }
        let query = labels.hashes(seq_hashes(&self.query));
        let w = self.witness;
        let _ = write!(
            out,
            "  query [{query}]: {} scored {}, but holds only a {}-block prefix ({} reachable)",
            labels.rank(w.rank),
            w.got,
            w.held,
            w.reachable
        );
        out
    }
}

/// Prints a seed's whole stream with `HARNESS_TRACE=1`.
fn trace_plan(seed: u64, plan: &[Step]) {
    let mut labels = Labeler::default();
    eprintln!("TRACE seed {seed}:");
    for (step, planned) in plan.iter().enumerate() {
        let mark = if planned.checkpoint {
            " | checkpoint"
        } else {
            ""
        };
        eprintln!("  step {step}: {}{mark}", labels.op(&planned.op));
    }
}

// ----------------------------------------------------------------------------
// Known repros
// ----------------------------------------------------------------------------

/// The grouped-removal overcount in ConcurrentRadixTreeCompressed at 22f7f2a103, as two
/// single-rank streams. The first is this differential's own shrunk repro (`DIFF repro`
/// output):
///
/// ```text
/// 1. w0.r0 store  parent=root blocks=[b1 b2 b3]
/// 2. w0.r0 store  parent=b1 blocks=[b4]   (splits [b1 b2 b3] after b1)
/// 3. w0.r0 remove [b1]                    (no full rank left: [b2 b3] and [b4] unlinked)
/// 4. w0.r0 store  parent=root blocks=[b1 b2]
/// 5. w0.r0 remove [b3 b2]
/// query [b1 b2]: w0.r0 scores 2 but holds only b1
/// ```
///
/// Step 4 puts b2 in a new child and repoints the rank's b2 lookup there, but the b3
/// lookup still names the unlinked `[b2 b3]` node. Step 5 resolves the run through b3,
/// finds b2 in that same stale edge, and applies the whole run there, so the live b2
/// keeps the rank's coverage; scrubbing b2's lookup entry also hides it from later
/// removals.
///
/// The second is the six-event form with the restore split in two (local hashes):
/// `store [1 30]; store [2 20 21] under [1]; remove [1]; store [1]; store [2 20] under
/// [1]; remove [seq(1 2 20 21) seq(1 2 20)]`, after which `[1 2 20]` scores 3 for a true 2.
pub(crate) fn known_grouped_removal_overcount<T: HarnessBackend>(backend: &str) {
    let mut overcounts = Vec::new();
    for (name, ops, query) in grouped_removal_repros() {
        let mut reference = Reference::default();
        for op in &ops {
            assert!(
                reference.is_valid(op),
                "{name}: the repro must be an engine-valid stream"
            );
            reference.apply(op);
        }
        let seqs = seq_hashes(&query);
        for mode in [
            Mode::Serial,
            Mode::Pool { lanes: 1 },
            Mode::Pool { lanes: 4 },
        ] {
            let driver = new_driver::<T>(mode);
            driver.apply_all(ops.clone());
            let scores = driver.find(&query);
            for (&rank, &score) in &scores.scores {
                let (held, reachable) = reference.rank(rank).map_or((0, 0), |s| s.prefix(&seqs));
                eprintln!(
                    "REPRO backend={backend} repro={name} mode={mode} {} scored {score} on a \
                     {}-block query, holds {held} ({reachable} reachable)",
                    rank_label(rank),
                    query.len()
                );
                if score as usize > held {
                    overcounts.push(format!(
                        "{name}/{mode}: {} scored {score} > {held}",
                        rank_label(rank)
                    ));
                }
            }
        }
    }
    assert!(
        overcounts.is_empty(),
        "{backend}: grouped-removal overcount: {overcounts:?}"
    );
}

type NamedRepro = (&'static str, Vec<Op>, Vec<LocalBlockHash>);

fn grouped_removal_repros() -> Vec<NamedRepro> {
    let locals = |hashes: &[u64]| -> Vec<LocalBlockHash> {
        hashes.iter().copied().map(LocalBlockHash).collect()
    };
    let rank = WorkerWithDpRank::new(0, 0);
    let event =
        |data: KvCacheEventData| Op::Event(router_event(rank.worker_id, 0, rank.dp_rank, data));
    // Stores `chain[start..]` under `chain[start - 1]`.
    let store = |chain: &[u64], start: usize| {
        let chain = locals(chain);
        let seqs = seq_hashes(&chain);
        event(KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: start
                .checked_sub(1)
                .map(|i| ExternalSequenceBlockHash(seqs[i])),
            start_position: Some(start as u32),
            blocks: stored_blocks_with_sequence_hashes(&chain[start..], &seqs[start..]),
        }))
    };
    // Removes the blocks of `chain` at `positions`, in that order.
    let remove = |chain: &[u64], positions: &[usize]| {
        let seqs = seq_hashes(&locals(chain));
        Op::Event(remove_event(
            rank.worker_id,
            0,
            rank.dp_rank,
            positions
                .iter()
                .map(|&i| ExternalSequenceBlockHash(seqs[i]))
                .collect(),
        ))
    };
    vec![
        (
            "shrunk-5",
            vec![
                store(&[1, 2, 3], 0),
                store(&[1, 4], 1),
                remove(&[1], &[0]),
                store(&[1, 2], 0),
                remove(&[1, 2, 3], &[2, 1]),
            ],
            locals(&[1, 2]),
        ),
        (
            "six-event",
            vec![
                store(&[1, 30], 0),
                store(&[1, 2, 20, 21], 1),
                remove(&[1], &[0]),
                store(&[1], 0),
                store(&[1, 2, 20], 1),
                remove(&[1, 2, 20, 21], &[3, 2]),
            ],
            locals(&[1, 2, 20]),
        ),
    ]
}
