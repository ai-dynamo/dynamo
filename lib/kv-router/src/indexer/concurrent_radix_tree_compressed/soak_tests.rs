// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in race soak for `ConcurrentRadixTreeCompressed`.
//!
//! Writer threads act as event lanes: each owns a disjoint set of ranks, as
//! `ThreadPoolIndexer` sticky routing would assign them, and its own `LaneLookup`. They
//! replay adversarial per-rank event streams over a pool of heavily shared prefixes
//! (system prompt, then document, then user turn): stores, some split into two chained
//! events; decode extensions, most of which share their tail with other ranks; tail
//! evictions whose blocks arrive in either order; and clears. Reader threads look up
//! concurrently through `find_matches_impl` and `find_match_details_impl`, and a cleanup
//! thread sweeps stale children.
//!
//! Checks:
//! - Every lookup score, even mid-race, must stay within the blocks its rank has ever
//!   stored, and detailed lookups must report the scored prefix's last sequence hash.
//!   This catches credit for blocks a rank never stored, but not for blocks it has since
//!   evicted.
//! - In strict mode, every `SOAK_CHECK_MS` and once at the end, writers pause at a batch
//!   boundary and lookups over every live sequence plus random queries must match a
//!   per-rank model keyed by sequence hash exactly. Only these checks catch credit for
//!   evicted blocks.
//! - Each lane's `LaneLookup` invariants are checked every 64 batches and at the end.
//!
//! With `SOAK_CHURN`, writers also remove a rank (`RemoveWorkerDpRank` on its own lane,
//! racing the other lanes) or a whole worker (`RemoveWorker`: every lane drops the
//! worker's ranks before its next event, then the removing lane sweeps while the others
//! keep applying events) and replace each removed rank with a fresh one, so released
//! slots are recycled to new ranks. `SOAK_SLOT_OFFSET` parks idle ranks on the lowest
//! slots before the run, so a value of 256 or more puts every live rank on the overflow
//! slot chunks.
//!
//! Modes (`SOAK_MODE`):
//! - `strict` (default): streams an engine could emit. Every event must apply, and the
//!   quiescent parity checks run.
//! - `chaos`: a quarter of evictions remove a single block anywhere in the rank's cached
//!   chain. That uncovers the rest of the block's compressed edge for the rank, so later
//!   stores can fail and the model no longer predicts exact scores. Only the ever-stored
//!   and last-hash checks apply, so chaos mode cannot detect credit for evicted blocks.
//!
//! Knobs (environment variables, default in parentheses):
//! - `SOAK_SECS` (10): run time in seconds.
//! - `SOAK_WRITERS` (8): writer threads, one event lane each.
//! - `SOAK_READERS` (4): reader threads.
//! - `SOAK_WORKERS` (64): ranks, two dp ranks per worker id, spread over the writers.
//! - `SOAK_SEED` (1): seed for every thread's random stream.
//! - `SOAK_MODE` (`strict`): `strict` or `chaos`.
//! - `SOAK_CHECK_MS` (500): interval between strict-mode parity checks.
//! - `SOAK_DOC_LEN` (10): maximum document length in blocks; longer documents give
//!   longer compressed edges.
//! - `SOAK_MAX_REMOVE` (4): maximum blocks per tail eviction.
//! - `SOAK_CHURN` (0): per-mille of each writer's batches that start with a removal.
//! - `SOAK_SLOT_OFFSET` (0): slots held by idle ranks before the run.
//!
//! Run it in release mode, for example:
//!
//! ```text
//! SOAK_SECS=60 SOAK_WRITERS=16 SOAK_READERS=6 SOAK_SEED=22 SOAK_DOC_LEN=40 \
//!   SOAK_MAX_REMOVE=32 SOAK_CHURN=20 SOAK_SLOT_OFFSET=300 \
//!   cargo test -p dynamo-kv-router --release --lib crtc_race_soak -- --ignored --nocapture
//! ```
//!
//! The run ends with one `SOAK` line on stderr: the configuration (`writers` counts only
//! the lanes that received a rank), then `events`, `apply_errors`, `reads`, `overcounts`
//! (scores past the blocks a rank ever stored), `hash_mismatches`, `checks` (parity
//! passes), `parity_queries`, `parity_mismatches`, `edges` (child edges left after a
//! final cleanup), and the churn counters `rank_retires`, `worker_retires`, and
//! `recycled_slots` (slots a rank received after a different rank held them).
//! `overcounts` and `hash_mismatches` must be zero in both modes; strict mode also
//! requires zero `apply_errors` and `parity_mismatches`. The first five failures of each
//! check, and in strict mode of `apply_errors`, are printed above the line.

use super::*;
use dashmap::DashSet;
use parking_lot::{Mutex, RwLock, RwLockWriteGuard};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Pool shape: system prompts, documents per system prompt, user turns per document.
const SYSTEM_PROMPTS: u64 = 6;
const DOCUMENTS: u64 = 12;
const USER_TURNS: u64 = 6;

const BATCH_STEPS: usize = 64;
const QUERY_POOL: usize = 4096;
const PRINTED_FAILURES: u64 = 5;

fn env_u64(name: &str, default: u64) -> u64 {
    let Ok(value) = std::env::var(name) else {
        return default;
    };
    value
        .parse()
        .unwrap_or_else(|_| panic!("{name} must be an unsigned integer, got {value:?}"))
}

struct Config {
    secs: u64,
    writers: usize,
    readers: usize,
    workers: u64,
    seed: u64,
    chaos: bool,
    check_ms: u64,
    doc_len: u64,
    max_remove: u64,
    churn_per_mille: u64,
    slot_offset: u64,
}

impl Config {
    fn from_env() -> Self {
        let chaos = match std::env::var("SOAK_MODE").as_deref() {
            Err(_) | Ok("strict") => false,
            Ok("chaos") => true,
            Ok(other) => panic!("SOAK_MODE must be strict or chaos, got {other:?}"),
        };
        let config = Self {
            secs: env_u64("SOAK_SECS", 10),
            writers: env_u64("SOAK_WRITERS", 8) as usize,
            readers: env_u64("SOAK_READERS", 4) as usize,
            workers: env_u64("SOAK_WORKERS", 64),
            seed: env_u64("SOAK_SEED", 1),
            chaos,
            check_ms: env_u64("SOAK_CHECK_MS", 500),
            doc_len: env_u64("SOAK_DOC_LEN", 10),
            max_remove: env_u64("SOAK_MAX_REMOVE", 4),
            churn_per_mille: env_u64("SOAK_CHURN", 0),
            slot_offset: env_u64("SOAK_SLOT_OFFSET", 0),
        };
        assert!(config.writers > 0, "SOAK_WRITERS must be at least 1");
        config
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.0)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }

    fn chance(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }
}

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn hash_parts(parts: &[u64]) -> u64 {
    parts
        .iter()
        .fold(0xD1B5_4A32_D192_ED03, |acc, &p| mix(acc ^ mix(p)))
}

/// Local hashes of a pool sequence: system prompt, then document, then user turn, each
/// with a fixed length per id.
fn pool_seq(doc_len: u64, system: u64, doc: u64, turn: u64) -> Vec<u64> {
    let ls = 1 + hash_parts(&[1, system]) % 12;
    let ld = hash_parts(&[2, system, doc]) % (doc_len + 1);
    let lu = 1 + hash_parts(&[3, system, doc, turn]) % 8;
    let mut seq = Vec::with_capacity((ls + ld + lu) as usize);
    seq.extend((0..ls).map(|i| hash_parts(&[10, system, i])));
    seq.extend((0..ld).map(|i| hash_parts(&[11, system, doc, i])));
    seq.extend((0..lu).map(|i| hash_parts(&[12, system, doc, turn, i])));
    seq
}

fn random_pool_seq(doc_len: u64, rng: &mut Rng) -> Vec<u64> {
    let system = rng.below(SYSTEM_PROMPTS);
    let doc = rng.below(DOCUMENTS);
    let turn = rng.below(USER_TURNS);
    pool_seq(doc_len, system, doc, turn)
}

fn seq_hashes(locals: &[u64]) -> Vec<u64> {
    let locals: Vec<LocalBlockHash> = locals.iter().copied().map(LocalBlockHash).collect();
    compute_seq_hash_for_block(&locals)
}

/// One rank's cached blocks as its event stream implies them.
#[derive(Default)]
struct WorkerModel {
    /// Sequence hash -> (parent sequence hash, number of cached children).
    cached: FxHashMap<u64, (Option<u64>, u32)>,
    /// The most recent sequences this rank stored or extended, as local hashes.
    live: VecDeque<Vec<u64>>,
}

impl WorkerModel {
    fn prefix_len(&self, seqs: &[u64]) -> usize {
        seqs.iter()
            .take_while(|s| self.cached.contains_key(s))
            .count()
    }

    fn insert(&mut self, parent: Option<u64>, seq: u64) {
        if self.cached.contains_key(&seq) {
            return;
        }
        if let Some(p) = parent
            && let Some(entry) = self.cached.get_mut(&p)
        {
            entry.1 += 1;
        }
        self.cached.insert(seq, (parent, 0));
    }

    fn remove(&mut self, seq: u64) {
        let Some((parent, _)) = self.cached.remove(&seq) else {
            return;
        };
        if let Some(p) = parent
            && let Some(entry) = self.cached.get_mut(&p)
        {
            entry.1 -= 1;
        }
    }

    fn push_live(&mut self, seq: Vec<u64>) {
        if self.live.len() >= 48 {
            self.live.pop_front();
        }
        self.live.push_back(seq);
    }
}

struct Shared {
    index: ConcurrentRadixTreeCompressed,
    config: Config,
    /// Ranks of a removed worker that their owning lane has not dropped yet.
    retired: Mutex<FxHashSet<WorkerWithDpRank>>,
    next_worker_id: AtomicU64,
    /// Rank last seen on each slot, to count slot reuse.
    slot_owners: Mutex<FxHashMap<coverage::Slot, WorkerWithDpRank>>,
    rank_retires: AtomicU64,
    worker_retires: AtomicU64,
    recycled_slots: AtomicU64,
    /// Every (rank, sequence hash) a rank has stored, recorded before the store applies.
    ever: DashSet<(WorkerWithDpRank, u64)>,
    /// Held shared by each writer batch and cleanup pass, and exclusively by parity
    /// checks and whole-worker removals to pause them.
    gate: RwLock<()>,
    /// Each lane's rank models.
    models: Vec<Mutex<FxHashMap<WorkerWithDpRank, WorkerModel>>>,
    /// Extended decode sequences that readers sample as queries.
    queries: RwLock<Vec<Vec<u64>>>,
    stop: AtomicBool,
    events: AtomicU64,
    apply_errors: AtomicU64,
    reads: AtomicU64,
    overcounts: AtomicU64,
    hash_mismatches: AtomicU64,
    checks: AtomicU64,
    parity_queries: AtomicU64,
    parity_mismatches: AtomicU64,
}

fn store_event(
    worker: WorkerWithDpRank,
    id: u64,
    parent: Option<u64>,
    locals: &[u64],
    seqs: &[u64],
) -> RouterEvent {
    RouterEvent::new(
        worker.worker_id,
        KvCacheEvent {
            event_id: id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent.map(ExternalSequenceBlockHash),
                start_position: None,
                blocks: locals
                    .iter()
                    .zip(seqs)
                    .map(|(&l, &s)| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(s),
                        tokens_hash: LocalBlockHash(l),
                        mm_extra_info: None,
                    })
                    .collect(),
            }),
            dp_rank: worker.dp_rank,
        },
    )
}

fn data_event(worker: WorkerWithDpRank, id: u64, data: KvCacheEventData) -> RouterEvent {
    RouterEvent::new(
        worker.worker_id,
        KvCacheEvent {
            event_id: id,
            data,
            dp_rank: worker.dp_rank,
        },
    )
}

struct Writer {
    shared: Arc<Shared>,
    /// This writer's index into `Shared::models`.
    lane: usize,
    workers: Vec<WorkerWithDpRank>,
    lookup: LaneLookup,
    rng: Rng,
    next_id: u64,
    /// Ranks whose slot has not been recorded in `slot_owners` yet.
    unslotted: FxHashSet<WorkerWithDpRank>,
}

impl Writer {
    /// Applies `event`. `describe` names its data for the strict-mode failure report.
    fn apply(&mut self, event: RouterEvent, describe: impl FnOnce() -> String) -> bool {
        self.next_id += 1;
        self.shared.events.fetch_add(1, Ordering::Relaxed);
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let Err(error) = self.shared.index.apply_event(&mut self.lookup, event, None) else {
            return true;
        };
        let errors = self.shared.apply_errors.fetch_add(1, Ordering::Relaxed);
        if !self.shared.config.chaos && errors < PRINTED_FAILURES {
            eprintln!("apply error: {worker:?} {}: {error}", describe());
        }
        false
    }

    fn store(
        &mut self,
        model: &mut WorkerModel,
        worker: WorkerWithDpRank,
        locals: &[u64],
        from: usize,
        to: usize,
    ) {
        let seqs = seq_hashes(&locals[..to]);
        for &s in &seqs[from..to] {
            self.shared.ever.insert((worker, s));
        }
        let parent = (from > 0).then(|| seqs[from - 1]);
        let event = store_event(
            worker,
            self.next_id,
            parent,
            &locals[from..to],
            &seqs[from..to],
        );
        let describe = || format!("store of {:?} under parent {parent:?}", &seqs[from..to]);
        if !self.apply(event, describe) {
            return;
        }
        let mut p = parent;
        for &s in &seqs[from..to] {
            model.insert(p, s);
            p = Some(s);
        }
        if self.unslotted.remove(&worker)
            && let Some(slot) = self.shared.index.slot_for_test(worker)
            && self
                .shared
                .slot_owners
                .lock()
                .insert(slot, worker)
                .is_some_and(|previous| previous != worker)
        {
            self.shared.recycled_slots.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Retires one of this writer's ranks, or every rank of its worker, and re-adds each
    /// under a fresh worker id with an empty model.
    fn churn(&mut self) {
        let pick = self.rng.below(self.workers.len() as u64) as usize;
        let shared = self.shared.clone();
        if self.rng.chance(50) {
            // RemoveWorkerDpRank on the rank's own lane, racing every other lane. Adopting
            // first keeps it off a rank whose worker another lane already removed.
            let _batch = shared.gate.read();
            self.adopt_retirements();
            let worker = self.workers[pick];
            shared.index.remove_worker_coverage(
                &mut self.lookup,
                WorkerRemovalTarget::DpRank(worker),
                true,
            );
            self.replace_rank(worker);
            shared.rank_retires.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // RemoveWorker: retiring the worker's ranks on every other lane while all lanes are
        // paused stands in for the cross-lane barrier, since each lane drops them before
        // its next event. As in `ThreadPoolIndexer`, this lane then sweeps while the other
        // lanes keep applying events.
        let paused = shared.gate.write();
        self.adopt_retirements();
        let worker = self.workers[pick];
        {
            let mut retired = shared.retired.lock();
            for (lane, models) in shared.models.iter().enumerate() {
                if lane == self.lane {
                    continue;
                }
                retired.extend(
                    models
                        .lock()
                        .keys()
                        .filter(|rank| rank.worker_id == worker.worker_id),
                );
            }
        }
        // Downgrading atomically keeps parity checks out until the sweep is done.
        let _batch = RwLockWriteGuard::downgrade(paused);
        shared.index.remove_worker_coverage(
            &mut self.lookup,
            WorkerRemovalTarget::WorkerId(worker.worker_id),
            true,
        );
        let own: Vec<_> = self
            .workers
            .iter()
            .copied()
            .filter(|rank| rank.worker_id == worker.worker_id)
            .collect();
        for rank in own {
            self.replace_rank(rank);
        }
        shared.worker_retires.fetch_add(1, Ordering::Relaxed);
    }

    /// Drops this writer's ranks of workers another writer removed.
    fn adopt_retirements(&mut self) {
        let mine: Vec<_> = {
            let mut retired = self.shared.retired.lock();
            if retired.is_empty() {
                return;
            }
            let mine: Vec<_> = self
                .workers
                .iter()
                .copied()
                .filter(|rank| retired.contains(rank))
                .collect();
            for rank in &mine {
                retired.remove(rank);
            }
            mine
        };
        for rank in mine {
            self.shared.index.remove_worker_coverage(
                &mut self.lookup,
                WorkerRemovalTarget::DpRank(rank),
                false,
            );
            self.replace_rank(rank);
        }
    }

    fn replace_rank(&mut self, old: WorkerWithDpRank) {
        let fresh = WorkerWithDpRank::new(
            self.shared.next_worker_id.fetch_add(1, Ordering::Relaxed),
            0,
        );
        let index = self
            .workers
            .iter()
            .position(|&rank| rank == old)
            .expect("replaced rank belongs to this writer");
        self.workers[index] = fresh;
        self.unslotted.remove(&old);
        self.unslotted.insert(fresh);
        let mut models = self.shared.models[self.lane].lock();
        models.remove(&old);
        models.insert(fresh, WorkerModel::default());
    }

    fn step(&mut self) {
        let worker = self.workers[self.rng.below(self.workers.len() as u64) as usize];
        let shared = self.shared.clone();
        let config = &shared.config;
        let mut models = shared.models[self.lane].lock();
        let model = models
            .get_mut(&worker)
            .expect("every writer rank has a model");
        let roll = self.rng.below(1000);

        if roll < 450 {
            // Request store: extend this rank's cached prefix of a pool sequence, sometimes
            // as two chained events.
            let seq = random_pool_seq(config.doc_len, &mut self.rng);
            let target = 1 + self.rng.below(seq.len() as u64) as usize;
            let seqs = seq_hashes(&seq);
            let cached = model.prefix_len(&seqs[..target]);
            if cached < target {
                if target - cached >= 2 && self.rng.chance(30) {
                    let mid = cached + 1 + self.rng.below((target - cached - 1) as u64) as usize;
                    self.store(model, worker, &seq, cached, mid);
                    self.store(model, worker, &seq, mid, target);
                } else {
                    self.store(model, worker, &seq, cached, target);
                }
            }
            model.push_live(seq[..target].to_vec());
        } else if roll < 650 {
            // Decode extension of a fully cached live sequence.
            if model.live.is_empty() {
                return;
            }
            let idx = self.rng.below(model.live.len() as u64) as usize;
            let mut seq = model.live[idx].clone();
            let seqs = seq_hashes(&seq);
            if model.prefix_len(&seqs) != seq.len() {
                return;
            }
            let tail = *seqs.last().unwrap();
            // A third of the extensions are unique to the rank; the rest take one of two
            // tails every rank shares, so ranks race to extend and split the same leaf.
            let variant = match self.rng.below(3) {
                0 => worker.worker_id * 4 + worker.dp_rank as u64 + 100,
                v => v,
            };
            let start = seq.len();
            let m = 1 + self.rng.below(4) as usize;
            seq.extend((0..m).map(|i| hash_parts(&[20, tail, variant, i as u64])));
            self.store(model, worker, &seq, start, seq.len());
            if self.rng.chance(5) {
                let mut queries = shared.queries.write();
                let victim = self.rng.below(QUERY_POOL as u64) as usize;
                if queries.len() < QUERY_POOL {
                    queries.push(seq.clone());
                } else {
                    queries[victim] = seq.clone();
                }
            }
            model.live[idx] = seq;
        } else if roll < 990 {
            // Eviction.
            if model.live.is_empty() {
                return;
            }
            let idx = self.rng.below(model.live.len() as u64) as usize;
            let seq = model.live[idx].clone();
            let seqs = seq_hashes(&seq);
            let cached = model.prefix_len(&seqs);
            if cached == 0 {
                return;
            }
            let mut removed = Vec::new();
            if config.chaos && self.rng.chance(25) {
                // Mid-chain eviction: the rank keeps the blocks after it.
                let pos = self.rng.below(cached as u64) as usize;
                removed.push(seqs[pos]);
            } else {
                // Tail eviction, stopping at a block that has other cached children.
                let want = 1 + self.rng.below(config.max_remove) as usize;
                let mut pos = cached;
                while pos > 0 && removed.len() < want {
                    let s = seqs[pos - 1];
                    let children = model.cached.get(&s).map_or(0, |e| e.1);
                    let pending_child = removed.last().is_some_and(|_| children == 1);
                    if children != 0 && !pending_child {
                        break;
                    }
                    removed.push(s);
                    pos -= 1;
                }
            }
            if removed.is_empty() {
                return;
            }
            // Event order within a batch is arbitrary.
            if self.rng.chance(50) {
                removed.reverse();
            }
            let event = data_event(
                worker,
                self.next_id,
                KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: removed
                        .iter()
                        .copied()
                        .map(ExternalSequenceBlockHash)
                        .collect(),
                }),
            );
            if self.apply(event, || format!("removal of {removed:?}")) {
                // Either order keeps the child counts right: a parent removed first is
                // skipped when its child is removed.
                for s in removed {
                    model.remove(s);
                }
            }
        } else {
            let event = data_event(worker, self.next_id, KvCacheEventData::Cleared);
            if self.apply(event, || "clear".to_string()) {
                model.cached.clear();
                model.live.clear();
            }
        }
    }
}

fn check_read(shared: &Shared, query: &[u64], rng: &mut Rng) {
    let seqs = seq_hashes(query);
    let locals: Vec<LocalBlockHash> = query.iter().copied().map(LocalBlockHash).collect();
    let details = rng.chance(25);
    let (scores, last) = if details {
        let d = shared.index.find_match_details_impl(&locals, false);
        (d.overlap_scores.scores, Some(d.last_matched_hashes))
    } else {
        (shared.index.find_matches_impl(&locals, false).scores, None)
    };
    shared.reads.fetch_add(1, Ordering::Relaxed);
    for (&worker, &score) in &scores {
        let score = score as usize;
        // A rank's `ever` set is prefix-closed, since every store chains from blocks the
        // rank already stored, so the scored prefix's last block vouches for all of it.
        let ok = score <= seqs.len()
            && score
                .checked_sub(1)
                .is_none_or(|i| shared.ever.contains(&(worker, seqs[i])));
        if !ok && shared.overcounts.fetch_add(1, Ordering::Relaxed) < PRINTED_FAILURES {
            eprintln!(
                "overcount: {worker:?} scored {score} on a {}-block query, past the blocks it \
                 ever stored",
                seqs.len()
            );
        }
        if let Some(last) = &last
            && let Some(&tail) = score.checked_sub(1).and_then(|i| seqs.get(i))
            && last.get(&worker).map(|h| h.0) != Some(tail)
            && shared.hash_mismatches.fetch_add(1, Ordering::Relaxed) < PRINTED_FAILURES
        {
            eprintln!(
                "last-hash mismatch: {worker:?} scored {score}, expected {:?}, got {:?}",
                ExternalSequenceBlockHash(tail),
                last.get(&worker)
            );
        }
    }
}

/// A pool sequence or a published decode sequence, truncated to a random length and
/// sometimes followed by a block no rank stores.
fn random_query(shared: &Shared, rng: &mut Rng) -> Vec<u64> {
    let doc_len = shared.config.doc_len;
    let mut q = if rng.chance(30) {
        let queries = shared.queries.read();
        if queries.is_empty() {
            random_pool_seq(doc_len, rng)
        } else {
            queries[rng.below(queries.len() as u64) as usize].clone()
        }
    } else {
        random_pool_seq(doc_len, rng)
    };
    let len = 1 + rng.below(q.len() as u64) as usize;
    q.truncate(len);
    if rng.chance(10) {
        q.push(rng.next());
    }
    q
}

fn quiescent_parity(shared: &Shared, rng: &mut Rng) {
    let _paused = shared.gate.write();
    let models: Vec<_> = shared.models.iter().map(|m| m.lock()).collect();
    // Ranks of a removed worker are gone from the index before their writer drops them
    // from its model at its next batch.
    let retired = shared.retired.lock().clone();
    let mut queries: Vec<Vec<u64>> = models
        .iter()
        .flat_map(|m| m.values().flat_map(|w| w.live.iter().cloned()))
        .collect();
    queries.extend((0..256).map(|_| random_query(shared, rng)));
    for query in queries {
        let seqs = seq_hashes(&query);
        let locals: Vec<LocalBlockHash> = query.iter().copied().map(LocalBlockHash).collect();
        let got = shared.index.find_matches_impl(&locals, false).scores;
        let mut expected = FxHashMap::default();
        for m in &models {
            for (&worker, model) in m.iter().filter(|(worker, _)| !retired.contains(worker)) {
                let len = model.prefix_len(&seqs);
                if len > 0 {
                    expected.insert(worker, len as u32);
                }
            }
        }
        shared.parity_queries.fetch_add(1, Ordering::Relaxed);
        if got != expected
            && shared.parity_mismatches.fetch_add(1, Ordering::Relaxed) < PRINTED_FAILURES
        {
            let mut diff: Vec<_> = expected
                .keys()
                .chain(got.keys())
                .copied()
                .collect::<FxHashSet<_>>()
                .into_iter()
                .filter(|w| expected.get(w) != got.get(w))
                .map(|w| (w, expected.get(&w).copied(), got.get(&w).copied()))
                .collect();
            diff.sort_by_key(|d| (d.0.worker_id, d.0.dp_rank));
            eprintln!(
                "parity mismatch on a {}-block query, (rank, expected, got): {diff:?}",
                seqs.len()
            );
        }
    }
    shared.checks.fetch_add(1, Ordering::Relaxed);
}

#[test]
#[ignore = "long-running race soak; run explicitly"]
fn crtc_race_soak() {
    let config = Config::from_env();
    let seed = config.seed;
    let chaos = config.chaos;

    let mut owned: Vec<Vec<WorkerWithDpRank>> = vec![Vec::new(); config.writers];
    for w in 0..config.workers {
        let worker = WorkerWithDpRank::new(w / 2, (w % 2) as u32);
        owned[(mix(w) % config.writers as u64) as usize].push(worker);
    }
    owned.retain(|ws| !ws.is_empty());
    let shared = Arc::new(Shared {
        index: ConcurrentRadixTreeCompressed::new(),
        retired: Mutex::new(FxHashSet::default()),
        next_worker_id: AtomicU64::new(1 << 32),
        slot_owners: Mutex::new(FxHashMap::default()),
        rank_retires: AtomicU64::new(0),
        worker_retires: AtomicU64::new(0),
        recycled_slots: AtomicU64::new(0),
        ever: DashSet::new(),
        gate: RwLock::new(()),
        models: owned
            .iter()
            .map(|ws| Mutex::new(ws.iter().map(|&w| (w, WorkerModel::default())).collect()))
            .collect(),
        queries: RwLock::new(Vec::new()),
        stop: AtomicBool::new(false),
        events: AtomicU64::new(0),
        apply_errors: AtomicU64::new(0),
        reads: AtomicU64::new(0),
        overcounts: AtomicU64::new(0),
        hash_mismatches: AtomicU64::new(0),
        checks: AtomicU64::new(0),
        parity_queries: AtomicU64::new(0),
        parity_mismatches: AtomicU64::new(0),
        config,
    });

    {
        // Idle ranks that never store hold the low slots for the whole run.
        let guard = crossbeam_epoch::pin();
        for id in 0..shared.config.slot_offset {
            shared
                .index
                .slots
                .acquire(WorkerWithDpRank::new(u64::MAX - id, 0), &guard)
                .expect("SOAK_SLOT_OFFSET exceeds the slot capacity");
        }
    }

    let mut handles = Vec::new();
    for (lane, ws) in owned.into_iter().enumerate() {
        let shared = shared.clone();
        handles.push(thread::spawn(move || {
            let mut writer = Writer {
                shared: shared.clone(),
                lane,
                unslotted: ws.iter().copied().collect(),
                workers: ws,
                lookup: LaneLookup::default(),
                rng: Rng(mix(seed ^ (lane as u64 + 1))),
                next_id: 0,
            };
            let mut batches = 0u64;
            while !shared.stop.load(Ordering::Relaxed) {
                if writer.rng.below(1000) < shared.config.churn_per_mille {
                    writer.churn();
                }
                let _batch = shared.gate.read();
                writer.adopt_retirements();
                for _ in 0..BATCH_STEPS {
                    writer.step();
                }
                batches += 1;
                if batches.is_multiple_of(64) {
                    writer.lookup.assert_invariants();
                }
            }
            writer.lookup.assert_invariants();
        }));
    }
    for r in 0..shared.config.readers {
        let shared = shared.clone();
        handles.push(thread::spawn(move || {
            let mut rng = Rng(mix(seed ^ (0xABCD + r as u64)));
            while !shared.stop.load(Ordering::Relaxed) {
                let q = random_query(&shared, &mut rng);
                check_read(&shared, &q, &mut rng);
            }
        }));
    }
    {
        let shared = shared.clone();
        handles.push(thread::spawn(move || {
            while !shared.stop.load(Ordering::Relaxed) {
                {
                    let _batch = shared.gate.read();
                    shared.index.run_cleanup_for_test();
                }
                thread::sleep(Duration::from_millis(20));
            }
        }));
    }

    let mut rng = Rng(mix(seed ^ 0x5151));
    let deadline = Instant::now() + Duration::from_secs(shared.config.secs);
    while Instant::now() < deadline {
        thread::sleep(Duration::from_millis(shared.config.check_ms));
        if !chaos {
            quiescent_parity(&shared, &mut rng);
        }
    }
    shared.stop.store(true, Ordering::Relaxed);
    for handle in handles {
        handle.join().expect("soak thread panicked");
    }
    shared.index.run_cleanup_for_test();
    if !chaos {
        quiescent_parity(&shared, &mut rng);
    }

    let config = &shared.config;
    let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
    let (overcounts, hash_mismatches) = (load(&shared.overcounts), load(&shared.hash_mismatches));
    let (apply_errors, parity_mismatches) =
        (load(&shared.apply_errors), load(&shared.parity_mismatches));
    eprintln!(
        "SOAK mode={} secs={} writers={} readers={} workers={} seed={seed} check_ms={} \
         doc_len={} max_remove={} churn={} slot_offset={} events={} apply_errors={apply_errors} \
         reads={} overcounts={overcounts} hash_mismatches={hash_mismatches} checks={} \
         parity_queries={} parity_mismatches={parity_mismatches} edges={} rank_retires={} \
         worker_retires={} recycled_slots={}",
        if chaos { "chaos" } else { "strict" },
        config.secs,
        shared.models.len(),
        config.readers,
        config.workers,
        config.check_ms,
        config.doc_len,
        config.max_remove,
        config.churn_per_mille,
        config.slot_offset,
        load(&shared.events),
        load(&shared.reads),
        load(&shared.checks),
        load(&shared.parity_queries),
        shared.index.raw_child_edge_count(),
        load(&shared.rank_retires),
        load(&shared.worker_retires),
        load(&shared.recycled_slots),
    );
    assert_eq!(
        overcounts, 0,
        "lookups credited ranks past the blocks they ever stored"
    );
    assert_eq!(
        hash_mismatches, 0,
        "detailed lookups reported a last matched hash other than the scored prefix's tail"
    );
    if chaos {
        return;
    }
    assert_eq!(apply_errors, 0, "strict-mode events failed to apply");
    assert_eq!(
        parity_mismatches, 0,
        "quiescent lookups disagreed with the sequence-hash model"
    );
}
