// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in adversarial race soak for any [`HarnessBackend`], adapted from the CRTC soak in
//! #15614 (`concurrent_radix_tree_compressed/soak_tests.rs`).
//!
//! Each writer thread owns one event lane, a thread running the backend's
//! `SyncIndexer::worker` loop, and a disjoint set of ranks, as `ThreadPoolIndexer`
//! sticky routing would assign them. Writers replay adversarial per-rank streams over a
//! pool of heavily shared prefixes (system prompt, then document, then user turn): stores,
//! some split into two chained events; decode extensions, most of which share their tail
//! with other ranks; tail evictions whose blocks arrive in either order; and clears. Reader
//! threads look up concurrently through `SyncIndexer::find_matches` and, when the backend
//! has them, detailed lookups. A maintenance thread sends `CleanupStaleChildren` to a
//! random lane every 20 ms.
//!
//! Checks:
//! - Every lookup score, even mid-race, must stay within the blocks its rank has ever
//!   stored, and detailed lookups must report the scored prefix's last sequence hash.
//!   This catches credit for blocks a rank never stored, but not for blocks it has since
//!   evicted.
//! - In strict mode, every `SOAK_CHECK_MS` and once at the end, writers pause at a batch
//!   boundary with their lanes drained, and lookups over every live sequence plus random
//!   queries must match a per-rank model keyed by sequence hash exactly. Only these checks
//!   catch credit for evicted blocks.
//! - Every event is acknowledged by its lane; strict mode requires every one to apply.
//!
//! With `SOAK_CHURN`, writers also remove a rank (`RemoveWorkerDpRank` on its own lane,
//! racing the other lanes) or a whole worker (`RemoveWorker`: every other lane drops the
//! worker's ranks while all writers are paused, as `ThreadPoolIndexer`'s barrier does, then
//! the removing lane sweeps while the others keep applying events) and replace each
//! removed rank with a fresh one, so released slots are recycled to new ranks.
//! `SOAK_SLOT_OFFSET` first has that many idle ranks store one private block each, so with
//! 256 or more every live rank lands past CRTC's inline slot words.
//!
//! In strict mode a writer sends a batch's events without waiting and collects the
//! acknowledgements at the batch boundary; in chaos mode it waits for each one, because
//! failed stores change what its model may generate next.
//!
//! Modes (`SOAK_MODE`, or the test name):
//! - `strict` (default): streams an engine could emit. Every event must apply, and the
//!   quiescent parity checks run.
//! - `chaos`: a quarter of evictions remove a single block anywhere in the rank's cached
//!   chain. Later stores can fail and the model no longer predicts exact scores, so only
//!   the ever-stored and last-hash checks apply.
//!
//! Knobs (environment variables, default in parentheses):
//! - `SOAK_SECS` (10), `SOAK_WRITERS` (8), `SOAK_READERS` (4), `SOAK_WORKERS` (64 ranks,
//!   two dp ranks per worker id), `SOAK_SEED` (1), `SOAK_MODE` (`strict`),
//!   `SOAK_CHECK_MS` (500), `SOAK_DOC_LEN` (10), `SOAK_MAX_REMOVE` (4), `SOAK_CHURN` (0,
//!   per mille of batches that start with a removal), `SOAK_SLOT_OFFSET` (0).
//!
//! The run ends with one `SOAK` line on stderr: the configuration, then `events`,
//! `apply_errors`, `reads`, `overcounts`, `hash_mismatches`, `checks`, `parity_queries`,
//! `parity_mismatches`, `structure` (the backend's structural size after a final cleanup,
//! if it reports one), `rank_retires`, `worker_retires` and `recycled_slots`.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use dashmap::DashSet;
use parking_lot::{Mutex, RwLock, RwLockWriteGuard};
use rustc_hash::{FxHashMap, FxHashSet};
use tokio::sync::oneshot;

use super::workload::{hash_parts, mix};
use super::{HarnessBackend, env_u64, new_backend};
use crate::indexer::WorkerTask;
use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheStoreData, LocalBlockHash, RouterEvent,
    WorkerWithDpRank, compute_seq_hash_for_block,
};
use crate::test_utils::{remove_event, router_event, stored_blocks_with_sequence_hashes};

/// Pool shape: system prompts, documents per system prompt, user turns per document.
const SYSTEM_PROMPTS: u64 = 6;
const DOCUMENTS: u64 = 12;
const USER_TURNS: u64 = 6;

const BATCH_STEPS: usize = 64;
const QUERY_POOL: usize = 4096;
const PRINTED_FAILURES: u64 = 5;

pub(super) struct Config {
    pub(super) secs: u64,
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
    /// Knobs from the environment; `chaos` overrides `SOAK_MODE`.
    pub(super) fn from_env(chaos: Option<bool>) -> Self {
        let chaos = chaos.unwrap_or_else(|| match std::env::var("SOAK_MODE").as_deref() {
            Err(_) | Ok("strict") => false,
            Ok("chaos") => true,
            Ok(other) => panic!("SOAK_MODE must be strict or chaos, got {other:?}"),
        });
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

/// Local hashes of a pool sequence: system prompt, then document, then user turn, each
/// with a fixed length per id.
fn pool_seq(doc_len: u64, system: u64, doc: u64, turn: u64) -> Vec<LocalBlockHash> {
    let ls = 1 + hash_parts(&[1, system]) % 12;
    let ld = hash_parts(&[2, system, doc]) % (doc_len + 1);
    let lu = 1 + hash_parts(&[3, system, doc, turn]) % 8;
    let mut seq = Vec::with_capacity((ls + ld + lu) as usize);
    seq.extend((0..ls).map(|i| hash_parts(&[10, system, i])));
    seq.extend((0..ld).map(|i| hash_parts(&[11, system, doc, i])));
    seq.extend((0..lu).map(|i| hash_parts(&[12, system, doc, turn, i])));
    seq.into_iter().map(LocalBlockHash).collect()
}

fn random_pool_seq(doc_len: u64, rng: &mut fastrand::Rng) -> Vec<LocalBlockHash> {
    let system = rng.u64(..SYSTEM_PROMPTS);
    let doc = rng.u64(..DOCUMENTS);
    let turn = rng.u64(..USER_TURNS);
    pool_seq(doc_len, system, doc, turn)
}

/// One rank's cached blocks as its event stream implies them.
#[derive(Default)]
struct WorkerModel {
    /// Sequence hash -> (parent sequence hash, number of cached children).
    cached: FxHashMap<u64, (Option<u64>, u32)>,
    /// The most recent sequences this rank stored or extended, as local hashes.
    live: VecDeque<Vec<LocalBlockHash>>,
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

    fn push_live(&mut self, seq: Vec<LocalBlockHash>) {
        if self.live.len() >= 48 {
            self.live.pop_front();
        }
        self.live.push_back(seq);
    }
}

struct Shared<T: HarnessBackend> {
    backend: Arc<T>,
    /// One event lane per writer.
    lanes: Vec<flume::Sender<WorkerTask>>,
    config: Config,
    /// Ranks of a removed worker that their owning writer has not replaced yet.
    retired: Mutex<FxHashSet<WorkerWithDpRank>>,
    next_worker_id: AtomicU64,
    /// Rank last seen on each slot, to count slot reuse.
    slot_owners: Mutex<FxHashMap<usize, WorkerWithDpRank>>,
    rank_retires: AtomicU64,
    worker_retires: AtomicU64,
    recycled_slots: AtomicU64,
    /// Every (rank, sequence hash) a rank has stored, recorded before the store is sent.
    ever: DashSet<(WorkerWithDpRank, u64)>,
    /// Held shared by each writer batch, and exclusively by parity checks and
    /// whole-worker removals to pause the writers.
    gate: RwLock<()>,
    /// Each writer's rank models.
    models: Vec<Mutex<FxHashMap<WorkerWithDpRank, WorkerModel>>>,
    /// Extended decode sequences that readers sample as queries.
    queries: RwLock<Vec<Vec<LocalBlockHash>>>,
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

impl<T: HarnessBackend> Shared<T> {
    fn send(&self, lane: usize, task: WorkerTask) {
        self.lanes[lane]
            .send(task)
            .expect("soak event lane is gone (did it panic?)");
    }

    fn flush_lane(&self, lane: usize) {
        let (resp, rx) = oneshot::channel();
        self.send(lane, WorkerTask::Flush(resp));
        rx.blocking_recv().expect("soak event lane dropped a flush");
    }
}

/// An event in flight and what to print if it fails.
struct PendingAck {
    rx: oneshot::Receiver<bool>,
    worker: WorkerWithDpRank,
    event_id: u64,
    kind: &'static str,
}

struct Writer<T: HarnessBackend> {
    shared: Arc<Shared<T>>,
    /// This writer's lane, and its index into `Shared::models`.
    lane: usize,
    workers: Vec<WorkerWithDpRank>,
    rng: fastrand::Rng,
    next_id: u64,
    pending: Vec<PendingAck>,
    /// Ranks whose slot has not been recorded in `slot_owners` yet.
    unslotted: FxHashSet<WorkerWithDpRank>,
}

impl<T: HarnessBackend> Writer<T> {
    fn record_failure(&self, worker: WorkerWithDpRank, event_id: u64, kind: &str) {
        let errors = self.shared.apply_errors.fetch_add(1, Ordering::Relaxed);
        if !self.shared.config.chaos && errors < PRINTED_FAILURES {
            eprintln!("apply error: {worker:?} event {event_id} ({kind})");
        }
    }

    /// Sends `event`. In chaos mode waits for it and returns whether it applied; in strict
    /// mode returns true and checks the acknowledgement at the batch boundary.
    fn apply(&mut self, event: RouterEvent, kind: &'static str) -> bool {
        self.next_id += 1;
        self.shared.events.fetch_add(1, Ordering::Relaxed);
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let event_id = event.event.event_id;
        let (resp, rx) = oneshot::channel();
        self.shared
            .send(self.lane, WorkerTask::EventWithAck { event, resp });
        if !self.shared.config.chaos {
            self.pending.push(PendingAck {
                rx,
                worker,
                event_id,
                kind,
            });
            return true;
        }
        let applied = rx.blocking_recv().unwrap_or(false);
        if !applied {
            self.record_failure(worker, event_id, kind);
        }
        applied
    }

    /// Waits for every acknowledgement this writer has outstanding.
    fn drain(&mut self) {
        for ack in std::mem::take(&mut self.pending) {
            if !ack.rx.blocking_recv().unwrap_or(false) {
                self.record_failure(ack.worker, ack.event_id, ack.kind);
            }
        }
    }

    fn record_slot(&mut self, worker: WorkerWithDpRank) {
        if !self.unslotted.contains(&worker) {
            return;
        }
        let Some(slot) = self.shared.backend.harness_rank_slot(worker) else {
            return;
        };
        self.unslotted.remove(&worker);
        if self
            .shared
            .slot_owners
            .lock()
            .insert(slot, worker)
            .is_some_and(|previous| previous != worker)
        {
            self.shared.recycled_slots.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn store(
        &mut self,
        model: &mut WorkerModel,
        worker: WorkerWithDpRank,
        locals: &[LocalBlockHash],
        from: usize,
        to: usize,
    ) {
        let seqs = compute_seq_hash_for_block(&locals[..to]);
        for &s in &seqs[from..to] {
            self.shared.ever.insert((worker, s));
        }
        let parent = (from > 0).then(|| seqs[from - 1]);
        let event = router_event(
            worker.worker_id,
            self.next_id,
            worker.dp_rank,
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent.map(ExternalSequenceBlockHash),
                start_position: None,
                blocks: stored_blocks_with_sequence_hashes(&locals[from..to], &seqs[from..to]),
            }),
        );
        if !self.apply(event, "store") {
            return;
        }
        let mut p = parent;
        for &s in &seqs[from..to] {
            model.insert(p, s);
            p = Some(s);
        }
    }

    /// Retires one of this writer's ranks, or every rank of its worker, and re-adds each
    /// under a fresh worker id with an empty model.
    fn churn(&mut self) {
        let pick = self.rng.usize(..self.workers.len());
        let shared = self.shared.clone();
        if self.rng.bool() {
            // RemoveWorkerDpRank on the rank's own lane, racing every other lane. Adopting
            // first keeps it off a rank whose worker another lane already removed.
            let _batch = shared.gate.read();
            self.adopt_retirements();
            let worker = self.workers[pick];
            shared.send(
                self.lane,
                WorkerTask::RemoveWorkerDpRank {
                    worker_id: worker.worker_id,
                    dp_rank: worker.dp_rank,
                    sweep_tree: true,
                },
            );
            shared.flush_lane(self.lane);
            self.replace_rank(worker);
            shared.rank_retires.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // RemoveWorker: with every writer paused at a batch boundary and its lane drained,
        // every other lane drops the worker's ranks, as `ThreadPoolIndexer`'s broadcast
        // does before its sweep. This lane then sweeps while the others resume.
        let paused = shared.gate.write();
        self.adopt_retirements();
        let worker = self.workers[pick];
        let mut acks = Vec::new();
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
                let (resp, rx) = oneshot::channel();
                shared.send(
                    lane,
                    WorkerTask::RemoveWorker {
                        worker_id: worker.worker_id,
                        sweep_tree: false,
                        resp,
                    },
                );
                acks.push(rx);
            }
        }
        for rx in acks {
            rx.blocking_recv()
                .expect("soak event lane dropped a removal acknowledgement");
        }
        // Downgrading atomically keeps parity checks out until the sweep is done.
        let _batch = RwLockWriteGuard::downgrade(paused);
        let (resp, rx) = oneshot::channel();
        shared.send(
            self.lane,
            WorkerTask::RemoveWorker {
                worker_id: worker.worker_id,
                sweep_tree: true,
                resp,
            },
        );
        rx.blocking_recv()
            .expect("soak event lane dropped a removal acknowledgement");
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

    /// Replaces this writer's ranks of workers another writer removed. Their lane already
    /// dropped them during the removal.
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
        let worker = self.workers[self.rng.usize(..self.workers.len())];
        let shared = self.shared.clone();
        let config = &shared.config;
        let mut models = shared.models[self.lane].lock();
        let model = models
            .get_mut(&worker)
            .expect("every writer rank has a model");
        let roll = self.rng.u32(..1000);

        if roll < 450 {
            // Request store: extend this rank's cached prefix of a pool sequence, sometimes
            // as two chained events.
            let seq = random_pool_seq(config.doc_len, &mut self.rng);
            let target = self.rng.usize(1..=seq.len());
            let seqs = compute_seq_hash_for_block(&seq);
            let cached = model.prefix_len(&seqs[..target]);
            if cached < target {
                if target - cached >= 2 && self.rng.u32(..100) < 30 {
                    let mid = self.rng.usize(cached + 1..target);
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
            let idx = self.rng.usize(..model.live.len());
            let mut seq = model.live[idx].clone();
            let seqs = compute_seq_hash_for_block(&seq);
            if model.prefix_len(&seqs) != seq.len() {
                return;
            }
            let tail = *seqs.last().unwrap();
            // A third of the extensions are unique to the rank; the rest take one of two
            // tails every rank shares, so ranks race to extend and split the same leaf.
            let variant = match self.rng.u64(..3) {
                0 => worker.worker_id * 4 + worker.dp_rank as u64 + 100,
                v => v,
            };
            let start = seq.len();
            let m = self.rng.u64(1..=4);
            seq.extend((0..m).map(|i| LocalBlockHash(hash_parts(&[20, tail, variant, i]))));
            self.store(model, worker, &seq, start, seq.len());
            if self.rng.u32(..100) < 5 {
                let mut queries = shared.queries.write();
                let victim = self.rng.usize(..QUERY_POOL);
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
            let idx = self.rng.usize(..model.live.len());
            let seq = model.live[idx].clone();
            let seqs = compute_seq_hash_for_block(&seq);
            let cached = model.prefix_len(&seqs);
            if cached == 0 {
                return;
            }
            let mut removed = Vec::new();
            if config.chaos && self.rng.u32(..100) < 25 {
                // Mid-chain eviction: the rank keeps the blocks after it.
                let pos = self.rng.usize(..cached);
                removed.push(seqs[pos]);
            } else {
                // Tail eviction, stopping at a block that has other cached children.
                let want = self.rng.usize(1..=config.max_remove.max(1) as usize);
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
            if self.rng.bool() {
                removed.reverse();
            }
            let event = remove_event(
                worker.worker_id,
                self.next_id,
                worker.dp_rank,
                removed
                    .iter()
                    .copied()
                    .map(ExternalSequenceBlockHash)
                    .collect(),
            );
            if self.apply(event, "removal") {
                // Either order keeps the child counts right: a parent removed first is
                // skipped when its child is removed.
                for s in removed {
                    model.remove(s);
                }
            }
        } else {
            let event = router_event(
                worker.worker_id,
                self.next_id,
                worker.dp_rank,
                KvCacheEventData::Cleared,
            );
            if self.apply(event, "clear") {
                model.cached.clear();
                model.live.clear();
            }
        }
    }
}

fn check_read<T: HarnessBackend>(
    shared: &Shared<T>,
    query: &[LocalBlockHash],
    rng: &mut fastrand::Rng,
) {
    let seqs = compute_seq_hash_for_block(query);
    let details = (rng.u32(..100) < 25)
        .then(|| shared.backend.harness_match_details(query))
        .flatten();
    let (scores, last) = match details {
        Some(d) => (d.overlap_scores.scores, Some(d.last_matched_hashes)),
        None => (shared.backend.find_matches(query, false).scores, None),
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
fn random_query<T: HarnessBackend>(
    shared: &Shared<T>,
    rng: &mut fastrand::Rng,
) -> Vec<LocalBlockHash> {
    let doc_len = shared.config.doc_len;
    let mut q = if rng.u32(..100) < 30 {
        let queries = shared.queries.read();
        if queries.is_empty() {
            random_pool_seq(doc_len, rng)
        } else {
            queries[rng.usize(..queries.len())].clone()
        }
    } else {
        random_pool_seq(doc_len, rng)
    };
    let len = rng.usize(1..=q.len());
    q.truncate(len);
    if rng.u32(..100) < 10 {
        q.push(LocalBlockHash(rng.u64(..)));
    }
    q
}

fn quiescent_parity<T: HarnessBackend>(shared: &Shared<T>, rng: &mut fastrand::Rng) {
    let _paused = shared.gate.write();
    let models: Vec<_> = shared.models.iter().map(|m| m.lock()).collect();
    // Ranks of a removed worker are gone from the index before their writer drops them
    // from its model at its next batch.
    let retired = shared.retired.lock().clone();
    let mut queries: Vec<Vec<LocalBlockHash>> = models
        .iter()
        .flat_map(|m| m.values().flat_map(|w| w.live.iter().cloned()))
        .collect();
    queries.extend((0..256).map(|_| random_query(shared, rng)));
    for query in queries {
        let seqs = compute_seq_hash_for_block(&query);
        let got = shared.backend.find_matches(&query, false).scores;
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

/// Parks `count` idle ranks on the lowest slots: each stores one private block on lane 0.
fn park_idle_ranks<T: HarnessBackend>(shared: &Shared<T>, count: u64) {
    let mut acks = Vec::new();
    for id in 0..count {
        let rank = WorkerWithDpRank::new(u64::MAX - id, 0);
        let local = LocalBlockHash(hash_parts(&[99, id]));
        let seqs = compute_seq_hash_for_block(&[local]);
        let event = router_event(
            rank.worker_id,
            0,
            rank.dp_rank,
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: stored_blocks_with_sequence_hashes(&[local], &seqs),
            }),
        );
        let (resp, rx) = oneshot::channel();
        shared.send(0, WorkerTask::EventWithAck { event, resp });
        acks.push(rx);
    }
    for rx in acks {
        assert!(
            rx.blocking_recv().unwrap_or(false),
            "SOAK_SLOT_OFFSET exceeds the backend's rank capacity"
        );
    }
}

pub(crate) fn run<T: HarnessBackend>(backend_name: &str, chaos: Option<bool>) {
    run_with::<T>(backend_name, Config::from_env(chaos));
}

pub(super) fn run_with<T: HarnessBackend>(backend_name: &str, config: Config) {
    let seed = config.seed;
    let chaos = config.chaos;

    let mut owned: Vec<Vec<WorkerWithDpRank>> = vec![Vec::new(); config.writers];
    for w in 0..config.workers {
        let worker = WorkerWithDpRank::new(w / 2, (w % 2) as u32);
        owned[(mix(w) % config.writers as u64) as usize].push(worker);
    }
    owned.retain(|ws| !ws.is_empty());

    let backend = new_backend::<T>();
    let mut lane_threads = Vec::new();
    let mut lanes = Vec::new();
    for _ in 0..owned.len() {
        let (tx, rx) = flume::unbounded();
        let lane_backend = Arc::clone(&backend);
        lane_threads.push(thread::spawn(move || {
            lane_backend
                .worker(rx, None)
                .expect("soak event lane exited with an error");
        }));
        lanes.push(tx);
    }

    let shared = Arc::new(Shared {
        backend,
        lanes,
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

    park_idle_ranks(&shared, shared.config.slot_offset);

    let mut handles = Vec::new();
    for (lane, ws) in owned.into_iter().enumerate() {
        let shared = shared.clone();
        handles.push(thread::spawn(move || {
            let mut writer = Writer {
                shared: shared.clone(),
                lane,
                unslotted: ws.iter().copied().collect(),
                workers: ws,
                rng: fastrand::Rng::with_seed(mix(seed ^ (lane as u64 + 1))),
                next_id: 0,
                pending: Vec::new(),
            };
            while !shared.stop.load(Ordering::Relaxed) {
                if writer.rng.u64(..1000) < shared.config.churn_per_mille {
                    writer.churn();
                }
                let _batch = shared.gate.read();
                writer.adopt_retirements();
                for _ in 0..BATCH_STEPS {
                    writer.step();
                }
                writer.drain();
                let workers = writer.workers.clone();
                for worker in workers {
                    writer.record_slot(worker);
                }
            }
        }));
    }
    for r in 0..shared.config.readers {
        let shared = shared.clone();
        handles.push(thread::spawn(move || {
            let mut rng = fastrand::Rng::with_seed(mix(seed ^ (0xABCD + r as u64)));
            while !shared.stop.load(Ordering::Relaxed) {
                let q = random_query(&shared, &mut rng);
                check_read(&shared, &q, &mut rng);
            }
        }));
    }
    {
        let shared = shared.clone();
        handles.push(thread::spawn(move || {
            let mut rng = fastrand::Rng::with_seed(mix(seed ^ 0xC1EA));
            while !shared.stop.load(Ordering::Relaxed) {
                let lane = rng.usize(..shared.lanes.len());
                shared.send(lane, WorkerTask::CleanupStaleChildren);
                thread::sleep(Duration::from_millis(20));
            }
        }));
    }

    let mut rng = fastrand::Rng::with_seed(mix(seed ^ 0x5151));
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
    shared.send(0, WorkerTask::CleanupStaleChildren);
    for lane in 0..shared.lanes.len() {
        shared.flush_lane(lane);
    }
    if !chaos {
        quiescent_parity(&shared, &mut rng);
    }

    let config = &shared.config;
    let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
    let (overcounts, hash_mismatches) = (load(&shared.overcounts), load(&shared.hash_mismatches));
    let (apply_errors, parity_mismatches) =
        (load(&shared.apply_errors), load(&shared.parity_mismatches));
    let structure = shared
        .backend
        .harness_structure_size()
        .map_or_else(|| "n/a".to_string(), |size| size.to_string());
    eprintln!(
        "SOAK backend={backend_name} mode={} secs={} writers={} readers={} workers={} \
         seed={seed} check_ms={} doc_len={} max_remove={} churn={} slot_offset={} events={} \
         apply_errors={apply_errors} reads={} overcounts={overcounts} \
         hash_mismatches={hash_mismatches} checks={} parity_queries={} \
         parity_mismatches={parity_mismatches} structure={structure} rank_retires={} \
         worker_retires={} recycled_slots={}",
        if chaos { "chaos" } else { "strict" },
        config.secs,
        shared.lanes.len(),
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
        load(&shared.rank_retires),
        load(&shared.worker_retires),
        load(&shared.recycled_slots),
    );

    for lane in &shared.lanes {
        let _ = lane.send(WorkerTask::Terminate);
    }
    for lane_thread in lane_threads {
        lane_thread.join().expect("soak event lane panicked");
    }

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
