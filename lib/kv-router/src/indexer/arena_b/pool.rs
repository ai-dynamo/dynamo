// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The work-stealing lane pool (spec B 8). `ThreadPoolIndexer` is unchanged: it still
//! routes each rank stickily to one lane's channel and calls `worker` on each lane. Here
//! each lane pumps its channel into per-rank mailboxes ([`RankCell`]) owned by the backend,
//! serves its own ready list, and steals whole ranks from busy lanes. A rank's block map
//! lives in its cell, so it moves with the rank.
//!
//! Lane-wide tasks become barriers: a lane records, for every rank it has fed, how many
//! tasks it had fed, stops pumping, keeps serving and stealing, and completes the task once
//! every such rank has applied that many.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crossbeam_utils::CachePadded;
use dashmap::DashMap;
use parking_lot::Mutex;
use rustc_hash::{FxBuildHasher, FxHashMap};
use tokio::sync::oneshot;

use super::ArenaIndex;
use super::protocol::Mailbox;
use super::runs::bump;
use super::slots::RemovalTarget;
use super::write::RankState;
#[cfg(feature = "bench")]
use crate::indexer::{EventCompletionWriter, ObservationSeal, WorkerObservationState};
use crate::indexer::{
    KvIndexerMetrics, KvRouterError, PreBoundEventCounters, WorkerLookupStats, WorkerTask,
};
use crate::protocols::{ExternalSequenceBlockHash, RouterEvent, WorkerId, WorkerWithDpRank};

/// Lanes one backend can host at once.
pub(crate) const MAX_LANES: usize = 256;
const MASK_WORDS: usize = MAX_LANES / 64;
/// Channel tasks a lane pumps before it serves a cell again.
const PUMP_BATCH: usize = 64;
/// The safety timeout of a parked lane.
const PARK_TIMEOUT: Duration = Duration::from_millis(10);
/// How long a lane waiting on a barrier parks between checks.
const BARRIER_PARK: Duration = Duration::from_micros(200);

/// One task for one rank.
pub(crate) enum RankTask {
    Event {
        event: RouterEvent,
        ack: Option<oneshot::Sender<bool>>,
    },
    #[cfg(feature = "bench")]
    Observed {
        event: RouterEvent,
        correlation_id: u32,
        /// The lane whose queue the event came through; its completion buffer records it.
        producer: usize,
    },
    Contains {
        hash: ExternalSequenceBlockHash,
        resp: oneshot::Sender<bool>,
    },
    /// `RemoveWorkerDpRank`, or one rank of a `RemoveWorker`.
    Remove,
    Anchor,
}

/// A rank's mailbox and state (8.1). Cells are created on a rank's first task and kept for
/// the index's lifetime.
pub(crate) struct RankCell {
    pub(crate) rank: WorkerWithDpRank,
    mailbox: Mailbox<RankTask>,
    /// Touched only by the lane that moved the cell to `RUNNING`; the mutex is therefore
    /// always uncontended, and a failed `try_lock` is a scheduler-invariant violation.
    pub(crate) data: Mutex<RankState>,
    /// The lane that created the cell; `Stats` and `RemoveWorker` act on a lane's home cells.
    home: usize,
    fed: AtomicU64,
    applied: AtomicU64,
    pub(crate) blocks: AtomicUsize,
}

impl RankCell {
    /// A cell no lane owns, for ranks driven on the caller's thread.
    pub(crate) fn new_detached(rank: WorkerWithDpRank) -> Self {
        Self::new(rank, usize::MAX)
    }

    fn new(rank: WorkerWithDpRank, home: usize) -> Self {
        Self {
            rank,
            mailbox: Mailbox::default(),
            data: Mutex::new(RankState::new(rank)),
            home,
            fed: AtomicU64::new(0),
            applied: AtomicU64::new(0),
            blocks: AtomicUsize::new(0),
        }
    }

    pub(crate) fn pending(&self) -> u64 {
        self.fed
            .load(Ordering::Acquire)
            .saturating_sub(self.applied.load(Ordering::Acquire))
    }
}

/// What other lanes see of a lane.
pub(crate) struct LaneShared {
    alive: AtomicBool,
    ready: CachePadded<Mutex<VecDeque<Arc<RankCell>>>>,
    depth: AtomicUsize,
    wake_tx: flume::Sender<()>,
    wake_rx: flume::Receiver<()>,
    #[cfg(feature = "bench")]
    observation: Mutex<WorkerObservationState>,
    /// Mailbox backlog of the ranks this lane fed, when it pumped `SealObservation`.
    backlog_at_seal: AtomicU64,
    steals: AtomicU64,
    inline: AtomicU64,
}

impl LaneShared {
    fn new() -> Self {
        let (wake_tx, wake_rx) = flume::bounded(1);
        Self {
            alive: AtomicBool::new(false),
            ready: CachePadded::new(Mutex::new(VecDeque::new())),
            depth: AtomicUsize::new(0),
            wake_tx,
            wake_rx,
            #[cfg(feature = "bench")]
            observation: Mutex::new(WorkerObservationState::default()),
            backlog_at_seal: AtomicU64::new(0),
            steals: AtomicU64::new(0),
            inline: AtomicU64::new(0),
        }
    }

    fn push_ready(&self, cell: Arc<RankCell>) -> usize {
        let mut ready = self.ready.lock();
        ready.push_back(cell);
        let depth = ready.len();
        self.depth.store(depth, Ordering::SeqCst);
        depth
    }

    fn pop_ready(&self) -> Option<Arc<RankCell>> {
        if self.depth.load(Ordering::Acquire) == 0 {
            return None;
        }
        let mut ready = self.ready.lock();
        let cell = ready.pop_front();
        self.depth.store(ready.len(), Ordering::SeqCst);
        cell
    }
}

/// Pool-wide state.
pub(crate) struct Pool {
    pub(crate) cells: DashMap<WorkerWithDpRank, Arc<RankCell>, FxBuildHasher>,
    lanes: Box<[OnceLock<Arc<LaneShared>>]>,
    /// Lane indices ever registered; registered lanes are `0..high`.
    high: AtomicUsize,
    free: Mutex<Vec<usize>>,
    parked: [AtomicU64; MASK_WORDS],
    barrier_parked: [AtomicU64; MASK_WORDS],
}

impl Default for Pool {
    fn default() -> Self {
        Self {
            cells: DashMap::with_hasher(FxBuildHasher),
            lanes: (0..MAX_LANES).map(|_| OnceLock::new()).collect(),
            high: AtomicUsize::new(0),
            free: Mutex::new(Vec::new()),
            parked: Default::default(),
            barrier_parked: Default::default(),
        }
    }
}

impl Pool {
    fn register(&self) -> anyhow::Result<(usize, Arc<LaneShared>)> {
        let index = match self.free.lock().pop() {
            Some(index) => index,
            None => {
                let index = self.high.fetch_add(1, Ordering::AcqRel);
                if index >= MAX_LANES {
                    self.high.fetch_sub(1, Ordering::AcqRel);
                    anyhow::bail!("arena index hosts at most {MAX_LANES} lanes");
                }
                index
            }
        };
        let lane = self.lanes[index]
            .get_or_init(|| Arc::new(LaneShared::new()))
            .clone();
        lane.alive.store(true, Ordering::Release);
        Ok((index, lane))
    }

    fn unregister(&self, index: usize) {
        if let Some(lane) = self.lanes[index].get() {
            lane.alive.store(false, Ordering::Release);
        }
        self.set_bit(&self.parked, index, false);
        self.set_bit(&self.barrier_parked, index, false);
        self.free.lock().push(index);
    }

    pub(crate) fn lane(&self, index: usize) -> Option<&Arc<LaneShared>> {
        self.lanes.get(index)?.get()
    }

    fn set_bit(&self, mask: &[AtomicU64; MASK_WORDS], index: usize, on: bool) {
        let bit = 1u64 << (index % 64);
        if on {
            mask[index / 64].fetch_or(bit, Ordering::SeqCst);
        } else {
            mask[index / 64].fetch_and(!bit, Ordering::SeqCst);
        }
    }

    /// Wakes the lowest parked lane other than `me` so it can steal.
    fn wake_one(&self, me: usize) {
        for (w, word) in self.parked.iter().enumerate() {
            let mut bits = word.load(Ordering::SeqCst);
            while bits != 0 {
                let index = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if index == me {
                    continue;
                }
                let bit = 1u64 << (index % 64);
                if word.fetch_and(!bit, Ordering::SeqCst) & bit == 0 {
                    continue;
                }
                if let Some(lane) = self.lane(index) {
                    let _ = lane.wake_tx.try_send(());
                }
                return;
            }
        }
    }

    /// Wakes every lane parked on a barrier.
    fn wake_barrier_waiters(&self) {
        for (w, word) in self.barrier_parked.iter().enumerate() {
            if word.load(Ordering::SeqCst) == 0 {
                continue;
            }
            let mut bits = word.swap(0, Ordering::SeqCst);
            while bits != 0 {
                let index = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if let Some(lane) = self.lane(index) {
                    let _ = lane.wake_tx.try_send(());
                }
            }
        }
    }

    /// Takes the oldest ready cell of another lane, starting at a rotating index.
    fn steal(&self, me: usize, cursor: &mut usize) -> Option<Arc<RankCell>> {
        let lanes = self.high.load(Ordering::Acquire).min(MAX_LANES);
        for step in 0..lanes {
            let index = (*cursor + step) % lanes;
            if index == me {
                continue;
            }
            let Some(lane) = self.lane(index) else {
                continue;
            };
            if lane.depth.load(Ordering::SeqCst) == 0 {
                continue;
            }
            if let Some(cell) = lane.pop_ready() {
                *cursor = index + 1;
                return Some(cell);
            }
        }
        None
    }

    fn any_ready(&self, me: usize) -> bool {
        let lanes = self.high.load(Ordering::Acquire).min(MAX_LANES);
        (0..lanes).any(|index| {
            index != me
                && self
                    .lane(index)
                    .is_some_and(|lane| lane.depth.load(Ordering::SeqCst) > 0)
        })
    }

    /// Tasks fed to `lane`'s home ranks and not applied yet.
    pub(crate) fn pending_events(&self, lane: usize) -> u64 {
        self.cells
            .iter()
            .filter(|cell| cell.home == lane)
            .map(|cell| cell.pending())
            .sum()
    }

    pub(crate) fn steals(&self) -> (u64, u64) {
        let lanes = self.high.load(Ordering::Acquire).min(MAX_LANES);
        (0..lanes)
            .filter_map(|index| self.lane(index))
            .fold((0, 0), |(steals, inline), lane| {
                (
                    steals + lane.steals.load(Ordering::Relaxed),
                    inline + lane.inline.load(Ordering::Relaxed),
                )
            })
    }

    pub(crate) fn backlog_at_seal(&self) -> Vec<u64> {
        let lanes = self.high.load(Ordering::Acquire).min(MAX_LANES);
        (0..lanes)
            .filter_map(|index| self.lane(index))
            .map(|lane| lane.backlog_at_seal.load(Ordering::Relaxed))
            .collect()
    }
}

/// A lane-wide task waiting for the ranks its lane fed.
enum BarrierTask {
    Flush(oneshot::Sender<()>),
    Dump(oneshot::Sender<anyhow::Result<Vec<RouterEvent>>>),
    Stats(oneshot::Sender<WorkerLookupStats>),
    RemoveWorker {
        worker_id: WorkerId,
        sweep_tree: bool,
        resp: oneshot::Sender<()>,
    },
    #[cfg(feature = "bench")]
    Install {
        writer: EventCompletionWriter,
        resp: oneshot::Sender<bool>,
    },
    #[cfg(feature = "bench")]
    Seal(oneshot::Sender<Option<ObservationSeal>>),
    #[cfg(feature = "bench")]
    Harvest(oneshot::Sender<crate::indexer::EventCompletionBuffer>),
    Terminate,
}

struct Barrier {
    targets: Vec<(Arc<RankCell>, u64)>,
    task: BarrierTask,
}

impl Barrier {
    fn done(&mut self) -> bool {
        self.targets
            .retain(|(cell, target)| cell.applied.load(Ordering::Acquire) < *target);
        self.targets.is_empty()
    }
}

/// One lane's private state.
struct Lane {
    index: usize,
    shared: Arc<LaneShared>,
    /// Cells this lane has fed, for barriers.
    fed: Vec<Arc<RankCell>>,
    cache: FxHashMap<WorkerWithDpRank, Arc<RankCell>>,
    barrier: Option<Barrier>,
    counters: Option<PreBoundEventCounters>,
    cursor: usize,
    exit: bool,
}

enum Woken {
    Task(Box<WorkerTask>),
    Disconnected,
    Nothing,
}

impl ArenaIndex {
    /// The body of `SyncIndexer::worker`.
    pub(crate) fn run_lane(
        &self,
        rx: flume::Receiver<WorkerTask>,
        metrics: Option<Arc<KvIndexerMetrics>>,
    ) -> anyhow::Result<()> {
        let (index, shared) = self.pool.register()?;
        let mut lane = Lane {
            index,
            shared,
            fed: Vec::new(),
            cache: FxHashMap::default(),
            barrier: None,
            counters: metrics.as_ref().map(|m| m.prebind()),
            cursor: index + 1,
            exit: false,
        };
        while !lane.exit {
            self.lane_step(&mut lane, &rx);
        }
        self.pool.unregister(index);
        tracing::debug!(lane = index, "arena index lane shutting down");
        Ok(())
    }

    fn lane_step(&self, lane: &mut Lane, rx: &flume::Receiver<WorkerTask>) {
        // 1. Pump.
        if lane.barrier.is_none() {
            for _ in 0..PUMP_BATCH {
                match rx.try_recv() {
                    Ok(task) => {
                        let backlog = !rx.is_empty();
                        self.dispatch(lane, task, backlog);
                        if lane.barrier.is_some() {
                            break;
                        }
                    }
                    Err(flume::TryRecvError::Empty) => break,
                    Err(flume::TryRecvError::Disconnected) => {
                        self.begin_barrier(lane, BarrierTask::Terminate);
                        break;
                    }
                }
            }
        }
        // 2. Serve the oldest cell on this lane's list.
        if let Some(cell) = lane.shared.pop_ready() {
            self.serve(lane, cell);
            return;
        }
        // 3. Steal.
        if self.config.steal
            && let Some(cell) = self.pool.steal(lane.index, &mut lane.cursor)
        {
            lane.shared.steals.fetch_add(1, Ordering::Relaxed);
            self.serve(lane, cell);
            return;
        }
        // 4. Complete a pending barrier.
        if let Some(barrier) = lane.barrier.as_mut()
            && barrier.done()
        {
            let barrier = lane.barrier.take().expect("checked above");
            self.complete(lane, barrier.task);
            return;
        }
        // 5. Park.
        if lane.barrier.is_some() {
            self.park_on_barrier(lane);
        } else {
            match self.park(lane, rx) {
                Woken::Task(task) => {
                    let backlog = !rx.is_empty();
                    self.dispatch(lane, *task, backlog);
                }
                Woken::Disconnected => self.begin_barrier(lane, BarrierTask::Terminate),
                Woken::Nothing => {}
            }
        }
    }

    fn park(&self, lane: &Lane, rx: &flume::Receiver<WorkerTask>) -> Woken {
        self.pool.set_bit(&self.pool.parked, lane.index, true);
        // Re-check after advertising, so a producer that pushed before it saw the bit is
        // not missed (both sides use SeqCst).
        let busy = lane.shared.depth.load(Ordering::SeqCst) > 0
            || (self.config.steal && self.pool.any_ready(lane.index))
            || !rx.is_empty();
        let woken = if busy {
            Woken::Nothing
        } else {
            flume::Selector::new()
                .recv(rx, |task| match task {
                    Ok(task) => Woken::Task(Box::new(task)),
                    Err(_) => Woken::Disconnected,
                })
                .recv(&lane.shared.wake_rx, |_| Woken::Nothing)
                .wait_timeout(PARK_TIMEOUT)
                .unwrap_or(Woken::Nothing)
        };
        self.pool.set_bit(&self.pool.parked, lane.index, false);
        woken
    }

    fn park_on_barrier(&self, lane: &mut Lane) {
        self.pool
            .set_bit(&self.pool.barrier_parked, lane.index, true);
        let done = lane.barrier.as_mut().is_some_and(Barrier::done);
        let busy = done
            || lane.shared.depth.load(Ordering::SeqCst) > 0
            || (self.config.steal && self.pool.any_ready(lane.index));
        if !busy {
            let _ = lane.shared.wake_rx.recv_timeout(BARRIER_PARK);
        }
        self.pool
            .set_bit(&self.pool.barrier_parked, lane.index, false);
    }

    fn cell(&self, lane: &mut Lane, rank: WorkerWithDpRank) -> Arc<RankCell> {
        if let Some(cell) = lane.cache.get(&rank) {
            return cell.clone();
        }
        let cell = self
            .pool
            .cells
            .entry(rank)
            .or_insert_with(|| Arc::new(RankCell::new(rank, lane.index)))
            .clone();
        lane.cache.insert(rank, cell.clone());
        lane.fed.push(cell.clone());
        cell
    }

    /// Feeds `task` to `rank`'s mailbox, or applies it inline when the rank is idle with an
    /// empty mailbox, this lane has nothing ready, and nothing else waits in its channel
    /// (8.1 inline fast path). With a backlog, tasks go through mailboxes so idle lanes can
    /// steal them.
    fn feed(&self, lane: &mut Lane, rank: WorkerWithDpRank, task: RankTask, backlog: bool) {
        let cell = self.cell(lane, rank);
        cell.fed.fetch_add(1, Ordering::AcqRel);
        if self.config.inline_fast_path
            && !backlog
            && lane.shared.depth.load(Ordering::Acquire) == 0
            && cell.mailbox.try_inline()
        {
            lane.shared.inline.fetch_add(1, Ordering::Relaxed);
            match cell.data.try_lock() {
                Some(mut data) => {
                    self.run_task(lane, &cell, &mut data, task);
                }
                None => bump(&self.runs.stats.claim_check_failures),
            }
            cell.applied.fetch_add(1, Ordering::AcqRel);
            self.release_cell(lane, &cell);
            return;
        }
        if cell.mailbox.push(task) {
            lane.shared.push_ready(cell);
            if self.config.steal {
                self.pool.wake_one(lane.index);
            }
        }
    }

    /// Runs up to a batch of a ready cell's tasks (8.1 owner protocol).
    fn serve(&self, lane: &mut Lane, cell: Arc<RankCell>) {
        if !cell.mailbox.take() {
            // Fix 8: a hard check, never a debug assertion. Skip the cell.
            bump(&self.runs.stats.claim_check_failures);
            tracing::error!(rank = ?cell.rank, "arena index: ready cell was not READY");
            return;
        }
        {
            let Some(mut data) = cell.data.try_lock() else {
                bump(&self.runs.stats.claim_check_failures);
                tracing::error!(rank = ?cell.rank, "arena index: running cell was locked");
                self.release_cell(lane, &cell);
                return;
            };
            for _ in 0..self.config.batch {
                let Some(task) = cell.mailbox.pop() else {
                    break;
                };
                self.run_task(lane, &cell, &mut data, task);
                cell.applied.fetch_add(1, Ordering::AcqRel);
            }
        }
        self.release_cell(lane, &cell);
    }

    fn release_cell(&self, lane: &mut Lane, cell: &Arc<RankCell>) {
        if cell.mailbox.release() {
            lane.shared.push_ready(cell.clone());
            if self.config.steal {
                self.pool.wake_one(lane.index);
            }
        }
        self.pool.wake_barrier_waiters();
    }

    fn run_task(&self, lane: &Lane, cell: &RankCell, data: &mut RankState, task: RankTask) {
        match task {
            RankTask::Event { event, ack } => {
                let result = self.apply_event(data, event, lane.counters.as_ref());
                if let Err(error) = &result {
                    tracing::warn!("Failed to apply event: {error:?}");
                }
                if let Some(ack) = ack {
                    let _ = ack.send(result.is_ok());
                }
            }
            #[cfg(feature = "bench")]
            RankTask::Observed {
                event,
                correlation_id,
                producer,
            } => {
                let result = self.apply_event(data, event, lane.counters.as_ref());
                if let Err(error) = &result {
                    tracing::warn!("Failed to apply event: {error:?}");
                }
                if let Some(producer) = self.pool.lane(producer) {
                    producer
                        .observation
                        .lock()
                        .record(correlation_id, result.is_ok());
                }
            }
            RankTask::Contains { hash, resp } => {
                let _ = resp.send(data.map.get(hash).is_some());
            }
            RankTask::Remove => self.remove_rank(data),
            RankTask::Anchor => {
                tracing::warn!(rank = ?cell.rank, "arena index does not support anchors yet");
            }
        }
        cell.blocks.store(data.map.len(), Ordering::Relaxed);
    }

    fn begin_barrier(&self, lane: &mut Lane, task: BarrierTask) {
        let targets = lane
            .fed
            .iter()
            .filter_map(|cell| {
                let target = cell.fed.load(Ordering::Acquire);
                (cell.applied.load(Ordering::Acquire) < target).then(|| (cell.clone(), target))
            })
            .collect();
        lane.barrier = Some(Barrier { targets, task });
    }

    fn dispatch(&self, lane: &mut Lane, task: WorkerTask, backlog: bool) {
        match task {
            WorkerTask::Event(event) => {
                let rank = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
                self.feed(lane, rank, RankTask::Event { event, ack: None }, backlog);
            }
            WorkerTask::EventWithAck { event, resp } => {
                let rank = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
                self.feed(
                    lane,
                    rank,
                    RankTask::Event {
                        event,
                        ack: Some(resp),
                    },
                    backlog,
                );
            }
            #[cfg(feature = "bench")]
            WorkerTask::ObservedEvent {
                event,
                correlation_id,
            } => {
                let rank = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
                let producer = lane.index;
                self.feed(
                    lane,
                    rank,
                    RankTask::Observed {
                        event,
                        correlation_id,
                        producer,
                    },
                    backlog,
                );
            }
            WorkerTask::Anchor { worker, .. } => self.feed(lane, worker, RankTask::Anchor, backlog),
            WorkerTask::ContainsWorkerBlock {
                worker,
                block_hash,
                resp,
            } => self.feed(
                lane,
                worker,
                RankTask::Contains {
                    hash: block_hash,
                    resp,
                },
                backlog,
            ),
            WorkerTask::RemoveWorkerDpRank {
                worker_id, dp_rank, ..
            } => self.feed(
                lane,
                WorkerWithDpRank::new(worker_id, dp_rank),
                RankTask::Remove,
                backlog,
            ),
            WorkerTask::RemoveWorker {
                worker_id,
                sweep_tree,
                resp,
            } => {
                let home: Vec<_> = self
                    .pool
                    .cells
                    .iter()
                    .filter(|cell| cell.rank.worker_id == worker_id && cell.home == lane.index)
                    .map(|cell| cell.rank)
                    .collect();
                for rank in home {
                    self.feed(lane, rank, RankTask::Remove, true);
                }
                self.begin_barrier(
                    lane,
                    BarrierTask::RemoveWorker {
                        worker_id,
                        sweep_tree,
                        resp,
                    },
                );
            }
            WorkerTask::CleanupStaleChildren => {
                crate::indexer::SyncIndexer::run_cleanup_task(self);
            }
            WorkerTask::DumpEvents(resp) => self.begin_barrier(lane, BarrierTask::Dump(resp)),
            WorkerTask::Stats(resp) => self.begin_barrier(lane, BarrierTask::Stats(resp)),
            WorkerTask::Flush(resp) => self.begin_barrier(lane, BarrierTask::Flush(resp)),
            WorkerTask::ApproximateLru(task) => {
                if let Some(response) = task.response {
                    let _ = response.send(Err(KvRouterError::Unsupported(
                        "the arena index does not support approximate LRU".to_string(),
                    )));
                }
            }
            #[cfg(feature = "bench")]
            WorkerTask::InstallObservation { writer, resp } => {
                self.begin_barrier(lane, BarrierTask::Install { writer, resp });
            }
            #[cfg(feature = "bench")]
            WorkerTask::SealObservation(resp) => {
                let backlog: u64 = lane.fed.iter().map(|cell| cell.pending()).sum();
                lane.shared
                    .backlog_at_seal
                    .store(backlog, Ordering::Relaxed);
                self.begin_barrier(lane, BarrierTask::Seal(resp));
            }
            #[cfg(feature = "bench")]
            WorkerTask::HarvestObservation(resp) => {
                self.begin_barrier(lane, BarrierTask::Harvest(resp));
            }
            WorkerTask::Terminate => self.begin_barrier(lane, BarrierTask::Terminate),
        }
    }

    fn complete(&self, lane: &mut Lane, task: BarrierTask) {
        match task {
            BarrierTask::Flush(resp) => {
                crossbeam_epoch::pin().flush();
                let _ = resp.send(());
            }
            BarrierTask::Dump(resp) => {
                let _ = resp.send(Ok(Vec::new()));
            }
            BarrierTask::Stats(resp) => {
                let blocks = self
                    .pool
                    .cells
                    .iter()
                    .filter(|cell| cell.home == lane.index)
                    .map(|cell| (cell.rank, cell.blocks.load(Ordering::Relaxed)))
                    .collect::<Vec<_>>();
                let _ = resp.send(WorkerLookupStats::from_worker_block_counts(blocks));
            }
            BarrierTask::RemoveWorker {
                worker_id,
                sweep_tree,
                resp,
            } => {
                if sweep_tree {
                    // Ranks of the worker still mapped belong to lanes that never got the
                    // removal; release their slots as CRTC's sweep does.
                    let leftover = {
                        let guard = crossbeam_epoch::pin();
                        !self
                            .slots
                            .table(&guard)
                            .mapped(RemovalTarget::Worker(worker_id))
                            .is_empty()
                    };
                    if leftover {
                        self.release_slots(RemovalTarget::Worker(worker_id));
                    }
                    self.slots
                        .wait_for_release(RemovalTarget::Worker(worker_id));
                }
                let _ = resp.send(());
            }
            #[cfg(feature = "bench")]
            BarrierTask::Install { writer, resp } => {
                lane.shared.observation.lock().install(writer, resp);
            }
            #[cfg(feature = "bench")]
            BarrierTask::Seal(resp) => lane.shared.observation.lock().seal(resp),
            #[cfg(feature = "bench")]
            BarrierTask::Harvest(resp) => lane.shared.observation.lock().harvest(resp),
            BarrierTask::Terminate => {
                if lane.shared.depth.load(Ordering::Acquire) > 0 {
                    // Serve what is still on this lane's list before leaving: only this
                    // lane pushes to it, so it drains.
                    lane.barrier = Some(Barrier {
                        targets: Vec::new(),
                        task: BarrierTask::Terminate,
                    });
                    return;
                }
                lane.exit = true;
            }
        }
    }
}
