// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ways to feed a backend: one serial event lane driven through `SyncIndexer::worker`,
//! or a [`ThreadPoolIndexer`] with pipelined, acknowledged events.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::thread::JoinHandle;

use futures_util::task::noop_waker_ref;
use tokio::sync::oneshot;

use super::{HarnessBackend, new_backend};
use crate::indexer::{KvIndexerInterface, MatchDetails, ThreadPoolIndexer, WorkerTask};
use crate::protocols::{LocalBlockHash, OverlapScores, RouterEvent, WorkerId, WorkerWithDpRank};

/// One input to the indexer, as the router would deliver it.
#[derive(Clone, Debug)]
pub(super) enum Op {
    Event(RouterEvent),
    /// `remove_worker_dp_rank`.
    RemoveRank(WorkerWithDpRank),
    /// `remove_worker`.
    RemoveWorker(WorkerId),
}

/// A backend behind one event lane on its own thread. Every operation is acknowledged
/// before the next is sent and lookups run while the lane is idle, so execution is serial
/// and deterministic.
pub(super) struct SerialLane<T: HarnessBackend> {
    backend: Arc<T>,
    tx: flume::Sender<WorkerTask>,
    thread: Option<JoinHandle<()>>,
}

impl<T: HarnessBackend> SerialLane<T> {
    pub(super) fn new() -> Self {
        let backend = new_backend::<T>();
        let (tx, rx) = flume::unbounded();
        let lane_backend = Arc::clone(&backend);
        let thread = std::thread::spawn(move || {
            lane_backend
                .worker(rx, None)
                .expect("harness event lane exited with an error");
        });
        Self {
            backend,
            tx,
            thread: Some(thread),
        }
    }

    pub(super) fn backend(&self) -> &T {
        &self.backend
    }

    fn send(&self, task: WorkerTask) {
        self.tx
            .send(task)
            .expect("harness event lane is gone (did it panic?)");
    }

    /// Applies `op` and returns whether the backend accepted it.
    pub(super) fn apply(&self, op: Op) -> bool {
        match op {
            Op::Event(event) => {
                let (resp, rx) = oneshot::channel();
                self.send(WorkerTask::EventWithAck { event, resp });
                rx.blocking_recv()
                    .expect("harness event lane dropped an acknowledgement")
            }
            Op::RemoveRank(rank) => {
                self.send(WorkerTask::RemoveWorkerDpRank {
                    worker_id: rank.worker_id,
                    dp_rank: rank.dp_rank,
                    sweep_tree: true,
                });
                self.flush();
                true
            }
            Op::RemoveWorker(worker_id) => {
                let (resp, rx) = oneshot::channel();
                self.send(WorkerTask::RemoveWorker {
                    worker_id,
                    sweep_tree: true,
                    resp,
                });
                rx.blocking_recv()
                    .expect("harness event lane dropped a removal acknowledgement");
                true
            }
        }
    }

    pub(super) fn flush(&self) {
        let (resp, rx) = oneshot::channel();
        self.send(WorkerTask::Flush(resp));
        rx.blocking_recv()
            .expect("harness event lane dropped a flush acknowledgement");
    }
}

impl<T: HarnessBackend> Drop for SerialLane<T> {
    fn drop(&mut self) {
        let _ = self.tx.send(WorkerTask::Terminate);
        let Some(thread) = self.thread.take() else {
            return;
        };
        if thread.join().is_err() && !std::thread::panicking() {
            panic!("harness event lane panicked");
        }
    }
}

/// An acknowledgement still in flight, or already in.
enum Ack<'a> {
    Pending(Pin<Box<dyn Future<Output = bool> + 'a>>),
    Done(bool),
}

impl<'a> Ack<'a> {
    /// Polls `future` once so it enqueues its task now, in call order.
    fn start(future: impl Future<Output = bool> + 'a) -> Self {
        let mut future: Pin<Box<dyn Future<Output = bool> + 'a>> = Box::pin(future);
        match future
            .as_mut()
            .poll(&mut Context::from_waker(noop_waker_ref()))
        {
            Poll::Ready(applied) => Self::Done(applied),
            Poll::Pending => Self::Pending(future),
        }
    }

    async fn finish(self) -> bool {
        match self {
            Self::Pending(future) => future.await,
            Self::Done(applied) => applied,
        }
    }
}

/// A backend behind a [`ThreadPoolIndexer`].
pub(super) struct PoolDriver<T: HarnessBackend> {
    indexer: ThreadPoolIndexer<T>,
    runtime: tokio::runtime::Runtime,
}

impl<T: HarnessBackend> PoolDriver<T> {
    pub(super) fn new(lanes: usize) -> Self {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("harness runtime");
        let indexer = ThreadPoolIndexer::new(T::harness_new(), lanes, 32);
        Self { indexer, runtime }
    }

    pub(super) fn backend(&self) -> &T {
        self.indexer.backend()
    }

    /// Applies `ops` in order: events and rank removals are enqueued back to back on
    /// their sticky lanes, and a worker removal waits for everything before it. Returns
    /// one acknowledgement per op, after a flush.
    pub(super) fn apply_batch(&self, ops: Vec<Op>) -> Vec<bool> {
        let indexer = &self.indexer;
        self.runtime.block_on(async move {
            let mut acks: Vec<Ack<'_>> = Vec::with_capacity(ops.len());
            for op in ops {
                match op {
                    Op::Event(event) => acks.push(Ack::start(async move {
                        indexer.apply_event_and_wait(event).await.is_ok()
                    })),
                    Op::RemoveRank(rank) => acks.push(Ack::start(async move {
                        indexer
                            .remove_worker_dp_rank(rank.worker_id, rank.dp_rank)
                            .await;
                        true
                    })),
                    Op::RemoveWorker(worker_id) => {
                        for ack in &mut acks {
                            if let Ack::Pending(_) = ack {
                                let pending = std::mem::replace(ack, Ack::Done(false));
                                *ack = Ack::Done(pending.finish().await);
                            }
                        }
                        indexer.remove_worker(worker_id).await;
                        acks.push(Ack::Done(true));
                    }
                }
            }
            let mut applied = Vec::with_capacity(acks.len());
            for ack in acks {
                applied.push(ack.finish().await);
            }
            indexer.flush().await;
            applied
        })
    }
}

impl<T: HarnessBackend> PoolDriver<T> {
    /// Applies `ops` one at a time, waiting for each, so no two lanes ever run at once.
    pub(super) fn apply_sequential(&self, ops: Vec<Op>) -> Vec<bool> {
        let indexer = &self.indexer;
        self.runtime.block_on(async move {
            let mut applied = Vec::with_capacity(ops.len());
            for op in ops {
                applied.push(match op {
                    Op::Event(event) => indexer.apply_event_and_wait(event).await.is_ok(),
                    Op::RemoveRank(rank) => {
                        indexer
                            .remove_worker_dp_rank(rank.worker_id, rank.dp_rank)
                            .await;
                        indexer.flush().await;
                        true
                    }
                    Op::RemoveWorker(worker_id) => {
                        indexer.remove_worker(worker_id).await;
                        true
                    }
                });
            }
            indexer.flush().await;
            applied
        })
    }
}

/// One way to feed a backend.
pub(super) enum Driver<T: HarnessBackend> {
    Serial(SerialLane<T>),
    Pool(PoolDriver<T>),
}

impl<T: HarnessBackend> Driver<T> {
    pub(super) fn backend(&self) -> &T {
        match self {
            Self::Serial(lane) => lane.backend(),
            Self::Pool(pool) => pool.backend(),
        }
    }

    pub(super) fn find(&self, query: &[LocalBlockHash]) -> OverlapScores {
        self.backend().find_matches(query, false)
    }

    pub(super) fn details(&self, query: &[LocalBlockHash]) -> Option<MatchDetails> {
        self.backend().harness_match_details(query)
    }

    pub(super) fn apply_all(&self, ops: Vec<Op>) -> Vec<bool> {
        match self {
            Self::Serial(lane) => ops.into_iter().map(|op| lane.apply(op)).collect(),
            Self::Pool(pool) => pool.apply_batch(ops),
        }
    }
}
