// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam_queue::ArrayQueue;
use rustc_hash::FxHashSet;
use tokio::sync::Notify;

use dynamo_router_protocol::worker::WorkerWithDpRank;

const CAPACITY_UPDATE_LIMIT: usize = 256;

/// Capacity changes are hints to re-read state, not lifecycle mutations. A full
/// buffer can therefore collapse into a full recheck without losing progress.
/// The dispatch gate's wake rides the same channel under its own flag, so the
/// actor can tell a freed worker from a gate that released a held head.
pub(super) struct CapacityUpdates {
    workers: ArrayQueue<WorkerWithDpRank>,
    full_recheck: AtomicBool,
    gate_recheck: AtomicBool,
    notify: Notify,
}

/// What one drain of the channel found.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Drained {
    /// Capacity changed on more workers than the buffer holds, or on an unnamed one.
    pub(super) full_recheck: bool,
    /// The dispatch gate asked for its held heads to be re-asked.
    pub(super) gate_recheck: bool,
}

impl CapacityUpdates {
    pub(super) fn new() -> Self {
        Self {
            workers: ArrayQueue::new(CAPACITY_UPDATE_LIMIT),
            full_recheck: AtomicBool::new(false),
            gate_recheck: AtomicBool::new(false),
            notify: Notify::new(),
        }
    }

    /// Record the hint before signalling, without an await or a pending-count
    /// check. notify_one retains a permit when the actor is not yet waiting.
    pub(super) fn record(&self, worker: Option<WorkerWithDpRank>) {
        if worker.is_none_or(|worker| self.workers.push(worker).is_err()) {
            self.full_recheck.store(true, Ordering::Release);
        }
        self.notify.notify_one();
    }

    /// Record that the dispatch gate may now let a held head go.
    pub(super) fn record_gate(&self) {
        self.gate_recheck.store(true, Ordering::Release);
        self.notify.notify_one();
    }

    pub(super) async fn notified(&self) {
        self.notify.notified().await;
    }

    /// Take one bounded batch into actor-owned scratch storage. Clear the flags
    /// before draining: overflow during this drain belongs to the next wake and
    /// must not be cleared after its producer has signalled.
    pub(super) fn drain_into(&self, workers: &mut FxHashSet<WorkerWithDpRank>) -> Drained {
        let drained = Drained {
            full_recheck: self.full_recheck.swap(false, Ordering::AcqRel),
            gate_recheck: self.gate_recheck.swap(false, Ordering::AcqRel),
        };
        workers.clear();
        for _ in 0..CAPACITY_UPDATE_LIMIT {
            let Some(worker) = self.workers.pop() else {
                break;
            };
            workers.insert(worker);
        }
        if !self.workers.is_empty()
            || self.full_recheck.load(Ordering::Acquire)
            || self.gate_recheck.load(Ordering::Acquire)
        {
            self.notify.notify_one();
        }
        drained
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;

    use super::*;

    #[tokio::test]
    async fn notifications_survive_idle_overflow_and_consumer_handoff() {
        let updates = CapacityUpdates::new();
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(2, 0);
        let mut workers = FxHashSet::default();

        // A burst before the actor waits consumes bounded storage and retains
        // a wakeup. Overflow requests a full recheck, not a dropped worker.
        for _ in 0..=CAPACITY_UPDATE_LIMIT {
            updates.record(Some(first));
        }
        updates.record(Some(second));
        updates.notified().await;
        assert!(updates.drain_into(&mut workers).full_recheck);
        assert_eq!(workers, FxHashSet::from_iter([first]));

        // A signalled wait can lose the actor's select race to an admission
        // command. Dropping that wait must leave a permit for the next turn.
        let mut waiting = Box::pin(updates.notified());
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(waiting.as_mut().poll(&mut context).is_pending());
        updates.record(Some(second));
        drop(waiting);
        updates.notified().await;
        assert_eq!(updates.drain_into(&mut workers), Drained::default());
        assert_eq!(workers, FxHashSet::from_iter([second]));

        updates.record(None);
        updates.notified().await;
        assert!(updates.drain_into(&mut workers).full_recheck);
        assert!(workers.is_empty());
        assert_eq!(updates.drain_into(&mut workers), Drained::default());

        // The gate's wake is told apart from a capacity change.
        updates.record_gate();
        updates.notified().await;
        assert_eq!(
            updates.drain_into(&mut workers),
            Drained {
                full_recheck: false,
                gate_recheck: true,
            }
        );
        assert!(workers.is_empty());
    }
}
