// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live per-rank resident block counts.
//!
//! Each event thread owns the block maps of its ranks. Once a policy reads resident blocks, the
//! thread keeps one counter per rank it owns and sets it to the rank's block count after each
//! event it applies. Readers find the counters through a copy-on-write map that changes only
//! when a rank appears or is removed.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arc_swap::{ArcSwap, Guard};
use crossbeam_utils::CachePadded;
use rustc_hash::FxHashMap;

use crate::protocols::WorkerWithDpRank;

/// Padded so counters written by different event threads never share a cache line.
type Counter = CachePadded<AtomicU64>;

fn counter(blocks: usize) -> Arc<Counter> {
    Arc::new(CachePadded::new(AtomicU64::new(blocks as u64)))
}

/// The counters of every rank an indexer holds.
#[derive(Clone, Debug, Default)]
pub(crate) struct ResidentBlockCounts {
    counters: FxHashMap<WorkerWithDpRank, Arc<Counter>>,
}

impl ResidentBlockCounts {
    /// Blocks held for `worker` as of the last event applied to it; `None` for a rank that has held
    /// no blocks while tracked, or that the indexer removed.
    pub(crate) fn get(&self, worker: WorkerWithDpRank) -> Option<u64> {
        self.counters
            .get(&worker)
            .map(|counter| counter.load(Ordering::Relaxed))
    }
}

/// The counters a policy reads, shared with the indexer's event threads.
#[derive(Clone, Debug, Default)]
pub struct ResidentBlockCountsHandle {
    counts: Arc<ArcSwap<ResidentBlockCounts>>,
}

impl ResidentBlockCountsHandle {
    /// The counters as of the last rank change. Hold the guard only for one selection.
    pub(crate) fn load(&self) -> Guard<Arc<ResidentBlockCounts>> {
        self.counts.load()
    }

    /// Counters fixed at `counts`, for tests and benchmarks.
    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn fixed(counts: impl IntoIterator<Item = (WorkerWithDpRank, u64)>) -> Self {
        let counters = counts
            .into_iter()
            .map(|(worker, blocks)| (worker, counter(blocks as usize)))
            .collect();
        Self {
            counts: Arc::new(ArcSwap::from_pointee(ResidentBlockCounts { counters })),
        }
    }

    /// Sum of every counter, read the way selections read them.
    #[cfg(feature = "bench")]
    #[doc(hidden)]
    pub fn bench_sum(&self) -> u64 {
        self.counts
            .load()
            .counters
            .values()
            .map(|counter| counter.load(Ordering::Relaxed))
            .sum()
    }

    /// Sets every counter to `blocks`, the way event threads write them.
    #[cfg(feature = "bench")]
    pub(crate) fn store_all(&self, blocks: u64) {
        for counter in self.counts.load().counters.values() {
            counter.store(blocks, Ordering::Relaxed);
        }
    }

    fn publish(&self, added: &[(WorkerWithDpRank, Arc<Counter>)]) {
        if added.is_empty() {
            return;
        }
        self.counts.rcu(|current| {
            let mut next = ResidentBlockCounts::clone(current);
            next.counters.extend(
                added
                    .iter()
                    .map(|(worker, counter)| (*worker, counter.clone())),
            );
            next
        });
    }

    fn unpublish(&self, removed: &[(WorkerWithDpRank, Arc<Counter>)]) {
        if removed.is_empty() {
            return;
        }
        self.counts.rcu(|current| {
            let mut next = ResidentBlockCounts::clone(current);
            for (worker, counter) in removed {
                if next
                    .counters
                    .get(worker)
                    .is_some_and(|published| Arc::ptr_eq(published, counter))
                {
                    next.counters.remove(worker);
                }
            }
            next
        });
    }
}

/// The counters for the ranks one event thread owns. Dropping it, as when the thread exits,
/// removes them, so readers get `None` instead of counts nothing updates anymore.
pub(crate) struct ResidentBlockTracker {
    counts: ResidentBlockCountsHandle,
    owned: FxHashMap<WorkerWithDpRank, Arc<Counter>>,
}

impl ResidentBlockTracker {
    /// Starts tracking, with counters for the ranks the thread already holds blocks for.
    pub(crate) fn new(
        counts: ResidentBlockCountsHandle,
        ranks: impl IntoIterator<Item = (WorkerWithDpRank, usize)>,
    ) -> Self {
        let owned: FxHashMap<_, _> = ranks
            .into_iter()
            .filter(|&(_, blocks)| blocks > 0)
            .map(|(worker, blocks)| (worker, counter(blocks)))
            .collect();
        let added: Vec<_> = owned
            .iter()
            .map(|(worker, counter)| (*worker, counter.clone()))
            .collect();
        counts.publish(&added);
        Self { counts, owned }
    }

    /// Records `worker`'s block count after an event. A rank is counted from its first block on;
    /// `None` means the thread holds no blocks for it, as after a clear, and a counted rank then
    /// reads 0.
    pub(crate) fn update(&mut self, worker: WorkerWithDpRank, blocks: Option<usize>) {
        if let Some(counter) = self.owned.get(&worker) {
            counter.store(blocks.unwrap_or(0) as u64, Ordering::Relaxed);
        } else if let Some(blocks) = blocks.filter(|&blocks| blocks > 0) {
            let counter = counter(blocks);
            self.owned.insert(worker, counter.clone());
            self.counts.publish(&[(worker, counter)]);
        }
    }

    /// Stops counting the ranks `removed` selects, once the indexer has removed them.
    pub(crate) fn remove(&mut self, removed: impl Fn(WorkerWithDpRank) -> bool) {
        let mut gone = Vec::new();
        self.owned.retain(|worker, counter| {
            let keep = !removed(*worker);
            if !keep {
                gone.push((*worker, counter.clone()));
            }
            keep
        });
        self.counts.unpublish(&gone);
    }
}

impl Drop for ResidentBlockTracker {
    fn drop(&mut self) {
        let owned: Vec<_> = self.owned.drain().collect();
        self.counts.unpublish(&owned);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_publishes_counts_and_withdraws_them() {
        let counts = ResidentBlockCountsHandle::default();
        let (held, emptied, removed, unseen) = (
            WorkerWithDpRank::new(1, 0),
            WorkerWithDpRank::new(1, 1),
            WorkerWithDpRank::new(2, 0),
            WorkerWithDpRank::new(3, 0),
        );
        // Empty block maps, as a remove for an unknown rank can leave, start no count.
        let mut tracker =
            ResidentBlockTracker::new(counts.clone(), [(held, 4), (emptied, 2), (unseen, 0)]);
        tracker.update(held, Some(5));
        tracker.update(emptied, None);
        tracker.update(removed, Some(7));
        tracker.update(unseen, Some(0));
        let read = |worker| counts.load().get(worker);
        assert_eq!(
            [read(held), read(emptied), read(removed), read(unseen)],
            [Some(5), Some(0), Some(7), None]
        );

        tracker.remove(|worker| worker.worker_id == 2);
        assert_eq!(read(removed), None);
        drop(tracker);
        assert_eq!(read(held), None, "an exited thread's ranks read None");
    }
}
