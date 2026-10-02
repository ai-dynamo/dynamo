// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dynamo_kv_router::protocols::WorkerWithDpRank;

/// Cached per-worker program-retention capacity derived from model deployment cards.
static NEXT_SNAPSHOT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct WorkerCapacitySnapshot {
    id: u64,
    capacities: HashMap<WorkerWithDpRank, usize>,
    live_workers: Option<HashSet<WorkerWithDpRank>>,
}

impl WorkerCapacitySnapshot {
    /// Build a snapshot from the MDC-backed capacities currently available.
    ///
    /// An empty capacity map is treated as MDC cold start. Use
    /// [`Self::with_live_workers`] to attach the independent discovery view.
    pub fn new(capacities: impl IntoIterator<Item = (WorkerWithDpRank, usize)>) -> Self {
        Self {
            id: NEXT_SNAPSHOT_ID.fetch_add(1, Ordering::Relaxed),
            capacities: capacities.into_iter().collect(),
            live_workers: None,
        }
    }

    /// Attach an authoritative discovery snapshot for worker-removal detection.
    pub fn with_live_workers(
        mut self,
        workers: impl IntoIterator<Item = WorkerWithDpRank>,
    ) -> Self {
        self.live_workers = Some(workers.into_iter().collect());
        self
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.capacities.is_empty()
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn has_usable_capacity(&self) -> bool {
        self.capacities.keys().any(|&worker| self.is_live(worker))
    }

    pub(crate) fn has_live_worker(&self) -> bool {
        self.live_workers
            .as_ref()
            .is_none_or(|workers| !workers.is_empty())
    }

    pub(crate) fn is_live(&self, worker: WorkerWithDpRank) -> bool {
        self.live_workers
            .as_ref()
            .is_none_or(|workers| workers.contains(&worker))
    }

    pub(crate) fn has_liveness(&self) -> bool {
        self.live_workers.is_some()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (WorkerWithDpRank, usize)> + '_ {
        self.capacities
            .iter()
            .map(|(&worker, &capacity)| (worker, capacity))
    }
}

impl Default for WorkerCapacitySnapshot {
    fn default() -> Self {
        Self::new(std::iter::empty())
    }
}

/// Supplies a non-blocking cached MDC capacity and discovery snapshot.
pub trait WorkerCapacityProvider: Send + Sync + 'static {
    fn snapshot(&self) -> Arc<WorkerCapacitySnapshot>;
}

impl<F> WorkerCapacityProvider for F
where
    F: Fn() -> Arc<WorkerCapacitySnapshot> + Send + Sync + 'static,
{
    fn snapshot(&self) -> Arc<WorkerCapacitySnapshot> {
        self()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinguishes_missing_capacity_from_worker_removal() {
        let worker_with_card = WorkerWithDpRank::new(1, 0);
        let live_without_card = WorkerWithDpRank::new(2, 0);
        let removed = WorkerWithDpRank::new(3, 0);
        let snapshot = WorkerCapacitySnapshot::new([(worker_with_card, 1_000)])
            .with_live_workers([worker_with_card, live_without_card]);

        assert!(snapshot.capacities.contains_key(&worker_with_card));
        assert!(!snapshot.capacities.contains_key(&live_without_card));
        assert!(snapshot.is_live(live_without_card));
        assert!(!snapshot.is_live(removed));
    }
}
