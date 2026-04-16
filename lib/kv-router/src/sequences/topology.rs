// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;

use super::prompt_membership_trie::WorkerLookup;
use super::single::ActiveSequences;
use crate::protocols::WorkerWithDpRank;

#[derive(Clone)]
pub(super) struct RemovedWorkerState {
    pub(super) worker: WorkerWithDpRank,
    pub(super) trie_lookup: Arc<RwLock<WorkerLookup>>,
}

impl std::fmt::Debug for RemovedWorkerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemovedWorkerState")
            .field("worker", &self.worker)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Default, Clone)]
pub(super) struct WorkerTopologyChange {
    pub(super) added: Vec<WorkerWithDpRank>,
    pub(super) removed: Vec<RemovedWorkerState>,
}

pub(super) struct WorkerSlot {
    pub(super) worker: WorkerWithDpRank,
    pub(super) sequences: RwLock<ActiveSequences>,
    pub(super) trie_lookup: Arc<RwLock<WorkerLookup>>,
}

impl WorkerSlot {
    fn new(worker: WorkerWithDpRank, block_size: usize) -> Self {
        Self {
            worker,
            sequences: RwLock::new(ActiveSequences::new(block_size)),
            trie_lookup: Arc::new(RwLock::new(WorkerLookup::default())),
        }
    }
}

pub(super) struct WorkerTable {
    pub(super) slots: Vec<WorkerSlot>,
    pub(super) index: FxHashMap<WorkerWithDpRank, usize>,
}

impl WorkerTable {
    pub(super) fn new(block_size: usize, dp_range: &HashMap<u64, (u32, u32)>) -> Self {
        let mut slots = Vec::new();
        let mut index = FxHashMap::default();
        for worker in workers_from_dp_range(dp_range) {
            let idx = slots.len();
            slots.push(WorkerSlot::new(worker, block_size));
            index.insert(worker, idx);
        }
        Self { slots, index }
    }

    pub(super) fn workers(&self) -> impl Iterator<Item = WorkerWithDpRank> + '_ {
        self.slots.iter().map(|slot| slot.worker)
    }

    pub(super) fn register_external(
        &mut self,
        block_size: usize,
        dp_range: &HashMap<u64, (u32, u32)>,
    ) -> WorkerTopologyChange {
        let mut change = WorkerTopologyChange::default();
        for worker in workers_from_dp_range(dp_range) {
            if self.index.contains_key(&worker) {
                continue;
            }

            let idx = self.slots.len();
            self.slots.push(WorkerSlot::new(worker, block_size));
            self.index.insert(worker, idx);
            change.added.push(worker);
        }
        change
    }

    pub(super) fn reconcile(
        &mut self,
        block_size: usize,
        new_dp_range: &HashMap<u64, (u32, u32)>,
    ) -> WorkerTopologyChange {
        let target_workers: FxHashSet<WorkerWithDpRank> =
            workers_from_dp_range(new_dp_range).into_iter().collect();

        let mut old: FxHashMap<WorkerWithDpRank, WorkerSlot> = self
            .slots
            .drain(..)
            .map(|slot| (slot.worker, slot))
            .collect();
        self.index.clear();

        let mut added = Vec::new();
        for worker in target_workers {
            if !old.contains_key(&worker) {
                added.push(worker);
            }
            let idx = self.slots.len();
            let slot = old
                .remove(&worker)
                .unwrap_or_else(|| WorkerSlot::new(worker, block_size));
            self.slots.push(slot);
            self.index.insert(worker, idx);
        }

        let removed = old
            .into_values()
            .map(|slot| RemovedWorkerState {
                worker: slot.worker,
                trie_lookup: slot.trie_lookup,
            })
            .collect();

        WorkerTopologyChange { added, removed }
    }

    pub(super) fn ensure_worker(
        &mut self,
        block_size: usize,
        worker: WorkerWithDpRank,
    ) -> WorkerTopologyChange {
        if self.index.contains_key(&worker) {
            return WorkerTopologyChange::default();
        }

        let idx = self.slots.len();
        self.slots.push(WorkerSlot::new(worker, block_size));
        self.index.insert(worker, idx);
        WorkerTopologyChange {
            added: vec![worker],
            removed: Vec::new(),
        }
    }
}

fn workers_from_dp_range(dp_range: &HashMap<u64, (u32, u32)>) -> Vec<WorkerWithDpRank> {
    let mut workers = Vec::new();
    for (&worker_id, &(dp_start, dp_size)) in dp_range {
        for dp_rank in dp_start..(dp_start + dp_size) {
            workers.push(WorkerWithDpRank::new(worker_id, dp_rank));
        }
    }
    workers
}

#[cfg(test)]
mod tests {
    use tokio::time::Instant;

    use super::*;

    fn worker(worker_id: u64, dp_rank: u32) -> WorkerWithDpRank {
        WorkerWithDpRank::new(worker_id, dp_rank)
    }

    #[test]
    fn new_expands_dp_ranges_into_slots_and_index() {
        let table = WorkerTable::new(4, &HashMap::from([(7, (2, 3)), (9, (0, 1))]));

        let workers: FxHashSet<_> = table.workers().collect();
        assert_eq!(
            workers,
            FxHashSet::from_iter([worker(7, 2), worker(7, 3), worker(7, 4), worker(9, 0)])
        );
        assert_eq!(table.index.len(), 4);
        assert_eq!(table.slots.len(), 4);
        for worker in workers {
            assert!(table.index.contains_key(&worker));
        }
    }

    #[test]
    fn register_external_only_adds_missing_workers() {
        let mut table = WorkerTable::new(4, &HashMap::from([(1, (0, 1))]));
        let change = table.register_external(4, &HashMap::from([(1, (0, 2)), (2, (0, 1))]));

        assert_eq!(
            change.added.into_iter().collect::<FxHashSet<_>>(),
            FxHashSet::from_iter([worker(1, 1), worker(2, 0)])
        );
        assert!(change.removed.is_empty());
        assert_eq!(table.index.len(), 3);
    }

    #[test]
    fn ensure_worker_is_idempotent() {
        let mut table = WorkerTable::new(4, &HashMap::from([(1, (0, 1))]));
        let target = worker(2, 0);

        let first = table.ensure_worker(4, target);
        let second = table.ensure_worker(4, target);

        assert_eq!(first.added, vec![target]);
        assert!(first.removed.is_empty());
        assert!(second.added.is_empty());
        assert!(second.removed.is_empty());
        assert_eq!(table.index.len(), 2);
    }

    #[test]
    fn reconcile_preserves_existing_worker_state_and_reports_delta() {
        let mut table = WorkerTable::new(4, &HashMap::from([(1, (0, 1)), (2, (0, 1))]));
        let existing = worker(1, 0);
        let removed = worker(2, 0);
        let added = worker(3, 0);

        {
            let idx = table.index[&existing];
            let mut seq = table.slots[idx].sequences.write();
            let outcome = seq.add_request(
                "req-1".to_string(),
                Some(vec![1, 2, 3]),
                12,
                0,
                None,
                Instant::now(),
            );
            assert_eq!(outcome.membership_delta.stores[0].hashes, vec![1, 2, 3],);
        }

        let change = table.reconcile(4, &HashMap::from([(1, (0, 1)), (3, (0, 1))]));

        assert_eq!(change.added, vec![added]);
        assert_eq!(
            change
                .removed
                .iter()
                .map(|state| state.worker)
                .collect::<Vec<_>>(),
            vec![removed]
        );
        assert!(table.index.contains_key(&existing));
        assert!(table.index.contains_key(&added));
        assert!(!table.index.contains_key(&removed));

        let existing_idx = table.index[&existing];
        assert_eq!(
            table.slots[existing_idx].sequences.read().active_blocks(),
            3
        );

        let added_idx = table.index[&added];
        assert_eq!(table.slots[added_idx].sequences.read().active_blocks(), 0);
    }
}
