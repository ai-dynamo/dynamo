// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};

use super::single::ActiveSequences;
use crate::protocols::{DpRank, WorkerId, WorkerWithDpRank};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerDpRange {
    pub worker_id: WorkerId,
    pub dp_start: DpRank,
    pub dp_size: u32,
}

impl WorkerDpRange {
    pub fn new(worker_id: WorkerId, dp_start: DpRank, dp_size: u32) -> Self {
        Self {
            worker_id,
            dp_start,
            dp_size,
        }
    }

    pub fn validate(self) -> Result<Self, WorkerTopologyError> {
        if self.dp_size == 0 {
            return Err(WorkerTopologyError::InvalidDpSize {
                worker_id: self.worker_id,
            });
        }
        if self.dp_start.checked_add(self.dp_size).is_none() {
            return Err(WorkerTopologyError::InvalidDpRange {
                worker_id: self.worker_id,
                dp_start: self.dp_start,
                dp_size: self.dp_size,
            });
        }
        Ok(self)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum WorkerTopologyError {
    #[error("dp_size must be greater than 0 for worker {worker_id}")]
    InvalidDpSize { worker_id: WorkerId },

    #[error("dp range overflows u32 for worker {worker_id}: start={dp_start} size={dp_size}")]
    InvalidDpRange {
        worker_id: WorkerId,
        dp_start: DpRank,
        dp_size: u32,
    },

    #[error("worker {worker_id} is already registered")]
    DuplicateWorker { worker_id: WorkerId },

    #[error("worker {worker_id} is not registered")]
    WorkerNotFound { worker_id: WorkerId },
}

#[derive(Clone)]
pub(super) struct RemovedWorkerState {
    pub(super) worker: WorkerWithDpRank,
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
}

impl WorkerSlot {
    fn new(worker: WorkerWithDpRank, block_size: usize, expiry_enabled: bool) -> Self {
        let sequences = if expiry_enabled {
            ActiveSequences::new(block_size)
        } else {
            ActiveSequences::new_without_expiry(block_size)
        };
        Self {
            worker,
            sequences: RwLock::new(sequences),
        }
    }
}

pub(super) struct WorkerTable {
    pub(super) slots: Vec<WorkerSlot>,
    pub(super) index: FxHashMap<WorkerWithDpRank, usize>,
    worker_ranges: HashMap<WorkerId, WorkerDpRange>,
    expiry_enabled: bool,
}

impl WorkerTable {
    pub(super) fn new(block_size: usize, dp_range: &HashMap<u64, (u32, u32)>) -> Self {
        Self::new_with_expiry(block_size, dp_range, true)
    }

    pub(super) fn new_without_expiry(
        block_size: usize,
        dp_range: &HashMap<u64, (u32, u32)>,
    ) -> Self {
        Self::new_with_expiry(block_size, dp_range, false)
    }

    fn new_with_expiry(
        block_size: usize,
        dp_range: &HashMap<u64, (u32, u32)>,
        expiry_enabled: bool,
    ) -> Self {
        let worker_ranges: HashMap<WorkerId, WorkerDpRange> = dp_range
            .iter()
            .map(|(&worker_id, &(dp_start, dp_size))| {
                let range = WorkerDpRange::new(worker_id, dp_start, dp_size);
                range
                    .validate()
                    .expect("initial worker DP ranges must be valid");
                (worker_id, range)
            })
            .collect();
        let mut slots = Vec::new();
        let mut index = FxHashMap::default();
        for worker in workers_from_ranges(worker_ranges.values().copied()) {
            let idx = slots.len();
            slots.push(WorkerSlot::new(worker, block_size, expiry_enabled));
            index.insert(worker, idx);
        }
        Self {
            slots,
            index,
            worker_ranges,
            expiry_enabled,
        }
    }

    pub(super) fn workers(&self) -> impl Iterator<Item = WorkerWithDpRank> + '_ {
        self.slots.iter().map(|slot| slot.worker)
    }

    pub(super) fn register_worker(
        &mut self,
        block_size: usize,
        range: WorkerDpRange,
    ) -> Result<WorkerTopologyChange, WorkerTopologyError> {
        range.validate()?;
        if self.worker_ranges.contains_key(&range.worker_id) {
            return Err(WorkerTopologyError::DuplicateWorker {
                worker_id: range.worker_id,
            });
        }
        self.worker_ranges.insert(range.worker_id, range);
        Ok(self.reconcile_worker_slots(block_size, range.worker_id))
    }

    pub(super) fn upsert_worker(
        &mut self,
        block_size: usize,
        range: WorkerDpRange,
    ) -> Result<WorkerTopologyChange, WorkerTopologyError> {
        range.validate()?;
        if self.worker_ranges.get(&range.worker_id) == Some(&range) {
            return Ok(WorkerTopologyChange::default());
        }
        self.worker_ranges.insert(range.worker_id, range);
        Ok(self.reconcile_worker_slots(block_size, range.worker_id))
    }

    pub(super) fn unregister_worker(
        &mut self,
        block_size: usize,
        worker_id: WorkerId,
    ) -> Result<WorkerTopologyChange, WorkerTopologyError> {
        if self.worker_ranges.remove(&worker_id).is_none() {
            return Err(WorkerTopologyError::WorkerNotFound { worker_id });
        }
        Ok(self.reconcile_worker_slots(block_size, worker_id))
    }

    pub(super) fn reconcile(
        &mut self,
        block_size: usize,
        ranges: Vec<WorkerDpRange>,
    ) -> Result<WorkerTopologyChange, WorkerTopologyError> {
        let mut worker_ranges = HashMap::with_capacity(ranges.len());
        for range in ranges {
            range.validate()?;
            if worker_ranges.insert(range.worker_id, range).is_some() {
                return Err(WorkerTopologyError::DuplicateWorker {
                    worker_id: range.worker_id,
                });
            }
        }
        self.worker_ranges = worker_ranges;
        Ok(self.reconcile_all_slots(block_size))
    }

    pub(super) fn worker_ranges(&self) -> Vec<WorkerDpRange> {
        let mut ranges: Vec<_> = self.worker_ranges.values().copied().collect();
        ranges.sort_by_key(|range| range.worker_id);
        ranges
    }

    pub(super) fn has_registered_workers(&self) -> bool {
        !self.worker_ranges.is_empty()
    }

    fn reconcile_worker_slots(
        &mut self,
        block_size: usize,
        worker_id: WorkerId,
    ) -> WorkerTopologyChange {
        let target_workers: Vec<_> = self
            .worker_ranges
            .get(&worker_id)
            .copied()
            .into_iter()
            .flat_map(workers_from_range)
            .collect();
        let mut old = FxHashMap::default();
        let mut retained = Vec::with_capacity(self.slots.len());
        for slot in self.slots.drain(..) {
            if slot.worker.worker_id == worker_id {
                old.insert(slot.worker, slot);
            } else {
                retained.push(slot);
            }
        }
        self.slots = retained;

        let mut added = Vec::new();
        for worker in target_workers {
            let slot = old.remove(&worker).unwrap_or_else(|| {
                added.push(worker);
                WorkerSlot::new(worker, block_size, self.expiry_enabled)
            });
            self.slots.push(slot);
        }

        let removed = old.into_values().map(RemovedWorkerState::from).collect();
        self.rebuild_index();
        WorkerTopologyChange { added, removed }
    }

    fn reconcile_all_slots(&mut self, block_size: usize) -> WorkerTopologyChange {
        let target_workers: FxHashSet<WorkerWithDpRank> =
            workers_from_ranges(self.worker_ranges.values().copied())
                .into_iter()
                .collect();
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
                .unwrap_or_else(|| WorkerSlot::new(worker, block_size, self.expiry_enabled));
            self.slots.push(slot);
            self.index.insert(worker, idx);
        }

        let removed = old.into_values().map(RemovedWorkerState::from).collect();

        WorkerTopologyChange { added, removed }
    }

    fn rebuild_index(&mut self) {
        self.index.clear();
        self.index.extend(
            self.slots
                .iter()
                .enumerate()
                .map(|(idx, slot)| (slot.worker, idx)),
        );
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
        self.slots
            .push(WorkerSlot::new(worker, block_size, self.expiry_enabled));
        self.index.insert(worker, idx);
        WorkerTopologyChange {
            added: vec![worker],
            removed: Vec::new(),
        }
    }
}

impl From<WorkerSlot> for RemovedWorkerState {
    fn from(slot: WorkerSlot) -> Self {
        Self {
            worker: slot.worker,
        }
    }
}

fn workers_from_ranges(ranges: impl IntoIterator<Item = WorkerDpRange>) -> Vec<WorkerWithDpRank> {
    let mut workers = Vec::new();
    for range in ranges {
        workers.extend(workers_from_range(range));
    }
    workers
}

fn workers_from_range(range: WorkerDpRange) -> impl Iterator<Item = WorkerWithDpRank> {
    (range.dp_start..(range.dp_start + range.dp_size))
        .map(move |dp_rank| WorkerWithDpRank::new(range.worker_id, dp_rank))
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
        assert_eq!(
            table.worker_ranges(),
            vec![WorkerDpRange::new(7, 2, 3), WorkerDpRange::new(9, 0, 1)]
        );
    }

    #[test]
    fn upsert_worker_updates_one_authoritative_range() {
        let mut table = WorkerTable::new(4, &HashMap::from([(1, (0, 1))]));
        let lazy_worker = worker(2, 0);
        table.ensure_worker(4, lazy_worker);
        let change = table.upsert_worker(4, WorkerDpRange::new(1, 0, 2)).unwrap();

        assert_eq!(change.added, vec![worker(1, 1)]);
        assert!(change.removed.is_empty());
        assert_eq!(table.index.len(), 3);
        assert!(table.index.contains_key(&lazy_worker));
        assert_eq!(table.worker_ranges(), vec![WorkerDpRange::new(1, 0, 2)]);
    }

    #[test]
    fn register_and_unregister_worker_are_strict() {
        let mut table = WorkerTable::new(4, &HashMap::new());
        let range = WorkerDpRange::new(1, 2, 2);

        let change = table.register_worker(4, range).unwrap();
        assert_eq!(
            change.added.into_iter().collect::<FxHashSet<_>>(),
            FxHashSet::from_iter([worker(1, 2), worker(1, 3)])
        );
        assert!(matches!(
            table.register_worker(4, range),
            Err(WorkerTopologyError::DuplicateWorker { worker_id: 1 })
        ));

        let change = table.unregister_worker(4, 1).unwrap();
        assert_eq!(change.removed.len(), 2);
        assert!(!table.has_registered_workers());
        assert!(matches!(
            table.unregister_worker(4, 1),
            Err(WorkerTopologyError::WorkerNotFound { worker_id: 1 })
        ));
    }

    #[test]
    fn invalid_ranges_are_rejected() {
        let mut table = WorkerTable::new(4, &HashMap::new());

        assert!(matches!(
            table.register_worker(4, WorkerDpRange::new(1, 0, 0)),
            Err(WorkerTopologyError::InvalidDpSize { worker_id: 1 })
        ));
        assert!(matches!(
            table.register_worker(4, WorkerDpRange::new(1, u32::MAX, 1)),
            Err(WorkerTopologyError::InvalidDpRange {
                worker_id: 1,
                dp_start: u32::MAX,
                dp_size: 1,
            })
        ));
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
    fn full_reconcile_removes_unregistered_lazy_workers() {
        let mut table = WorkerTable::new(4, &HashMap::from([(1, (0, 1))]));
        let lazy_worker = worker(2, 0);
        table.ensure_worker(4, lazy_worker);

        let change = table
            .reconcile(4, vec![WorkerDpRange::new(1, 0, 1)])
            .unwrap();

        assert_eq!(
            change
                .removed
                .iter()
                .map(|state| state.worker)
                .collect::<Vec<_>>(),
            vec![lazy_worker]
        );
        assert!(!table.index.contains_key(&lazy_worker));
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
            let outcome = seq.add_request_with_prefill_tracking(
                "req-1".to_string(),
                Some(vec![1, 2, 3]),
                None,
                true,
                Some(crate::protocols::PrefillLoadHint {
                    initial_effective_prefill_tokens: 12,
                    expected_prefill_duration: None,
                }),
                Instant::now(),
            );
            assert_eq!(outcome.membership_delta.stores[0].path, vec![1, 2, 3]);
        }

        let change = table
            .reconcile(
                4,
                vec![WorkerDpRange::new(1, 0, 1), WorkerDpRange::new(3, 0, 1)],
            )
            .unwrap();

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
