// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independent diagnostic ledger for native replay's event-visible device cache.
//! Native event block hashes already include the prefix chain; token-only hashes do not.

use std::collections::{BTreeMap, BTreeSet};

use dynamo_kv_router::protocols::{KvCacheEventData, RouterEvent, WorkerId, WorkerWithDpRank};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;

#[derive(Clone, Default, Serialize)]
struct Counters {
    store_events: u64,
    stored_block_mentions: u64,
    added_copies: u64,
    duplicate_stores: u64,
    removal_events: u64,
    removed_block_mentions: u64,
    removed_copies: u64,
    unknown_removals: u64,
    global_final_copy_removals: u64,
    clear_events: u64,
    cleared_copies: u64,
    worker_removals: u64,
    topology_removed_copies: u64,
}

#[derive(Serialize)]
struct WorkerSnapshot {
    worker_id: WorkerId,
    dp_rank: u32,
    resident_copies: usize,
    capacity: u64,
    occupancy_ratio: f64,
    added_copies: u64,
    removed_copies: u64,
}

#[derive(Serialize)]
struct Snapshot {
    time_secs: f64,
    resident_copies: usize,
    unique_blocks: usize,
    excess_copies: usize,
    replication_factor: f64,
    total_capacity: u64,
    occupancy_ratio: f64,
    workers: Vec<WorkerSnapshot>,
    counters: Counters,
}

#[derive(Default)]
pub(super) struct ResidentCache {
    workers: BTreeMap<WorkerWithDpRank, FxHashSet<u64>>,
    references: FxHashMap<u64, usize>,
    capacities: BTreeMap<WorkerWithDpRank, u64>,
    rank_counters: BTreeMap<WorkerWithDpRank, (u64, u64)>,
    copies: usize,
    counters: Counters,
    peak_resident_copies: usize,
    peak_excess_copies: usize,
    latest_time_secs: f64,
    series: Vec<Snapshot>,
}

impl ResidentCache {
    pub(super) fn add_worker(&mut self, worker: WorkerWithDpRank, capacity: u64) {
        self.capacities.insert(worker, capacity);
    }

    pub(super) fn remove_worker(&mut self, worker_id: WorkerId) {
        let ranks = self
            .workers
            .keys()
            .filter(|worker| worker.worker_id == worker_id)
            .copied()
            .collect::<Vec<_>>();
        for rank in ranks {
            self.counters.topology_removed_copies += self.clear_rank(rank) as u64;
        }
        self.capacities
            .retain(|worker, _| worker.worker_id != worker_id);
        self.counters.worker_removals += 1;
    }

    pub(super) fn observe(&mut self, event: &RouterEvent) {
        if !event.storage_tier.is_gpu() {
            return;
        }
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        match &event.event.data {
            KvCacheEventData::Stored(stored) => {
                self.counters.store_events += 1;
                self.counters.stored_block_mentions += stored.blocks.len() as u64;
                let present = self.workers.entry(worker).or_default();
                for block in &stored.blocks {
                    if !present.insert(block.block_hash.0) {
                        self.counters.duplicate_stores += 1;
                        continue;
                    }
                    *self.references.entry(block.block_hash.0).or_default() += 1;
                    self.copies += 1;
                    self.counters.added_copies += 1;
                    self.rank_counters.entry(worker).or_default().0 += 1;
                }
                self.peak_resident_copies = self.peak_resident_copies.max(self.copies);
                self.peak_excess_copies = self
                    .peak_excess_copies
                    .max(self.copies.saturating_sub(self.references.len()));
            }
            KvCacheEventData::Removed(removed) => {
                self.counters.removal_events += 1;
                self.counters.removed_block_mentions += removed.block_hashes.len() as u64;
                for hash in &removed.block_hashes {
                    let removed = self
                        .workers
                        .get_mut(&worker)
                        .is_some_and(|present| present.remove(&hash.0));
                    if !removed {
                        self.counters.unknown_removals += 1;
                        continue;
                    }
                    self.counters.removed_copies += 1;
                    self.rank_counters.entry(worker).or_default().1 += 1;
                    self.counters.global_final_copy_removals += u64::from(self.release(hash.0));
                }
            }
            KvCacheEventData::Cleared => {
                self.counters.clear_events += 1;
                self.counters.cleared_copies += self.clear_rank(worker) as u64;
            }
        }
    }

    fn release(&mut self, hash: u64) -> bool {
        let references = self
            .references
            .get_mut(&hash)
            .expect("resident worker entry must have a global reference");
        *references -= 1;
        self.copies -= 1;
        if *references > 0 {
            return false;
        }
        self.references.remove(&hash);
        true
    }

    fn clear_rank(&mut self, worker: WorkerWithDpRank) -> usize {
        let Some(present) = self.workers.remove(&worker) else {
            return 0;
        };
        let count = present.len();
        for hash in present {
            self.release(hash);
        }
        count
    }

    pub(super) fn sample(&mut self, time_secs: f64) {
        self.latest_time_secs = time_secs;
        let snapshot = self.snapshot();
        if self
            .series
            .last()
            .is_some_and(|previous| previous.time_secs.floor() == time_secs.floor())
        {
            *self.series.last_mut().unwrap() = snapshot;
        } else {
            self.series.push(snapshot);
        }
    }

    fn snapshot(&self) -> Snapshot {
        let ranks = self
            .workers
            .keys()
            .chain(self.capacities.keys())
            .chain(self.rank_counters.keys())
            .copied()
            .collect::<BTreeSet<_>>();
        let workers = ranks
            .into_iter()
            .map(|worker| {
                let resident_copies = self.workers.get(&worker).map_or(0, FxHashSet::len);
                let capacity = self.capacities.get(&worker).copied().unwrap_or_default();
                let (added_copies, removed_copies) =
                    self.rank_counters.get(&worker).copied().unwrap_or_default();
                WorkerSnapshot {
                    worker_id: worker.worker_id,
                    dp_rank: worker.dp_rank,
                    resident_copies,
                    capacity,
                    occupancy_ratio: ratio(resident_copies, capacity as usize),
                    added_copies,
                    removed_copies,
                }
            })
            .collect();
        let total_capacity = self.capacities.values().sum::<u64>();
        Snapshot {
            time_secs: self.latest_time_secs,
            resident_copies: self.copies,
            unique_blocks: self.references.len(),
            excess_copies: self.copies.saturating_sub(self.references.len()),
            replication_factor: ratio(self.copies, self.references.len()),
            total_capacity,
            occupancy_ratio: ratio(self.copies, total_capacity as usize),
            workers,
            counters: self.counters.clone(),
        }
    }

    pub(super) fn diagnostics(&self) -> serde_json::Value {
        serde_json::json!({
            "measurement": "event_visible_device_sequence_block_presence",
            "limitations": [
                "observed at native replay event boundaries, not intra-pass physical occupancy",
                "excludes partial, private, uncomputed blocks and same-rank physical duplicates",
                "includes published full input and generated-output blocks",
                "removal counters reflect events; event schema alone does not prove removal cause"
            ],
            "final": self.snapshot(),
            "series": self.series,
            "peak_resident_copies": self.peak_resident_copies,
            "peak_excess_copies": self.peak_excess_copies,
        })
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        return 0.0;
    }
    numerator as f64 / denominator as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheRemoveData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, StorageTier,
    };

    fn event(worker: u64, rank: u32, data: KvCacheEventData) -> RouterEvent {
        RouterEvent::with_storage_tier(
            worker,
            KvCacheEvent {
                event_id: 0,
                data,
                dp_rank: rank,
            },
            StorageTier::Device,
        )
    }

    fn store(hashes: &[u64]) -> KvCacheEventData {
        KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: hashes
                .iter()
                .map(|hash| KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(*hash),
                    tokens_hash: LocalBlockHash(7),
                    mm_extra_info: None,
                })
                .collect(),
        })
    }

    fn remove(hashes: &[u64]) -> KvCacheEventData {
        KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: hashes
                .iter()
                .copied()
                .map(ExternalSequenceBlockHash)
                .collect(),
        })
    }

    #[test]
    fn resident_cache_counts_sequence_copies_and_idempotent_stores() {
        let mut ledger = ResidentCache::default();
        ledger.add_worker(WorkerWithDpRank::new(0, 0), 8);
        ledger.add_worker(WorkerWithDpRank::new(0, 1), 8);
        ledger.observe(&event(0, 0, store(&[1, 2, 2])));
        ledger.observe(&event(0, 0, store(&[1])));
        ledger.observe(&event(0, 1, store(&[1])));
        let snapshot = ledger.snapshot();
        assert_eq!(
            (
                snapshot.resident_copies,
                snapshot.unique_blocks,
                snapshot.excess_copies
            ),
            (3, 2, 1)
        );
        assert_eq!(snapshot.replication_factor, 1.5);
        assert_eq!(snapshot.occupancy_ratio, 3.0 / 16.0);
        assert_eq!(snapshot.counters.duplicate_stores, 2);
        assert_eq!(
            snapshot
                .workers
                .iter()
                .map(|worker| worker.resident_copies)
                .collect::<Vec<_>>(),
            vec![2, 1]
        );
    }

    #[test]
    fn resident_cache_removal_distinguishes_global_last_copy_and_unknown_hashes() {
        let mut ledger = ResidentCache::default();
        ledger.observe(&event(0, 0, store(&[1])));
        ledger.observe(&event(1, 0, store(&[1])));
        ledger.observe(&event(0, 0, remove(&[1, 1])));
        assert_eq!((ledger.copies, ledger.references.len()), (1, 1));
        assert_eq!(ledger.counters.global_final_copy_removals, 0);
        assert_eq!(ledger.counters.unknown_removals, 1);
        ledger.observe(&event(1, 0, remove(&[1])));
        assert_eq!((ledger.copies, ledger.references.len()), (0, 0));
        assert_eq!(ledger.counters.removed_copies, 2);
        assert_eq!(ledger.counters.global_final_copy_removals, 1);
    }

    #[test]
    fn resident_cache_clear_is_rank_scoped_and_separate_from_eviction() {
        let mut ledger = ResidentCache::default();
        ledger.add_worker(WorkerWithDpRank::new(0, 0), 8);
        ledger.add_worker(WorkerWithDpRank::new(0, 1), 8);
        ledger.observe(&event(0, 0, store(&[1, 2])));
        ledger.observe(&event(0, 1, store(&[1, 3])));
        ledger.observe(&event(0, 0, KvCacheEventData::Cleared));
        assert_eq!((ledger.copies, ledger.references.len()), (2, 2));
        assert_eq!(ledger.counters.cleared_copies, 2);
        assert_eq!(ledger.counters.removed_copies, 0);
        ledger.remove_worker(0);
        assert!(ledger.references.is_empty());
        assert_eq!(ledger.snapshot().total_capacity, 0);
        assert_eq!(ledger.counters.topology_removed_copies, 2);
    }

    #[test]
    fn resident_cache_samples_latest_virtual_second_and_ignores_other_tiers() {
        let mut ledger = ResidentCache::default();
        let mut host = event(0, 0, store(&[1]));
        host.storage_tier = StorageTier::HostPinned;
        ledger.observe(&host);
        ledger.sample(10.1);
        ledger.observe(&event(0, 0, store(&[2])));
        ledger.sample(10.9);
        ledger.observe(&event(0, 0, remove(&[2])));
        ledger.sample(300.0);
        assert_eq!(ledger.series.len(), 2);
        assert_eq!(ledger.series[0].time_secs, 10.9);
        assert_eq!(ledger.series[0].resident_copies, 1);
        assert_eq!(ledger.series[1].counters.removed_copies, 1);
        assert_eq!(ledger.series[1].time_secs, 300.0);
    }
}
