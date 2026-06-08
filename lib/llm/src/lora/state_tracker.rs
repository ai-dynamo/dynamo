// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA State Tracker
//!
//! Tracks the runtime state of LoRA adapters across workers by watching MDC discovery
//! events. Maintains which LoRAs are loaded on which workers and per-worker LoRA capacity.
//!
//! ## MDC Event Model
//!
//! Each MDC entry corresponds to a single `(worker, lora_name)` pair carrying one
//! `LoraInfo`. The tracker handles three discrete event types:
//!
//! - `handle_mdc_addition`: a worker registered (or re-published) a LoRA adapter.
//! - `handle_mdc_removal`: a worker unregistered one specific LoRA adapter.
//! - `handle_worker_removal`: a worker left the cluster (drops all of its LoRAs).
//!
//! `handle_mdc_addition` is purely additive and idempotent — re-publishing the same
//! `(worker, lora_name)` updates the stored `LoraInfo` in place. State reconciliation
//! when a worker drops an adapter is the responsibility of the corresponding removal
//! event, not the addition handler.
//!
//! ## Worker Capacity Invariant
//!
//! `LoraInfo::max_gpu_lora_count` is a *worker-level* property (the engine's
//! `--max-loras` setting; see `docs/dev/dep/000N-lora-placement/lora-allocation-v2.md`),
//! but is duplicated into every `LoraInfo` a
//! worker publishes for convenience. All `LoraInfo` values coming from the same
//! worker MUST carry the same `max_gpu_lora_count`. If a mismatch is observed
//! across updates, we log a warning and adopt the latest value — but this should
//! never happen with a correctly-configured worker.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use dashmap::DashMap;

use crate::kv_router::protocols::WorkerWithDpRank;
use crate::model_card::LoraInfo;

const DEFAULT_MAX_GPU_LORA_COUNT: u32 = 4;

/// Tracks the loaded state of LoRAs across workers and per-worker capacity.
///
/// Concurrent data structure updated from MDC discovery events and read from
/// the allocation controller and filter.
///
/// ## Cross-map consistency
///
/// State is spread across four `DashMap`s that must stay mutually consistent
/// (e.g. `loaded_locations` and `worker_to_loras` are inverse indexes of the
/// same fact). Per-map atomicity is not enough: a concurrent addition and
/// removal touching the same `(worker, lora)` could interleave their
/// individual map writes and leave the indexes disagreeing. All three mutating
/// handlers therefore serialize on `write_lock` for the duration of their
/// multi-map update, so writers observe a consistent snapshot relative to one
/// another. Readers stay lock-free on the individual `DashMap`s.
#[derive(Clone)]
pub struct LoraStateTracker {
    /// LoRA name -> set of workers where it is loaded
    loaded_locations: Arc<DashMap<String, HashSet<WorkerWithDpRank>>>,
    /// LoRA name, worker -> LoraInfo from MDC
    lora_info: Arc<DashMap<(String, WorkerWithDpRank), LoraInfo>>,
    /// Worker -> set of LoRA names loaded on that worker
    worker_to_loras: Arc<DashMap<WorkerWithDpRank, HashSet<String>>>,
    /// Worker -> max_gpu_lora_count capacity
    worker_capacity: Arc<DashMap<WorkerWithDpRank, u32>>,
    /// Serializes the multi-map mutations in the `handle_*` methods so the four
    /// indexes above can never be left mutually inconsistent by interleaved
    /// writers. Reads do not take this lock.
    write_lock: Arc<Mutex<()>>,
}

impl LoraStateTracker {
    pub fn new() -> Self {
        Self {
            loaded_locations: Arc::new(DashMap::new()),
            lora_info: Arc::new(DashMap::new()),
            worker_to_loras: Arc::new(DashMap::new()),
            worker_capacity: Arc::new(DashMap::new()),
            write_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Acquire the writer-serialization lock, tolerating poisoning (a prior
    /// writer panic must not wedge all future updates).
    fn lock_writes(&self) -> std::sync::MutexGuard<'_, ()> {
        self.write_lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Handle an MDC addition event: a worker registered (or re-published) a LoRA adapter.
    ///
    /// Each MDC entry uniquely identifies a `(worker, lora_name)` pair, so this
    /// function is purely additive and idempotent: re-publishing the same pair
    /// updates the stored `LoraInfo` in place. State reconciliation when a worker
    /// drops an adapter is handled by [`handle_mdc_removal`](Self::handle_mdc_removal),
    /// and full worker departure by [`handle_worker_removal`](Self::handle_worker_removal).
    ///
    /// `lora.max_gpu_lora_count` is treated as a worker-level capacity: see the
    /// "Worker Capacity Invariant" note in the module docs. A mismatch against a
    /// previously-recorded capacity for the same worker is logged at warn level
    /// and the latest value is adopted.
    pub fn handle_mdc_addition(&self, worker: WorkerWithDpRank, lora: &LoraInfo) {
        let _guard = self.lock_writes();
        let lora_name = lora.name.clone();

        self.loaded_locations
            .entry(lora_name.clone())
            .or_default()
            .insert(worker);

        self.lora_info
            .insert((lora_name.clone(), worker), lora.clone());

        self.worker_to_loras
            .entry(worker)
            .or_default()
            .insert(lora_name);

        let capacity = lora
            .max_gpu_lora_count
            .unwrap_or(DEFAULT_MAX_GPU_LORA_COUNT);
        if let Some(prev) = self.worker_capacity.get(&worker)
            && *prev != capacity
        {
            tracing::warn!(
                worker_id = worker.worker_id,
                dp_rank = worker.dp_rank,
                lora_name = lora.name,
                previous_capacity = *prev,
                new_capacity = capacity,
                "Worker capacity changed across MDC updates"
            );
        }
        self.worker_capacity.insert(worker, capacity);

        tracing::debug!(
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            lora_name = lora.name,
            capacity = capacity,
            "LoRA state tracker: MDC addition"
        );
    }

    /// Handle an MDC removal event: a worker unregistered a LoRA adapter.
    pub fn handle_mdc_removal(&self, worker: WorkerWithDpRank, lora_name: &str) {
        let _guard = self.lock_writes();
        let became_empty = if let Some(mut workers) = self.loaded_locations.get_mut(lora_name) {
            workers.remove(&worker);
            workers.is_empty()
        } else {
            false
        };
        if became_empty {
            // remove_if re-checks the predicate under the shard lock, so a
            // concurrent handle_mdc_addition that races between the is_empty
            // check above and this call cannot have its entry silently deleted.
            self.loaded_locations
                .remove_if(lora_name, |_, v| v.is_empty());
        }

        self.lora_info.remove(&(lora_name.to_string(), worker));

        let became_empty = if let Some(mut loras) = self.worker_to_loras.get_mut(&worker) {
            loras.remove(lora_name);
            loras.is_empty()
        } else {
            false
        };
        if became_empty {
            // remove_if re-checks the predicate under the shard lock, so a
            // concurrent handle_mdc_addition that races between the drop above
            // and this call cannot have its newly inserted entry deleted.
            self.worker_to_loras.remove_if(&worker, |_, v| v.is_empty());
        }

        tracing::debug!(
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            lora_name = lora_name,
            "LoRA state tracker: MDC removed"
        );
    }

    /// Handle a worker being completely removed.
    pub fn handle_worker_removal(&self, worker: WorkerWithDpRank) {
        let _guard = self.lock_writes();

        // Remove the capacity entry FIRST. `worker_capacity` is the enumeration
        // spine for every slot/capacity reader (`workers_with_free_slots`,
        // `list_workers`, `get_worker_slot_usage`, `slot_info`), so dropping it
        // first makes the worker vanish from those snapshots atomically. If we
        // cleared `worker_to_loras` first instead, a concurrent reader could
        // momentarily see the worker as capacity-present with zero loaded LoRAs
        // and wrongly report it as having free slots.
        self.worker_capacity.remove(&worker);

        if let Some((_, loras)) = self.worker_to_loras.remove(&worker) {
            for lora_name in &loras {
                let became_empty =
                    if let Some(mut workers) = self.loaded_locations.get_mut(lora_name) {
                        workers.remove(&worker);
                        workers.is_empty()
                    } else {
                        false
                    };
                if became_empty {
                    self.loaded_locations
                        .remove_if(lora_name, |_, v| v.is_empty());
                }
                self.lora_info.remove(&(lora_name.clone(), worker));
            }
        }

        tracing::debug!(
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            "LoRA state tracker: worker removed"
        );
    }

    pub fn get_loaded_workers(&self, lora_name: &str) -> HashSet<WorkerWithDpRank> {
        self.loaded_locations
            .get(lora_name)
            .map(|entry| entry.value().clone())
            .unwrap_or_default()
    }

    pub fn is_loaded(&self, lora_name: &str, worker: &WorkerWithDpRank) -> bool {
        self.loaded_locations
            .get(lora_name)
            .map(|entry| entry.contains(worker))
            .unwrap_or(false)
    }

    pub fn list_loras(&self) -> Vec<String> {
        self.loaded_locations
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    pub fn list_workers(&self) -> Vec<WorkerWithDpRank> {
        self.worker_capacity
            .iter()
            .map(|entry| *entry.key())
            .collect()
    }

    fn slot_info(&self, worker: &WorkerWithDpRank) -> (u32, u32) {
        let capacity = self.worker_capacity.get(worker).map(|v| *v).unwrap_or(0);
        let loaded = self
            .worker_to_loras
            .get(worker)
            .map(|v| v.len() as u32)
            .unwrap_or(0);
        (loaded, capacity)
    }

    pub fn free_slots(&self, worker: &WorkerWithDpRank) -> u32 {
        let (loaded, capacity) = self.slot_info(worker);
        capacity.saturating_sub(loaded)
    }

    pub fn total_lora_slots(&self) -> u32 {
        self.worker_capacity
            .iter()
            .map(|entry| *entry.value())
            .sum()
    }

    pub fn get_worker_capacities(&self) -> HashMap<WorkerWithDpRank, u32> {
        self.worker_capacity
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect()
    }

    pub fn get_worker_slot_usage(&self) -> HashMap<WorkerWithDpRank, (usize, usize)> {
        // Snapshot capacities first, releasing all worker_capacity shard guards
        // before touching worker_to_loras. Reading a second DashMap while
        // holding an iterator guard risks a lock-order deadlock against a
        // writer that locks the two maps in the opposite order.
        let caps: Vec<(WorkerWithDpRank, u32)> = self
            .worker_capacity
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect();
        caps.into_iter()
            .filter_map(|(worker, cap)| {
                let loaded = self.loaded_count(&worker);
                // Drop workers removed since the snapshot. handle_worker_removal
                // clears worker_capacity before worker_to_loras, so if loaded
                // dropped to 0 due to a concurrent removal, the capacity entry
                // is already gone and this recheck filters out the stale row.
                self.worker_capacity
                    .contains_key(&worker)
                    .then_some((worker, (loaded, cap as usize)))
            })
            .collect()
    }

    pub fn workers_with_free_slots(&self) -> Vec<WorkerWithDpRank> {
        // Snapshot first (see get_worker_slot_usage) to avoid holding a
        // worker_capacity iterator guard while reading worker_to_loras.
        let caps: Vec<(WorkerWithDpRank, u32)> = self
            .worker_capacity
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect();
        caps.into_iter()
            .filter(|(worker, capacity)| {
                // loaded_count first, then re-confirm the worker still exists:
                // a worker removed since the snapshot has its capacity cleared
                // before worker_to_loras, so a transient loaded==0 cannot make
                // a removed worker look free.
                (self.loaded_count(worker) as u32) < *capacity
                    && self.worker_capacity.contains_key(worker)
            })
            .map(|(worker, _)| worker)
            .collect()
    }

    pub fn loaded_count(&self, worker: &WorkerWithDpRank) -> usize {
        self.worker_to_loras
            .get(worker)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.worker_capacity.is_empty()
    }
}

impl Default for LoraStateTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_card::LoraInfo;

    fn make_worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    fn make_lora_info(name: &str, max_count: Option<u32>) -> LoraInfo {
        LoraInfo {
            name: name.to_string(),
            max_gpu_lora_count: max_count,
        }
    }

    #[test]
    fn test_mdc_update_and_query() {
        let tracker = LoraStateTracker::new();
        let w1 = make_worker(1);
        let lora = make_lora_info("lora-math", Some(8));

        tracker.handle_mdc_addition(w1, &lora);

        assert!(!tracker.is_empty());
        assert_eq!(tracker.list_workers().len(), 1);
        assert_eq!(tracker.list_loras(), vec!["lora-math"]);
        assert!(tracker.is_loaded("lora-math", &w1));
        assert_eq!(tracker.total_lora_slots(), 8);
        assert_eq!(tracker.free_slots(&w1), 7);
    }

    #[test]
    fn test_multiple_workers_same_lora() {
        let tracker = LoraStateTracker::new();
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        let lora = make_lora_info("lora-code", Some(4));

        tracker.handle_mdc_addition(w1, &lora);
        tracker.handle_mdc_addition(w2, &lora);

        let loaded = tracker.get_loaded_workers("lora-code");
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains(&w1));
        assert!(loaded.contains(&w2));
        assert_eq!(tracker.total_lora_slots(), 8);
    }

    #[test]
    fn test_mdc_removal() {
        let tracker = LoraStateTracker::new();
        let w1 = make_worker(1);
        let lora = make_lora_info("lora-math", Some(4));

        tracker.handle_mdc_addition(w1, &lora);
        assert!(tracker.is_loaded("lora-math", &w1));

        tracker.handle_mdc_removal(w1, "lora-math");
        assert!(!tracker.is_loaded("lora-math", &w1));
        assert!(tracker.list_loras().is_empty());
    }

    #[test]
    fn test_worker_removal() {
        let tracker = LoraStateTracker::new();
        let w1 = make_worker(1);
        let lora1 = make_lora_info("lora-a", Some(4));
        let lora2 = make_lora_info("lora-b", Some(4));

        tracker.handle_mdc_addition(w1, &lora1);
        tracker.handle_mdc_addition(w1, &lora2);

        assert_eq!(tracker.loaded_count(&w1), 2);
        assert_eq!(tracker.free_slots(&w1), 2);

        tracker.handle_worker_removal(w1);
        assert!(tracker.is_empty());
        assert!(tracker.list_loras().is_empty());
    }

    #[test]
    fn test_slot_usage() {
        let tracker = LoraStateTracker::new();
        let w1 = make_worker(1);
        let lora1 = make_lora_info("lora-a", Some(8));
        let lora2 = make_lora_info("lora-b", Some(8));

        tracker.handle_mdc_addition(w1, &lora1);
        tracker.handle_mdc_addition(w1, &lora2);

        let usage = tracker.get_worker_slot_usage();
        assert_eq!(usage.get(&w1), Some(&(2, 8)));
    }

    #[test]
    fn test_workers_with_free_slots() {
        let tracker = LoraStateTracker::new();
        let w1 = make_worker(1);
        let w2 = make_worker(2);

        // w1 has capacity 1, load 1 lora → 0 free slots
        let lora1 = make_lora_info("lora-a", Some(1));
        tracker.handle_mdc_addition(w1, &lora1);

        // w2 has capacity 4, load 1 lora → 3 free slots
        let lora2 = make_lora_info("lora-b", Some(4));
        tracker.handle_mdc_addition(w2, &lora2);

        let free = tracker.workers_with_free_slots();
        assert_eq!(free.len(), 1);
        assert!(free.contains(&w2));
    }

    #[test]
    fn test_concurrent_add_remove_keeps_indexes_consistent() {
        // Hammer the tracker with concurrent additions and removals across many
        // (worker, lora) pairs, then assert the two inverse indexes
        // (loaded_locations and worker_to_loras) agree. Without writer
        // serialization, interleaved multi-map updates could leave them
        // disagreeing; the write_lock prevents that.
        use std::thread;

        let tracker = LoraStateTracker::new();
        let workers = 8u64;
        let loras = 8u64;
        let iters = 200;

        let mut handles = Vec::new();
        for t in 0..workers {
            let tk = tracker.clone();
            handles.push(thread::spawn(move || {
                let w = make_worker(t);
                for i in 0..iters {
                    let lname = format!("lora-{}", i % loras);
                    let info = make_lora_info(&lname, Some(loras as u32));
                    tk.handle_mdc_addition(w, &info);
                    if i % 3 == 0 {
                        tk.handle_mdc_removal(w, &lname);
                    }
                    if i % 50 == 49 {
                        tk.handle_worker_removal(w);
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }

        // Invariant: every (lora -> worker) entry in loaded_locations has a
        // matching (worker -> lora) entry in worker_to_loras, and vice versa.
        for lora in tracker.list_loras() {
            for w in tracker.get_loaded_workers(&lora) {
                let loras_on_w = tracker
                    .worker_to_loras
                    .get(&w)
                    .map(|s| s.contains(&lora))
                    .unwrap_or(false);
                assert!(
                    loras_on_w,
                    "loaded_locations says {lora} on {w:?} but worker_to_loras disagrees"
                );
            }
        }
        for entry in tracker.worker_to_loras.iter() {
            let w = *entry.key();
            for lora in entry.value() {
                assert!(
                    tracker.is_loaded(lora, &w),
                    "worker_to_loras says {lora} on {w:?} but loaded_locations disagrees"
                );
            }
        }
    }
}
