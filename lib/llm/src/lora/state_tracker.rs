// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA State Tracker
//!
//! Publishes a coherent, immutable view of loaded LoRA adapters and worker capacity.
//! Discovery replaces the complete committed projection for an endpoint; legacy
//! event-style mutation methods remain for in-process callers. Readers pin one
//! [`LoraObservedSnapshot`] so a routing decision cannot mix indexes from different
//! discovery generations.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;

use crate::kv_router::protocols::WorkerWithDpRank;
use crate::model_card::LoraInfo;

const DEFAULT_MAX_GPU_LORA_COUNT: u32 = 4;

#[derive(Clone, Debug, Default)]
pub struct LoraObservedSnapshot {
    incarnation: u64,
    loaded_locations: HashMap<String, HashSet<WorkerWithDpRank>>,
    lora_info: HashMap<(String, WorkerWithDpRank), LoraInfo>,
    worker_to_loras: HashMap<WorkerWithDpRank, HashSet<String>>,
    worker_capacity: HashMap<WorkerWithDpRank, u32>,
}

impl LoraObservedSnapshot {
    pub fn incarnation(&self) -> u64 {
        self.incarnation
    }

    pub fn get_loaded_workers(&self, lora_name: &str) -> HashSet<WorkerWithDpRank> {
        self.loaded_locations
            .get(lora_name)
            .cloned()
            .unwrap_or_default()
    }

    pub fn is_loaded(&self, lora_name: &str, worker: &WorkerWithDpRank) -> bool {
        self.loaded_locations
            .get(lora_name)
            .is_some_and(|workers| workers.contains(worker))
    }

    pub fn list_loras(&self) -> Vec<String> {
        self.loaded_locations.keys().cloned().collect()
    }

    pub fn list_workers(&self) -> Vec<WorkerWithDpRank> {
        self.worker_capacity.keys().copied().collect()
    }

    fn slot_info(&self, worker: &WorkerWithDpRank) -> (u32, u32) {
        let capacity = self.worker_capacity.get(worker).copied().unwrap_or(0);
        let loaded = self
            .worker_to_loras
            .get(worker)
            .map(|loras| loras.len() as u32)
            .unwrap_or(0);
        (loaded, capacity)
    }

    pub fn free_slots(&self, worker: &WorkerWithDpRank) -> u32 {
        let (loaded, capacity) = self.slot_info(worker);
        capacity.saturating_sub(loaded)
    }

    pub fn total_lora_slots(&self) -> u32 {
        self.worker_capacity.values().sum()
    }

    pub fn get_worker_capacities(&self) -> HashMap<WorkerWithDpRank, u32> {
        self.worker_capacity.clone()
    }

    pub fn get_worker_slot_usage(&self) -> HashMap<WorkerWithDpRank, (usize, usize)> {
        self.worker_capacity
            .iter()
            .map(|(worker, capacity)| (*worker, (self.loaded_count(worker), *capacity as usize)))
            .collect()
    }

    pub fn workers_with_free_slots(&self) -> Vec<WorkerWithDpRank> {
        self.worker_capacity
            .iter()
            .filter_map(|(worker, capacity)| {
                ((self.loaded_count(worker) as u32) < *capacity).then_some(*worker)
            })
            .collect()
    }

    pub fn loaded_count(&self, worker: &WorkerWithDpRank) -> usize {
        self.worker_to_loras
            .get(worker)
            .map(HashSet::len)
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.worker_capacity.is_empty()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LoraWorkerProjection {
    pub(crate) capacity: u32,
    pub(crate) loras: Vec<LoraInfo>,
}

/// Tracks one endpoint's complete observed LoRA state.
///
/// Writers build a new immutable generation under `write_lock` and publish it
/// with one pointer swap. A request or allocation tick can therefore pin one
/// [`LoraObservedSnapshot`] and never combine indexes from different discovery
/// generations.
#[derive(Clone)]
pub struct LoraStateTracker {
    observed: Arc<ArcSwap<LoraObservedSnapshot>>,
    write_lock: Arc<Mutex<()>>,
}

impl LoraStateTracker {
    pub fn new() -> Self {
        Self {
            observed: Arc::new(ArcSwap::from_pointee(LoraObservedSnapshot::default())),
            write_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Acquire the writer-serialization lock, tolerating poisoning (a prior
    /// writer panic must not wedge all future updates).
    fn lock_writes(&self) -> std::sync::MutexGuard<'_, ()> {
        self.write_lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn snapshot(&self) -> Arc<LoraObservedSnapshot> {
        self.observed.load_full()
    }

    fn mutate(&self, update: impl FnOnce(&mut LoraObservedSnapshot)) {
        let _guard = self.lock_writes();
        let current = self.observed.load_full();
        let was_empty = current.is_empty();
        let mut next = (*current).clone();
        update(&mut next);
        if was_empty && !next.is_empty() {
            next.incarnation = next.incarnation.wrapping_add(1).max(1);
        }
        self.observed.store(Arc::new(next));
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
        let capacity = lora.max_gpu_lora_count.unwrap_or_else(|| {
            tracing::warn!(
                worker_id = worker.worker_id,
                dp_rank = worker.dp_rank,
                lora_name = lora.name,
                default = DEFAULT_MAX_GPU_LORA_COUNT,
                "LoRA MDC carries no max_gpu_lora_count; using default for placement capacity. \
                 The worker backend should publish its configured per-worker LoRA slot count \
                 (e.g. vLLM --max-loras) so allocation reflects real capacity."
            );
            DEFAULT_MAX_GPU_LORA_COUNT
        });
        self.mutate(|next| {
            let lora_name = lora.name.clone();
            next.loaded_locations
                .entry(lora_name.clone())
                .or_default()
                .insert(worker);
            next.lora_info
                .insert((lora_name.clone(), worker), lora.clone());
            next.worker_to_loras
                .entry(worker)
                .or_default()
                .insert(lora_name);
            record_worker_capacity(next, worker, capacity);
        });

        tracing::debug!(
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            lora_name = lora.name,
            capacity = capacity,
            "LoRA state tracker: MDC addition"
        );
    }

    /// Set a worker's LoRA slot capacity directly, without registering any adapter on it.
    ///
    /// Capacity-only counterpart to `handle_mdc_addition` (which records a loaded adapter AND
    /// sets capacity). Makes the worker appear in `list_workers` with the given capacity and no
    /// loaded LoRAs, so callers can establish cluster topology / per-worker `max_gpu_lora_count`
    /// without consuming a slot with a phantom adapter.
    pub fn set_worker_capacity(&self, worker: WorkerWithDpRank, capacity: u32) {
        self.mutate(|next| record_worker_capacity(next, worker, capacity));
    }

    /// Replace one retained worker's complete committed LoRA projection.
    ///
    /// Desired entries are installed before obsolete entries are withdrawn, so lock-free
    /// routing readers never observe an unchanged adapter as temporarily unloaded.
    #[cfg(test)]
    pub(crate) fn replace_worker_projection(
        &self,
        worker: WorkerWithDpRank,
        base_capacity: Option<u32>,
        loras: &[LoraInfo],
    ) {
        let capacity = effective_capacity(worker, base_capacity, loras);
        let projection = capacity.map(|capacity| LoraWorkerProjection {
            capacity,
            loras: loras.to_vec(),
        });
        self.mutate(|next| replace_worker(next, worker, projection));
    }

    pub(crate) fn replace_endpoint_projection(
        &self,
        projections: HashMap<WorkerWithDpRank, LoraWorkerProjection>,
    ) {
        let _guard = self.lock_writes();
        let current = self.observed.load_full();
        let mut next = LoraObservedSnapshot {
            incarnation: current.incarnation,
            ..Default::default()
        };
        for (worker, projection) in projections {
            replace_worker(&mut next, worker, Some(projection));
        }
        if current.is_empty() && !next.is_empty() {
            next.incarnation = next.incarnation.wrapping_add(1).max(1);
        }
        self.observed.store(Arc::new(next));
    }

    /// Handle an MDC removal event: a worker unregistered a LoRA adapter.
    pub fn handle_mdc_removal(&self, worker: WorkerWithDpRank, lora_name: &str) {
        self.mutate(|next| remove_lora(next, worker, lora_name));

        tracing::debug!(
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            lora_name = lora_name,
            "LoRA state tracker: MDC removed"
        );
    }

    /// Handle a worker being completely removed.
    pub fn handle_worker_removal(&self, worker: WorkerWithDpRank) {
        self.mutate(|next| replace_worker(next, worker, None));

        tracing::debug!(
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            "LoRA state tracker: worker removed"
        );
    }

    pub fn get_loaded_workers(&self, lora_name: &str) -> HashSet<WorkerWithDpRank> {
        self.snapshot().get_loaded_workers(lora_name)
    }

    pub fn is_loaded(&self, lora_name: &str, worker: &WorkerWithDpRank) -> bool {
        self.snapshot().is_loaded(lora_name, worker)
    }

    pub fn list_loras(&self) -> Vec<String> {
        self.snapshot().list_loras()
    }

    pub fn list_workers(&self) -> Vec<WorkerWithDpRank> {
        self.snapshot().list_workers()
    }

    pub fn free_slots(&self, worker: &WorkerWithDpRank) -> u32 {
        self.snapshot().free_slots(worker)
    }

    pub fn total_lora_slots(&self) -> u32 {
        self.snapshot().total_lora_slots()
    }

    pub fn get_worker_capacities(&self) -> HashMap<WorkerWithDpRank, u32> {
        self.snapshot().get_worker_capacities()
    }

    pub fn get_worker_slot_usage(&self) -> HashMap<WorkerWithDpRank, (usize, usize)> {
        self.snapshot().get_worker_slot_usage()
    }

    pub fn workers_with_free_slots(&self) -> Vec<WorkerWithDpRank> {
        self.snapshot().workers_with_free_slots()
    }

    pub fn loaded_count(&self, worker: &WorkerWithDpRank) -> usize {
        self.snapshot().loaded_count(worker)
    }

    pub fn is_empty(&self) -> bool {
        self.snapshot().is_empty()
    }
}

#[cfg(test)]
fn effective_capacity(
    worker: WorkerWithDpRank,
    base_capacity: Option<u32>,
    loras: &[LoraInfo],
) -> Option<u32> {
    if let Some(capacity) = base_capacity {
        return Some(capacity);
    }
    let mut assertions = loras
        .iter()
        .filter_map(|lora| lora.max_gpu_lora_count)
        .collect::<Vec<_>>();
    assertions.sort_unstable();
    assertions.dedup();
    if assertions.len() > 1 {
        tracing::warn!(
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            capacities = ?assertions,
            "LoRA adapter cards disagree on worker capacity; using the conservative minimum"
        );
    }
    assertions.first().copied().or_else(|| {
        (!loras.is_empty()).then(|| {
            tracing::warn!(
                worker_id = worker.worker_id,
                dp_rank = worker.dp_rank,
                default = DEFAULT_MAX_GPU_LORA_COUNT,
                "LoRA MDC carries no max_gpu_lora_count; using compatibility default"
            );
            DEFAULT_MAX_GPU_LORA_COUNT
        })
    })
}

fn record_worker_capacity(
    snapshot: &mut LoraObservedSnapshot,
    worker: WorkerWithDpRank,
    capacity: u32,
) {
    if let Some(previous) = snapshot.worker_capacity.get(&worker)
        && *previous != capacity
    {
        tracing::warn!(
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            previous_capacity = *previous,
            new_capacity = capacity,
            "Worker LoRA capacity changed across registrations"
        );
    }
    snapshot.worker_capacity.insert(worker, capacity);
}

fn remove_lora(snapshot: &mut LoraObservedSnapshot, worker: WorkerWithDpRank, lora_name: &str) {
    if let Some(workers) = snapshot.loaded_locations.get_mut(lora_name) {
        workers.remove(&worker);
        if workers.is_empty() {
            snapshot.loaded_locations.remove(lora_name);
        }
    }
    snapshot.lora_info.remove(&(lora_name.to_string(), worker));
    if let Some(loras) = snapshot.worker_to_loras.get_mut(&worker) {
        loras.remove(lora_name);
        if loras.is_empty() {
            snapshot.worker_to_loras.remove(&worker);
        }
    }
}

fn replace_worker(
    snapshot: &mut LoraObservedSnapshot,
    worker: WorkerWithDpRank,
    projection: Option<LoraWorkerProjection>,
) {
    let previous = snapshot
        .worker_to_loras
        .get(&worker)
        .cloned()
        .unwrap_or_default();
    let Some(projection) = projection else {
        snapshot.worker_capacity.remove(&worker);
        for name in previous {
            remove_lora(snapshot, worker, &name);
        }
        return;
    };

    snapshot.worker_capacity.insert(worker, projection.capacity);
    let desired = projection
        .loras
        .into_iter()
        .map(|lora| (lora.name.clone(), lora))
        .collect::<HashMap<_, _>>();
    let desired_names = desired.keys().cloned().collect::<HashSet<_>>();
    for (name, lora) in desired {
        snapshot
            .loaded_locations
            .entry(name.clone())
            .or_default()
            .insert(worker);
        snapshot.lora_info.insert((name, worker), lora);
    }
    if desired_names.is_empty() {
        snapshot.worker_to_loras.remove(&worker);
    } else {
        snapshot
            .worker_to_loras
            .insert(worker, desired_names.clone());
    }
    for name in previous.difference(&desired_names) {
        remove_lora(snapshot, worker, name);
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
    fn replace_worker_projection_reconciles_adapters_and_capacity() {
        let tracker = LoraStateTracker::new();
        let worker = make_worker(1);
        let retained = make_lora_info("retained", Some(4));
        let obsolete = make_lora_info("obsolete", Some(4));
        tracker.handle_mdc_addition(worker, &retained);
        tracker.handle_mdc_addition(worker, &obsolete);

        let retained = make_lora_info("retained", Some(6));
        let added = make_lora_info("added", Some(6));
        tracker.replace_worker_projection(worker, Some(6), &[retained, added]);

        assert!(tracker.is_loaded("retained", &worker));
        assert!(tracker.is_loaded("added", &worker));
        assert!(!tracker.is_loaded("obsolete", &worker));
        assert_eq!(tracker.loaded_count(&worker), 2);
        assert_eq!(tracker.free_slots(&worker), 4);
    }

    #[test]
    fn pinned_snapshot_survives_atomic_projection_replacement() {
        let tracker = LoraStateTracker::new();
        let worker = make_worker(1);
        tracker.replace_worker_projection(worker, Some(4), &[make_lora_info("old", Some(4))]);
        let pinned = tracker.snapshot();

        tracker.replace_worker_projection(worker, Some(8), &[make_lora_info("new", Some(8))]);

        assert!(pinned.is_loaded("old", &worker));
        assert!(!pinned.is_loaded("new", &worker));
        assert_eq!(pinned.total_lora_slots(), 4);
        let current = tracker.snapshot();
        assert!(!current.is_loaded("old", &worker));
        assert!(current.is_loaded("new", &worker));
        assert_eq!(current.total_lora_slots(), 8);
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

        let snapshot = tracker.snapshot();
        // Invariant: every (lora -> worker) entry in loaded_locations has a
        // matching (worker -> lora) entry in worker_to_loras, and vice versa.
        for lora in snapshot.list_loras() {
            for w in snapshot.get_loaded_workers(&lora) {
                let loras_on_w = snapshot
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
        for (w, loras) in &snapshot.worker_to_loras {
            for lora in loras {
                assert!(
                    snapshot.is_loaded(lora, w),
                    "worker_to_loras says {lora} on {w:?} but loaded_locations disagrees"
                );
            }
        }
    }
}
