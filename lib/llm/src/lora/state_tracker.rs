// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA State Tracker
//!
//! Tracks the runtime state of LoRA adapters across workers by watching MDC discovery
//! events. Maintains which LoRAs are loaded on which workers and per-worker LoRA capacity.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dashmap::DashMap;

use crate::kv_router::protocols::WorkerWithDpRank;
use crate::model_card::LoraInfo;

const DEFAULT_MAX_GPU_LORA_COUNT: u32 = 4;

/// Tracks the loaded state of LoRAs across workers and per-worker capacity.
///
/// Concurrent data structure updated from MDC discovery events and read from
/// the allocation controller and filter.
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
}

impl LoraStateTracker {
    pub fn new() -> Self {
        Self {
            loaded_locations: Arc::new(DashMap::new()),
            lora_info: Arc::new(DashMap::new()),
            worker_to_loras: Arc::new(DashMap::new()),
            worker_capacity: Arc::new(DashMap::new()),
        }
    }

    /// Handle an MDC update event: a worker published an MDC with LoRA info.
    pub fn handle_mdc_update(&self, worker: WorkerWithDpRank, lora: &LoraInfo) {
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
            "LoRA state tracker: MDC added"
        );
    }

    /// Handle an MDC removal event: a worker unregistered a LoRA adapter.
    pub fn handle_mdc_removal(&self, worker: WorkerWithDpRank, lora_name: &str) {
        if let Some(mut workers) = self.loaded_locations.get_mut(lora_name) {
            workers.remove(&worker);
            if workers.is_empty() {
                drop(workers);
                self.loaded_locations.remove(lora_name);
            }
        }

        self.lora_info.remove(&(lora_name.to_string(), worker));

        if let Some(mut loras) = self.worker_to_loras.get_mut(&worker) {
            loras.remove(lora_name);
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
        if let Some((_, loras)) = self.worker_to_loras.remove(&worker) {
            for lora_name in &loras {
                if let Some(mut workers) = self.loaded_locations.get_mut(lora_name) {
                    workers.remove(&worker);
                    if workers.is_empty() {
                        drop(workers);
                        self.loaded_locations.remove(lora_name);
                    }
                }
                self.lora_info.remove(&(lora_name.clone(), worker));
            }
        }
        self.worker_capacity.remove(&worker);

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
        self.worker_capacity
            .iter()
            .map(|entry| {
                let worker = *entry.key();
                let (loaded, cap) = self.slot_info(&worker);
                (worker, (loaded as usize, cap as usize))
            })
            .collect()
    }

    pub fn workers_with_free_slots(&self) -> Vec<WorkerWithDpRank> {
        self.worker_capacity
            .iter()
            .filter(|entry| {
                let (loaded, capacity) = self.slot_info(entry.key());
                loaded < capacity
            })
            .map(|entry| *entry.key())
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

        tracker.handle_mdc_update(w1, &lora);

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

        tracker.handle_mdc_update(w1, &lora);
        tracker.handle_mdc_update(w2, &lora);

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

        tracker.handle_mdc_update(w1, &lora);
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

        tracker.handle_mdc_update(w1, &lora1);
        tracker.handle_mdc_update(w1, &lora2);

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

        tracker.handle_mdc_update(w1, &lora1);
        tracker.handle_mdc_update(w1, &lora2);

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
        tracker.handle_mdc_update(w1, &lora1);

        // w2 has capacity 4, load 1 lora → 3 free slots
        let lora2 = make_lora_info("lora-b", Some(4));
        tracker.handle_mdc_update(w2, &lora2);

        let free = tracker.workers_with_free_slots();
        assert_eq!(free.len(), 1);
        assert!(free.contains(&w2));
    }
}
