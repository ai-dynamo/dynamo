// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA Allocation Controller
//!
//! Periodically recomputes LoRA allocations based on load estimates and cluster state.
//! Uses batch load-fraction proportional allocation for active LoRAs, and deterministic
//! HRW cold-start pinning for inactive LoRAs.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::kv_router::protocols::WorkerWithDpRank;
use crate::lora::config::LoraAllocationConfig;
use crate::lora::load_estimator::LoadEstimator;
use crate::lora::routing::mcf_allocator::{
    LoraInput, McfPlacementSolver, McfSolveParams, WorkerInput,
};
use crate::lora::routing::table::{LoraReplicaConfig, LoraRoutingTable};
use crate::lora::routing::{AllocationAlgorithmType, LoraAllocator, create_lora_allocator};
use crate::lora::state_tracker::LoraStateTracker;

#[derive(Debug, Clone)]
struct HysteresisState {
    last_scale_down_tick: u64,
}

/// The LoRA allocation controller.
///
/// Runs a periodic background loop that:
/// 1. Reads current load from `LoadEstimator`
/// 2. Reads cluster topology from `LoraStateTracker`
/// 3. Computes proportional replica counts for active LoRAs
/// 4. Assigns deterministic HRW cold-start pins for inactive LoRAs
/// 5. Updates the `LoraRoutingTable`
pub struct LoraController {
    config: LoraAllocationConfig,
    allocator: Box<dyn LoraAllocator>,
    routing_table: LoraRoutingTable,
    state_tracker: LoraStateTracker,
    load_estimator: Arc<LoadEstimator>,
    hysteresis: HashMap<String, HysteresisState>,
    tick: u64,
    // MCF-specific state
    mcf_solver: Option<McfPlacementSolver>,
    prev_assignment: HashMap<String, HashSet<WorkerWithDpRank>>,
    prev_workers: HashSet<WorkerWithDpRank>,
    prev_replica_counts: HashMap<String, usize>,
}

impl LoraController {
    pub fn new(
        config: LoraAllocationConfig,
        routing_table: LoraRoutingTable,
        state_tracker: LoraStateTracker,
        load_estimator: Arc<LoadEstimator>,
    ) -> Self {
        let allocator = create_lora_allocator(config.algorithm);
        let mcf_solver = if config.algorithm == AllocationAlgorithmType::MinCostFlow {
            let params = McfSolveParams {
                candidate_m: config.mcf.candidate_m,
                alpha_pref: config.mcf.alpha_pref,
                gamma_load: config.mcf.gamma_load,
                beta_keep: config.mcf.beta_keep,
                overflow_cost: config.mcf.overflow_cost,
                allow_overflow: config.mcf.allow_overflow,
            };
            Some(McfPlacementSolver::new(params))
        } else {
            None
        };
        Self {
            config,
            allocator,
            routing_table,
            state_tracker,
            load_estimator,
            hysteresis: HashMap::new(),
            tick: 0,
            mcf_solver,
            prev_assignment: HashMap::new(),
            prev_workers: HashSet::new(),
            prev_replica_counts: HashMap::new(),
        }
    }

    /// Start the controller background loop.
    pub fn start(
        config: LoraAllocationConfig,
        routing_table: LoraRoutingTable,
        state_tracker: LoraStateTracker,
        load_estimator: Arc<LoadEstimator>,
        cancel_token: tokio_util::sync::CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        let timestep = Duration::from_secs(config.timestep_secs);
        let mut controller = Self::new(config, routing_table, state_tracker, load_estimator);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(timestep);
            tracing::info!(
                timestep_secs = controller.config.timestep_secs,
                algorithm = controller.allocator.name(),
                "LoRA allocation controller started"
            );

            loop {
                tokio::select! {
                    _ = cancel_token.cancelled() => {
                        tracing::debug!("LoRA allocation controller shutting down");
                        break;
                    }
                    _ = interval.tick() => {
                        controller.recompute_allocations();
                    }
                }
            }
        })
    }

    pub fn recompute_now(&mut self) {
        self.recompute_allocations();
    }

    fn recompute_allocations(&mut self) {
        self.tick += 1;

        let workers = self.state_tracker.list_workers();
        if workers.is_empty() {
            tracing::debug!("No workers available, skipping recompute");
            return;
        }

        let total_slots = self.state_tracker.total_lora_slots() as usize;
        if total_slots == 0 {
            tracing::debug!("No LoRA slots available, skipping recompute");
            return;
        }

        let worker_slot_usage = self.state_tracker.get_worker_slot_usage();
        let loads = self.load_estimator.get_current_load();
        let total_load: usize = loads.values().sum();

        if !loads.is_empty() {
            tracing::debug!(
                tick = self.tick,
                ?loads,
                total_load,
                "Load estimator snapshot"
            );
        }

        // Collect all known LoRAs (from state tracker + from load estimator)
        let mut seen: std::collections::HashSet<String> =
            self.state_tracker.list_loras().into_iter().collect();
        for lora_name in loads.keys() {
            seen.insert(lora_name.clone());
        }
        let all_loras: Vec<String> = seen.into_iter().collect();

        let active_loras: Vec<(String, usize)> = all_loras
            .iter()
            .filter_map(|name| {
                let load = loads.get(name).copied().unwrap_or(0);
                if load > 0 {
                    Some((name.clone(), load))
                } else {
                    None
                }
            })
            .collect();

        let inactive_loras: Vec<String> = all_loras
            .iter()
            .filter(|name| loads.get(*name).copied().unwrap_or(0) == 0)
            .cloned()
            .collect();

        // Active LoRAs: load-fraction proportional allocation
        let active_replica_counts = if !active_loras.is_empty() && total_load > 0 {
            Self::compute_active_replica_counts(
                &active_loras,
                total_load,
                total_slots,
                workers.len(),
            )
        } else {
            HashMap::new()
        };

        if self.mcf_solver.is_some() {
            self.recompute_mcf(
                &workers,
                &all_loras,
                &active_loras,
                &inactive_loras,
                &active_replica_counts,
            );
        } else {
            self.recompute_per_lora(
                &workers,
                &inactive_loras,
                &active_replica_counts,
                &worker_slot_usage,
            );
        }

        // Cleanup stale entries
        let known_set: std::collections::HashSet<&str> =
            all_loras.iter().map(|s| s.as_str()).collect();
        let table_snapshot = self.routing_table.snapshot_configs();

        for (name, _) in &table_snapshot {
            if !known_set.contains(name.as_str()) {
                self.routing_table.remove_lora(name);
                self.hysteresis.remove(name);
                tracing::debug!(lora = name, "Removed stale routing table entry");
            }
        }

        tracing::debug!(
            tick = self.tick,
            active = active_loras.len(),
            inactive = inactive_loras.len(),
            total_workers = workers.len(),
            total_slots = total_slots,
            "Recompute complete"
        );

        if !table_snapshot.is_empty() {
            tracing::debug!(
                tick = self.tick,
                entries = table_snapshot.len(),
                "LoRA allocation table"
            );
            for (name, config) in &table_snapshot {
                if known_set.contains(name.as_str()) {
                    let worker_ids: Vec<u64> =
                        config.replica_set.iter().map(|w| w.worker_id).collect();
                    tracing::debug!(
                        lora = name,
                        replica_factor = config.replica_factor,
                        workers = ?worker_ids,
                        is_active = config.is_active,
                        "  allocation"
                    );
                }
            }
        }

        let raw_arrival_counts = self.load_estimator.get_raw_arrival_counts();
        self.update_prometheus_metrics(&table_snapshot, &loads, &raw_arrival_counts);
    }

    /// Per-LoRA allocation path (HRW / Random): processes each LoRA independently.
    fn recompute_per_lora(
        &mut self,
        workers: &[WorkerWithDpRank],
        inactive_loras: &[String],
        active_replica_counts: &HashMap<String, usize>,
        worker_slot_usage: &HashMap<WorkerWithDpRank, (usize, usize)>,
    ) {
        for (lora_name, desired_replicas) in active_replica_counts {
            let current = self.routing_table.get_config(lora_name);
            let current_replicas = current.as_ref().map(|c| c.replica_factor).unwrap_or(0);

            let final_replicas =
                self.apply_hysteresis(lora_name, *desired_replicas, current_replicas);

            let replica_set = self.allocator.compute_replica_set_with_slots(
                lora_name,
                workers,
                final_replicas,
                worker_slot_usage,
            );

            if self.update_routing_entry(lora_name, final_replicas, replica_set.clone(), true) {
                tracing::info!(
                    lora = lora_name,
                    replicas = final_replicas,
                    workers = ?replica_set.iter().map(|w| w.worker_id).collect::<Vec<_>>(),
                    "Updated active LoRA allocation"
                );
            }
        }

        // Inactive LoRAs: single HRW-pinned cold-start replica
        for lora_name in inactive_loras {
            let pin = self.allocator.compute_replica_set(lora_name, workers, 1);
            if pin.is_empty() {
                continue;
            }
            let pinned_worker = pin.first().map(|w| w.worker_id);
            if self.update_routing_entry(lora_name, 1, pin, false) {
                tracing::info!(
                    lora = lora_name,
                    pinned_worker_id = ?pinned_worker,
                    "Inactive LoRA allocation (cold-start pin)"
                );
            }
            self.hysteresis.remove(lora_name);
        }
    }

    /// Global MCF allocation path: solves all LoRA placements simultaneously.
    fn recompute_mcf(
        &mut self,
        workers: &[WorkerWithDpRank],
        all_loras: &[String],
        active_loras: &[(String, usize)],
        inactive_loras: &[String],
        active_replica_counts: &HashMap<String, usize>,
    ) {
        let churn_weight = self.config.mcf.churn_weight_default;
        let capacities = self.state_tracker.get_worker_capacities();

        // Build worker inputs
        let worker_inputs: Vec<WorkerInput> = workers
            .iter()
            .map(|w| WorkerInput {
                worker: *w,
                capacity: capacities.get(w).copied().unwrap_or(0) as usize,
            })
            .collect();

        // Build LoRA inputs: active with proportional replicas, inactive with 1
        let mut lora_inputs: Vec<LoraInput> = Vec::new();
        for (lora_name, desired_replicas) in active_replica_counts {
            let current = self.routing_table.get_config(lora_name);
            let current_replicas = current.as_ref().map(|c| c.replica_factor).unwrap_or(0);
            let final_replicas =
                self.apply_hysteresis(lora_name, *desired_replicas, current_replicas);
            lora_inputs.push(LoraInput {
                name: lora_name.clone(),
                replicas: final_replicas,
                churn_weight,
            });
        }
        for lora_name in inactive_loras {
            lora_inputs.push(LoraInput {
                name: lora_name.clone(),
                replicas: 1,
                churn_weight,
            });
            self.hysteresis.remove(lora_name);
        }

        // Detect changes for delta solving
        let current_workers: HashSet<WorkerWithDpRank> = workers.iter().copied().collect();
        let changed_workers: HashSet<WorkerWithDpRank> = current_workers
            .symmetric_difference(&self.prev_workers)
            .copied()
            .collect();

        let mut changed_loras: HashSet<String> = HashSet::new();
        // New or removed LoRAs
        let current_lora_set: HashSet<&str> = all_loras.iter().map(|s| s.as_str()).collect();
        let prev_lora_set: HashSet<&str> =
            self.prev_assignment.keys().map(|s| s.as_str()).collect();
        for l in current_lora_set.difference(&prev_lora_set) {
            changed_loras.insert(l.to_string());
        }
        for l in prev_lora_set.difference(&current_lora_set) {
            changed_loras.insert(l.to_string());
        }
        // Replica count changes
        for li in &lora_inputs {
            let prev_rep = self.prev_replica_counts.get(&li.name).copied().unwrap_or(0);
            if li.replicas != prev_rep {
                changed_loras.insert(li.name.clone());
            }
        }

        let use_delta = !self.prev_assignment.is_empty();
        let (changed_l, changed_w) = if use_delta {
            (Some(&changed_loras), Some(&changed_workers))
        } else {
            (None, None)
        };

        // Borrow solver separately to avoid conflicting borrows on self
        let solver = self.mcf_solver.as_ref().expect("mcf_solver must be Some");
        let solve_result = solver.solve(
            &worker_inputs,
            &lora_inputs,
            &self.prev_assignment,
            changed_l,
            changed_w,
        );

        match solve_result {
            Ok(result) => {
                let total_loads: usize = result.loads.values().map(|s| s.len()).sum();
                let total_unloads: usize = result.unloads.values().map(|s| s.len()).sum();

                if total_loads > 0 || total_unloads > 0 {
                    tracing::info!(
                        tick = self.tick,
                        total_loads,
                        total_unloads,
                        overflow = result.overflow_count,
                        "MCF placement diff"
                    );
                }

                // Update routing table from MCF result
                let active_set: HashSet<&str> =
                    active_loras.iter().map(|(n, _)| n.as_str()).collect();

                for (lora_name, hosts) in &result.assignment {
                    let replica_set: Vec<WorkerWithDpRank> = {
                        let mut v: Vec<_> = hosts.iter().copied().collect();
                        v.sort();
                        v
                    };
                    let is_active = active_set.contains(lora_name.as_str());

                    if self.update_routing_entry(
                        lora_name,
                        replica_set.len(),
                        replica_set.clone(),
                        is_active,
                    ) {
                        tracing::info!(
                            lora = lora_name,
                            replicas = replica_set.len(),
                            workers = ?replica_set.iter().map(|w| w.worker_id).collect::<Vec<_>>(),
                            is_active,
                            "MCF updated LoRA allocation"
                        );
                    }
                }

                // Store state for next tick
                self.prev_assignment = result.assignment;
                self.prev_workers = current_workers;
                self.prev_replica_counts = lora_inputs
                    .iter()
                    .map(|li| (li.name.clone(), li.replicas))
                    .collect();
            }
            Err(e) => {
                tracing::error!(
                    tick = self.tick,
                    error = %e,
                    "MCF solver failed, keeping previous allocations"
                );
            }
        }
    }

    /// Apply scale-down hysteresis to a desired replica count.
    fn apply_hysteresis(
        &mut self,
        lora_name: &str,
        desired_replicas: usize,
        current_replicas: usize,
    ) -> usize {
        if desired_replicas < current_replicas {
            if let Some(hyst) = self.hysteresis.get(lora_name) {
                let ticks_since = self.tick.saturating_sub(hyst.last_scale_down_tick);
                if ticks_since < self.config.scale_down_cooldown_ticks as u64 {
                    tracing::debug!(
                        lora = lora_name,
                        desired = desired_replicas,
                        current = current_replicas,
                        cooldown_remaining =
                            self.config.scale_down_cooldown_ticks as u64 - ticks_since,
                        "Scale-down deferred by hysteresis"
                    );
                    return current_replicas;
                }
                desired_replicas
            } else {
                self.hysteresis.insert(
                    lora_name.to_string(),
                    HysteresisState {
                        last_scale_down_tick: self.tick,
                    },
                );
                current_replicas
            }
        } else {
            self.hysteresis.remove(lora_name);
            desired_replicas
        }
    }

    fn update_routing_entry(
        &self,
        lora_name: &str,
        replica_factor: usize,
        replica_set: Vec<WorkerWithDpRank>,
        is_active: bool,
    ) -> bool {
        let current = self.routing_table.get_config(lora_name);
        let changed = current
            .as_ref()
            .map(|c| {
                c.replica_set != replica_set
                    || c.replica_factor != replica_factor
                    || c.is_active != is_active
            })
            .unwrap_or(true);

        if changed {
            self.routing_table.update_allocation(
                lora_name.to_string(),
                LoraReplicaConfig {
                    lora_name: lora_name.to_string(),
                    replica_factor,
                    replica_set,
                    updated_at: Instant::now(),
                    is_active,
                },
            );
        }
        changed
    }

    fn update_prometheus_metrics(
        &self,
        table_snapshot: &[(String, LoraReplicaConfig)],
        loads: &HashMap<String, usize>,
        raw_arrival_counts: &HashMap<String, u64>,
    ) {
        use crate::http::service::metrics::{
            LORA_ACTIVE_REQUESTS_GAUGE, LORA_ESTIMATED_LOAD_GAUGE, LORA_IS_ACTIVE_GAUGE,
            LORA_RAW_ARRIVAL_COUNT_GAUGE, LORA_REPLICA_FACTOR_GAUGE,
        };

        LORA_REPLICA_FACTOR_GAUGE.reset();
        LORA_IS_ACTIVE_GAUGE.reset();
        LORA_RAW_ARRIVAL_COUNT_GAUGE.reset();
        LORA_ESTIMATED_LOAD_GAUGE.reset();
        LORA_ACTIVE_REQUESTS_GAUGE.reset();

        for (lora_name, config) in table_snapshot {
            LORA_REPLICA_FACTOR_GAUGE
                .with_label_values(&[lora_name])
                .set(config.replica_factor as i64);
            LORA_IS_ACTIVE_GAUGE
                .with_label_values(&[lora_name])
                .set(if config.is_active { 1 } else { 0 });
            let raw_count = raw_arrival_counts.get(lora_name).copied().unwrap_or(0);
            LORA_RAW_ARRIVAL_COUNT_GAUGE
                .with_label_values(&[lora_name])
                .set(raw_count as i64);
            let load = loads.get(lora_name).copied().unwrap_or(0);
            LORA_ESTIMATED_LOAD_GAUGE
                .with_label_values(&[lora_name])
                .set(load as i64);
        }

        let inflight = self.load_estimator.get_inflight_counts();
        for (lora_name, count) in &inflight {
            LORA_ACTIVE_REQUESTS_GAUGE
                .with_label_values(&[lora_name.as_str()])
                .set(*count as i64);
        }
    }

    /// Compute proportional replica counts for active LoRAs.
    fn compute_active_replica_counts(
        active_loras: &[(String, usize)],
        total_load: usize,
        total_slots: usize,
        num_workers: usize,
    ) -> HashMap<String, usize> {
        let mut result = HashMap::new();

        if active_loras.is_empty() || total_load == 0 {
            return result;
        }

        let mut raw_counts: Vec<(String, f64)> = active_loras
            .iter()
            .map(|(name, load)| {
                let fraction = *load as f64 / total_load as f64;
                let raw = (fraction * total_slots as f64).ceil().max(1.0);
                (name.clone(), raw)
            })
            .collect();

        for (_, count) in raw_counts.iter_mut() {
            *count = count.min(num_workers as f64);
        }

        // Budget normalization if sum exceeds total_slots
        let sum: f64 = raw_counts.iter().map(|(_, c)| *c).sum();
        if sum > total_slots as f64 {
            let scale = total_slots as f64 / sum;
            for (_, count) in raw_counts.iter_mut() {
                *count = (*count * scale).max(1.0);
            }

            let mut floored: Vec<(String, usize, f64)> = raw_counts
                .iter()
                .map(|(name, c)| {
                    let f = c.floor().max(1.0) as usize;
                    let remainder = *c - f as f64;
                    (name.clone(), f, remainder)
                })
                .collect();

            let floored_sum: usize = floored.iter().map(|(_, f, _)| *f).sum();
            let mut leftover = total_slots.saturating_sub(floored_sum);

            floored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            for (_, count, _) in floored.iter_mut() {
                if leftover == 0 {
                    break;
                }
                if *count < num_workers {
                    *count += 1;
                    leftover -= 1;
                }
            }

            for (name, count, _) in floored {
                result.insert(name, count.max(1).min(num_workers));
            }
        } else {
            for (name, count) in raw_counts {
                result.insert(name, (count as usize).max(1).min(num_workers));
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::load_estimator::LoadEstimator;
    use crate::lora::routing::table::LoraRoutingTable;
    use crate::lora::state_tracker::LoraStateTracker;
    use crate::model_card::LoraInfo;

    fn make_worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    fn make_lora_info(name: &str, cap: u32) -> LoraInfo {
        LoraInfo {
            name: name.to_string(),
            max_gpu_lora_count: Some(cap),
        }
    }

    fn setup_controller() -> (
        LoraController,
        LoraStateTracker,
        Arc<LoadEstimator>,
        LoraRoutingTable,
    ) {
        let config = LoraAllocationConfig::default();
        let routing_table = LoraRoutingTable::new();
        let state_tracker = LoraStateTracker::new();
        let load_estimator = Arc::new(LoadEstimator::new());
        let controller = LoraController::new(
            config,
            routing_table.clone(),
            state_tracker.clone(),
            load_estimator.clone(),
        );
        (controller, state_tracker, load_estimator, routing_table)
    }

    #[test]
    fn test_no_workers_skips_recompute() {
        let (mut controller, _st, _le, rt) = setup_controller();
        controller.recompute_now();
        assert!(rt.is_empty());
    }

    #[test]
    fn test_inactive_lora_gets_single_pin() {
        let (mut controller, st, _le, rt) = setup_controller();
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        st.handle_mdc_update(w1, &make_lora_info("lora-a", 4));
        st.handle_mdc_update(w2, &make_lora_info("lora-a", 4));

        controller.recompute_now();

        let config = rt.get_config("lora-a").unwrap();
        assert!(!config.is_active);
        assert_eq!(config.replica_factor, 1);
        assert_eq!(config.replica_set.len(), 1);
    }

    #[test]
    fn test_active_lora_gets_proportional_allocation() {
        let (mut controller, st, le, rt) = setup_controller();
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        let w3 = make_worker(3);
        st.handle_mdc_update(w1, &make_lora_info("lora-a", 4));
        st.handle_mdc_update(w2, &make_lora_info("lora-b", 4));
        st.handle_mdc_update(w3, &make_lora_info("lora-a", 4));

        le.increment_load("lora-a");
        le.increment_load("lora-a");
        le.increment_load("lora-a");

        controller.recompute_now();

        let config = rt.get_config("lora-a").unwrap();
        assert!(config.is_active);
        assert!(config.replica_factor >= 1);
    }

    #[test]
    fn test_proportional_allocation_math() {
        let active = vec![
            ("lora-a".to_string(), 60),
            ("lora-b".to_string(), 30),
            ("lora-c".to_string(), 10),
        ];
        let result = LoraController::compute_active_replica_counts(&active, 100, 12, 5);

        assert!(result["lora-a"] >= 1 && result["lora-a"] <= 5);
        assert!(result["lora-b"] >= 1);
        assert!(result["lora-c"] >= 1);
        assert!(result["lora-a"] >= result["lora-b"]);
        assert!(result["lora-b"] >= result["lora-c"]);
    }

    #[test]
    fn test_budget_normalization() {
        let active = vec![("lora-a".to_string(), 50), ("lora-b".to_string(), 50)];
        let result = LoraController::compute_active_replica_counts(&active, 100, 2, 10);

        assert_eq!(result["lora-a"], 1);
        assert_eq!(result["lora-b"], 1);
    }

    #[test]
    fn test_cleanup_removes_stale_entries() {
        let (mut controller, st, _le, rt) = setup_controller();
        let w1 = make_worker(1);
        st.handle_mdc_update(w1, &make_lora_info("lora-a", 4));

        controller.recompute_now();
        assert!(rt.get_config("lora-a").is_some());

        st.handle_mdc_removal(w1, "lora-a");
        controller.recompute_now();

        assert!(rt.get_config("lora-a").is_none());
    }
}
