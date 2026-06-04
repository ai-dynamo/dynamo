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
    prev_worker_capacities: HashMap<WorkerWithDpRank, u32>,
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
            prev_worker_capacities: HashMap::new(),
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
            // Cluster drained: every routing entry now points at gone workers. Leaving it would
            // make the filter fall back to all-available (scatter) and the gauges show phantom
            // allocations. Clear routing state and reset metrics instead of returning early.
            tracing::debug!("No workers available; clearing stale LoRA routes and metrics");
            self.clear_lora_routing_and_metrics();
            return;
        }

        let total_slots = self.state_tracker.total_lora_slots() as usize;
        if total_slots == 0 {
            // Workers exist but report zero LoRA capacity, so there is nothing to (re)allocate.
            // Don't wipe bounded routes wholesale (they'd drop to the filter's no-table scatter),
            // but a route whose replica set has lost ALL its live workers (e.g. some workers were
            // removed) WOULD scatter — the filter finds no live replica intersection and falls back
            // to all available workers. Prune every entry against the live worker set: narrow to
            // the live intersection, or rebind to a single deterministic live pin when none of its
            // workers are live, so every LoRA keeps a bounded, live route. Then skip recompute.
            let live: std::collections::HashSet<WorkerWithDpRank> =
                workers.iter().copied().collect();
            for (name, cfg) in self.routing_table.snapshot_configs() {
                let live_set: Vec<WorkerWithDpRank> = cfg
                    .replica_set
                    .iter()
                    .copied()
                    .filter(|w| live.contains(w))
                    .collect();
                if live_set.is_empty() {
                    // Deterministic HRW top-1, NOT self.allocator: the Random allocator returns
                    // all workers regardless of replica_factor, which would re-scatter the dead
                    // route instead of binding it to a single worker.
                    let pin = crate::lora::routing::RendezvousHasher
                        .compute_replica_set(&name, &workers, 1);
                    if pin.is_empty() {
                        self.routing_table.remove_lora(&name);
                        self.hysteresis.remove(&name);
                    } else if pin != cfg.replica_set {
                        self.update_routing_entry(&name, pin.len(), pin, cfg.is_active);
                    }
                } else if live_set.len() != cfg.replica_set.len() {
                    self.update_routing_entry(&name, live_set.len(), live_set, cfg.is_active);
                }
            }
            // Refresh gauges from the pruned table so a narrowed/rebound route's replica_factor
            // (and the other LoRA gauges) don't stay stale while capacity remains zero. No
            // allocation ran this tick, so churn/overflow are zero.
            let table_snapshot = self.routing_table.snapshot_configs();
            let loads = self.load_estimator.get_current_load();
            let raw_arrival_counts = self.load_estimator.get_raw_arrival_counts();
            self.update_prometheus_metrics(&table_snapshot, &loads, &raw_arrival_counts);
            crate::http::service::metrics::LORA_CHURN_LOADS_GAUGE.set(0);
            crate::http::service::metrics::LORA_CHURN_UNLOADS_GAUGE.set(0);
            crate::http::service::metrics::LORA_OVERFLOW_COUNT_GAUGE.set(0);
            tracing::debug!(
                "No LoRA slots available; pruned routes to live workers, skipping recompute"
            );
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
        // Sort for deterministic ordering across all router instances (REQ: deterministic
        // placement). active/inactive/MCF inputs all derive from this order, and the
        // largest-remainder tie-break below is by name, so every router converges identically.
        let mut all_loras: Vec<String> = seen.into_iter().collect();
        all_loras.sort();

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

        // Cleanup stale entries.
        let known_set: std::collections::HashSet<&str> =
            all_loras.iter().map(|s| s.as_str()).collect();
        let live_workers: std::collections::HashSet<WorkerWithDpRank> =
            workers.iter().copied().collect();

        // Capacity-dropped active LoRAs: active (load>0) but truncated out of this tick's budget
        // by the R3 cap, so they received no fresh allocation. Each must stay routable (REQ 7)
        // without triggering uncontrolled loads that defeat the cap:
        //   * If still warm somewhere (prior set ∩ loaded ∩ live non-empty), keep ONLY those
        //     workers — routes to existing adapters with zero new loads. (Rewrites only when the
        //     warm set shrank, so a still-warm dropped LoRA stays churn-free.)
        //   * Otherwise (adapter evicted everywhere, OR a brand-new over-budget active LoRA with
        //     no entry at all), pin it to a SINGLE deterministic slot-aware HRW worker. This bounds
        //     the unavoidable load to one worker and is identical across router instances, instead
        //     of removing the entry — which would scatter to all workers via the filter's no-table
        //     path (returns all when nothing is loaded).
        // Pin worker selection is slot-aware against this tick's PROJECTED usage (the in-budget
        // HRW/MCF placements already written to the routing table, plus earlier dropped pins), so a
        // pin never targets a slot already claimed this tick.
        let dropped_active: Vec<&str> = active_loras
            .iter()
            .map(|(n, _)| n.as_str())
            .filter(|n| !active_replica_counts.contains_key(*n))
            .collect();
        if !dropped_active.is_empty() {
            let dropped_set: std::collections::HashSet<&str> =
                dropped_active.iter().copied().collect();
            let caps = self.state_tracker.get_worker_capacities();
            // Committed placements this tick, excluding the dropped LoRAs themselves (their entries
            // are stale and re-decided below).
            let mut projected: HashMap<WorkerWithDpRank, usize> = HashMap::new();
            for (name, cfg) in self.routing_table.snapshot_configs() {
                if dropped_set.contains(name.as_str()) {
                    continue;
                }
                for w in &cfg.replica_set {
                    *projected.entry(*w).or_insert(0) += 1;
                }
            }

            for &name in &dropped_active {
                let prior = self
                    .routing_table
                    .get_config(name)
                    .map(|c| c.replica_set)
                    .unwrap_or_default();
                let loaded = self.state_tracker.get_loaded_workers(name);
                let warm: Vec<WorkerWithDpRank> = prior
                    .iter()
                    .copied()
                    .filter(|w| loaded.contains(w) && live_workers.contains(w))
                    .collect();

                let chosen = if !warm.is_empty() {
                    warm
                } else {
                    let proj_usage: HashMap<WorkerWithDpRank, (usize, usize)> = workers
                        .iter()
                        .map(|w| {
                            let used = projected.get(w).copied().unwrap_or(0);
                            let cap = caps.get(w).copied().unwrap_or(0) as usize;
                            (*w, (used, cap))
                        })
                        .collect();
                    self.allocator
                        .compute_replica_set_with_slots(name, &workers, 1, &proj_usage)
                };

                if chosen.is_empty() {
                    self.routing_table.remove_lora(name);
                    self.hysteresis.remove(name);
                    continue;
                }
                // Charge the chosen workers so later dropped pins see them occupied.
                for w in &chosen {
                    *projected.entry(*w).or_insert(0) += 1;
                }
                // No-ops when nothing changed (still-warm, unchanged set) — keeps churn at zero.
                self.update_routing_entry(name, chosen.len(), chosen, true);
            }
        }

        let table_snapshot = self.routing_table.snapshot_configs();

        for (name, _) in &table_snapshot {
            if !known_set.contains(name.as_str()) {
                self.routing_table.remove_lora(name);
                self.hysteresis.remove(name);
                tracing::debug!(lora = name, "Removed stale routing table entry");
            }
        }

        // Re-snapshot after cleanup so the logs and Prometheus gauges below reflect the
        // post-cleanup table and don't surface a stale LoRA for an extra tick (RF-1).
        let table_snapshot = self.routing_table.snapshot_configs();

        // Prune the load estimator of LoRAs that are no longer loaded (and any
        // unknown/typo request names), bounding its memory over time (F12).
        self.load_estimator.retain_known(&known_set);

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
        // Track residual capacity across LoRAs within this tick so multiple active LoRAs
        // are not all placed on the same "free" slots and exceed per-worker capacity (F7).
        // Iterate in a deterministic (sorted) order so every router instance charges
        // residual capacity identically and converges on the same placement.
        let mut residual_usage = worker_slot_usage.clone();
        let mut active_sorted: Vec<(&String, &usize)> = active_replica_counts.iter().collect();
        active_sorted.sort_by(|a, b| a.0.cmp(b.0));

        for (lora_name, desired_replicas) in active_sorted {
            let current = self.routing_table.get_config(lora_name);
            let current_replicas = current.as_ref().map(|c| c.replica_factor).unwrap_or(0);
            // Anchor on the LoRA's existing placement so transient sibling activity within
            // this tick cannot move a LoRA whose own inputs are unchanged (preserves the HRW
            // churn-minimization guarantee while still charging residual capacity below).
            let prior: Vec<WorkerWithDpRank> = current
                .as_ref()
                .map(|c| c.replica_set.clone())
                .unwrap_or_default();

            let final_replicas =
                self.apply_hysteresis(lora_name, *desired_replicas, current_replicas);

            // Discount this LoRA's OWN already-loaded slots before the fullness check: a worker
            // that currently hosts this adapter is not "full" with respect to re-placing the same
            // adapter (retaining it consumes no new slot). Without this, a LoRA pinned to a worker
            // that is full largely because of itself (e.g. cap=1) would be evicted and moved every
            // tick — defeating the sticky placement. Charging (below) still uses the undiscounted
            // shared residual so sibling LoRAs see true remaining capacity.
            let own_loaded = self.state_tracker.get_loaded_workers(lora_name);
            let mut eff_usage = residual_usage.clone();
            for w in &own_loaded {
                if let Some(usage) = eff_usage.get_mut(w) {
                    usage.0 = usage.0.saturating_sub(1);
                }
            }

            let replica_set = self.allocator.compute_replica_set_with_slots_sticky(
                lora_name,
                workers,
                final_replicas,
                &eff_usage,
                &prior,
            );

            // Charge this LoRA's placements against the shared residual capacity for later LoRAs,
            // but ONLY for workers where it is not already loaded: an already-loaded placement is
            // already counted in the base `worker_slot_usage`, so charging it again would
            // double-count and make the worker look full to subsequent same-worker LoRAs (e.g. a
            // cap=2 worker hosting A and B: charging A's retained slot would push B off it).
            for w in &replica_set {
                if own_loaded.contains(w) {
                    continue;
                }
                if let Some(usage) = residual_usage.get_mut(w) {
                    usage.0 += 1;
                }
            }

            // Store the EFFECTIVE replica count (the set the allocator actually returned), not the
            // desired `final_replicas`. Under partial-capacity pressure the slot-aware allocator
            // returns a smaller-than-requested set, so using `final_replicas` would make
            // LoraReplicaConfig::replica_factor overstate replica_set.len() — wrong for the
            // replica-factor gauge and any consumer of that field. The router already narrows by
            // the set itself.
            if self.update_routing_entry(lora_name, replica_set.len(), replica_set.clone(), true) {
                tracing::info!(
                    lora = lora_name,
                    desired = final_replicas,
                    replicas = replica_set.len(),
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
            // Store the effective set size (not a literal 1): the per-LoRA allocators return one
            // worker for HRW/MCF cold-start, but the test-only Random allocator returns every
            // worker regardless of the requested count, so replica_factor must track the set.
            if self.update_routing_entry(lora_name, pin.len(), pin, false) {
                tracing::info!(
                    lora = lora_name,
                    pinned_worker_id = ?pinned_worker,
                    "Inactive LoRA allocation (cold-start pin)"
                );
            }
            self.hysteresis.remove(lora_name);
        }
    }

    /// Workers that changed since the previous MCF tick: those added or removed (set difference)
    /// AND those whose per-worker LoRA capacity changed. The MCF delta solver only reconsiders
    /// workers it is told changed, so a capacity change (e.g. a base-card `max_gpu_lora_count`
    /// update, or a worker gaining/losing slots) must be reported here — otherwise its frozen
    /// placements stay stale in delta mode and the solver never uses new headroom. (Capacity
    /// shrink past current usage is also caught by the solver's over-commit unfreeze, but increases
    /// and non-overcommitting shrinks rely on this signal.)
    fn compute_changed_workers(
        current_workers: &HashSet<WorkerWithDpRank>,
        prev_workers: &HashSet<WorkerWithDpRank>,
        current_caps: &HashMap<WorkerWithDpRank, u32>,
        prev_caps: &HashMap<WorkerWithDpRank, u32>,
    ) -> HashSet<WorkerWithDpRank> {
        let mut changed: HashSet<WorkerWithDpRank> = current_workers
            .symmetric_difference(prev_workers)
            .copied()
            .collect();
        for w in current_workers {
            if current_caps.get(w).copied().unwrap_or(0) != prev_caps.get(w).copied().unwrap_or(0) {
                changed.insert(*w);
            }
        }
        changed
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

        // R3-7: reset churn/overflow gauges at tick start so a failed MCF solve does not leave
        // stale values from a prior successful tick; the success branch sets the actual diff.
        crate::http::service::metrics::LORA_CHURN_LOADS_GAUGE.set(0);
        crate::http::service::metrics::LORA_CHURN_UNLOADS_GAUGE.set(0);
        crate::http::service::metrics::LORA_OVERFLOW_COUNT_GAUGE.set(0);

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

        // Deterministic solver input order across router instances (RR3-2): active inputs are
        // built by iterating a HashMap, so sort by name before the MCF solve.
        lora_inputs.sort_by(|a, b| a.name.cmp(&b.name));

        // Detect changes for delta solving
        let current_workers: HashSet<WorkerWithDpRank> = workers.iter().copied().collect();
        // Worker-set changes (added/removed) AND per-worker capacity changes both count as worker
        // changes for the MCF delta solver (see compute_changed_workers).
        let changed_workers = Self::compute_changed_workers(
            &current_workers,
            &self.prev_workers,
            &capacities,
            &self.prev_worker_capacities,
        );

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

                // N7: export churn + overflow gauges (registered but previously never set).
                crate::http::service::metrics::LORA_CHURN_LOADS_GAUGE.set(total_loads as i64);
                crate::http::service::metrics::LORA_CHURN_UNLOADS_GAUGE.set(total_unloads as i64);
                crate::http::service::metrics::LORA_OVERFLOW_COUNT_GAUGE
                    .set(result.overflow_count as i64);

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

                // F8: the MCF solver omits fully-overflowed LoRAs from `assignment`. A
                // still-loaded LoRA left unplaced must stay routable (REQ 7) without thrashing
                // the routing table or polluting the solver's keep/delta state:
                //   * Continuity first — if the LoRA already has a routing entry (it was placed
                //     on a recent tick, so its adapter is likely still warm there), leave it
                //     untouched. No write means no churn, and the prior workers remain a valid
                //     route. The filter's runtime loaded-worker fallback (N5) also keeps these
                //     reachable.
                //   * Only a brand-new unplaced LoRA (no entry at all) gets a deterministic HRW
                //     cold-start pin so it has at least one home.
                //   * Crucially, fallback pins are NEVER inserted into `result.assignment`: that
                //     value becomes `prev_assignment`, and feeding it phantom (non-MCF) placements
                //     corrupts the keep-reward/delta logic on the next tick and inflates churn.
                // LoRAs intentionally dropped by the capacity cap (active but truncated out of
                // active_replica_counts) must NOT be re-pinned (RR3-1) — they rely on the runtime
                // loaded-worker fallback, exactly as in the HRW path.
                let intended: std::collections::HashSet<&str> = active_replica_counts
                    .keys()
                    .map(String::as_str)
                    .chain(inactive_loras.iter().map(String::as_str))
                    .collect();
                let unplaced: Vec<String> = all_loras
                    .iter()
                    .filter(|n| {
                        intended.contains(n.as_str()) && !result.assignment.contains_key(*n)
                    })
                    .cloned()
                    .collect();
                for lora_name in unplaced {
                    let is_active = active_set.contains(lora_name.as_str());
                    // The solver produced NO placement for this LoRA (capacity overflow), but it
                    // must stay routable (REQ 7) without bypassing that decision. We must NOT keep
                    // its full prior entry: the filter's tier-2 lazy-load path (`replica_set ∩
                    // available`) would reload the adapter onto a still-live replica it was evicted
                    // from, defeating the overflow cap. So narrow any existing entry to the workers
                    // where it is STILL warm (prior set ∩ loaded ∩ live) — pure existing routes, no
                    // new load. If nothing is warm, fall back to a single deterministic HRW
                    // cold-start pin (bounded, cross-instance-consistent). Fallback pins are never
                    // inserted into `result.assignment`, so `prev_assignment` stays the pure solver
                    // output.
                    let loaded = self.state_tracker.get_loaded_workers(&lora_name);
                    let warm: Vec<WorkerWithDpRank> = self
                        .routing_table
                        .get_config(&lora_name)
                        .map(|c| c.replica_set)
                        .unwrap_or_default()
                        .into_iter()
                        .filter(|w| current_workers.contains(w) && loaded.contains(w))
                        .collect();

                    let chosen = if !warm.is_empty() {
                        warm
                    } else {
                        self.allocator.compute_replica_set(&lora_name, workers, 1)
                    };
                    if chosen.is_empty() {
                        // No live worker at all — drop any stale entry rather than keep a dead route.
                        self.routing_table.remove_lora(&lora_name);
                        self.hysteresis.remove(&lora_name);
                        continue;
                    }
                    if self.update_routing_entry(
                        &lora_name,
                        chosen.len(),
                        chosen.clone(),
                        is_active,
                    ) {
                        tracing::warn!(
                            lora = %lora_name,
                            workers = ?chosen.iter().map(|w| w.worker_id).collect::<Vec<_>>(),
                            "MCF left LoRA unplaced (capacity overflow); narrowed to warm workers or HRW pin"
                        );
                    }
                }

                // Store state for next tick (pure MCF assignment; no fallback pins injected)
                self.prev_assignment = result.assignment;
                self.prev_workers = current_workers;
                self.prev_worker_capacities = capacities;
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

    /// Clear all LoRA routing state and reset every LoRA gauge.
    ///
    /// Called when the cluster has no live workers (a full drain). Every routing entry is then
    /// stale — its replica set points at gone workers — and such an entry makes `LoraFilter` fall
    /// back to all available workers (none), so dropping the entries and resetting the gauges
    /// (which would otherwise show phantom allocations) is correct. The separate zero-capacity
    /// path (workers live, no slots) instead prunes/rebinds routes against the live worker set,
    /// since those entries can still name live workers.
    fn clear_lora_routing_and_metrics(&mut self) {
        use crate::http::service::metrics::{
            LORA_ACTIVE_REQUESTS_GAUGE, LORA_CHURN_LOADS_GAUGE, LORA_CHURN_UNLOADS_GAUGE,
            LORA_ESTIMATED_LOAD_GAUGE, LORA_IS_ACTIVE_GAUGE, LORA_OVERFLOW_COUNT_GAUGE,
            LORA_RAW_ARRIVAL_COUNT_GAUGE, LORA_REPLICA_FACTOR_GAUGE,
        };

        for (name, _) in self.routing_table.snapshot_configs() {
            self.routing_table.remove_lora(&name);
        }
        self.hysteresis.clear();
        // Drop MCF delta state so a later recovery re-solves from scratch rather than against a
        // phantom prior assignment referencing gone workers.
        self.prev_assignment.clear();
        self.prev_workers.clear();
        self.prev_worker_capacities.clear();
        self.prev_replica_counts.clear();

        LORA_REPLICA_FACTOR_GAUGE.reset();
        LORA_IS_ACTIVE_GAUGE.reset();
        LORA_RAW_ARRIVAL_COUNT_GAUGE.reset();
        LORA_ESTIMATED_LOAD_GAUGE.reset();
        LORA_CHURN_LOADS_GAUGE.set(0);
        LORA_CHURN_UNLOADS_GAUGE.set(0);
        LORA_OVERFLOW_COUNT_GAUGE.set(0);

        // In-flight requests can still be draining even with no workers; keep reporting them
        // (reset then repopulate from the estimator, matching the normal metrics path) rather than
        // forcing the gauge to zero.
        LORA_ACTIVE_REQUESTS_GAUGE.reset();
        for (lora_name, count) in &self.load_estimator.get_inflight_counts() {
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

        if active_loras.is_empty() || total_load == 0 || total_slots == 0 {
            return result;
        }

        // R3-1: when there are more active LoRAs than the cluster slot budget, we cannot give
        // every active LoRA a replica without overcommitting workers (the min-1 floor below
        // would otherwise sum past `total_slots`). Rank by load (desc, tie by name for
        // determinism) and keep only the top `total_slots`; the remainder get no allocation
        // here and rely on the runtime loaded-worker fallback. This avoids forcing min-1
        // placements onto already-full workers, and (by keeping the placed set to a stable
        // top-by-load) also stabilizes the MCF solver's input across ticks.
        let mut ranked: Vec<(String, usize)> = active_loras.to_vec();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if ranked.len() > total_slots {
            ranked.truncate(total_slots);
        }
        let eff_total_load: usize = ranked.iter().map(|(_, l)| *l).sum();
        if eff_total_load == 0 {
            return result;
        }

        let mut raw_counts: Vec<(String, f64)> = ranked
            .iter()
            .map(|(name, load)| {
                let fraction = *load as f64 / eff_total_load as f64;
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

            // Sort by fractional remainder descending, breaking ties by LoRA name so the
            // largest-remainder distribution is deterministic across all router instances.
            floored.sort_by(|a, b| {
                b.2.partial_cmp(&a.2)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
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
        st.handle_mdc_addition(w1, &make_lora_info("lora-a", 4));
        st.handle_mdc_addition(w2, &make_lora_info("lora-a", 4));

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
        st.handle_mdc_addition(w1, &make_lora_info("lora-a", 4));
        st.handle_mdc_addition(w2, &make_lora_info("lora-b", 4));
        st.handle_mdc_addition(w3, &make_lora_info("lora-a", 4));

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
        st.handle_mdc_addition(w1, &make_lora_info("lora-a", 4));

        controller.recompute_now();
        assert!(rt.get_config("lora-a").is_some());

        st.handle_mdc_removal(w1, "lora-a");
        controller.recompute_now();

        assert!(rt.get_config("lora-a").is_none());
    }

    #[test]
    fn test_capacity_one_worker_sticky_no_eviction() {
        // F1: a LoRA pinned to a cap=1 worker is "full" only because of itself. The fullness
        // check must discount the LoRA's own loaded slot, or sticky placement would evict it and
        // scatter it across other workers (here w2 is full with a different adapter, so without
        // the discount the empty-result fallback would return the full HRW set [w1, w2]).
        let (mut controller, st, le, rt) = setup_controller();
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        st.handle_mdc_addition(w1, &make_lora_info("lora-a", 1));
        st.handle_mdc_addition(w2, &make_lora_info("lora-b", 1)); // w2 full with a different adapter
        le.increment_load("lora-a");

        controller.recompute_now();
        let set1: Vec<u64> = rt
            .get_config("lora-a")
            .unwrap()
            .replica_set
            .iter()
            .map(|w| w.worker_id)
            .collect();
        assert_eq!(
            set1,
            vec![1],
            "lora-a must stay on its only non-full home w1, not scatter onto the full w2"
        );

        // Nothing changed → placement must be identical across ticks (no eviction/churn).
        controller.recompute_now();
        let set2: Vec<u64> = rt
            .get_config("lora-a")
            .unwrap()
            .replica_set
            .iter()
            .map(|w| w.worker_id)
            .collect();
        assert_eq!(
            set2, set1,
            "stable LoRA on a cap=1 worker must not move across ticks"
        );
    }

    #[test]
    fn test_two_adapters_share_cap2_worker_neither_evicted() {
        // N1: a cap=2 worker hosting two of its own adapters must keep BOTH. The shared residual
        // must not be charged for already-loaded placements; otherwise placing the first adapter
        // pushes residual to 3, the second adapter's own-slot discount only brings it back to cap,
        // and sticky evicts the second one onto another worker (churn from the fix itself).
        let (mut controller, st, le, rt) = setup_controller();
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        // w1 hosts A,B; w2 hosts C,D (both cap=2, both full with their own adapters).
        st.handle_mdc_addition(w1, &make_lora_info("lora-a", 2));
        st.handle_mdc_addition(w1, &make_lora_info("lora-b", 2));
        st.handle_mdc_addition(w2, &make_lora_info("lora-c", 2));
        st.handle_mdc_addition(w2, &make_lora_info("lora-d", 2));
        // All four active with equal load => proportional gives each replica_factor 1.
        for n in ["lora-a", "lora-b", "lora-c", "lora-d"] {
            le.increment_load(n);
        }

        controller.recompute_now();

        let set_ids = |name: &str| -> Vec<u64> {
            rt.get_config(name)
                .unwrap()
                .replica_set
                .iter()
                .map(|w| w.worker_id)
                .collect()
        };
        assert_eq!(set_ids("lora-a"), vec![1], "A keeps its warm worker w1");
        assert_eq!(
            set_ids("lora-b"),
            vec![1],
            "B must NOT be evicted from the shared cap=2 worker w1"
        );
    }

    #[test]
    fn test_capacity_dropped_active_lora_without_warm_worker_is_pinned() {
        // F2 + N2: an active LoRA truncated out of the slot budget (capacity-dropped) must NOT
        // keep a stale replica-set entry pointing at workers where it is no longer loaded — the
        // filter would lazy-load it there (a new load that defeats the cap) or widen to all
        // workers. With no warm worker, the entry is replaced with a single deterministic HRW pin
        // (bounded, REQ 7), NOT removed (which would scatter via the filter's no-table path) and
        // NOT left stale.
        let (mut controller, st, le, rt) = setup_controller();
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        // Only w1 is a live worker (cap=1 → total budget = 1). lora-a is the high-load adapter.
        st.handle_mdc_addition(w1, &make_lora_info("lora-a", 1));
        // Seed a STALE entry for lora-b pointing at w2 (not live, lora-b not loaded anywhere).
        rt.update_allocation(
            "lora-b".to_string(),
            LoraReplicaConfig {
                lora_name: "lora-b".to_string(),
                replica_factor: 1,
                replica_set: vec![w2],
                updated_at: Instant::now(),
                is_active: true,
            },
        );
        // Both active; lora-a outranks lora-b by load, so the budget=1 cap drops lora-b.
        for _ in 0..5 {
            le.increment_load("lora-a");
        }
        le.increment_load("lora-b");

        controller.recompute_now();

        assert!(
            rt.get_config("lora-a").is_some(),
            "the in-budget LoRA keeps its allocation"
        );
        let cfg_b = rt.get_config("lora-b").expect(
            "capacity-dropped LoRA stays routable via a bounded pin, not removed/scattered",
        );
        assert_eq!(
            cfg_b.replica_set.len(),
            1,
            "no-warm cap-dropped LoRA must be pinned to a single worker, not scattered"
        );
        assert_eq!(
            cfg_b.replica_set[0].worker_id, 1,
            "pinned to the only live worker w1, not the stale w2"
        );
        assert!(cfg_b.is_active, "the LoRA is still active");
    }

    #[test]
    fn test_new_cap_dropped_active_lora_without_entry_is_pinned_not_scattered() {
        // R3-1: a brand-new active LoRA that is dropped by the budget cap and has NO routing entry
        // must still get a single bounded pin — not be skipped (which would let the filter's
        // no-table path scatter it across all workers).
        let (mut controller, st, le, rt) = setup_controller();
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        // Two cap=1 workers (budget = 2). lora-a and lora-c are the high-load in-budget adapters;
        // lora-b is new, low-load, no prior entry, not loaded anywhere.
        st.handle_mdc_addition(w1, &make_lora_info("lora-a", 1));
        st.handle_mdc_addition(w2, &make_lora_info("lora-c", 1));
        for _ in 0..5 {
            le.increment_load("lora-a");
            le.increment_load("lora-c");
        }
        le.increment_load("lora-b"); // active but lowest load -> dropped by the budget=2 cap

        controller.recompute_now();

        let cfg_b = rt
            .get_config("lora-b")
            .expect("new cap-dropped active LoRA must get a bounded pin, not be skipped");
        assert_eq!(
            cfg_b.replica_set.len(),
            1,
            "must be pinned to a single worker, not scattered across all"
        );
        assert!(cfg_b.is_active);
    }

    #[test]
    fn test_worker_drain_clears_stale_routes() {
        // P1: when the cluster loses all workers, the controller must clear the routing table
        // instead of leaving stale entries that point at gone workers (which would make the
        // filter scatter LoRA traffic to all available workers).
        let (mut controller, st, le, rt) = setup_controller();
        let w1 = make_worker(1);
        st.handle_mdc_addition(w1, &make_lora_info("lora-a", 4));
        le.increment_load("lora-a");
        controller.recompute_now();
        assert!(
            rt.get_config("lora-a").is_some(),
            "LoRA placed while a worker exists"
        );

        // Drain the cluster.
        st.handle_worker_removal(w1);
        controller.recompute_now();
        assert!(
            rt.get_config("lora-a").is_none(),
            "stale routing entry must be cleared after a full worker drain"
        );
    }

    #[test]
    fn test_zero_capacity_rebinds_dead_route_to_live_worker() {
        // P1 follow-up: with live workers but zero LoRA capacity, a route pointing only at a
        // now-removed worker must be rebound to a live worker (bounded), not left pointing at the
        // gone worker (which makes the filter scatter to all available workers).
        let (mut controller, _st, _le, rt) = setup_controller();
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        // w1 is live with zero capacity; w2 is never registered (gone).
        _st.set_worker_capacity(w1, 0);
        rt.update_allocation(
            "lora-a".to_string(),
            LoraReplicaConfig {
                lora_name: "lora-a".to_string(),
                replica_factor: 1,
                replica_set: vec![w2],
                updated_at: Instant::now(),
                is_active: true,
            },
        );

        controller.recompute_now(); // exercises the total_slots == 0 path

        let cfg = rt
            .get_config("lora-a")
            .expect("route should be rebound to a live worker, not dropped to scatter");
        assert_eq!(
            cfg.replica_set,
            vec![w1],
            "dead route must be rebound to the live worker w1, not left at the gone w2"
        );
    }

    #[test]
    fn test_replica_factor_matches_set_under_partial_capacity() {
        // P1/P2: under partial-capacity pressure the slot-aware allocator returns fewer workers
        // than desired. The stored replica_factor must equal the actual replica_set length, not
        // the (larger) desired count.
        let (mut controller, st, le, rt) = setup_controller();
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        st.set_worker_capacity(w1, 4); // free
        st.handle_mdc_addition(w2, &make_lora_info("lora-b", 1)); // w2 full (1/1) with lora-b
        // lora-a is the only active LoRA; its proportional desired is 2, but w2 is full so the
        // allocator can only place it on w1.
        le.increment_load("lora-a");
        le.increment_load("lora-a");

        controller.recompute_now();

        let cfg = rt.get_config("lora-a").unwrap();
        assert_eq!(
            cfg.replica_factor,
            cfg.replica_set.len(),
            "replica_factor must equal the actual replica_set size"
        );
        assert!(
            !cfg.replica_set.iter().any(|w| w.worker_id == 2),
            "must not place on the full worker w2"
        );
    }

    #[test]
    fn test_idle_capacity_only_worker_is_placement_target() {
        // jh-nv (watcher base-card seeding): a LoRA-capable worker with NO adapter loaded yet —
        // capacity seeded from the base MDC via set_worker_capacity — must be visible to the
        // controller and eligible for placement, instead of being excluded until some request has
        // already lazy-loaded an adapter onto it. Without the seeding, list_workers() would be
        // empty here and recompute would clear/return with no placement at all.
        let (mut controller, st, le, rt) = setup_controller();
        let w1 = make_worker(1);
        // Capacity-only registration: the worker advertises 4 LoRA slots, no adapter loaded.
        st.set_worker_capacity(w1, 4);
        assert_eq!(
            st.list_workers(),
            vec![w1],
            "capacity-only worker must be visible to the controller"
        );
        assert_eq!(
            st.total_lora_slots(),
            4,
            "its slots must count toward budget"
        );

        // A request arrives for lora-a (active), but no adapter is loaded anywhere yet.
        le.increment_load("lora-a");
        controller.recompute_now();

        let cfg = rt
            .get_config("lora-a")
            .expect("active LoRA must be placed on the idle capacity-only worker");
        assert_eq!(
            cfg.replica_set,
            vec![w1],
            "placement must target the idle-but-capable worker w1"
        );
        assert!(cfg.is_active);
    }

    #[test]
    fn test_mcf_changed_workers_includes_capacity_changes() {
        // MCF delta mode only reconsiders workers reported as changed. A capacity change on an
        // existing worker (same id, same LoRA set, same replica counts) must mark that worker
        // changed — otherwise delta MCF leaves its frozen placements stale and never uses the new
        // headroom.
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        let workers: HashSet<WorkerWithDpRank> = [w1, w2].into_iter().collect();
        let prev_caps: HashMap<WorkerWithDpRank, u32> = [(w1, 2), (w2, 2)].into_iter().collect();
        // Identical worker set; w1 capacity grew 2 -> 4, w2 unchanged.
        let cur_caps: HashMap<WorkerWithDpRank, u32> = [(w1, 4), (w2, 2)].into_iter().collect();

        let changed =
            LoraController::compute_changed_workers(&workers, &workers, &cur_caps, &prev_caps);
        assert!(
            changed.contains(&w1),
            "a capacity change must mark the worker changed for delta MCF"
        );
        assert!(
            !changed.contains(&w2),
            "an unchanged-capacity worker must not be marked changed"
        );

        // No id or capacity change -> nothing changed.
        assert!(
            LoraController::compute_changed_workers(&workers, &workers, &prev_caps, &prev_caps)
                .is_empty(),
            "no id/capacity change must yield no changed workers"
        );
    }
}
