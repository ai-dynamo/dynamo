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
use dynamo_runtime::protocols::EndpointId;

#[derive(Debug, Clone)]
struct HysteresisState {
    last_scale_down_tick: u64,
}

/// Process-global guard for the legacy unscoped controller entrypoint. Endpoint-scoped
/// controllers publish into independent Prometheus label sets and do not use this guard.
static CONTROLLER_RUNNING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// RAII release for [`CONTROLLER_RUNNING`]. Held inside the controller's spawned task only when
/// this controller actually acquired the guard; its `Drop` clears the flag on ANY task exit —
/// clean cancel, a panic that escapes the per-tick `catch_unwind`, or task abort (the future is
/// dropped) — so a legitimate restart is never wedged. A controller that did NOT acquire the guard
/// (a duplicate) carries no releaser and so cannot clear the real owner's flag.
struct ControllerRunningGuard;

impl Drop for ControllerRunningGuard {
    fn drop(&mut self) {
        CONTROLLER_RUNNING.store(false, std::sync::atomic::Ordering::SeqCst);
    }
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
    metrics_endpoint: String,
    published_metric_loras: HashSet<String>,
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
            metrics_endpoint: "unscoped".to_string(),
            published_metric_loras: HashSet::new(),
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
    ///
    /// This legacy entrypoint retains the single unscoped-controller guard. New serving paths use
    /// [`Self::start_for_endpoint`] so controller state and metrics remain endpoint-qualified.
    pub fn start(
        config: LoraAllocationConfig,
        routing_table: LoraRoutingTable,
        state_tracker: LoraStateTracker,
        load_estimator: Arc<LoadEstimator>,
        cancel_token: tokio_util::sync::CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        Self::start_inner(
            None,
            config,
            routing_table,
            state_tracker,
            load_estimator,
            cancel_token,
        )
    }

    pub fn start_for_endpoint(
        endpoint_id: EndpointId,
        config: LoraAllocationConfig,
        routing_table: LoraRoutingTable,
        state_tracker: LoraStateTracker,
        load_estimator: Arc<LoadEstimator>,
        cancel_token: tokio_util::sync::CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        Self::start_inner(
            Some(endpoint_id),
            config,
            routing_table,
            state_tracker,
            load_estimator,
            cancel_token,
        )
    }

    fn start_inner(
        endpoint_id: Option<EndpointId>,
        config: LoraAllocationConfig,
        routing_table: LoraRoutingTable,
        state_tracker: LoraStateTracker,
        load_estimator: Arc<LoadEstimator>,
        cancel_token: tokio_util::sync::CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        use std::sync::atomic::Ordering;
        // Process-global guard: the LoRA metrics gauges are unlabeled singletons, so only one
        // controller may own them. `acquired` is true iff we won the swap; only the winner carries
        // a `ControllerRunningGuard` into the task, whose Drop releases the flag on ANY exit
        // (cancel/panic/abort) so a clean restart is never wedged.
        let acquired = endpoint_id
            .is_none()
            .then(|| !CONTROLLER_RUNNING.swap(true, Ordering::SeqCst));
        if acquired == Some(false) {
            tracing::error!(
                "A LoRA allocation controller is already running in this process. Starting a \
                 second one is unsupported: they share process-global, unlabeled Prometheus LoRA \
                 gauges and would clobber each other's metrics every tick. There must be exactly \
                 one LoRA controller per process."
            );
        }
        let release_guard =
            acquired.and_then(|acquired| acquired.then_some(ControllerRunningGuard));

        let timestep = Duration::from_secs(config.timestep_secs);
        let mut controller = Self::new(config, routing_table, state_tracker, load_estimator);
        if let Some(endpoint_id) = &endpoint_id {
            controller.metrics_endpoint = endpoint_id.to_string();
        }

        tokio::spawn(async move {
            // Released on ANY exit of this task (clean cancel, escaped panic, or abort).
            let _release_guard = release_guard;
            let mut interval = tokio::time::interval(timestep);
            tracing::info!(
                endpoint = endpoint_id.as_ref().map(ToString::to_string),
                timestep_secs = controller.config.timestep_secs,
                algorithm = controller.allocator.name(),
                "LoRA allocation controller started"
            );

            loop {
                tokio::select! {
                    _ = cancel_token.cancelled() => {
                        tracing::debug!(
                            endpoint = endpoint_id.as_ref().map(ToString::to_string),
                            "LoRA allocation controller shutting down"
                        );
                        // `_release_guard` Drop clears CONTROLLER_RUNNING as this task unwinds.
                        break;
                    }
                    _ = interval.tick() => {
                        // A panic inside recompute must not silently kill the controller loop —
                        // that would freeze all LoRA allocation for the process lifetime with no
                        // restart and no alert. Catch it, log, and continue; the next tick
                        // recomputes fresh from current cluster state (a partially-updated table
                        // self-heals). `&mut controller` across the unwind boundary needs
                        // AssertUnwindSafe.
                        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            controller.recompute_allocations();
                        }));
                        if let Err(panic) = outcome {
                            let msg = panic
                                .downcast_ref::<&str>()
                                .map(|s| s.to_string())
                                .or_else(|| panic.downcast_ref::<String>().cloned())
                                .unwrap_or_else(|| "unknown panic".to_string());
                            tracing::error!(
                                tick = controller.tick,
                                panic = %msg,
                                "LoRA allocation recompute panicked; skipping this tick and continuing"
                            );
                        }
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

        // State-tracker iteration order is intentionally unspecified (DashMap). Keep worker input
        // order stable so equal-cost MCF paths and capacity-aware per-LoRA allocation converge to
        // the same routing table across controllers and repeated runs.
        let mut workers = self.state_tracker.list_workers();
        workers.sort();
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
            // Workers exist but advertise zero LoRA capacity. Backends only report LoRA slots when
            // LoRA serving is enabled/capable, so zero total slots means no live worker can accept
            // a lazy LoRA load. Fail closed: keep a route ONLY where the adapter is already loaded
            // (warm) on a live worker — those routes serve an in-memory adapter without triggering
            // a new load. Drop every other entry so the LoRA becomes unroutable instead of being
            // HRW-pinned to a live (possibly non-LoRA-capable) worker that would just fail the
            // load — pinning would route LoRA traffic to a worker that cannot serve it. A removed
            // entry's LoRA can still be served via the filter's runtime loaded-worker fallback if
            // it is loaded somewhere; otherwise it stays intentionally unroutable until capacity
            // appears. Then skip recompute.
            let live: std::collections::HashSet<WorkerWithDpRank> =
                workers.iter().copied().collect();
            for (name, cfg) in self.routing_table.snapshot_configs() {
                let loaded = self.state_tracker.get_loaded_workers(&name);
                let warm: Vec<WorkerWithDpRank> = cfg
                    .replica_set
                    .iter()
                    .copied()
                    .filter(|w| live.contains(w) && loaded.contains(w))
                    .collect();
                if warm.is_empty() {
                    self.routing_table.remove_lora(&name);
                    self.hysteresis.remove(&name);
                } else if warm.len() != cfg.replica_set.len() {
                    self.update_routing_entry(&name, warm.len(), warm, cfg.is_active);
                }
            }
            // Refresh gauges from the pruned table so a narrowed route's replica_factor (and the
            // other LoRA gauges) don't stay stale while capacity remains zero. No allocation ran
            // this tick, so churn/overflow are zero.
            let table_snapshot = self.routing_table.snapshot_configs();
            let loads = self.load_estimator.get_current_load();
            let raw_arrival_counts = self.load_estimator.get_raw_arrival_counts();
            self.update_prometheus_metrics(&table_snapshot, &loads, &raw_arrival_counts);
            self.update_churn_metrics(0, 0, 0);
            tracing::debug!(
                "No LoRA slots available; preserved only warm live routes (fail-closed), \
                 skipping recompute"
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
        self.update_churn_metrics(0, 0, 0);

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
                self.update_churn_metrics(
                    total_loads as i64,
                    total_unloads as i64,
                    result.overflow_count as i64,
                );

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
                // still-loaded LoRA left unplaced must stay routable (REQ 7) without thrashing the
                // routing table, polluting the solver's keep/delta state, or overriding the overflow
                // decision by ignoring slot capacity:
                //   * Warm-first — narrow any existing entry to the workers where it is STILL warm
                //     (prior set ∩ loaded ∩ live): pure existing routes, no new load. (The full
                //     prior entry must NOT be kept as-is — the filter's tier-2 lazy-load path
                //     `replica_set ∩ available` would reload the adapter onto a still-live replica
                //     it was evicted from, defeating the overflow cap.)
                //   * Otherwise pin to a single deterministic SLOT-AWARE HRW worker via
                //     `compute_replica_set_with_slots`, EXACTLY as the HRW `dropped_active` path
                //     does (parity — the previous MCF fallback used the slot-blind
                //     `compute_replica_set`, which could cold-pin onto a full worker even while a
                //     free slot existed elsewhere; the slot-aware variant prefers a free slot and
                //     only falls back to a single bounded worker when every worker is full, which is
                //     unavoidable and bounded to one worker, never scattered). The backend enforces
                //     `max_*_lora_count`, so the bounded last-resort pin cannot over-subscribe.
                //   * Crucially, fallback pins are NEVER inserted into `result.assignment`: that
                //     value becomes `prev_assignment`, and feeding it phantom (non-MCF) placements
                //     corrupts the keep-reward/delta logic on the next tick and inflates churn.
                // LoRAs intentionally dropped by the capacity cap (active but truncated out of
                // active_replica_counts) are handled separately in recompute_allocations and must
                // NOT be re-pinned here (RR3-1).
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

                // Build this tick's projected per-worker usage from the committed routing table (the
                // placements the MCF solve already wrote above), excluding the unplaced LoRAs' own
                // stale entries since those are re-decided here. The slot-aware pin reads this so it
                // prefers a worker that still has a free slot and never overfills the same slot
                // across multiple unplaced LoRAs within the tick. Mirrors the HRW `dropped_active`
                // path so both algorithms place capacity-pressured LoRAs identically.
                let caps = self.state_tracker.get_worker_capacities();
                let unplaced_set: HashSet<&str> = unplaced.iter().map(String::as_str).collect();
                let mut projected: HashMap<WorkerWithDpRank, usize> = HashMap::new();
                for (name, cfg) in self.routing_table.snapshot_configs() {
                    if unplaced_set.contains(name.as_str()) {
                        continue;
                    }
                    for w in &cfg.replica_set {
                        *projected.entry(*w).or_insert(0) += 1;
                    }
                }

                for lora_name in unplaced {
                    let is_active = active_set.contains(lora_name.as_str());
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
                        // Slot-aware bounded pin against this tick's projected usage (parity with
                        // the HRW dropped_active path). Prefers a free slot; falls back to a single
                        // worker only when all are full.
                        let proj_usage: HashMap<WorkerWithDpRank, (usize, usize)> = workers
                            .iter()
                            .map(|w| {
                                let used = projected.get(w).copied().unwrap_or(0);
                                let cap = caps.get(w).copied().unwrap_or(0) as usize;
                                (*w, (used, cap))
                            })
                            .collect();
                        self.allocator.compute_replica_set_with_slots(
                            &lora_name,
                            workers,
                            1,
                            &proj_usage,
                        )
                    };
                    if chosen.is_empty() {
                        // No live worker at all — drop any stale entry rather than keep a dead route.
                        self.routing_table.remove_lora(&lora_name);
                        self.hysteresis.remove(&lora_name);
                        continue;
                    }
                    // Charge the chosen worker(s) so later unplaced pins see them occupied.
                    for w in &chosen {
                        *projected.entry(*w).or_insert(0) += 1;
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
                            "MCF left LoRA unplaced (capacity overflow); narrowed to warm workers or slot-aware bounded pin"
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
                    "MCF solver failed; keeping current routes but resetting delta baseline so the next tick re-solves from scratch"
                );
                // F6: the solve failed, so prev_* still describes the LAST SUCCESSFUL assignment —
                // one this tick did NOT apply. Keeping it would make the next tick's delta solve
                // diff against a stale baseline and could freeze placements indefinitely (new
                // in-budget LoRAs never reconsidered; removed workers stay frozen-in). Drop the
                // delta state so the next tick runs a full (non-delta) solve from current cluster
                // state. Existing routing-table entries are left intact for continuity; the
                // post-branch cleanup in recompute_allocations still prunes stale/dropped entries.
                self.prev_assignment.clear();
                self.prev_workers.clear();
                self.prev_worker_capacities.clear();
                self.prev_replica_counts.clear();
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
        // A zero cooldown is an explicit off switch (used by allocator comparisons and useful
        // operationally when immediate convergence matters). The generic first-scale-down path
        // below always defers once while it arms per-LoRA state, which would otherwise turn a
        // configured zero into an undocumented one-tick cooldown.
        if self.config.scale_down_cooldown_ticks == 0 {
            self.hysteresis.remove(lora_name);
            return desired_replicas;
        }

        if desired_replicas < current_replicas {
            // Copy the timestamp so the immutable borrow ends before the mutable re-arm below.
            match self
                .hysteresis
                .get(lora_name)
                .map(|h| h.last_scale_down_tick)
            {
                Some(last_scale_down_tick) => {
                    let ticks_since = self.tick.saturating_sub(last_scale_down_tick);
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
                    // Cooldown elapsed: apply this scale-down AND re-arm the cooldown to THIS tick
                    // so the next scale-down is rate-limited too. Without this refresh the timestamp
                    // stays at the first-deferral tick, which would let a continuing decline shrink
                    // replicas on every subsequent tick (the cooldown would only delay the first
                    // scale-down, not space out successive ones).
                    if let Some(h) = self.hysteresis.get_mut(lora_name) {
                        h.last_scale_down_tick = self.tick;
                    }
                    desired_replicas
                }
                None => {
                    self.hysteresis.insert(
                        lora_name.to_string(),
                        HysteresisState {
                            last_scale_down_tick: self.tick,
                        },
                    );
                    current_replicas
                }
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

    /// Republish this controller's endpoint-qualified LoRA gauges.
    fn update_prometheus_metrics(
        &mut self,
        table_snapshot: &[(String, LoraReplicaConfig)],
        loads: &HashMap<String, usize>,
        raw_arrival_counts: &HashMap<String, u64>,
    ) {
        use crate::http::service::metrics::{
            LORA_ACTIVE_REQUESTS_GAUGE, LORA_ESTIMATED_LOAD_GAUGE, LORA_IS_ACTIVE_GAUGE,
            LORA_RAW_ARRIVAL_COUNT_GAUGE, LORA_REPLICA_FACTOR_GAUGE,
        };

        let inflight = self.load_estimator.get_inflight_counts();
        let current_loras = table_snapshot
            .iter()
            .map(|(name, _)| name.clone())
            .chain(inflight.keys().cloned())
            .collect::<HashSet<_>>();
        for stale_lora in self.published_metric_loras.difference(&current_loras) {
            let labels = [self.metrics_endpoint.as_str(), stale_lora.as_str()];
            let _ = LORA_REPLICA_FACTOR_GAUGE.remove_label_values(&labels);
            let _ = LORA_IS_ACTIVE_GAUGE.remove_label_values(&labels);
            let _ = LORA_RAW_ARRIVAL_COUNT_GAUGE.remove_label_values(&labels);
            let _ = LORA_ESTIMATED_LOAD_GAUGE.remove_label_values(&labels);
            let _ = LORA_ACTIVE_REQUESTS_GAUGE.remove_label_values(&labels);
        }

        for (lora_name, config) in table_snapshot {
            let labels = [self.metrics_endpoint.as_str(), lora_name.as_str()];
            LORA_REPLICA_FACTOR_GAUGE
                .with_label_values(&labels)
                .set(config.replica_factor as i64);
            LORA_IS_ACTIVE_GAUGE
                .with_label_values(&labels)
                .set(if config.is_active { 1 } else { 0 });
            let raw_count = raw_arrival_counts.get(lora_name).copied().unwrap_or(0);
            LORA_RAW_ARRIVAL_COUNT_GAUGE
                .with_label_values(&labels)
                .set(raw_count as i64);
            let load = loads.get(lora_name).copied().unwrap_or(0);
            LORA_ESTIMATED_LOAD_GAUGE
                .with_label_values(&labels)
                .set(load as i64);
        }

        for (lora_name, count) in &inflight {
            LORA_ACTIVE_REQUESTS_GAUGE
                .with_label_values(&[self.metrics_endpoint.as_str(), lora_name.as_str()])
                .set(*count as i64);
        }
        self.published_metric_loras = current_loras;
    }

    fn update_churn_metrics(&self, loads: i64, unloads: i64, overflow: i64) {
        use crate::http::service::metrics::{
            LORA_CHURN_LOADS_GAUGE, LORA_CHURN_UNLOADS_GAUGE, LORA_OVERFLOW_COUNT_GAUGE,
        };

        let endpoint = [self.metrics_endpoint.as_str()];
        LORA_CHURN_LOADS_GAUGE
            .with_label_values(&endpoint)
            .set(loads);
        LORA_CHURN_UNLOADS_GAUGE
            .with_label_values(&endpoint)
            .set(unloads);
        LORA_OVERFLOW_COUNT_GAUGE
            .with_label_values(&endpoint)
            .set(overflow);
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

        // In-flight requests can still be draining even with no workers; the normal publisher
        // keeps those series while removing stale allocation series for only this endpoint.
        self.update_prometheus_metrics(&[], &HashMap::new(), &HashMap::new());
        self.update_churn_metrics(0, 0, 0);
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

    fn setup_mcf_controller() -> (
        LoraController,
        LoraStateTracker,
        Arc<LoadEstimator>,
        LoraRoutingTable,
    ) {
        let config = LoraAllocationConfig {
            algorithm: AllocationAlgorithmType::MinCostFlow,
            ..LoraAllocationConfig::default()
        };
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
    fn test_zero_capacity_fails_closed_drops_non_warm_route() {
        // With live workers but zero total LoRA capacity, no worker can accept a lazy load, so the
        // controller must FAIL CLOSED: a route to a non-warm worker (here a gone w2, and the
        // adapter is not loaded on the live w1) must be REMOVED, not rebound onto the live worker.
        // Rebinding would pin LoRA traffic to a worker that has no LoRA capacity and would just
        // fail the load.
        let (mut controller, st, _le, rt) = setup_controller();
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        // w1 is live with zero capacity; w2 is never registered (gone). lora-a is not loaded
        // anywhere, so there is no warm worker.
        st.set_worker_capacity(w1, 0);
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

        assert!(
            rt.get_config("lora-a").is_none(),
            "with zero LoRA capacity and no warm worker, the route must be dropped (fail closed), \
             not pinned to a non-LoRA-capable worker"
        );
    }

    #[test]
    fn test_zero_capacity_preserves_warm_live_route() {
        // Fail-closed must still preserve an already-loaded (warm) route: if the adapter is loaded
        // on a live worker, that route serves the in-memory adapter without a new load and must be
        // kept (narrowed to the warm live subset).
        let (mut controller, st, _le, rt) = setup_controller();
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        // w1 is live and HAS lora-a loaded; w2 is gone. Force zero total capacity by registering
        // w1 with zero slots AFTER recording the loaded adapter would normally bump capacity, so
        // instead model "loaded but zero advertised slots" directly.
        st.set_worker_capacity(w1, 0);
        // Mark lora-a as loaded on the live w1 (warm) without granting capacity.
        st.handle_mdc_addition(w1, &make_lora_info("lora-a", 0));
        assert_eq!(st.total_lora_slots(), 0, "test requires zero total slots");
        rt.update_allocation(
            "lora-a".to_string(),
            LoraReplicaConfig {
                lora_name: "lora-a".to_string(),
                replica_factor: 2,
                replica_set: vec![w1, w2],
                updated_at: Instant::now(),
                is_active: true,
            },
        );

        controller.recompute_now(); // total_slots == 0 path

        let cfg = rt
            .get_config("lora-a")
            .expect("a warm live route must be preserved under zero capacity");
        assert_eq!(
            cfg.replica_set,
            vec![w1],
            "route must be narrowed to the warm live worker w1, dropping the gone w2"
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

    #[test]
    fn test_zero_cooldown_disables_scale_down_hysteresis() {
        let (mut controller, _st, _le, _rt) = setup_controller();
        controller.config.scale_down_cooldown_ticks = 0;
        controller.tick = 1;

        assert_eq!(
            controller.apply_hysteresis("l", 1, 4),
            1,
            "zero cooldown must apply the first scale-down immediately"
        );
        assert!(
            !controller.hysteresis.contains_key("l"),
            "disabled hysteresis must not retain per-LoRA state"
        );
    }

    #[test]
    fn test_hysteresis_rate_limits_successive_scale_downs() {
        // Hysteresis must space out SUCCESSIVE scale-downs by the cooldown, not just delay the
        // first one. Regression: previously last_scale_down_tick was set only on the first
        // deferral and never refreshed, so once the cooldown elapsed a continuing decline shrank
        // replicas on every tick.
        let (mut controller, _st, _le, _rt) = setup_controller();
        assert_eq!(
            controller.config.scale_down_cooldown_ticks, 3,
            "test assumes the default 3-tick cooldown"
        );

        // First desire to scale down 5->3 is deferred for the cooldown window.
        controller.tick = 1;
        assert_eq!(
            controller.apply_hysteresis("l", 3, 5),
            5,
            "first scale-down is deferred"
        );
        controller.tick = 2;
        assert_eq!(
            controller.apply_hysteresis("l", 3, 5),
            5,
            "still within cooldown"
        );
        controller.tick = 3;
        assert_eq!(
            controller.apply_hysteresis("l", 3, 5),
            5,
            "still within cooldown"
        );

        // Cooldown elapsed (1 -> 4): scale-down 5->3 applied AND cooldown re-armed at tick 4.
        controller.tick = 4;
        assert_eq!(
            controller.apply_hysteresis("l", 3, 5),
            3,
            "scale-down applied after the cooldown"
        );

        // A further decline 3->1 on the very next tick MUST be deferred (this is the bug fix:
        // previously it returned 1 immediately because the timestamp was stale).
        controller.tick = 5;
        assert_eq!(
            controller.apply_hysteresis("l", 1, 3),
            3,
            "a successive scale-down must wait a fresh cooldown, not apply next tick"
        );
        controller.tick = 6;
        assert_eq!(
            controller.apply_hysteresis("l", 1, 3),
            3,
            "still within the re-armed cooldown"
        );

        // Re-armed cooldown elapsed (4 -> 7): now 3->1 applies.
        controller.tick = 7;
        assert_eq!(
            controller.apply_hysteresis("l", 1, 3),
            1,
            "scale-down applied after the re-armed cooldown"
        );

        // Scale-up applies immediately and clears the hysteresis entry, so a later decline
        // re-arms the cooldown from scratch (first decline deferred again).
        controller.tick = 8;
        assert_eq!(
            controller.apply_hysteresis("l", 4, 1),
            4,
            "scale-up applies immediately"
        );
        controller.tick = 9;
        assert_eq!(
            controller.apply_hysteresis("l", 2, 4),
            4,
            "after a scale-up, the next decline is deferred again (cooldown re-armed from scratch)"
        );
    }

    #[test]
    fn test_mcf_overflow_bounded_pin_keeps_loras_routable() {
        // MCF genuine overflow: more cold-start LoRAs than slots. The MCF solver places what fits
        // and OMITS the rest. Each omitted LoRA must stay routable via a single bounded pin (REQ 7
        // keep-one-home; backend enforces max_*_lora_count), NOT be scattered across all workers.
        let (mut controller, st, _le, rt) = setup_mcf_controller();
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        st.handle_mdc_addition(w1, &make_lora_info("lora-a", 1));
        st.handle_mdc_addition(w2, &make_lora_info("lora-b", 1));
        st.handle_mdc_addition(w1, &make_lora_info("lora-c", 1));
        assert_eq!(st.total_lora_slots(), 2, "two cap=1 workers => two slots");

        controller.recompute_now();

        let entries = rt.snapshot_configs();
        assert_eq!(
            entries.len(),
            3,
            "every cold-start LoRA stays routable (placed or slot-aware bounded pin)"
        );
        for (name, cfg) in &entries {
            assert_eq!(
                cfg.replica_set.len(),
                1,
                "LoRA {name} must get a single bounded replica, not be scattered across all workers"
            );
            let w = cfg.replica_set[0];
            assert!(w == w1 || w == w2, "LoRA {name} must pin to a live worker");
        }
    }

    #[test]
    fn test_mcf_overflow_fallback_is_slot_aware_not_slot_blind() {
        // Discriminating test (N-1): proves the MCF unplaced fallback uses the SLOT-AWARE
        // `compute_replica_set_with_slots` rather than the old slot-blind `compute_replica_set`.
        //
        // Construct an overflow that leaves a FREE worker so the two behaviors diverge: with
        // `candidate_m = 1` the MCF solve only considers each LoRA's single top-HRW worker. Two
        // LoRAs that both rank w1 as their top contend for w1 (cap 1) — one is placed, the other
        // overflows — while w2 is never a candidate and stays free. The overflowed LoRA's top-HRW
        // worker is the (now-full) w1, so:
        //   * slot-blind compute_replica_set would re-pin it onto the FULL w1 (cold load on a
        //     worker with no slot — the bug), landing BOTH LoRAs on w1.
        //   * slot-aware compute_replica_set_with_slots skips full w1 and pins it to the FREE w2.
        // Asserting the two LoRAs land on DIFFERENT workers therefore fails for slot-blind and
        // passes only for slot-aware — and also exercises projected-usage charging (w1 is marked
        // full by the committed MCF placement, which is what makes the fallback avoid it).
        let w1 = make_worker(1);
        let w2 = make_worker(2);

        // Find two adapter names that both rank w1 as their top HRW worker over {w1, w2}.
        let mut colliding: Vec<String> = Vec::new();
        for i in 0..10_000 {
            let name = format!("lora-{i}");
            let ranked = crate::lora::routing::RendezvousHasher::rank_workers(&name, &[w1, w2]);
            if ranked[0].0 == w1 {
                colliding.push(name);
                if colliding.len() == 2 {
                    break;
                }
            }
        }
        assert_eq!(
            colliding.len(),
            2,
            "need two adapters that both top-rank w1"
        );

        let config = LoraAllocationConfig {
            algorithm: AllocationAlgorithmType::MinCostFlow,
            mcf: crate::lora::config::McfConfig {
                candidate_m: 1,
                ..crate::lora::config::McfConfig::default()
            },
            ..LoraAllocationConfig::default()
        };
        let rt = LoraRoutingTable::new();
        let st = LoraStateTracker::new();
        let le = Arc::new(LoadEstimator::new());
        let mut controller = LoraController::new(config, rt.clone(), st.clone(), le.clone());

        // Both colliding adapters cold-loaded on w1 (cap 1); w2 is a free capacity-only worker.
        st.handle_mdc_addition(w1, &make_lora_info(&colliding[0], 1));
        st.handle_mdc_addition(w1, &make_lora_info(&colliding[1], 1));
        st.set_worker_capacity(w2, 1);
        assert_eq!(st.total_lora_slots(), 2);

        controller.recompute_now();

        let c0 = rt
            .get_config(&colliding[0])
            .expect("first colliding LoRA must stay routable");
        let c1 = rt
            .get_config(&colliding[1])
            .expect("second colliding LoRA must stay routable");
        assert_eq!(c0.replica_set.len(), 1);
        assert_eq!(c1.replica_set.len(), 1);
        let placed: std::collections::HashSet<u64> =
            [c0.replica_set[0].worker_id, c1.replica_set[0].worker_id]
                .into_iter()
                .collect();
        assert_eq!(
            placed,
            [w1.worker_id, w2.worker_id].into_iter().collect(),
            "slot-aware fallback must move the overflowed LoRA off the full w1 onto the free w2 \
             (slot-blind would pin both onto w1)"
        );
    }
}
