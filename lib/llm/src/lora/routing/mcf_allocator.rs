// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Min-Cost Flow LoRA Placement Solver
//!
//! Wraps the generic SSAP solver with the LoRA placement domain logic:
//! - HRW top-M candidate generation + prior-host inclusion (edge sparsification)
//! - Delta freezing: only re-solve for changed LoRAs/workers
//! - Cost function: α·rank + γ·w_l·(new) − β·w_l·(keep)
//! - Overflow handling via dummy worker
//!

use std::collections::{HashMap, HashSet};

use crate::kv_router::protocols::WorkerWithDpRank;
use crate::lora::routing::hrw::RendezvousHasher;

use super::min_cost_flow::MinCostFlowGraph;

/// Parameters for the MCF placement solver.
#[derive(Debug, Clone)]
pub struct McfSolveParams {
    /// Number of HRW top-M candidates per LoRA (default 16).
    pub candidate_m: usize,
    /// Preference weight for HRW rank (default 1).
    pub alpha_pref: i64,
    /// Penalty weight for loading a new LoRA on a worker (default 1000).
    pub gamma_load: i64,
    /// Reward weight for keeping a LoRA on its prior worker (default 250).
    pub beta_keep: i64,
    /// Cost assigned to the overflow dummy worker (default 10^12).
    pub overflow_cost: i64,
    /// Whether to allow overflow (soft infeasibility) or fail hard.
    pub allow_overflow: bool,
}

impl Default for McfSolveParams {
    fn default() -> Self {
        Self {
            candidate_m: 16,
            alpha_pref: 1,
            gamma_load: 1000,
            beta_keep: 250,
            overflow_cost: 1_000_000_000_000,
            allow_overflow: true,
        }
    }
}

/// Input for a single LoRA in the placement problem.
#[derive(Debug, Clone)]
pub struct LoraInput {
    pub name: String,
    /// Required number of replicas.
    pub replicas: usize,
    /// Churn weight (proportional to load time / impact). 1 = uniform.
    pub churn_weight: i64,
}

/// Input for a single worker in the placement problem.
#[derive(Debug, Clone)]
pub struct WorkerInput {
    pub worker: WorkerWithDpRank,
    /// Distinct-LoRA capacity K_s.
    pub capacity: usize,
}

/// Result of a placement solve.
#[derive(Debug, Clone)]
pub struct McfPlacementResult {
    /// LoRA name -> set of workers assigned.
    pub assignment: HashMap<String, HashSet<WorkerWithDpRank>>,
    /// Workers that need to load a new LoRA (per worker).
    pub loads: HashMap<WorkerWithDpRank, HashSet<String>>,
    /// Workers that should unload a LoRA (per worker).
    pub unloads: HashMap<WorkerWithDpRank, HashSet<String>>,
    /// Number of replica placements that overflowed (could not be placed).
    pub overflow_count: usize,
}

/// The MCF-based placement solver.
pub struct McfPlacementSolver {
    params: McfSolveParams,
}

impl McfPlacementSolver {
    pub fn new(params: McfSolveParams) -> Self {
        Self { params }
    }

    /// Solve the LoRA placement problem.
    ///
    /// # Arguments
    /// * `workers` - Available workers with their capacities.
    /// * `loras` - LoRAs with their replica requirements.
    /// * `prev_assignment` - Previous tick's assignment (for churn minimization).
    /// * `changed_loras` - LoRAs whose demand changed (None = treat all as changed).
    /// * `changed_workers` - Workers that joined or left (None = treat none as changed).
    pub fn solve(
        &self,
        workers: &[WorkerInput],
        loras: &[LoraInput],
        prev_assignment: &HashMap<String, HashSet<WorkerWithDpRank>>,
        changed_loras: Option<&HashSet<String>>,
        changed_workers: Option<&HashSet<WorkerWithDpRank>>,
    ) -> Result<McfPlacementResult, String> {
        if workers.is_empty() || loras.is_empty() {
            return Ok(McfPlacementResult {
                assignment: HashMap::new(),
                loads: HashMap::new(),
                unloads: HashMap::new(),
                overflow_count: 0,
            });
        }

        let worker_map: HashMap<WorkerWithDpRank, &WorkerInput> =
            workers.iter().map(|w| (w.worker, w)).collect();

        let changed_w = changed_workers.cloned().unwrap_or_default();

        // ── Step 1: Identify impacted LoRAs ──────────────────────────────────
        let impacted: HashSet<String> = if let Some(cl) = changed_loras {
            let mut imp = cl.clone();
            // Also include any LoRA whose prior hosts overlap with changed workers
            for lora in loras {
                if let Some(prev) = prev_assignment.get(&lora.name) {
                    if prev.iter().any(|w| changed_w.contains(w)) {
                        imp.insert(lora.name.clone());
                    }
                }
            }
            imp
        } else {
            // Treat all as impacted (first tick or full recompute)
            loras.iter().map(|l| l.name.clone()).collect()
        };

        // ── Step 2: Freeze unaffected assignments ────────────────────────────
        let mut frozen_hosts: HashMap<String, HashSet<WorkerWithDpRank>> = HashMap::new();
        let mut used_slots: HashMap<WorkerWithDpRank, usize> = HashMap::new();

        for lora in loras {
            if impacted.contains(&lora.name) {
                continue;
            }
            if let Some(prev) = prev_assignment.get(&lora.name) {
                // Keep prior hosts that are still alive (in worker_map)
                let keep: HashSet<WorkerWithDpRank> = prev
                    .iter()
                    .filter(|w| worker_map.contains_key(w))
                    .copied()
                    .take(lora.replicas)
                    .collect();
                for w in &keep {
                    *used_slots.entry(*w).or_insert(0) += 1;
                }
                frozen_hosts.insert(lora.name.clone(), keep);
            }
        }

        // Compute residual capacities and replica demands
        let rem_cap: HashMap<WorkerWithDpRank, usize> = workers
            .iter()
            .map(|w| {
                let used = used_slots.get(&w.worker).copied().unwrap_or(0);
                (w.worker, w.capacity.saturating_sub(used))
            })
            .collect();

        let active_loras: Vec<&LoraInput> = loras
            .iter()
            .filter(|l| {
                let frozen = frozen_hosts.get(&l.name).map(|s| s.len()).unwrap_or(0);
                l.replicas > frozen
            })
            .collect();

        let active_workers: Vec<&WorkerInput> = workers
            .iter()
            .filter(|w| rem_cap.get(&w.worker).copied().unwrap_or(0) > 0)
            .collect();

        let total_demand: usize = active_loras
            .iter()
            .map(|l| {
                let frozen = frozen_hosts.get(&l.name).map(|s| s.len()).unwrap_or(0);
                l.replicas.saturating_sub(frozen)
            })
            .sum();

        let total_cap: usize = active_workers
            .iter()
            .map(|w| rem_cap.get(&w.worker).copied().unwrap_or(0))
            .sum();

        // ── Step 3: Build MCF graph ──────────────────────────────────────────
        // Node layout: SRC | lora_0..lora_N | worker_0..worker_M [| overflow] | SNK
        let src = 0usize;
        let mut next_id = 1usize;

        let mut lora_node: HashMap<&str, usize> = HashMap::new();
        for l in &active_loras {
            lora_node.insert(&l.name, next_id);
            next_id += 1;
        }

        let mut worker_node: HashMap<WorkerWithDpRank, usize> = HashMap::new();
        for w in &active_workers {
            worker_node.insert(w.worker, next_id);
            next_id += 1;
        }

        let overflow_node = if self.params.allow_overflow && total_demand > total_cap {
            let id = next_id;
            next_id += 1;
            Some(id)
        } else {
            None
        };

        let snk = next_id;
        let total_nodes = snk + 1;

        let mut mcf = MinCostFlowGraph::new(total_nodes);

        // SRC -> LoRA nodes
        for l in &active_loras {
            let frozen_count = frozen_hosts.get(&l.name).map(|s| s.len()).unwrap_or(0);
            let rem_rep = l.replicas.saturating_sub(frozen_count);
            if rem_rep > 0 {
                mcf.add_edge(src, lora_node[l.name.as_str()], rem_rep as i64, 0);
            }
        }

        // Worker nodes -> SNK
        for w in &active_workers {
            let cap = rem_cap.get(&w.worker).copied().unwrap_or(0);
            if cap > 0 {
                mcf.add_edge(worker_node[&w.worker], snk, cap as i64, 0);
            }
        }

        // Overflow -> SNK
        if let Some(ov) = overflow_node {
            let overflow_cap = total_demand.saturating_sub(total_cap);
            mcf.add_edge(ov, snk, overflow_cap as i64, 0);
        }

        // ── Step 4: LoRA -> Worker edges (sparsified) ────────────────────────
        // Track which forward edge indices correspond to which worker for each LoRA
        let all_workers_sorted: Vec<WorkerWithDpRank> = {
            let mut ws: Vec<WorkerWithDpRank> = workers.iter().map(|w| w.worker).collect();
            ws.sort();
            ws
        };

        // For each active LoRA, build candidate set and add edges
        let mut lora_edge_info: HashMap<&str, Vec<(usize, WorkerWithDpRank)>> = HashMap::new();

        for l in &active_loras {
            let prev_hosts = prev_assignment.get(&l.name).cloned().unwrap_or_default();

            // HRW top-M candidates
            let ranked = RendezvousHasher::rank_workers(&l.name, &all_workers_sorted);
            let top_m: Vec<WorkerWithDpRank> = ranked
                .iter()
                .take(self.params.candidate_m)
                .map(|(w, _)| *w)
                .collect();

            // Build candidate list: top-M ∪ prev_hosts
            let mut cand: Vec<WorkerWithDpRank> = Vec::new();
            let mut seen: HashSet<WorkerWithDpRank> = HashSet::new();

            // HRW-ranked first
            for w in &top_m {
                if seen.insert(*w) {
                    cand.push(*w);
                }
            }
            // Prior hosts
            let mut prev_sorted: Vec<WorkerWithDpRank> = prev_hosts.iter().copied().collect();
            prev_sorted.sort();
            for w in prev_sorted {
                if seen.insert(w) {
                    cand.push(w);
                }
            }

            // Compute HRW rank index for each candidate
            let rank_map: HashMap<WorkerWithDpRank, usize> = ranked
                .iter()
                .enumerate()
                .map(|(i, (w, _))| (*w, i))
                .collect();

            let lora_node_id = lora_node[l.name.as_str()];
            let mut edges = Vec::new();

            for (rnk, w) in cand.iter().enumerate() {
                if let Some(&w_node) = worker_node.get(w) {
                    let cost = self.build_edge_cost(
                        l.churn_weight,
                        *rank_map.get(w).unwrap_or(&rnk),
                        prev_hosts.contains(w),
                    );
                    let edge_idx = mcf.edge_count(lora_node_id);
                    mcf.add_edge(lora_node_id, w_node, 1, cost);
                    edges.push((edge_idx, *w));
                }
            }

            // Overflow edge
            if let Some(ov) = overflow_node {
                let edge_idx = mcf.edge_count(lora_node_id);
                mcf.add_edge(lora_node_id, ov, 1, self.params.overflow_cost);
                edges.push((edge_idx, WorkerWithDpRank::new(u64::MAX, 0))); // sentinel
            }

            lora_edge_info.insert(&l.name, edges);
        }

        // ── Step 5: Solve ────────────────────────────────────────────────────
        let flow_needed = total_demand as i64;
        let result = mcf.min_cost_flow(src, snk, flow_needed);

        match result {
            Err(e) => {
                return Err(format!(
                    "MCF solver failed: {e}. Try increasing candidate_m or enabling overflow."
                ));
            }
            Ok((_flow, _cost)) => {}
        }

        // ── Step 6: Extract assignments ──────────────────────────────────────
        let overflow_sentinel = WorkerWithDpRank::new(u64::MAX, 0);
        let mut solved_hosts: HashMap<String, HashSet<WorkerWithDpRank>> = HashMap::new();
        let mut overflow_count = 0usize;

        for l in &active_loras {
            let lora_node_id = lora_node[l.name.as_str()];
            let edges = &lora_edge_info[l.name.as_str()];
            let mut hosts = HashSet::new();

            for &(edge_idx, worker) in edges {
                let flow = mcf.flow_on_edge(lora_node_id, edge_idx);
                if flow > 0 {
                    if worker == overflow_sentinel {
                        overflow_count += flow as usize;
                    } else {
                        hosts.insert(worker);
                    }
                }
            }

            solved_hosts.insert(l.name.clone(), hosts);
        }

        // ── Step 7: Merge frozen + solved ────────────────────────────────────
        let mut assignment: HashMap<String, HashSet<WorkerWithDpRank>> = HashMap::new();

        for l in loras {
            let mut hosts = frozen_hosts.get(&l.name).cloned().unwrap_or_default();
            if let Some(solved) = solved_hosts.get(&l.name) {
                hosts.extend(solved);
            }
            // Trim to exact replica count (deterministic: sort by worker_id)
            if hosts.len() > l.replicas {
                let mut sorted: Vec<WorkerWithDpRank> = hosts.into_iter().collect();
                sorted.sort();
                hosts = sorted.into_iter().take(l.replicas).collect();
            }
            if !hosts.is_empty() {
                assignment.insert(l.name.clone(), hosts);
            }
        }

        // ── Step 8: Compute diffs ────────────────────────────────────────────
        let mut loads: HashMap<WorkerWithDpRank, HashSet<String>> = HashMap::new();
        let mut unloads: HashMap<WorkerWithDpRank, HashSet<String>> = HashMap::new();

        for l in loras {
            let prev = prev_assignment.get(&l.name).cloned().unwrap_or_default();
            let now = assignment.get(&l.name).cloned().unwrap_or_default();

            for w in now.difference(&prev) {
                loads.entry(*w).or_default().insert(l.name.clone());
            }
            for w in prev.difference(&now) {
                if worker_map.contains_key(w) {
                    unloads.entry(*w).or_default().insert(l.name.clone());
                }
            }
        }

        Ok(McfPlacementResult {
            assignment,
            loads,
            unloads,
            overflow_count,
        })
    }

    /// Compute the cost for placing a LoRA on a worker.
    ///
    /// cost = α·rank + γ·w_l if new, or α·rank − β·w_l if keeping
    fn build_edge_cost(&self, churn_weight: i64, rank_index: usize, is_keep: bool) -> i64 {
        let mut c = self.params.alpha_pref * rank_index as i64;
        if is_keep {
            c -= self.params.beta_keep * churn_weight;
        } else {
            c += self.params.gamma_load * churn_weight;
        }
        c
    }

    pub fn params(&self) -> &McfSolveParams {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_workers(count: usize, capacity: usize) -> Vec<WorkerInput> {
        (0..count)
            .map(|i| WorkerInput {
                worker: WorkerWithDpRank::new(i as u64, 0),
                capacity,
            })
            .collect()
    }

    fn make_lora(name: &str, replicas: usize) -> LoraInput {
        LoraInput {
            name: name.to_string(),
            replicas,
            churn_weight: 1,
        }
    }

    #[test]
    fn test_simple_placement() {
        let solver = McfPlacementSolver::new(McfSolveParams::default());
        let workers = make_workers(3, 4);
        let loras = vec![make_lora("A", 2), make_lora("B", 1)];
        let prev = HashMap::new();

        let result = solver.solve(&workers, &loras, &prev, None, None).unwrap();

        assert_eq!(result.assignment["A"].len(), 2);
        assert_eq!(result.assignment["B"].len(), 1);
        assert_eq!(result.overflow_count, 0);
    }

    #[test]
    fn test_determinism() {
        let solver = McfPlacementSolver::new(McfSolveParams::default());
        let workers = make_workers(5, 4);
        let loras = vec![make_lora("A", 3), make_lora("B", 2), make_lora("C", 1)];
        let prev = HashMap::new();

        let r1 = solver.solve(&workers, &loras, &prev, None, None).unwrap();
        let r2 = solver.solve(&workers, &loras, &prev, None, None).unwrap();

        assert_eq!(r1.assignment, r2.assignment);
    }

    #[test]
    fn test_capacity_respected() {
        let solver = McfPlacementSolver::new(McfSolveParams::default());
        // 2 workers, capacity 2 each = 4 total slots
        let workers = make_workers(2, 2);
        let loras = vec![make_lora("A", 2), make_lora("B", 2)];
        let prev = HashMap::new();

        let result = solver.solve(&workers, &loras, &prev, None, None).unwrap();

        // Count per-worker assignments
        let mut worker_counts: HashMap<WorkerWithDpRank, usize> = HashMap::new();
        for hosts in result.assignment.values() {
            for w in hosts {
                *worker_counts.entry(*w).or_insert(0) += 1;
            }
        }
        for (w, count) in &worker_counts {
            let cap = workers.iter().find(|wi| wi.worker == *w).unwrap().capacity;
            assert!(
                *count <= cap,
                "Worker {:?} has {} assignments but capacity {}",
                w,
                count,
                cap
            );
        }
    }

    #[test]
    fn test_churn_minimization() {
        let solver = McfPlacementSolver::new(McfSolveParams::default());
        let workers = make_workers(4, 4);
        let loras = vec![make_lora("A", 2), make_lora("B", 2)];

        // First solve
        let prev = HashMap::new();
        let r1 = solver.solve(&workers, &loras, &prev, None, None).unwrap();

        // Second solve with same inputs but using r1 as prev_assignment
        let r2 = solver
            .solve(&workers, &loras, &r1.assignment, None, None)
            .unwrap();

        // With identical demand, assignment should be identical (zero churn)
        assert_eq!(r1.assignment, r2.assignment);
        assert!(
            r2.loads.values().all(|s| s.is_empty()),
            "Expected zero loads on stable demand"
        );
        assert!(
            r2.unloads.values().all(|s| s.is_empty()),
            "Expected zero unloads on stable demand"
        );
    }

    #[test]
    fn test_overflow_detection() {
        let solver = McfPlacementSolver::new(McfSolveParams::default());
        // 1 worker, capacity 1, but 2 LoRAs each needing 1 replica
        let workers = make_workers(1, 1);
        let loras = vec![make_lora("A", 1), make_lora("B", 1)];
        let prev = HashMap::new();

        let result = solver.solve(&workers, &loras, &prev, None, None).unwrap();
        assert!(result.overflow_count > 0, "Should detect overflow");
    }

    #[test]
    fn test_overflow_disabled() {
        let solver = McfPlacementSolver::new(McfSolveParams {
            allow_overflow: false,
            ..Default::default()
        });
        let workers = make_workers(1, 1);
        let loras = vec![make_lora("A", 1), make_lora("B", 1)];
        let prev = HashMap::new();

        let result = solver.solve(&workers, &loras, &prev, None, None);
        assert!(result.is_err(), "Should fail without overflow");
    }

    #[test]
    fn test_delta_solving() {
        let solver = McfPlacementSolver::new(McfSolveParams::default());
        let workers = make_workers(4, 4);
        let loras_v1 = vec![make_lora("A", 2), make_lora("B", 2)];

        let r1 = solver
            .solve(&workers, &loras_v1, &HashMap::new(), None, None)
            .unwrap();

        // Add a new LoRA C, only C is changed
        let loras_v2 = vec![make_lora("A", 2), make_lora("B", 2), make_lora("C", 1)];
        let changed = HashSet::from(["C".to_string()]);
        let r2 = solver
            .solve(&workers, &loras_v2, &r1.assignment, Some(&changed), None)
            .unwrap();

        // A and B should keep their assignments (frozen)
        assert_eq!(r2.assignment["A"], r1.assignment["A"]);
        assert_eq!(r2.assignment["B"], r1.assignment["B"]);
        // C should be placed
        assert_eq!(r2.assignment["C"].len(), 1);
    }

    #[test]
    fn test_empty_inputs() {
        let solver = McfPlacementSolver::new(McfSolveParams::default());
        let result = solver.solve(&[], &[], &HashMap::new(), None, None).unwrap();
        assert!(result.assignment.is_empty());
        assert_eq!(result.overflow_count, 0);
    }

    #[test]
    fn test_worker_removal_bounded_churn() {
        let solver = McfPlacementSolver::new(McfSolveParams::default());
        let workers_v1 = make_workers(4, 4);
        let loras = vec![make_lora("A", 2), make_lora("B", 2), make_lora("C", 1)];

        let r1 = solver
            .solve(&workers_v1, &loras, &HashMap::new(), None, None)
            .unwrap();

        // Remove worker 2
        let removed = WorkerWithDpRank::new(2, 0);
        let workers_v2: Vec<WorkerInput> = workers_v1
            .iter()
            .filter(|w| w.worker != removed)
            .cloned()
            .collect();
        let changed_w = HashSet::from([removed]);
        let r2 = solver
            .solve(&workers_v2, &loras, &r1.assignment, None, Some(&changed_w))
            .unwrap();

        // Removed worker should not appear in any assignment
        for hosts in r2.assignment.values() {
            assert!(
                !hosts.contains(&removed),
                "Removed worker should not be in assignment"
            );
        }

        // Churn should be bounded
        let total_loads: usize = r2.loads.values().map(|s| s.len()).sum();
        assert!(
            total_loads <= loras.len(),
            "Churn should be bounded by number of LoRAs"
        );
    }
}
