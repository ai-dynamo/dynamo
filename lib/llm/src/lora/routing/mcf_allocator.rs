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
        // Trivial: no LoRAs to place.
        if loras.is_empty() {
            return Ok(McfPlacementResult {
                assignment: HashMap::new(),
                loads: HashMap::new(),
                unloads: HashMap::new(),
                overflow_count: 0,
            });
        }

        // LoRAs exist but there are no workers. Every required replica
        // overflows: route through the overflow path if allowed, otherwise
        // fail hard to match the behaviour of the main solver under
        // allow_overflow = false.
        if workers.is_empty() {
            let total_demand: usize = loras.iter().map(|l| l.replicas).sum();
            if !self.params.allow_overflow {
                return Err(format!(
                    "MCF solver failed: no workers available but {} replica(s) required across {} LoRA(s); \
                     enable overflow or provision workers.",
                    total_demand,
                    loras.len(),
                ));
            }
            return Ok(McfPlacementResult {
                assignment: HashMap::new(),
                loads: HashMap::new(),
                unloads: HashMap::new(),
                overflow_count: total_demand,
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
                if let Some(prev) = prev_assignment.get(&lora.name)
                    && prev.iter().any(|w| changed_w.contains(w))
                {
                    imp.insert(lora.name.clone());
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

        // ── Candidate pre-computation ────────────────────────────────────────
        // Build candidate sets for all active LoRAs before constructing the
        // MCF graph. The HRW window is expanded to max(candidate_m, rem_rep)
        // so that every LoRA always has enough outgoing edges to place all its
        // replicas on real workers when the global capacity allows it.
        //
        // Note: we do NOT try to pre-compute per-LoRA reach deficits here.
        // Candidate-graph matching conflicts (two LoRAs competing for the same
        // top-ranked worker) are invisible to a per-LoRA reachability check
        // but can still leave the MCF infeasible. The robust solution is to
        // always attach an overflow escape when allow_overflow=true (see Step 3).
        let all_workers_sorted: Vec<WorkerWithDpRank> = {
            let mut ws: Vec<WorkerWithDpRank> = workers.iter().map(|w| w.worker).collect();
            ws.sort();
            ws
        };

        struct LoraCandInfo {
            cand: Vec<WorkerWithDpRank>,
            rank_map: HashMap<WorkerWithDpRank, usize>,
            prev_hosts: HashSet<WorkerWithDpRank>,
            rem_rep: usize,
        }

        // One entry per active LoRA, stored in the same order as `active_loras`.
        let mut lora_cands: Vec<LoraCandInfo> = Vec::with_capacity(active_loras.len());

        for l in &active_loras {
            let frozen_count = frozen_hosts.get(&l.name).map(|s| s.len()).unwrap_or(0);
            let rem_rep = l.replicas.saturating_sub(frozen_count);
            let prev_hosts = prev_assignment.get(&l.name).cloned().unwrap_or_default();

            let ranked = RendezvousHasher::rank_workers(&l.name, &all_workers_sorted);

            // Expand the HRW window to at least rem_rep so every solvable
            // placement has a corresponding edge in the flow graph.
            let take_n = self.params.candidate_m.max(rem_rep);

            let mut cand: Vec<WorkerWithDpRank> = Vec::new();
            let mut seen: HashSet<WorkerWithDpRank> = HashSet::new();
            for (w, _) in ranked.iter().take(take_n) {
                if seen.insert(*w) {
                    cand.push(*w);
                }
            }
            let mut prev_sorted: Vec<WorkerWithDpRank> = prev_hosts.iter().copied().collect();
            prev_sorted.sort();
            for w in prev_sorted {
                if seen.insert(w) {
                    cand.push(w);
                }
            }

            let rank_map: HashMap<WorkerWithDpRank, usize> = ranked
                .iter()
                .enumerate()
                .map(|(i, (w, _))| (*w, i))
                .collect();

            lora_cands.push(LoraCandInfo {
                cand,
                rank_map,
                prev_hosts,
                rem_rep,
            });
        }

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

        // Always attach an overflow escape when allow_overflow=true.
        // A conditional check (total_demand > total_cap, or per-LoRA reach
        // deficit) is insufficient: candidate-graph matching conflicts — two
        // LoRAs competing for the same sparsified top-ranked worker — are
        // invisible to any per-LoRA reachability heuristic. Providing the
        // escape unconditionally makes overflow the last resort in every
        // topology, while the high overflow_cost keeps it truly last-resort.
        let overflow_node = if self.params.allow_overflow && total_demand > 0 {
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

        // Overflow -> SNK: capacity = total_demand is a safe upper bound
        // (at most all demand can overflow).
        if let Some(ov) = overflow_node {
            mcf.add_edge(ov, snk, total_demand as i64, 0);
        }

        // ── Step 4: LoRA -> Worker edges (sparsified) ────────────────────────
        // Candidate sets were pre-computed above; reuse them here to avoid
        // re-running the HRW ranking.
        let mut lora_edge_info: HashMap<&str, Vec<(usize, WorkerWithDpRank)>> = HashMap::new();

        for (l, lc) in active_loras.iter().zip(lora_cands.iter()) {
            let lora_node_id = lora_node[l.name.as_str()];
            let mut edges = Vec::new();

            for (rnk, w) in lc.cand.iter().enumerate() {
                if let Some(&w_node) = worker_node.get(w) {
                    let cost = self.build_edge_cost(
                        l.churn_weight,
                        *lc.rank_map.get(w).unwrap_or(&rnk),
                        lc.prev_hosts.contains(w),
                    );
                    let edge_idx = mcf.edge_count(lora_node_id);
                    mcf.add_edge(lora_node_id, w_node, 1, cost);
                    edges.push((edge_idx, *w));
                }
            }

            // Overflow edge: capacity = rem_rep so a LoRA needing multiple
            // replicas can route all unplaceable demand through overflow
            // (cap=1 would silently bound per-LoRA overflow and surface as
            // InsufficientFlow even when an overflow path exists).
            if let Some(ov) = overflow_node {
                let edge_idx = mcf.edge_count(lora_node_id);
                mcf.add_edge(
                    lora_node_id,
                    ov,
                    lc.rem_rep as i64,
                    self.params.overflow_cost,
                );
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
    fn test_overflow_single_lora_multiple_replicas() {
        // 1 worker, capacity 1, single LoRA needing 3 replicas. 1 replica fits;
        // the other 2 must overflow. Regression test for the per-LoRA overflow
        // edge capacity bug (cap=1 silently bounded overflow per LoRA).
        let solver = McfPlacementSolver::new(McfSolveParams::default());
        let workers = make_workers(1, 1);
        let loras = vec![make_lora("A", 3)];
        let prev = HashMap::new();

        let result = solver
            .solve(&workers, &loras, &prev, None, None)
            .expect("MCF should solve when overflow has correct capacity");
        assert_eq!(
            result.overflow_count, 2,
            "2 of 3 replicas must overflow when only 1 worker slot is available"
        );
        assert_eq!(
            result.assignment.get("A").map(|s| s.len()).unwrap_or(0),
            1,
            "exactly 1 replica should be placed on the available worker"
        );
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
    fn test_no_workers_with_loras_overflows() {
        let solver = McfPlacementSolver::new(McfSolveParams::default());
        let loras = vec![make_lora("A", 2), make_lora("B", 3)];
        let result = solver
            .solve(&[], &loras, &HashMap::new(), None, None)
            .expect("allow_overflow defaults to true");
        assert!(
            result.assignment.is_empty(),
            "no workers means no assignments"
        );
        assert_eq!(
            result.overflow_count, 5,
            "all required replicas should overflow when no workers exist"
        );
    }

    #[test]
    fn test_small_candidate_m_solvable_without_overflow() {
        // Regression: 5 workers (cap=1 each), candidate_m=1, 1 LoRA needing
        // 3 replicas. total_cap (5) >= total_demand (3) so a valid placement
        // exists, but with candidate_m=1 the old code only built 1 outgoing
        // edge from the LoRA node and returned InsufficientFlow.
        // Fix: expand the HRW window to max(candidate_m, rem_rep)=3, giving
        // the solver enough edges to place all 3 replicas on real workers.
        let solver = McfPlacementSolver::new(McfSolveParams {
            candidate_m: 1,
            ..Default::default()
        });
        let workers = make_workers(5, 1);
        let loras = vec![make_lora("A", 3)];
        let prev = HashMap::new();

        let result = solver
            .solve(&workers, &loras, &prev, None, None)
            .expect("should succeed: 5 workers can satisfy 3 replicas");
        assert_eq!(
            result.overflow_count, 0,
            "all replicas should land on real workers, not overflow"
        );
        assert_eq!(
            result.assignment.get("A").map(|s| s.len()).unwrap_or(0),
            3,
            "all 3 replicas of LoRA A must be assigned"
        );
    }

    #[test]
    fn test_candidate_conflict_with_surplus_cap_does_not_hard_fail() {
        // Regression: candidate_m=1, two LoRAs needing 1 replica each, two
        // workers with capacity 1 each. total_cap (2) == total_demand (2) so
        // a valid placement exists globally, but if both LoRAs' HRW top-1 is
        // the same worker the old reach-deficit heuristic gave overflow_needed=0
        // and omitted the overflow node, causing InsufficientFlow.
        //
        // With allow_overflow=true the overflow node is always present, so the
        // solver can route one LoRA to the contested worker and overflow the
        // other — no hard failure regardless of HRW tiebreaking.
        let solver = McfPlacementSolver::new(McfSolveParams {
            candidate_m: 1,
            ..Default::default()
        });
        // Three workers give total_cap=3 > total_demand=2 (clear global surplus)
        // so the old global-deficit guard definitely would not have created overflow.
        let workers = make_workers(3, 1);
        let loras = vec![make_lora("A", 1), make_lora("B", 1)];
        let prev = HashMap::new();

        let result = solver
            .solve(&workers, &loras, &prev, None, None)
            .expect("solver must not hard-fail with allow_overflow=true");

        let total_placed: usize = result.assignment.values().map(|s| s.len()).sum();
        assert_eq!(
            total_placed + result.overflow_count,
            2,
            "placed + overflow must equal total demand"
        );
    }

    #[test]
    fn test_small_candidate_m_overflows_when_truly_insufficient() {
        // 2 workers (cap=1 each), candidate_m=1, 1 LoRA needing 3 replicas.
        // Even after expanding to max(1,3)=3 candidates, only 2 active
        // workers exist, so 1 replica must overflow.
        let solver = McfPlacementSolver::new(McfSolveParams {
            candidate_m: 1,
            ..Default::default()
        });
        let workers = make_workers(2, 1);
        let loras = vec![make_lora("A", 3)];
        let prev = HashMap::new();

        let result = solver
            .solve(&workers, &loras, &prev, None, None)
            .expect("should not hard-fail: overflow handles the shortfall");
        assert_eq!(
            result.overflow_count, 1,
            "exactly 1 replica should overflow when only 2 workers are available"
        );
        assert_eq!(
            result.assignment.get("A").map(|s| s.len()).unwrap_or(0),
            2,
            "2 replicas should be placed on the available workers"
        );
    }

    #[test]
    fn test_no_workers_with_loras_overflow_disabled_fails() {
        let solver = McfPlacementSolver::new(McfSolveParams {
            allow_overflow: false,
            ..Default::default()
        });
        let loras = vec![make_lora("A", 1)];
        let result = solver.solve(&[], &loras, &HashMap::new(), None, None);
        assert!(
            result.is_err(),
            "with allow_overflow=false, missing workers must surface as an error"
        );
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
