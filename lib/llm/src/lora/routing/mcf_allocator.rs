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
//! ## Overflow is a genuine last resort
//!
//! `overflow_count` means "replicas that could not be placed on any real
//! worker." Two properties guarantee this:
//!
//! 1. Every real LoRA→worker edge has a cost clamped strictly below
//!    `overflow_cost`, so the min-cost solver always prefers a real worker
//!    (even a non-preferred one) over the overflow escape.
//! 2. The sparse top-M candidate graph is only an optimization. If the sparse
//!    solve overflows while real capacity is still free (a candidate-matching
//!    conflict — e.g. several LoRAs contending for the same top-ranked worker),
//!    the solver retries once with a full bipartite graph over all active
//!    workers. The dense retry satisfies Hall's condition, so overflow then
//!    reflects only a true aggregate-capacity shortage.

use std::collections::{HashMap, HashSet};

use crate::kv_router::protocols::WorkerWithDpRank;
use crate::lora::routing::hrw::RendezvousHasher;

use super::min_cost_flow::MinCostFlowGraph;

/// Per-LoRA data shared across graph-build attempts (HRW order, prior hosts,
/// residual replica demand). Computed once; reused by both the sparse and the
/// dense fallback solve.
struct LoraMeta {
    /// All workers in HRW-preference order (best first).
    ranked: Vec<WorkerWithDpRank>,
    /// Worker -> its index in `ranked` (the HRW rank used for edge cost).
    rank_map: HashMap<WorkerWithDpRank, usize>,
    /// Workers that hosted this LoRA on the previous tick.
    prev_hosts: HashSet<WorkerWithDpRank>,
    /// Replicas still to be placed after freezing (`replicas - frozen`).
    rem_rep: usize,
}

/// Candidate-set density for a single graph-build attempt.
#[derive(Clone, Copy)]
enum CandStrategy {
    /// HRW top-`max(candidate_m, rem_rep)` plus prior hosts (sparse fast path).
    Sparse,
    /// Every active worker is a candidate (full bipartite). Used only as a
    /// fallback when the sparse attempt overflowed with real capacity to spare.
    Dense,
}

/// Destination of a LoRA's outgoing flow edge. Replaces an in-band sentinel
/// `WorkerWithDpRank` value so a real worker can never be misread as overflow.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EdgeTarget {
    Worker(WorkerWithDpRank),
    Overflow,
}

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
    /// * `loras` - LoRAs with their replica requirements. To remove a LoRA
    ///   cleanly, pass it with `replicas = 0`; the solver will emit the
    ///   necessary unloads and produce no assignment for it. LoRAs omitted
    ///   entirely also produce unloads (belt-and-suspenders), but the
    ///   `replicas = 0` path is the preferred contract.
    /// * `prev_assignment` - Previous tick's assignment (for churn minimization).
    /// * `changed_loras` - LoRAs whose demand changed (None = treat all as changed).
    /// * `changed_workers` - Workers that joined or left (None = treat none as
    ///   changed). Pass a worker here when its capacity changes so frozen
    ///   assignments are re-evaluated; the solver detects over-committed workers
    ///   defensively but callers should still mark them explicitly.
    pub fn solve(
        &self,
        workers: &[WorkerInput],
        loras: &[LoraInput],
        prev_assignment: &HashMap<String, HashSet<WorkerWithDpRank>>,
        changed_loras: Option<&HashSet<String>>,
        changed_workers: Option<&HashSet<WorkerWithDpRank>>,
    ) -> Result<McfPlacementResult, String> {
        // No LoRAs desired. Compute unloads for any prior placements that
        // are still on live workers (workers that have since left handle
        // their own cleanup via handle_worker_removal / changed_workers).
        if loras.is_empty() {
            let live_workers: HashSet<WorkerWithDpRank> =
                workers.iter().map(|w| w.worker).collect();
            let mut unloads: HashMap<WorkerWithDpRank, HashSet<String>> = HashMap::new();
            for (lora_name, prev_workers) in prev_assignment {
                for w in prev_workers {
                    if live_workers.contains(w) {
                        unloads.entry(*w).or_default().insert(lora_name.clone());
                    }
                }
            }
            return Ok(McfPlacementResult {
                assignment: HashMap::new(),
                loads: HashMap::new(),
                unloads,
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

        // Reject duplicate worker or LoRA identities. The flow graph keys nodes
        // on them; duplicates would silently overwrite a node and double-count
        // sink capacity, corrupting the solve.
        let mut seen_workers: HashSet<WorkerWithDpRank> = HashSet::with_capacity(workers.len());
        for w in workers {
            if !seen_workers.insert(w.worker) {
                return Err(format!(
                    "MCF solver: duplicate worker {:?} in input; workers must be unique",
                    w.worker
                ));
            }
        }
        let mut seen_loras: HashSet<&str> = HashSet::with_capacity(loras.len());
        for l in loras {
            if !seen_loras.insert(l.name.as_str()) {
                return Err(format!(
                    "MCF solver: duplicate LoRA '{}' in input; LoRA names must be unique",
                    l.name
                ));
            }
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

        // Belt-and-suspenders guard: if a worker's capacity shrank since
        // the last tick and frozen assignments now exceed that capacity,
        // saturating_sub would silently hide the violation. Detect every
        // over-committed worker and unfreeze all LoRAs touching it so the
        // MCF solver re-places them within the new capacity bounds.
        // Callers should already pass changed_workers for such workers,
        // which adds their LoRAs to `impacted` before the freeze step; this
        // handles any cases that slip through that path.
        let over_committed: HashSet<WorkerWithDpRank> = workers
            .iter()
            .filter(|w| used_slots.get(&w.worker).copied().unwrap_or(0) > w.capacity)
            .map(|w| w.worker)
            .collect();

        if !over_committed.is_empty() {
            let names_to_unfreeze: Vec<String> = frozen_hosts
                .iter()
                .filter(|(_, hosts)| hosts.iter().any(|w| over_committed.contains(w)))
                .map(|(name, _)| name.clone())
                .collect();
            for name in names_to_unfreeze {
                if let Some(hosts) = frozen_hosts.remove(&name) {
                    for w in &hosts {
                        if let Some(c) = used_slots.get_mut(w) {
                            *c = c.saturating_sub(1);
                        }
                    }
                }
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

        // ── Per-LoRA metadata (computed once, reused by both solve attempts) ──
        // HRW ranking, prior hosts, and residual demand do not depend on the
        // candidate density, so derive them once here. The candidate list
        // itself is built inside `build_and_solve` per attempt.
        let all_workers_sorted: Vec<WorkerWithDpRank> = {
            let mut ws: Vec<WorkerWithDpRank> = workers.iter().map(|w| w.worker).collect();
            ws.sort();
            ws
        };

        // One entry per active LoRA, in the same order as `active_loras`.
        let metas: Vec<LoraMeta> = active_loras
            .iter()
            .map(|l| {
                let frozen_count = frozen_hosts.get(&l.name).map(|s| s.len()).unwrap_or(0);
                let rem_rep = l.replicas.saturating_sub(frozen_count);
                let prev_hosts = prev_assignment.get(&l.name).cloned().unwrap_or_default();
                let ranked_pairs = RendezvousHasher::rank_workers(&l.name, &all_workers_sorted);
                let rank_map: HashMap<WorkerWithDpRank, usize> = ranked_pairs
                    .iter()
                    .enumerate()
                    .map(|(i, (w, _))| (*w, i))
                    .collect();
                let ranked: Vec<WorkerWithDpRank> =
                    ranked_pairs.into_iter().map(|(w, _)| w).collect();
                LoraMeta {
                    ranked,
                    rank_map,
                    prev_hosts,
                    rem_rep,
                }
            })
            .collect();

        // ── Solve: sparse fast path, dense fallback on spurious overflow ──────
        let total_cap: usize = active_workers
            .iter()
            .map(|w| rem_cap.get(&w.worker).copied().unwrap_or(0))
            .sum();

        let sparse = self.build_and_solve(
            &active_loras,
            &active_workers,
            &rem_cap,
            &metas,
            total_demand,
            CandStrategy::Sparse,
        );

        // Retry with a full bipartite graph when the sparse attempt either
        // overflowed while real capacity was still free (a candidate-matching
        // conflict), or failed outright under allow_overflow=false (a denser
        // graph is a strict superset and may be feasible).
        let retry_dense = match &sparse {
            Ok((_, overflow)) => {
                *overflow > 0 && total_cap > total_demand.saturating_sub(*overflow)
            }
            Err(_) => true,
        };

        let (solved_hosts, overflow_count) = if retry_dense {
            self.build_and_solve(
                &active_loras,
                &active_workers,
                &rem_cap,
                &metas,
                total_demand,
                CandStrategy::Dense,
            )?
        } else {
            sparse?
        };

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

        // LoRAs present in the current desired set.
        let lora_names_in_input: HashSet<&str> = loras.iter().map(|l| l.name.as_str()).collect();

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

        // LoRAs that existed in prev_assignment but were omitted from the
        // desired `loras` slice entirely. Callers should pass replicas=0
        // entries for clean removal, but as a correctness guarantee we also
        // emit unloads here for any prior placements on live workers.
        for (lora_name, prev_workers) in prev_assignment {
            if lora_names_in_input.contains(lora_name.as_str()) {
                continue;
            }
            for w in prev_workers {
                if worker_map.contains_key(w) {
                    unloads.entry(*w).or_default().insert(lora_name.clone());
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

    /// Build the flow graph for one candidate-density `strategy`, solve it, and
    /// return `(solved_hosts, overflow_count)`.
    ///
    /// `metas` must be parallel to `active_loras`. Saturating/clamped edge costs
    /// (see [`Self::build_edge_cost`]) guarantee every real worker edge is
    /// strictly cheaper than the overflow escape, so the min-cost solver only
    /// routes to overflow when no real worker path exists.
    fn build_and_solve(
        &self,
        active_loras: &[&LoraInput],
        active_workers: &[&WorkerInput],
        rem_cap: &HashMap<WorkerWithDpRank, usize>,
        metas: &[LoraMeta],
        total_demand: usize,
        strategy: CandStrategy,
    ) -> Result<(HashMap<String, HashSet<WorkerWithDpRank>>, usize), String> {
        // Node layout: SRC | lora_0..lora_N | worker_0..worker_M [| overflow] | SNK
        let src = 0usize;
        let mut next_id = 1usize;

        let mut lora_node: HashMap<&str, usize> = HashMap::new();
        for l in active_loras {
            lora_node.insert(l.name.as_str(), next_id);
            next_id += 1;
        }

        let mut worker_node: HashMap<WorkerWithDpRank, usize> = HashMap::new();
        for w in active_workers {
            worker_node.insert(w.worker, next_id);
            next_id += 1;
        }

        let overflow_node = if self.params.allow_overflow && total_demand > 0 {
            let id = next_id;
            next_id += 1;
            Some(id)
        } else {
            None
        };

        let snk = next_id;
        let mut mcf = MinCostFlowGraph::new(snk + 1);

        // SRC -> LoRA
        for (l, m) in active_loras.iter().zip(metas) {
            if m.rem_rep > 0 {
                mcf.add_edge(src, lora_node[l.name.as_str()], m.rem_rep as i64, 0);
            }
        }

        // Worker -> SNK
        for w in active_workers {
            let cap = rem_cap.get(&w.worker).copied().unwrap_or(0);
            if cap > 0 {
                mcf.add_edge(worker_node[&w.worker], snk, cap as i64, 0);
            }
        }

        // Overflow -> SNK: capacity = total_demand is a safe upper bound.
        if let Some(ov) = overflow_node {
            mcf.add_edge(ov, snk, total_demand as i64, 0);
        }

        // LoRA -> Worker (+ overflow) edges.
        let mut lora_edge_info: HashMap<&str, Vec<(usize, EdgeTarget)>> = HashMap::new();
        for (l, m) in active_loras.iter().zip(metas) {
            // Sparse keeps the HRW top-`max(candidate_m, rem_rep)`; dense takes
            // every active worker. Prior hosts are always included.
            let take_n = match strategy {
                CandStrategy::Sparse => self.params.candidate_m.max(m.rem_rep),
                CandStrategy::Dense => usize::MAX,
            };

            let mut seen: HashSet<WorkerWithDpRank> = HashSet::new();
            let mut cand: Vec<WorkerWithDpRank> = Vec::new();
            for w in m.ranked.iter().take(take_n) {
                if worker_node.contains_key(w) && seen.insert(*w) {
                    cand.push(*w);
                }
            }
            let mut prev_sorted: Vec<WorkerWithDpRank> = m.prev_hosts.iter().copied().collect();
            prev_sorted.sort();
            for w in prev_sorted {
                if worker_node.contains_key(&w) && seen.insert(w) {
                    cand.push(w);
                }
            }

            let lora_node_id = lora_node[l.name.as_str()];
            let mut edges = Vec::with_capacity(cand.len() + 1);
            for (fallback_rnk, w) in cand.iter().enumerate() {
                let w_node = worker_node[w];
                let rank = *m.rank_map.get(w).unwrap_or(&fallback_rnk);
                let cost = self.build_edge_cost(l.churn_weight, rank, m.prev_hosts.contains(w));
                let edge_idx = mcf.edge_count(lora_node_id);
                mcf.add_edge(lora_node_id, w_node, 1, cost);
                edges.push((edge_idx, EdgeTarget::Worker(*w)));
            }

            // Overflow edge: capacity = rem_rep so a multi-replica LoRA can route
            // all of its unplaceable demand through overflow (cap=1 would bound
            // per-LoRA overflow to 1 and surface as a spurious InsufficientFlow).
            if let Some(ov) = overflow_node {
                let edge_idx = mcf.edge_count(lora_node_id);
                mcf.add_edge(
                    lora_node_id,
                    ov,
                    m.rem_rep as i64,
                    self.params.overflow_cost,
                );
                edges.push((edge_idx, EdgeTarget::Overflow));
            }

            lora_edge_info.insert(l.name.as_str(), edges);
        }

        // Solve.
        mcf.min_cost_flow(src, snk, total_demand as i64)
            .map_err(|e| {
                format!("MCF solver failed: {e}. Try increasing candidate_m or enabling overflow.")
            })?;

        // Extract per-LoRA placements + overflow.
        let mut solved_hosts: HashMap<String, HashSet<WorkerWithDpRank>> = HashMap::new();
        let mut overflow_count = 0usize;
        for l in active_loras {
            let lora_node_id = lora_node[l.name.as_str()];
            let mut hosts = HashSet::new();
            for &(edge_idx, target) in &lora_edge_info[l.name.as_str()] {
                let flow = mcf.flow_on_edge(lora_node_id, edge_idx);
                if flow > 0 {
                    match target {
                        EdgeTarget::Worker(w) => {
                            hosts.insert(w);
                        }
                        EdgeTarget::Overflow => overflow_count += flow as usize,
                    }
                }
            }
            solved_hosts.insert(l.name.clone(), hosts);
        }

        Ok((solved_hosts, overflow_count))
    }

    /// Compute the cost for placing a LoRA on a worker.
    ///
    /// `cost = α·rank + γ·w_l` if new, or `α·rank − β·w_l` if keeping.
    ///
    /// All arithmetic is saturating and `churn_weight` is floored at 0 (a
    /// negative weight would invert the keep/load incentive). The result is
    /// clamped strictly below `overflow_cost` so that placing on any real
    /// worker — even a far-down-HRW fallback worker in a dense solve — is always
    /// cheaper than overflowing.
    fn build_edge_cost(&self, churn_weight: i64, rank_index: usize, is_keep: bool) -> i64 {
        let w = churn_weight.max(0);
        let rank_term = self.params.alpha_pref.saturating_mul(rank_index as i64);
        let c = if is_keep {
            rank_term.saturating_sub(self.params.beta_keep.saturating_mul(w))
        } else {
            rank_term.saturating_add(self.params.gamma_load.saturating_mul(w))
        };
        c.min(self.params.overflow_cost.saturating_sub(1))
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
    fn test_candidate_conflict_resolved_by_dense_fallback() {
        // candidate_m=1 forces each LoRA's sparse candidate set to a single
        // HRW-top worker. With many LoRAs and many capacity-1 workers, several
        // LoRAs inevitably contend for the same top worker, so the sparse solve
        // overflows even though aggregate capacity is ample. The dense fallback
        // must then place every replica on a real worker (overflow == 0),
        // proving overflow is a genuine last resort and not an artifact of
        // candidate sparsification.
        let solver = McfPlacementSolver::new(McfSolveParams {
            candidate_m: 1,
            ..Default::default()
        });
        let workers = make_workers(8, 1); // total_cap = 8
        let loras: Vec<LoraInput> = (0..8).map(|i| make_lora(&format!("L{i}"), 1)).collect();
        let prev = HashMap::new();

        let result = solver
            .solve(&workers, &loras, &prev, None, None)
            .expect("dense fallback must yield a feasible solve");

        assert_eq!(
            result.overflow_count, 0,
            "with capacity == demand the dense fallback must place every replica"
        );
        let total_placed: usize = result.assignment.values().map(|s| s.len()).sum();
        assert_eq!(total_placed, 8, "all 8 replicas must land on real workers");

        // No worker exceeds its capacity of 1.
        let mut counts: HashMap<WorkerWithDpRank, usize> = HashMap::new();
        for hosts in result.assignment.values() {
            for w in hosts {
                *counts.entry(*w).or_insert(0) += 1;
            }
        }
        assert!(
            counts.values().all(|&c| c <= 1),
            "capacity-1 workers must not be double-assigned"
        );
    }

    #[test]
    fn test_duplicate_worker_rejected() {
        let solver = McfPlacementSolver::new(McfSolveParams::default());
        let dup = WorkerWithDpRank::new(1, 0);
        let workers = vec![
            WorkerInput {
                worker: dup,
                capacity: 4,
            },
            WorkerInput {
                worker: dup,
                capacity: 4,
            },
        ];
        let loras = vec![make_lora("A", 1)];
        let result = solver.solve(&workers, &loras, &HashMap::new(), None, None);
        assert!(
            result.is_err_and(|e| e.contains("duplicate worker")),
            "duplicate workers must be rejected"
        );
    }

    #[test]
    fn test_duplicate_lora_rejected() {
        let solver = McfPlacementSolver::new(McfSolveParams::default());
        let workers = make_workers(2, 4);
        let loras = vec![make_lora("A", 1), make_lora("A", 2)];
        let result = solver.solve(&workers, &loras, &HashMap::new(), None, None);
        assert!(
            result.is_err_and(|e| e.contains("duplicate LoRA")),
            "duplicate LoRA names must be rejected"
        );
    }

    #[test]
    fn test_extreme_churn_weight_still_prefers_real_worker() {
        // A churn_weight large enough that gamma_load * weight would overflow
        // i64 or exceed overflow_cost must NOT cause the LoRA to overflow when
        // a real worker is free: build_edge_cost clamps real edge cost below
        // overflow_cost and uses saturating arithmetic.
        let solver = McfPlacementSolver::new(McfSolveParams::default());
        let workers = make_workers(2, 4);
        let loras = vec![LoraInput {
            name: "A".to_string(),
            replicas: 1,
            churn_weight: i64::MAX,
        }];
        let result = solver
            .solve(&workers, &loras, &HashMap::new(), None, None)
            .expect("solve must not panic on extreme churn_weight");
        assert_eq!(
            result.overflow_count, 0,
            "a free real worker must always beat overflow regardless of churn_weight"
        );
        assert_eq!(result.assignment["A"].len(), 1);
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
    fn test_empty_loras_produces_unloads() {
        // Regression: loras=[] with a non-empty prev_assignment used to return
        // empty unloads. Now it must emit unloads for every live prior placement.
        let solver = McfPlacementSolver::new(McfSolveParams::default());
        let workers = make_workers(2, 4);
        let w0 = WorkerWithDpRank::new(0, 0);
        let w1 = WorkerWithDpRank::new(1, 0);

        let mut prev = HashMap::new();
        prev.insert("A".to_string(), HashSet::from([w0]));
        prev.insert("B".to_string(), HashSet::from([w1]));

        let result = solver
            .solve(&workers, &[], &prev, None, None)
            .expect("empty loras should succeed");

        assert!(result.assignment.is_empty());
        assert_eq!(result.overflow_count, 0);
        // Both prior placements must surface as unloads.
        assert!(
            result
                .unloads
                .get(&w0)
                .map(|s| s.contains("A"))
                .unwrap_or(false),
            "A should be unloaded from w0"
        );
        assert!(
            result
                .unloads
                .get(&w1)
                .map(|s| s.contains("B"))
                .unwrap_or(false),
            "B should be unloaded from w1"
        );
    }

    #[test]
    fn test_omitted_lora_produces_unload() {
        // A LoRA present in prev_assignment but absent from the loras input
        // must still generate an unload for every live worker it was on.
        let solver = McfPlacementSolver::new(McfSolveParams::default());
        let workers = make_workers(2, 4);
        let w0 = WorkerWithDpRank::new(0, 0);

        let mut prev = HashMap::new();
        prev.insert("A".to_string(), HashSet::from([w0]));
        prev.insert("B".to_string(), HashSet::from([w0]));

        // Only pass LoRA A; B is omitted entirely (simulates a stale removal).
        let loras = vec![make_lora("A", 1)];
        let result = solver
            .solve(&workers, &loras, &prev, None, None)
            .expect("solve should succeed");

        assert!(
            result
                .unloads
                .get(&w0)
                .map(|s| s.contains("B"))
                .unwrap_or(false),
            "omitted LoRA B must be unloaded from w0"
        );
        assert!(
            !result
                .unloads
                .get(&w0)
                .map(|s| s.contains("A"))
                .unwrap_or(false),
            "LoRA A should not be unloaded (it is still desired)"
        );
    }

    #[test]
    fn test_frozen_over_capacity_worker_gets_rebalanced() {
        // Regression: if a worker's capacity shrinks between ticks and frozen
        // assignments exceed the new capacity, saturating_sub used to hide the
        // violation. Now the over-committed LoRAs are unfrozen and re-solved.
        let solver = McfPlacementSolver::new(McfSolveParams::default());

        // First tick: 1 worker with capacity 2, two LoRAs placed on it.
        let workers_v1 = make_workers(1, 2);
        let loras = vec![make_lora("A", 1), make_lora("B", 1)];
        let r1 = solver
            .solve(&workers_v1, &loras, &HashMap::new(), None, None)
            .unwrap();
        assert_eq!(r1.overflow_count, 0);

        // Second tick: same worker, capacity reduced to 1.
        // Neither LoRA is in changed_loras, but the worker capacity dropped.
        let workers_v2 = vec![WorkerInput {
            worker: WorkerWithDpRank::new(0, 0),
            capacity: 1,
        }];
        let r2 = solver
            .solve(&workers_v2, &loras, &r1.assignment, None, None)
            .unwrap();

        // Worker can hold at most 1 LoRA; any excess must overflow.
        let placed: usize = r2.assignment.values().map(|s| s.len()).sum();
        assert_eq!(
            placed + r2.overflow_count,
            2,
            "placed + overflow must equal total demand"
        );
        // Worker must not be over-assigned.
        let w0 = WorkerWithDpRank::new(0, 0);
        let on_w0 = r2
            .assignment
            .values()
            .filter(|hosts| hosts.contains(&w0))
            .count();
        assert!(on_w0 <= 1, "worker capacity=1 must not be exceeded");
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
