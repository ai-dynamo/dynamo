// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic Min-Cost Flow Solver
//!
//! Implements the Successive Shortest Augmenting Paths (SSAP) algorithm with
//! Johnson's potentials (Dijkstra on reduced costs). All capacities and costs
//! are integers, guaranteeing integral optimal solutions.
//!
//! Reference: the Python SSAP implementation in the LoRA allocation design doc.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Infinite cost sentinel (must be larger than any real cost in the graph).
pub const INF_COST: i64 = 1_000_000_000_000_000_000; // 10^18

/// Error returned when the solver cannot send the requested flow.
#[derive(Debug, Clone)]
pub enum McfError {
    /// Could not send the full requested flow.
    InsufficientFlow { sent: i64, requested: i64 },
}

impl std::fmt::Display for McfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McfError::InsufficientFlow { sent, requested } => {
                write!(
                    f,
                    "Could not send full flow: sent={sent} requested={requested}"
                )
            }
        }
    }
}

impl std::error::Error for McfError {}

/// A directed edge in the residual graph.
#[derive(Debug, Clone)]
struct Edge {
    to: usize,
    rev: usize,
    cap: i64,
    cost: i64,
}

/// Min-cost flow graph using adjacency lists.
///
/// Nodes are numbered `0..n`. The caller is responsible for assigning node
/// indices (source, sink, LoRA nodes, worker nodes, etc.).
pub struct MinCostFlowGraph {
    n: usize,
    graph: Vec<Vec<Edge>>,
}

impl MinCostFlowGraph {
    /// Create a graph with `n` nodes (indexed `0..n`).
    pub fn new(n: usize) -> Self {
        Self {
            n,
            graph: vec![Vec::new(); n],
        }
    }

    /// Add a directed edge from `from` to `to` with the given capacity and cost.
    /// Also adds the reverse edge with capacity 0 and negative cost.
    pub fn add_edge(&mut self, from: usize, to: usize, cap: i64, cost: i64) {
        let rev_from = self.graph[to].len();
        let rev_to = self.graph[from].len();
        self.graph[from].push(Edge {
            to,
            rev: rev_from,
            cap,
            cost,
        });
        self.graph[to].push(Edge {
            to: from,
            rev: rev_to,
            cap: 0,
            cost: -cost,
        });
    }

    /// Compute the min-cost flow of at most `max_flow` units from `s` to `t`.
    ///
    /// Returns `(flow_sent, min_cost)`.
    /// Returns `Err` if unable to send `max_flow` units (the partial flow sent
    /// is included in the error).
    pub fn min_cost_flow(
        &mut self,
        s: usize,
        t: usize,
        max_flow: i64,
    ) -> Result<(i64, i64), McfError> {
        let n = self.n;
        let mut pot = vec![0i64; n];
        let mut prevv = vec![0usize; n];
        let mut preve = vec![0usize; n];

        let mut flow: i64 = 0;
        let mut cost: i64 = 0;

        while flow < max_flow {
            // Dijkstra on reduced costs
            let mut dist = vec![INF_COST; n];
            dist[s] = 0;
            let mut pq: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
            pq.push(Reverse((0, s)));

            while let Some(Reverse((d, v))) = pq.pop() {
                if d != dist[v] {
                    continue;
                }
                for (i, e) in self.graph[v].iter().enumerate() {
                    if e.cap <= 0 {
                        continue;
                    }
                    let nd = d + e.cost + pot[v] - pot[e.to];
                    if nd < dist[e.to] {
                        dist[e.to] = nd;
                        prevv[e.to] = v;
                        preve[e.to] = i;
                        pq.push(Reverse((nd, e.to)));
                    }
                }
            }

            if dist[t] >= INF_COST {
                break; // No augmenting path
            }

            // Update potentials
            for v in 0..n {
                if dist[v] < INF_COST {
                    pot[v] += dist[v];
                }
            }

            // Find bottleneck capacity along the path
            let mut add_flow = max_flow - flow;
            let mut v = t;
            while v != s {
                let e = &self.graph[prevv[v]][preve[v]];
                add_flow = add_flow.min(e.cap);
                v = prevv[v];
            }

            // Augment flow along the path
            v = t;
            while v != s {
                let pv = prevv[v];
                let pe = preve[v];
                self.graph[pv][pe].cap -= add_flow;
                let rev = self.graph[pv][pe].rev;
                self.graph[v][rev].cap += add_flow;
                v = pv;
            }

            flow += add_flow;
            cost += add_flow * pot[t];
        }

        if flow < max_flow {
            Err(McfError::InsufficientFlow {
                sent: flow,
                requested: max_flow,
            })
        } else {
            Ok((flow, cost))
        }
    }

    /// Read the flow on a forward edge. The forward edge from `from` at index
    /// `edge_idx` has residual capacity `graph[from][edge_idx].cap`; the
    /// original capacity minus residual gives the flow.
    ///
    /// Since we only need to check whether flow passed through an edge (cap=1),
    /// a simpler approach: check if the reverse edge has cap > 0.
    pub fn flow_on_edge(&self, from: usize, edge_idx: usize) -> i64 {
        let e = &self.graph[from][edge_idx];
        let rev_e = &self.graph[e.to][e.rev];
        rev_e.cap
    }

    /// Number of edges (forward only) from a node.
    pub fn edge_count(&self, node: usize) -> usize {
        self.graph[node].len()
    }

    /// Get the target node of edge `edge_idx` from `node`.
    pub fn edge_to(&self, node: usize, edge_idx: usize) -> usize {
        self.graph[node][edge_idx].to
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_flow() {
        // S(0) --cap=2,cost=1--> T(1)
        let mut g = MinCostFlowGraph::new(2);
        g.add_edge(0, 1, 2, 1);
        let (flow, cost) = g.min_cost_flow(0, 1, 2).unwrap();
        assert_eq!(flow, 2);
        assert_eq!(cost, 2); // 2 units * cost 1
    }

    #[test]
    fn test_multi_path_chooses_cheapest() {
        // S(0) --cap=1,cost=10--> T(2)  via node 1 (cheap)
        // S(0) --cap=1,cost=100-> T(2)  via node 3 (expensive)
        let mut g = MinCostFlowGraph::new(4);
        g.add_edge(0, 1, 1, 10); // S -> A cheap
        g.add_edge(1, 3, 1, 0); // A -> T
        g.add_edge(0, 2, 1, 100); // S -> B expensive
        g.add_edge(2, 3, 1, 0); // B -> T

        let (flow, cost) = g.min_cost_flow(0, 3, 1).unwrap();
        assert_eq!(flow, 1);
        assert_eq!(cost, 10); // Uses the cheap path
    }

    #[test]
    fn test_multi_path_both_needed() {
        // Need 2 units of flow, cheap + expensive paths
        let mut g = MinCostFlowGraph::new(4);
        g.add_edge(0, 1, 1, 10);
        g.add_edge(1, 3, 1, 0);
        g.add_edge(0, 2, 1, 100);
        g.add_edge(2, 3, 1, 0);

        let (flow, cost) = g.min_cost_flow(0, 3, 2).unwrap();
        assert_eq!(flow, 2);
        assert_eq!(cost, 110); // 10 + 100
    }

    #[test]
    fn test_insufficient_flow() {
        let mut g = MinCostFlowGraph::new(2);
        g.add_edge(0, 1, 1, 1);
        let result = g.min_cost_flow(0, 1, 5);
        assert!(result.is_err());
        match result {
            Err(McfError::InsufficientFlow { sent, requested }) => {
                assert_eq!(sent, 1);
                assert_eq!(requested, 5);
            }
            _ => panic!("expected InsufficientFlow"),
        }
    }

    #[test]
    fn test_zero_flow() {
        let mut g = MinCostFlowGraph::new(2);
        g.add_edge(0, 1, 10, 5);
        let (flow, cost) = g.min_cost_flow(0, 1, 0).unwrap();
        assert_eq!(flow, 0);
        assert_eq!(cost, 0);
    }

    #[test]
    fn test_bipartite_assignment() {
        // Classic assignment: 2 LoRAs, 2 workers
        // S=0, L0=1, L1=2, W0=3, W1=4, T=5
        let mut g = MinCostFlowGraph::new(6);
        // S -> LoRAs (cap = replica count)
        g.add_edge(0, 1, 1, 0); // S -> L0 (1 replica)
        g.add_edge(0, 2, 1, 0); // S -> L1 (1 replica)
        // LoRA -> Worker edges (cap=1, with costs)
        g.add_edge(1, 3, 1, 5); // L0 -> W0 cost 5
        g.add_edge(1, 4, 1, 10); // L0 -> W1 cost 10
        g.add_edge(2, 3, 1, 10); // L1 -> W0 cost 10
        g.add_edge(2, 4, 1, 5); // L1 -> W1 cost 5
        // Workers -> T (cap = worker capacity)
        g.add_edge(3, 5, 2, 0); // W0 -> T cap 2
        g.add_edge(4, 5, 2, 0); // W1 -> T cap 2

        let (flow, cost) = g.min_cost_flow(0, 5, 2).unwrap();
        assert_eq!(flow, 2);
        assert_eq!(cost, 10); // L0->W0(5) + L1->W1(5) = 10 (optimal)
    }

    #[test]
    fn test_flow_on_edge() {
        let mut g = MinCostFlowGraph::new(3);
        g.add_edge(0, 1, 1, 5); // edge_idx 0 from node 0
        g.add_edge(0, 2, 1, 10); // edge_idx 1 from node 0 (fwd only)
        g.add_edge(1, 2, 1, 0);
        g.add_edge(2, 2, 0, 0); // dummy to make sink
        // Actually let's make a proper source->sink
        let mut g = MinCostFlowGraph::new(4);
        g.add_edge(0, 1, 1, 5); // edge 0 from node 0: S->A cost 5
        g.add_edge(0, 2, 1, 10); // edge 1 from node 0: S->B cost 10 (fwd; idx=2 because rev of first also on 0... no)
        // edges from node 0: [S->A, rev(A->S is on node1), S->B, rev(B->S is on node2)]
        // Actually edges from node 0 are just the forward edges we add: idx0 = S->A, idx1 = S->B
        // (reverse edges are added to nodes 1 and 2 respectively)
        g.add_edge(1, 3, 1, 0); // A -> T
        g.add_edge(2, 3, 1, 0); // B -> T

        let (flow, _cost) = g.min_cost_flow(0, 3, 1).unwrap();
        assert_eq!(flow, 1);

        // The cheap path S->A->T should be used
        assert_eq!(g.flow_on_edge(0, 0), 1); // S->A has flow
        assert_eq!(g.flow_on_edge(0, 1), 0); // S->B has no flow
    }

    #[test]
    fn test_negative_costs_keep_reward() {
        // Simulate keep-reward: placing L0 on W0 has negative cost (reward)
        // S=0, L0=1, W0=2, W1=3, T=4
        let mut g = MinCostFlowGraph::new(5);
        g.add_edge(0, 1, 1, 0); // S -> L0
        g.add_edge(1, 2, 1, -100); // L0 -> W0 (keep reward: -100)
        g.add_edge(1, 3, 1, 50); // L0 -> W1 (new load penalty: +50)
        g.add_edge(2, 4, 1, 0); // W0 -> T
        g.add_edge(3, 4, 1, 0); // W1 -> T

        let (flow, cost) = g.min_cost_flow(0, 4, 1).unwrap();
        assert_eq!(flow, 1);
        assert_eq!(cost, -100); // Should pick the keep path
    }
}
