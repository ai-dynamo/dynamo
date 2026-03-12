---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Optimal LoRA Placement: From Adapter Churn to Global Optimization"
subtitle: "Biswa Ranjan Panda, Anish Modipatti, Vikram Sharma Mailthody, March 2026"
description: "Dynamo's LoRA placement controller evolved through four stages (from naive scattering, through rendezvous hashing, to min-cost flow optimization), reducing adapter churn by 99.7% and achieving 95% churn-free ticks across realistic traffic patterns."
keywords: LoRA, adapter serving, placement optimization, min-cost flow, rendezvous hashing, LLM inference, Dynamo, GPU cluster, adapter churn
last-updated: March 12, 2026
---

Dynamo's **LoRA placement controller** dynamically assigns adapter replicas across a GPU cluster, reducing adapter churn by **99.7% compared to naive placement** and achieving **95% churn-free ticks** under realistic traffic. It evolved through four stages: from naive scattering that produces ~35,000 swap operations, through rendezvous hashing that cuts that to ~250, to a global min-cost flow optimizer that brings it down to **~98**. Each stage exposed a new bottleneck and a specific algorithm to break through it.

We're shipping both rendezvous hashing and min-cost flow as first-class placement algorithms in Dynamo.

This post walks through that engineering journey, from 35,000 churn operations to 98.

---

## 1. The Adapter Churn Problem

### 1.1 LoRA Serving at Scale

[LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) lets you fine-tune a base model with a small set of weight matrices (typically 0.1-1% of the base model's parameters). A single GPU cluster can serve hundreds of these adapters simultaneously, each specializing the base model for a different task: code generation, medical Q&A, legal summarization, customer support in different languages.

Each GPU worker has a limited number of **adapter slots**, determined by its available VRAM. A worker with 4 slots can hold four adapters in GPU memory and serve them concurrently. When a request arrives for an adapter that isn't loaded, the worker must evict one and load the new one, known as an **adapter swap**.

### 1.2 What Adapter Churn Costs

Each swap transfers adapter weights over the GPU memory bus. Even for a rank-16 LoRA on a small 7B model, that's roughly 20-50 MB per direction. At scale, the costs compound:

- **GPU memory bandwidth** consumed by weight transfers instead of inference
- **Tail latency** spikes when a request triggers a cold load
- **Cascading evictions** when a burst of new adapters displaces warm ones

With hundreds of adapters and shifting traffic patterns, the swap rate itself becomes the bottleneck. We call this **adapter churn**: the total number of load and unload operations per control cycle.

### 1.3 Why Naive Approaches Fail

| Approach | Strategy | Why it fails |
|----------|----------|-------------|
| **Stateless scattering** | Random or round-robin | Spreads each adapter's traffic across all workers, maximizing churn |
| **Static pinning** | Pre-assign adapters to workers | Cannot react to load shifts; hot adapters overwhelm pinned workers |

Scattering is the worst case: every tick reshuffles placements, producing ~35,000 churn operations over 200 ticks in our simulations. Static pinning avoids churn but creates hotspots. We need something that adapts to traffic while minimizing movement.

---

## 2. Architecture: A Closed-Loop Controller

Before diving into placement algorithms, here's the system they plug into. The architecture cleanly separates a periodic **control plane** from a per-request **data plane**, connected through shared concurrent state.

<Frame caption="Figure 1: System Architecture">
  <img src="./images/fig-1-architecture-overview.svg" alt="System Architecture" style={{width: "100%"}} />
</Frame>

<Info icon="hand-point-up" className="fig-caption">
The control plane (Controller, Load Estimator, Allocator) runs periodically, computing replica placements. The data plane (LoRA Filter, KV Router) runs per-request, narrowing candidates to the designated replica set. Shared state (Routing Table, State Tracker) bridges the two via thread-safe concurrent maps.
</Info>

**The feedback loop:**

<Frame caption="Figure 2: Control Loop">
  <img src="./images/fig-2-control-loop.svg" alt="Control Loop" style={{width: "100%"}} />
</Frame>

<Info icon="hand-point-up" className="fig-caption">

Request traffic feeds the load estimator, flows through the controller and routing table to the LoRA filter, and reaches the worker. When the response stream completes, the load counter is decremented automatically, closing the loop.
</Info>

Request traffic feeds the load estimator, which maintains a windowed arrival-rate counter per adapter. The controller reads load estimates and cluster topology, computes allocations, and writes to the routing table. The LoRA filter reads the routing table on each request to narrow worker candidates. When the response stream completes, the load counter is decremented automatically, closing the loop.

**Two-stage routing:** Stage 1 (LoRA filter) narrows candidates to the designated replica set. Stage 2 (existing KV-cache-aware router or round-robin) selects the final worker from that narrowed set. The placement controller is transparent: it only constrains *which workers* are eligible, not *how* the final selection happens.

**Proportional allocation:** The controller computes per-adapter replica counts proportional to load fractions. An adapter receiving 60% of traffic gets more replicas than one receiving 5%. The [largest remainder method](https://en.wikipedia.org/wiki/Largest_remainder_method) distributes the total slot budget (N workers x K slots each) across adapters, ensuring every known adapter gets at least one replica.

**Pluggable algorithms:** The placement algorithm is a runtime choice, not a compile-time decision. You can switch between rendezvous hashing, random placement, or min-cost flow with a single configuration change. All three share the same feedback loop, shared state, and two-stage routing; only the placement logic differs. No need to change any other configuration or redeploy workers.

**Adapter lifecycle:** Each LoRA adapter moves through a simple set of states. When an operator loads an adapter onto a worker, it registers with the discovery service. The controller picks it up on the next tick and creates an inactive routing entry (cold-start pin). When the first request arrives, it transitions to active with proportional replicas. When traffic drops to zero (after a hysteresis cooldown), it returns to inactive. The router never loads or unloads adapters itself; workers handle lazy loading, and operators manage the lifecycle.

<Frame caption="Figure 3: LoRA Adapter Lifecycle">
  <img src="./images/fig-9-lifecycle.svg" alt="LoRA Adapter Lifecycle" style={{width: "100%"}} />
</Frame>

<Info icon="hand-point-up" className="fig-caption">
Each adapter transitions through three states: Loaded (in VRAM), Inactive (cold-start pinned to one deterministic worker), and Active (proportional replicas across multiple workers). The router only routes; operators manage the adapter lifecycle via load/unload calls.
</Info>

---

## 3. Stage 1: Random Placement (The Baseline)

The simplest placement: assign each adapter's replicas to randomly selected workers every tick. This establishes the performance floor.

The problem is immediate. Every tick re-rolls placements. An adapter on workers {W1, W3, W5} this tick lands on {W2, W4, W6} next tick, resulting in six swap operations for a single adapter, even though nothing about the traffic changed.

Over 200 simulated ticks with 100 adapters on an 8-worker cluster (4 slots each): **~35,000 total churn operations**. Zero ticks with zero churn. The cluster never rests.

Random placement tells us the shape of the problem. The baseline is spectacularly bad, which means there's a lot of room for a smarter algorithm.

---

## 4. Stage 2: Rendezvous Hashing (Deterministic Placement)

The first real improvement: replace randomness with **Rendezvous Hashing (HRW)**. For each (adapter, worker) pair, the algorithm computes a deterministic score using a cryptographic hash. Workers are ranked by score; the highest-ranked ones form the replica set. The ranking is deterministic: same inputs always produce the same output, on every router instance, with no coordination.

| Property | Guarantee |
|----------|-----------|
| Determinism | Same (adapter, worker set) produces the same ranking everywhere |
| Stability | Adding or removing a worker changes only a small fraction of placements |
| Uniformity | Cryptographic hashing gives near-uniform score distribution |
| No coordination | Any router instance computes identical placements independently |

**Slot-aware placement:** The allocator walks the ranked list, skipping workers that are already at capacity. If the result set is under-filled (all workers full), it falls back to basic rendezvous hashing rather than under-scheduling.

**Cold-start pinning:** When an adapter has zero traffic, the controller maintains an inactive entry pointing to its top-ranked worker. When the first request arrives at multiple router instances simultaneously, they all converge on the same worker, preventing thundering herd with no coordination needed.

### 4.1 Four Layers of Churn Minimization

Deterministic ranking alone isn't enough. Four complementary mechanisms stack to minimize churn:

1. **Hashing stability.** Rendezvous hashing guarantees that adding or removing one worker disturbs only a small fraction of existing placements. The vast majority stay put.

2. **Hysteresis.** Scale-down is deferred by a cooldown period (default: 3 ticks, or 9 seconds). A transient traffic dip doesn't trigger an immediate replica reduction. Scale-up is always immediate.

3. **Change detection.** Before writing to the routing table, the controller compares the new replica set against the current one. If nothing changed (same workers, same count, same active flag), the write is skipped entirely.

4. **Transition continuity.** The top-ranked worker is always the first element of any active replica set. When an adapter transitions from inactive (1 replica, cold-start pin) to active (R replicas), the original pinned worker is preserved, and only new workers are added. The reverse transition only removes the extras.

When min-cost flow is used, it adds a **fifth layer** (global optimization with keep-rewards) that replaces layers 1 and 4 with a single global solver. Layers 2 and 3 remain shared by both algorithms.

<Frame caption="Figure 4: Five-Layer Churn Minimization Strategy">
  <img src="./images/fig-8-churn-layers.svg" alt="Five-Layer Churn Minimization Strategy" style={{width: "100%"}} />
</Frame>

<Info icon="hand-point-up" className="fig-caption">
The first three layers are shared by both rendezvous hashing and min-cost flow. The path diverges at layer 3: rendezvous hashing uses transition continuity (layer 4), while min-cost flow replaces it with global optimization using keep-rewards and delta solving (layer 5). The result: min-cost flow achieves 51-65% less churn than rendezvous hashing's four-layer strategy.
</Info>

### 4.2 Where Rendezvous Hashing Plateaus

Rendezvous hashing with four churn-minimization layers brings total churn down to **~250 operations** over 200 ticks (from Random's ~35,000, a 99.3% reduction) and achieves **61-74% churn-free ticks**. A huge improvement, but not optimal.

The fundamental limitation: rendezvous hashing is a **per-adapter heuristic**. Each adapter's placement is computed independently. When adapter A's replica set shifts from {W1, W3} to {W1, W5}, the algorithm doesn't know that adapter B was about to vacate W5 anyway. Per-adapter decisions cannot see the global picture, missing opportunities to coordinate moves and cancel out churn.

---

## 5. Stage 3: Min-Cost Flow (The Global Optimizer)

The insight: placement is a **bipartite matching problem**. We have L adapters on one side, N workers on the other, and we want an assignment that minimizes total movement cost. This is exactly what min-cost flow solves.

### 5.1 Bipartite Flow Formulation

The placement problem is modelled as a flow network with four layers:

<Frame caption="Figure 5: Bipartite Flow Network">
  <img src="./images/fig-3-mcf-bipartite.svg" alt="MCF Bipartite Flow Network" style={{width: "100%"}} />
</Frame>

<Info icon="hand-point-up" className="fig-caption">
Each adapter node receives flow equal to its replica count from the source. Each worker node drains flow equal to its slot capacity to the sink. Edge costs encode placement preferences: lower cost for keeping existing placements, higher cost for new assignments. An overflow node absorbs excess demand when the cluster is oversubscribed.
</Info>

- **Source to Adapter**: capacity equals the adapter's replica count. Each adapter must place exactly that many replicas.
- **Adapter to Worker**: capacity of 1 per edge, with a placement cost (see below). At most one replica of each adapter per worker.
- **Worker to Sink**: capacity equals the worker's slot count. Enforces the per-worker adapter limit.
- **Overflow (dummy)**: absorbs excess flow when total demand exceeds capacity, with a very high cost. The solver prefers real workers but never fails.

The solver finds a flow that places all replicas while **minimizing total cost** subject to capacity constraints.

### 5.2 The Cost Function

Each adapter-to-worker edge carries a cost that encodes three signals:

| Signal | What it does |
|--------|-------------|
| **HRW rank preference** | Favors workers that rendezvous hashing would naturally select. Provides a uniform tiebreaker when no prior assignment exists. |
| **New-load penalty** | Heavily penalizes placing an adapter onto a worker it wasn't on last tick. Discourages unnecessary migration. |
| **Keep-reward** | Rewards keeping an adapter on its current worker. Creates a strong incentive to preserve existing placements. |

The key insight: the keep-reward means the solver actively *prefers* the status quo. Only when the proportional allocation genuinely shifts (because traffic patterns changed) does the solver move adapters. And when it does, it moves the minimum number needed to satisfy the new allocation.

### 5.3 What Min-Cost Flow Achieves

With the same four-layer churn minimization stack (hysteresis, change detection, transition continuity) plus global optimization as a **fifth layer**, min-cost flow reduces total churn to **88-136 operations** over 200 ticks. Here's how that stacks up across the full journey:

| Comparison | Churn Reduction | What it means |
|-----------|----------------|---------------|
| MCF vs Random | **99.7%** | The full journey: 35,658 ops down to 98 |
| MCF vs HRW | **51-65%** | MCF's incremental gain over the already-good HRW |
| HRW vs Random | **99.3%** | Deterministic hashing alone is a massive win |

Min-cost flow also achieves **86-95% churn-free ticks** (the cluster is completely stable with zero GPU memory transfers for the vast majority of control cycles), and its worst-case per-tick churn is bounded at 32 operations (vs rendezvous hashing's 40, random's 226).

---

## 6. Stage 4: Making Min-Cost Flow Practical

A naive min-cost flow formulation connects every adapter to every worker. With 20,000 adapters and 200 workers, that's 4 million edges, too slow for a 3-second control loop. Two optimizations make it practical at scale.

### 6.1 Edge Sparsification

Instead of connecting every adapter to every worker, the solver only creates edges to a small **candidate set**: the top 16 workers from rendezvous hashing plus any workers that hosted the adapter last tick. This reduces the number of edges by roughly 12x for a 200-worker cluster. The guarantee: rendezvous hashing's top candidates are the "natural" placements, and including previous hosts ensures the keep-reward can take effect.

### 6.2 Delta Solving

Most ticks see small changes: a few adapters shift load, no topology changes. Solving the full problem every tick is wasteful.

**Delta solving** identifies which adapters are *impacted* (replica count changed, or prior hosts overlap with changed workers), freezes unaffected assignments in place, and solves only the residual problem:

1. **Identify impacted adapters**: those whose replica count changed or whose prior hosts overlap with changed workers
2. **Freeze unaffected**: carry their current assignments forward, consuming their slots from worker capacities
3. **Solve residual**: only impacted adapters participate in the flow network, with remaining worker capacities
4. **Merge**: combine frozen and solved assignments

During steady-state traffic, the residual problem is tiny (often zero impacted adapters), making the tick near-instantaneous. Only when traffic genuinely shifts does the solver engage with a meaningful problem.

---

## 7. Simulation Results

### 7.1 Test Configuration

All simulations use an **8-worker cluster with 4 slots each** (32 total slots), **100 LoRA adapters** drawn from a Zipf popularity distribution, running for **200 ticks** at 3-second intervals.

Four load patterns stress different aspects of the placement system:

| Scenario | Load Model | What it stresses |
|----------|-----------|------------------|
| **Zipf + Poisson** | Power-law popularity, independent Poisson arrivals | Skewed popularity; long-tail cold adapter flickering |
| **Daily traffic** | Sinusoidal day/night envelope over Zipf | Smooth ramp-up/down; gradual load shifts |
| **Traffic spikes** | Zipf baseline with 5x viral surge events | Sudden bursts; recovery stability |
| **Markov-modulated** | 3-state Markov chain (calm/busy/surge) | Bursty, correlated regime switches |

**Zipf + Poisson** models the steady-state reality of most LoRA deployments: a handful of popular adapters (code generation, general chat) dominate traffic while a long tail of specialized adapters (niche languages, domain-specific tasks) flicker in and out with low-frequency requests. The Zipf distribution controls popularity skew; Poisson arrivals make each tick's load stochastic.

**Daily traffic** adds a time-of-day dimension. Traffic follows a smooth sinusoidal curve: low at night, peaking at midday. This tests how gracefully the placement controller scales replicas up and down without over-reacting to gradual shifts.

**Traffic spikes** simulate viral events: a previously-cold adapter suddenly receives 5x normal load (think: a trending topic hitting a specialized model). Two spike events at different times test whether the controller can absorb a sudden surge and then stabilize quickly.

**Markov-modulated** traffic is the most challenging. The system randomly transitions between three regimes (calm, busy, surge), each with a different total load level. Unlike smooth daily cycles, these transitions are abrupt and unpredictable, testing how the controller handles step-changes in demand.

### 7.2 Total Churn

<Frame caption="Figure 6: Total Churn Across Load Patterns">
  <img src="./images/fig-4-total-churn.svg" alt="Total Churn Comparison" />
</Frame>

<Info icon="hand-point-up" className="fig-caption">
Total adapter churn (loads + unloads) over 200 simulated ticks. The full journey: Random (~35,000) to rendezvous hashing (~250) to min-cost flow (~98), a 99.7% end-to-end reduction. Min-cost flow provides an additional 51-65% reduction over the already-good rendezvous hashing. Note the log-scale axis.
</Info>

### 7.3 Cluster Stability

<Frame caption="Figure 7: Churn-Free Tick Ratio">
  <img src="./images/fig-5-churn-free-ratio.svg" alt="Churn-Free Tick Ratio" />
</Frame>

<Info icon="hand-point-up" className="fig-caption">
Percentage of ticks with zero adapter movement. Min-cost flow keeps the cluster completely stable 86-95% of the time. Rendezvous hashing manages 61-74%. Random never achieves a single churn-free tick.
</Info>

### 7.4 Spike Resilience

The Traffic Spikes scenario is the most revealing. At ticks 50 and 130, traffic surges to 5x baseline, activating dozens of previously-cold adapters simultaneously.

<Frame caption="Figure 8: Per-Tick Churn During Traffic Spikes">
  <img src="./images/fig-6-spike-timeline.svg" alt="Per-Tick Churn Timeline" style={{width: "100%"}} />
</Frame>

<Info icon="hand-point-up" className="fig-caption">
Per-tick churn for the Traffic Spikes scenario. Min-cost flow absorbs each spike with a single concentrated rebalance, then immediately returns to zero churn. Rendezvous hashing produces cascading adjustments that take many ticks to settle. Random churns continuously regardless of traffic.
</Info>

Min-cost flow handles the 5x viral spike with only **103 total operations** over 200 ticks. The spike triggers a single rebalance event: the solver places the newly-active adapters optimally in one shot, then returns to zero churn. Rendezvous hashing produces **238 operations** (2.3x more) as independent per-adapter re-evaluations cascade through overlapping replica sets.

### 7.5 Summary

<Frame caption="Figure 9: Min-Cost Flow vs Rendezvous Hashing Improvement">
  <img src="./images/fig-7-churn-efficiency.svg" alt="MCF vs HRW Improvement" />
</Frame>

<Info icon="hand-point-up" className="fig-caption">
Min-cost flow's incremental churn reduction over rendezvous hashing across all four load patterns. While the full journey from random to min-cost flow is a 99.7% reduction, this chart isolates min-cost flow's contribution: a consistent 51-65% improvement over rendezvous hashing across fundamentally different traffic dynamics.
</Info>

| Metric | Min-Cost Flow | Rendezvous Hashing | Random |
|--------|----:|----:|-------:|
| Best total churn (Markov-modulated) | 88 | 251 | 35,477 |
| Best churn-free ratio (Zipf) | 95% | 74% | 0% |
| Peak per-tick churn | 32 | 40 | 226 |
| Routing table additions | 40-53 | 108 | 100 |

Min-cost flow's routing table is more compact: it adds fewer adapters (40-53 vs 108) and removes fewer (12-40 vs 74-95), avoiding unnecessary cold-start pin churn that rendezvous hashing produces.

---

## 8. Choosing an Algorithm

The placement algorithm is selected at runtime with a single configuration change. All algorithms share the same architecture (the same control loop, shared state, two-stage routing, and load estimator). Only the placement logic changes.

| Algorithm | Best for | Tradeoff |
|-----------|----------|----------|
| **Rendezvous Hashing** | Most deployments | Simple, deterministic, 99.3% churn reduction vs random. No solver overhead. |
| **Min-Cost Flow** | Large clusters with many adapters | Additional 51-65% churn reduction over rendezvous hashing. Small solver cost per tick (mitigated by delta solving). |
| **Random** | Benchmarking only | Establishes the performance floor. Not recommended for production. |

The controller is transparent to workers. Workers do not know which algorithm is running; they simply receive requests for adapters they may or may not have loaded, and handle lazy loading on first request.

---

## 9. Future Directions

- **LoRA size awareness.** Weight allocation by adapter rank and VRAM footprint instead of uniform slots. The min-cost flow solver already supports per-adapter churn weights, a natural extension: set them proportional to adapter size so the solver preferentially avoids moving large adapters.

- **Predictive scaling.** Use time-series history to anticipate load shifts. Diurnal patterns could pre-warm adapters before the daily ramp-up, avoiding the cold-start penalty entirely.

- **Adaptive solver tuning.** Auto-tune the new-load penalty and keep-reward weights based on observed churn rates and measured GPU memory bandwidth costs, closing the loop between placement decisions and hardware impact.

---

## 10. Conclusion

The journey from random scattering to global optimization spans four stages, each motivated by a concrete limitation in the previous approach:

1. **Random Placement**: simple but catastrophic at ~35,000 churn operations and 0% stability.
2. **Rendezvous Hashing**: deterministic, coordination-free placement with four churn-minimization layers. Reduces churn by 99.3% vs random, but per-adapter heuristics miss global optimization opportunities.
3. **Min-Cost Flow**: global optimizer that sees all adapters and workers simultaneously. Cost function with keep-rewards and new-load penalties preserves existing placements by design. Reduces churn by 51-65% vs rendezvous hashing.
4. **Practical Min-Cost Flow**: edge sparsification and delta solving make global optimization viable in a 3-second control loop at scale.

The result: from ~35,000 churn operations down to ~98 (a **99.7% reduction**) with **95% churn-free ticks**. The GPU cluster spends its memory bandwidth on inference, not shuffling adapter weights.
