<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adaptive Routing Evaluation

## Decision

Start with the plugin-only AIMD controller for further testing. Keep sigmoid available for comparison. Neither is ready for promotion to Dynamo's default. The experiment supports adaptive scorer budgets, but does not support the claim that a request-level bandit will automatically find a good policy under every workload.

The implementation uses the existing picker and catalog contracts. No router-host, scheduler, dispatch, binding, or telemetry API changes are included. Both reactive algorithms are available in real Dynamo selection. Reward-based algorithms are evaluated only in the simulator because the current plugin contract lacks correlated outcome feedback.

## Algorithm Comparison

| Algorithm | Adaptation | Fit and decision |
|---|---|---|
| llm-d sigmoid budgets | Map imbalance and pressure to affinity/distribution weights | Implemented with Dynamo's current signals, bounded updates, and no async detector. Explainable but still needs calibrated load scales; decode-heavy regression in this experiment. [Pinned implementation](https://github.com/nirrozenbaum/gateway-api-inference-extension/blob/6c0f8d8496b56d68c30369520f3c4477adf8cd42/pkg/epp/scheduling/adaptive_configurator.go). |
| AIMD | Add distribution weight under skew; slowly reduce it after recovery | Implemented. Small state and hysteresis; strongest candidate across this matrix. This borrows a control-law shape, not a proof of routing stability or fairness from TCP. [Chiu and Jain](https://pages.cs.wisc.edu/~suman/courses/740/papers/chiu89isdn.pdf). |
| Discounted/sliding-window UCB | Forget old rewards and explore scorer-weight arms | Discounted version implemented in simulation with delayed credit. Handles shifting rewards, but exploration can destroy shared cache state. The paper's stochastic switching-reward assumptions do not establish safety in this stateful routing environment. [Garivier and Moulines](https://arxiv.org/abs/0805.3415). |
| Contextual bandits / LinUCB | Condition arm reward on request and pool features | Implemented a small bucketed discounted-UCB experiment, not LinUCB. True LinUCB would learn a linear reward model from context; request length, reuse, and stage are plausible features, but the host still needs feedback. Context buckets alone did not prevent cache thrashing. [Li et al.](https://arxiv.org/abs/1003.0146). |
| Thompson sampling | Sample an uncertain reward model | Deferred. Attractive exploration behavior, but requires a latency/noise model, nonstationary forgetting, and the same feedback and cache-interference controls. [Russo et al.](https://arxiv.org/abs/1707.02038). |
| EXP3 / policy-gradient weight learning | Learn sampling probabilities or continuous preferences | Deferred. EXP3 addresses adversarial rewards, but unconstrained exploration and noisy importance-weighted updates do not solve shared-cache damage or outcome attribution. A continuous weight learner also needs an observable objective. [Auer et al.](https://www.microsoft.com/en-us/research/?p=579916). |

The source [llm-d design](https://docs.google.com/document/d/1NFfIz-BdUUOJrfErD2pfCizgncsWOYgFak8Z_HaP7e4/edit) explicitly describes reactive signal control and excludes ML-based/per-request adaptation. Its pressure branch reduces the distribution ceiling as pressure rises. This prototype preserves that direction, replaces detector inputs with an active-count proxy, normalizes the sigmoid endpoints, and adds AIMD as an alternative. It does not reproduce llm-d's published benchmark results.

## Method

Run `phase_shift` and `summarize.py` using the [validation commands](README.md#validation). The final matrix routes **2,400,000 simulated requests**: six workloads, eight policies, five seeds, and 10,000 requests per run. Every policy sees identical exogenous arrivals for a given workload and seed. Each run owns fresh worker queues, cache state, RNG, and controller state; state persists across its five traffic phases. Parameters are unchanged across workloads.

- Eight FIFO workers, each with a 32-prefix LRU cache. Cache becomes visible at first token, not when a queued request is selected. A queued job's service cost is calculated when it starts, so earlier prefill completions can benefit later jobs.
- Exponential interarrival times; successive 2,000-request phases use 8, 55, 30, 55, and 8 requests/s. The overloaded phases deliberately exceed capacity in the cold and decode-heavy cases. All queues are drained; nothing is silently dropped or excluded.
- Uncached prefill: `4 ms + input_tokens / 30,000`; a cache hit avoids 90% of token work. Decode: `output_tokens / 400` seconds. These are declared synthetic constants, not measured GPU rates.
- Normal traffic uses 4,096 input and 16 output tokens. Decode-heavy uses 256/128. Cache-working-set uses 16,384/16 and 256 prefixes, initially partitioned exactly across the eight caches. Other reuse workloads start with 64 prefixes partitioned across workers; burst phases concentrate 85% of requests on one prefix. Mixed traffic replaces one third with unique prefixes. Heterogeneous workers run two workers at 0.4x speed.
- All fixed and learned weights share the plugin's exact scoring function and tie-breaker. Static controls are weights 0.1, 0.5, 0.9, and least-load (1.0). **They are not Dynamo's built-in selector.** Dynamo host behavior is separately covered by integration tests.
- TTFT is queue wait plus modeled prefill. Report full-cohort mean and P99, completions divided by total time through final drain, cache-hit fraction, and the fraction meeting a 500 ms TTFT SLO. A phase is attributed by request arrival, so late completions stay in their original cohort.
- UCB arms are weights `[0.1, 0.3, 0.5, 0.7, 0.9]`; discount is 0.999 per decision and exploration coefficient is 0.4. Reward is `1 / (1 + TTFT / 0.5s)`, delivered at first token. Maximizing this bounded reward differs from minimizing arithmetic mean TTFT; the results apply to this reward and exploration configuration. Pending pulls limit repeated cold-arm exploration. Delayed rewards use the original arm/context and are discounted by decision age. Contextual UCB uses four buckets: any cache hit versus none, crossed with pool active count above versus below 64.

The simulator does not model continuous batching, GPU compute interference, memory capacity in tokens, prefix trees, preemption, transport, stale metrics, multi-router synchronization, failures, or request cancellation. In particular, FIFO workers are a strong simplification for decode-heavy traffic. These results are falsification tests of policy behavior, not serving-performance predictions or statistical evidence of a production win.

## Results

Measured on 2026-09-09. TTFT values below are arithmetic means across five complete seeds. Deltas are means of paired per-seed percentages against static 0.5, so they need not equal a ratio of the displayed means. Ranges show seed variability, not confidence intervals.

| Workload | Static 0.5 mean TTFT | AIMD mean TTFT | Paired TTFT delta | Per-seed delta range | AIMD throughput delta |
|---|---:|---:|---:|---:|---:|
| Hotspot shift | 0.073 s | 0.027 s | -50.0% | -86.1% to -31.3% | +0.00% |
| Mixed warm/cold | 0.156 s | 0.094 s | -37.5% | -60.8% to -21.3% | +0.00% |
| Heterogeneous | 0.131 s | 0.047 s | -50.1% | -88.0% to -33.8% | +0.00% |
| Decode-heavy | 40.383 s | 38.347 s | -5.1% | -6.3% to -3.9% | +0.01% |
| Cold burst | 2.473 s | 2.473 s | 0.0% | 0.0% to 0.0% | 0.00% |
| Cache working set | 0.108 s | 0.108 s | 0.0% | 0.0% to 0.0% | 0.00% |

Material counterexamples:

- **Sigmoid is not a universal improvement.** Decode-heavy mean TTFT rises to 57.527 s, a paired **42.4% regression**, with throughput down **2.49%** against static 0.5. AIMD is therefore the example default.
- **A good static setting can still beat adaptation.** Static 0.9 reaches 0.020 s on hotspot traffic and 0.076 s on mixed traffic, versus AIMD's 0.027 s and 0.094 s. Adaptive does not dominate the best configuration chosen separately for each workload.
- **Blind distribution destroys the working set in this model.** Static 0.9 drops cache hits from 100% to 41.5% and raises mean TTFT from 0.108 s to 102.877 s. Both controllers keep the exact initial cache partition and match static 0.5 on this workload.
- **The tested bandits also fail the working-set case.** UCB and contextual UCB reach 48.583 s and 54.504 s, with about 72.8% cache hits. Elsewhere they often approach or beat static 0.9. Reward feedback by itself is insufficient: one arm's cache pollution changes the future rewards of every arm. These specific results do not rule out conservative, epoch-based, or model-based learning.

## Selection Overhead

The release microbenchmark measures the shared controller/scoring code with preallocated candidate inputs, excluding the Dynamo host, input materialization, `Instant::elapsed()`, tracing, and networking. An initial profile by timing showed variance computation dominating between-update selections; gating variance to control ticks reduced the 4,096-worker path from about 24 microseconds to 6.6 microseconds. Hardware-counter profiling was unavailable on the test host. The table reports medians of five repetitions; the command also emits minimum and maximum times.

| Workers | Fixed-weight scoring | AIMD between ticks | AIMD control tick | Sigmoid control tick |
|---|---:|---:|---:|---:|
| 8 | 20.5 ns | 19.2 ns | 66.8 ns | 83.5 ns |
| 64 | 116 ns | 111 ns | 376 ns | 388 ns |
| 1,024 | 1.64 us | 1.59 us | 5.62 us | 5.58 us |
| 4,096 | 6.79 us | 6.59 us | 23.31 us | 22.45 us |

Microbenchmarks have scheduling and frequency noise; the small between-tick differences are not speedup claims. The expensive update occurs at most once per configured interval per policy instance. End-to-end frontend overhead still needs measurement.

## What Would Justify a Host Change

A production reward learner needs a generic feedback contract, not a router-host implementation of a particular algorithm:

1. Associate a policy decision token with a successfully dispatched attempt, including model/group/role and selected worker/rank. Selection can be advisory, pinned, retried, or fail before dispatch; a picker call is not proof of an executed arm.
2. Return correlated TTFT, terminal success/error/cancellation, and token counts through a bounded, nonblocking path. Learn errors as outcomes; do not silently train only on fast successes. Distinguish stage-local measurements in disaggregated serving.
3. Preserve the arm and context from decision time. Bound pending decisions, expire missing feedback, reject duplicates, and drop late observations from an older policy generation.
4. Evaluate epoch-level scorer-weight arms with guarded exploration and cache warm-up/settling periods. Log arm probabilities, cache effects, latency, and SLO goodput so a regression can be attributed and rolled back.

None of those hooks were added here. The signal controller can be deployed through today's catalog. Before promoting it, run paired GPU tests against both Dynamo default and a tuned static policy on real request traces, including mixed lengths, constrained pools, stale metrics, prefill/decode stages, and cache eviction. The present implementation and measurements do not establish that those checks pass.
