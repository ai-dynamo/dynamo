<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DEP: Pluggable Duru fairness policy with a ThunderAgent adapter

Area: router, with a separately reviewed vLLM adapter.

## Summary

Duru's philosophy is to measure how far each agent or multi-agent task has fallen
behind its fair share, then use that common signal to coordinate router admission
and runtime scheduling. This approach can be applied to different serving systems.

This DEP proposes an opt-in implementation on top of Dynamo's ThunderAgent router
and vLLM. ThunderAgent and vLLM adapters connect Duru’s fairness policy to the router
and engine schedulers.
ThunderAgent continues to own capacity checks, worker placement,
reservations, locking, pause/resume execution, and cancellation. Duru supplies
priority and admission-exception eligibility.

This proposal breaks it into small changes with independent tests and explicit
dependencies.

## Motivation

Agent requests alternate between generation and tool execution. Serving individual
requests in arrival order does not directly account for service received across
an agent's lifetime or across children of the same root. Duru uses accumulated
entitlement and accepted work to prioritize underserved roots.

ThunderAgent is the initial integration target because it already provides
program-level working-set accounting and admission control. Reusing those
mechanisms lets us add Duru's fairness decisions without maintaining a second
implementation of reservation and lifecycle management.

## Proposal

### General philosophy and this implementation

The accompanying paper, [*Towards Fair Resource Sharing in Agentic LLM Serving*](https://arxiv.org/abs/2611.23558v1),
defines normalized slowdown relative to ideal processor sharing (§3) and uses it
as a shared control signal across routing and runtime scheduling (§§5.1–5.5).
The goal is to reduce disproportionate agent-level delay while limiting throughput
loss.

**1. Slowdown: how much longer did the agent take under sharing?**

Compare the time an agent spends executing and waiting under sharing with the
inference time it would need alone:

$$
S = \frac{T_{\mathrm{shared}} + T_{\mathrm{queue}}}{T_{\mathrm{alone}}}.
$$

Here, $T_{\mathrm{shared}}$ is execution time under sharing and
$T_{\mathrm{queue}}$ is waiting time, following the paper's notation. Both
accumulate across the agent's LLM calls.

**2. Normalize for load: with a fixed number of agents, is that slowdown fair?**

Start with one worker and $N$ competing single-agent tasks, each with an
outstanding request. Under ideal equal sharing, each receives $1/N$ of the
worker, so its reference slowdown is $N$. Normalize by that reference:

$$
U = \frac{S}{N}.
$$

$U > 1$ identifies an agent that has fallen behind its fair share; $U < 1$ means it has received more.

**3. Let the number of competing agents change over time.**

Agents arrive, finish, and enter tool phases, so there is no single fixed $N$.
Instead, accumulate the agent's fair share at each instant when it has an
outstanding request:

$$
U = \frac{1}{T_{\mathrm{alone}}}
    \int_{\mathcal{I}} \frac{1}{N(t)}\,dt.
$$

Here, $\mathcal{I}$ contains the agent's waiting and execution intervals, and
$N(t)$ counts agents with outstanding requests at time $t$. When $N$ is
constant, $U$ reduces to $\frac{S}{N}$, recovering
the previous equation.

**4. Extend to multiple workers and multi-agent tasks.**

With $G(t)$ workers and $N(t)$ competing single-agent tasks, each agent's fair
share becomes $\min(1, G(t)/N(t))$, capped at one worker:

$$
U = \frac{1}{T_{\mathrm{alone}}}
    \int_{\mathcal{I}} \min\!\left(1, \frac{G(t)}{N(t)}\right)\,dt.
$$

To calculate this score while a task is running, use only the waiting and
execution intervals observed so far and the estimated standalone cost of
completed, accepted engine iterations. This requires no prediction of the task's
future requests or total work.

In the code, `E` accumulates $\int_{\mathcal{I}\cap[0,t]} \min(1, G(\tau)/N(\tau))\,d\tau$ and `W` estimates
$T_{\mathrm{alone,so\ far}}$, the standalone time for completed, accepted work,
giving `U = E / W` when `W > 0`.
We update both incrementally, estimating each completed iteration's standalone
time from offline points because it cannot be measured directly under sharing.

We call the root agent and its sub-agents working on the same multi-agent task
an **agent group**. The normalized slowdown for the group is the
$T_{\mathrm{alone,so\ far}}$-weighted average of its agents' normalized-slowdown
scores, using the per-agent share defined in step 4.

### Responsibilities

Components describe runtime responsibilities; PRs describe how we deliver them.
Some components span multiple PRs.

| Component | Responsibility | PR |
| --- | --- | --- |
| ThunderAgent adapter | We reuse ThunderAgent's capacity checks, placement, reservations and lifecycle management, and expose hooks for Duru decisions. | 1 |
| Event collection | Capture router lifecycle events and accepted vLLM work with request and root identities. | 2 |
| Event and score transport | Deliver ordered events to accounting and fresh score snapshots to engine processes. | 3 |
| Calibration | Measure and predict standalone execution costs for the deployment configuration. | 4 |
| U accounting and engine scheduling | Maintain E/W, clean up completed roots, publish scores and prioritize vLLM waiting requests. | 5 |
| Router ordering and serving integration | Prioritize waiting programs and connect Duru to the public serving path. | 6 |
| Admission exception | Authorize alpha-based exceptions after normal admission fails; ThunderAgent applies placement and reservation changes. | 7 |
| Documentation | Explain setup, configuration, architecture and evaluation in the READMEs and guides. | 8 |
| Diagnostics | Provide optional counters, logs and recordings alongside the components they describe. | 2–3, 5–7 |

The intended data flow is:

```mermaid
flowchart LR
    H["Router events"] -->|"In-process"| A["Duru accounting<br/>in the router process"]
    E["vLLM engine events"] -->|"IPC"| A
    C["Calibration"] -->|"Estimated standalone costs"| A
    A --> S["Normalized slowdown"]
    S -->|"In-process"| R["ThunderAgent adapter<br/>Router ordering and admission"]
    S -->|"IPC"| W["vLLM adapter<br/>Engine waiting-queue ordering"]
```

### Scheduling behavior

- **Score:** Duru accumulates `E`, the group's fair-share worker time, and `W`, the estimated standalone cost of completed, accepted work. It prioritizes larger `U = E / W`.
- **Lookahead:** Before the group completes any work, each waiting request uses `E` divided by the estimated standalone cost of its next five engine steps.
- **Router admission:** Try agents in decreasing score order. Admit requests that fit normally; otherwise, allow the admission exception when a fresh, valid score reaches `alpha`. If neither applies, try the next agent.
- **Engine scheduling:** Apply the same ordering within vLLM's waiting queues, breaking ties by arrival time. vLLM retains physical memory checks; Duru does not preempt running requests for fairness.
- **Updates and failures:** Calculate `E` and `W` through the same confirmed event time. Accounting failures return explicit 503 errors to new and waiting requests without cancelling already-dispatched requests.

## Implementation plan: eight PRs

Organize the implementation around seven responsibilities, followed by one documentation PR. Each implementation PR includes its own tests, docstrings, and CLI help.
| PR | Responsibility | What it does | Depends on |
| --- | --- | --- | --- |
| 1 | ThunderAgent adapter | Define the boundary between Duru decisions and ThunderAgent mechanisms. Extract the minimum extension points for policy decisions and lifecycle observation, including reusable serving startup in `thunderagent_router/__main__.py` and CLI argument configuration in `thunderagent_router/args.py`. ThunderAgent retains capacity checks, worker placement, reservations, locking, pause/resume and cancellation. Preserve native defaults; introduce no Duru decisions. | None |
| 2 | Event collection | Define observations and capture request start/end, program/root identity, accepted engine work, and worker completion. Use an injected event sink so callback semantics can be tested without IPC. Observe native execution without changing queue order. | 1 |
| 3 | Transport | Connect producers and consumers across local processes. Deliver work events through bounded ordered channels, and deliver score snapshots through a separate latest-value channel. Include readers, freshness metadata, producer progress, shutdown and failure reporting. Test with synthetic events/scores; do not depend on the accounting implementation. | 2 |
| 4 | Calibration | Define the model/configuration-specific calibration artifact, measure solo work costs, and predict the cost of accepted work. Include planning and measurement tools, interpolation and contract validation. Keep measurement orchestration separate from the runtime predictor. | None |
| 5 | Duru runtime and engine scheduler | Consume observations, compute entitlement E and accepted work W, derive U, retain state through tool gaps, and release completed roots only after all work is accounted for. Publish fresh snapshots and use them to order waiting requests inside vLLM. Include background lifecycle, failure supervision and runtime diagnostics. | 2, 3, 4 |
| 6 | Router ordering and serving integration | Use the runtime's fresh scores to order programs waiting for ThunderAgent admission, including first admission and capacity backfill. Add the opt-in public entrypoint, configuration, identity/lifecycle wiring and CLI help. Keep the alpha exception disabled in this PR. | 1, 5 |
| 7 | Admission exception | After a candidate's normal router capacity check fails, permit an exception when its waiting request has a fresh, valid score at least `alpha` (default: `1`). ThunderAgent retains placement and reservation changes; the engine still requires real KV space. Include alpha configuration and matched end-to-end evaluation. | 6 |
| 8 | Documentation | Add the Duru and calibration READMEs, architecture explanation, setup and configuration guide, supported-topology limitations, and evaluation instructions for the completed implementation. | 7 |

For example, a request completes a prefill step. Collection reports the accepted
work; transport delivers it; calibration estimates its cost; the runtime updates
that root's U. The router uses the resulting scores to order admissions. If a
candidate cannot fit normally, the exception rule may authorize admission. Once
requests are inside vLLM, Duru's normalized slowdown based ordering chooses which one to try
next, while native memory allocation remains authoritative.

## Requirements

- Duru MUST be opt-in and preserve native ThunderAgent behavior when disabled.
- The fairness policy MUST remain independent of ThunderAgent and vLLM; system-specific integration MUST live in adapters.
- ThunderAgent MUST retain responsibility for placement and state changes. Engine scheduling MUST respect physical KV-cache limits, including when the alpha exception admits a request.
- Accounting MUST report missing work as an explicit error, bound retained state, and release completed agent groups. Scheduling MUST NOT depend on diagnostic logs or score files.
- Evaluation SHOULD report fairness, throughput, and latency against unchanged baselines under matching workloads and serving configurations.

## Alternate Solutions

The [paper's related-work discussion](https://arxiv.org/html/2611.23558v1#S8)
motivates the following choices:

- **Weighted-token fairness (VTC, DLPM):** balances service across clients, with DLPM also considering prefix locality. Token counts do not fully capture differences in execution cost across context lengths and inference phases, so Duru uses estimated standalone time.
- **Token-latency isolation (FairInference):** protects per-token latency. Our objective is cumulative slowdown across an agent group's successive LLM calls, which requeset level fairness alone does not establish.
- **Prediction-based workflow scheduling (SAGA):** uses predicted remaining work and task deadlines. Agent execution can change with tool results and retries, so Duru tracks observed waiting and completed work without requiring a prediction of remaining task execution.
- **Router-only or engine-only scheduling:** simplifies integration but leaves the other queue outside fairness control. Duru coordinates both, as motivated by the [paper's scheduling analysis](https://arxiv.org/html/2611.23558v1#S4).

## Open review questions

1. Which policy/lifecycle hooks belong in ThunderAgent, and which should stay in
   the Duru adapter?
2. Is same-node IPC an acceptable initial scope, or is cross-node delivery a
   requirement before upstream integration?
3. What benchmark coverage and acceptable throughput tradeoff are required?

## References

- [*Towards Fair Resource Sharing in Agentic LLM Serving*](https://arxiv.org/abs/2611.23558v1), arXiv:2611.23558v1: §3 for normalized/live slowdown and calibration; §§5.1–5.5 for the control signal, coordinated ordering, and admission exceptions; §5.6 for autoscaling, which is outside this DEP.
- [Dynamo contribution guidance](https://github.com/ai-dynamo/dynamo/blob/main/CONTRIBUTING.md)
- [Dynamo DEP template](https://github.com/ai-dynamo/dynamo/blob/main/.github/ISSUE_TEMPLATE/dep.yml)
