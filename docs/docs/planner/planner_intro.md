---
title: "Planner"
---

# Planner

The planner monitors the state of the system and adjusts workers to
ensure that the system runs efficiently.

Currently, the planner can scale the number of vllm workers up and down
based on the kv cache load and prefill queue size:

Key features include:

- **SLA-based scaling** that uses predictive modeling and performance
  interpolation to proactively meet TTFT and ITL targets
- **Graceful scaling** that ensures no requests are dropped during
  scale-down operations

<div class="admonition seealso">

🚀 Quick Start

**New to SLA Planner?** Start with the \[SLA Planner Quick Start
Guide\](/docs/planner/sla_planner_quickstart.md) for a complete,
step-by-step workflow.

**Prerequisites**: SLA-based planner requires pre-deployment profiling
(2-4 hours on real silicon or a few minutes using simulator) before
deployment. The Quick Start guide includes everything you need.

</div>

||
||
||
||
||
||
||
||
||
||
||
||

<div class="toctree" hidden="">

Overview \<self\> SLA Planner Quick Start \<sla_planner_quickstart\>
SLA-Driven Profiling \<../benchmarks/sla_driven_profiling.md\> SLA-based
Planner \<sla_planner.md\>

</div>
