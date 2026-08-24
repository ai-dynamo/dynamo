<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Optimization Lever Priority

Use this guide before opening the Dynamo or engine knob catalogs.

The hierarchy decides the level first: topology before configuration within that topology.

## Entry Gates

Before selecting a lever:

- require a successfully deployed current DGD and valid, comparable AIPerf analysis;
- state the primary objective or failed SLO and the client-visible symptom;
- identify the target concurrency or request-rate region;
- verify that the current manifest and intended configuration actually engaged;
- preserve fixed target constraints such as model identity, serving framework, hardware, workload, and model-weight
  precision;
- check prior candidates for an equivalent tested change; and
- satisfy [evidence before spend](../../rules/optimization/evidence-before-spend.md).

If the current configuration did not engage or the run is invalid, repair or remeasure it before proposing another
optimization.


## Category 1 — Deployment Topology and Operating Regime

First ask whether the deployment shape can efficiently serve the user workload. Topology sets four things at once,
and only one of them is memory: per-rank effective batch and arithmetic intensity (the throughput axis); weight and
KV memory footprint (the fit axis); the collective-communication pattern and volume (TP all-reduce, EP all-to-all,
none for DP); and whether attention and weights are replicated or sharded. Do not evaluate a topology change on
memory alone: when memory is not binding, the topology question is not closed; it moves to compute and
communication. A configuration that fits can still be the wrong shape for the operating point. Evaluate topology in
this order:

1. Compute the model, activation, and KV-memory fit and establish the minimum viable TP, PP, and EP.
2. Compute the effective per-rank batch at the target concurrency: data parallelism serves batch/N per replica,
   shrinking per-rank batch and starving compute even though everything fits, while tensor parallelism runs the
   whole batch through one sharded forward pass, raising per-rank batch and freeing memory at the cost of
   collective communication.
3. Among layouts that fit with operating headroom, prefer the one that maximizes useful per-rank batch and compute
   efficiency at the target concurrency, then weigh its communication cost; fit is a constraint, not the objective.
   Use remaining fixed-budget GPUs for replicas when the workload can benefit from them.
4. Consider aggregated versus disaggregated serving only when workload shape, scale, or independent prefill/decode
   objectives justify the transfer and coordination cost.
5. For an existing disaggregated deployment, check prefill/decode allocation and rate matching before adding workers.
6. Verify node placement and the required fast fabric when the selected topology crosses GPUs or nodes.

Choose a topology hypothesis when model-sizing arithmetic, per-rank batch at the target concurrency, Kubernetes or
engine evidence, rate imbalance, transfer behavior, same-regime history, or a sibling recipe for the same model on
other hardware supports a structural mismatch. Sibling recipes are hypothesis priors, not adoption evidence: even
when their checkpoint or hardware is incompatible, their parallelism and serving-mode choices transfer as candidates.
Rank candidate layouts cheaply (sizing arithmetic, projections, published same-model recipes) before spending a
deployment on one, and revisit topology after baseline characterization and whenever the measured regime changes; an
inherited layout is a candidate like any other, not a settled decision. Consult the
[model-sizing guides](../model-sizing/classification.md) and, for disaggregated serving,
[rate matching](../rate-matching/matching.md).

A topology change may require several YAML fields to move together. Treat that as one functionality-required mechanism,
record the full GPU-resource change, and follow the
[one-variable rule](../../rules/optimization/one-variable.md). Do not change topology merely because a lower-level knob
failed or because unused cluster capacity exists.

## Category 2 — Configuration Within the Chosen Topology

When the topology is viable, hold its component graph, parallelism, worker counts, and GPU budget fixed. Select one
configuration family whose condition is visible in the evidence.

| Lever family | Promote when | Demote or skip when |
|---|---|---|
| CUDA graph engagement and coverage | graphs are disabled, startup reports capture failure, or observed engine batch shapes exceed capture coverage | startup and runtime evidence proves graphs cover the target operating region |
| Admission, batching, prefill scheduling, and workspace | useful batch occupancy is low, token limits do not cover the target input, small prefill chunks repeat fixed overhead, or the first load burst approaches OOM | the engine already admits the intended work with stable memory headroom |
| Speculative decoding | the workload is decode- or latency-bound at low to moderate concurrency, the model and engine support it, and representative prompts provide useful acceptance | the target is prefill- or TTFT-bound, high-concurrency throughput is primary, prompts are not representative, or draft state reduces capacity |
| KV-cache dtype and capacity | long context or high concurrency is KV-bound and additional cache capacity can admit useful work | KV capacity is not limiting or the selected attention path does not support the dtype |
| Engine backend or autotuner selection | logs prove an unsuitable path, or same-version and same-hardware evidence predicts a gain at the target parallelism and concurrency | support, engagement, or version-specific behavior is uncertain |
| Dynamo routing and prefix reuse | worker load is skewed or the real workload has reusable prefixes that the current routing or cache policy misses | reuse exists only because synthetic inputs repeat, or cache bookkeeping costs dominate at the target load |
| KVBM or engine KV offload | repeated long prefixes make prefill or TTFT dominant and host or disk capacity can retain useful KV | decode latency is primary, prefixes do not repeat, or transfer and host-memory costs are unmeasured |
| Frontend, transport, and pod resources | CPU or memory throttling, request-plane overhead, connection handling, or KV-transfer fallback limits the request path | engine execution or admission remains the measured limit |

Use the exact engine catalog for engine-owned fields:
[vLLM](vllm.md), [SGLang](sglang.md), or [TensorRT-LLM](tensorrt-llm.md). Use the
[Dynamo catalog](dynamo.md) for DGD shape, routing, transport, KVBM, and other Dynamo-owned controls.


## Concurrency and Workload Overrides

Several high-impact controls can reverse direction as load increases:

- speculative decoding is usually strongest for low-concurrency decode latency and can become neutral or harmful for
  high-concurrency throughput;
- prefix caching requires genuine repeated prefixes and can help at low load while its bookkeeping reduces throughput
  at high load;
- TRT-LLM autotuner results depend on TP, concurrency, engine version, and the active collective path; and
- CUDA graph capture must cover the observed peak engine batch shape at every target operating point.


## Conditional Lane — Local Planner

Consider Local Planner controls only when autoscaling behavior is an explicit objective and the current single DGD
already includes the Planner. Establish a sound fixed-capacity configuration first. Do not use autoscaling to hide a
topology, admission, or rate-matching problem, and do not propose Planner changes during a fixed-capacity AIPerf
comparison.

## Bookkeeping

Record the selected tier, default priority, lever family, why earlier choices were retained or skipped, the exact
changed fields, the intended mechanism, and the measurements that would support or falsify it.
