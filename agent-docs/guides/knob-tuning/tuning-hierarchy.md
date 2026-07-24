<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Optimization Lever Priority

Use this guide before opening the Dynamo or engine knob catalogs. It ranks only controls that can be expressed in the
current DGD or its engine configuration. It is not a sweep order and does not require testing a knob from every row.

The hierarchy decides the level first: topology before configuration within that topology. The default rankings within
each level help break ties for a new model; direct evidence from the current workload overrides the generic ranking.

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

Engine selection and model-weight quantization are target-definition choices, not ordinary candidates in this guide.
KV-cache dtype remains an engine tuning lever because it changes the runtime cache rather than the model weights.

## Tier 1 — Deployment Topology and Fit

First ask whether the deployment shape can efficiently serve the target workload. This is a screening decision, not an
automatic topology experiment. Evaluate topology in this order:

1. Compute the model, activation, and KV-memory fit and establish the minimum viable TP, PP, and EP.
2. Prefer the smallest parallelism that fits with operating headroom, then use remaining fixed-budget GPUs for
   replicas when the workload can benefit from them.
3. Consider aggregated versus disaggregated serving only when workload shape, scale, or independent prefill/decode
   objectives justify the transfer and coordination cost.
4. For an existing disaggregated deployment, check prefill/decode allocation and rate matching before adding workers.
5. Verify node placement and the required fast fabric when the selected topology crosses GPUs or nodes.

Choose a topology hypothesis only when model sizing, Kubernetes or engine evidence, rate imbalance, transfer behavior,
or same-regime history supports a structural mismatch. Consult the
[model-sizing guides](../model-sizing/classification.md) and, for disaggregated serving,
[rate matching](../rate-matching/matching.md).

A topology change may require several YAML fields to move together. Treat that as one functionality-required mechanism,
record the full GPU-resource change, and follow the
[one-variable rule](../../rules/optimization/one-variable.md). Do not change topology merely because a lower-level knob
failed or because unused cluster capacity exists.

## Tier 2 — Configuration Within the Chosen Topology

When the topology is viable, hold its component graph, parallelism, worker counts, and GPU budget fixed. Select one
configuration family whose condition is visible in the evidence.

| Default priority | Lever family | Promote when | Demote or skip when |
|---|---|---|---|
| 1 | CUDA graph engagement and coverage | graphs are disabled, startup reports capture failure, or observed engine batch shapes exceed capture coverage | startup and runtime evidence proves graphs cover the target operating region |
| 2 | Admission, batching, prefill scheduling, and workspace | useful batch occupancy is low, token limits do not cover the target input, small prefill chunks repeat fixed overhead, or the first load burst approaches OOM | the engine already admits the intended work with stable memory headroom |
| 3 | Speculative decoding | the workload is decode- or latency-bound at low to moderate concurrency, the model and engine support it, and representative prompts provide useful acceptance | the target is prefill- or TTFT-bound, high-concurrency throughput is primary, prompts are not representative, or draft state reduces capacity |
| 4 | KV-cache dtype and capacity | long context or high concurrency is KV-bound and additional cache capacity can admit useful work | KV capacity is not limiting or the selected attention path does not support the dtype |
| 5 | Engine backend or autotuner selection | logs prove an unsuitable path, or same-version and same-hardware evidence predicts a gain at the target parallelism and concurrency | support, engagement, or version-specific behavior is uncertain |
| 6 | Dynamo routing and prefix reuse | worker load is skewed or the real workload has reusable prefixes that the current routing or cache policy misses | reuse exists only because synthetic inputs repeat, or cache bookkeeping costs dominate at the target load |
| 7 | KVBM or engine KV offload | repeated long prefixes make prefill or TTFT dominant and host or disk capacity can retain useful KV | decode latency is primary, prefixes do not repeat, or transfer and host-memory costs are unmeasured |
| 8 | Frontend, transport, and pod resources | CPU or memory throttling, request-plane overhead, connection handling, or KV-transfer fallback limits the request path | engine execution or admission remains the measured limit |

Use the exact engine catalog for engine-owned fields:
[vLLM](vllm.md), [SGLang](sglang.md), or [TensorRT-LLM](tensorrt-llm.md). Use the
[Dynamo catalog](dynamo.md) for DGD shape, routing, transport, KVBM, and other Dynamo-owned controls.

The first two rows are universal checks, not automatic changes. A directly observed disaggregated-stage imbalance or
slow-path KV transfer moves the matching rate or transport lever ahead of generic speculative-decoding and cache
priors.

## Concurrency and Workload Overrides

Several high-impact controls can reverse direction as load increases:

- speculative decoding is usually strongest for low-concurrency decode latency and can become neutral or harmful for
  high-concurrency throughput;
- prefix caching requires genuine repeated prefixes and can help at low load while its bookkeeping reduces throughput
  at high load;
- TRT-LLM autotuner results depend on TP, concurrency, engine version, and the active collective path; and
- CUDA graph capture must cover the observed peak engine batch shape at every target operating point.

Treat concurrency 8–16 as a possible crossover region to investigate, not a fixed threshold. Rank against the current
workload's AIPerf and engine evidence. Do not use repeated synthetic prompts to justify prefix caching or to tune
speculative-decoding draft length for natural traffic.

## Conditional Lane — Local Planner

Consider Local Planner controls only when autoscaling behavior is an explicit objective and the current single DGD
already includes the Planner. Establish a sound fixed-capacity configuration first. Do not use autoscaling to hide a
topology, admission, or rate-matching problem, and do not propose Planner changes during a fixed-capacity AIPerf
comparison.

## Selection Procedure

1. Screen Tier 1 and keep the topology unchanged when the evidence does not implicate it.
2. For a viable topology, start with the Tier 2 default ranking and retain only levers whose mechanism matches the
   observed symptom.
3. Promote a lower-ranked lever when current-run evidence is stronger than the generic prior.
4. Remove fixed, unsupported, duplicate, already-failed, or unverified choices.
5. Rank the remainder by direct evidence, expected effect on the primary objective, information value, risk,
   reversibility, experiment cost, and diff size.
6. Select one independently testable knob or one justified coupled mechanism.
7. Return `no_proposal` when no candidate satisfies the evidence threshold.

Record the selected tier, default priority, lever family, why earlier choices were retained or skipped, the exact
changed fields, the intended mechanism, and the measurements that would support or falsify it.
