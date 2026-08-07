<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmark Isolation

Keep the serving path and benchmark client isolated from unrelated work that can change latency or throughput.

## Exclusivity

- Never run more than one benchmark against the same Dynamo endpoint at the same time.
- Never run an unrelated benchmark, training job, evaluation, profile capture, or GPU workload on a serving GPU used
  by the candidate under test.
- Keep the AIPerf client off the serving GPU nodes when practical. Do not co-locate CPU-, disk-, or network-intensive
  analysis with an active benchmark when that work can contend with the frontend, workers, or AIPerf client.
- Run benchmarks in parallel only when their serving allocations, clients, endpoints, and relevant node resources are
  disjoint. Serialize them whenever exclusivity is uncertain.

## Preflight And Evidence

Before starting AIPerf:

1. Resolve the Kubernetes context, namespace, DGD pods, serving nodes, and benchmark Job placement.
2. Inspect the serving nodes for competing pods and determine whether MIG, time slicing, or another sharing mechanism
   is active.
3. Confirm that no other benchmark is targeting the candidate endpoint.

Record the AIPerf Job and pod, endpoint, serving pod and node list, relevant GPU allocation information, competing-work
check, and any isolation limitation in `benchmark_execution.json`.

If the serving allocation or measurement path cannot be shown to be isolated, address if possible, otherwise stop or mark the run invalid. Do not use
an unexplained contention risk as candidate-promotion evidence.
