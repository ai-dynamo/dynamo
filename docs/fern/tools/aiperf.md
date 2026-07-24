---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Benchmark a Kubernetes Deployment with AIPerf
subtitle: Measure latency and throughput through a DynamoGraphDeployment frontend
---

AIPerf sends load to an OpenAI-compatible endpoint and measures latency and throughput. Use this
tutorial after deploying a model with a DynamoGraphDeployment (DGD).

AIPerf measures the live deployment. It does not predict a configuration like AIConfigurator or run
a GPU-free simulation like Mocker and DynoSim.

## Prerequisites

Install AIPerf on the workstation or benchmark host:

```bash
pip install aiperf
```

Set the namespace and forward the DGD frontend service to port 8000:

```bash
export NAMESPACE=dynamo-system
kubectl port-forward -n "$NAMESPACE" svc/<deployment-name>-frontend 8000:8000
```

The service name is the DGD `metadata.name` followed by `-frontend`.

## Run a baseline benchmark

In another terminal, set `--model` to the model name served by the deployment:

```bash
aiperf profile \
  --model Qwen/Qwen3-32B-FP8 \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 10 \
  --request-count 100 \
  --synthetic-input-tokens-mean 2000 \
  --output-tokens-mean 256
```

AIPerf writes artifacts to `artifacts/` and prints a metrics summary. Record Time to First Token
(TTFT), Inter-Token Latency (ITL), end-to-end latency, and output throughput.

## Compare Kubernetes configurations

Change one deployment setting at a time, wait for the DGD to become ready, and rerun the same AIPerf
command. Common comparisons include:

- aggregated versus disaggregated serving
- KV-aware versus round-robin routing
- different worker replica counts
- different TP or PP sizes recommended by AIConfigurator
- backend or engine-argument changes

Keep the request shape, concurrency, and request count fixed so the results remain comparable. Then
sweep concurrency to find the saturation point of the selected deployment.

## Next steps

- Use [Sizing with AIConfigurator](../kubernetes/dgd-aiconfigurator.md) to select additional DGD
  configurations.
- Use the full [Dynamo Benchmarking guide](../benchmarks/benchmarking.md) for in-cluster load
  generation, concurrency sweeps, arrival patterns, trace replay, visualization, and GPU telemetry.
- Start from a [Dynamo Recipe](https://github.com/ai-dynamo/dynamo/tree/main/recipes) when a supported
  model and hardware combination already has a known-good baseline.
