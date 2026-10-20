<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Beam-1.0-501B-A23B Benchmark

[perf.yaml](perf.yaml) runs an AIPerf synthetic chat benchmark through the Dynamo
frontend: 8,192 input tokens, 1,024 requested output tokens, streaming, and one
concurrency point per Job.

## Run

First deploy a profile using the [recipe](../README.md). In `perf.yaml`, set:

- `BASE_URL`: `http://<deployment-name>-frontend:8000` for your chosen profile.
- `TOKENIZER`: `reflection/Beam-1.0-501B-A23B-FP8` for H200, or
  `reflection/Beam-1.0-501B-A23B-NVFP4` for B200/B300/GB200/GB300.
  The tokenizer resolves from the shared Hugging Face cache while offline.
- `ALLOCATED_GPUS`: all GPUs in the deployment, including prefill and decode
  workers (default `8`; use `16` for H200 disaggregated).
- `CONCURRENCY`: the number of concurrent requests (default `32`). The Job sends
  one warmup request and 20 measured requests per concurrency slot.

The Job uses `nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0`, as in the other model recipes.

Run from `recipes/beam-1.0-501b` with `NAMESPACE` set:

```bash
kubectl apply -n "$NAMESPACE" -f perf/perf.yaml
kubectl logs -n "$NAMESPACE" -f job/beam-synthetic-bench --pod-running-timeout=10m
kubectl wait -n "$NAMESPACE" --for=condition=Complete job/beam-synthetic-bench --timeout=7200s
```

## Results

Raw AIPerf exports, the log, and the client version are saved on
`shared-model-cache` under `/shared-model-cache/perf/beam/<pod-uid>/`. Copy them
from a running helper Pod mounting that claim:

```bash
kubectl cp "$NAMESPACE/<cache-helper-pod>:/shared-model-cache/perf/beam/<pod-uid>" ./beam-results
```

To test another concurrency, save the results, delete the completed Job, edit
`CONCURRENCY`, and apply it again. Keep cache state consistent across comparisons.
Report TTFT, inter-token latency, output tokens/s, and failed requests; normalize
GPU throughput by every allocated GPU (8 or 16, including both roles for
disaggregated targets). Record the image and checkpoint revisions with each run.
