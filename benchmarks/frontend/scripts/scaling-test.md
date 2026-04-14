<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Frontend Scaling Test: Finding the Saturation Point

This guide walks through using the sweep runner to find the saturation point of
a Dynamo frontend serving a real vLLM backend.  The saturation point is the
request rate at which latency begins to degrade -- prefill requests start
queuing instead of being served immediately, TTFT p99 spikes, and throughput
plateaus.

---

## Overview

The test sweeps increasing request rates (`--rps`) at a fixed input sequence
length while keeping the backend warm (`--reset-strategy frontend`).  Each data
point is a 60-second aiperf run at a controlled RPS.  The sweep stops
automatically after consecutive failures (`--max-consecutive-fails`).

**What you get:**

- Per-RPS throughput (actual req/s vs target), TTFT p50/p99, ITL p50/p99
- Prometheus pre/post metrics for pipeline stage breakdown
- CSV + summary for easy comparison

---

## Prerequisites

1. **K8s namespace** with:
   - `hf-token-secret` (HuggingFace token)
   - `nvcrimagepullsecret` (image pull credentials)
   - `model-cache` PVC (RWX, large enough for model weights)
   - Model weights downloaded to PVC (see "Model Download" below)

2. **DGD deployed** with the target model and backend.

3. **sweep_runner.py** accessible from a machine with `kubectl` access to the
   cluster.

---

## Model Download (gpt-oss-20b example)

Download the model to the PVC, excluding large non-inference directories:

```bash
# Create a download Job (adjust image and namespace)
kubectl apply -n <namespace> -f - <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: model-download-gpt-oss-20b
spec:
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
      imagePullSecrets:
        - name: nvcrimagepullsecret
      containers:
        - name: download
          image: nvcr.io/nvidian/dynamo-dev/biswa:vllm-runtime-1a8bce12ea
          command: ["python3", "-c"]
          args:
            - |
              import os, subprocess, sys, pathlib
              model = "openai/gpt-oss-20b"
              os.environ["HF_HOME"] = "/model-store"
              cmd = ["huggingface-cli", "download", model,
                     "--exclude", "metal/*", "--exclude", "original/*",
                     "--local-dir", "/model-store/hub/models--openai--gpt-oss-20b/snapshots/main"]
              sys.exit(subprocess.run(cmd).returncode)
          env:
            - name: HF_HOME
              value: /model-store
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: HF_TOKEN
          volumeMounts:
            - name: model-cache
              mountPath: /model-store
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache
EOF

# Monitor
kubectl logs -n <namespace> -l job-name=model-download-gpt-oss-20b -f
```

---

## Deploy the DGD

Use the provided template for gpt-oss-20b with TP=2:

```bash
# Template path (relative to repo root)
# benchmarks/frontend/dgd/templates/vllm-gpt-oss-20b.yaml
#
# Key settings in the template:
#   - tensor-parallel-size 2 (2 GPUs per worker)
#   - max-model-len 65536
#   - gpu-memory-utilization 0.90
#   - GPU toleration for scheduling

# Deploy directly (adjust values as needed):
kubectl apply -n <namespace> -f - <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: gpt-oss-20b-bench
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        imagePullSecrets:
          - name: nvcrimagepullsecret
        mainContainer:
          image: <your-image>
          command: ["/bin/sh", "-c"]
          args: ["python3 -m dynamo.frontend --router-mode round-robin --http-port 8000"]
          env:
            - name: DYN_TOKENIZER_BACKEND
              value: "default"
            - name: DYN_PERF_DIAG
              value: "1"
            - name: HF_HOME
              value: /model-store
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: HF_TOKEN
          volumeMounts:
            - name: model-cache
              mountPath: /model-store
        volumes:
          - name: model-cache
            persistentVolumeClaim:
              claimName: model-cache

    VllmWorker:
      componentType: worker
      replicas: 4                    # <-- number of backend replicas
      extraPodSpec:
        imagePullSecrets:
          - name: nvcrimagepullsecret
        tolerations:
          - effect: NoSchedule
            key: nvidia.com/gpu
            operator: Exists
        mainContainer:
          image: <your-image>
          command: ["/bin/sh", "-c"]
          args:
            - >-
              python3 -m dynamo.vllm
              --model /model-store/hub/models--openai--gpt-oss-20b/snapshots/main
              --served-model-name openai/gpt-oss-20b
              --tensor-parallel-size 2
              --max-model-len 65536
              --gpu-memory-utilization 0.90
          env:
            - name: HF_HOME
              value: /model-store
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: HF_TOKEN
          resources:
            limits:
              nvidia.com/gpu: "2"    # <-- 2 GPUs for TP=2
          volumeMounts:
            - name: model-cache
              mountPath: /model-store
        volumes:
          - name: model-cache
            persistentVolumeClaim:
              claimName: model-cache
EOF

# Wait for all pods to be ready
kubectl get pods -n <namespace> -w
```

---

## Run the Saturation Sweep

### Baseline: HF tokenizer, RPS sweep

```bash
cd benchmarks/frontend/scripts

python3 sweep_runner.py --mode k8s \
    --dgd-name gpt-oss-20b-bench \
    --namespace <namespace> \
    --endpoint gpt-oss-20b-bench-frontend:8000 \
    --model openai/gpt-oss-20b \
    --backend vllm \
    --image <your-image> \
    --tokenizers hf \
    --concurrency 200 \
    --rps 10,20,30,40,50,60,70,80,90,100 \
    --isl 6144 \
    --osl 256 \
    --benchmark-duration 60 \
    --reset-strategy frontend \
    --isolation reuse_by_deploy_key \
    --worker-replicas 4 \
    --max-consecutive-fails 2
```

**Flag explanations:**

| Flag | Value | Purpose |
|------|-------|---------|
| `--rps 10,20,...,100` | Sweep dimension | Each run targets a fixed request rate. aiperf uses `--request-rate` to cap submission. |
| `--concurrency 200` | High ceiling | Maximum in-flight requests. Set high so aiperf can sustain the target RPS without being limited by available connection slots. This is NOT a sweep dimension. |
| `--isl 6144` | Fixed ISL | Holds input length constant to isolate throughput scaling. |
| `--osl 256` | Fixed OSL | Consistent output length across all runs. |
| `--benchmark-duration 60` | 60s per point | Long enough for vLLM scheduling to stabilize. |
| `--reset-strategy frontend` | Frontend-only | Resets Prometheus counters between runs, but keeps vLLM workers alive with warm KV caches and CUDA graphs. Avoids the ~90s full DGD restart per point. |
| `--isolation reuse_by_deploy_key` | Reuse deployment | Since tokenizer=hf is constant, no DGD restart between runs. Only a frontend pod restart for clean metrics. |
| `--max-consecutive-fails 2` | Auto-stop | After 2 consecutive failures at a given RPS, remaining higher RPS values are skipped. |

### Follow-up: FastTokens comparison

Once you have the baseline, run the same sweep with fastokens to see if the
saturation point shifts:

```bash
python3 sweep_runner.py --mode k8s \
    --dgd-name gpt-oss-20b-bench \
    --namespace <namespace> \
    --endpoint gpt-oss-20b-bench-frontend:8000 \
    --model openai/gpt-oss-20b \
    --backend vllm \
    --image <your-image> \
    --tokenizers fastokens \
    --concurrency 200 \
    --rps 10,20,30,40,50,60,70,80,90,100 \
    --isl 6144 \
    --osl 256 \
    --benchmark-duration 60 \
    --reset-strategy frontend \
    --isolation reuse_by_deploy_key \
    --worker-replicas 4 \
    --max-consecutive-fails 2
```

### Fine-grained sweep around the inflection

If the baseline shows saturation between, say, RPS=40 and RPS=60:

```bash
python3 sweep_runner.py --mode k8s \
    ... \
    --rps 35,40,45,50,55,60 \
    --reset-strategy frontend \
    --isolation reuse_by_deploy_key
```

---

## Reading the Results

The sweep produces `results.csv` and `summary.md` in the output directory.

### Identifying the saturation point

Look for these signals in the CSV:

| RPS | Actual Req/s | TTFT p50 | TTFT p99 | ITL p99 | Status |
|----:|-----------:|--------:|--------:|-------:|--------|
| 10 | 10.0 | 800ms | 1200ms | 30ms | ok |
| 20 | 19.8 | 850ms | 1400ms | 32ms | ok |
| 30 | 29.5 | 900ms | 2000ms | 35ms | ok |
| 40 | 38.0 | 1200ms | 5000ms | 45ms | ok -- onset |
| 50 | 42.0 | 3000ms | 15000ms | 80ms | ok -- saturated |
| 60 | 41.5 | 8000ms | 30000ms | 120ms | ok -- overloaded |
| 70 | -- | -- | -- | -- | fail |

**Saturation indicators:**

1. **Actual req/s < target RPS**: The system cannot sustain the requested rate.
   At RPS=50, only 42 req/s are achieved.
2. **TTFT p99 spike**: A sharp increase (e.g., 2x-5x) means prefill requests
   are queuing behind each other.
3. **ITL p99 degradation**: Decode throughput drops because the vLLM scheduler
   is overloaded with concurrent prefills.
4. **Errors/failures**: Timeouts, OOM, or vLLM rejecting requests.

The **saturation point** in the example above is **RPS ~40** -- the last rate
where actual throughput tracks the target and TTFT p99 is still reasonable.

### Prometheus metrics

Each run captures `frontend_metrics_pre.txt` and `frontend_metrics_post.txt`.
Key metrics for saturation analysis:

- `dynamo_frontend_stage_duration_seconds{stage="preprocess"}` -- tokenization time
- `dynamo_frontend_stage_duration_seconds{stage="transport_roundtrip"}` -- backend latency
- `dynamo_frontend_queued_requests` -- requests waiting in HTTP queue (should be 0 below saturation)
- `dynamo_frontend_inflight_requests` -- concurrent in-flight requests
- `dynamo_frontend_time_to_first_token_seconds` -- TTFT histogram buckets

---

## DGD Template Reference

The `dgd/templates/vllm-gpt-oss-20b.yaml` template is pre-configured for
gpt-oss-20b with TP=2.  To use it with `--deploy-template`:

```bash
python3 sweep_runner.py --mode k8s \
    --deploy-template benchmarks/frontend/dgd/templates/vllm-gpt-oss-20b.yaml \
    --dgd-name gpt-oss-20b-bench \
    --model /model-store/hub/models--openai--gpt-oss-20b/snapshots/main \
    --image <your-image> \
    --worker-replicas 4 \
    ...
```

The template substitutes these variables at deploy time:
`${DGD_NAME}`, `${IMAGE}`, `${MODEL}`, `${MODEL_NAME}`,
`${WORKER_REPLICAS}`, `${DYN_TOKENIZER_BACKEND}`, `${FRONTEND_PORT}`,
`${ROUTER_MODE}`.

---

## Tuning Parameters

| Parameter | Recommended Range | Notes |
|-----------|-------------------|-------|
| `--benchmark-duration` | 60-120s | Longer = more stable averages but slower sweep |
| `--concurrency` | 2-4x max target RPS | Must be high enough that aiperf can reach the target rate |
| `--rps` | Start at 10, double until failures | Geometric progression finds the order of magnitude fast |
| `--worker-replicas` | 1-8 | More replicas = higher saturation point but more GPUs |
| `--reset-strategy` | `frontend` for saturation tests | `graph` for clean-baseline TTFT measurements |
| `--isolation` | `reuse_by_deploy_key` for same-tokenizer sweeps | Avoids unnecessary DGD restarts |
| `--max-consecutive-fails` | 2-3 | Higher = more data points at the failure boundary |
