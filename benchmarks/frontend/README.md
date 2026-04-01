<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Frontend Performance Benchmark Suite

A configurable sweep runner for measuring Dynamo frontend model serving performance.  It drives [aiperf](https://github.com/ai-dynamo/aiperf) load against a frontend/mocker (or frontend/vLLM) stack and collects throughput, latency, and observability data across a grid of parameters.

The primary use case is **HuggingFace tokenizer vs. fastokens comparison** -- sweeping across concurrency levels, input sequence lengths (ISL), and worker counts to quantify the tokenizer's impact on end-to-end performance.

---

## Architecture

The codebase follows a three-layer design that separates pure logic from execution and infrastructure concerns.

| Layer | Package | Responsibility |
|-------|---------|----------------|
| **Core** | `scripts/sweep_core/` | Pure data models, plan construction, artifact writing, reporting. No subprocess or kubectl calls. |
| **Executors** | `scripts/sweep_executors/` | `SweepExecutor` protocol with two implementations -- `LocalExecutor` (delegates to `run_perf.sh`) and `K8sDgdExecutor` (DynamoGraphDeployment-based k8s runs). |
| **K8s helpers** | `scripts/sweep_k8s/` | kubectl wrappers, DGD patching, template rendering, aiperf Job launching, and Prometheus metrics capture. |

The entry point is `scripts/sweep_runner.py`, a thin CLI that wires the three layers together: it builds a `SweepPlan` from CLI arguments, selects an executor based on `--mode`, and feeds the plan to the orchestrator.

**Data flow:**

```
CLI args --> SweepConfig --> SweepPlan (Cartesian grid of RunSpecs)
                                |
                          Orchestrator
                                |
                   LocalExecutor  or  K8sDgdExecutor
                        |                   |
                  run_perf.sh        DGD + aiperf Job
                        |                   |
                  artifacts/           artifacts/
```

---

## Quick Start -- Local

Local mode starts a mocker backend and frontend process on the current machine, runs aiperf against them, and tears everything down between runs.

**Prerequisites:**

- `dynamo.mocker` and `dynamo.frontend` installed (from the Dynamo repo)
- `aiperf` installed and on `$PATH`
- A HuggingFace model accessible locally (default: `Qwen/Qwen3-0.6B`)

**Smoke test (2 runs, ~30 s each):**

```bash
cd benchmarks/frontend/scripts

python3 sweep_runner.py \
    --tokenizers hf,fastokens \
    --concurrency 32 \
    --isl 512 \
    --benchmark-duration 30 \
    --speedup-ratio 1000000
```

**Full local sweep:**

```bash
python3 sweep_runner.py \
    --tokenizers hf,fastokens \
    --concurrency 32,64,128 \
    --isl 512,1024,2048
```

**Transport saturation sweep (high concurrency, vary workers):**

```bash
python3 sweep_runner.py \
    --tokenizers hf \
    --concurrency 4096 \
    --num-requests 16384,32768 \
    --workers 1,2,4,8 \
    --speedup-ratio 1000000
```

Results are written to `artifacts/sweep_<timestamp>/`.

---

## Quick Start -- Kubernetes

K8s mode deploys a DynamoGraphDeployment (DGD) into a Kubernetes namespace and launches aiperf as an in-cluster Job that targets the frontend service endpoint.

### Prerequisites

1. **Namespace** -- a dedicated namespace for the benchmark (default: `dynamo-bench`).
2. **HuggingFace token secret** -- a Kubernetes Secret named `hf-token-secret`
   containing your HF token, if the model requires authentication.
3. **Model cache PVC** -- a PersistentVolumeClaim for caching model weights
   (avoids repeated downloads across runs).
4. **DGD deployed** -- either pre-deploy the DGD yourself, or use the
   `--deploy --deploy-template` flags to let the sweep runner create it.
5. **kubectl** configured with access to the target cluster and namespace.

### Example: mocker backend

```bash
python3 sweep_runner.py \
    --mode k8s \
    --dgd-name dynamo-bench-mocker \
    --tokenizers hf,fastokens \
    --concurrency 50,100 \
    --isl 512
```

### Example: template-based deployment

When `--deploy-template` is provided, the runner renders the template with per-run variables (tokenizer, workers, model, etc.) and applies it via kubectl before each run group:

```bash
python3 sweep_runner.py \
    --mode k8s \
    --deploy \
    --deploy-template dgd/templates/mocker.yaml \
    --dgd-name dynamo-bench-mocker \
    --image nvcr.io/.../image:tag \
    --tokenizers hf,fastokens \
    --concurrency 50,100 \
    --isl 512
```

### How aiperf runs in-cluster

The sweep runner creates a short-lived Kubernetes Job in the same namespace as the DGD. The Job pod runs `aiperf` against the frontend's in-cluster service DNS name (e.g., `dynamo-bench-mocker-frontend:8000`). Once the Job completes, artifacts are copied back to the local host via `kubectl cp`.

### Reset strategy

Between runs, the `--reset-strategy` flag controls how the deployed stack is
recycled:

| Strategy | Behavior |
|----------|----------|
| `none` | No resets; runs back-to-back on the same deployment. |
| `frontend` | Restart only the frontend pod between runs. |
| `graph` (default) | Redeploy the entire DGD graph between run groups. |

---

## CLI Reference

All flags for `sweep_runner.py`:

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `local` | Execution mode: `local` or `k8s`. |
| `--backend` | `mocker` | Engine backend: `mocker` (synthetic) or `vllm` (real inference). |
| `--model` | `Qwen/Qwen3-0.6B` | HuggingFace model path. |
| `--model-name` | same as `--model` | Served model name (for multi-model setups). |
| `--tokenizers` | `hf,fastokens` | Comma-separated tokenizer backends. |
| `--concurrency` | `50,100,200` | Comma-separated concurrency levels. |
| `--isl` | `512,1024,2048` | Comma-separated input sequence lengths. |
| `--osl` | `256` | Output sequence length. |
| `--workers` | `2` | Comma-separated worker counts per model. |
| `--num-models` | `1` | Number of model instances. |
| `--speedup-ratio` | `1.0` | Mocker speedup divisor; use large values (e.g., 1000000) for near-instant mocker. |
| `--benchmark-duration` | `60` | aiperf duration in seconds. |
| `--num-requests` | none | Comma-separated request counts (overrides `--benchmark-duration`). |
| `--rps` | none | Comma-separated target request rates (req/s). |
| `--output-dir` | auto-timestamped | Output directory. |
| `--cooldown` | `3` | Seconds between runs. |
| `--max-consecutive-fails` | `2` | Abort sweep after N consecutive failures. |
| `--isolation` | `fresh_per_run` | Isolation policy: `fresh_per_run` or `reuse_by_deploy_key`. |
| `--no-report` | off | Skip per-run report generation. |

### Execution control

| Flag | Description |
|------|-------------|
| `--dry-run` | Print the sweep plan without executing any runs. |
| `--emit-plan` | Print the sweep plan as JSON and exit (useful for Argo or MCP integration). |

### K8s mode options

| Flag | Default | Description |
|------|---------|-------------|
| `--namespace` | `dynamo-bench` | Kubernetes namespace. |
| `--endpoint` | auto-derived | Frontend endpoint (`host:port`). |
| `--dgd-name` | none | DynamoGraphDeployment name. |
| `--image` | none | Container image for k8s deployment. |
| `--deploy-template` | none | Path to a DGD YAML template (enables template-based deployment). |
| `--deploy` | off | Deploy infrastructure before sweeping. |
| `--reset-strategy` | `graph` | Per-run reset: `none`, `frontend`, or `graph`. |
| `--frontend-port` | `8000` | Frontend HTTP port. |
| `--worker-replicas` | `1` | Number of worker pod replicas. |
| `--request-plane` | `tcp` | Request plane transport. |
| `--event-plane` | `nats` | Event plane transport. |
| `--router-mode` | `round-robin` | Frontend router mode. |
| `--hf-token` | none | HuggingFace token for k8s. |
| `--image-pull-secret` | none | Image pull secret name. |
| `--export-level` | `summary` | aiperf export level. |

---

## Artifact Structure

Each sweep produces a timestamped output directory:

```
artifacts/sweep_20260330_143000/
    sweep_config.json        # Full SweepConfig used for this run
    results.csv              # One row per run with key metrics
    summary.md               # Markdown summary table

    mocker_hf_w2_c50_isl512/
        aiperf/              # aiperf JSON output
        prometheus/          # Prometheus metric snapshots
        report.md            # Per-run analysis report (unless --no-report)

    mocker_fastokens_w2_c50_isl512/
        aiperf/
        prometheus/
        report.md
    ...
```

**results.csv columns:**

`run_id`, `backend`, `tokenizer`, `concurrency`, `isl`, `osl`, `workers`,
`speedup_ratio`, `status`, `req_per_sec`, `output_tok_per_sec`,
`ttft_p50_ms`, `ttft_p99_ms`, `itl_p50_ms`, `itl_p99_ms`, `duration_sec`,
`run_dir`

---

## DGD Templates

The `dgd/templates/` directory contains DynamoGraphDeployment YAML templates
for k8s mode. Template variables (e.g., `${DGD_NAME}`, `${IMAGE}`,
`${DYN_TOKENIZER_BACKEND}`) are substituted by the sweep runner at deploy time.

| Template | Backend | GPU required | Description |
|----------|---------|-------------|-------------|
| `mocker.yaml` | mocker | No | Synthetic backend for isolating frontend/tokenizer overhead. |
| `vllm.yaml` | vLLM | Yes | Real inference backend for end-to-end benchmarking. |

---

## Analysis

Post-sweep analysis scripts live in `scripts/analysis/`:

| Script | Purpose |
|--------|---------|
| `create_report.py` | Generates a per-run observability report from aiperf JSON, Prometheus snapshots, NVTX traces, syscall profiles, and BPF data. |
| `frontend_perf_analysis.py` | Produces scalability curves (TTFT/ITL/throughput vs. concurrency), ISL heatmaps, stage waterfall breakdowns, and regression detection. Supports single-run analysis, A/B comparison, and heatmap generation. |

**Single-run report:**

```bash
python3 scripts/analysis/create_report.py analyze artifacts/sweep_*/mocker_hf_w2_c50_isl512/
```

**A/B comparison:**

```bash
python3 scripts/analysis/frontend_perf_analysis.py compare \
    artifacts/sweep_*/mocker_hf_w2_c50_isl512/ \
    artifacts/sweep_*/mocker_fastokens_w2_c50_isl512/
```
