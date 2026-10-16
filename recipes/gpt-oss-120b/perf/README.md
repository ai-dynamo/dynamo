<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# gpt-oss-120b Benchmark Recipe

A single [AIPerf](https://github.com/ai-dynamo/aiperf) trace-replay Job — [perf.yaml](perf.yaml) — covers every gpt-oss-120b variant (agg/disagg × B200/H200). The benchmark is identical across variants; only `ENDPOINT`, `TRACE_FILE`, and `CONCURRENCY` need to change.

The Job waits for `GET /v1/models` on the frontend to return `openai/gpt-oss-120b` (up to ~1h by default), runs a short warmup, then replays the configured trace at a single `CONCURRENCY` value and writes raw artifacts to the shared `model-cache` PVC.

The bench pod is **co-located with the DGD frontend** (`podAffinity` on the frontend's host) so client → server traffic stays on a single node.

## Targeting a variant

Edit the `env` block in [perf.yaml](perf.yaml):

| Variant target       | `ENDPOINT`                                        |
|----------------------|---------------------------------------------------|
| B200 agg, agentic    | `turbo-gptoss-120b-agg-b200-agentic-frontend:8000`|
| H200 agg, agentic    | `turbo-gptoss-120b-agg-h200-agentic-frontend:8000`|
| B200 disagg, agentic | `turbo-gptoss-120b-disagg-b200-agentic-frontend:8000`      |
| H200 disagg, agentic | `turbo-gptoss-120b-disagg-h200-agentic-frontend:8000`      |

All variants serve `openai/gpt-oss-120b`, so any trace can be replayed against any variant by swapping `TRACE_FILE`.

> **Note:** The `podAffinity` in `perf.yaml` matches the `nvidia.com/dynamo-component-type: frontend` +
> `nvidia.com/dynamo-graph-deployment-name` labels. The operator sets these on DGD (agg) frontends, and the disagg deploy
> YAMLs carry the same labels on their Pod, so the one perf Job co-locates with both agg and disagg targets unchanged.

If you run more than one benchmark in the same namespace, also update `metadata.name` / `labels.app` so jobs and artifact directories stay distinct.

## Dataset

The benchmark replays a [Mooncake-format](https://github.com/kvcache-ai/Mooncake) trace via `aiperf --custom-dataset-type mooncake_trace`. Each JSONL line describes one request (`input_length`, `output_length`, `hash_ids`). The traces are token-length-only, so model-agnostic — this recipe reuses the traces shipped with the Kimi-K2.6 recipe in [../../kimi-k2.6/perf/traces/](../../kimi-k2.6/perf/traces/).

> **Note:** The trace files are stored in **Git LFS** — a plain `git clone` leaves ~132-byte pointer files.
> Fetch the actual data before staging:
>
> ```bash
> git lfs install
> git lfs pull --include "recipes/kimi-k2.6/perf/traces/*"
> ```
>
> A real trace file is several MB (`ls -lh`); if the first line starts with `version https://git-lfs...`,
> it is still a pointer.

**Only the agentic 15% subset is needed** — the recipes target the agentic workload (64k/400/90%-KV), and the 15% subset keeps run time short while preserving the KV-hit-rate:

```text
/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl
```

This is the default `TRACE_FILE` in `perf.yaml`. The full trace and 30% subset (and the chat flavour) are available in the same folder for longer runs or experimentation.

> **Note:** The trace was generated for models with a longer context window than gpt-oss-120b. A small, deterministic
> fraction of requests (~7%) has `input_length` > 131072 (the recipes' `--max-model-len`) and is rejected by the server.
> The rejected set is identical on every run, so comparisons across configs and concurrencies remain valid — just expect
> the error count in the aiperf export to be non-zero.

## KV-cache setup

KV-offload defaults differ by SKU: **B200 agg ships prefix-caching only** (net-neutral there) while **H200 agg ships CPU offload ON** (`--kv-transfer-config=$(KV_TRANSFER_CONFIG)`, +9% at the 50-tps floor, quality-neutral). Details:

- **Do not enable LMCache for gpt-oss.** `LMCacheConnectorV1` crashes the engine on the request-completion path
  (`lmcache vllm_v1_adapter request_finished: assert self.lmcache_engine is not None` → `EngineDeadError`) and is not
  `SupportsHMA`, so it force-disables the hybrid KV cache manager — costing KV capacity on gpt-oss sliding-window attention.
- **The working opt-in backend is `SimpleCPUOffloadConnector`** (native, `SupportsHMA` → hybrid KV manager stays on;
  verified coherent and stable: 384-request concurrent soak, 0 crashes). It's pre-wired as `KV_TRANSFER_CONFIG` in the agg
  deploy YAMLs — enable by adding `--kv-transfer-config=$(KV_TRANSFER_CONFIG)` to the worker args (no
  `--disable-hybrid-kv-cache-manager` needed).
- Disagg variants stay NixlConnector-only — combining offload connectors with NIXL breaks the P→D handshake.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy the variant

See instructions in the [README](../README.md).

### 2. Stage the trace on the PVC

Spin up a short-lived helper pod that mounts `model-cache`, then `kubectl cp` the agentic 15% trace in from the Kimi-K2.6 recipe:

```bash
kubectl run pvc-helper -n ${NAMESPACE} \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"model-cache","mountPath":"/model-cache"}]}],"volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"model-cache"}}]}}' \
  --command -- sleep 3600

git lfs pull --include "recipes/kimi-k2.6/perf/traces/*"   # traces are Git LFS objects
kubectl exec -n ${NAMESPACE} pvc-helper -- mkdir -p /model-cache/traces
kubectl cp ../../kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl \
  ${NAMESPACE}/pvc-helper:/model-cache/traces/
```

Keep `pvc-helper` around for fetching artifacts later, or `kubectl delete pod pvc-helper -n ${NAMESPACE}` once you're done staging.

### 3. Run the benchmark

```bash
kubectl apply -f perf.yaml -n ${NAMESPACE}

# Stream logs
kubectl logs -n ${NAMESPACE} -l job-name=turbo-gptoss-120b-bench -f

# Wait for completion (2h hard cap on the Job)
kubectl wait --for=condition=Complete \
  job/turbo-gptoss-120b-bench \
  -n ${NAMESPACE} --timeout=7200s
```

### 4. Fetch artifacts

```bash
kubectl cp ${NAMESPACE}/pvc-helper:/model-cache/perf/<epoch>_turbo-gptoss-120b-bench ./results
```

### 5. Cleanup

```bash
kubectl delete job turbo-gptoss-120b-bench -n ${NAMESPACE}
kubectl delete pod pvc-helper -n ${NAMESPACE}   # if you kept it around
```

## Running a concurrency sweep

`perf.yaml` runs a **single** `CONCURRENCY` value. To measure multiple concurrencies you must clear server state between runs — otherwise residual KV cache / prefix-cache hits from the previous run skew results.

For each concurrency value you want to measure:

```bash
# 1. Delete the previous bench job
kubectl delete job turbo-gptoss-120b-bench -n ${NAMESPACE} --ignore-not-found

# 2. Drop KV / prefix-cache state.
#    agg (DGD): DGDs are Grove-managed (DGD → PodCliqueSet → PodClique → Pod), not
#    Deployments, so `kubectl rollout restart deployment` won't match anything —
#    delete the worker pods directly and let Grove recreate them.
DGD=turbo-gptoss-120b-agg-b200-agentic
kubectl delete pods -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD},nvidia.com/dynamo-component-type=worker
kubectl wait --for=condition=Ready pod -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD},nvidia.com/dynamo-component-type=worker \
  --timeout=900s

#    disagg (also a DGD): delete its worker pods and let the operator recreate them.
# kubectl delete pods -n ${NAMESPACE} -l nvidia.com/dynamo-graph-deployment-name=turbo-gptoss-120b-disagg-b200-agentic,nvidia.com/dynamo-component-type=worker

# 3. Bump CONCURRENCY in perf.yaml, then re-apply
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/turbo-gptoss-120b-bench -n ${NAMESPACE} --timeout=7200s
```

(The bench Job's `wait_for_model_ready` loop handles the worker restart window — it re-polls `/v1/models` until the frontend reports ready again.)

## Tunable environment variables

Edit the `env` block on the `Job` to adjust:

| Variable       | Default                                            | Notes                                                                          |
|----------------|----------------------------------------------------|--------------------------------------------------------------------------------|
| `ENDPOINT`     | `turbo-gptoss-120b-agg-b200-agentic-frontend:8000` | Frontend service:port — change per variant (see [Targeting a variant](#targeting-a-variant)) |
| `TRACE_FILE`   | agentic 15% subset                                 | Swap to the full trace, the chat flavour, or a smaller subset                  |
| `CONCURRENCY`  | `384`                                              | B200 agentic ~50-tps floor; H200 ~128. See [Running a concurrency sweep](#running-a-concurrency-sweep) |
| `TARGET_MODEL` | `openai/gpt-oss-120b`                              | Must match `--served-model-name`                                               |

## Artifacts

Results are written to:

```text
/model-cache/perf/<epoch>_<job-name>/
  warmup/
  gpt-oss-120b_trace_c<concurrency>_<timestamp>/
    profile_export.json
    inputs.json
    ...
```

Per-GPU throughput = `output_token_throughput` / 8 (all variants run on 8 GPUs); the interactivity target is ≥50 tok/s/user.
