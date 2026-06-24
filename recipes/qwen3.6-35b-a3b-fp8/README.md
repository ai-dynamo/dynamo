# Qwen3.6-35B-A3B-FP8 — Dynamo deployment recipe

K8s recipe for serving
[`Qwen/Qwen3.6-35B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
on a single GPU (H100 or GB200). It ships three deploy targets that share one
hardware target:

| Config         | Stack         | Multimodal | Frontend-decoding | Embedding cache |
|----------------|---------------|------------|-------------------|-----------------|
| `vllm-serve`   | vanilla vLLM  | n/a        | n/a               | n/a             |
| `dynamo-fd`    | Dynamo + vLLM | on         | on                | off             |
| `dynamo-fd-ec` | Dynamo + vLLM | on         | on                | 8 GiB           |

If you just want a running deployment, follow **[Deploy](#deploy)** below —
that's the standalone install path. To reproduce the 3-way performance
comparison across the configs, see
**[`benchmark/README.md`](benchmark/README.md)**.

## Pre-requisites

1. Kubectl context pointing at a cluster with the right GPUs.
2. A namespace you have write access to (`$NAMESPACE` below).
3. A `shared-model-cache` PVC in that namespace (RWX). If your cluster
   pre-provisions it (common on platform-managed AWS / FSx clusters),
   you don't need to do anything. Otherwise see
   [Storage: shared-model-cache](#storage-shared-model-cache).
4. **Fill in your hostname** in `hw/h100.env` or `hw/gb200.env` —
   replace the `<FILL-IN-…-HOSTNAME>` placeholder. See
   [Hardware targets](#hardware-targets) for the lookup command.
5. `envsubst` on the laptop driving the recipe (Ubuntu:
   `apt install gettext-base`; macOS: `brew install gettext`).
6. **HuggingFace token: not required.** `Qwen/Qwen3.6-35B-A3B-FP8` is
   public (`gated: false`), so neither the download Job nor `vllm serve`
   needs one. To swap in a gated model, uncomment the `hf-token-secret`
   blocks in `model-cache/model-download.yaml` + `deploy/<config>.yaml` and create:
   ```bash
   kubectl -n "$NAMESPACE" create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="$HF_TOKEN"
   ```

## Deploy

This is the standalone install path — no benchmark tooling involved. Pick a
hardware target and a config, download the weights once, then apply the deploy
manifest.

```bash
export NAMESPACE=<your-namespace>
export HW=gb200            # or h100

# Load image tag + node pin + tolerations for the chosen hardware.
set -a; . hw/${HW}.env; set +a

# 1. Download the model into shared-model-cache (one-time, config-agnostic).
kubectl -n "$NAMESPACE" apply -f model-cache/model-download.yaml
kubectl -n "$NAMESPACE" wait --for=condition=Complete \
  job/qwen36-model-download --timeout=3600s

# 2. Deploy one config. envsubst is limited to the hardware vars so the
#    embedded ${MODEL_NAME} etc. stay literal.
envsubst '$VLLM_IMAGE $HW_NODE_SELECTOR $HW_TOLERATIONS' \
  < deploy/dynamo-fd.yaml | kubectl -n "$NAMESPACE" apply -f -
```

Per-config wait + endpoint:

| Config         | Kind                  | Wait command |
|----------------|-----------------------|--------------|
| `vllm-serve`   | Deployment + Service  | `kubectl -n $NAMESPACE rollout status deploy/qwen36-vllm-serve --timeout=1800s` |
| `dynamo-fd`    | DynamoGraphDeployment | `kubectl -n $NAMESPACE wait --for=condition=Ready pod -l nvidia.com/dynamo-graph-deployment-name=qwen36-dynamo-fd --timeout=1500s` |
| `dynamo-fd-ec` | DynamoGraphDeployment | `kubectl -n $NAMESPACE wait --for=condition=Ready pod -l nvidia.com/dynamo-graph-deployment-name=qwen36-dynamo-fd-ec --timeout=1500s` |

The frontend listens on port 8000. For the DGD configs the dynamo operator
auto-creates a `<deploy-name>-frontend` Service (e.g.
`qwen36-dynamo-fd-frontend`); `vllm-serve` exposes the `qwen36-vllm-serve`
Service directly. Verify it serves:

```bash
# DGD example — swap the service name per the table above.
kubectl -n "$NAMESPACE" port-forward svc/qwen36-dynamo-fd-frontend 8000:8000 &
curl -s http://localhost:8000/v1/models | jq '.data[].id'
# → "Qwen/Qwen3.6-35B-A3B-FP8"
```

### Tear down

```bash
# DGD configs:
kubectl -n "$NAMESPACE" delete dynamographdeployment qwen36-dynamo-fd
# vllm-serve:
kubectl -n "$NAMESPACE" delete deploy,service qwen36-vllm-serve
```

The PVC is intentionally left in place so the model isn't re-downloaded. To
wipe everything: `kubectl -n $NAMESPACE delete pvc shared-model-cache`.

## Directory layout

```text
qwen3.6-35b-a3b-fp8/
├── README.md                   # this file — standalone deploy guide
├── hw/                         # per-cluster user state — edit hostname here
│   ├── h100.env
│   └── gb200.env
├── model-cache/                # model-caching subsystem
│   └── model-download.yaml
├── deploy/                     # 3 deploy targets
│   ├── vllm-serve.yaml         # Plain Deployment + Service (baseline)
│   ├── dynamo-fd.yaml          # DynamoGraphDeployment, frontend-decoding ON
│   └── dynamo-fd-ec.yaml       # DynamoGraphDeployment, FD + embedding cache
└── benchmark/                  # perf comparison harness (built on the deploys)
    ├── README.md
    ├── data-gen-job.yaml       # sliding-window jsonl generator Job
    ├── perf.yaml               # shared aiperf bench Pod template
    ├── run-benchmark.sh        # single-config driver
    └── run-all-benchmarks.sh   # sequential 3-config orchestrator
```

`hw/`, `model-cache/`, and `deploy/` are the deployment surface and are shared
by the benchmark harness. Everything benchmark-specific lives under
`benchmark/`.

## Hardware targets

`hw/h100.env` and `hw/gb200.env` are shared across all three configs. Each file
exports three vars the YAML templates substitute via `envsubst`:

- `VLLM_IMAGE` — `nvcr.io/nvidia/ai-dynamo/vllm-runtime:<tag>` (multi-arch
  manifest, same tag works on amd64 / arm64).
- `HW_NODE_SELECTOR` — JSON-flow nodeSelector (currently
  `{"kubernetes.io/hostname":"…"}` for both targets).
- `HW_TOLERATIONS` — JSON-flow toleration array. H100 has `[]`; GB200
  carries the `kubernetes.io/arch=arm64:NoSchedule` toleration.

**Before first use**: edit `hw/h100.env` and `hw/gb200.env` and replace
the `<FILL-IN-…-HOSTNAME>` placeholders with `kubernetes.io/hostname`
values from your cluster:

```bash
# H100
kubectl get nodes -L nvidia.com/gpu.product | awk '/H100/'
# GB200
kubectl get nodes -L kubernetes.io/arch -L nvidia.com/gpu.product \
  | awk '/arm64/ && /GB200/'
```

Adding a new hardware target later is a one-file change in `hw/`.

## Storage: shared-model-cache

The recipe expects a single PVC named `shared-model-cache` (RWX) in
the target namespace — typically backed by FSx Lustre on AWS or any
RWX storage class on your cluster. It's mounted at three locations:

| Mount in pod | subPath | What lives there |
|--------------|---------|------------------|
| `/home/dynamo/.cache/huggingface` | — (root) | Shared HF Hub cache (anything else in the namespace re-uses it) |
| `/home/dynamo/.cache/vllm`        | `qwen36-bench/vllm-cache` | vllm cudagraph compilation cache |
| `/perf-cache`                     | `qwen36-bench/perf-cache` | Generated dataset + aiperf artifacts (benchmark only) |

The per-recipe subPath prefix `qwen36-bench/` keeps this recipe's
private state from colliding with future recipes (e.g.
`qwen3vl30b-bench/`).

The HF cache is mounted at the root so any model already cached in
the namespace is reused. The Qwen3.6-35B-A3B-FP8 download lands in
the standard `hub/models--Qwen--Qwen3.6-35B-A3B-FP8/` directory.

If your cluster doesn't pre-provision `shared-model-cache`, create it
out-of-band before running the recipe, picking an RWX storage class
(e.g. `dgxc-enterprise-file` on dgxc, FSx Lustre on AWS):

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: shared-model-cache
spec:
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: 200Gi
  storageClassName: <your-rwx-storage-class>
```

Prefer RWX/Retain (e.g. FSx Lustre) over RWO/Delete (e.g. EBS) —
RWO EBS volumes get pinned to whichever AZ the first-consumer pod
schedules into, leaving the GPU pod unschedulable if your GPU
nodes live in a different AZ.

## Naming & ownership

All resources carry a `qwen36-` prefix (per-model) and these labels:

```yaml
labels:
  app.kubernetes.io/name: qwen3.6-35b
  app.kubernetes.io/managed-by: dynamo-recipe
```

So in a shared namespace you can find this recipe's resources via:

```bash
kubectl -n "$NAMESPACE" get pvc,deploy,job,pod \
  -l app.kubernetes.io/name=qwen3.6-35b
```

## Notes

- The vllm command in each `deploy/*.yaml` uses `--mm-processor-cache-gb 30`
  and `--max-model-len 32768` to handle the multimodal context on 1 GPU.
- `dynamo-fd` and `dynamo-fd-ec` share the same processor cache size (30 GiB)
  so the only flag that differs between them is the embedding-cache capacity
  — keeping the comparison fair.
- All YAML uses `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1`, pinned in
  `hw/*.env` (`VLLM_IMAGE`) and `benchmark/data-gen-job.yaml`.
