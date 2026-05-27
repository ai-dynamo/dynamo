# Qwen3.6-35B-A3B-FP8 — single-GPU deploy recipe

K8s recipe for serving
[`Qwen/Qwen3.6-35B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
on a single GPU. Three deployable variants share one hardware target:

| Config         | Stack         | Multimodal | Frontend-decoding | Embedding cache |
|----------------|---------------|------------|-------------------|-----------------|
| `vllm-serve`   | vanilla vLLM  | n/a        | n/a               | n/a             |
| `dynamo-fd`    | Dynamo + vLLM | on         | on                | off             |
| `dynamo-fd-ec` | Dynamo + vLLM | on         | on                | 8 GiB           |

Pick a hardware target with `--hw {h100,gb200}` and one config at a time.
See [Hardware targets](#hardware-targets) below.

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
   public (`gated: false`), so neither the download Job nor the workers
   need one. To swap in a gated model, uncomment the `hf-token-secret`
   blocks in `model-cache/model-download.yaml` + `deploy/<config>.yaml`
   and create:
   ```bash
   kubectl -n "$NAMESPACE" create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="$HF_TOKEN"
   ```

## Quick start

```bash
export NAMESPACE=<your-namespace>
export HW=gb200   # or h100

# Pick one config and stand it up. `all` runs pvc check → model download
# → deploy. The download Job is config-agnostic, so the next config you
# pick reuses the cached weights.
./deploy.sh -n "$NAMESPACE" --hw "$HW" --config vllm-serve
./deploy.sh -n "$NAMESPACE" --hw "$HW" --config dynamo-fd
./deploy.sh -n "$NAMESPACE" --hw "$HW" --config dynamo-fd-ec
```

What you get per config:

- `vllm-serve` → a `Deployment` + `Service` named `qwen36-vllm-serve`.
  Hit it at `http://qwen36-vllm-serve:8000/v1/...` from inside the
  namespace.
- `dynamo-fd` / `dynamo-fd-ec` → a `DynamoGraphDeployment`. The Dynamo
  operator stands up a `<dgd>-frontend` Service exposing the same
  OpenAI-compatible API on port 8000.

`deploy.sh` accepts `--step {pvc|download|deploy|clean|all}` for
granular control. `pvc` and `download` are config-agnostic (any
`--config` value works to run them once).

## Directory layout

```text
qwen3.6-35b/
├── README.md
├── deploy.sh                   # Deploy driver — branches on --config/--hw
├── hw/                         # Per-cluster user state — edit hostname here
│   ├── h100.env
│   └── gb200.env
├── model-cache/                # Model-caching subsystem
│   └── model-download.yaml
└── deploy/                     # 3 deploy targets — grouped because >1 sibling
    ├── vllm-serve.yaml         # Plain Deployment + Service (baseline)
    ├── dynamo-fd.yaml          # DynamoGraphDeployment, frontend-decoding ON
    └── dynamo-fd-ec.yaml       # DynamoGraphDeployment, FD + embedding cache
```

Layout rule: **singletons flatten to root**; **dirs hold ≥2 files**
(`hw/`, `deploy/`); **`model-cache/` is the exception** — a role bucket
kept for future model-caching siblings.

## Hardware targets

`hw/h100.env` and `hw/gb200.env` are shared across all three deploy
targets. Each file exports three vars the YAML templates substitute via
`envsubst`:

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
RWX storage class on your cluster. It's mounted at two locations:

| Mount in pod | subPath | What lives there |
|--------------|---------|------------------|
| `/home/dynamo/.cache/huggingface` | — (root) | Shared HF Hub cache (anything else in the namespace re-uses it) |
| `/home/dynamo/.cache/vllm`        | `qwen36-deploy/vllm-cache` | vllm cudagraph compilation cache |

The per-recipe subPath prefix `qwen36-deploy/` keeps this recipe's
private state from colliding with other recipes in the same namespace.

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

## Cleanup

```bash
./deploy.sh -n "$NAMESPACE" --hw "$HW" --config <name> --step clean
```

Deletes the Deployment+Service (for `vllm-serve`) or the
DynamoGraphDeployment (for `dynamo-fd` / `dynamo-fd-ec`). PVCs are
intentionally left intact so the cached weights survive across configs.
To wipe everything:

```bash
kubectl -n "$NAMESPACE" delete pvc shared-model-cache
```

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

- The vllm command in `deploy/*.yaml` uses `--mm-processor-cache-gb 30`
  and `--max-model-len 32768` to handle multimodal contexts up to 5
  images per request.
