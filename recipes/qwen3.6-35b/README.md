# Qwen3.6-35B-A3B-FP8 — single-GPU recipe

K8s recipe for [`Qwen/Qwen3.6-35B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
on a single GPU. Three deployable variants share one hardware target:

| Config         | Stack         | Multimodal | Frontend-decoding | Embedding cache |
|----------------|---------------|------------|-------------------|-----------------|
| `vllm-serve`   | vanilla vLLM  | n/a        | n/a               | n/a             |
| `dynamo-fd`    | Dynamo + vLLM | on         | on                | off             |
| `dynamo-fd-ec` | Dynamo + vLLM | on         | on                | 8 GiB           |

## Pick a path

- **Production / standalone deploy** → see [`deploy/README.md`](deploy/README.md).
  Stands up one of the three configs on the GPU node. No benchmark
  artifacts, no aiperf, no dataset generation.
- **Performance benchmarking** → see [`benchmark/README.md`](benchmark/README.md).
  Runs aiperf against an already-deployed config with a sliding-window
  multimodal dataset. Layers on top of a deploy.

The two paths share the model-download Job and the hardware env files
at the recipe root, so the same `shared-model-cache` PVC and the same
hostname pin serve both flows.

## Shared pre-requisites

Both paths need:

1. Kubectl context pointing at a cluster with the right GPUs.
2. A namespace you have write access to (`$NAMESPACE` below).
3. A `shared-model-cache` PVC in that namespace (RWX). If your cluster
   pre-provisions it (common on platform-managed AWS / FSx clusters),
   you don't need to do anything. Otherwise see
   [Storage: shared-model-cache](#storage-shared-model-cache).
4. **Fill in your hostname** in `hw/h100.env` or `hw/gb200.env` —
   replace the `<FILL-IN-…-HOSTNAME>` placeholder. See
   [Hardware targets](#hardware-targets) below for the lookup command.
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

## Directory layout

```text
qwen3.6-35b/
├── README.md                   # this file — path picker
├── hw/                         # shared by both paths — edit hostname here
│   ├── h100.env
│   └── gb200.env
├── model-cache/                # shared by both paths
│   └── model-download.yaml
├── deploy/
│   ├── README.md               # standalone deploy guide
│   ├── deploy.sh
│   ├── vllm-serve.yaml         # Plain Deployment + Service (baseline)
│   ├── dynamo-fd.yaml          # DynamoGraphDeployment, frontend-decoding ON
│   └── dynamo-fd-ec.yaml       # DynamoGraphDeployment, FD + embedding cache
└── benchmark/
    ├── README.md               # standalone benchmark guide
    ├── benchmark.sh
    ├── perf.yaml               # aiperf Pod template
    └── data-gen-job.yaml       # sliding-window dataset generator
```

## Hardware targets

`hw/h100.env` and `hw/gb200.env` are shared by both subdirs. Each file
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
RWX storage class on your cluster. Both subdirs mount it; the
per-recipe subPath prefix `qwen36/` keeps this recipe's private state
from colliding with other recipes in the same namespace:

| Mount in pod                       | subPath                | Used by    | What lives there |
|------------------------------------|------------------------|------------|------------------|
| `/home/dynamo/.cache/huggingface`  | — (root)               | deploy     | Shared HF Hub cache (anything else in the namespace re-uses it) |
| `/home/dynamo/.cache/vllm`         | `qwen36/vllm-cache`    | deploy     | vllm cudagraph compilation cache |
| `/perf-cache`                      | `qwen36/perf-cache`    | benchmark  | Generated dataset + aiperf artifacts |

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
