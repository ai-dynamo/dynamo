# Qwen3.6-35B-A3B-FP8 — 3-way `vllm serve` vs Dynamo benchmark

K8s recipe for benchmarking
[`Qwen/Qwen3.6-35B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
across three configs on the same single-GPU hardware target:

| Config         | Stack         | Multimodal | Frontend-decoding | Embedding cache |
|----------------|---------------|------------|-------------------|-----------------|
| `vllm-serve`   | vanilla vLLM  | n/a        | n/a               | n/a             |
| `dynamo-fd`    | Dynamo + vLLM | on         | on                | off             |
| `dynamo-fd-ec` | Dynamo + vLLM | on         | on                | 8 GiB           |

All three configs share one hardware target (H100 or GB200) chosen at
deploy time via `--hw {h100,gb200}`. See [Hardware targets](#hardware-targets) below.

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

## Quick start

```bash
export NAMESPACE=<your-namespace>
export HW=gb200   # or h100

# Run all three configs sequentially (prep + deploy + bench + retrieve + clean
# per config). Artifacts land under
# ~/workspace/dynamo-tmp/logs/<MM-DD>/qwen36-fp8-${HW}/{vllm-serve,dynamo-fd,dynamo-fd-ec}/.
./run-all-benchmarks.sh -n ${NAMESPACE} --hw ${HW}
```

Each config's `profile_export_aiperf.json` is retrieved into the matching
sub-directory; throughput / TTFT / ITL numbers can be read directly from
that file.

Or step-by-step for a single config:

```bash
./run-benchmark.sh -n ${NAMESPACE} --hw ${HW} --config vllm-serve
./run-benchmark.sh -n ${NAMESPACE} --hw ${HW} --config dynamo-fd
./run-benchmark.sh -n ${NAMESPACE} --hw ${HW} --config dynamo-fd-ec
```

`run-benchmark.sh` accepts `--step {pvc|download|dataset|deploy|bench|retrieve|clean}` for granular control. `pvc`, `download`, and `dataset` are config-agnostic (any `--config` works to run them once).

## Directory layout

```text
qwen3.6-35b/
├── README.md
├── run-benchmark.sh            # Unified driver — branches on --config/--hw
├── run-all-benchmarks.sh       # Sequential 3-config orchestrator
├── perf.yaml                   # Single shared aiperf bench Pod template
├── data-gen-job.yaml           # Sliding-window jsonl generator Job
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

Layout rule: **singletons flatten to root** (`perf.yaml`, `data-gen-job.yaml`); **dirs hold ≥2 files** (`hw/`, `deploy/`); **`model-cache/` is the exception** — a role bucket kept for future model-caching siblings.

The three deploy targets share one `perf.yaml` because the only deltas
across them (pod name, frontend service, run-label) are exported as
`${BENCH_POD}` / `${BENCH_FRONTEND}` / `${BENCH_RUN_LABEL}` by
`run-benchmark.sh` and resolved via `envsubst` at apply time.

## Hardware targets

`hw/h100.env` and `hw/gb200.env` are sibling to the three config
directories and shared across all three. Each file exports three vars
the YAML templates substitute via `envsubst`:

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
| `/perf-cache`                     | `qwen36-bench/perf-cache` | Generated dataset + aiperf artifacts |

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

## aiperf install

We install `aiperf==0.10.0` from PyPI. This release includes
[PR 824](https://github.com/ai-dynamo/aiperf/pull/824)
(`feat(dataset): add session_id to single-turn for causal ordering`),
which makes `single_turn` mode honor `session_id` ordering so
prefix-cache hits across the 8 turns of a given user are real.

The version is pinned in `perf.yaml` and applies identically across
all three configs. Bump the `aiperf==` pin there to roll forward.

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

- Dataset is sliding-window with `window=5`, `turns=8`, `users=30`,
  `image_size=2400x1080`, `user_text_tokens=8000`. Yields 240 requests
  (`users × turns`) over 12 unique images per user
  (`window + turns - 1`). Base64-inlined.
- Each jsonl row carries `session_id=user_<N>`. With aiperf PR 824, the
  `single_turn` dataset type honors session ordering so the 8 turns of
  any one user are sent in causal order, letting prefix-cache hits land.
- The vllm command in `deploy.yaml` uses `--mm-processor-cache-gb 30`
  and `--max-model-len 32768` to handle the 5-image multimodal context
  (mirrors the 397B sweep yaml's settings adapted for 1 GPU).
