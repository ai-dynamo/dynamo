# Qwen3.5-397B-A17B-FP8 — `vllm serve` vs Dynamo frontend-decoding benchmark

K8s recipe for benchmarking
[`Qwen/Qwen3.5-397B-A17B-FP8`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8)
across two configs on a single 8×H100 node (TP=8, expert-parallel,
multimodal):

| Config         | Stack         | Multimodal | Frontend-decoding |
|----------------|---------------|------------|-------------------|
| `vllm-serve`   | vanilla vLLM  | yes        | n/a               |
| `dynamo-fd`    | Dynamo + vLLM | on         | on                |

Flags mirror the `vllm-serve` and `dynamo-fd` configs in
[PR 8235's sweep_397b.yaml](https://github.com/ai-dynamo/dynamo/pull/8235),
with one deliberate divergence: **`--mm-processor-cache-gb` is set to
30 GiB on both** (PR 8235 uses 60 GiB on `dynamo-fd`). Matching the
processor cache across configs makes the only delta `frontend-decoding`,
which is what we're measuring.

## Pre-requisites

1. Kubectl context pointing at a cluster with 8×H100-80GB nodes.
2. A namespace you have write access to (`$NAMESPACE` below).
3. A `shared-model-cache` PVC in that namespace (RWX, ≥600 GiB). If
   your cluster pre-provisions it (common on platform-managed AWS /
   FSx clusters), you don't need to do anything. Otherwise see
   [Storage: shared-model-cache](#storage-shared-model-cache).
4. **Pick a hostname**: find a node with 8 free H100 GPUs and fill it
   into `hw/h100.env` — replace the `<FILL-IN-H100-HOSTNAME>`
   placeholder. The survey query MUST account for Pending pods too on
   kai-scheduler clusters (a node showing `used_gpu=0` from running
   pods can still have another tenant's Pending pod queued for those
   GPUs):
   ```bash
   for n in $(kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-H100-80GB-HBM3 -o name); do
     used=$(kubectl get pods --all-namespaces -o json \
       --field-selector=spec.nodeName=$(basename $n) \
       | jq '[.items[] | select(.status.phase=="Running" or .status.phase=="Pending")
              | .spec.containers[].resources.limits."nvidia.com/gpu" // "0" | tonumber] | add // 0')
     echo "$(basename $n) used=$used"
   done
   ```
5. `envsubst` on the laptop driving the recipe (Ubuntu:
   `apt install gettext-base`; macOS: `brew install gettext`).
6. **HuggingFace token: not required.** `Qwen/Qwen3.5-397B-A17B-FP8`
   is public, so neither the download Job nor the model server needs
   one. To swap in a gated model, uncomment the `hf-token-secret`
   blocks in `model-download.yaml` + `<config>/deploy.yaml` and create:
   ```bash
   kubectl -n "$NAMESPACE" create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="$HF_TOKEN"
   ```

## Quick start

```bash
export NAMESPACE=<your-namespace>

# Run both configs sequentially (prep + deploy + bench + retrieve + clean
# per config). Artifacts land under
# ~/workspace/dynamo-tmp/logs/<MM-DD>/qwen35-fp8-h100/{vllm-serve,dynamo-fd}/.
./run-all-benchmarks.sh -n ${NAMESPACE}

# 2-way comparison:
python3 compare.py ~/workspace/dynamo-tmp/logs/$(date +%m-%d)/qwen35-fp8-h100/
```

Or step-by-step for a single config:

```bash
./run-benchmark.sh -n ${NAMESPACE} --hw h100 --config vllm-serve
./run-benchmark.sh -n ${NAMESPACE} --hw h100 --config dynamo-fd
```

`run-benchmark.sh` accepts `--step {pvc|download|dataset|deploy|bench|retrieve|clean}` for granular control. `pvc`, `download`, and `dataset` are config-agnostic (any `--config` works to run them once).

## Directory layout

```text
qwen3.5-397b-fp8/
├── README.md
├── run-benchmark.sh            # Unified driver — branches on --config/--hw
├── run-all-benchmarks.sh       # Sequential 2-config orchestrator
├── compare.py                  # 2-way comparison (throughput, TTFT, ITL)
├── hw/
│   └── h100.env                # Hardware axis (8×H100, shared across both configs)
├── model-cache/
│   └── model-download.yaml
├── data-gen/
│   └── generate-datasets-job.yaml
├── vllm-serve/
│   ├── deploy.yaml             # Templated (image, nodeSelector, tolerations)
│   └── perf.yaml               # Templated
└── dynamo-fd/
    ├── deploy.yaml             # DynamoGraphDeployment, frontend-decoding ON
    └── perf.yaml
```

## Hardware target

`hw/h100.env` is shared across both configs. It exports three vars
the YAML templates substitute via `envsubst`:

- `VLLM_IMAGE` — `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0`.
- `HW_NODE_SELECTOR` — JSON-flow nodeSelector pinning every pod
  (Frontend, VllmWorker, bench) to one specific 8×H100 host.
- `HW_TOLERATIONS` — JSON-flow toleration array (empty for H100).

Why hostname pinning instead of inter-pod podAffinity: on kai-scheduler
clusters the Frontend + VllmWorker pods of a DGD are gang-scheduled
atomically. `required` podAffinity deadlocks (each pod waits for the
other's node which doesn't exist yet) and `preferred` podAffinity is a
no-op at decision time. A hostname `nodeSelector` on both pods is the
only reliable way to land them on the same node.

If the pinned host runs out of free GPUs at apply time, both pods stay
Pending — re-survey and update `hw/h100.env`.

## Storage: shared-model-cache

The recipe expects a single PVC named `shared-model-cache` (RWX,
≥600 GiB) in the target namespace — typically backed by FSx Lustre on
AWS or any RWX storage class on your cluster. It's mounted at three
locations:

| Mount in pod | subPath | What lives there |
|--------------|---------|------------------|
| `/home/dynamo/.cache/huggingface` | — (root) | Shared HF Hub cache (anything else in the namespace re-uses it) |
| `/home/dynamo/.cache/vllm`        | `qwen35-bench/vllm-cache` | vllm cudagraph + DeepGEMM JIT compilation cache |
| `/perf-cache`                     | `qwen35-bench/perf-cache` | Generated dataset + aiperf artifacts |

The per-recipe subPath prefix `qwen35-bench/` keeps this recipe's
private state from colliding with other recipes (e.g. `qwen36-bench/`,
`qwen3vl30b-bench/`).

The HF cache is mounted at the root so any model already cached in
the namespace is reused. The Qwen3.5-397B-A17B-FP8 download lands in
the standard `hub/models--Qwen--Qwen3.5-397B-A17B-FP8/` directory.

If your cluster doesn't pre-provision `shared-model-cache`, create one
yourself before running the recipe — RWX access mode, ≥600 GiB, pick
an RWX storage class (e.g. `dgxc-enterprise-file` on dgxc, FSx Lustre
on AWS). The recipe will not create it for you; `run-benchmark.sh`
errors out if the PVC is missing.

## aiperf install

We install aiperf from source pinned to a recent `main` SHA that
includes [PR 824](https://github.com/ai-dynamo/aiperf/pull/824)
(`feat(dataset): add session_id to single-turn for causal ordering`).
Our sliding-window dataset writes one row per `(user, turn)` with
`session_id=user_<N>`; PR 824 is what makes aiperf's `single_turn`
mode honor that ordering so prefix-cache hits across turns are real.

The pin lives in `<config>/perf.yaml` (`AIPERF_GIT_REF` env var, identical
across both configs). Bump when you want newer aiperf fixes.

## Naming & ownership

All resources carry a `qwen35-` prefix (per-model) and these labels:

```yaml
labels:
  app.kubernetes.io/name: qwen3.5-397b-fp8
  app.kubernetes.io/managed-by: dynamo-recipe
```

So in a shared namespace you can find this recipe's resources via:

```bash
kubectl -n "$NAMESPACE" get pvc,deploy,job,pod \
  -l app.kubernetes.io/name=qwen3.5-397b-fp8
```

## Notes

- Dataset is sliding-window with `window=5`, `turns=8`, `users=30`,
  `image_size=2400x1080`, `user_text_tokens=8000`. Yields 240 requests
  (`users × turns`) over 12 unique images per user
  (`window + turns - 1`). Base64-inlined.
- Each jsonl row carries `session_id=user_<N>`. With aiperf PR 824, the
  `single_turn` dataset type honors session ordering so the 8 turns of
  any one user are sent in causal order, letting prefix-cache hits land.
- The vllm command in `deploy.yaml` uses `--tensor-parallel-size 8
  --enable-expert-parallel --mm-encoder-tp-mode data
  --mm-processor-cache-gb 30 --max-model-len 32768` to fit
  Qwen3.5-397B-A17B-FP8 across the 8 H100s with multimodal
  context (5×2400x1080 base64 images per turn). NCCL watchdog is
  relaxed to 1800 s on both server and worker pods so DeepGEMM JIT +
  cudagraph capture don't trigger a spurious heartbeat SIGABRT.
- First-time deploy is slow: image pull (~3 min) + 400 GiB weight load
  from FSx (~10-20 min) + torch.compile + cudagraph capture (~5-10 min)
  ≈ 25-40 min total. `progressDeadlineSeconds` is bumped to 2700s
  (45 min) accordingly.
