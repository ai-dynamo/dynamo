# Qwen3.5-397B-A17B-FP8 — `vllm serve` vs Dynamo frontend-decoding benchmark

K8s recipe for benchmarking
[`Qwen/Qwen3.5-397B-A17B-FP8`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8)
across two configs on a single 8×H100 node (TP=8, expert-parallel,
multimodal):

| Config         | Stack         | Multimodal | Frontend-decoding |
|----------------|---------------|------------|-------------------|
| `vllm-serve`   | vanilla vLLM  | yes        | n/a               |
| `dynamo-fd`    | Dynamo + vLLM | on         | on                |

**`--mm-processor-cache-gb` is set to 30 GiB on both configs.**
Matching the processor cache across configs makes the only delta
`frontend-decoding`, which is what we're measuring.

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
   blocks in `model-download.yaml` + `deploy-<config>.yaml` and create:
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
./benchmark/run-all-benchmarks.sh -n ${NAMESPACE}
```

Each config's `profile_export_aiperf.json` is the contract output;
extract numbers or run analysis tooling over them separately
(deliberately not shipped in the recipe).

Or step-by-step for a single config:

```bash
./benchmark/run-benchmark.sh -n ${NAMESPACE} --hw h100 --config vllm-serve
./benchmark/run-benchmark.sh -n ${NAMESPACE} --hw h100 --config dynamo-fd
```

`benchmark/run-benchmark.sh` accepts `--step {pvc|download|dataset|deploy|bench|retrieve|clean}` for granular control. `pvc`, `download`, and `dataset` are config-agnostic (any `--config` works to run them once).

## Directory layout

```text
qwen3.5-397b-fp8/
├── README.md
├── model-download.yaml         # Job: hf download into shared HF cache
├── deploy-vllm-serve.yaml      # Deployment + Service (vanilla vLLM)
├── deploy-dynamo-fd.yaml       # DynamoGraphDeployment (Dynamo + FD)
├── hw/
│   └── h100.env                # Hardware axis (8×H100, shared across both configs)
└── benchmark/
    ├── run-all-benchmarks.sh   # Sequential 2-config orchestrator
    ├── run-benchmark.sh        # Per-config driver (branches on --config/--hw)
    ├── perf.yaml               # Shared aiperf bench Pod template
    └── data-gen-job.yaml       # Job: sliding-window jsonl generator
```

Recipe root holds "what gets deployed" — model server templates, HF
cache prep, hardware config. `benchmark/` holds "how we measure it" —
the aiperf bench Pod, the dataset generator, and the driver scripts.

The shared `benchmark/perf.yaml` is parameterized via envsubst on three
per-config vars exported by `benchmark/run-benchmark.sh`'s case statement:
`$BENCH_POD`, `$BENCH_FRONTEND` (Service the bench Pod hits), and
`$BENCH_RUN_LABEL` (sub-dir under `/perf-cache/artifacts/qwen35_fp8/`).
Only `deploy-<config>.yaml` legitimately differs per config.

## Hardware target

`hw/h100.env` is shared across both configs. It exports three vars
the YAML templates substitute via `envsubst`:

- `VLLM_IMAGE` — `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0`.
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
| `/perf-cache`                     | `qwen35-bench/perf-cache` | Generated dataset + aiperf artifacts (artifacts are run-isolated, see below) |

The per-recipe subPath prefix `qwen35-bench/` keeps this recipe's
private state from colliding with other recipes (e.g. `qwen36-bench/`,
`qwen3vl30b-bench/`).

The HF cache is mounted at the root so any model already cached in
the namespace is reused. The Qwen3.5-397B-A17B-FP8 download lands in
the standard `hub/models--Qwen--Qwen3.5-397B-A17B-FP8/` directory.

### Run isolation (aiperf artifacts)

The per-namespace `shared-model-cache` PVCs are distinct PV names over
**one** FSx Lustre filesystem, so a fixed artifact path is shared by
every namespace on the cluster. Writing each run to a constant
`.../qwen35_fp8/<config>/` therefore lets two runs — even in different
namespaces — silently read and clobber each other's
`profile_export_aiperf.json`. To prevent that, the bench writes to a
run-isolated path:

```text
/perf-cache/artifacts/qwen35_fp8/<namespace>/<run-id>/<config>/
```

`<run-id>` comes from `--run-id` / `$RUN_ID`; `run-all-benchmarks.sh`
pins one id per pass (so both configs group under it and each step's
`retrieve` agrees with its `bench`). A standalone `bench` mints a
timestamp; a standalone `retrieve` with no id discovers the newest run
for that `(namespace, config)`. Model weights and the generated dataset
are intentionally **not** isolated — they're content-deterministic and
shared by design.

If your cluster doesn't pre-provision `shared-model-cache`, create
one yourself before running the recipe. The driver's `pvc()` step
asserts the PVC exists and errors out otherwise — it deliberately
does NOT apply a stub manifest. Copy-paste, swap the
`storageClassName` for an RWX/Retain class on your cluster (e.g.
`dgxc-enterprise-file` on dgxc, an FSx Lustre class on AWS — never
`ebs`, which is RWO + AZ-pinned), and apply:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: shared-model-cache
spec:
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: 600Gi
  storageClassName: <your-rwx-storage-class>
```

## aiperf install

The bench pod installs aiperf from a pinned PyPI release
(`AIPERF_VERSION` env var in `benchmark/perf.yaml`, default `0.9.0`).
Use 0.8.0+: that's when `single_turn` mode gained support for
`session_id` ordering, which is what ensures the 3 turns of any one
user are issued in causal order so prefix-cache hits across turns land.

Bump `AIPERF_VERSION` in `benchmark/perf.yaml` when you want newer aiperf
fixes.

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

- Dataset is sliding-window with `window=5`, `turns=3`, `users=30`,
  `image_size=2400x1080`, `user_text_tokens=8000`. Yields 90 requests
  (`users × turns`) over 7 unique images per user
  (`window + turns - 1`). Base64-inlined. 3 turns keeps the warm-request
  fraction at 2/3 (turn-1 cold, turns 2-3 reuse 4-of-5 images), so the
  benchmark exercises fresh-image preprocessing rather than a
  cache-saturated steady state.
- Each jsonl row carries `session_id=user_<N>`. aiperf's `single_turn`
  dataset type honors session ordering so the 3 turns of any one user
  are sent in causal order, letting prefix-cache hits land.
- The vllm command in `deploy-<config>.yaml` uses `--tensor-parallel-size 8
  --enable-expert-parallel --mm-encoder-tp-mode data
  --mm-processor-cache-gb 30 --max-model-len 32768
  --gpu-memory-utilization 0.85` to fit Qwen3.5-397B-A17B-FP8
  across the 8 H100s with multimodal context (5×2400x1080 base64
  images per turn). `--gpu-memory-utilization` is deliberately
  matched at 0.85 across both configs — the obvious move is to set
  vllm-serve to 0.90 (it has no encoder cache consumer) but matching
  keeps the comparison apples-to-apples since the headline metrics
  are throughput / prefill-bound, not VRAM-bound. NCCL watchdog is
  relaxed to 1800 s on both server and worker pods so DeepGEMM JIT +
  cudagraph capture don't trigger a spurious heartbeat SIGABRT.
- First-time deploy is slow: image pull (~3 min) + 400 GiB weight load
  from FSx (~10-20 min) + torch.compile + cudagraph capture (~5-10 min)
  ≈ 25-40 min total. `progressDeadlineSeconds` is bumped to 2700s
  (45 min) accordingly.
