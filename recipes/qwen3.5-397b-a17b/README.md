<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.5-397B-A17B-FP8 — 1P1D Disaggregation (K8s)

Prefill/decode (1P1D) disaggregation benchmark for **`Qwen/Qwen3.5-397B-A17B-FP8`**
(397B total / 17B active MoE, **hybrid Gated-Delta-Net + full-attention**) on
Dynamo + vLLM, with an aggregated single-node baseline for comparison.

## Why this needs vLLM 0.22.0 (Dynamo `main`)

P/D disagg of a hybrid GDN model requires the decode worker to receive the GDN
**conv/recurrent state** over the NIXL connector — not just the attention KV
cache. That support is vLLM **PR #41869** (*"PD disagg with NIXL Connector: GDN
support (Qwen3.5)"*), which is in **vLLM 0.22.0 but NOT 0.20.1**. Dynamo
`release/1.2.0` pins 0.20.1; **Dynamo `main` pins `vllm==0.22.0` + `nixl==1.1.0`**.
So the recipe runs on an image built from Dynamo `main`.

Corruption of the GDN state over NIXL is **silent** — the server stays healthy
and emits tokens, they are just wrong past the prefill boundary. The GSM8K
**accuracy canary** (below) is the real correctness gate, not a crash-free run.

## Topology

FP8 weights ≈ 397 GB fit **TP8 on one 8×H100 node** (~50 GB/GPU), so each worker
is a single-node TP8 pod:

| Config   | Services                                   | Nodes |
|----------|--------------------------------------------|-------|
| `disagg` | Frontend + Prefill(TP8) + Decode(TP8)      | **2** |
| `agg`    | Frontend + single Worker(TP8)              | **1** |

No `multinode.nodeCount` — two `gpu:"8"` workers land on separate 8-GPU nodes by
scheduler placement. Prefill carries `--disaggregation-mode prefill` +
`--kv-transfer-config` (NixlConnector); decode omits `--disaggregation-mode`.

## Prerequisites

1. **Runtime image.** `hw/h100.env` sets `VLLM_IMAGE` to a runtime image built
   from Dynamo `main` (vLLM 0.22.0 + nixl 1.1.0) — default
   `nvcr.io/nvstaging/ai-dynamo/vllm-runtime:qiwa-release-vllm-x86-06-03`, built
   via `build_image(local_path=<dynamo-main>, framework=vllm, target=runtime,
   platform=linux/amd64, push=True)`. No public release carries vLLM 0.22.0
   yet; point `VLLM_IMAGE` at your own tag.
2. **`shared-model-cache` PVC** pre-provisioned (RWX, FSx Lustre), **≥500 GB
   free** for the FP8 checkpoint. Already provisioned on `aws-dev-02` (ns `qiwa`).
3. **Image pull secret** for the private image: a `dockerconfigjson` secret in
   the namespace (name in `hw/h100.env` `IMAGE_PULL_SECRET`, default
   `ngc-pull-secret`). DGD pods run under an operator-stamped SA
   (`<dgd>-k8s-service-discovery`, **not** `default`), so that SA is patched
   after deploy — see Usage step 2.
4. **HF token** — only if the checkpoint is gated: create `hf-token-secret`
   (key `HF_TOKEN`) and uncomment the `envFrom` block in
   `model-cache/model-download.yaml`. (`Qwen/Qwen3.5-397B-A17B-FP8` is public.)

## Usage

All commands assume `kubectl -n <ns>` (e.g. `-n qiwa`). The deploy / benchmark /
accuracy YAMLs are templated — source the hardware env, then render each with
`envsubst` before applying:

```bash
set -a; . hw/h100.env; set +a   # VLLM_IMAGE, HW_NODE_SELECTOR, HW_TOLERATIONS, IMAGE_PULL_SECRET
```

**1. Download the model** (idempotent, ~400 GB — skip if already on the PVC):
```bash
kubectl -n <ns> apply -f model-cache/model-download.yaml
kubectl -n <ns> wait --for=condition=Complete job/qwen35-model-download --timeout=3600s
```

**2. Deploy** a config (`disagg` shown; baseline = `qwen35-agg` + `deploy/agg.yaml`):
```bash
DGD=qwen35-disagg
envsubst '$VLLM_IMAGE $HW_NODE_SELECTOR $HW_TOLERATIONS' \
  < deploy/disagg.yaml | kubectl -n <ns> apply -f -

# Operator stamps a dedicated SA ~5-10s later — patch it for the private image,
# then recycle pods so they pick up the pull secret:
until kubectl -n <ns> get sa ${DGD}-k8s-service-discovery >/dev/null 2>&1; do sleep 2; done
kubectl -n <ns> patch sa ${DGD}-k8s-service-discovery \
  -p "{\"imagePullSecrets\":[{\"name\":\"${IMAGE_PULL_SECRET}\"}]}"
kubectl -n <ns> delete pod -l nvidia.com/dynamo-graph-deployment-name=${DGD} --wait=false

# Wait for Ready (397B FP8 load + GDN JIT warmup is slow):
kubectl -n <ns> wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD},nvidia.com/dynamo-component-type=worker \
  --timeout=2400s
```

**3. Benchmark** (8K ISL / 3K OSL):
```bash
export BENCH_POD=qwen35-disagg-bench BENCH_FRONTEND=qwen35-disagg-frontend BENCH_RUN_LABEL=disagg
envsubst '$HW_NODE_SELECTOR $HW_TOLERATIONS $BENCH_POD $BENCH_FRONTEND $BENCH_RUN_LABEL' \
  < benchmark/8k-3k.yaml | kubectl -n <ns> apply -f -
kubectl -n <ns> logs -f qwen35-disagg-bench    # aiperf table prints at the end
```

**4. Accuracy canary** (GSM8K — the correctness gate):
```bash
export BENCH_ACC_POD=qwen35-disagg-acc BENCH_FRONTEND=qwen35-disagg-frontend BENCH_RUN_LABEL=disagg
envsubst '$HW_NODE_SELECTOR $HW_TOLERATIONS $BENCH_ACC_POD $BENCH_FRONTEND $BENCH_RUN_LABEL' \
  < accuracy-job.yaml | kubectl -n <ns> apply -f -
kubectl -n <ns> logs -f qwen35-disagg-acc
```

Run 2-4 for **both** `disagg` and `agg`, then compare GSM8K pass rates:

- `disagg ≈ agg` → NIXL GDN-state transfer is healthy ✅
- `disagg` materially below `agg` (>2-3 % abs) → PR #41869 path misbehaving ❌

GDN-state corruption over NIXL is **silent** (healthy server, wrong tokens past
the prefill boundary), so the disagg-vs-agg accuracy delta — not a crash-free
run — is the real gate.

**5. Retrieve / clean:**
```bash
kubectl -n <ns> cp qwen35-disagg-bench:/perf-cache/artifacts/qwen35_fp8/disagg ./disagg-results
kubectl -n <ns> delete -f deploy/disagg.yaml      # PVC (model cache) is left intact
```

## NIXL handshake

After deploy, both worker pods must be `Ready` and a first request must return a
token — that proves the prefill→decode KV + GDN-conv-state transfer works. If
decode **hangs with no first token**, NIXL side-channel host auto-detection
picked an unroutable NIC; set `VLLM_NIXL_SIDE_CHANNEL_HOST` (commented stub in
`deploy/disagg.yaml`).

## Storage layout (subPath `qwen35-bench/perf-cache`)

- Model weights: PVC root → `/home/dynamo/.cache/huggingface/hub/…` (workers
  mount the root with `HF_HUB_OFFLINE=1`).
- Bench artifacts: `/perf-cache/artifacts/qwen35_fp8/<cfg>/`.
- Accuracy results: `/perf-cache/accuracy/<cfg>/`.

## Scheduling

`hw/h100.env` sets `HW_NODE_SELECTOR='{}'` (empty), so the scheduler places the
prefill and decode pods on separate 8-GPU nodes. Do **not** pin both to one node
— they won't both fit. On contended kai-scheduler clusters gang-scheduling can
stall with `no nodes with enough resources` (it counts other tenants' **Pending**
pods too); set `HW_NODE_SELECTOR` to a specific free-node hostname per worker if
needed.

## Tunables (defaults in the YAMLs)

| Knob | Default | Notes |
|------|---------|-------|
| `--gpu-memory-utilization` | `0.90` | 397B FP8 is tight — drop to `0.85` if OOM at load |
| `--max-model-len` | `16384` | must exceed ISL+OSL (8K+3K); smaller = more KV headroom |
| `--block-size` + `--no-enable-prefix-caching` | `128`, off (disagg) | conservative hybrid/NIXL path (keeps `mamba_cache_mode=none`); agg uses prefix caching ON |
| `--enable-expert-parallel` | off | TP8 already shards the A17B experts; EP is a later throughput experiment — add to both workers |
| `benchmark/8k-3k.yaml` `ISL_MEAN`/`OSL_MEAN`/`CONCURRENCY`/`REQUEST_COUNT` | 8000/3000/20/200 | aiperf synthetic text load (8K ISL / 3K OSL) |
| `hw/h100.env` `VLLM_IMAGE` / `HW_NODE_SELECTOR` / `IMAGE_PULL_SECRET` | nvstaging tag / `{}` / `ngc-pull-secret` | image, node placement, pull secret |

## Files

```
qwen3.5-397b-a17b/
├── README.md
├── hw/
│   └── h100.env                # VLLM_IMAGE, HW_NODE_SELECTOR, IMAGE_PULL_SECRET
├── deploy/
│   ├── disagg.yaml             # DGD: Frontend + Prefill(TP8) + Decode(TP8)
│   └── agg.yaml                # DGD: Frontend + Worker(TP8)
├── benchmark/
│   └── 8k-3k.yaml              # aiperf bench Pod (synthetic 8K ISL / 3K OSL)
├── accuracy-job.yaml           # GSM8K canary Pod
└── model-cache/
    └── model-download.yaml
```
