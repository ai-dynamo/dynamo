<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1-Flash Recipes

Dynamo deployments of `deepseek-ai/DeepSeek-V4.1-Flash`, a Mixture-of-Experts
model serving up to 1,048,576 tokens of context.

Six targets across two frameworks. The **vLLM** targets are performance-tuned
and benchmarked on an agentic trace, on both B200 and GB200. The **SGLang**
targets are the original day-0 deployment on GB200 and carry no performance
claim.

Refer to the [documentation](https://docs.nvidia.com/dynamo/dev/recipes/deepseek-v4-1-flash)
for the rendered guide.

## vLLM configurations

All four targets share this much: TP4 with expert parallelism and EPLB over
`torch_gloo`, the `deep_gemm_mega_moe` MoE backend, MXFP4 experts with an
`mxfp4` sparse-indexer KV cache, the engine-resolved `FLASHMLA_SPARSE_DSV41`
attention backend with DeepSeek's `fp8_ds_mla` KV format at block size 128,
DSpark speculative decoding with 3 draft tokens and adaptive verification off,
prefix caching with a 1024 retention interval, NUMA binding on, text-only
serving, and 1,048,576 tokens of context. Engine defaults are kept for
`max-num-seqs` and `gpu-memory-utilization`, and no KV offloading is used.

What differs:

|                                | B200 aggregated | B200 disaggregated | GB200 aggregated | GB200 disaggregated |
| ------------------------------ | --- | --- | --- | --- |
| **Recipe**                     | [`vllm/agg-b200-agentic`](vllm/agg-b200-agentic/deploy-generic.yaml) | [`vllm/disagg-b200-agentic`](vllm/disagg-b200-agentic/deploy-generic.yaml) | [`vllm/agg-gb200-agentic`](vllm/agg-gb200-agentic/deploy-generic.yaml) | [`vllm/disagg-gb200-agentic`](vllm/disagg-gb200-agentic/deploy-generic.yaml) |
| **GPUs**                       | 8 (2 workers) | 12 (2P1D) | 8 (2 workers) | 12 (2P1D) |
| **Architecture**               | amd64 | amd64 | arm64 | arm64 |
| **Max batched tokens**         | Engine default | 32,768 prefill | Engine default | 32,768 prefill |
| **Long-prefill threshold**     | Not set | 4,096 prefill only | Not set | 4,096 prefill only |
| **CUDA graph capture size**    | 512 | 512 prefill / 1,024 decode | 512 | 512 prefill / 1,024 decode |
| **Routing**                    | KV-aware, decode-load weight 50 | KV-aware | KV-aware, decode-load weight 50 | KV-aware |
| **Conditional disaggregation** | N/A | Off (unset) | N/A | Off (unset) |
| **KV transfer**                | N/A | NIXL/UCX, `num_threads` 4 | N/A | NIXL/UCX, engine defaults |
| **UCX transports**             | N/A | `rc_x,rc,cuda_copy,cuda_ipc,tcp` | N/A | `cuda_copy,cuda_ipc,tcp,rc` |
| **UCX rendezvous**             | N/A | `get_zcopy`, threshold 0 | N/A | Engine defaults |
| **Multi-node NVLink**          | N/A | Off (`MNNVL=n`, memtype cache on) | N/A | On (`MNNVL=y`, memtype cache off) |
| **Clique requirement**         | N/A | None | N/A | All pools in one clique, via ComputeDomain |
| **RDMA request**               | None | `rdma/ib: 4` per worker | None | `rdma/ib: 4` per worker |
| **Provider variant**           | — | `deploy-ib-device-pin.yaml` | — | `deploy-gke-rdma.yaml` |

The two aggregated targets are the same serving configuration: only the
architecture label, the GPU product label, and the deployment name differ.

## SGLang configurations

|                          | GB200 aggregated | GB200 disaggregated |
| ------------------------ | --- | --- |
| **Recipe**               | [`sglang/agg-gb200`](sglang/agg-gb200/deploy.yaml) | [`sglang/disagg-gb200`](sglang/disagg-gb200/deploy-generic.yaml) |
| **GPUs**                 | 8 (2 workers) | 8 (1P1D) |
| **Parallelism**          | TP4, EP4 | TP4, EP4 per role |
| **Precision**            | FP8 dense, FP4 experts, FP8 KV | FP8 dense, FP4 experts, FP8 KV |
| **Page size**            | 256 | 256 |
| **Memory fraction**      | 0.8 | 0.8 |
| **Max running requests** | 256 | 256 |
| **Max prefill tokens**   | Engine default | 16,384 (prefill role) |
| **Speculative decoding** | DSpark, block size 5 | None — SGLang refuses it under disaggregation |
| **Decode CUDA graph**    | 64 | Engine default |
| **KV transfer**          | N/A | Mooncake over TCP, or RDMA on GKE |
| **Context length**       | 1,048,576 | 1,048,576 |

## Features

Every vLLM target sets `--dyn-reasoning-parser deepseek_v41` paired with the
engine-side `--reasoning-parser`, `--dyn-tool-call-parser deepseek_v41`, and
`--dyn-enable-structural-tag`, so reasoning, tool calling, and structured
output all work. `indexer_kv_dtype` accepts only `fp8` and `mxfp4`, and `mxfp4`
requires compute capability 10.0. The checkpoint accepts images; every recipe
here serves text only.

KV-aware routing is wired end to end on the vLLM targets. On the SGLang
targets the aggregated recipe publishes KV events without
`enable_kv_cache_events` and the disaggregated recipe publishes none, so
`DYN_ROUTER_MODE=kv` degrades to load-based routing there.

## Prerequisites

- A Kubernetes cluster with the Dynamo platform installed.
- A ReadWriteMany PVC named `shared-model-cache` with at least 1000Gi.
- Access to `deepseek-ai/DeepSeek-V4.1-Flash` — 510 GB over 48 shards.
- 512 GiB of host memory per worker, for the pinned host-side tables.
- Disaggregated vLLM targets: four RDMA devices per worker (`rdma/ib: 4`).
- `vllm/disagg-gb200-agentic` and `sglang/disagg-gb200`: the NVIDIA DRA driver
  with ComputeDomain support, and all pools in one NVLink clique.

## Quick start

### 1. Namespace and token secret

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" -n ${NAMESPACE}
```

### 2. Storage and weights

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

Edit `storageClassName` in `model-cache/model-cache.yaml` first. The download
Job pins snapshot `dba1be0a40aa45a94ad051997016db3960a90277`.

### 3. Deploy

```bash
# vLLM, pick one
kubectl apply -f vllm/agg-b200-agentic/deploy-generic.yaml     -n ${NAMESPACE}
kubectl apply -f vllm/agg-gb200-agentic/deploy-generic.yaml    -n ${NAMESPACE}
kubectl apply -f vllm/disagg-b200-agentic/deploy-generic.yaml  -n ${NAMESPACE}
kubectl apply -f vllm/disagg-gb200-agentic/deploy-generic.yaml -n ${NAMESPACE}
```

Provider variants: `vllm/disagg-gb200-agentic/deploy-gke-rdma.yaml` for GKE
multi-network RDMA, and `vllm/disagg-b200-agentic/deploy-ib-device-pin.yaml`
to pin `UCX_NET_DEVICES` to named HCAs.

### 4. Smoke test

```bash
kubectl port-forward svc/dsv41-flash-vllm-b200-agg-frontend 8000:8000 -n ${NAMESPACE}

curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash",
       "messages":[{"role":"user","content":"Reply with exactly: READY"}],
       "temperature":0,"max_tokens":512}' | jq '.choices[0].message'
```

Read the message body, not the status code: on a disaggregated target a KV
transfer failure still returns HTTP 200 with `content: null`. Give `max_tokens`
real room — the model reasons first, and a small budget returns `content: null`
with `finish_reason: length`, which reads as a broken deployment and is not.

### 5. Benchmark

See [`perf/README.md`](perf/README.md).

## Performance results

Agentic Mooncake trace, 3,541 requests, input median ~67,600 tokens and output
median ~400, ~90 percent designed KV reuse. Each row is that target's best
SLO-clearing operating point, not the peak of its throughput curve. SLO floors,
both p50: per-user output >= 50 tok/s, time to first token (TTFT) < 5 s.

| Target | Concurrency | tok/s/GPU | Total tok/s | tok/s/user p50 | TTFT p50 | ITL p50 | ITL p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B200 aggregated | 88 | 488.9 | 3,911 | 54.65 | **210 ms** | 18.30 ms | 123.67 ms |
| B200 disaggregated | 160 | 610.8 | 7,329 | 52.14 | 2,389 ms | 19.18 ms | **37.97 ms** |
| GB200 aggregated | 88 | 533.2 | 4,266 | 54.23 | **296 ms** | 18.44 ms | 128.36 ms |
| GB200 disaggregated | 168 | **694.7** | **8,336** | 59.86 | 3,237 ms | 16.71 ms | **36.16 ms** |

Disaggregation gains about 30 percent per GPU on GB200 and 25 percent on B200,
on 50 percent more hardware, and cuts the inter-token latency (ITL) tail by
more than 3x. It pays for that with an order of magnitude higher TTFT. The two
topologies are not interchangeable; pick by the TTFT the workload accepts.

GB200 measures about 9 percent above B200 on the identical aggregated
configuration. The cause is KV capacity, not compute: a B200 TP4 worker holds
roughly 19 percent fewer KV tokens and prefix-cache hit rate follows, and on a
67,600-token input median each miss is a full prefill.

> [!IMPORTANT]
> Throughput on this workload repeats to about 3 percent across identical runs.
> Do not read a smaller difference as a result. Latency percentiles are far
> more stable.

Accuracy for this checkpoint is published on its
[model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). These
recipes publish serving performance.

## Configuration notes

**The attention path is engine-resolved, and the recipes leave it alone.**
vLLM selects `FLASHMLA_SPARSE_DSV41` from the checkpoint on Blackwell, which
brings DeepSeek's `fp8_ds_mla` KV cache format and fixes the KV block size at
128. Two related settings are overridden rather than left at their defaults:
`--moe-backend deep_gemm_mega_moe` (resolved default
`FLASHINFER_TRTLLM_MXFP4_MXFP8`) and `indexer_kv_dtype: mxfp4` (resolved
default `fp8`).

**Expert parallelism is a hard prerequisite, not a tuning knob.** Every member
of the MegaMoE backend family refuses to load without `--enable-expert-parallel`.

**EPLB names its communicator explicitly.** `--eplb-config
'{"communicator":"torch_gloo"}'` is set because the default of `None` prefers
NIXL, which is not present in this image.

**Prefix-cache retention is the largest single lever.** The resolved default of
0 tells the sliding-window group to retain only a 160-token tail at the
previous prompt end, and the reconciled prefix hit is the minimum across
groups, so that tail caps every request. 1024 keeps one tail per 1024 tokens.

**The decode pool captures larger CUDA graphs than prefill.** At 512 the decode
batch caps at 128 sequences, which the decode pool exceeds at the shipped
concurrency; prefill stays at 512.

**The disaggregated prefill knobs target the tail, not the median.** Raising the
batch budget to 32,768 stops one request from serializing the pool, and the
4,096 long-prefill threshold stops one giant request from taking a whole step.
Queued requests still fill the remainder of the step, so this redistributes
wait rather than shrinking the batch. 8,192 is the tail-safe alternative, about
3 percent slower with a shorter p99 TTFT.

**2 prefill to 1 decode is balanced, not a free parameter.** Both sides
saturate together at the operating point: the prefill queue grows into TTFT,
which still holds headroom, while decode ITL grows into the per-user token
rate, which is what binds. Other ratios measured worse or the same per GPU —
scaling moves the absolute number, not the number per GPU.

**The speculative config must be identical on both roles.** The draft head
state and KV layout have to agree across the NIXL handoff.

**`num_threads` bounds outstanding KV transfers** before they queue behind the
UCX worker threads. B200 uses 4, which beat 8 at this ratio; the answer inverts
at 1P1D, so re-measure if the ratio changes. GB200 uses the NIXL defaults.

**`tcp` stays in `UCX_TLS` on both SKUs.** `cuda_ipc` and `cuda_copy` are
memory-type transports only and cannot carry NIXL's UCX active-message control
plane, so removing `tcp` breaks the handshake rather than just the data path.
`rc` is what carries KV over RDMA.

**`UCX_IB_GID_INDEX` is set only on the GKE variant.** Index 3 is the RoCEv2
convention, which applies to GKE's RDMA networks and does not apply where ports
report `link_layer: InfiniBand`.

**No `NCCL_*` tuning on GB200, deliberately.** KV moves over NIXL/UCX; NCCL
only forms the intra-worker TP4 group inside one node, and enabling
`NCCL_MNNVL_ENABLE`, `NCCL_NVLS_ENABLE`, or `NCCL_P2P_LEVEL` broke it with
`unhandled system error` in `ncclCommInitRank`.

**Conditional disaggregation is left unset rather than `false`.** Removal is
the documented default. The valve keys on KV cache overlap, so it diverts
cached requests straight to decode, which has no spare compute on this
workload.

**The frontend weights decode load at 50.** The default of 0.0 optimizes purely
for prefix overlap, which concentrates work on one worker past concurrency 80.

## Operational settings

These are not performance knobs, but each one is deliberate:

| Setting | Value | Why |
| --- | --- | --- |
| `startupProbe` | 30 s × 120 | First boot measures ~46 min, dominated by a silent host-memory load with the GPUs at 0 percent |
| `livenessProbe` / `readinessProbe` | `failureThreshold: 60` | The operator default restarts a worker after one failed probe; the failure mode here is a 503 while the scheduler is inside a forward pass |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 3600 | The 600 s default is far short of this checkpoint's load time |
| `DYN_HEALTH_CHECK_REQUEST_TIMEOUT` | 300 | The 3 s default fails healthy workers: the health canary queues behind real work |
| `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN` | 65536 | The default profiles at 8,192 tokens, against a 67,600-token input median |
| `memory` request, no limit | 512Gi | A hard limit turns a pinned-allocation spike into an OOMKill rather than reclaim |
| `securityContext.runAsUser` | 0 | The FlashInfer FP4 MoE JIT writes cubins into a root-owned site-packages directory at startup |

## Editing and rendering

The vLLM recipes are Kustomize matrices. Edit `kustomize/base/` or the
Components, never a `deploy-*.yaml`:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/deepseek-v4.1-flash/vllm/<target>/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/deepseek-v4.1-flash/vllm/<target>/.kustomize-matrix.yaml
```

`scripts/kustomize-matrix.py check` fails if a checked-in manifest is stale.
Use `compose` to build an ad-hoc variant without checking in an overlay.

## Limitations

- **Adaptive verification must stay off.** It is the established cause of a
  CUDA illegal memory access on this model. Turning it off also rules out
  `--enforce-eager`, because that path structurally requires captured graphs.
- **KV-aware routing degrades silently** if `enable_kv_cache_events` is dropped
  from `--kv-events-config`: it defaults to false, the router receives no
  events, and `DYN_ROUTER_MODE=kv` falls back to load-based routing with no
  error.
- **Disaggregation without an RDMA device falls back to TCP silently**, costing
  roughly an order of magnitude of TTFT. The `rdma/ib` resource name is
  cluster-specific; other clusters expose `rdma/shared_ib` or a vendor name.
- **A cross-clique GB200 placement does not error.** Confirm with
  `kubectl get nodes -L nvidia.com/gpu.clique` and one real request.
- **Do not pair the engine-side and Dynamo tool-call parsers.** Dynamo refuses
  that combination. The two *reasoning* parsers are both required and are not
  duplicates: the engine-side one gates guided decoding, the Dynamo-side one
  splits `reasoning_content` from `content`.
- The SGLang disaggregated target has no speculative decoding, and its
  KV-aware routing is inert at `replicas: 1`.
