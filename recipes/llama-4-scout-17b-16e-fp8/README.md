# Llama-4-Scout-17B-16E-Instruct-FP8 — Single-Node Disaggregated (vLLM)

Disaggregated serving recipe for `nvidia/Llama-4-Scout-17B-16E-Instruct-FP8`
on a single 8×H100-80GB node. Default layout is **1 prefill + 3 decode**
(TP=2 each), using all 8 GPUs — chosen because Scout's decode is HBM-bandwidth-
bound and adds throughput linearly with decode replicas at the workload
points we measured.

## Why this layout

| Knob | Choice | Reason |
|---|---|---|
| TP | 2 | ~109 GB FP8 / 2 = ~55 GB/GPU on 80 GB H100. TP=1 won't fit; TP=4 burns idle GPUs. |
| Replicas | 1P + 3D | Matched-concurrency head-to-head: 1P+3D beat the 1P+2D baseline on throughput (+39%), p50 latency (−29%), and tok/Wh (+16%) at c=36. Use 1P+1D (4 GPUs) for the minimum useful disagg layout. |
| `--max-model-len` | 4096 | Llama 4 native context is huge (10M); pinning keeps KV pool predictable and ensures the prefill/decode NIXL shape contract is met (they MUST match). |
| Router | `kv` | Required for disagg. `round_robin` silently bypasses NIXL handoff. |
| Prefill GPU util | 0.85 | Flashinfer MoE workspace needs transient scratch. |
| Decode GPU util | 0.88 | Validated decode value; raise to 0.92 only if KV pool becomes the binding constraint. |
| `UCX_MEMTYPE_CACHE=n` | both | Without it, PyTorch VMM remap invalidates UCX's cached (addr→memtype) → `cuMemcpyAsync illegal memory access` kills all TP ranks. |
| `expandable_segments` | **decode only** | Required on decode to avoid fragmentation OOM. Causes a CUDAGraph replay segfault on prefill (VMM remap leaves captured raw pointers dangling), so prefill uses classic cudaMalloc + lower util. |
| `NCCL_CUMEM_ENABLE=0` | decode only | Companion to `expandable_segments` — avoids VMM/NCCL allocator collision. |
| `--disable-custom-all-reduce` | both | vLLM custom AR + `hostIPC` collide between prefill+decode on the same node. |

See inline comments in `vllm/disagg-single-node/deploy.yaml` for the full
provenance.

## Runtime compatibility

Tested on `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1`. The disagg flags
match the 1.0.x API:

- `--is-prefill-worker` on prefill
- Split `kv_role`: `kv_producer` (prefill) / `kv_consumer` (decode)

For 1.2.0+: replace with `--disaggregation-mode {prefill,decode}` and
`"kv_role":"kv_both"` (see the `qwen3-32b` recipe for the 1.2.0 pattern).

## Prerequisites

1. Dynamo platform installed — see the [Kubernetes Deployment Guide](../../docs/kubernetes/README.md).
2. One 8×H100-80GB node (or larger; 4 GPUs suffice for the minimum 1P+1D layout).
3. HuggingFace token (the FP8 checkpoint is gated):
   ```bash
   export NAMESPACE=your-namespace
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="your-token" -n ${NAMESPACE}
   ```

## Quick start

```bash
# 1. Storage — edit model-cache/model-cache.yaml to set storageClassName first
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}

# 2. Download weights (~109 GB)
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=1800s

# 3. Deploy
kubectl apply -f vllm/disagg-single-node/deploy.yaml -n ${NAMESPACE}
kubectl wait --for=condition=ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=vllm-llama-4-scout-disagg \
  -n ${NAMESPACE} --timeout=1800s
# First boot warms CUDA graphs across the prompt-length range (~20 min);
# subsequent restarts hit the compilation-cache in <5 min.

# 4. Benchmark
kubectl apply -f vllm/disagg-single-node/perf.yaml -n ${NAMESPACE}
kubectl exec -it -n ${NAMESPACE} vllm-llama-4-scout-disagg-benchmark -- tmux a -t benchmark
```

## Scaling notes

- **1P + 1D (4 GPUs)** — set decode `replicas: 1`. Minimum disagg layout; leaves 4 GPUs free.
- **1P + 3D (8 GPUs, default)** — best tok/s and tok/Wh on ISL=1024/OSL=512.
- **2P + 2D (8 GPUs)** — better for very long-prompt workloads where prefill becomes the queue.

## Cleanup

```bash
kubectl delete pod -l app=benchmark -n ${NAMESPACE}
kubectl delete dynamographdeployment vllm-llama-4-scout-disagg -n ${NAMESPACE}
```
