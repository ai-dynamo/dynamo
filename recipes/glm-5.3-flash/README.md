# GLM-5.3-Flash — vLLM Recipes

Dynamo serving recipes for [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) on GB200.
GLM-5.3-Flash is a hybrid GLA/KDA attention model with multimodal support (image/video inputs
disabled in these recipes for text-only serving).

## Recipes

| Recipe | Hardware | Mode | GPUs | KV dtype | Notes |
|--------|----------|------|------|----------|-------|
| `vllm/agg-gb200-agentic` | GB200 | Aggregated | 4 (TP=4, 1 node) | fp8 | Prefix caching; high throughput |
| `vllm/disagg-gb200-agentic` | GB200 | Disaggregated 1P1D | 8 (TP=4 × 2 workers) | bf16 | NIXL KV transfer; prefill/decode specialization |

Benchmark workload: agentic coding, 90% KV reuse, SSP=57600 / ISL=6400 / OSL=400 (64K context).

## Measured performance (dynamo-gcp-dev-02, 2026-08-27)

| Recipe | Concurrency | Server throughput | tok/s/GPU | TTFT p50 |
|--------|-------------|-------------------|-----------|----------|
| agg-gb200-agentic | 4 | 258 tok/s | 64.5 | 2.9 s |
| agg-gb200-agentic | 6 | ~300 tok/s (peak) | ~75 | 4.8 s |
| disagg-gb200-agentic | 4 | 103 tok/s | 12.9 | 10.8 s |
| disagg-gb200-agentic | 16 | 173 tok/s | 21.6 | 32 s |
| disagg-gb200-agentic | 24 | 206 tok/s (peak) | 25.8 | 47 s |

Disagg TTFT is transport-dominated over cuda_copy+tcp (~3–4 GB KV per 64K request over ethernet).
See [MNNVL transport upgrade](#mnnvl-transport-upgrade) below for the path to NVLink KV transfer.

## Image

These recipes use `vllm/vllm-openai:glm53-flash`, a GLM-specific vLLM build that includes:

- GLA/KDA attention kernels (`gla_cuda`, `kda_cache`)
- `sitecustomize.py` cudnn workaround (WAR for segfault in `_kpool_*` kernels on GB200)
- Standard vLLM with `--trust-remote-code` model support

`ai-dynamo==1.4.1` is pip-installed at pod startup. Expect ~60–90 s additional startup time
for the pip install on first run (subsequent runs pull from layer cache if unchanged).

## Prerequisites

- A `model-cache` PersistentVolumeClaim with the model weights at:
  `models--zai-org--GLM-5.3-Flash/snapshots/<sha>/`
  (populated by `huggingface-cli download zai-org/GLM-5.3-Flash`)
- Dynamo operator installed in the cluster
- GB200 nodes with `nvidia.com/gpu.product: NVIDIA-GB200` label

## Deployment

```bash
# Aggregated
kubectl apply -f recipes/glm-5.3-flash/vllm/agg-gb200-agentic/deploy.yaml

# Disaggregated 1P1D
kubectl apply -f recipes/glm-5.3-flash/vllm/disagg-gb200-agentic/deploy.yaml
```

Watch startup (workers take 15–25 min on first run for CUDA graph capture):

```bash
kubectl get dgd -w
kubectl logs -f deployment/glm53-flash-agg-gb200-agentic-vllmworker
```

## Disagg: ComputeDomain requirements

The disagg recipe requires `numNodes: 0` in the ComputeDomain spec. This creates an
on-demand domain that spans the full NVL72 rack (all 72 GPUs), ensuring MNNVL fabric
handles are valid between any pair of nodes. Using `numNodes: 2` creates a partial
domain that causes C-level crashes (`cudaErrorInvalidValue` / SIGBUS) when prefill and
decode workers land on different nodes.

**ComputeDomain spec is immutable.** To change `numNodes`, delete the DGD first
(releases resource claim references), then delete the ComputeDomain, then re-apply.

## MNNVL transport upgrade

The disagg recipe ships with `UCX_TLS: cuda_copy,^cuda_ipc,tcp` (KV transfer over TCP).
The `^cuda_ipc` excludes single-host IPC to prevent a C-level crash when the generic
`glm53-flash` UCX library attempts MNNVL IPC handles cross-node.

To enable MNNVL NVLink KV transfer (~10x lower TTFT):
1. Rebuild `vllm/vllm-openai:glm53-flash` on `nvcr.io/nvidia/ai-dynamo/vllm-runtime` base
   (the vllm-runtime image has a UCX build with MNNVL IPC support)
2. In `deploy.yaml`, replace:
   ```yaml
   - {name: UCX_TLS, value: "cuda_copy,^cuda_ipc,tcp"}
   ```
   with:
   ```yaml
   - {name: UCX_TLS,                   value: "cuda_copy,cuda_ipc,tcp"}
   - {name: UCX_CUDA_IPC_ENABLE_MNNVL, value: "y"}
   ```

## Key environment variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `VLLM_SSM_CONV_STATE_LAYOUT` | `DS` | GLM SSM conv state memory layout (disagg only) |
| `VLLM_KV_CACHE_LAYOUT` | `HND` | KV cache layout compatible with NIXL transfer (disagg only) |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | `5560` (prefill) / `5558` (decode) | NIXL handshake ZMQ ports |
| `VLLM_ALLREDUCE_USE_FLASHINFER` | `1` | FlashInfer all-reduce for MNNVL bandwidth |
| `VLLM_FLASHINFER_ALLREDUCE_BACKEND` | `mnnvl` | Use MNNVL NVLink for intra-worker all-reduce |
| `NCCL_MNNVL_ENABLE` | `1` | Enable MNNVL in NCCL collective ops |
