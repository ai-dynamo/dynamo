# DeepSeek-V3.2 Recipes

Deployment recipes for **DeepSeek-V3.2** using TensorRT-LLM with Dynamo's KV-aware routing and ModelExpress P2P weight transfer for fast scale-up.

## Available Configurations

| Configuration | Modality | Deploy Config | Notes |
|---------------|----------|---------------|-------|
| **dep8x2 disaggregated** | Text | [`trtllm/disagg/dep8x2/deploy.yaml`](trtllm/disagg/dep8x2/deploy.yaml) | TP=8 prefill + TP=8 decode (8 GPUs each, 2-node multinode each), ModelExpress P2P enabled for sub-second weight loading on subsequent replica scale-up. |

## A note on the missing `model-cache/`

This recipe deviates from the standard structure documented in [`recipes/CONTRIBUTING.md`](../CONTRIBUTING.md) — there's no `model-cache/` directory. That's intentional:

DeepSeek-V3.2 is ~685 GB on disk; downloading it into a per-deployment cache PVC every time is the bottleneck this recipe is specifically designed to bypass. The deployment uses ModelExpress P2P (`--model-express-url modelexpress-server:8001`) so:

- The **first replica** loads the model from a shared `shared-model-cache` PVC (provisioned out-of-band by the operator) and publishes its weights via NIXL RDMA.
- **Every subsequent replica** auto-detects the existing source and pulls weights via RDMA in ~2 seconds per rank instead of pulling from disk.

For clusters that need the standard model-cache flow (no MX server available), use a sibling DeepSeek recipe such as [`recipes/deepseek-r1/`](../deepseek-r1/) or [`recipes/deepseek-v32-fp4/`](../deepseek-v32-fp4/) which include `model-cache/` with `model-cache.yaml` and `model-download.yaml`.

`run.sh` only validates that `<model>/<framework>/<deployment>/deploy.yaml` exists, so this recipe runs cleanly without `model-cache/`.

## Prerequisites

1. **Dynamo platform** installed (etcd + NATS, the DGD operator, optionally HPA + DGDSA controllers).
2. **GB200 cluster** with at least 4 nodes (the dep8x2 deployment uses 2 nodes for prefill + 2 for decode; aggregated configs would need fewer).
3. **HuggingFace token** with access to `deepseek-ai/DeepSeek-V3.2`.
4. **`shared-model-cache` PVC** populated with the model in HF cache layout (downloaded once, out-of-band — for example via a one-shot Job).
5. **ComputeDomain** sized for the deployment's `replicas × multinode.nodeCount` nodes (IMEX channels via DRA on GB200).
6. **ModelExpress server + Redis** reachable in the namespace. A reference deployment is at [`ai-dynamo/modelexpress`](https://github.com/ai-dynamo/modelexpress) (`examples/p2p_transfer_k8s/client/trtllm/mx-infra-decode.yaml`).
7. **`tensorrtllm-runtime` image at TRT-LLM 1.3.0rc15+** (carrying [PR #13531](https://github.com/NVIDIA/TensorRT-LLM/pull/13531) — `MXCheckpointLoader`) plus `pip install modelexpress` (or use the equivalent NVIDIA-built image once available).

## Quick Deploy

```bash
kubectl -n <namespace> apply -f trtllm/disagg/dep8x2/deploy.yaml
```

After all 8 source ranks publish to MX, scale the prefill or decode component up; the new replicas auto-detect the source and complete weight loading in ~5 minutes (vs ~22 minutes for the cold-load source).

## Companion PRs

| PR | Repo | Status | Notes |
|----|------|--------|-------|
| [#13531](https://github.com/NVIDIA/TensorRT-LLM/pull/13531) | TRT-LLM | **Merged 2026-05-06** | `MXCheckpointLoader`, `checkpoint_format="MX"` |
| [#202](https://github.com/ai-dynamo/modelexpress/pull/202) | ModelExpress | **Merged** | `MxLiveWeightLoader`, `publish_model_params` |
| [#267](https://github.com/ai-dynamo/modelexpress/pull/267) | ModelExpress | **Merged** | `MX_POOL_REG` allocation-based registration |
| [#8037](https://github.com/ai-dynamo/dynamo/pull/8037) | Dynamo | This PR | `--model-express-url` engine integration with auto-detect source/target |
