<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron 3.5 Super VL Certified Recipe

Deploy `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` with vLLM on B200 GPUs, using the
Dynamo 1.7.0 enterprise images `dynamo-frontend-enterprise` and `vllm-runtime-enterprise`. The
recipes are tuned for agentic coding traffic like the SWE-bench Pro trace: long, multi-turn
prompts with high prefix reuse.

See the
[recipe page](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/recipes/model-recipes/nemotron-3-5-super-vl-120b-a12b-certified.mdx)
for certification results.

The open-source recipes for this model, including other GPUs and TensorRT-LLM, are in
[`recipes/nemotron-3.5-super-vl-120b-a12b`](https://github.com/ai-dynamo/dynamo/tree/main/recipes/nemotron-3.5-super-vl-120b-a12b).

## Targets

| Manifest | GPUs | Topology | Parallelism | Speculative decoding | Context length |
|----------|-----:|----------|-------------|----------------------|---------------:|
| [`vllm/agg-b200-swebenchpro`](vllm/agg-b200-swebenchpro/deploy.yaml) | 4 | 4 aggregated workers | TP1 per worker | MTP, 5 draft tokens | 1M |
| [`vllm/disagg-b200-swebenchpro`](vllm/disagg-b200-swebenchpro/deploy.yaml) | 5 | 3 prefill and 2 decode workers | TP1 per worker | MTP, 5 draft tokens | 1M |

Both targets use NVFP4 weights, an FP8 KV cache, Mamba prefix caching, 200 GB of CPU KV cache
offload on each aggregated and prefill worker, and the KV router with session affinity. The
disaggregated target transfers KV cache between workers with NIXL over UCX on RDMA.

## Prerequisites

- A Kubernetes cluster with the Dynamo platform installed from the enterprise Helm chart, release
  1.7.0. See
  [Dynamo Enterprise Certified Recipes](../README.md#prerequisites).
- 4 B200 GPUs for the aggregated target, or 5 for the disaggregated target.
- Nodes with room for 16 CPUs and 328 GiB of memory per worker pod, plus 64 GiB (aggregated) or
  128 GiB (disaggregated) of ephemeral storage.
- Room for 16 CPUs and 16 GiB of memory for the frontend pod.
- For the disaggregated target, an RDMA device plugin that exposes `rdma/shared_hca`.
- A Hugging Face token with access to the checkpoint.

Create the namespace and token secret:

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

## Deploy

Each worker downloads the checkpoint into its own `emptyDir` volume at startup, so no model-cache
PVC is needed. Allow time for the download on first start; the startup probe waits up to 4 hours.

To deploy the aggregated target:

```bash
export DGD=super-3-5-vllm-b200-agg-swebenchpro
kubectl apply -f recipes/certified/nemotron-3.5-super-vl-120b-a12b/vllm/agg-b200-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

To deploy the disaggregated target:

```bash
export DGD=super-3-5-vllm-b200-disagg-swebenchpro
kubectl apply -f recipes/certified/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-b200-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

Wait for every pod to become ready:

```bash
kubectl get pods -n ${NAMESPACE} -w
```

## Smoke Test

Forward the frontend port:

```bash
kubectl port-forward svc/${DGD}-frontend 8000:8000 -n ${NAMESPACE}
```

Send a request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/nemotron-3.5-super-vl-120b-a12b","messages":[{"role":"user","content":"Write a one-sentence readiness check."}],"max_tokens":64}'
```

The model reasons before it answers. The deployment parses reasoning with `nemotron_v3` and tool
calls with `qwen3_coder`.

## Multimodal Workloads

The manifests are tuned for the SWE-bench Pro trace. To serve image or video requests,
reserve GPU memory for passing multimodal inputs between vLLM processes: add these arguments to
every worker, after `--gpu-memory-utilization`:

```yaml
- --mm-ipc-gpu-memory-gb
- "10"
```

Raise the request size limits for large images and videos. Add these variables to the frontend
and every worker; `DYN_HTTP_BODY_LIMIT_MB` applies only to the frontend:

```yaml
- name: DYN_TCP_MAX_MESSAGE_SIZE
  value: "2147483648"
- name: DYN_MM_MAX_DATA_URL_MB
  value: "1024"
- name: DYN_HTTP_BODY_LIMIT_MB
  value: "1024"
```

To sample up to 256 frames from each video, in place of the default 32, add this variable to
every worker, and to the frontend if it decodes video (`--frontend-decoding`):

```yaml
- name: DYN_MM_VIDEO_NUM_FRAMES
  value: "256"
```

NIM Factory image and video accuracy testing used these settings.

## Benchmark

Use the SWE-bench Pro trace-replay workflow in
[`perf/README.md`](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/perf/README.md).
Set `ENDPOINT` in `perf.yaml` to `http://${DGD}-frontend:8000`. The benchmark Job reads the trace
and tokenizer from the `shared-model-cache` PVC, which these recipes do not create. Create it from
the open-source recipe's
[`model-cache`](https://github.com/ai-dynamo/dynamo/tree/main/recipes/nemotron-3.5-super-vl-120b-a12b/model-cache)
manifests before running the Job.

For synthetic throughput tests, each manifest carries a commented `--speculative-config` value that
fixes the MTP acceptance length at 4.06. The default value uses natural acceptance; use it for
accuracy and production traffic.

## Notes

- The disaggregated workers set `VLLM_SSM_CONV_STATE_LAYOUT=DS`. NIXL needs this layout to transfer
  the Mamba convolution state between prefill and decode.
- In the disaggregated target, only the prefill workers offload KV cache to CPU. Their capacity is
  set by `cpu_bytes_to_use` inside the `MultiConnector` configuration in `KV_TRANSFER_CONFIG`. Don't
  add `--kv-offloading-size` to those workers; it replaces the `MultiConnector` configuration. The
  decode workers use `NixlConnector` alone.
