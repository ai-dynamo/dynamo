<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron 3.5 Super VL Certified Recipe

Deploy `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` with vLLM on B200 GPUs, using the
Dynamo 1.7.0 enterprise images `dynamo-frontend-enterprise` and `vllm-runtime-enterprise`. The
recipes are tuned for agentic coding traffic like the SWE-bench Pro trace: long, multi-turn
prompts with high prefix reuse.

See the [recipe page](https://docs.nvidia.com/dynamo/dev/recipes/nemotron-3-5-super-vl) and pick
**B200 Certified** for certification results.

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
- Nodes with room for 16 CPUs, 328 GiB of memory, and 128 GiB of ephemeral storage per worker
  pod. The checkpoint downloads into the ephemeral storage.
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

### Shared Model Cache

With `emptyDir`, every worker pod downloads the checkpoint again each time it starts, including on
scale-up and rescheduling. To download it once and reuse it, keep the checkpoint on the
`shared-model-cache` PVC instead. This needs a ReadWriteMany storage class whose filesystem
supports file locking, such as NFSv4.

Create the PVC from the open-source recipe's
[`model-cache.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/model-cache/model-cache.yaml),
after setting `storageClassName` to that class:

```bash
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/model-cache/model-cache.yaml -n ${NAMESPACE}
```

Then, in every worker, replace the `hf-cache` `emptyDir` with the PVC and mount its `hub`
directory:

```yaml
volumeMounts:
- name: hf-cache
  mountPath: /hf-cache
  subPath: hub
# ...
volumes:
- name: hf-cache
  persistentVolumeClaim:
    claimName: shared-model-cache
```

The first workers to start download the checkpoint into the PVC. They lock each file as they
download it, so the others wait instead of downloading it again. Later pods find the checkpoint in
the cache and skip the download, but still contact Hugging Face at startup, so keep the token. The
weights no longer use ephemeral storage, so the workers' ephemeral-storage requests can come down.
The same PVC also serves the [benchmark](#benchmark) Job.

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

To serve video on the disaggregated target, make two more changes. A large inline video can take
longer than the default 5 s request timeout to reach a prefill worker, and the decode worker
reloads each video before it pulls KV from prefill, which can take minutes under load. Prefill
releases unread KV blocks after 30 s by default, and decode then generates from stale KV.

- Add this variable to the frontend and every worker:

  ```yaml
  - name: DYN_TCP_REQUEST_TIMEOUT
    value: "120"
  ```

- Add `"kv_lease_duration":2400` to the `NixlConnector` settings in `KV_TRANSFER_CONFIG` on the
  prefill and decode workers. On prefill, it goes in the `NixlConnector` entry inside the
  `MultiConnector`. On decode, the value becomes:

  ```yaml
  - name: KV_TRANSFER_CONFIG
    value: '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda","kv_connector_extra_config":{"backends":["UCX"],"kv_lease_duration":2400}}'
  ```

The 2400 s lease covers the longest KV pull wait seen in Video-MME testing, about 1,180 s; raise
it if your decode queue runs longer. Prefill holds the KV blocks of abandoned requests for the
whole lease. See [#16155](https://github.com/ai-dynamo/dynamo/issues/16155) and
[vllm-project/vllm#61229](https://github.com/vllm-project/vllm/issues/61229).

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
