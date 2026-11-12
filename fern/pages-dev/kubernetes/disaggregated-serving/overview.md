---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Disaggregated Serving
subtitle: Write a DynamoGraphDeployment that splits prefill and decode into independent worker pools
---

Disaggregated serving separates the two phases of LLM inference into their own
worker pools so you can scale, size, and place them independently:

| Phase | What it does | Scaling pressure |
|---|---|---|
| Prefill | Processes the prompt and produces the initial KV cache. | Input length, prompt reuse, context size |
| Decode | Generates output tokens using the KV cache. | Concurrency, output length, active KV memory |

In an **aggregated** deployment, one worker does both phases. In a
**disaggregated** deployment, prefill and decode are separate pools on separate
GPUs: Dynamo routes each request through prefill first, transfers the KV cache to
the decode worker, then streams the response from decode.

Structurally, that difference is one worker versus two. The rest of the
`DynamoGraphDeployment` (DGD) is the same:

<Tabs>
<Tab title="Aggregated (1 worker)">

```yaml
  services:
    Frontend:
      componentType: frontend
    worker:
      componentType: worker
      # one worker does both prefill and decode
```

</Tab>
<Tab title="Disaggregated (prefill + decode)">

```yaml
  services:
    Frontend:
      componentType: frontend
    prefill:
      componentType: worker
      subComponentType: prefill   # prompt processing only
    decode:
      componentType: worker
      subComponentType: decode    # token generation only
```

</Tab>
</Tabs>

This guide walks through writing the disaggregated spec end to end. If you have
not authored a DGD before, read the [Deploy with DGD](../model-deployment/deploy-with-dgd.md)
guide first — this page assumes you know the overall shape and focuses on what
disaggregation adds.

## Should you disaggregate?

Disaggregation helps most when prefill and decode need different resource shapes:

- long prompts or retrieval-heavy traffic make prefill expensive
- long generations or high concurrency make decode the bottleneck
- you want to scale prefill and decode replicas independently
- large models need different parallelism for prompt processing and generation

It is not automatically better. For small models, short prompts, low concurrency,
or clusters without a fast KV-transfer fabric, an aggregated deployment is
simpler and often faster. Use [Sizing with AIConfigurator](../../developer-guide/additional-resources/aiconfigurator-reference.md) to
compare aggregated vs. disaggregated layouts before committing.

This guide uses a **single-node, multi-GPU** example — prefill and decode run on
separate GPUs in the same node (for instance, a node with 8×H100 serving
Qwen3-32B, one GPU per worker). Each worker runs in its own pod. By default,
Kubernetes isolation stops the KV cache from moving over NVLink between two pods,
even on the same node, so the transfer uses RDMA if the pods request it and TCP
otherwise. The [Enable KV transfer](#enable-kv-transfer) step explains why, and
[Use NVLink between pods on one node](#use-nvlink-between-pods-on-one-node)
describes the opt-in. Scaling prefill and decode across **multiple nodes** adds
RDMA networking; see [Enable multi-node transfer](#enable-multi-node-transfer)
at the end.

## Write the deployment

<Steps toc={true} tocDepth={2}>

<Step title="Check prerequisites">

For the single-node example in this guide you need:

1. **A node with multiple GPUs** — one per prefill and decode worker (this
   example uses 8×H100 with `gpu: "1"` per worker).
2. **ETCD and NATS** deployed for Dynamo coordination.
3. **A HuggingFace token secret** (`hf-token-secret`) if the model is gated.

Model caching (the PVC and `HF_HOME` wiring the workers mount) is configured
separately — see [Model caching](../kv-cache-offloading/overview.mdx). This
guide omits those fields for clarity; without them the model downloads to the
container's default cache.

Scaling across multiple nodes additionally requires an RDMA-capable network and
device plugin — covered in [Enable multi-node transfer](#enable-multi-node-transfer).

</Step>

<Step title="Start the spec: the frontend">

Begin with the DGD skeleton — just a frontend. This is identical to an
aggregated deployment; nothing about disaggregation changes it.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: dynamo-disagg
  namespace: your-namespace
spec:
  backendFramework: vllm
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0
          imagePullPolicy: IfNotPresent
```

</Step>

<Step title="Add the prefill worker">

The prefill worker runs the prompt phase. Two things make it a prefill worker
rather than a plain worker:

- `subComponentType: prefill` tags the role for the operator and router.
- `--disaggregation-mode prefill` tells the engine to stop after producing the
  KV cache instead of generating tokens.

```yaml
    prefill:
      envFromSecret: hf-token-secret
      componentType: worker
      subComponentType: prefill
      replicas: 3
      resources:
        limits:
          gpu: "1"
      sharedMemory:
        size: 16Gi
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0
          imagePullPolicy: IfNotPresent
          command: ["python3", "-m", "dynamo.vllm"]
          args:
            - --model
            - Qwen/Qwen3-32B
            - --disaggregation-mode
            - prefill
```

Qwen3-32B fits on a single H100, so each worker takes `gpu: "1"` with no tensor
parallelism. `replicas: 3` gives three prefill workers; size the count for your
prompt load independently of decode. (`sharedMemory` is explained in the KV
transfer step.)

</Step>

<Step title="Add the decode worker">

The decode worker continues generation after prefill hands off the KV cache. It
mirrors the prefill worker with `subComponentType: decode` and
`--disaggregation-mode decode`, and its replica count is sized for concurrency
rather than prompt load.

```yaml
    decode:
      envFromSecret: hf-token-secret
      componentType: worker
      subComponentType: decode
      replicas: 4
      resources:
        limits:
          gpu: "1"
      sharedMemory:
        size: 16Gi
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0
          imagePullPolicy: IfNotPresent
          command: ["python3", "-m", "dynamo.vllm"]
          args:
            - --model
            - Qwen/Qwen3-32B
            - --disaggregation-mode
            - decode
```

<Warning>
Prefill and decode workers must pass the **same `--model`** (and, if you set
them, the same dtype, block size, and KV layout). A mismatch means the KV cache
prefill produces cannot be consumed by decode — you get transfer errors or
silently corrupt output.
</Warning>

</Step>

<Step title="Enable KV transfer">

Because prefill and decode run on **different GPUs**, the KV cache prefill
produces has to be copied into the decode worker's GPU memory before decode can
generate. Dynamo's transfer layer, **NIXL**, does this automatically, and with
its UCX backend, UCX picks the transport.

UCX moves the cache GPU-to-GPU over NVLink with **CUDA IPC** only when both of
these hold for the two worker processes:

- **They share a PID namespace.** Starting with UCX 1.23
  ([openucx/ucx#11514](https://github.com/openucx/ucx/pull/11514)), UCX treats
  CUDA IPC as unreachable across PID namespaces unless both GPUs support fabric
  handles (IMEX). HGX systems such as 8×H100 nodes do not have them.
- **The importing process can see the peer GPU.** With the default one GPU per
  pod, opening the peer's CUDA IPC handle fails.

Default worker pods meet neither condition, even when they run on the same
node. UCX then uses RDMA if the pods request RDMA devices, and TCP otherwise.
NIXL logs no error when this happens. TCP is slow enough that transfers of long
prompts can exceed the engine's KV transfer timeout, and requests fail. Setting
`UCX_TLS` does not change this. To get NVLink between pods, see
[Use NVLink between pods on one node](#use-nvlink-between-pods-on-one-node). To
check which transport a deployment uses, see the next step.

The field you always set is shared memory, already in the worker specs above:

| Setting | Why |
|---|---|
| `sharedMemory.size: 16Gi` | NIXL stages transfer metadata through the pod's `/dev/shm`. The pod default (64 MB) is too small and NIXL fails to initialize. `16Gi` is comfortable headroom, not a tightly-computed per-request budget — it does not consume that memory unless needed. |

With these fields the KV transfer works: two workers tagged prefill/decode,
matching `--model`, and enough shared memory for NIXL. The transport it uses
depends on the pod settings described above.

</Step>

<Step title="Apply and verify the transfer path">

Apply the assembled spec:

```bash
kubectl apply -f dynamo-disagg.yaml -n your-namespace
```

Then confirm NIXL initialized its transfer backend — the key check for a
disaggregated deployment. Grep a prefill worker's logs:

```bash
kubectl logs <prefill-worker-pod> | grep -i "NIXL"
```

You want to see NIXL bring up a backend, for example:

```text
NIXL INFO Backend UCX was instantiated
```

If NIXL fails to initialize, the usual cause is `sharedMemory.size` left at the
default. Symptoms of a broken transfer path include high TTFT despite free
prefill capacity, decode workers sitting idle while prefill is busy, or
disaggregated throughput falling below your aggregated baseline.

This check does not show which transport UCX chose. To see it, add
`UCX_PROTO_INFO=y` to the `env` of both workers, send a request, and read the
protocol lines in the decode worker's log:

```bash
kubectl logs <decode-worker-pod> | grep -E "intra-node|inter-node|cuda_ipc|tcp/|rc_mlx5"
```

For a GPU-to-GPU KV read, `cuda_ipc` means CUDA IPC, an RDMA device such as
`rc_mlx5` means RDMA, and `tcp` means the transfer goes over TCP. CUDA IPC uses
NVLink only when the two GPUs are connected by NVLink; otherwise it goes over
PCIe. To check, run `nvidia-smi topo -m` on the node: `NV#` between the two
GPUs means NVLink. For example, with vLLM, two default pods on one node show
this line for the KV read:

```text
intra-node ... 0..inf | software emulation | tcp/eth0
```

Remove `UCX_PROTO_INFO` after the check; it adds log output for every new
connection.

</Step>

<Step title="Enable multi-node transfer">

Everything above keeps prefill and decode on **one node**. When a worker's
parallelism exceeds the GPUs on a node, or you want prefill and decode pools on
separate machines, the KV cache travels over the **network**, and that path needs
**RDMA** (InfiniBand or RoCE). Systems with multi-node NVLink, such as GB200 or
GB300 racks with IMEX, can instead move the KV cache over NVLink between nodes in
one NVLink domain; that setup is not covered here. Without either, transfers fall
back to TCP and KV movement can dominate TTFT and throughput. The same RDMA
setup also serves separate pods on one node when you do not use the NVLink
opt-in.

Multi-node adds RDMA fields to each worker (`rdma/ib` resource requests, the
`IPC_LOCK` capability, and `UCX_*` transport env vars) plus an RDMA device plugin
on the cluster. That setup is out of scope here — see the
[RDMA Setup](../installation/rdma-setup/overview.md)
for the transport configuration and [Multinode Orchestration](../installation/multinode-orchestration.md)
for spanning workers across machines.

</Step>

</Steps>

## Use NVLink between pods on one node

To move the KV cache over NVLink between prefill and decode pods on the same
node, each worker pod must share the host PID namespace and see the other
worker's GPU. One pattern that does this and keeps GPU scheduling under the
device plugin:

<Warning>
This pattern runs workers with `hostPID: true` and `privileged: true`. The
namespace must allow Pod Security level `privileged`, and each worker can see
every process and device on the host. It helps only when prefill and decode
run on the same node; across nodes you still need RDMA or multi-node NVLink.
Decide with your cluster administrator whether this trade-off is acceptable.
</Warning>

```yaml
    decode:
      envFromSecret: hf-token-secret
      componentType: worker
      subComponentType: decode
      replicas: 4
      resources:
        limits:
          gpu: "1"
      sharedMemory:
        size: 16Gi
      extraPodSpec:
        hostPID: true                 # share the host PID namespace
        initContainers:
          - name: gpu-ids             # record the GPUs the device plugin allocated
            image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0
            command: ["sh", "-c", "nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, > /gpu-alloc/uuids"]
            resources:
              limits:
                nvidia.com/gpu: "1"   # same count as the worker
            volumeMounts:
              - name: gpu-alloc
                mountPath: /gpu-alloc
        volumes:
          - name: gpu-alloc
            emptyDir: {}
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0
          securityContext:
            privileged: true          # make every GPU on the node visible
          volumeMounts:
            - name: gpu-alloc
              mountPath: /gpu-alloc
          command:
            - bash
            - -c
            - |
              # Pin CUDA back to the GPUs allocated to this pod
              export CUDA_DEVICE_ORDER=PCI_BUS_ID
              export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
                | grep -F -f <(tr , '\n' < /gpu-alloc/uuids) | cut -d, -f1 | paste -sd,)
              exec python3 -m dynamo.vllm "$@"
            - worker
          args:
            - --model
            - Qwen/Qwen3-32B
            - --disaggregation-mode
            - decode
```

Apply the same change to the prefill worker, and keep its own `replicas` and
`envFromSecret`.

The workers can use CUDA IPC only when they run on the same node. On a cluster
with more than one GPU node, place all prefill and decode replicas on one node,
for example with a `nodeSelector` on `kubernetes.io/hostname` in each worker's
`extraPodSpec`. That node needs a free GPU for every replica: seven for this
example (three prefill, four decode).

The pieces work together:

- `hostPID: true` lets UCX treat CUDA IPC as reachable between the two pods.
- `privileged: true` makes the peer GPU visible, so the importing worker can
  open its CUDA IPC handle.
- The init container requests the same GPU count as the worker. The kubelet
  gives the main container the same devices, so the init container's
  `nvidia-smi` output lists the GPUs allocated to this pod.
- The command wrapper sets `CUDA_VISIBLE_DEVICES` to those GPUs, so each worker
  still uses only the GPUs the scheduler assigned to it.

After you apply it, run the `UCX_PROTO_INFO=y` check from
[Apply and verify the transfer path](#apply-and-verify-the-transfer-path). The
GPU-to-GPU KV read should show `cuda_ipc`:

```text
intra-node ... 0..inf | zero-copy | cuda_ipc/cuda
```

[GitHub issue #16646](https://github.com/ai-dynamo/dynamo/issues/16646) has the
measurements behind this pattern and tracks first-class support in the operator.

## Faster starting points

Typing the full spec from scratch is rarely the fastest path. Copy a validated
template and adapt it:

| Starting point | Use when |
|---|---|
| [Dynamo Recipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes) | A recipe matches your model, backend, and hardware. Best for validated baselines and `perf.yaml` benchmarks. |
| Backend `disagg.yaml` templates | You want a working DGD to adapt. Each backend ships `disagg.yaml`, `disagg_router.yaml`, and `disagg_planner.yaml` under its `deploy/` folder. |
| [DGDR](../../reference/kubernetes-api/dynamo-graph-deployment-request.mdx) | You want Dynamo to generate a DGD from model, backend, hardware, and SLA intent. |
| [Sizing with AIConfigurator](../../developer-guide/additional-resources/aiconfigurator-reference.md) | You want to compare aggregated vs. disaggregated layouts and estimate prefill/decode sizing first. |

Good recipe starting points:

- [Qwen3-32B vLLM disagg + KV router](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3-32b)

Backend deployment examples with concrete worker flags:

| Backend | Examples |
|---|---|
| vLLM | [Deployment examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy) — `disagg.yaml`, `disagg_router.yaml`, `disagg_planner.yaml` |
| TensorRT-LLM | [Deployment examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/deploy) — disaggregated, router, and planner variants |
| SGLang | [Deployment examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/deploy) — NIXL-based disaggregated serving |

## How disaggregation relates to KV-aware routing

These are separate features that pair well. Disaggregation splits prefill and
decode. KV-aware routing chooses workers based on cache locality. Many
production deployments use both, but you can reason about them independently.

For router behavior, see [Router: Disaggregated Serving](../../developer-guide/knowledge-base/modular-components/router/disaggregated-serving.md)
and [KV Cache Aware Routing](../../developer-guide/knowledge-base/modular-components/router/router-guide.md).

## Before production

- Confirm NIXL initialized on each worker (the log check above).
- Confirm the KV transfer uses the transport you expect (`cuda_ipc`, RDMA, or
  `tcp`) with the `UCX_PROTO_INFO=y` check above. Separate pods on one node use
  TCP unless they request RDMA or use the NVLink opt-in.
- Confirm prefill and decode workers agree on model, dtype, block size, and KV layout.
- Confirm pods have the required GPU and shared memory.
- Confirm frontend/router flags match your routing strategy.
- Run benchmarks inside the cluster, not through local port-forwarding.
- If you scale across nodes, validate the RDMA transfer path — see
  [Enable multi-node transfer](#enable-multi-node-transfer).

Use [Dynamo Benchmarking](../../recipes/feature-benchmarks/benchmarking-guide.md) to compare
aggregated and disaggregated configurations under the same workload.

## Next steps

1. Start from a matching [Dynamo Recipe](https://github.com/ai-dynamo/dynamo/tree/main/recipes) when one exists.
2. Read the backend-specific `disagg.yaml` for your engine.
3. Use [Sizing with AIConfigurator](../../developer-guide/additional-resources/aiconfigurator-reference.md) or DGDR to choose prefill/decode sizing.
4. Validate with [Dynamo Benchmarking](../../recipes/feature-benchmarks/benchmarking-guide.md).
