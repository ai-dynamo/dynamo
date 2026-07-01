---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Deploy with DGD
subtitle: Author a DynamoGraphDeployment spec step by step — choose a model and backend, size parallelism for your hardware, then apply it and send requests.
---

A **DynamoGraphDeployment (DGD)** is the Kubernetes Custom Resource (CRD) that describes how to deploy a model with Dynamo. You write the spec, `kubectl apply` it, and the Dynamo operator reconciles it into the running pods, services, and scheduling resources that serve your model. One DGD describes one inference graph: a Frontend plus one or more workers.

This guide walks through authoring that spec end to end: choose a model, pick a backend, size the parallelism for your hardware, then apply the deployment and send it a request. Each step fills in more of a single DGD spec, with optional detours for caching, disaggregation, multinode, and Mixture-of-Experts along the way.

The examples use the `nvidia.com/v1beta1` API: `spec.components` is a list, and each component carries a standard Kubernetes `podTemplate`. (The older `nvidia.com/v1alpha1` API used a `spec.services` map with `extraPodSpec`; the operator still serves it and converts between the two.)

## Prerequisites

Before authoring a DGD, make sure you have:

- A Kubernetes cluster with the **Dynamo Platform installed**. See the [Installation Guide](installation-guide.md).
- `kubectl` access to that cluster and a target namespace.
- A **HuggingFace token** for gated or rate-limited models (you create the Secret in step 1).
- New to Dynamo on Kubernetes? Run one model end to end with the [Kubernetes Quickstart](README.md) first, then come back here to author your own spec.

For the concepts behind the CRDs and the operator, see the [Deployment Overview](model-deployment-guide.md) and the [API Reference](api-reference.md).

## How a DGD is structured

Every DGD has the same top-level shape. `spec.components` is a list, where each entry is one Dynamo component with a stable `name` and a `type`:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: my-deployment
spec:
  components:
  - name: Frontend        # the OpenAI-compatible API gateway
    type: frontend
    # ...
  - name: MyWorker        # one or more inference workers
    type: worker
    # ...
```

- `type: frontend` is the HTTP entry point — see [Frontend](../components/frontend/README.md).
- `type: worker` runs the inference engine (vLLM, SGLang, or TensorRT-LLM).
- Per-component fields you will use most: `replicas`, `multinode`, `sharedMemorySize`, and `podTemplate` — a standard Kubernetes [PodTemplateSpec](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#podtemplatespec-v1-core). The operator injects its defaults into the container named `main` inside `podTemplate.spec.containers`, where you set the `image`, the `command`/`args` that launch the engine, `resources` (CPU/memory/GPU), `envFrom`, `env`, and `volumeMounts`.

The steps below fill in these fields, building one spec from a model to a running deployment.

<Steps toc={true} tocDepth={2}>

<Step title="Choose your model and create the HF token secret">

Pick the model you want to serve. This guide uses **Qwen3-32B** as the running example; substitute your own anywhere you see it. Gated or rate-limited models need a HuggingFace token — create it once as a Secret in your namespace:

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=<your-hf-token> \
  -n <namespace>
```

Start the spec with a Frontend and one worker. Set the model once as a `MODEL` environment variable and reference it from the launch command, so the model name lives in a single place:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-32b-agg
spec:
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: <runtime-image>             # set in the next step
  - name: VllmWorker
    type: worker
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: <runtime-image>             # set in the next step
          envFrom:
          - secretRef:
              name: hf-token-secret          # the Secret you just created
          env:
          - name: MODEL
            value: Qwen/Qwen3-32B            # substitute: your model
          command:
          - /bin/bash
          - -c
          - exec python3 -m dynamo.vllm --model $MODEL
```

The `envFrom` block injects the HuggingFace token; `env: MODEL` sets the model name that `$MODEL` expands to in the launch command.

<Accordion title="Optional: cache model weights">

By default each worker pod downloads the model from HuggingFace on startup. For large models (>70B) this is slow per pod, and many replicas hit HuggingFace rate limits. Create a shared PVC, download the model once with a Job, then mount it into the worker with a standard Kubernetes volume:

```yaml
  - name: VllmWorker
    type: worker
    podTemplate:
      spec:
        containers:
        - name: main
          volumeMounts:
          - name: model-cache
            mountPath: /home/dynamo/.cache/huggingface
        volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache
```

For the PVC definition, the model-download Job, finding the snapshot path, and the faster RDMA-based alternative, see [Model Caching](model-caching.md) (and [ModelExpress](modelexpress.md) for fleet-scale weight distribution).

</Accordion>

</Step>

<Step title="Choose your backend">

The launch command above runs **vLLM**. Dynamo also supports **SGLang** and **TensorRT-LLM**. The backend sets three things: the runtime image, the launch module, and how you pass parallelism.

| Backend | Runtime image | Launch module | Tensor-parallel flag |
|---|---|---|---|
| vLLM | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:<tag>` | `python3 -m dynamo.vllm` | `--tensor-parallel-size` |
| SGLang | `nvcr.io/nvidia/ai-dynamo/sglang-runtime:<tag>` | `python3 -m dynamo.sglang` | `--tp` |
| TensorRT-LLM | `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:<tag>` | `python3 -m dynamo.trtllm` | set in the engine config passed via `--extra-engine-args` |

Fill in the `<runtime-image>` placeholder on both the Frontend and the worker with your chosen image and release tag:

```yaml
        - name: main
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1   # substitute: release tag
```

vLLM and SGLang take parallelism as CLI flags. TensorRT-LLM bakes parallelism into a pre-built engine config file that you reference with `--extra-engine-args`, so its worker command looks different:

```yaml
    # TensorRT-LLM worker
    command: [python3, -m, dynamo.trtllm]
    args:
    - --model-path
    - Qwen/Qwen3-32B
    - --extra-engine-args
    - ./engine_configs/qwen3/agg.yaml
```

You normally do not need to set the top-level `spec.backendFramework` field — the operator infers the backend from the worker command. Set it explicitly (`vllm`, `sglang`, or `trtllm`) only when a feature needs the framework known up front, such as GMS failover or multinode TensorRT-LLM.

For per-backend setup and tuning, see [vLLM](../backends/vllm/README.md), [SGLang](../backends/sglang/README.md), and [TensorRT-LLM](../backends/trtllm/README.md). The rest of this guide uses vLLM.

</Step>

<Step title="Size parallelism for your hardware">

Now decide how many GPUs the worker needs and how to split the model across them. Two questions drive this: **does the model fit**, and **how do you split it across your nodes and GPUs**.

**Will it fit?** A rough lower bound for the weights is `parameters × bytes-per-parameter`: 2 bytes for BF16/FP16, 1 for FP8. Qwen3-32B in BF16 is ~64 GB of weights, so it needs at least two 40 GB GPUs or one 80 GB GPU — and that is before the KV cache, which grows with context length and concurrency. Leave headroom.

**How to split it.** These are engine CLI arguments in the worker command, not dedicated CRD fields. The worker's GPU limit must equal the product of the parallel sizes per node.

| Argument | What it does | Rule of thumb |
|---|---|---|
| `--tensor-parallel-size` (TP) | Splits each layer's tensors across GPUs | Keep **within a single node** (uses NVLink); size up to fit weights + KV cache |
| `--pipeline-parallel-size` (PP) | Splits layers into stages across GPUs | Use **across nodes** when one node can't hold the model |
| `--data-parallel-size` (DP) | Replicates the model for more throughput | Scale out once a single replica meets latency |

For Qwen3-32B on 80 GB GPUs, TP-2 fits the weights with room for KV cache. Add the flag to the launch command and matching GPU resources to the container:

```yaml
  - name: VllmWorker
    type: worker
    podTemplate:
      spec:
        containers:
        - name: main
          env:
          - name: MODEL
            value: Qwen/Qwen3-32B
          command:
          - /bin/bash
          - -c
          - exec python3 -m dynamo.vllm --model $MODEL --tensor-parallel-size 2
          resources:
            limits:
              nvidia.com/gpu: "2"            # must equal TP × PP per node
```

**Context length.** There is no uniform DGD field for maximum context length — it is an engine flag: vLLM's `--max-model-len`, with equivalents on the other backends. Longer context needs more free GPU memory for the KV cache, so it trades off against tensor-parallel size and concurrency. Add it to the launch command when the default does not match your workload, for example `--max-model-len 32000`.

> [!TIP]
> For a first deployment, pick TP to fit the model and move on. To hit a specific TTFT/ITL latency target at minimum cost, [AIConfigurator](../features/disaggregated-serving/aiconfigurator.md) sweeps TP/PP/DP layouts and emits ready-to-use DGD YAML, and the [Profiler](../components/profiler/profiler-guide.md) profiles your model on your hardware and produces a deployable DGD.

The three choices below also shape TP/PP/DP — expand the one that fits your model and cluster.

<AccordionGroup>
  <Accordion title="Aggregated vs. disaggregated serving">

So far the worker does both prefill and decode (**aggregated**). **Disaggregated** serving splits these into separate prefill and decode workers, each sized, scaled, and parallelized independently, with the KV cache transferred between them over the network. This changes the sizing above: you choose TP/PP for prefill and decode separately.

Use aggregated for the simplest deployment or uniform traffic. Move to disaggregated when prefill and decode have different bottlenecks — long prompts saturating prefill while decode sits idle, or vice versa.

A disaggregated graph replaces the single worker with two, tagged by role:

```yaml
  components:
  - name: Frontend
    type: frontend
  - name: VllmPrefillWorker
    type: prefill
    sharedMemorySize: 16Gi
    podTemplate:
      spec:
        containers:
        - name: main
          command: [python3, -m, dynamo.vllm]
          args:
          - --model
          - Qwen/Qwen3-32B
          - --disaggregation-mode
          - prefill
          - --kv-transfer-config
          - '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
  - name: VllmDecodeWorker
    type: decode
    sharedMemorySize: 16Gi
    podTemplate:
      spec:
        containers:
        - name: main
          command: [python3, -m, dynamo.vllm]
          args:
          - --model
          - Qwen/Qwen3-32B
          - --disaggregation-mode
          - decode
          - --kv-transfer-config
          - '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

`type: prefill` and `type: decode` tag the roles, the KV cache moves over NIXL (`--kv-transfer-config`), and `sharedMemorySize` is raised for the transfer. For the full walkthrough — RDMA resources, UCX env vars, and prefill/decode scaling — see [Disaggregated Serving](../features/disaggregated-serving/README.md).

> [!IMPORTANT]
> Disaggregated serving moves KV cache between workers over the network. For acceptable performance the cluster needs RDMA — see the [Disaggregated Communication Guide](disagg-communication-guide.md). Without it, transfers fall back to TCP with severe latency penalties.

  </Accordion>
  <Accordion title="Deploy across multiple nodes">

When a worker needs more GPUs than a single node provides — a large model, or TP/PP that exceeds one node — set `multinode.nodeCount`. Total GPUs = `nodeCount` × the per-node `nvidia.com/gpu` limit, and your tensor/pipeline-parallel sizes must match that total. The operator spreads the leader and worker pods across nodes using Grove or LeaderWorkerSet — you do **not** author those pods yourself.

```yaml
  - name: VllmDecodeWorker
    type: worker
    multinode:
      nodeCount: 2                 # total GPUs = nodeCount × per-node gpu limit
    podTemplate:
      spec:
        containers:
        - name: main
          resources:
            limits:
              nvidia.com/gpu: "8"        # per node → 16 GPUs total
```

Multinode requires a gang scheduler installed first. For the orchestrator choice, the GPU/parallelism math, and per-backend operator behavior, see [Multinode Deployments](deployment/multinode-deployment.md) and the install-time [Multinode Orchestration](multinode-installation.md) prerequisites.

  </Accordion>
  <Accordion title="Mixture-of-Experts: expert parallelism">

Mixture-of-Experts (MoE) models — DeepSeek-R1, Qwen3-235B, Kimi-K2 — add a parallelism axis: **expert parallelism (EP)** distributes the experts across GPUs. In vLLM, enable it with `--enable-expert-parallel`, typically alongside data parallelism for the attention layers:

```yaml
    command:
    - /bin/bash
    - -c
    - |
      exec python3 -m dynamo.vllm --model $MODEL \
        --data-parallel-size 8 \
        --enable-expert-parallel
```

Expert parallelism spans the data-parallel ranks, so the GPU count follows DP (and TP/PP where used) — `nvidia.com/gpu` must still equal the total per node. Wide-EP MoE deployments usually pair this with multinode and a fast all-to-all backend (for example `VLLM_ALL2ALL_BACKEND=deepep_low_latency`). For a worked multinode wide-EP example, see the [DeepSeek-R1 recipe](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-r1/vllm/disagg/deploy_hopper_16gpu.yaml).

  </Accordion>
</AccordionGroup>

</Step>

<Step title="Apply the deployment and check status">

Validate the spec with a server-side dry run, then apply it:

```bash
kubectl apply -f qwen3-32b-agg.yaml -n <namespace> --dry-run=server
kubectl apply -f qwen3-32b-agg.yaml -n <namespace>
```

The DGD is the live serving resource. It becomes ready once the Frontend and workers are up — the `Ready` condition reports `True`. Watch it reconcile:

```bash
kubectl get dynamographdeployment -n <namespace> -w
```

Or block until it is ready:

```bash
kubectl wait --for=condition=Ready dynamographdeployment/qwen3-32b-agg \
  -n <namespace> --timeout=600s
```

If it stalls, `kubectl describe dynamographdeployment <name>` and the operator events surface the cause. Common rejections: a multinode component with neither Grove nor LWS installed, an `nvidia.com/gpu` limit that does not equal tensor-parallel × pipeline-parallel size, or a missing `envFrom` Secret.

</Step>

<Step title="Send a request">

The operator creates a `ClusterIP` Service named `<name>-frontend` on port 8000. Port-forward it and send requests to the OpenAI-compatible API:

```bash
kubectl port-forward svc/qwen3-32b-agg-frontend 8000:8000 -n <namespace>
```

List the served models:

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

Send a chat completion — the `model` field must match the model you deployed:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "messages": [{"role": "user", "content": "What is NVIDIA Dynamo?"}],
    "max_tokens": 200
  }' | python3 -m json.tool
```

For a stable external address instead of `port-forward`, see [Expose the Frontend](dgd-expose-frontend.md).

</Step>

</Steps>

## Put it all together: Qwen3-32B

The following spec assembles the steps above into one aggregated Qwen3-32B deployment, adapted from the [agg-round-robin recipe](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b/vllm/agg-round-robin/deploy.yaml). Comments mark the values you substitute for your model, hardware, and scale:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-32b-agg
spec:
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1   # substitute: release tag
  - name: VllmWorker
    type: worker
    replicas: 8                              # substitute: scale out for throughput
    podTemplate:
      spec:
        containers:
        - name: main
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1   # substitute: release tag
          workingDir: /workspace
          envFrom:
          - secretRef:
              name: hf-token-secret          # substitute: your HuggingFace token Secret name
          env:
          - name: MODEL
            value: Qwen/Qwen3-32B            # substitute: your model
          - name: HF_HOME
            value: /home/dynamo/.cache/huggingface
          command:
          - /bin/bash
          - -c
          - exec python3 -m dynamo.vllm --model $MODEL --tensor-parallel-size 2 --max-model-len 32000
          resources:
            requests:
              nvidia.com/gpu: "2"
            limits:
              nvidia.com/gpu: "2"             # substitute: must equal --tensor-parallel-size
          volumeMounts:
          - name: model-cache                 # substitute: drop the volume + mount if not caching
            mountPath: /home/dynamo/.cache/huggingface
        volumes:
        - name: model-cache                   # substitute: omit to download from HuggingFace per pod
          persistentVolumeClaim:              #   (see Model Caching to create the PVC + download Job)
            claimName: model-cache
```

This runs eight TP-2 workers (16 GPUs). To turn it into one of the variations above — disaggregated, multinode, MoE expert-parallel, cached, KV-routed, or offloaded — apply the change from that step or the matching page below.

## Optional next steps

These are independent capabilities you opt into per workload. None are required for a working deployment.

<CardGroup cols={2}>
  <Card title="Set up KV-Aware Routing" icon="regular route" href="/dynamo/dev/kubernetes/dgd-kv-routing">
    Route requests to the worker that already holds the prompt's KV cache prefix to cut TTFT.
  </Card>
  <Card title="Set up KV Cache Offloading" icon="regular layer-group" href="/dynamo/dev/kubernetes/dgd-kv-offloading">
    Spill KV blocks to host memory or disk to serve longer contexts and reuse cache.
  </Card>
  <Card title="Set up Disaggregated Serving" icon="regular arrows-split-up-and-left" href="/dynamo/dev/features/disaggregated-serving">
    Split prefill and decode into independently scaled workers with KV transfer over RDMA.
  </Card>
  <Card title="Size with AIConfigurator" icon="regular ruler-combined" href="/dynamo/dev/features/disaggregated-serving/aiconfigurator">
    Auto-pick TP/PP/DP and replica counts to meet a TTFT/ITL latency target.
  </Card>
  <Card title="Expose the Frontend" icon="regular globe" href="/dynamo/dev/kubernetes/dgd-expose-frontend">
    Give the Frontend a stable external address with a Kubernetes Ingress, a LoadBalancer Service, or GAIE.
  </Card>
  <Card title="Customize Health Probes" icon="regular heart-pulse" href="/dynamo/dev/kubernetes/dgd-probes">
    Override the operator's default liveness, readiness, and startup probes when needed.
  </Card>
  <Card title="Observability and Metrics" icon="regular chart-line" href="/dynamo/dev/kubernetes/observability/metrics">
    Scrape Prometheus metrics — on by default; opt out with an annotation.
  </Card>
  <Card title="Auto Deploy with DGDR" icon="regular wand-magic-sparkles" href="/dynamo/dev/kubernetes/dgdr-guide">
    Let the Planner pick parallelism and autoscale to an SLA — authored as a DGDR, not a DGD.
  </Card>
</CardGroup>

> [!IMPORTANT]
> **The Planner is not part of the DGD spec.** SLA-driven autoscaling with the [Planner](../components/planner/README.md) is configured through a **DynamoGraphDeploymentRequest (DGDR)** — see [Auto Deploy with DGDR](dgdr-guide.md). For HPA/KEDA scaling of a DGD service, see [Autoscaling](autoscaling.md). For the full field reference, see the [API Reference](api-reference.md).
