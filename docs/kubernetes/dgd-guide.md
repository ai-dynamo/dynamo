---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Deploy with DGD
subtitle: Author a DynamoGraphDeployment spec step by step — choose a model and backend, size parallelism for your hardware, then apply it and send requests.
---

A **DynamoGraphDeployment (DGD)** is the Kubernetes Custom Resource (CRD) that describes how to deploy a model with Dynamo. You write the spec, `kubectl apply` it, and the Dynamo operator reconciles it into the running pods, services, and scheduling resources that serve your model. One DGD describes one inference graph: a Frontend plus one or more workers.

This guide walks through authoring that spec end to end: choose a model, pick a backend, choose aggregated or disaggregated serving, size the topology and parallelism for your hardware, then apply the deployment and send it a request. Each step fills in more of a single DGD spec. Environment-specific values — namespace, runtime image, HuggingFace token — stay in shell variables and are filled into the spec at apply time with `envsubst`, so the same YAML file moves between clusters unchanged.

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

<Step title="Choose your model">

Pick the model you want to serve from HuggingFace. This guide uses **Qwen3-32B** as the running example; substitute your own anywhere you see it.

A few values change per environment — the target namespace, the runtime image, and your HuggingFace token. Rather than hard-code them, keep them in shell variables and fill them into the spec at apply time with `envsubst` (from the `gettext` package). Export the two you know now — you pick the image in the next step:

```bash
export NAMESPACE=your-namespace
export HF_TOKEN=hf_xxxxxxxx                 # your HuggingFace token
```

Gated or rate-limited models need the token as a Secret in your namespace. Create it once from `$HF_TOKEN`:

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

Start the spec with a Frontend and one worker. The `${NAMESPACE}` and `${RUNTIME_IMAGE}` placeholders are expanded by `envsubst` when you apply (see the *Apply* step); the worker pulls the token from the Secret via `envFrom`, and the model name is passed to the engine as an argument. The worker's launch command is backend-specific — this guide defaults to **vLLM**; select your backend and the choice syncs across every code block on this page (you set the matching image in the next step):

<Tabs>
<Tab title="vLLM" language="vllm">

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-32b-agg
  namespace: ${NAMESPACE}
spec:
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
  - name: VllmWorker
    type: worker
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
          envFrom:
          - secretRef:
              name: hf-token-secret
          command:
          - python3
          - -m
          - dynamo.vllm
          args:
          - --model
          - Qwen/Qwen3-32B
```

</Tab>
<Tab title="SGLang" language="sglang">

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-32b-agg
  namespace: ${NAMESPACE}
spec:
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
  - name: SGLangWorker
    type: worker
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
          envFrom:
          - secretRef:
              name: hf-token-secret
          command:
          - python3
          - -m
          - dynamo.sglang
          args:
          - --model-path
          - Qwen/Qwen3-32B
          - --served-model-name
          - Qwen/Qwen3-32B
          - --page-size
          - "16"
          - --tp
          - "1"
          - --trust-remote-code
          - --skip-tokenizer-init
```

</Tab>
<Tab title="TensorRT-LLM" language="trtllm">

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-32b-agg
  namespace: ${NAMESPACE}
spec:
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
  - name: TRTLLMWorker
    type: worker
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
          envFrom:
          - secretRef:
              name: hf-token-secret
          command:
          - python3
          - -m
          - dynamo.trtllm
          args:
          - --model-path
          - Qwen/Qwen3-32B
          - --served-model-name
          - Qwen/Qwen3-32B
          - --extra-engine-args
          - ./engine_configs/qwen3/agg.yaml
```

</Tab>
</Tabs>

Substitute `Qwen/Qwen3-32B` with your model, `hf-token-secret` with your Secret name if you changed it. Keeping the namespace, image, and token in the environment means the same YAML file works across clusters — you change the exports, not the spec.

<Tip>
**Large model?** By default every worker pod downloads the weights from HuggingFace on startup — slow for big models and prone to rate limits across many replicas. Cache the weights once on a shared volume and mount them into each worker. See [Model Caching](model-caching.md) (and [ModelExpress](modelexpress.md) for fleet-scale distribution).
</Tip>

</Step>

<Step title="Choose your backend">

The worker in the previous step runs **vLLM**. Dynamo also supports **SGLang** and **TensorRT-LLM**, and the backend you picked there already synced every code block on this page. The backend sets three things: the runtime image, the launch module (`dynamo.vllm` / `dynamo.sglang` / `dynamo.trtllm`), and how you pass the model and parallelism. Export the matching image so `envsubst` fills the `${RUNTIME_IMAGE}` placeholder on both the Frontend and the worker:

<Tabs>
<Tab title="vLLM" language="vllm">

```bash
export RUNTIME_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
```

</Tab>
<Tab title="SGLang" language="sglang">

```bash
export RUNTIME_IMAGE=nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.1
```

</Tab>
<Tab title="TensorRT-LLM" language="trtllm">

```bash
export RUNTIME_IMAGE=nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.2.1
```

</Tab>
</Tabs>

The launch module and the way each engine takes the model differ, which is why the worker block changed when you switched tabs above:

| | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| **Image** | `vllm-runtime` | `sglang-runtime` | `tensorrtllm-runtime` |
| **Launch module** | `dynamo.vllm` | `dynamo.sglang` | `dynamo.trtllm` |
| **Model argument** | `--model` | `--model-path` (+ `--served-model-name`) | `--model-path` (+ `--served-model-name`) |
| **Where parallelism lives** | CLI flags | CLI flags | engine-config YAML (`--extra-engine-args`) |

vLLM and SGLang take tensor/pipeline/data parallelism as CLI flags (covered in the topology step). TensorRT-LLM is the exception: it reads **all** parallelism from the engine-config YAML referenced by `--extra-engine-args`, not from worker flags.

<Info>
**Check the model works on your hardware and backend before you deploy.** Not every model, GPU, and backend combination is valid — a quantization can be unsupported on your GPU architecture (for example, FP8 needs Hopper/Ada or newer; some FP4/NVFP4 paths need Blackwell), or an engine may not yet implement a model's architecture. Confirm against the backend's supported-models list ([vLLM](../backends/vllm/README.md), [SGLang](../backends/sglang/README.md), [TensorRT-LLM](../backends/trtllm/README.md)) and, for sizing, the [AIConfigurator support matrix](../features/disaggregated-serving/aiconfigurator.md).
</Info>

You normally do not need to set the top-level `spec.backendFramework` field — the operator infers the backend from the worker command. Set it explicitly (`vllm`, `sglang`, or `trtllm`) only when a feature needs the framework known up front, such as GMS failover or multinode TensorRT-LLM.

For per-backend setup and tuning, see [vLLM](../backends/vllm/README.md), [SGLang](../backends/sglang/README.md), and [TensorRT-LLM](../backends/trtllm/README.md).

<Note>
**Private registry?** The operator **auto-discovers and injects** image pull secrets by matching Docker config Secrets in the namespace against your image's registry host, so private registries usually work with no extra config. To take manual control for a component, set the `nvidia.com/disable-image-pull-secret-discovery: "true"` annotation on it and list your own `imagePullSecrets` under `podTemplate.spec`.
</Note>

</Step>

<Step title="Choose aggregated or disaggregated serving">

Before sizing hardware, decide how prefill (prompt processing) and decode (token generation) are placed. Both patterns work with the same DGD spec — you are choosing how many worker roles it has.

- **Aggregated** — the single worker you have been building does both phases. Simplest to deploy and debug; a good default for smaller models and uniform traffic.
- **Disaggregated** — separate prefill and decode worker pools, each sized, scaled, and parallelized independently, with the KV cache transferred between them over the network. Reach for it when prefill and decode have different bottlenecks — long prompts saturating prefill while decode sits idle, or high concurrency saturating decode.

If you are staying aggregated, keep the single worker and continue to the next step. To go disaggregated, replace the one worker with two, tagged by role. The tabs below use **vLLM** as a concrete example to highlight the structural difference — one worker versus a prefill/decode pair; SGLang and TensorRT-LLM follow the same pattern with their own flags:

<Tabs>
<Tab title="Aggregated (1 worker)">

```yaml
  components:
  - name: Frontend
    type: frontend
  - name: VllmWorker
    type: worker
    podTemplate:
      spec:
        containers:
        - name: main
          command: [python3, -m, dynamo.vllm]
          args:
          - --model
          - Qwen/Qwen3-32B
```

</Tab>
<Tab title="Disaggregated (prefill + decode)">

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

</Tab>
</Tabs>

In the disaggregated spec, `type: prefill` and `type: decode` tag the roles, the KV cache moves over NIXL (`--kv-transfer-config`), and `sharedMemorySize` is raised for the transfer. When you disaggregate, the next step's sizing applies to prefill and decode **separately** — you pick TP/PP for each role.

The fastest way to start is to copy a backend template rather than type the spec from scratch. Each backend ships `agg.yaml` and `disagg.yaml` under its `deploy/` folder — for vLLM, [`examples/backends/vllm/deploy/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy/README.md) (SGLang and TensorRT-LLM have equivalent `deploy/` folders). Adapt the model, image, and parallelism to your case.

For disaggregation-specific configuration — RDMA resources, UCX environment variables, and prefill/decode scaling — see [Disaggregated Serving](../features/disaggregated-serving/README.md).

<Info>
Disaggregated serving moves KV cache between workers over the network. For acceptable performance the cluster needs RDMA — see the [Disaggregated Communication Guide](disagg-communication-guide.md). Without it, transfers fall back to TCP with severe latency penalties.
</Info>

</Step>

<Step title="Determine topology and parallelism">

Now decide how many GPUs each worker needs and how to split the model across them. (If you disaggregated in the previous step, answer this once for the prefill worker and once for the decode worker — they size independently.)

**Will it fit?** A rough lower bound for the weights is `parameters × bytes-per-parameter`: 2 bytes for BF16/FP16, 1 for FP8. Qwen3-32B in BF16 is ~64 GB of weights, so it needs at least two 40 GB GPUs or one 80 GB GPU — and that is before the KV cache, which grows with context length and concurrency. Leave headroom.

**The parallelism knobs.** A worker's total GPUs equals the product of its parallel sizes; when it spans machines, that product also equals `multinode.nodeCount × per-node GPUs`. Five knobs do different jobs:

- **Tensor parallelism (TP)** splits every layer's weight matrices across GPUs so one model copy spans them, cutting memory per GPU. The split is communication-heavy (an all-reduce every layer), so TP wants fast intra-node interconnect (NVLink) — keep it **within a node**.
- **Pipeline parallelism (PP)** splits the model into sequential stages — contiguous blocks of layers, one block per GPU or node. It also cuts memory per GPU, but only passes activations at stage boundaries, so it tolerates slower links and is how you span **across nodes**.
- **Data parallelism (DP) / replicas** runs multiple independent model copies for throughput. It does **not** help a model fit — each copy still needs enough GPUs to hold the model. Scale whole pods with `replicas`, or use the engine's own `--data-parallel-size` (common with MoE attention).
- **Expert parallelism (EP)** applies to Mixture-of-Experts models only: it spreads a layer's experts across GPUs so no single GPU holds them all.
- **`multinode.nodeCount`** is infrastructure, not a model split: it tells the operator how many physical nodes one worker spans. You set it when TP×PP exceeds the GPUs on a single node.

The exact flag differs per backend. vLLM and SGLang take them as CLI args on the worker; TensorRT-LLM reads them from the engine-config YAML instead:

| Knob | vLLM (CLI) | SGLang (CLI) | TensorRT-LLM (engine YAML) |
|---|---|---|---|
| **TP** | `--tensor-parallel-size` | `--tp` | `tensor_parallel_size` |
| **PP** | `--pipeline-parallel-size` | `--pp` | `pipeline_parallel_size` |
| **DP** | `--data-parallel-size` | `--dp` | `enable_attention_dp` |
| **EP** (MoE) | `--enable-expert-parallel` | `--ep-size` | `moe_expert_parallel_size` |
| **MoE TP** | (follows TP) | `--tp` | `moe_tensor_parallel_size` |
| **nodeCount** | `multinode.nodeCount` | `multinode.nodeCount` | `multinode.nodeCount` |

**How to choose.** Work from where the model fits:

- **Fits on one GPU** — no parallelism needed. Leave TP/PP/DP at 1, request one GPU, and skip to the *Apply* step. Add replicas later only for throughput.
- **Doesn't fit on one GPU, fits on one node** — raise **TP** to the number of GPUs the weights (plus KV cache headroom) require, and set the worker's `nvidia.com/gpu` limit to match. TP keeps one model copy on the fast intra-node fabric.
- **Doesn't fit on one node** — combine **PP** with `multinode.nodeCount` so contiguous layer blocks land on separate nodes, typically with TP within each node. Total GPUs = TP × PP = `nodeCount × per-node GPUs`.
- **Need more throughput** — the model already fits; add **replicas** (or `--data-parallel-size`) to run more copies. This scales requests-per-second, not model size.
- **Mixture-of-Experts** — add **EP** on top of the above (see the MoE note below).

To pick actual numbers, start from a **[recipe](https://github.com/ai-dynamo/dynamo/tree/main/recipes)**: if one matches your model, backend, and hardware, copy its TP/PP/DP and adapt. If none fits, use **AIConfigurator**, which sweeps and ranks TP/PP/DP + replica configs for your model, GPU SKU, and latency target — see [Sizing with AIConfigurator](../features/disaggregated-serving/aiconfigurator.md).

**Worked example.** Qwen3-32B in BF16 doesn't fit on one 80 GB GPU with room for KV cache, but TP-2 does. Set the parallel size on the worker and match the GPU limit — for TensorRT-LLM the size lives in the engine-config YAML instead, so only the GPU limit changes on the worker:

<Tabs>
<Tab title="vLLM" language="vllm">

```yaml
  - name: VllmWorker
    type: worker
    podTemplate:
      spec:
        containers:
        - name: main
          command: [python3, -m, dynamo.vllm]
          args:
          - --model
          - Qwen/Qwen3-32B
          - --tensor-parallel-size
          - "2"
          resources:
            limits:
              nvidia.com/gpu: "2"            # must equal TP × PP
```

</Tab>
<Tab title="SGLang" language="sglang">

```yaml
  - name: SGLangWorker
    type: worker
    podTemplate:
      spec:
        containers:
        - name: main
          command: [python3, -m, dynamo.sglang]
          args:
          - --model-path
          - Qwen/Qwen3-32B
          - --served-model-name
          - Qwen/Qwen3-32B
          - --tp
          - "2"
          resources:
            limits:
              nvidia.com/gpu: "2"            # must equal TP × PP
```

</Tab>
<Tab title="TensorRT-LLM" language="trtllm">

```yaml
  - name: TRTLLMWorker
    type: worker
    podTemplate:
      spec:
        containers:
        - name: main
          command: [python3, -m, dynamo.trtllm]
          args:
          - --model-path
          - Qwen/Qwen3-32B
          - --served-model-name
          - Qwen/Qwen3-32B
          - --extra-engine-args
          - ./engine_configs/qwen3/agg.yaml  # set tensor_parallel_size: 2 here
          resources:
            limits:
              nvidia.com/gpu: "2"            # must equal TP × PP from the engine config
```

</Tab>
</Tabs>

<Info>
**TensorRT-LLM sets parallelism in the engine config, not on the CLI.** TP, PP, and MoE parallelism are keys in the engine-config YAML referenced by `--extra-engine-args` — `tensor_parallel_size`, `pipeline_parallel_size`, `moe_tensor_parallel_size`, and `moe_expert_parallel_size` — not worker flags. Edit them there. Start from a per-model config under [`examples/backends/trtllm/engine_configs/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/engine_configs/README.md) (it ships `agg.yaml`, `prefill.yaml`, and `decode.yaml` variants), and keep the worker's `nvidia.com/gpu` limit equal to the product of those sizes.
</Info>

**Context length.** There is no uniform DGD field for maximum context length — it is an engine flag: `--max-model-len` (vLLM), `--context-length` (SGLang), or `max_seq_len` in the engine config (TensorRT-LLM). Longer context needs more free GPU memory for the KV cache, so it trades off against parallel size and concurrency. Set it when the default does not match your workload, for example `--max-model-len 32000`.

**Mixture-of-Experts.** MoE models — DeepSeek-R1, Qwen3-235B, Kimi-K2 — add **expert parallelism (EP)** on top of the knobs above, and expose it as **two** sizes: how experts are split (EP) and how the non-expert (attention) weights are split (a tensor-parallel size for the MoE layers). In vLLM, enable EP with `--enable-expert-parallel`, typically alongside `--data-parallel-size` for the attention layers. In SGLang, use `--ep-size` with `--enable-dp-attention`. In TensorRT-LLM, set both `moe_expert_parallel_size` and `moe_tensor_parallel_size` in the engine config. Wide-EP deployments usually pair this with multinode and a fast all-to-all backend (for example `VLLM_ALL2ALL_BACKEND=deepep_low_latency`). For a worked multinode wide-EP example, see the [DeepSeek-R1 recipe](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-r1/vllm/disagg/deploy_hopper_16gpu.yaml).

**Multinode.** When TP × PP exceeds the GPUs on one node, set `multinode.nodeCount` on the worker so it spans machines; the operator schedules the pods and wires the engine's cross-node communication (Ray for vLLM, `--dist-init-addr`/`--nnodes` for SGLang, MPI for TensorRT-LLM). This needs Grove or LWS installed. See [Multinode Deployments](deployment/multinode-deployment.md) and the `disagg-multinode.yaml` templates under each backend's `deploy/` folder.

<Warning>
**Leave room for the KV cache.** The weights are only part of GPU memory — the KV cache grows with context length and concurrency, and if it has no room the worker OOMs at load or under traffic. Each engine caps the fraction of GPU memory it will use: `--gpu-memory-utilization` (vLLM, default 0.90), `--mem-fraction-static` (SGLang), or `free_gpu_memory_fraction` in the engine config (TensorRT-LLM). If you hit OOM, lower the fraction, add a GPU (raise TP), or reduce max context length.
</Warning>

</Step>

<Step title="Apply the deployment and check status">

You have exported `NAMESPACE` and `RUNTIME_IMAGE` (and created the token Secret). Expand the placeholders and validate the spec with a server-side dry run, then apply it — pipe the file through `envsubst` both times:

```bash
envsubst < qwen3-32b-agg.yaml | kubectl apply -f - -n ${NAMESPACE} --dry-run=server
envsubst < qwen3-32b-agg.yaml | kubectl apply -f - -n ${NAMESPACE}
```

<Note>
`envsubst` only substitutes `${VAR}` tokens. Confirm the exports are set (`echo ${NAMESPACE} ${RUNTIME_IMAGE}`) before applying, or the placeholders expand to empty strings. To see the fully rendered spec first, run `envsubst < qwen3-32b-agg.yaml | less`.
</Note>

The DGD is the live serving resource. It becomes ready once the Frontend and workers are up — the `Ready` condition reports `True`. Watch it reconcile:

```bash
kubectl get dynamographdeployment -n ${NAMESPACE} -w
```

Or block until it is ready:

```bash
kubectl wait --for=condition=Ready dynamographdeployment/qwen3-32b-agg \
  -n ${NAMESPACE} --timeout=600s
```

If it stalls, `kubectl describe dynamographdeployment <name> -n ${NAMESPACE}` and the operator events surface the cause. Common rejections: a multinode component with neither Grove nor LWS installed, an `nvidia.com/gpu` limit that does not equal tensor-parallel × pipeline-parallel size, or a missing `envFrom` Secret.

</Step>

<Step title="Send a request">

The operator creates a `ClusterIP` Service named `<name>-frontend` on port 8000. Port-forward it and send requests to the OpenAI-compatible API:

```bash
kubectl port-forward svc/qwen3-32b-agg-frontend 8000:8000 -n ${NAMESPACE}
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

The following spec assembles the steps above into one aggregated Qwen3-32B deployment, adapted from the [agg-round-robin recipe](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b/vllm/agg-round-robin/deploy.yaml). It reflects the backend you selected earlier. Comments mark the values you substitute for your model, hardware, and scale:

<Tabs>
<Tab title="vLLM" language="vllm">

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-32b-agg
  namespace: ${NAMESPACE}
spec:
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}             # export: vllm-runtime image + tag
  - name: VllmWorker
    type: worker
    replicas: 8                               # substitute: scale out for throughput
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}             # export: vllm-runtime image + tag
          workingDir: /workspace
          envFrom:
          - secretRef:
              name: hf-token-secret           # substitute: your HuggingFace token Secret name
          env:
          - name: HF_HOME
            value: /home/dynamo/.cache/huggingface
          command:
          - python3
          - -m
          - dynamo.vllm
          args:
          - --model
          - Qwen/Qwen3-32B                     # substitute: your model
          - --tensor-parallel-size
          - "2"
          - --max-model-len
          - "32000"
          resources:
            requests:
              nvidia.com/gpu: "2"
            limits:
              nvidia.com/gpu: "2"              # substitute: must equal --tensor-parallel-size
          volumeMounts:
          - name: model-cache                  # substitute: drop the volume + mount if not caching
            mountPath: /home/dynamo/.cache/huggingface
        volumes:
        - name: model-cache                    # substitute: omit to download from HuggingFace per pod
          persistentVolumeClaim:               #   (see Model Caching to create the PVC + download Job)
            claimName: model-cache
```

</Tab>
<Tab title="SGLang" language="sglang">

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-32b-agg
  namespace: ${NAMESPACE}
spec:
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}             # export: sglang-runtime image + tag
  - name: SGLangWorker
    type: worker
    replicas: 8                               # substitute: scale out for throughput
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}             # export: sglang-runtime image + tag
          workingDir: /workspace
          envFrom:
          - secretRef:
              name: hf-token-secret           # substitute: your HuggingFace token Secret name
          env:
          - name: HF_HOME
            value: /home/dynamo/.cache/huggingface
          command:
          - python3
          - -m
          - dynamo.sglang
          args:
          - --model-path
          - Qwen/Qwen3-32B                     # substitute: your model
          - --served-model-name
          - Qwen/Qwen3-32B
          - --page-size
          - "16"
          - --tp
          - "2"
          - --context-length
          - "32000"
          - --trust-remote-code
          - --skip-tokenizer-init
          resources:
            requests:
              nvidia.com/gpu: "2"
            limits:
              nvidia.com/gpu: "2"              # substitute: must equal --tp
          volumeMounts:
          - name: model-cache                  # substitute: drop the volume + mount if not caching
            mountPath: /home/dynamo/.cache/huggingface
        volumes:
        - name: model-cache                    # substitute: omit to download from HuggingFace per pod
          persistentVolumeClaim:               #   (see Model Caching to create the PVC + download Job)
            claimName: model-cache
```

</Tab>
<Tab title="TensorRT-LLM" language="trtllm">

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-32b-agg
  namespace: ${NAMESPACE}
spec:
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}             # export: tensorrtllm-runtime image + tag
  - name: TRTLLMWorker
    type: worker
    replicas: 8                               # substitute: scale out for throughput
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}             # export: tensorrtllm-runtime image + tag
          workingDir: /workspace
          envFrom:
          - secretRef:
              name: hf-token-secret           # substitute: your HuggingFace token Secret name
          env:
          - name: HF_HOME
            value: /home/dynamo/.cache/huggingface
          command:
          - python3
          - -m
          - dynamo.trtllm
          args:
          - --model-path
          - Qwen/Qwen3-32B                     # substitute: your model
          - --served-model-name
          - Qwen/Qwen3-32B
          - --extra-engine-args
          - ./engine_configs/qwen3/agg.yaml    # set tensor_parallel_size: 2, max_seq_len: 32000 here
          resources:
            requests:
              nvidia.com/gpu: "2"
            limits:
              nvidia.com/gpu: "2"              # substitute: must equal TP × PP from the engine config
          volumeMounts:
          - name: model-cache                  # substitute: drop the volume + mount if not caching
            mountPath: /home/dynamo/.cache/huggingface
        volumes:
        - name: model-cache                    # substitute: omit to download from HuggingFace per pod
          persistentVolumeClaim:               #   (see Model Caching to create the PVC + download Job)
            claimName: model-cache
```

</Tab>
</Tabs>


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
  <Card title="Size with AIConfigurator" icon="regular ruler-combined" href="/dynamo/dev/kubernetes/dgd-aiconfigurator">
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

<Info>
**The Planner is not part of the DGD spec.** SLA-driven autoscaling with the [Planner](../components/planner/README.md) is configured through a **DynamoGraphDeploymentRequest (DGDR)** — see [Auto Deploy with DGDR](dgdr-guide.md). For HPA/KEDA scaling of a DGD service, see [Autoscaling](autoscaling.md). For the full field reference, see the [API Reference](api-reference.md).
</Info>
