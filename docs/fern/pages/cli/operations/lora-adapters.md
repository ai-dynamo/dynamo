---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: LoRA Adapters
subtitle: Serve fine-tuned LoRA adapters with dynamic loading and routing in Dynamo
---

## What Are LoRA Adapters

LoRA (Low-Rank Adaptation) serves specialized model variants without duplicating full base weights. In Dynamo you can load adapters at runtime, route inference to workers that hold them, and manage them declaratively on Kubernetes. It provides:

- **Dynamic loading**: Load and unload adapters without restarting workers
- **Multiple sources**: `file://`, `s3://`, or `hf://` URIs
- **Automatic caching**: Downloaded adapters are cached under `DYN_LORA_PATH`
- **Discovery**: Loaded adapters appear in `/v1/models`
- **KV-aware routing**: Route requests to workers with the matching adapter and cached prefix blocks
- **Kubernetes native**: Declarative loading through the `DynamoModel` CRD

<AccordionGroup>
  <Accordion title="Architecture">
    ```mermaid
    flowchart TD
        Frontend["Frontend"] --> Router["Router<br/>(LoRA-aware)"]
        Router --> Workers["Workers<br/>(LoRA-loaded)"]
        Workers --> ManagerNode["LoRA Manager"]

        subgraph ManagerGroup["LoRA Manager"]
            Downloader
            Cache
        end

        ManagerNode --> Local["file://<br/>Local"]
        ManagerNode --> S3["s3://<br/>S3/MinIO"]
        ManagerNode --> HF["hf://<br/>(custom)"]
    ```

    - **Rust core** (`lib/llm/src/lora/`): Downloading, caching, and validation
    - **Python manager** (`components/src/dynamo/common/lora/`): Custom source support
    - **Worker handlers** (`components/src/dynamo/vllm/handlers.py`): Load/unload API and inference integration
  </Accordion>
</AccordionGroup>

## Serve a LoRA Adapter

<Tabs>
<Tab title="Kubernetes">

<Steps>
  <Step title="Prerequisites">
    - A Kubernetes cluster with the Dynamo Platform installed and a vLLM runtime image
    - A LoRA adapter compatible with your base model
    - For `s3://` sources: AWS credentials in a Kubernetes Secret

    Full manifests and MinIO setup: [Kubernetes LoRA deployment example](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy/lora).
  </Step>

  <Step title="Deploy a LoRA-enabled worker">
    Enable LoRA on the **worker**: set LoRA and system API environment variables, and pass vLLM LoRA flags in `args:`. `DYN_SYSTEM_ENABLED` and `DYN_SYSTEM_PORT` expose load/unload on the worker system port.

    Adapted from the [`agg_lora.yaml` example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/lora/agg_lora.yaml):

    ```yaml
    apiVersion: nvidia.com/v1beta1
    kind: DynamoGraphDeployment
    metadata:
      name: vllm-agg-lora
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
      - name: VllmDecodeWorker
        type: decode
        replicas: 1
        podTemplate:
          spec:
            containers:
            - name: main
              image: ${RUNTIME_IMAGE}
              workingDir: /workspace/examples/backends/vllm
              envFrom:
              - secretRef:
                  name: hf-token-secret
              env:
              - name: DYN_LORA_ENABLED
                value: "true"
              - name: DYN_LORA_PATH
                value: /tmp/dynamo_loras
              - name: DYN_SYSTEM_ENABLED
                value: "true"
              - name: DYN_SYSTEM_PORT
                value: "9090"
              command:
              - python3
              - -m
              - dynamo.vllm
              args:
              - --model
              - Qwen/Qwen3-0.6B
              - --enable-lora
              - --max-lora-rank
              - "64"
              - --enforce-eager
              resources:
                limits:
                  nvidia.com/gpu: "1"
    ```

    Apply and wait for readiness:

    ```bash
    kubectl apply -f vllm-agg-lora.yaml -n ${NAMESPACE}
    kubectl wait --for=condition=Ready dynamographdeployment/vllm-agg-lora \
      -n ${NAMESPACE} --timeout=600s
    ```

    **Worker environment variables**

    | Variable | Description | Default |
    |----------|-------------|---------|
    | `DYN_LORA_ENABLED` | Enable LoRA adapter support | `false` |
    | `DYN_LORA_PATH` | Local cache directory for downloaded LoRAs | `~/.cache/dynamo_loras` |
    | `DYN_SYSTEM_ENABLED` | Expose worker load/unload API | `false` |
    | `DYN_SYSTEM_PORT` | System API port | — |
    | `AWS_ACCESS_KEY_ID` | S3 access key (for `s3://` URIs) | — |
    | `AWS_SECRET_ACCESS_KEY` | S3 secret key (for `s3://` URIs) | — |
    | `AWS_ENDPOINT` | Custom S3 endpoint (MinIO, etc.) | — |
    | `AWS_REGION` | AWS region | `us-east-1` |
    | `AWS_ALLOW_HTTP` | Allow HTTP (non-TLS) connections | `false` |

    **vLLM worker arguments**

    | Argument | Description |
    |----------|-------------|
    | `--enable-lora` | Enable LoRA adapter support in vLLM |
    | `--max-lora-rank` | Maximum LoRA rank (must be >= your adapter's rank) |
    | `--max-loras` | Maximum number of LoRAs loaded simultaneously |

    > [!WARNING]
    > Set `--max-lora-rank` to at least your adapter's rank. A lower value causes load failures.

    <AccordionGroup>
      <Accordion title="S3 or MinIO credentials on the worker">
        Store secret keys in a Kubernetes Secret and set non-secret endpoint/region inline:

        ```yaml
                  env:
                  - name: DYN_LORA_ENABLED
                    value: "true"
                  - name: AWS_ENDPOINT
                    value: http://minio:9000        # for MinIO; omit for AWS S3
                  - name: AWS_REGION
                    value: us-east-1
                  - name: AWS_ALLOW_HTTP
                    value: "true"                    # MinIO/non-TLS only
                  - name: AWS_ACCESS_KEY_ID
                    valueFrom:
                      secretKeyRef:
                        name: minio-secret
                        key: AWS_ACCESS_KEY_ID
                  - name: AWS_SECRET_ACCESS_KEY
                    valueFrom:
                      secretKeyRef:
                        name: minio-secret
                        key: AWS_SECRET_ACCESS_KEY
        ```
      </Accordion>
    </AccordionGroup>
  </Step>

  <Step title="Load an adapter">
    Use a `DynamoModel` CRD for declarative, cluster-native loading. It discovers worker endpoints for `baseModelName`, creates a Service, and calls the load API on each pod.

    ```yaml
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoModel
    metadata:
      name: customer-support-lora
      namespace: dynamo-system
    spec:
      modelName: customer-support-adapter-v1
      baseModelName: Qwen/Qwen3-0.6B  # Must match the worker --model value
      modelType: lora
      source:
        uri: s3://my-models-bucket/loras/customer-support/v1
    ```

    Verify readiness:

    ```bash
    kubectl get dynamomodel customer-support-lora
    # NAME                    TOTAL   READY   AGE
    # customer-support-lora   2       2       30s
    ```

    See [Managing Models with DynamoModel](../../developer-guide/knowledge-base/kubernetes/kubernetes-operator/managing-models-dynamomodel.md) for the full CRD workflow.

    <AccordionGroup>
      <Accordion title="Load imperatively with the worker system API">
        Port-forward the worker system port and POST to `/v1/loras`:

        ```bash
        kubectl port-forward svc/vllm-agg-lora-vllmdecodeworker 9090:9090 -n ${NAMESPACE}
        ```

        ```bash
        curl -X POST http://localhost:9090/v1/loras \
          -H "Content-Type: application/json" \
          -d '{
            "lora_name": "customer-support-lora",
            "source": {
              "uri": "s3://my-loras/customer-support-v1"
            }
          }'
        ```

        List loaded adapters with `GET /v1/loras`. Unload with `DELETE /v1/loras/{lora_name}`.
      </Accordion>
    </AccordionGroup>
  </Step>

  <Step title="Run inference">
    Port-forward the Frontend and set the request `model` field to the adapter name (`lora_name` or `DynamoModel` `modelName`):

    ```bash
    kubectl port-forward svc/vllm-agg-lora-frontend 8000:8000 -n ${NAMESPACE}
    ```

    ```bash
    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "customer-support-lora",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
      }'
    ```

    > [!WARNING]
    > The `model` field is case-sensitive and must match the loaded adapter name exactly. For disaggregated serving, load the adapter on both prefill and decode workers.
  </Step>
</Steps>

</Tab>
<Tab title="Local">

<Steps>
  <Step title="Prerequisites">
    - Dynamo with the vLLM backend installed and a CUDA GPU
    - A LoRA adapter compatible with your base model

    For S3-compatible local storage, run the MinIO setup script first:

    ```bash
    cd examples/backends/vllm/launch/lora
    ./setup_minio.sh
    ```

    See the [local LoRA example](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/launch/lora) for the full workflow.
  </Step>

  <Step title="Deploy a LoRA-enabled worker">
    Launch the frontend and a LoRA-enabled vLLM worker:

    ```bash
    cd examples/backends/vllm/launch/lora
    ./agg_lora.sh
    ```

    The script sets `DYN_LORA_ENABLED`, `DYN_LORA_PATH`, MinIO credentials, and starts `dynamo.frontend` plus `dynamo.vllm` with `--enable-lora` and `--max-lora-rank 64`. Override the base model with the `MODEL` environment variable.

    To run the commands manually instead:

    ```bash
    export DYN_LORA_ENABLED=true
    export DYN_LORA_PATH=/tmp/dynamo_loras
    export DYN_SYSTEM_ENABLED=true
    export DYN_SYSTEM_PORT=8081

    python -m dynamo.frontend &
    python -m dynamo.vllm --model Qwen/Qwen3-0.6B \
      --enable-lora --max-lora-rank 64 --enforce-eager
    ```

    Set `AWS_*` variables when loading from `s3://` URIs (the launch script configures MinIO defaults).
  </Step>

  <Step title="Load an adapter">
    POST to the worker system API on `DYN_SYSTEM_PORT` (default `8081` in `agg_lora.sh`):

    ```bash
    curl -X POST http://localhost:8081/v1/loras \
      -H "Content-Type: application/json" \
      -d '{
        "lora_name": "my-lora",
        "source": {
          "uri": "s3://my-loras/my-lora"
        }
      }'
    ```

    Use `file:///path/to/my-lora` for local adapters. List loaded adapters with `GET /v1/loras`. Unload with `DELETE /v1/loras/{lora_name}`.
  </Step>

  <Step title="Run inference">
    Send a chat completion request to the frontend (default port `8000`) with `model` set to the adapter name:

    ```bash
    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "my-lora",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
      }'
    ```

    > [!WARNING]
    > The `model` field is case-sensitive and must match `lora_name` exactly. For disaggregated serving, load the adapter on both prefill and decode workers.
  </Step>
</Steps>

</Tab>
</Tabs>

## KV Cache-Aware LoRA Routing

> [!IMPORTANT]
> With `DYN_LORA_ENABLED`, only KV, random, and round-robin routing are LoRA-aware.
> Direct, power-of-two, least-loaded, and device-aware-weighted modes fail startup.
> Session affinity with LoRA is supported only in KV mode; random and round-robin
> plus affinity are rejected.

<AccordionGroup>
  <Accordion title="How LoRA-aware KV routing works">
    When KV-aware routing is enabled, the router accounts for LoRA adapter identity when computing block hashes:

    - **Distinct hash spaces per adapter**: Blocks cached under adapter `A` are never confused with adapter `B` or the base model, even when token sequences match. The adapter name is mixed into the `LocalBlockHash` computation.
    - **Prefix sharing within the same adapter**: Requests targeting the same LoRA adapter reuse KV prefix blocks like base-model requests.
    - **No extra configuration**: The LoRA name propagates through KV events (`BlockStored`) from the engine to the router. The router uses the `lora_name` field to route requests to workers with matching cached blocks.

    This works across the publisher pipeline, the KV consolidator, and the routing query path.

    For a local two-worker demo with KV-aware routing, run [`agg_lora_router.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/lora/agg_lora_router.sh) and load the adapter on both worker system ports.
  </Accordion>
</AccordionGroup>

## Troubleshooting

<AccordionGroup>
  <Accordion title="LoRA fails to load">
    **Check S3 connectivity:**

    ```bash
    aws --endpoint-url=$AWS_ENDPOINT s3 ls s3://my-loras/ --recursive
    ```

    **Check the cache directory:**

    ```bash
    ls -la ~/.cache/dynamo_loras/
    # or the path set in DYN_LORA_PATH
    ```

    **Check worker logs:**

    ```bash
    kubectl logs deployment/my-worker | grep -i lora
    ```

    Confirm `--max-lora-rank` is at least your adapter's rank.
  </Accordion>

  <Accordion title="Model not found after loading">
    - Verify the LoRA name matches exactly (case-sensitive)
    - List loaded adapters: `curl http://localhost:9090/v1/loras` (port-forward the worker's `DYN_SYSTEM_PORT` first on Kubernetes)
    - Check worker logs for discovery registration errors
  </Accordion>

  <Accordion title="Inference returns the base model response">
    - Confirm the request `model` field matches the loaded `lora_name`
    - Verify the adapter is loaded on the worker handling the request
    - For disaggregated serving, load the adapter on both prefill and decode workers
  </Accordion>
</AccordionGroup>

## Backend Support

| Backend | Status | Notes |
|---------|--------|-------|
| vLLM | <Badge intent="success" minimal>Yes</Badge> | Full support including KV-aware routing |
| SGLang | <Badge intent="warning" minimal>WIP</Badge> | Support is in progress |
| TensorRT-LLM | <Badge intent="note" minimal>No</Badge> | Not yet supported |

See the [feature support matrix](../../reference/general/compatibility.mdx#feature-support) for full compatibility details.
