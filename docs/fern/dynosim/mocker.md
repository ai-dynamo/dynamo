---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Simulate a Kubernetes Deployment with Mocker
subtitle: Exercise the Dynamo frontend, router, and worker lifecycle without GPUs
---

Mocker is a simulated Dynamo backend. It registers workers, publishes KV events, and returns mock
responses while exercising the same frontend and routing path as a model-serving deployment.

Use this tutorial to deploy Mocker with a `DynamoGraphDeployment` (DGD). For local development, see
[Simulate a Local Deployment](mocker-local.md). For configuration details, see the
[Mocker CLI Reference](../components/mocker/mocker-cli-reference.mdx). For implementation details,
see [Mocker Engine Architecture](../design-docs/mocker-architecture.md).

## Prerequisites

Before you begin:

- Install the Dynamo Kubernetes Platform.
- Set `NAMESPACE` to the namespace where you deploy Dynamo resources.
- Set `PLANNER_IMAGE` to a released or locally built `dynamo-planner` image.
- Configure cluster access with `kubectl`.

Mocker runs in the planner image and does not request GPU resources.

## Deploy an aggregated simulation

Create `mocker-agg.yaml`:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: mocker-agg
spec:
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${PLANNER_IMAGE}
  - name: decode
    type: decode
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${PLANNER_IMAGE}
          workingDir: /workspace
          command:
          - python3
          - -m
          - dynamo.mocker
          args:
          - --model-path
          - Qwen/Qwen3-0.6B
          - --speedup-ratio
          - "1.0"
```

Replace `${PLANNER_IMAGE}` with the image URI, and then apply the DGD:

```bash
kubectl apply -n "$NAMESPACE" -f mocker-agg.yaml
kubectl wait -n "$NAMESPACE" \
  --for=condition=Ready pod \
  --selector=nvidia.com/dynamo-graph-deployment-name=mocker-agg \
  --timeout=300s
```

The checked-in
[aggregated Mocker example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/mocker/deploy/v1beta1/agg.yaml)
contains the complete deployment manifest.

## Send a request

Forward the frontend service to port 8000:

```bash
kubectl port-forward -n "$NAMESPACE" svc/mocker-agg-frontend 8000:8000
```

In another terminal, send an OpenAI-compatible request:

```bash
curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32
  }'
```

A successful response confirms that the frontend discovered the Mocker worker and routed the
request through the deployment.

## Simulate disaggregated serving

Start from the checked-in
[disaggregated Mocker example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/mocker/deploy/v1beta1/disagg.yaml).
It defines separate prefill and decode components. The prefill worker uses:

```yaml
args:
- --model-path
- Qwen/Qwen3-0.6B
- --disaggregation-mode
- prefill
- --bootstrap-ports
- "50100"
```

The decode worker uses:

```yaml
args:
- --model-path
- Qwen/Qwen3-0.6B
- --disaggregation-mode
- decode
```

Apply the manifest, repeat the port-forward and request steps, and inspect the worker logs:

```bash
kubectl logs -n "$NAMESPACE" \
  --selector=nvidia.com/dynamo-graph-deployment=mocker-disagg \
  --all-containers --prefix
```

Use the logs to confirm that both worker stages registered and that requests moved from prefill to
decode.

## Try another simulation

Change one setting at a time and repeat the request:

- Increase `replicas` to exercise routing across more workers.
- Add Mocker scheduling or KV-cache flags from the
  [Mocker CLI Reference](../components/mocker/mocker-cli-reference.mdx).
- Enable the Planner to test scaling behavior with simulated workers.
- Compare the simulation with a real-GPU deployment by using
  [Benchmarking with AIPerf](../tools/aiperf.md).

DynoSim helps test control-plane behavior and shortlist configurations. Validate performance
conclusions on the target GPU hardware before production use.
