<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hierarchical Planner Example

This example demonstrates a hierarchical routing setup with:
- A **Global Router** that routes to different pools based on request characteristics
- **Local Routers** in each pool namespace
- **Workers** (Mocker for local testing, vLLM for Kubernetes deployment)

## Architecture

```
                    Frontend (round-robin routing)
                         |
                         v
                    Global Router
                   (registers as both prefill + decode)
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
   Prefill Pool 0   Prefill Pool 1   Decode Pool 0
   (prefill-pool-0) (prefill-pool-1) (decode-pool-0)
        |                |                |
        v                v                v
   Local Router     Local Router     Local Router
        |                |                |
        v                v                v
      Worker           Worker           Worker
   (prefill)        (prefill)        (decode)
```

## Configuration

The `global_router_config.json` defines:
- 2 prefill pools (`prefill-pool-0`, `prefill-pool-1`)
- 1 decode pool (`decode-pool-0`)
- Grid-based pool selection strategy

Pool selection is based on a 2x2 grid:
- **Prefill**: (ISL, TTFT_target) maps to prefill pool index
- **Decode**: (context_length, ITL_target) maps to decode pool index

## Running Locally (with Mocker)

For local testing without GPUs, use the mocker-based script:

```bash
cd examples/hierarchical_planner
./run_example.sh
```

This starts all components in the background and provides instructions for testing.

## Kubernetes Deployment (with vLLM)

The `vllm-2p1d.yaml` file provides a multi-DGD deployment with real vLLM workers (1 GPU each).

### Prerequisites

- Kubernetes cluster with GPU nodes
- `hf-token-secret` secret containing your HuggingFace token
- The Dynamo operator installed

### Deployment

The YAML uses environment variable placeholders:
- `${K8S_NAMESPACE}` - Your Kubernetes namespace
- `${VLLM_IMAGE}` - Dynamo vLLM runtime container image

Use `envsubst` to substitute these before applying:

```bash
# Set your Kubernetes namespace and image
export K8S_NAMESPACE=<your-k8s-namespace>
export VLLM_IMAGE=<dynamo-vllm-image>

# Deploy all DGDs
envsubst < vllm-2p1d.yaml | kubectl apply -n ${K8S_NAMESPACE} -f -
```

### Verify Deployment

```bash
# Check DGD status
kubectl get dgd -n ${K8S_NAMESPACE}

# Check pods
kubectl get pods -n ${K8S_NAMESPACE}

# Check logs for a specific component
kubectl logs -n ${K8S_NAMESPACE} -l nvidia.com/dynamo-component=Frontend
```

### Cleanup

```bash
export K8S_NAMESPACE=<your-k8s-namespace>
export VLLM_IMAGE=<dynamo-vllm-image>
envsubst < vllm-2p1d.yaml | kubectl delete -n ${K8S_NAMESPACE} -f -
```

### Namespace Convention

The Dynamo operator prepends the Kubernetes namespace to the `dynamoNamespace` field:
- K8s namespace: `my-namespace`
- `dynamoNamespace: prefill-pool-0`
- Actual Dynamo namespace: `my-namespace-prefill-pool-0`

This is why the global router config and local router endpoints must use the full namespace path.

## Testing

Once all components are running, send a request to the frontend:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50,
    "stream": true
  }'
```

For Kubernetes, port-forward the frontend service first:

```bash
kubectl port-forward -n ${K8S_NAMESPACE} svc/hierarchical-frontend-frontend 8000:8000
```

## Request Flow

1. Request arrives at **Frontend**
2. Frontend's `PrefillRouter` detects both prefill and decode registered for the model
3. Frontend sends prefill request to **Global Router** (registered as prefill)
4. Global Router selects prefill pool based on (ISL, TTFT_target) grid
5. Request forwarded to **Local Router** in selected prefill pool namespace
6. Local Router forwards to **Worker** (prefill mode)
7. Prefill response returns with `disaggregated_params`
8. Frontend sends decode request to **Global Router** (registered as decode)
9. Global Router selects decode pool based on (context_length, ITL_target) grid
10. Tokens stream back through the chain

## Customizing Pool Selection

Edit `global_router_config.json` (or the ConfigMap in `vllm-2p1d.yaml`) to change:

- **Number of pools**: Adjust `num_prefill_pools`, `num_decode_pools` and corresponding namespace lists
- **Selection grid**: Modify `isl_resolution`, `ttft_resolution` etc. to change grid granularity
- **Pool mapping**: Edit `prefill_pool_mapping` and `decode_pool_mapping` matrices

Example: To always route to pool 0 regardless of request characteristics:
```json
"prefill_pool_mapping": [[0, 0], [0, 0]]
```
