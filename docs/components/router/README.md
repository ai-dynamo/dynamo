<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Router

The Dynamo KV Router intelligently routes requests by evaluating their computational costs across different workers. It considers both decoding costs (from active blocks) and prefill costs (from newly computed blocks), using KV cache overlap to minimize redundant computation. Optimizing the KV Router is critical for achieving maximum throughput and minimum latency in distributed inference setups.

## Quick Start

### Python / CLI Deployment

To launch the Dynamo frontend with the KV Router:

```bash
python -m dynamo.frontend --router-mode kv --http-port 8000
```

This command:
- Launches the Dynamo frontend service with KV routing enabled
- Exposes the service on port 8000 (configurable)
- Automatically handles all backend workers registered to the Dynamo endpoint

Backend workers register themselves using the `register_llm` API, after which the KV Router automatically tracks worker state and makes routing decisions based on KV cache overlap.

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--router-mode kv` | `round_robin` | Enable KV cache-aware routing |
| `--router-temperature <float>` | `0.0` | Controls routing randomness (0.0 = deterministic, higher = more random) |
| `--kv-cache-block-size <size>` | Backend-specific | KV cache block size (should match backend config) |
| `--kv-events` / `--no-kv-events` | `--kv-events` | Enable/disable real-time KV event tracking |
| `--kv-overlap-score-weight <float>` | `1.0` | Balance prefill vs decode optimization (higher = better TTFT) |

For all available options: `python -m dynamo.frontend --help`

### Kubernetes Deployment

To enable the KV Router in Kubernetes, add the `DYN_ROUTER_MODE` environment variable to your frontend service:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-deployment
spec:
  services:
    Frontend:
      dynamoNamespace: my-namespace
      componentType: frontend
      replicas: 1
      envs:
        - name: DYN_ROUTER_MODE
          value: kv  # Enable KV Smart Router
```

**Key Points:**
- Set `DYN_ROUTER_MODE=kv` on the **Frontend** service only
- Workers automatically report KV cache events to the router
- No worker-side configuration changes needed

#### Environment Variables

All CLI arguments can be configured via environment variables using the `DYN_` prefix:

| CLI Argument | Environment Variable | Default |
|--------------|---------------------|---------|
| `--router-mode kv` | `DYN_ROUTER_MODE=kv` | `round_robin` |
| `--router-temperature` | `DYN_ROUTER_TEMPERATURE` | `0.0` |
| `--kv-cache-block-size` | `DYN_KV_CACHE_BLOCK_SIZE` | Backend-specific |
| `--no-kv-events` | `DYN_KV_EVENTS=false` | `true` |
| `--kv-overlap-score-weight` | `DYN_KV_OVERLAP_SCORE_WEIGHT` | `1.0` |

For complete K8s examples and advanced configuration, see [K8s Examples](router_examples.md#k8s-examples).

For A/B testing and advanced K8s setup, see the [KV Router A/B Benchmarking Guide](../../benchmarks/kv-router-ab-testing.md).

For more configuration options and tuning guidelines, see the [Router Guide](router_guide.md).

## Prerequisites and Limitations

**Requirements:**
- **Dynamic endpoints only**: KV router requires `register_llm()` with `model_input=ModelInput.Tokens`. Your backend handler receives pre-tokenized requests with `token_ids` instead of raw text.
- Backend workers must call `register_llm()` with `model_input=ModelInput.Tokens` (see [Backend Guide](../../development/backend-guide.md))
- You cannot use `--static-endpoint` mode with KV routing (use dynamic discovery instead)

**Multimodal Support:**
- **vLLM and TRT-LLM**: Multimodal routing supported for images via multimodal hashes
- **SGLang**: Image routing not yet supported
- **Other modalities** (audio, video, etc.): Not yet supported

**Limitations:**
- Static endpoints not supported—KV router requires dynamic model discovery via etcd to track worker instances and their KV cache states

For basic model registration without KV routing, use `--router-mode round-robin` or `--router-mode random` with both static and dynamic endpoints.

## Next Steps

- **[Router Guide](router_guide.md)**: Deep dive into KV cache routing, configuration, disaggregated serving, and tuning
- **[Router Examples](router_examples.md)**: Python API usage, K8s examples, and custom routing patterns
- **[Router Design](../../design_docs/router_design.md)**: Architecture details, algorithms, and event transport modes
