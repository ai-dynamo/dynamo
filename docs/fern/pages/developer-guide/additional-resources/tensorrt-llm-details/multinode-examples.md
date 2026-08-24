---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multinode Deployments
---

For general TensorRT-LLM features and engine configuration, see the [Reference Guide](../../knowledge-base/modular-components/backends/tensorrt-llm/reference-guide.md).

## Deployment starting points

The maintained [TensorRT-LLM examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/deployments/kubernetes/llm/trtllm) show both aggregated and disaggregated serving with Qwen3-32B-FP8. They are self-contained and include the model cache, download Job, DynamoGraphDeployment, and AIPerf workload.

For a multinode deployment, start from the disaggregated example and adapt the parallelism, replica count, placement constraints, and interconnect settings to your cluster. Validate the NIXL/UCX path before scaling out; no model-specific multinode TensorRT-LLM manifest is maintained in this repository.

## Common workflow

1. Install the Dynamo platform on Kubernetes. See the [Kubernetes Deployment Guide](../../../kubernetes/getting-started/quickstart.mdx).
2. Select an example and update its `model-cache.yaml` storage class.
3. Create a namespace and a `hf-token-secret` containing your Hugging Face token.
4. Apply `model-cache.yaml`, then `model-download.yaml`, and wait for the download Job to complete.
5. Apply `deploy.yaml`, then port-forward the frontend and send a request to `/v1/models` or `/v1/chat/completions`.

## Notes

- The TensorRT-LLM engine configuration files used by launch and deploy flows live under [`examples/backends/trtllm/engine_configs/`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/trtllm/engine_configs/README.md).
- Tune model parallelism, replica counts, routing, and network settings in your copied example rather than relying on a model-specific recipe.
