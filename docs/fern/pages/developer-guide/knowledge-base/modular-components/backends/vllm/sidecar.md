---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Note to AI agents: keep this page minimal (intro, support matrix, launch
# example, Kubernetes example). Do not edit it unless the user explicitly asks.
title: vLLM Sidecar
subtitle: Run Dynamo beside a stock vLLM engine through native gRPC.
---

> [!WARNING]
> **Experimental.** The sidecars and their deployment examples are
> experimental. Manifests, flags, and behavior may change without notice.

`dynamo-vllm-sidecar` connects a Dynamo worker to vLLM's native gRPC server
(`vllm-rs`). See [Sidecar Backends](../../../concepts/system-architecture/sidecar-backends.md)
for the architecture.

## Support Matrix

| Feature | Supported |
|---|---|
| Aggregated | Yes |
| Disaggregated | Yes |
| KV routing | Yes |

## Launch Locally

See [`lib/sidecar/vllm/launch/`](https://github.com/ai-dynamo/dynamo/tree/main/lib/sidecar/vllm/launch)
for all topologies. For example, aggregated serving on one GPU:

```bash
docker compose -f dev/docker-compose.yml up -d
./lib/sidecar/vllm/launch/agg.sh
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Deploy on Kubernetes

See [`lib/sidecar/vllm/deploy/`](https://github.com/ai-dynamo/dynamo/tree/main/lib/sidecar/vllm/deploy)
for all manifests. For example, aggregated serving:

```bash
kubectl apply -f lib/sidecar/vllm/deploy/agg.yaml -n <namespace>
kubectl port-forward -n <namespace> svc/vllm-sidecar-agg-frontend 8000:8000
```
