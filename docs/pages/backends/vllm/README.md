---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: vLLM
---

# LLM Deployment using vLLM

Dynamo vLLM integrates [vLLM](https://github.com/vllm-project/vllm) engines into Dynamo's distributed runtime, enabling disaggregated serving, KV-aware routing, and request cancellation while maintaining full compatibility with vLLM's native engine arguments. Dynamo leverages vLLM's native KV cache events, NIXL-based transfer mechanisms, and metric reporting to enable KV-aware routing and P/D disaggregation.

## Use the Latest Release

We recommend using the [latest stable release](https://github.com/ai-dynamo/dynamo/releases/latest) of Dynamo to avoid breaking changes.

---

<Accordion title="Build and run container">

We have public images available on [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts). If you'd like to build your own container from source:

```bash
python container/render.py --framework vllm --output-short-filename
docker build -f container/rendered.Dockerfile -t dynamo:latest-vllm .
```

```bash
./container/run.sh -it --framework VLLM [--mount-workspace]
```

This includes the specific commit [vllm-project/vllm#19790](https://github.com/vllm-project/vllm/pull/19790) which enables support for external control of the DP ranks.
</Accordion>

## Feature Support Matrix

### Core Dynamo Features

| Feature | vLLM | Notes |
|---------|------|-------|
| [**Disaggregated Serving**](../../design-docs/disagg-serving.md) | ✅ |  |
| [**Conditional Disaggregation**](../../design-docs/disagg-serving.md) | 🚧 | WIP |
| [**KV-Aware Routing**](../../components/router/README.md) | ✅ |  |
| [**SLA-Based Planner**](../../components/planner/planner-guide.md) | ✅ |  |
| [**Load Based Planner**](../../components/planner/README.md) | 🚧 | WIP |
| [**KVBM**](../../components/kvbm/README.md) | ✅ |  |
| [**LMCache**](../../integrations/lmcache-integration.md) | ✅ |  |
| [**Prompt Embeddings**](./prompt-embeddings.md) | ✅ | Requires `--enable-prompt-embeds` flag |

### Large Scale P/D and WideEP Features

| Feature            | vLLM | Notes                                                                 |
|--------------------|------|-----------------------------------------------------------------------|
| **WideEP**         | ✅   | Support for PPLX / DeepEP not verified                                           |
| **DP Rank Routing**| ✅   | Supported via external control of DP ranks |
| **GB200 Support**  | 🚧   | Container functional on main |

## Quick Start

Start infrastructure services for local development:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

Launch an aggregated serving deployment:

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/agg.sh
```

Verify the deployment:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32
  }'
```

## Next Steps

- **[Reference Guide](vllm-reference-guide.md)**: Configuration, arguments, and operational details
- **[Examples](vllm-examples.md)**: All deployment patterns with launch scripts
- **[Multi-Node](multi-node.md)**: Multi-node deployment guide
- **[DeepSeek-R1](deepseek-r1.md)**: DeepSeek-R1 deployment guide
- **[GPT-OSS](gpt-oss.md)**: GPT-OSS deployment guide
- **[Prometheus](prometheus.md)**: Metrics and monitoring
- **[Prompt Embeddings](prompt-embeddings.md)**: Pre-computed prompt embeddings
- **[vLLM-Omni](vllm-omni.md)**: Multimodal model serving
- **[Deploying vLLM with Dynamo on Kubernetes](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy/README.md)**: Kubernetes deployment guide
- **[vLLM Documentation](https://docs.vllm.ai/en/v0.9.2/configuration/serve_args.html)**: Upstream vLLM serve arguments
