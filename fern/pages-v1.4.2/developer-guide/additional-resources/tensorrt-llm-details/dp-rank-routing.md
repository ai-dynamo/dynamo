---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: DP Rank Routing (Attention Data Parallelism)
---

For general TensorRT-LLM features and configuration, see the [Reference Guide](../../knowledge-base/modular-components/backends/tensorrt-llm/reference-guide.md).

---

TensorRT-LLM supports [attention data parallelism](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models) (attention DP) for models like DeepSeek. When enabled, multiple attention DP ranks run within a single worker, each with its own KV cache. Dynamo can route requests to specific DP ranks based on KV cache state.

### Dynamo vs TRT-LLM Internal Routing

- **Dynamo DP Rank Routing**: The router selects the optimal DP rank based on KV cache overlap and instructs TRT-LLM to use that rank with strict routing (`attention_dp_relax=False`). Use this with `--router-mode kv` for cache-aware routing.
- **TRT-LLM Internal Routing**: TRT-LLM's scheduler assigns DP ranks internally. Use this with `--router-mode round-robin` or `random` when KV-aware routing isn't needed.

### Conversation-Aware DP Rank Routing

To keep a conversation on the same worker and attention-DP rank, enable conversation affinity in both the [Dynamo router](../../knowledge-base/modular-components/router/configuration-and-tuning.md#session-affinity) and the [TensorRT-LLM ADP router](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md#rank-level-routing-within-a-context-server). Dynamo passes the stable conversation ID to TensorRT-LLM through `ConversationParams`.

Enable conversation-aware routing in the TensorRT-LLM engine configuration:

```yaml
enable_attention_dp: true
attention_dp_config:
  kv_cache_routing_conversation_affinity: true
```

Then pass that configuration and enable conversation affinity on the Dynamo TensorRT-LLM worker:

```bash
python3 -m dynamo.trtllm \
  --model-path <MODEL_PATH> \
  --extra-engine-args engine.yaml \
  --conversation-affinity
```

Conversation affinity requires a stable conversation ID, and TensorRT-LLM owns the conversation-to-rank binding after initial placement.

### Enabling DP Rank Routing

```bash
# Worker with attention DP
# (TP=2 acts as the "world size", in effect creating 2 attention DP ranks)
CUDA_VISIBLE_DEVICES=0,1 python3 -m dynamo.trtllm \
  --model-path <MODEL_PATH> \
  --tensor-parallel-size 2 \
  --enable-attention-dp \
  --publish-events-and-metrics

# Frontend with KV routing
python3 -m dynamo.frontend --router-mode kv
```

The `--enable-attention-dp` flag sets `attention_dp_size = tensor_parallel_size` and configures Dynamo to publish KV events per DP rank. The router automatically creates routing targets for each `(worker_id, dp_rank)` combination.

<Note>Attention DP requires TRT-LLM's PyTorch backend. AutoDeploy does not support attention DP.</Note>
