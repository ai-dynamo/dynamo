---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Gemma3 Sliding Window
---

For general TensorRT-LLM features and configuration, see the [Reference Guide](trtllm-reference-guide.md).

---

This guide demonstrates how to deploy google/gemma-3-1b-it with Variable Sliding Window Attention (VSWA) using Dynamo. Since google/gemma-3-1b-it is a small model, each aggregated, decode, or prefill worker only requires one H100 GPU or one GB200 GPU.
VSWA is a mechanism in which a model’s layers alternate between multiple sliding window sizes. An example of this is Gemma 3, which incorporates both global attention layers and sliding window layers.

> [!Note]
> - Ensure that required services such as `nats` and `etcd` are running before starting.
> - Request access to `google/gemma-3-1b-it` on Hugging Face and set your `HF_TOKEN` environment variable for authentication.

## Aggregated Serving
```bash
cd $DYNAMO_HOME/examples/backends/trtllm
export MODEL_PATH=google/gemma-3-1b-it
export SERVED_MODEL_NAME=$MODEL_PATH
export AGG_ENGINE_ARGS=$DYNAMO_HOME/examples/backends/trtllm/engine_configs/gemma3/vswa_agg.yaml
./launch/agg.sh
```

## Aggregated Serving with KV Routing
```bash
cd $DYNAMO_HOME/examples/backends/trtllm
export MODEL_PATH=google/gemma-3-1b-it
export SERVED_MODEL_NAME=$MODEL_PATH
export AGG_ENGINE_ARGS=$DYNAMO_HOME/examples/backends/trtllm/engine_configs/gemma3/vswa_agg.yaml
./launch/agg_router.sh
```

## Disaggregated Serving
```bash
cd $DYNAMO_HOME/examples/backends/trtllm
export MODEL_PATH=google/gemma-3-1b-it
export SERVED_MODEL_NAME=$MODEL_PATH
export PREFILL_ENGINE_ARGS=$DYNAMO_HOME/examples/backends/trtllm/engine_configs/gemma3/vswa_prefill.yaml
export DECODE_ENGINE_ARGS=$DYNAMO_HOME/examples/backends/trtllm/engine_configs/gemma3/vswa_decode.yaml
./launch/disagg.sh
```

## Disaggregated Serving with KV Routing
```bash
cd $DYNAMO_HOME/examples/backends/trtllm
export MODEL_PATH=google/gemma-3-1b-it
export SERVED_MODEL_NAME=$MODEL_PATH
export PREFILL_ENGINE_ARGS=$DYNAMO_HOME/examples/backends/trtllm/engine_configs/gemma3/vswa_prefill.yaml
export DECODE_ENGINE_ARGS=$DYNAMO_HOME/examples/backends/trtllm/engine_configs/gemma3/vswa_decode.yaml
./launch/disagg_router.sh
```
