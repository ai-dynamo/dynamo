---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Running gpt-oss-120b Disaggregated with SGLang

The gpt-oss-120b guide for SGLang is largely identical to the [guide for vLLM](../vllm/gpt-oss.md),
please ues the vLLM guide as a reference with the different deployment steps as highlighted below:

# Launch the Deployment

Note that GPT-OSS is a reasoning model with tool calling support. To
ensure the response is being processed correctly, the worker should be
launched with proper `--dyn-reasoning-parser` and `--dyn-tool-call-parser`.

**Start frontend**
```bash
python3 -m dynamo.frontend --http-port 8000 &
```

**Run decode worker**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m dynamo.sglang \
  --model-path openai/gpt-oss-120b \
  --served-model-name openai/gpt-oss-120b \
  --tp 4 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --dyn-reasoning-parser gpt_oss \
  --dyn-tool-call-parser harmony
```

**Run prefill workers**
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m dynamo.sglang \
  --model-path openai/gpt-oss-120b \
  --served-model-name openai/gpt-oss-120b \
  --tp 4 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --dyn-reasoning-parser gpt_oss \
  --dyn-tool-call-parser harmony
```
