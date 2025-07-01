#!/bin/bash

# run ingress
dynamo run in=http out=dyn &
# run prefill worker
python3 components/worker_inc.py \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --page-size 16 \
  --tp 2 \
  --dp-size 2 \
  --enable-dp-attention \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl & 
# run decode worker
CUDA_VISIBLE_DEVICES=2,3 python3 components/decode_worker_inc.py \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --page-size 16 \
  --tp 2 \
  --dp-size 2 \
  --enable-dp-attention \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl