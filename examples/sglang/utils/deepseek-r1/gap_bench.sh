# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

MODEL=deepseek-ai/DeepSeek-R1
OSL=5
INPUT_FILE=data.jsonl
HEAD_PREFILL_NODE_IP=<ip>
PORT=8000
ARTIFACT_DIR=/benchmarks/

# concurrency 1 and 2 are for warmup
for concurrency in 1 2 8192; do
  genai-perf profile \
    --model ${MODEL} \
    --tokenizer ${MODEL} \
    --endpoint-type completions \
    --streaming \
    --url ${HEAD_PREFILL_NODE_IP}:${PORT} \
    --input-file ${INPUT_FILE} \
    --extra-inputs max_tokens:${OSL} \
    --extra-inputs min_tokens:${OSL} \
    --extra-inputs ignore_eos:true \
    --concurrency ${concurrency} \
    --request-count ${concurrency} \
    --random-seed 100 \
    --artifact-dir ${ARTIFACT_DIR} \
    -- \
    -v -v \
    --max-threads 256 \
    -H 'Authorization: Bearer NOT USED' \
    -H 'Accept: text/event-stream'
done