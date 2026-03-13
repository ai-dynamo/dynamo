#!/usr/bin/env bash
# migrate-s3-models-to-hf-cache.sh
#
# Restructures models in s3://dynamo-infra-utils/ci/models/ from
# flat repo-ID layout to HF cache layout.
#
# Usage: bash migrate-s3-models-to-hf-cache.sh [--dry-run]

set -euo pipefail

BUCKET="s3://dynamo-infra-utils"
PREFIX="ci/models"
REVISION="ci-local"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# All models currently in the S3 bucket
MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-32B"
  "Qwen/Qwen3-Embedding-4B"
  "Qwen/Qwen2-VL-2B-Instruct"
  "Qwen/Qwen2-VL-7B-Instruct"
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "Qwen/Qwen2-Audio-7B-Instruct"
  "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/deepseek-llm-7b-base"
  "deepseek-ai/DeepSeek-V2-Lite"
  "openai/gpt-oss-20b"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.1-405B"
  "nvidia/Llama-3.1-8B-Instruct-FP8"
  "llava-hf/llava-1.5-7b-hf"
  "llava-hf/LLaVA-NeXT-Video-7B-hf"
  "llava-hf/llava-v1.6-mistral-7b-hf"
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  "silence09/DeepSeek-R1-Small-2layers"
  "codelion/Qwen3-0.6B-accuracy-recovery-lora"
)

for MODEL in "${MODELS[@]}"; do
  # Qwen/Qwen3-0.6B -> models--Qwen--Qwen3-0.6B
  CACHE_NAME="models--${MODEL//\//--}"

  SRC="${BUCKET}/${PREFIX}/${MODEL}/"
  DST="${BUCKET}/${PREFIX}/${CACHE_NAME}/snapshots/${REVISION}/"
  REFS="${BUCKET}/${PREFIX}/${CACHE_NAME}/refs/main"

  echo "=== ${MODEL} ==="
  echo "  src: ${SRC}"
  echo "  dst: ${DST}"

  if $DRY_RUN; then
    echo "  [dry-run] would copy and create refs/main"
    echo ""
    continue
  fi

  # 1. Copy model files into snapshots/ci-local/
  aws s3 cp "${SRC}" "${DST}" --recursive

  # 2. Create refs/main containing the revision string
  echo -n "${REVISION}" | aws s3 cp - "${REFS}"

  # 3. Remove old flat layout (uncomment after verifying)
  # aws s3 rm "${SRC}" --recursive

  echo "  done"
  echo ""
done

echo "Migration complete. Verify with:"
echo "  aws s3 ls ${BUCKET}/${PREFIX}/models-- --recursive | head -40"
echo ""
echo "After verification, remove old layout by uncommenting the 'aws s3 rm' line."
