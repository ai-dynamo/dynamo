#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
GENERATOR_DIR="${DYNAMO_ROOT}/benchmarks/multimodal/jsonl"
GENERATOR_MAIN="${GENERATOR_DIR}/main.py"

NUM_REQUESTS="${NUM_REQUESTS:-5000}"
IMAGES_PER_REQUEST="${IMAGES_PER_REQUEST:-3}"
USER_TEXT_TOKENS="${USER_TEXT_TOKENS:-1000}"
IMAGE_MODE="${IMAGE_MODE:-base64}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/datasets}"

if [[ ! -f "${GENERATOR_MAIN}" ]]; then
  echo "Generator not found at ${GENERATOR_MAIN}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

total_slots=$((NUM_REQUESTS * IMAGES_PER_REQUEST))

# Approximate mapping from duplicate probability to image pool size.
# pool_size ~= total_slots * (1 - duplicate_probability)
declare -A POOLS=(
  [r00]="${total_slots}"     # duplicate probability 0.00
  [r10]="$((total_slots * 90 / 100))"
  [r25]="$((total_slots * 75 / 100))"
  [r50]="$((total_slots * 50 / 100))"
  [r75]="$((total_slots * 25 / 100))"
)

for tag in r00 r10 r25 r50 r75; do
  output_file="${OUTPUT_DIR}/qwen3_vl_${NUM_REQUESTS}req_${IMAGES_PER_REQUEST}img_${tag}.jsonl"
  pool_size="${POOLS[$tag]}"

  echo "Generating ${output_file} (images-pool=${pool_size})"
  python3 "${GENERATOR_MAIN}" \
    -n "${NUM_REQUESTS}" \
    --images-per-request "${IMAGES_PER_REQUEST}" \
    --images-pool "${pool_size}" \
    --user-text-tokens "${USER_TEXT_TOKENS}" \
    --image-mode "${IMAGE_MODE}" \
    -o "${output_file}"
done

echo
echo "Dataset generation complete."
echo "Output directory: ${OUTPUT_DIR}"
echo "To copy to perf PVC pod:"
echo "  kubectl cp ${OUTPUT_DIR} <namespace>/<benchmark-pod-name>:/perf-cache/"
