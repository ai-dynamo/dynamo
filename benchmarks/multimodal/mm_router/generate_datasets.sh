#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generate HTTP and DataURI benchmark datasets for MM router concurrency sweep.
#
# Request count = concurrency * 100 (100 rounds per slot for stable statistics).
#
# Pool sizes scale proportionally with request count to maintain consistent reuse rates:
#   ~90% reuse : pool = n / 10        (birthday problem: ~90% collision rate)
#   ~50% reuse : pool = n * 0.6       (birthday problem: ~50% collision rate)
#     0% reuse : pool = n, --no-reuse  (each image used exactly once)
#
# Usage:
#   bash generate_datasets.sh [output_dir]            # all concurrency levels
#   bash generate_datasets.sh [output_dir] 5          # single level
#   bash generate_datasets.sh [output_dir] 1,2,4,8    # comma-separated subset
#   CONC_LEVELS="1 5 10" bash generate_datasets.sh    # space-separated env override
#
# Output dir defaults to: ./datasets

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSONL_GEN="${JSONL_GEN:-${SCRIPT_DIR}/../jsonl}"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/datasets}"
IMAGE_DIR="${IMAGE_DIR:-/tmp/bench_images}"
PROMPT="${PROMPT:-Please describe this image.}"
IMAGE_SEED="${IMAGE_SEED:-42}"   # fixed seed ensures disjoint slices are reproducible

# Concurrency levels: positional arg $2 (comma-separated), env CONC_LEVELS, or default sweep
if [[ -n "${2:-}" ]]; then
    IFS=',' read -ra CONC_LEVELS <<< "${2}"
elif [[ -n "${CONC_LEVELS:-}" ]]; then
    read -ra CONC_LEVELS <<< "${CONC_LEVELS}"
else
    CONC_LEVELS=(1 4 8 16 32 64)
fi

mkdir -p "${OUTPUT_DIR}"

echo "=== MM Router Dataset Generator ==="
echo "JSONL generator : ${JSONL_GEN}"
echo "Output directory: ${OUTPUT_DIR}"
echo

for conc in "${CONC_LEVELS[@]}"; do
    n=$(( conc * 100 ))

    # pool sizes scaled to maintain consistent reuse rates across concurrency levels
    pool_90=$(( n / 10 ))
    pool_50=$(( n * 6 / 10 ))  # 0.6 * n → ~50% reuse

    # disjoint image offsets per case so no image is shared across cases
    #   90% case : offset=0,                    uses images [0,        pool_90)
    #   50% case : offset=pool_90,               uses images [pool_90,  pool_90+pool_50)
    #    0% case : offset=pool_90+pool_50,        uses images [pool_90+pool_50, pool_90+pool_50+n)
    offset_90=0
    offset_50=$(( pool_90 ))
    offset_0=$(( pool_90 + pool_50 ))

    echo "--- conc=${conc}  req=${n}  pool_90%=${pool_90}(offset=${offset_90})  pool_50%=${pool_50}(offset=${offset_50})  pool_0%=${n}(offset=${offset_0}) ---"

    for mode in http base64; do
        # label used in filename
        label="${mode}"
        [[ "${mode}" == "base64" ]] && label="datauri"

        # ~90% reuse
        out="${OUTPUT_DIR}/${n}req_1img_${pool_90}pool_${label}.jsonl"
        echo "  ${out}"
        python3 "${JSONL_GEN}/main.py" \
            -n "${n}" \
            --images-per-request 1 \
            --images-pool "${pool_90}" \
            --image-offset "${offset_90}" \
            --seed "${IMAGE_SEED}" \
            --image-mode "${mode}" \
            --image-dir "${IMAGE_DIR}" \
            --prompt "${PROMPT}" \
            -o "${out}"

        # ~50% reuse
        out="${OUTPUT_DIR}/${n}req_1img_${pool_50}pool_${label}.jsonl"
        echo "  ${out}"
        python3 "${JSONL_GEN}/main.py" \
            -n "${n}" \
            --images-per-request 1 \
            --images-pool "${pool_50}" \
            --image-offset "${offset_50}" \
            --seed "${IMAGE_SEED}" \
            --image-mode "${mode}" \
            --image-dir "${IMAGE_DIR}" \
            --prompt "${PROMPT}" \
            -o "${out}"

        # 0% reuse (no replacement)
        out="${OUTPUT_DIR}/${n}req_1img_${n}pool_${label}.jsonl"
        echo "  ${out}"
        python3 "${JSONL_GEN}/main.py" \
            -n "${n}" \
            --images-per-request 1 \
            --images-pool "${n}" \
            --no-reuse \
            --image-offset "${offset_0}" \
            --seed "${IMAGE_SEED}" \
            --image-mode "${mode}" \
            --image-dir "${IMAGE_DIR}" \
            --prompt "${PROMPT}" \
            -o "${out}"
    done
    echo
done

# warmup dataset: 50 requests, small pool (http only — enough to prime KV cache)
echo "--- warmup ---"
out="${OUTPUT_DIR}/warmup_50req_1img_5pool_http.jsonl"
echo "  ${out}"
python3 "${JSONL_GEN}/main.py" \
    -n 50 \
    --images-per-request 1 \
    --images-pool 5 \
    --image-mode http \
    --prompt "${PROMPT}" \
    -o "${out}"
echo

echo "=== Done. Files written to ${OUTPUT_DIR} ==="
echo
echo "Expected pool sizes per concurrency level:"
printf "  %-6s %-8s  %-14s %-14s %s\n" "conc" "req" "~90% pool" "~50% pool" "0% pool"
printf "  %-6s %-8s  %-14s %-14s %s\n" "------" "--------" "--------------" "--------------" "-------"
for conc in "${CONC_LEVELS[@]}"; do
    n=$(( conc * 100 ))
    pool_90=$(( n / 10 ))
    pool_50=$(( n * 6 / 10 ))
    printf "  %-6s %-8s  %-14s %-14s %s\n" "${conc}" "${n}" "${pool_90}" "${pool_50}" "${n}"
done
