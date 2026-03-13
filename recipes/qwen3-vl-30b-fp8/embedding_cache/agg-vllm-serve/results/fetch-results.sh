#!/usr/bin/env bash
# Fetch aiperf results from the benchmark pod to local results directory.
# Usage: bash fetch-results.sh
set -euo pipefail

NAMESPACE="${NAMESPACE:-qiwa}"
POD="agg-vllm-serve-1gpu-benchmark"
REMOTE_BASE="/perf-cache-tmp/artifacts/fp8/gb200-1gpu/agg-vllm-serve"
LOCAL_BASE="${HOME}/workspace/dynamo-aiperf/aws-1 osl/agg-vllm-serve/gb200-1gpu"
DATASETS=(
  "1000req_1img_200pool_400word_base64"
  "1000req_1img_200pool_400word_http"
  "1000req_1img_500pool_400word_base64"
  "1000req_1img_500pool_400word_http"
  "1000req_1img_800pool_400word_base64"
  "1000req_1img_800pool_400word_http"
  "1000req_2img_400pool_400word_base64"
  "1000req_2img_400pool_400word_http"
  "1000req_2img_1000pool_400word_base64"
  "1000req_2img_1000pool_400word_http"
  "1000req_2img_1600pool_400word_base64"
  "1000req_2img_1600pool_400word_http"
)
RATES=(r4 r8 r16 r32 r64)

for mode in cache_on cache_off; do
  for ds in "${DATASETS[@]}"; do
    for rate in "${RATES[@]}"; do
      local_dir="${LOCAL_BASE}/${ds}/${mode}/${rate}"
      remote_dir="${REMOTE_BASE}/${mode}/${ds}/${rate}"
      mkdir -p "${local_dir}"
      echo "Fetching ${mode}/${ds}/${rate}..."
      for ext in json csv; do
        kubectl -n "${NAMESPACE}" cp \
          "${POD}:${remote_dir}/profile_export_aiperf.${ext}" \
          "${local_dir}/profile_export_aiperf.${ext}" 2>/dev/null || true
      done
      if [ ! -f "${local_dir}/profile_export_aiperf.json" ]; then
        echo "  Not ready yet: ${mode}/${ds}/${rate}"
      fi
    done
  done
done

echo "Done. Results in ${LOCAL_BASE}"
