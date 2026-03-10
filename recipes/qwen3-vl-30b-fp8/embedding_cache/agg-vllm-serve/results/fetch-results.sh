#!/usr/bin/env bash
# Fetch aiperf results from the benchmark pod to local results directory.
# Usage: bash fetch-results.sh
set -euo pipefail

NAMESPACE="${NAMESPACE:-qiwa}"
POD="qwen3-vl-vllm-serve-benchmark"
REMOTE_BASE="/perf-cache-tmp/artifacts/fp8/gb200/agg-vllm-serve"
LOCAL_BASE="$(cd "$(dirname "$0")" && pwd)/gb200"
DATASET="1000req_1img_200pool_400word_base64"

for mode in cache_on cache_off; do
  for conc in c32 c64 c128; do
    local_dir="${LOCAL_BASE}/${DATASET}/${mode}/${conc}"
    remote_dir="${REMOTE_BASE}/${mode}/${DATASET}/${conc}"
    mkdir -p "${local_dir}"
    echo "Fetching ${mode}/${conc}..."
    for ext in json csv; do
      kubectl -n "${NAMESPACE}" cp \
        "${POD}:${remote_dir}/profile_export_aiperf.${ext}" \
        "${local_dir}/profile_export_aiperf.${ext}" 2>/dev/null || true
    done
    if [ ! -f "${local_dir}/profile_export_aiperf.json" ]; then
      echo "  Not ready yet: ${mode}/${conc}"
    fi
  done
done

echo "Done. Results in ${LOCAL_BASE}"
