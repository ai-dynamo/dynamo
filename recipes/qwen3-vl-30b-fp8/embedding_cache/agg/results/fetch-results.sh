#!/usr/bin/env bash
# Fetch aiperf results from the benchmark pod to local results directory.
# Usage: bash fetch-results.sh
set -euo pipefail

NAMESPACE="${NAMESPACE:-qiwa}"
POD="qwen3-vl-agg-gb200-benchmark"
REMOTE_BASE="/perf-cache/artifacts/fp8/gb200-4gpu/agg"
LOCAL_BASE="$(cd "$(dirname "$0")" && pwd)/gb200-4gpu"
DATASETS=(
  "1000req_1img_200pool_400word_base64"
  "1000req_1img_200pool_400word_http"
)
CONCURRENCIES=(c4 c8 c16 c32 c64 c128)

for mode in cache_on cache_off; do
  for ds in "${DATASETS[@]}"; do
    for conc in "${CONCURRENCIES[@]}"; do
      local_dir="${LOCAL_BASE}/${ds}/${mode}/${conc}"
      remote_dir="${REMOTE_BASE}/${mode}/${ds}/${conc}"
      mkdir -p "${local_dir}"
      echo "Fetching ${mode}/${ds}/${conc}..."
      for ext in json csv; do
        kubectl -n "${NAMESPACE}" cp \
          "${POD}:${remote_dir}/profile_export_aiperf.${ext}" \
          "${local_dir}/profile_export_aiperf.${ext}" 2>/dev/null || true
      done
      if [ ! -f "${local_dir}/profile_export_aiperf.json" ]; then
        echo "  Not ready yet: ${mode}/${ds}/${conc}"
      fi
    done
  done
done

echo "Done. Results in ${LOCAL_BASE}"
