#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
CLIENT=${CLIENT:-$HERE/static_decode_pareto.py}
BASE_URL=${BASE_URL:-http://127.0.0.1:18000}
MODEL=${MODEL:-deepseek-ai/DeepSeek-V2-Lite}
RESULT_DIR=${RESULT_DIR:-/tmp/decode-migration-pareto}
MODES=${MODES:-migration}
RATES=${RATES:-"40 50 60 70 78 82 86"}
REQUESTS=${REQUESTS:-256}
WARMUP_SECONDS=${WARMUP_SECONDS:-5}
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-5}
MAX_TOKENS=${MAX_TOKENS:-512}
OSL_STDDEV=${OSL_STDDEV:-0}
OSL_MIN=${OSL_MIN:-256}
OSL_MAX=${OSL_MAX:-768}
SOURCE_FRACTION=${SOURCE_FRACTION:-0.6}
MIN_VISIBLE_RATE=${MIN_VISIBLE_RATE:-20}
BASELINE_GPUS=${BASELINE_GPUS:-8}
MIGRATION_GPUS=${MIGRATION_GPUS:-10}

if [[ ! -f "$CLIENT" ]]; then
    echo "Static benchmark client not found: $CLIENT" >&2
    exit 2
fi

mkdir -p "$RESULT_DIR"
request_offset=0

for mode in $MODES; do
    case "$mode" in
        baseline)
            gpu_count=$BASELINE_GPUS
            ;;
        migration)
            gpu_count=$MIGRATION_GPUS
            ;;
        *)
            echo "Unknown mode '$mode'; expected baseline or migration" >&2
            exit 2
            ;;
    esac

    for rate in $RATES; do
        output="$RESULT_DIR/${mode}-rps-${rate}.json"
        args=(
            --base-url "$BASE_URL"
            --model "$MODEL"
            --mode "$mode"
            --run-label "dep8-to-dep2-${mode}-rps-${rate}"
            --requests "$REQUESTS"
            --arrival-rate "$rate"
            --max-tokens "$MAX_TOKENS"
            --source-fraction "$SOURCE_FRACTION"
            --warmup-seconds "$WARMUP_SECONDS"
            --cooldown-seconds "$COOLDOWN_SECONDS"
            --gpu-count "$gpu_count"
            --min-visible-rate "$MIN_VISIBLE_RATE"
            --request-offset "$request_offset"
            --output "$output"
        )
        if [[ "$OSL_STDDEV" != "0" ]]; then
            args+=(
                --osl-stddev "$OSL_STDDEV"
                --osl-min "$OSL_MIN"
                --osl-max "$OSL_MAX"
            )
        fi
        echo "Running $mode at $rate requests/s -> $output"
        python3 "$CLIENT" "${args[@]}"
        request_offset=$((request_offset + REQUESTS + 100000))
    done
done
