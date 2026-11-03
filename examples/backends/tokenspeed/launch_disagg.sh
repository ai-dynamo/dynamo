#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

role="${1:?Usage: launch_disagg.sh frontend|prefill|decode [additional arguments]}"
shift
: "${ETCD_ENDPOINTS:?Set ETCD_ENDPOINTS to the shared etcd service}"
: "${NATS_SERVER:?Set NATS_SERVER to the shared NATS service}"
export DYN_NAMESPACE="${DYN_NAMESPACE:-tokenspeed}"
export DYN_EVENT_PLANE=nats

if [[ "$role" == frontend ]]; then
    exec python3 -m dynamo.frontend \
        --namespace "$DYN_NAMESPACE" \
        --discovery-backend etcd \
        --http-host 0.0.0.0 --http-port "${HTTP_PORT:-8000}" \
        --router-mode kv --router-kv-events \
        --kv-cache-block-size 64 "$@"
fi
if [[ "$role" != prefill && "$role" != decode ]]; then
    echo "Unknown role: $role" >&2
    exit 2
fi
: "${MODEL_PATH:?Set MODEL_PATH to the LongCat-Flash-Chat-FP8 checkpoint directory}"
if [[ "$role" == prefill ]]; then
    : "${DYN_TOKENSPEED_BOOTSTRAP_HOST:?Set this to an address reachable by decode workers}"
fi
extra=()
if [[ -n "${DISAGG_IB_DEVICE:-}" ]]; then
    extra+=(--disaggregation-ib-device "$DISAGG_IB_DEVICE")
fi
exec python3 -m dynamo.tokenspeed \
    --namespace "$DYN_NAMESPACE" --discovery-backend etcd \
    --request-plane tcp \
    --model "$MODEL_PATH" --served-model-name longcat-flash \
    --trust-remote-code \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-8}" --enable-expert-parallel \
    --host 0.0.0.0 --port "${ENGINE_PORT:-8100}" \
    --max-model-len 8192 --chunked-prefill-size 4096 --max-num-seqs 16 \
    --max-total-tokens "${MAX_TOTAL_TOKENS:-32768}" \
    --gpu-memory-utilization 0.85 --quantization fp8 \
    --moe-backend flashinfer_cutlass \
    --prefix-granularity 64 \
    --disaggregation-mode "$role" \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT:-8998}" \
    --kv-events-config '{"enable_kv_cache_events":true}' \
    --enable-log-request-stats \
    "${extra[@]}" "$@"
