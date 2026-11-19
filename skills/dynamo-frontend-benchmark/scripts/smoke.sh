#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single streaming request to confirm the frontend -> KV router -> mock worker
# pipeline is alive before running the full aiperf profile.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

echo "[smoke] GET /v1/models:"
curl -sf "http://localhost:${HTTP_PORT}/v1/models" | head -c 400; echo; echo

echo "[smoke] streaming chat completion:"
set +o pipefail
curl -sf -X POST "http://localhost:${HTTP_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Accept: text/event-stream" \
    -d "{
        \"model\": \"${MODEL}\",
        \"messages\": [
            {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
            {\"role\": \"user\", \"content\": \"Say hello in five words.\"}
        ],
        \"stream\": true,
        \"max_tokens\": 16,
        \"ignore_eos\": true
    }" | head -20
stream_status=("${PIPESTATUS[@]}")
set -o pipefail
if [[ "${stream_status[0]}" -ne 0 && "${stream_status[0]}" -ne 23 ]]; then
    exit "${stream_status[0]}"
fi
if [[ "${stream_status[1]}" -ne 0 ]]; then
    exit "${stream_status[1]}"
fi
echo
echo "[smoke] done. If you saw SSE 'data:' chunks above, the pipeline works."
