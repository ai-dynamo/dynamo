#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Send chat completions through the frontend -> global router -> a selected agg pool.
# The optional `nvext.router.ttft_target` (ms) drives pool selection per agg_config.json:
# a tight TTFT target lands in pool 0; a relaxed one lands in pool 1.
set -u
PORT="${1:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
URL="http://localhost:${PORT}/v1/chat/completions"

send() { # $1 = label, $2 = ttft_target (ms, empty for default)
  local label="$1" ttft="$2" nvext=""
  [ -n "$ttft" ] && nvext=",\"nvext\":{\"router\":{\"ttft_target\":${ttft}}}"
  echo "== ${label} =="
  curl -s "$URL" -H 'Content-Type: application/json' -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\":\"user\",\"content\":\"One-sentence fun fact about the moon.\"}],
    \"max_tokens\": 32, \"stream\": false ${nvext}
  }" | python3 -m json.tool
  echo
}

echo "== models =="; curl -s "http://localhost:${PORT}/v1/models" | python3 -m json.tool; echo
send "default routing"            ""
send "tight TTFT target -> pool 0" 100
send "relaxed TTFT target -> pool 1" 900

echo "Tip: which pool served a request is visible in the global router log (logs/global_router.log)."
