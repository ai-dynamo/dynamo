#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end verification: client → SMG → Dynamo Frontend → SGLang.
#
# Usage:
#   ./tests/smg-roundtrip.sh                     # default ports + namespaces
#   SMG_NS=smg DYNAMO_NS=dynamo-system ./tests/smg-roundtrip.sh
#   MODEL=BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-GatedNorm-G1 ./tests/smg-roundtrip.sh
#
# Exit codes:
#   0  full chain healthy
#   2  SMG /health failed (gateway down)
#   3  Dynamo Frontend /health failed (router down — SMG would also fail, but
#      we differentiate so the failure points at the right layer)
#   4  SMG returned non-200 on a /v1/chat/completions probe
#   5  response did not parse as a chat.completion (engine returned an error)

set -euo pipefail

SMG_NS=${SMG_NS:-smg}
SMG_SVC=${SMG_SVC:-smg-router}
SMG_PORT=${SMG_PORT:-80}
DYNAMO_NS=${DYNAMO_NS:-dynamo-system}
DYNAMO_SVC=${DYNAMO_SVC:-deepseek-v32-reap-sglang-frontend}
DYNAMO_PORT=${DYNAMO_PORT:-8000}
MODEL=${MODEL:-BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-GatedNorm-G1}
LOCAL_SMG_PORT=${LOCAL_SMG_PORT:-18080}
LOCAL_DYNAMO_PORT=${LOCAL_DYNAMO_PORT:-18000}

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
warn() { echo -e "${YELLOW}[..]${NC} $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*"; }

cleanup_pids=()
cleanup() { for pid in "${cleanup_pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT

start_pf() {
  local ns=$1 svc=$2 local_port=$3 remote_port=$4
  kubectl -n "$ns" port-forward "svc/$svc" "$local_port:$remote_port" >/dev/null 2>&1 &
  cleanup_pids+=($!)
  # Wait until the local socket is up. Bound the wait to 15s.
  for _ in $(seq 1 30); do
    if (echo > "/dev/tcp/127.0.0.1/$local_port") 2>/dev/null; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

warn "port-forward $SMG_NS/$SMG_SVC:$SMG_PORT -> :$LOCAL_SMG_PORT"
start_pf "$SMG_NS" "$SMG_SVC" "$LOCAL_SMG_PORT" "$SMG_PORT" \
  || { err "could not port-forward SMG service"; exit 2; }

warn "port-forward $DYNAMO_NS/$DYNAMO_SVC:$DYNAMO_PORT -> :$LOCAL_DYNAMO_PORT"
start_pf "$DYNAMO_NS" "$DYNAMO_SVC" "$LOCAL_DYNAMO_PORT" "$DYNAMO_PORT" \
  || { err "could not port-forward Dynamo Frontend service"; exit 3; }

warn "SMG /health"
if ! curl -sS -fL --max-time 5 "http://127.0.0.1:$LOCAL_SMG_PORT/health" >/dev/null; then
  err "SMG /health failed — gateway down"; exit 2
fi
ok "SMG healthy"

warn "Dynamo Frontend /health"
if ! curl -sS -fL --max-time 5 "http://127.0.0.1:$LOCAL_DYNAMO_PORT/health" >/dev/null; then
  err "Dynamo Frontend /health failed — router down"; exit 3
fi
ok "Dynamo Frontend healthy"

warn "round-trip: client -> SMG -> Dynamo -> SGLang"
resp=$(curl -sS --max-time 60 \
  -H 'Content-Type: application/json' \
  -X POST "http://127.0.0.1:$LOCAL_SMG_PORT/v1/chat/completions" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: pong\"}],
    \"max_tokens\": 8,
    \"temperature\": 0
  }" -w "\nHTTP_CODE=%{http_code}")
http_code=$(printf "%s\n" "$resp" | sed -n 's/^HTTP_CODE=\([0-9]*\)$/\1/p')
body=$(printf "%s\n" "$resp" | sed '/^HTTP_CODE=/d')

if [[ "$http_code" != "200" ]]; then
  err "SMG returned HTTP $http_code"; printf "%s\n" "$body" | head -10
  exit 4
fi

# Parse with python (no jq dependency).
if ! python3 -c "
import json, sys
r = json.loads(sys.stdin.read())
assert r.get('object') == 'chat.completion', f\"object={r.get('object')}\"
choices = r.get('choices') or []
assert choices and choices[0].get('message', {}).get('content'), 'missing content'
print(choices[0]['message']['content'][:80])
" <<< "$body"; then
  err "response did not parse as chat.completion"; printf "%s\n" "$body" | head -20
  exit 5
fi

ok "full chain (SMG -> Dynamo -> SGLang) round-trip succeeded"
