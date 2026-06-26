#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Send a completion to the FRONTEND (not the engine system port) and assert the
# shadow has taken over. Retries because, right after a failover, the shadow's
# KV cache may briefly hit allocation backpressure until the dead primary's GPU
# memory is reclaimed.
#
# Success: HTTP 200 with a non-empty "choices" array. Prints "TAKEOVER OK".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

PROMPT="${PROMPT:-The capital of France is}"
MAX_TOKENS="${MAX_TOKENS:-20}"
RETRIES="${RETRIES:-30}"
RETRY_INTERVAL="${RETRY_INTERVAL:-2}"

url="http://localhost:${FRONTEND_PORT}/v1/completions"
body="{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${MAX_TOKENS}}"

echo "    sending completion to ${url} (model ${MODEL}), up to ${RETRIES} retries ..."

for ((attempt = 1; attempt <= RETRIES; attempt++)); do
  # -s silent, -w status code on its own trailing line, capture body too.
  response="$(curl -s -w $'\n%{http_code}' -X POST "${url}" \
    -H 'Content-Type: application/json' \
    -d "${body}" 2>/dev/null || true)"

  http_code="$(printf '%s' "${response}" | tail -n1)"
  payload="$(printf '%s' "${response}" | sed '$d')"

  if [[ "${http_code}" == "200" ]]; then
    # Confirm a non-empty choices array via a strict JSON parse.
    if printf '%s' "${payload}" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)
choices = data.get("choices") or []
sys.exit(0 if choices else 1)
' 2>/dev/null; then
      echo "    attempt ${attempt}: 200 with choices"
      echo "TAKEOVER OK"
      exit 0
    fi
  fi

  echo "    attempt ${attempt}/${RETRIES}: not ready yet (http=${http_code:-none}); retrying in ${RETRY_INTERVAL}s ..."
  sleep "${RETRY_INTERVAL}"
done

echo "TAKEOVER FAILED: frontend did not return a valid completion after ${RETRIES} attempts" >&2
exit 1
