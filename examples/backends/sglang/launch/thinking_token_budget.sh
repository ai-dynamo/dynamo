#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Send an OpenAI-compatible request with a per-request SGLang thinking budget.
#
# Start Dynamo with the SGLang chat processor, then start an SGLang worker with
# --enable-strict-thinking and a reasoning parser before running this script.

set -euo pipefail

FRONTEND_URL="${DYN_FRONTEND_URL:-http://localhost:8000}"
MODEL="${DYN_MODEL_NAME:-Qwen/Qwen3-0.6B}"
THINKING_TOKEN_BUDGET="${THINKING_TOKEN_BUDGET:-32}"

curl --fail-with-body --silent --show-error \
  "${FRONTEND_URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": \"Return a short answer for this synthetic prompt.\"
      }
    ],
    \"max_completion_tokens\": 128,
    \"thinking_token_budget\": ${THINKING_TOKEN_BUDGET}
  }"
