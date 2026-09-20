#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start `dynamo.frontend --dyn-chat-processor sglang --reasoning-parser qwen3`
# and an SGLang worker with `--enable-strict-thinking --reasoning-parser qwen3`
# and an enabled grammar backend before running this script.

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
