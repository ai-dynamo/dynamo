#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
    echo "Bash 4.4 or later is required" >&2
    exit 1
fi

for tool in curl jq; do
    command -v "$tool" >/dev/null || { echo "$tool is required" >&2; exit 1; }
done

BASE_URL="${BASE_URL:-http://localhost:${DYN_HTTP_PORT:-8000}}"
BASE_URL="${BASE_URL%/}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
READY_TIMEOUT="${READY_TIMEOUT:-300}"
if [[ ! "$READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "READY_TIMEOUT must be a positive integer number of seconds" >&2
    exit 2
fi
AUTH_ARGS=()
if [[ -n "${API_KEY:-}" ]]; then AUTH_ARGS=(-H "Authorization: Bearer $API_KEY"); fi
SMOKE_DIR=$(mktemp -d)
trap 'rm -r -- "$SMOKE_DIR"' EXIT

deadline=$((SECONDS + READY_TIMEOUT))
until curl -fsS --max-time 2 "${AUTH_ARGS[@]}" "$BASE_URL/health" >/dev/null 2>&1 \
    && curl -fsS --max-time 2 "${AUTH_ARGS[@]}" "$BASE_URL/v1/models" > "$SMOKE_DIR/models.json" 2>/dev/null \
    && jq -e --arg model "$MODEL" '.data | any(.id == $model)' "$SMOKE_DIR/models.json" >/dev/null; do
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for healthy frontend and registered model $MODEL at $BASE_URL" >&2
        exit 1
    fi
    sleep 1
done
echo "Health and model discovery passed"

post() {
    local endpoint="$1" expected="$2"
    local status
    status=$(curl -sS --max-time 120 "${AUTH_ARGS[@]}" \
        -H 'Content-Type: application/json' \
        -D "$SMOKE_DIR/headers" -o "$SMOKE_DIR/response.json" -w '%{http_code}' \
        --data-binary "@$SMOKE_DIR/request.json" "$BASE_URL$endpoint")
    if [[ "$status" != "$expected" ]]; then
        echo "$endpoint: expected HTTP $expected, received $status" >&2
        cat "$SMOKE_DIR/response.json" >&2
        exit 1
    fi
}

jq -n --arg model "$MODEL" '{
    model: $model,
    state: {ticket: "A customer was charged twice and requests a refund today."},
    chat_template_kwargs: {enable_thinking: false},
    questions: {
        route: {type: "choice", instructions: "Choose the team responsible for this ticket.", criteria: {billing: "Payments and refunds", technical: "Software and integration problems"}},
        urgent: {type: "noul", instructions: "The customer asks for action today."},
        severity: {type: "score", instructions: "Rate the impact of this ticket.", criteria: ["Low impact", "Moderate impact", "High impact"]}
    }
}' > "$SMOKE_DIR/request.json"
post /v1/systemone 200
if ! tr -d '\r' < "$SMOKE_DIR/headers" | grep -qi '^x-dynamo-systemone-version: 1$'; then
    echo "Missing System One version response header" >&2
    exit 1
fi
if ! tr -d '\r' < "$SMOKE_DIR/headers" | grep -Eqi '^x-request-id: .+'; then
    echo "Missing request ID response header" >&2
    exit 1
fi
jq -e --arg model "$MODEL" '
    def unit: type == "number" and . >= 0 and . <= 1;
    def distribution: all(.[]; unit) and (([.[]] | add) - 1 | fabs) < 0.000001;
    .model == $model and (.answers | keys_unsorted) == ["route", "urgent", "severity"]
    and .answers.route.type == "choice"
    and (.answers.route.choice == "billing" or .answers.route.choice == "technical")
    and (.answers.route.probabilities | keys) == ["billing", "technical"]
    and (.answers.route.probabilities | distribution)
    and (.answers.route.confidence | unit)
    and (.answers.route.x_label_mass | unit)
    and .answers.urgent.type == "noul" and (.answers.urgent.noul | unit)
    and (.answers.urgent.x_label_mass | unit)
    and .answers.severity.type == "score"
    and (.answers.severity.score | type == "number" and . >= 0 and . <= 2)
    and (.answers.severity.probabilities | keys) == ["0", "1", "2"]
    and (.answers.severity.probabilities | distribution)
    and (.answers.severity.confidence | unit)
    and (.answers.severity.x_label_mass | unit)
    and (.answers.severity.legend | keys) == ["0", "1", "2"]
    and .usage.input_tokens > 0 and .usage.output_tokens == 0
' "$SMOKE_DIR/response.json" >/dev/null
echo "Mixed choice/noul/score request passed"
jq . "$SMOKE_DIR/response.json"

jq -n --arg model "$MODEL" '{model: $model, messages: [{role: "user", content: "Reply with one short greeting."}], max_tokens: 16, stream: false, chat_template_kwargs: {enable_thinking: false}}' > "$SMOKE_DIR/request.json"
post /v1/chat/completions 200
jq -e '.choices | length > 0' "$SMOKE_DIR/response.json" >/dev/null
echo "Chat completions regression check passed"

jq -n --arg model "$MODEL" '{model: $model, state: "A support ticket", questions: {route: {type: "choice", criteria: {}}}}' > "$SMOKE_DIR/request.json"
post /v1/systemone 422
jq -e '.error.message | contains("criteria")' "$SMOKE_DIR/response.json" >/dev/null
echo "Semantic validation error check passed"
echo "System One smoke test passed"
