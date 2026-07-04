#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
SMALL_MODEL=Qwen/Qwen3.5-122B-A10B-FP8
LARGE_MODEL=nvidia/GLM-5.2-NVFP4

request() {
  local model=$1
  local prompt=$2
  curl --fail-with-body --silent --show-error --max-time 600 \
    "$BASE_URL/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "$(jq -cn --arg model "$model" --arg prompt "$prompt" \
      '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:128,stream:false,chat_template_kwargs:{enable_thinking:false}}')"
}

assert_model() {
  local expected=$1
  local response=$2
  test "$(jq -r '.model' <<<"$response")" = "$expected"
  test "$(jq -r '[(.choices[0].message.content // ""), (.choices[0].message.reasoning_content // "")] | map(length) | add' <<<"$response")" -gt 0
}

small=$(request auto 'Say hello in one short sentence.')
assert_model "$SMALL_MODEL" "$small"

large=$(request auto 'What is 15% of 240? Answer with only the number.')
assert_model "$LARGE_MODEL" "$large"
[[ $(jq -r '[(.choices[0].message.content // ""), (.choices[0].message.reasoning_content // "")] | join(" ")' <<<"$large") == *36* ]]

auto=$(request auto 'Write a Python function that computes a matrix determinant.')
test "$(jq -r '.model' <<<"$auto")" = "$LARGE_MODEL"

spoof=$(curl --fail-with-body --silent --show-error --max-time 600 \
  "$BASE_URL/v1/chat/completions" \
  -H 'content-type: application/json' \
  -H "x-selected-model: $LARGE_MODEL" \
    -d '{"model":"auto","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":64}')
assert_model "$SMALL_MODEL" "$spoof"

unknown_code=$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 30 \
  "$BASE_URL/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"unknown/model","messages":[{"role":"user","content":"hello"}]}')
test "$unknown_code" -ge 400

curl --fail-with-body --silent --show-error --no-buffer --max-time 600 \
  "$BASE_URL/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Say hello briefly."}],"max_tokens":64,"stream":true}' \
  | awk '/^data:/{found=1} END{exit !found}'

anthropic=$(curl --fail-with-body --silent --show-error --max-time 600 \
  "$BASE_URL/v1/messages" \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -H 'x-api-key: local-dev-token' \
  -d '{"model":"auto","max_tokens":128,"messages":[{"role":"user","content":"What is 15% of 240? Answer with only the number."}]}')
test "$(jq -r '.type' <<<"$anthropic")" = message
test "$(jq -r '.model' <<<"$anthropic")" = "$LARGE_MODEL"
[[ $(jq -r '[.content[]?.text // "", .content[]?.thinking // ""] | join(" ")' <<<"$anthropic") == *36* ]]

curl --fail-with-body --silent --show-error --no-buffer --max-time 600 \
  "$BASE_URL/v1/messages" \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -H 'x-api-key: local-dev-token' \
  -d '{"model":"auto","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"Say hello briefly."}]}' \
  | awk '/^event: message_start/{start=1} /^event: message_stop/{stop=1} END{exit !(start && stop)}'

printf 'model routing smoke passed\n'
