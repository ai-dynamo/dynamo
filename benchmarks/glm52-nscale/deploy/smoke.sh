#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {dynamo-vllm|vllm-serve|dynamo-sglang|sglang-serve}" >&2
  exit 2
fi

variant="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }

service="glm52-${variant}"
if [[ "${variant}" == dynamo-* ]]; then
  service="${service}-frontend"
fi

local_port="${GLM52_SMOKE_PORT:-18080}"
log="$(mktemp)"
kubectl port-forward "service/${service}" -n "${NAMESPACE}" \
  "${local_port}:8000" >"${log}" 2>&1 &
forward_pid=$!
cleanup() {
  kill "${forward_pid}" 2>/dev/null || true
  wait "${forward_pid}" 2>/dev/null || true
  rm -f "${log}"
}
trap cleanup EXIT

ready=false
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${local_port}/health" >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "${forward_pid}" 2>/dev/null; then
    cat "${log}" >&2
    exit 1
  fi
  sleep 1
done

if [[ "${ready}" != true ]]; then
  cat "${log}" >&2
  echo "Timed out waiting for ${service}" >&2
  exit 1
fi

curl_args=(-fsS --connect-timeout 10 --max-time 120)
curl "${curl_args[@]}" "http://127.0.0.1:${local_port}/v1/models" | jq -e \
  --arg model "${SERVED_MODEL_NAME}" --argjson context "${MAX_MODEL_LEN}" '
    any(.data[]; .id == $model and .context_window == $context)
  ' >/dev/null

curl "${curl_args[@]}" "http://127.0.0.1:${local_port}/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d "$(jq -n --arg model "${SERVED_MODEL_NAME}" '{
        model: $model,
        messages: [{role:"user",content:"Reply with exactly READY."}],
        temperature: 0,
        max_tokens: 512
      }')" | jq -e '.choices[0].message.content == "READY"' >/dev/null

curl "${curl_args[@]}" "http://127.0.0.1:${local_port}/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d "$(jq -n --arg model "${SERVED_MODEL_NAME}" '{
        model: $model,
        messages: [{role:"user",content:"What is the weather in Paris? Use the tool."}],
        tools: [{type:"function",function:{name:"get_weather",description:"Get weather",parameters:{type:"object",properties:{city:{type:"string"}},required:["city"]}}}],
        tool_choice: {type:"function",function:{name:"get_weather"}},
        temperature: 0,
        max_tokens: 256
      }')" | jq -e '
        .choices[0].message.tool_calls[0].function as $function
        | $function.name == "get_weather"
          and (($function.arguments | fromjson).city == "Paris")
      ' >/dev/null

curl "${curl_args[@]}" "http://127.0.0.1:${local_port}/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d "$(jq -n --arg model "${SERVED_MODEL_NAME}" '{
        model: $model,
        messages: [{role:"user",content:"Use get_weather to look up the weather in Paris. Do not answer directly."}],
        tools: [{type:"function",function:{name:"get_weather",description:"Get weather",parameters:{type:"object",properties:{city:{type:"string"}},required:["city"]}}}],
        temperature: 0,
        max_tokens: 512
      }')" | jq -e '
        .choices[0].message.tool_calls[0].function as $function
        | $function.name == "get_weather"
          and (($function.arguments | fromjson).city == "Paris")
      ' >/dev/null

echo "Smoke PASS: ${variant}"
