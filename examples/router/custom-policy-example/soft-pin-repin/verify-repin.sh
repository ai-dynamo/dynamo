#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${DYNAMO_FRONTEND_URL:=http://localhost:8000}"
: "${DYNAMO_MODEL:=Qwen/Qwen3-0.6B}"

session_id="soft-pin-repin-example"
temp_directory="$(mktemp -d)"
first_response="${temp_directory}/first-response.sse"
first_pid=""

cleanup() {
  if [[ -n "${first_pid}" ]] && kill -0 "${first_pid}" 2>/dev/null; then
    kill "${first_pid}" 2>/dev/null || true
    wait "${first_pid}" 2>/dev/null || true
  fi
  rm -r "${temp_directory}"
}
trap cleanup EXIT

request_worker() {
  local turn="$1"

  curl --fail --silent --show-error \
    "${DYNAMO_FRONTEND_URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -H "X-Dynamo-Session-ID: ${session_id}" \
    -d "{
      \"model\": \"${DYNAMO_MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"soft pin turn ${turn}\"}],
      \"max_tokens\": 4,
      \"stream\": false,
      \"nvext\": {\"extra_fields\": [\"worker_id\"]}
    }" | jq --exit-status --raw-output '.nvext.worker_id.decode_worker_id'
}

stream_worker() {
  sed -n '/^data: {/s/^data: //p' "${first_response}" |
    jq --slurp --exit-status --raw-output \
      'map(.nvext.worker_id.decode_worker_id // empty) | first // empty' 2>/dev/null
}

curl --fail --silent --show-error --no-buffer \
  "${DYNAMO_FRONTEND_URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H "X-Dynamo-Session-ID: ${session_id}" \
  -d "{
    \"model\": \"${DYNAMO_MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"hold the initial soft pin\"}],
    \"max_tokens\": 128,
    \"stream\": true,
    \"nvext\": {\"extra_fields\": [\"worker_id\"]}
  }" >"${first_response}" &
first_pid="$!"

first_worker=""
for _ in $(seq 1 100); do
  if first_worker="$(stream_worker)" && [[ -n "${first_worker}" ]]; then
    break
  fi
  if ! kill -0 "${first_pid}" 2>/dev/null; then
    break
  fi
  sleep 0.1
done

if [[ -z "${first_worker}" ]]; then
  printf 'first request did not expose worker attribution while in flight\n' >&2
  exit 1
fi

if ! kill -0 "${first_pid}" 2>/dev/null; then
  printf 'first request completed before overload could be created; start Mocker with the documented decode slowdown\n' >&2
  exit 1
fi

second_worker="$(request_worker 2)"

wait "${first_pid}"
first_pid=""

third_worker="$(request_worker 3)"

printf 'selected workers: %s -> %s -> %s\n' \
  "${first_worker}" "${second_worker}" "${third_worker}"

if [[ "${first_worker}" == "${second_worker}" ]]; then
  printf 'expected the policy to move the second request off the first soft pin\n' >&2
  exit 1
fi

if [[ "${second_worker}" != "${third_worker}" ]]; then
  printf 'expected the third request to retain the updated soft pin after load drained\n' >&2
  exit 1
fi

printf 'the overload threshold repinned the session and the new soft pin was retained\n'
