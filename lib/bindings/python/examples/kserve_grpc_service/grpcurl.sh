#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Invoke the mock KServe gRPC endpoint using grpcurl. Requires grpcurl installed.
# The service does not expose server reflection, so we point grpcurl at the proto files directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="${SCRIPT_DIR}/../../../../llm/src/grpc/protos"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8787}"
MODEL="mock_model"

if [[ $# -gt 0 ]]; then
  PROMPTS=("$@")
else
  PROMPTS=(
    "Hello from Dynamo!"
    "How are you today?"
    "Tell me a joke."
  )
fi

encode_base64() {
  local text="$1"
  python - "$text" <<'PY'
import base64
import sys

print(base64.b64encode(sys.argv[1].encode("utf-8")).decode("ascii"))
PY
}

run_infer() {
  local prompt="$1"
  local encoded
  encoded="$(encode_base64 "$prompt")"

  printf -- '---\nSending prompt: %s\n' "$prompt"

  grpcurl \
    -plaintext \
    -import-path "${PROTO_DIR}" \
    -proto kserve.proto \
    -d "{
      \"model_name\": \"${MODEL}\",
      \"inputs\": [
        {
          \"name\": \"text_input\",
          \"datatype\": \"BYTES\",
          \"shape\": [1],
          \"contents\": { \"bytesContents\": [\"${encoded}\"] }
        }
      ]
    }" \
    "${HOST}:${PORT}" inference.GRPCInferenceService/ModelInfer
}

for prompt in "${PROMPTS[@]}"; do
  run_infer "$prompt"
done
