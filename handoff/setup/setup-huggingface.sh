#!/usr/bin/env bash

set -euo pipefail

PY="${PYTHON:-python3}"
MODEL="${DYNAMO_TEST_MODEL:-cerebras/DeepSeek-V3.2-REAP-345B-A37B}"

if ! "$PY" -c 'import huggingface_hub' >/dev/null 2>&1; then
  "$PY" -m pip install --user --upgrade huggingface_hub
fi

if command -v hf >/dev/null 2>&1; then
  HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=huggingface-cli
else
  user_bin="$("$PY" -m site --user-base)/bin"
  export PATH="$user_bin:$PATH"
  if command -v hf >/dev/null 2>&1; then
    HF=hf
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF=huggingface-cli
  else
    echo "ERROR: Hugging Face CLI not found after installing huggingface_hub" >&2
    exit 1
  fi
fi

if [ ! -s "$HOME/.cache/huggingface/token" ]; then
  "$HF" login
fi

"$HF" whoami

if [ "${HF_PREPULL:-0}" = "1" ]; then
  "$HF" download "$MODEL"
else
  echo "Skipping model download. Set HF_PREPULL=1 to download $MODEL."
fi
