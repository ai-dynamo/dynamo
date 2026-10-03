#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Idempotent Cloud Agent bootstrap for Dynamo development (CPU / no GPU required
# for build and most unit tests). Installs system packages, uv, Python venv,
# maturin-built runtime bindings, and the editable ai-dynamo wheel.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CXX="${CXX:-g++}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"

UV_VERSION="${UV_VERSION:-0.8.22}"

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  local tmp
  tmp="$(mktemp -d)"
  curl -fsSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    | tar -xzf - -C "$tmp"
  sudo install -m 0755 "$tmp/uv-x86_64-unknown-linux-gnu/uv" /usr/local/bin/uv
  rm -rf "$tmp"
}

install_system_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  # g++-14 headers: default `c++` is clang and expects gcc-14 includes for zmq-sys.
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    build-essential g++ g++-14 libstdc++-14-dev \
    libhwloc-dev libudev-dev pkg-config libclang-dev protobuf-compiler python3-dev cmake \
    gh \
    || true
}

ensure_demo_model() {
  local demo="$REPO_ROOT/.cursor/demo-model"
  [[ -f "$demo/config.json" ]] && return 0
  mkdir -p "$demo"
  cat >"$demo/config.json" <<'EOF'
{
  "model_type": "qwen3",
  "num_hidden_layers": 2,
  "num_key_value_heads": 4,
  "num_attention_heads": 8,
  "hidden_size": 64,
  "torch_dtype": "bfloat16"
}
EOF
  DEMO_MODEL_DIR="$demo" python3 - <<'PY'
import json
import os
from pathlib import Path

demo = Path(os.environ["DEMO_MODEL_DIR"])
tok = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": None,
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": None,
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": False,
        "byte_fallback": False,
        "vocab": {"<|endoftext|>": 0, "Hello": 1, "!": 2},
        "merges": [],
    },
}
(demo / "tokenizer.json").write_text(json.dumps(tok))
(demo / "tokenizer_config.json").write_text(
    json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
)
PY
}

install_system_packages
install_uv

if [[ ! -d "$REPO_ROOT/.venv" ]]; then
  uv venv "$REPO_ROOT/.venv"
fi

# shellcheck source=/dev/null
source "$REPO_ROOT/.venv/bin/activate"

uv pip install -q pip 'maturin[patchelf]' pytest pre-commit

pushd "$REPO_ROOT/lib/bindings/python" >/dev/null
maturin develop --uv
popd >/dev/null

uv pip install -q -e "$REPO_ROOT/lib/gpu_memory_service"
uv pip install -q -e "$REPO_ROOT"

ensure_demo_model
bash "$REPO_ROOT/.cursor/setup-personal-skills.sh"

python3 -m dynamo.frontend --help >/dev/null
python3 -m dynamo.mocker --help >/dev/null

echo "Dynamo Cloud Agent install complete."
