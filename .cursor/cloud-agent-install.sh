#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

export PATH="/usr/local/bin:/usr/local/cargo/bin:${PATH:-}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

ensure_libstdcxx_link() {
  if [[ -e /usr/lib/x86_64-linux-gnu/libstdc++.so ]]; then
    return 0
  fi
  local gcc_ver
  gcc_ver="$(gcc -dumpversion | cut -d. -f1)"
  if [[ -f "/usr/lib/gcc/x86_64-linux-gnu/${gcc_ver}/libstdc++.so" ]]; then
    sudo ln -sf "/usr/lib/gcc/x86_64-linux-gnu/${gcc_ver}/libstdc++.so" \
      /usr/lib/x86_64-linux-gnu/libstdc++.so
  fi
}

ensure_libstdcxx_link

ensure_protoc() {
  if ! command -v protoc >/dev/null; then
    echo "protoc not found on PATH; install system packages via .cursor/Dockerfile or apt." >&2
    exit 1
  fi
  if [[ ! -f /usr/local/include/google/protobuf/struct.proto ]]; then
    local ver="${PROTOC_VERSION:-30.2}"
    local arch="$(uname -m)"
    local protoc_zip
    case "${arch}" in
      x86_64) protoc_zip="protoc-${ver}-linux-x86_64.zip" ;;
      aarch64) protoc_zip="protoc-${ver}-linux-aarch_64.zip" ;;
      *) echo "unsupported arch for protoc: ${arch}" >&2; exit 1 ;;
    esac
    local tmp="$(mktemp -d)"
    curl -fsSL "https://github.com/protocolbuffers/protobuf/releases/download/v${ver}/${protoc_zip}" \
      -o "${tmp}/protoc.zip"
    sudo unzip -o "${tmp}/protoc.zip" -d /usr/local bin/protoc include/*
    rm -rf "${tmp}"
  fi
}

ensure_protoc

ensure_nats_server() {
  if command -v nats-server >/dev/null; then
    return 0
  fi
  NATS_VERSION="${NATS_VERSION:-v2.14.8}"
  arch="$(uname -m)"
  case "${arch}" in
    x86_64) NATS_ARCH=amd64 ;;
    aarch64) NATS_ARCH=arm64 ;;
    *) echo "unsupported arch for nats-server: ${arch}" >&2; exit 1 ;;
  esac
  tmp="$(mktemp -d)"
  curl -fsSL "https://github.com/nats-io/nats-server/releases/download/${NATS_VERSION}/nats-server-${NATS_VERSION}-linux-${NATS_ARCH}.tar.gz" \
    | tar -xzf - -C "${tmp}"
  sudo install -m 0755 "${tmp}/nats-server-${NATS_VERSION}-linux-${NATS_ARCH}/nats-server" /usr/local/bin/nats-server
  rm -rf "${tmp}"
}

ensure_nats_server

if ! command -v uv >/dev/null; then
  curl -fsSL https://astral.sh/uv/install.sh | sh
  sudo ln -sf "${HOME}/.local/bin/uv" /usr/local/bin/uv
  sudo ln -sf "${HOME}/.local/bin/uvx" /usr/local/bin/uvx
fi

if [[ ! -f .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install -q pip 'maturin[patchelf]'

MATURIN_FEATURES="kv-indexer,slot-tracker,select-service,mm-routing,ais-forward-pass,request-trace-s3"
(
  cd lib/bindings/python
  maturin develop --uv --features "${MATURIN_FEATURES}"
)

uv pip install -q -e lib/gpu_memory_service
uv pip install -q -e .
uv pip install -q -r container/deps/requirements.test.txt
uv pip install -q pre-commit

pre-commit install --install-hooks || true

python3 -m dynamo.frontend --help >/dev/null
echo "cloud-agent-install: dynamo.frontend OK"
