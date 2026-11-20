#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

: "${DYNAMO_SOURCE_COMMIT:?set the checked-out Dynamo commit}"
: "${TRTLLM_TEST_IMAGE:?set the resolved upstream image digest}"
# Artifacts are test-only, but use current source and the production feature set.
CARGO_TARGET_DIR="$(mktemp -d)"
export CARGO_TARGET_DIR CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0
# cudarc 0.19.8 supports CUDA through 13.3; native nightlies may ship newer toolkits.
export CUDARC_CUDA_VERSION=13030
trap 'rm -rf "$CARGO_TARGET_DIR"' EXIT
mkdir -p /out/serve-runtime/bin /out/serve-runtime/wheels
cargo build --locked -p dynamo-trtllm-sidecar --bin dynamo-trtllm-sidecar
install "$CARGO_TARGET_DIR/debug/dynamo-trtllm-sidecar" /out/serve-runtime/bin/
strip /out/serve-runtime/bin/dynamo-trtllm-sidecar
maturin build --locked --profile dev --strip --compatibility linux \
    --manifest-path lib/bindings/python/Cargo.toml \
    --features kv-indexer,slot-tracker,select-service,mm-routing,ais-forward-pass,request-trace-s3 \
    --out /out/serve-runtime/wheels
python3 - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import pathlib
import sys

root = pathlib.Path('/out/serve-runtime')
files = {}
for path in sorted(root.rglob('*')):
    if path.is_file():
        files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / 'build.json').write_text(json.dumps({
    'source_commit': os.environ['DYNAMO_SOURCE_COMMIT'],
    'native_image': os.environ['TRTLLM_TEST_IMAGE'],
    'native_commit': os.environ['TRT_LLM_GIT_COMMIT'],
    'native_version': importlib.metadata.version('tensorrt_llm'),
    'cudarc_cuda_version': os.environ['CUDARC_CUDA_VERSION'],
    'python': sys.version,
    'files': files,
}, indent=2) + '\n')
PY
