#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Never reuse compiled artifacts across checkouts on shared builders.
CARGO_TARGET_DIR="$(mktemp -d)"
export CARGO_TARGET_DIR
trap 'rm -rf "$CARGO_TARGET_DIR"' EXIT

CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
    cargo test --locked -p dynamo-trtllm-sidecar --lib --no-run \
    --message-format=json > /tmp/trtllm-test-artifacts.json
python3 - <<'PY'
import json
import pathlib
import shutil

artifacts = [json.loads(line) for line in pathlib.Path('/tmp/trtllm-test-artifacts.json').read_text().splitlines()]
executables = [item['executable'] for item in artifacts
               if item.get('reason') == 'compiler-artifact'
               and item.get('profile', {}).get('test')
               and item.get('target', {}).get('name') == 'dynamo_trtllm_sidecar'
               and item.get('executable')]
assert len(executables) == 1, executables
pathlib.Path('/out/tests').mkdir(parents=True, exist_ok=True)
shutil.copy2(executables[0], '/out/tests/trtllm-image-tests')
PY
strip /out/tests/trtllm-image-tests
