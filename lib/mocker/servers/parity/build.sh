#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Builds each mocker's `sidecar` test binary as $OUT/<engine>-parity.
set -euo pipefail

OUT="${OUT:-/out/tests}"
export OUT

# Never reuse compiled artifacts across checkouts on shared builders.
CARGO_TARGET_DIR="$(mktemp -d)"
export CARGO_TARGET_DIR
trap 'rm -rf "$CARGO_TARGET_DIR"' EXIT

CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
    cargo test --locked --no-run --test sidecar \
    -p dynamo-vllm-mocker -p dynamo-sglang-mocker -p dynamo-trtllm-mocker \
    --message-format=json-render-diagnostics > "$CARGO_TARGET_DIR/test-artifacts.json"
mkdir -p "$OUT"
python3 - <<'PY'
import json
import os
import pathlib
import shutil

artifacts = pathlib.Path(os.environ["CARGO_TARGET_DIR"]) / "test-artifacts.json"
for line in artifacts.read_text().splitlines():
    item = json.loads(line)
    if (
        item.get("reason") == "compiler-artifact"
        and item.get("profile", {}).get("test")
        and item.get("target", {}).get("name") == "sidecar"
        and item.get("executable")
    ):
        engine = pathlib.Path(item["manifest_path"]).parent.name
        shutil.copy2(item["executable"], pathlib.Path(os.environ["OUT"]) / f"{engine}-parity")
assert sorted(os.listdir(os.environ["OUT"])) == ["sglang-parity", "trtllm-parity", "vllm-parity"]
PY
strip "$OUT"/*-parity
