#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Every dynamo-kv-router feature builds on its own and all together, so no adapter leaks
# into another adapter's gate (SPLIT_V2 section 3: adapters behind features, default = []).
# Needs cargo-hack (cargo install cargo-hack --locked).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if ! command -v cargo-hack >/dev/null 2>&1; then
  echo "ERROR cargo-hack is not installed (cargo install cargo-hack --locked)" >&2
  exit 2
fi
cargo hack check --locked -p dynamo-kv-router --each-feature --all-targets
echo "router features: every dynamo-kv-router feature builds alone and together"
