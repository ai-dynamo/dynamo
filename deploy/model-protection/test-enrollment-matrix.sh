#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-test}"
[[ "$MODE" == test || "$MODE" == lint || "$MODE" == simulator || "$MODE" == hardware ]] || { echo 'Expected test, lint, simulator or hardware' >&2; exit 2; }
cd "$ROOT_DIR"

if [[ "$MODE" == simulator ]]; then
  command -v swtpm >/dev/null
  command -v swtpm_ioctl >/dev/null
  cargo test --locked -p dynamo-model-protection --features enrollment-authority,enrollment-client \
    --lib swtpm_ -- --ignored --nocapture
  exit
fi

if [[ "$MODE" == hardware ]]; then
  : "${DYNAMO_TPM_HARDWARE_TEST_DIR:?Set the private directory for already approved test handles 0x81012001-03}"
  cargo test --locked -p dynamo-model-protection --features enrollment-authority,enrollment-client \
    --lib physical_tpm_ -- --ignored --nocapture
  exit
fi

FEATURES=("" packager tpm2 packager,tpm2 enrollment-client enrollment-authority enrollment-authority,enrollment-client)
for feature in "${FEATURES[@]}"; do
  ARGS=(--locked -p dynamo-model-protection)
  [[ -z "$feature" ]] || ARGS+=(--features "$feature")
  if [[ "$MODE" == test ]]; then
    cargo test "${ARGS[@]}" --lib
  else
    cargo clippy "${ARGS[@]}" --no-deps --all-targets -- -D warnings
  fi
done
if [[ "$MODE" == test ]]; then
  cargo test --locked -p dynamo-model-protection --features enrollment-authority,enrollment-client --all-targets
else
  cargo fmt --package dynamo-model-protection -- --check
fi
