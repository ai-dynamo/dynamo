#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Regression test for the architecture-selection prompt in build_and_deploy.sh.
#
# An invalid answer must produce the script's own "Error: Invalid architecture"
# message and let the user retry, per prompt_for_steps/validate_architecture's
# documented contract. Under `set -euo pipefail`, a bare `VAR=$(fn)` assignment
# whose command substitution returns non-zero terminates the shell immediately
# -- before the `if [ $? -eq 0 ]` check that prints the retry message ever
# runs -- so a single typo at that prompt silently kills the whole script.
#
# No test framework is used anywhere in this repo for shell scripts, so this
# is a plain, self-contained bash script matching the style already used for
# `tests/**/run_*_test.sh`.

set -uo pipefail # not -e: we need to observe the target script's own exit code

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/build_and_deploy.sh"

FAKEBIN="$(mktemp -d)"
trap 'rm -rf "$FAKEBIN"' EXIT
for tool in wget unzip kubectl docker; do
  printf '#!/bin/bash\nexit 0\n' >"$FAKEBIN/$tool"
  chmod +x "$FAKEBIN/$tool"
done

# A single invalid architecture answer at the first prompt.
OUTPUT=$(printf 'not-a-real-arch\n' | PATH="$FAKEBIN:$PATH" timeout 10 bash "$TARGET" 2>&1)

if grep -q 'Error: Invalid architecture' <<<"$OUTPUT"; then
  echo "1 passed, 0 failed"
  echo "PASS: invalid architecture input produced the documented retry error"
  exit 0
else
  echo "0 passed, 1 failed"
  echo "FAIL: invalid architecture input silently killed the script instead of"
  echo "      printing 'Error: Invalid architecture...' and retrying. Observed output:"
  echo "$OUTPUT"
  exit 1
fi
