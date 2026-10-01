#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../../.."

cd "${REPO_ROOT}"

export RUN_IMAGE_DECODE_SWEEP=1
export DYNAMO_REQUIRE_LIBJPEG_TURBO_TEST=1

exec cargo bench -p dynamo-llm --bench image_decode -- "$@"
