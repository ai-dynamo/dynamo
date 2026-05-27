#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${MODELEXPRESS_REF:?MODELEXPRESS_REF must be set}"

uv pip install --system --no-deps \
  "https://github.com/ai-dynamo/modelexpress/archive/${MODELEXPRESS_REF}.tar.gz#subdirectory=modelexpress_client/python"
