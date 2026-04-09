#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Wrapper: launch agg_multimodal.sh with logging suppressed to warn level.
# Used to measure tracing/logging overhead vs the default (info) config.

export DYN_LOG=warn
exec "$(dirname "$0")/../../../examples/backends/vllm/launch/agg_multimodal.sh" "$@"
