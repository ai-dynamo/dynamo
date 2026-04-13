# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM Integration for Dynamo KVBM

This module provides vLLM-specific configuration and utilities for
integrating with the Dynamo KVBM library.
"""

try:
    from kvbm._core import v2 as _v2

    # Note: version_check() is intentionally NOT called here. Calling it
    # at package init would force `import vllm` (transitively pulling
    # vllm._version, vllm.envs, vllm.logger, vllm.utils, …) every time
    # something just imports `kvbm.v2.vllm.connector` to scan the module
    # path. The check runs lazily from `kvbm.v2.vllm.config` instead,
    # which is the gateway every code path that actually needs vllm
    # transits through.
    from .version_check import version_check

    KvbmVllmConfig = _v2.KvbmVllmConfig
except ImportError:
    from kvbm._feature_stubs import _make_feature_stub

    KvbmVllmConfig = _make_feature_stub("KvbmVllmConfig", "v2")

__all__ = ["KvbmVllmConfig", "version_check"]
