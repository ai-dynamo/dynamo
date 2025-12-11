# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM Integration for Dynamo KVBM

This module provides vLLM-specific configuration and utilities for
integrating with the Dynamo KVBM library.
"""

try:
    from kvbm._core import v2 as _v2

    KvbmVllmConfig = _v2.KvbmVllmConfig
except ImportError:
    from kvbm._feature_stubs import _make_feature_stub

    KvbmVllmConfig = _make_feature_stub("KvbmVllmConfig", "v2")

__all__ = ["KvbmVllmConfig"]
