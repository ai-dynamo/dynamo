# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM Integration for Dynamo KVBM

This module provides vLLM-specific configuration and utilities for
integrating with the Dynamo KVBM library.
"""

from kvbm._core.v2 import KvbmVllmConfig

__all__ = ["KvbmVllmConfig"]
