# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM logits processor adapters for Dynamo.

This module provides adapters that wrap Dynamo's BaseLogitsProcessor instances
to work with vLLM's logits processor interface.
"""

from .adapter import VllmDynamoLogitsAdapter, create_vllm_adapters

__all__ = ["VllmDynamoLogitsAdapter", "create_vllm_adapters"]
