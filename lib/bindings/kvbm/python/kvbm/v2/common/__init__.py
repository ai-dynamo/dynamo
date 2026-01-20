# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for v2 connectors.

This module provides shared classes and functions used by both vLLM and TRTLLM
v2 connector implementations.
"""

from .handshake import (
    ConnectorLeaderProtocol,
    NovaPeerMetadata,
    register_workers_from_handshake,
)

__all__ = [
    "NovaPeerMetadata",
    "ConnectorLeaderProtocol",
    "register_workers_from_handshake",
]
