# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Internal bindings for Dynamo components.

These classes are used by internal Dynamo components (components/) and are not
part of the stable public API. They may change without notice.

For public APIs, use dynamo.runtime and dynamo.llm.
"""

# Re-export from _core
from dynamo._core import ModelDeploymentCard as ModelDeploymentCard
from dynamo._core import start_kv_block_indexer as start_kv_block_indexer

__all__ = [
    "ModelDeploymentCard",
    "start_kv_block_indexer",
]
