# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service server components."""

from gpu_memory_service.server.client import AllocationServerClient
from gpu_memory_service.server.registry import ArtifactRegistry
from gpu_memory_service.server.server import AllocationServer

__all__ = [
    "AllocationServer",
    "AllocationServerClient",
    "ArtifactRegistry",
]
