# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implementations of :class:`BlockIndexStore`."""

from gpu_memory_service.kv_cache.backends.mmap import MmapBlockIndexStore

__all__ = ["MmapBlockIndexStore"]
