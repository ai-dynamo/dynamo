# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

_LEGACY_GPU_SKU_ALIASES = {
    "gb200_sxm": "gb200",
}


def resolve_ais_system(gpu_sku: str | Enum) -> str:
    """Return the canonical AISimulate system identifier for a DGDR GPU SKU."""
    value = gpu_sku.value if isinstance(gpu_sku, Enum) else str(gpu_sku)
    normalized = value.lower()
    return _LEGACY_GPU_SKU_ALIASES.get(normalized, normalized)
