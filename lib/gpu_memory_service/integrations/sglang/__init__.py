# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service integration for SGLang.

Usage:
    from gpu_memory_service.integrations.sglang import setup_gms

    if server_args.load_format == "gms":
        server_args.load_format = setup_gms(server_args)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from gpu_memory_service.integrations.sglang.model_loader import GMSModelLoader

logger = logging.getLogger(__name__)


def setup_gms(server_args) -> Type["GMSModelLoader"]:
    """Setup GPU Memory Service for SGLang.

    Validates config and returns the GMSModelLoader class.
    Patches are applied automatically when GMSModelLoader is imported.

    Args:
        server_args: SGLang ServerArgs instance.

    Returns:
        GMSModelLoader class to use as load_format.

    Raises:
        ValueError: If incompatible options are enabled.
    """
    # Validate config - GMS provides its own VA-stable unmap/remap for weights
    if getattr(server_args, "enable_weights_cpu_backup", False):
        raise ValueError(
            "Cannot use --enable-weights-cpu-backup with --load-format gms."
        )
    if getattr(server_args, "enable_draft_weights_cpu_backup", False):
        raise ValueError(
            "Cannot use --enable-draft-weights-cpu-backup with --load-format gms."
        )

    # Import triggers patches at module level
    from gpu_memory_service.integrations.sglang.model_loader import GMSModelLoader

    logger.info("[GMS] Using GMSModelLoader...")
    return GMSModelLoader
