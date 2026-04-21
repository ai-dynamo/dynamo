# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shadow mode utilities for GMS TensorRT-LLM integration."""

import logging
import os

logger = logging.getLogger(__name__)


def is_shadow_mode() -> bool:
    """True when DYN_GMS_SHADOW_MODE=1 (set by llm_worker.py at startup)."""
    return os.environ.get("DYN_GMS_SHADOW_MODE", "0") == "1"


def configure_gms_lock_mode(extra_config: dict) -> dict:
    """Set gms_read_only in model_loader_extra_config based on ENGINE_ID.

    Shadow-engine failover ensures that only ENGINE_ID="0" loads weights
    from disk (RW_OR_RO). All other engines import from the committed GMS
    layout (RO). This avoids deadlock when multiple engines contend for RW
    locks across TP ranks.

    Raises if user-specified gms_read_only conflicts with ENGINE_ID.
    Mutates and returns the same dict so callers can chain.
    """
    engine_id = os.environ.get("ENGINE_ID", "0")
    user_read_only = extra_config.get("gms_read_only", None)

    if engine_id == "0":
        if user_read_only:
            raise ValueError(
                "ENGINE_ID=0 is the primary writer but "
                "gms_read_only=True was explicitly set. "
                "The primary engine must be able to write weights."
            )
    else:
        if user_read_only is not None and not user_read_only:
            raise ValueError(
                f"ENGINE_ID={engine_id} requires gms_read_only=True, "
                f"but gms_read_only=False was explicitly set."
            )
        extra_config["gms_read_only"] = True

    logger.info(
        "[Shadow] ENGINE_ID=%s → gms_read_only=%s",
        engine_id,
        extra_config.get("gms_read_only", False),
    )
    return extra_config
