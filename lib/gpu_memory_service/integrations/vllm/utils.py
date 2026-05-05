# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS vLLM integration utilities."""

import logging
import os

logger = logging.getLogger(__name__)


def configure_gms_lock_mode(engine_args) -> None:
    """Set gms_read_only in model_loader_extra_config based on ENGINE_ID.

    In a failover setup with TP>1, only ENGINE_ID="0" loads weights from
    disk (RW_OR_RO). All other engines import from GMS (RO). This avoids
    deadlock: if multiple engines tried to acquire RW locks across TP ranks
    simultaneously, they could block each other indefinitely.

    Raises if user-specified gms_read_only conflicts with ENGINE_ID.
    """
    engine_id = os.environ.get("ENGINE_ID", "0")
    extra = engine_args.model_loader_extra_config or {}
    user_read_only = extra.get("gms_read_only", None)

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
        extra["gms_read_only"] = True

    engine_args.model_loader_extra_config = extra
