# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shadow mode utilities for GMS vLLM integration."""

import logging
import os

from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.integrations.common.utils import parse_requested_lock_type

logger = logging.getLogger(__name__)


def is_shadow_mode() -> bool:
    """True when DYN_GMS_SHADOW_MODE=1 (set by main.py at startup)."""
    return os.environ.get("DYN_GMS_SHADOW_MODE", "0") == "1"


def validate_cudagraph_mode(engine_args) -> None:
    """Validate and set cudagraph mode for shadow engines.

    Defaults unset mode to PIECEWISE (attention stubbed during graph capture).
    Accepts NONE (e.g. enforce_eager). Rejects FULL variants which need
    KV cache tensors that don't exist during shadow init.
    """
    from vllm.config import CompilationConfig, CUDAGraphMode

    cc = engine_args.compilation_config
    assert isinstance(cc, CompilationConfig), (
        f"Expected CompilationConfig, got {type(cc).__name__}. "
        f"vLLM's arg parsing may have changed."
    )

    if cc.cudagraph_mode is None:
        cc.cudagraph_mode = CUDAGraphMode.PIECEWISE
        logger.info("[Shadow] cudagraph_mode defaulted to PIECEWISE")
    elif cc.cudagraph_mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE):
        pass  # compatible
    else:
        raise ValueError(
            f"Shadow mode requires PIECEWISE or NONE cudagraph mode, "
            f"got {cc.cudagraph_mode.name}. FULL modes capture attention ops "
            f"that need KV cache tensors, which don't exist during shadow init."
        )


def configure_gms_lock_mode(engine_args) -> None:
    """Set gms_lock_mode in model_loader_extra_config based on ENGINE_ID.

    In a failover setup with TP>1, only ENGINE_ID="0" loads weights from
    disk (RW_OR_RO). All other engines import from GMS (RO). This avoids
    deadlock: if multiple engines tried to acquire RW locks across TP ranks
    simultaneously, they could block each other indefinitely.

    Raises if user-specified gms_lock_mode conflicts with ENGINE_ID.
    """
    engine_id = os.environ.get("ENGINE_ID", "0")
    extra = dict(engine_args.model_loader_extra_config or {})
    requested_mode = extra.get("gms_lock_mode")
    if requested_mode is not None:
        try:
            requested_mode = parse_requested_lock_type(requested_mode)
            extra["gms_lock_mode"] = requested_mode.value
        except ValueError as exc:
            raise ValueError(
                f"Invalid gms_lock_mode {requested_mode!r}. "
                "Expected one of: rw, ro, rw_or_ro."
            ) from exc

    if engine_id == "0":
        if requested_mode == RequestedLockType.RO:
            raise ValueError(
                "ENGINE_ID=0 is the primary writer but "
                "gms_lock_mode='ro' was explicitly set. "
                "The primary engine must be able to write weights."
            )
    else:
        if requested_mode is None:
            extra["gms_lock_mode"] = RequestedLockType.RO.value
        elif requested_mode != RequestedLockType.RO:
            raise ValueError(
                f"ENGINE_ID={engine_id} requires gms_lock_mode='ro', "
                f"but gms_lock_mode={requested_mode.value!r} was explicitly set."
            )

    engine_args.model_loader_extra_config = extra
