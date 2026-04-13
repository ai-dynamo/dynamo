# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shadow mode utilities for GMS vLLM integration."""

import logging
import os

logger = logging.getLogger(__name__)

_logging_configured = False


def configure_gms_logging() -> None:
    """Attach a handler to the gpu_memory_service logger.

    vLLM only configures the ``vllm`` logger. Without this, all
    ``gpu_memory_service.*`` log messages are silently dropped inside
    the EngineCore worker process.

    Reuses vLLM's handler and formatter when available so that GMS
    log lines match the surrounding vLLM output style.

    ModelExpress configures its own logger separately via
    ``modelexpress.configure_logging()``.
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    vllm_logger = logging.getLogger("vllm")
    if vllm_logger.handlers:
        handler = vllm_logger.handlers[0]
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
                datefmt="%m-%d %H:%M:%S",
            )
        )

    gms_logger = logging.getLogger("gpu_memory_service")
    if not gms_logger.handlers:
        gms_logger.addHandler(handler)
        gms_logger.setLevel(logging.INFO)


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


def configure_mx_ports(engine_args) -> None:
    """Offset MX metadata and gRPC base ports per engine to avoid bind collisions.

    In failover pods, both engines share a network namespace. MX binds
    NIXL metadata on ``MX_METADATA_PORT + device_id`` and worker gRPC on
    ``MX_WORKER_GRPC_PORT + device_id``. Without offsetting, engines with
    the same device_id collide (EADDRINUSE on the NIXL listener).

    Offsets by ``engine_id * tp_size`` so each engine's port range is
    non-overlapping:
        engine 0, TP=2: metadata 5555-5556, gRPC 6555-6556
        engine 1, TP=2: metadata 5557-5558, gRPC 6557-6558
    """
    if os.environ.get("MX_ENABLED", "0") != "1":
        return

    engine_id = int(os.environ.get("ENGINE_ID", "0"))
    offset = engine_id * (engine_args.tensor_parallel_size or 1)
    mx_metadata_base = int(os.environ.get("MX_METADATA_PORT", "5555")) + offset
    mx_grpc_base = int(os.environ.get("MX_WORKER_GRPC_PORT", "6555")) + offset
    os.environ["MX_METADATA_PORT"] = str(mx_metadata_base)
    os.environ["MX_WORKER_GRPC_PORT"] = str(mx_grpc_base)

    logger.info(
        "[Shadow] MX ports for engine-%d: metadata=%d, grpc=%d",
        engine_id,
        mx_metadata_base,
        mx_grpc_base,
    )
