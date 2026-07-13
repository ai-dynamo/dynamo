# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo Snapshot integration for SGLang workers."""

import asyncio
import gc
import logging
import time
from typing import Any

import sglang as sgl

from dynamo.common.snapshot.lifecycle import (
    EngineSnapshotController,
    SnapshotConfig,
    configure_snapshot_capture_env,
    unified_snapshot_outcome,
)
from dynamo.common.backend import PreRuntimeOutcome
from dynamo.llm.exceptions import Cancelled, EngineShutdown, InvalidArgument

from .pause import SGLangEnginePauseController

logger = logging.getLogger(__name__)


def _configure_server_args_for_snapshot(server_args: Any) -> None:
    # Enable memory_saver so GPU memory can be released for CRIU. GMS uses
    # VA-stable unmap/remap and forbids CPU weight backup.
    server_args.enable_memory_saver = True
    try:
        from gpu_memory_service.integrations.sglang import is_gms_active

        using_gms = is_gms_active()
    except ImportError:
        using_gms = False
    if not using_gms:
        server_args.enable_weights_cpu_backup = True


async def prepare_unified_snapshot(
    engine: Any,
    *,
    argv: list[str] | None,
    context: Any,
) -> PreRuntimeOutcome:
    snapshot_config = SnapshotConfig.from_env()
    if snapshot_config is None:
        return PreRuntimeOutcome.continue_startup()

    try:
        configure_snapshot_capture_env()
        _configure_server_args_for_snapshot(engine.server_args)
        logger.info("Unified SGLang snapshot mode enabled")
        await engine._initialize_engine()
        await warmup_engine(engine.engine, engine.server_args)
        if engine._pause_controller is None:
            raise RuntimeError("SGLang snapshot pause controller was not initialized")
    except (Cancelled, InvalidArgument):
        raise
    except Exception as exc:
        raise EngineShutdown(f"SGLang snapshot_prepare failed: {exc}") from exc
    gc.collect()

    controller = EngineSnapshotController(
        engine=engine,
        pause_controller=engine._pause_controller,
        snapshot_config=snapshot_config,
    )
    return await unified_snapshot_outcome(
        controller,
        argv=argv,
        context=context,
        backend_name="SGLang",
    )


async def warmup_engine(engine: sgl.Engine, server_args: Any) -> None:
    """Warm up the direct SGLang Engine generation path before snapshot capture."""
    DUMMY_WARMUP_MAX_NEW_TOKENS = 8
    DUMMY_WARMUP_TEMPERATURE = 0
    DUMMY_WARMUP_INPUT_IDS = [10, 11, 12]
    DUMMY_WARMUP_PROMPT = "The capital city of France is"
    DUMMY_DEBUG_TENSOR_MAX_NEW_TOKENS = 0
    DUMMY_DISAGG_WARMUP_TEMPERATURE = 0.0
    DUMMY_DISAGG_WARMUP_INPUT_IDS = [10, 11, 12, 13]
    DUMMY_BOOTSTRAP_ROOM_RANGE = 2**63
    DEFAULT_WARMUP_TIMEOUT = 600
    DEFAULT_DISAGG_WARMUP_TIMEOUT = 1800

    # The direct Engine API does not run SGLang's HTTP server warmup. Snapshot
    # mode only needs to warm the generation path before the engine is frozen.
    if getattr(server_args, "skip_server_warmup", False):
        return

    tokenizer_manager = engine.tokenizer_manager
    if not tokenizer_manager.is_generation:
        logger.info("Skipping SGLang snapshot warmup for non-generation model")
        return

    from sglang.srt.environ import envs

    warmup_args: dict[str, Any] = {
        "sampling_params": {
            "temperature": DUMMY_WARMUP_TEMPERATURE,
            "max_new_tokens": DUMMY_WARMUP_MAX_NEW_TOKENS,
        }
    }
    if server_args.skip_tokenizer_init:
        warmup_args["input_ids"] = [
            DUMMY_WARMUP_INPUT_IDS for _ in range(server_args.dp_size)
        ]
        if server_args.dp_size == 1:
            warmup_args["input_ids"] = warmup_args["input_ids"][0]
    else:
        warmup_args["prompt"] = [DUMMY_WARMUP_PROMPT] * server_args.dp_size
        if server_args.dp_size == 1:
            warmup_args["prompt"] = warmup_args["prompt"][0]

    if server_args.debug_tensor_dump_input_file:
        import numpy as np

        warmup_args.pop("prompt", None)
        warmup_args["input_ids"] = np.load(
            server_args.debug_tensor_dump_input_file
        ).tolist()
        warmup_args["sampling_params"]["max_new_tokens"] = (
            DUMMY_DEBUG_TENSOR_MAX_NEW_TOKENS
        )

    is_disaggregated = server_args.disaggregation_mode != "null"
    if is_disaggregated:
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

        logger.info("Start of pd disaggregation warmup ...")
        warmup_args = {
            "sampling_params": {
                "temperature": DUMMY_DISAGG_WARMUP_TEMPERATURE,
                "max_new_tokens": DUMMY_WARMUP_MAX_NEW_TOKENS,
                "ignore_eos": True,
            },
            "bootstrap_host": [FAKE_BOOTSTRAP_HOST] * server_args.dp_size,
            # This is a hack to ensure fake transfer is enabled during
            # prefill warmup and each DP rank has a unique room.
            "bootstrap_room": [
                i * (DUMMY_BOOTSTRAP_ROOM_RANGE // server_args.dp_size)
                + (i % server_args.tp_size)
                for i in range(server_args.dp_size)
            ],
            "input_ids": [DUMMY_DISAGG_WARMUP_INPUT_IDS] * server_args.dp_size,
        }

    warmup_timeout = envs.SGLANG_WARMUP_TIMEOUT.get()
    timeout = warmup_timeout if warmup_timeout > 0 else DEFAULT_WARMUP_TIMEOUT
    if is_disaggregated:
        timeout = (
            warmup_timeout if warmup_timeout > 0 else DEFAULT_DISAGG_WARMUP_TIMEOUT
        )

    logger.info("SGLang snapshot warmup starting")
    await asyncio.wait_for(
        engine.async_generate(**warmup_args),
        timeout=timeout,
    )
    logger.info("SGLang snapshot warmup complete")


async def prepare_snapshot_engine(
    server_args,
) -> EngineSnapshotController[sgl.Engine] | None:
    """Single entry point for Dynamo Snapshot integration.

    Must be called BEFORE runtime creation so the engine can be snapshotted
    without active NATS/etcd connections.

    Returns:
        None when not in snapshot mode.
        A snapshot controller when restore completed and the caller should use
        the restored engine.

        If snapshotting completed successfully, this function exits the
        process with status 0.
    """
    snapshot_config = SnapshotConfig.from_env()
    if snapshot_config is None:
        return None

    configure_snapshot_capture_env()
    logger.info("Snapshot mode enabled (watcher-driven signals)")

    _configure_server_args_for_snapshot(server_args)

    start_time = time.time()
    engine = sgl.Engine(server_args=server_args)
    logger.info(
        f"SGLang engine loaded in {time.time() - start_time:.2f}s (snapshot mode)"
    )
    await warmup_engine(engine, server_args)

    gc.collect()

    snapshot_controller = EngineSnapshotController(
        engine=engine,
        pause_controller=SGLangEnginePauseController(engine),
        snapshot_config=snapshot_config,
    )
    if not await snapshot_controller.wait_for_restore():
        raise SystemExit(0)

    return snapshot_controller
