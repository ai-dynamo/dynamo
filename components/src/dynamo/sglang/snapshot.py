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
)

from .pause import SGLangEnginePauseController

logger = logging.getLogger(__name__)


async def warmup_engine(engine: sgl.Engine, server_args: Any) -> None:
    """Warm up the direct SGLang Engine generation path before snapshot capture."""
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
            "temperature": 0,
            "max_new_tokens": 8,
        }
    }
    if server_args.skip_tokenizer_init:
        warmup_args["input_ids"] = [[10, 11, 12] for _ in range(server_args.dp_size)]
        if server_args.dp_size == 1:
            warmup_args["input_ids"] = warmup_args["input_ids"][0]
    else:
        warmup_args["prompt"] = ["The capital city of France is"] * server_args.dp_size
        if server_args.dp_size == 1:
            warmup_args["prompt"] = warmup_args["prompt"][0]

    if server_args.debug_tensor_dump_input_file:
        import numpy as np

        warmup_args.pop("prompt", None)
        warmup_args["input_ids"] = np.load(
            server_args.debug_tensor_dump_input_file
        ).tolist()
        warmup_args["sampling_params"]["max_new_tokens"] = 0

    is_disaggregated = server_args.disaggregation_mode != "null"
    if is_disaggregated:
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

        logger.info("Start of pd disaggregation warmup ...")
        warmup_args = {
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 8,
                "ignore_eos": True,
            },
            "bootstrap_host": [FAKE_BOOTSTRAP_HOST] * server_args.dp_size,
            # This is a hack to ensure fake transfer is enabled during
            # prefill warmup and each DP rank has a unique room.
            "bootstrap_room": [
                i * (2**63 // server_args.dp_size) + (i % server_args.tp_size)
                for i in range(server_args.dp_size)
            ],
            "input_ids": [[10, 11, 12, 13]] * server_args.dp_size,
        }

    warmup_timeout = envs.SGLANG_WARMUP_TIMEOUT.get()
    timeout = warmup_timeout if warmup_timeout > 0 else 600
    if is_disaggregated:
        timeout = warmup_timeout if warmup_timeout > 0 else 1800

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

    # Enable memory_saver so GPU memory can be released for CRIU.
    # When using GMS, weights use VA-stable unmap/remap (no CPU backup); GMS
    # forbids enable_weights_cpu_backup. Otherwise use CPU backup for weights.
    server_args.enable_memory_saver = True
    try:
        from gpu_memory_service.integrations.sglang import is_gms_active

        _using_gms = is_gms_active()
    except ImportError:
        _using_gms = False
    if not _using_gms:
        server_args.enable_weights_cpu_backup = True

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
