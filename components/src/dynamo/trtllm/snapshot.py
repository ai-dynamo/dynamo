# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo Snapshot integration for TRT-LLM workers."""

import gc
import logging
import time
from collections.abc import Callable

from dynamo.common.utils.snapshot import CheckpointConfig, EngineSnapshotController

from .args import Config
from .engine import TensorRTLLMEngine
from .request_handlers.handler_base import TRTLLMEngineQuiesceController

logger = logging.getLogger(__name__)


async def prepare_snapshot_engine(
    config: Config,
    setup_trtllm_engine: Callable[[Config], TensorRTLLMEngine],
) -> EngineSnapshotController[TensorRTLLMEngine] | None:
    """Single entry point for Dynamo Snapshot integration with TRT-LLM.

    Must be called BEFORE runtime creation so the engine can be checkpointed
    without active NATS/etcd connections.

    Weight quiesce (GMS) is intentionally excluded from the snapshot scope.
    Snapshot mode only quiesces KV cache via TRT-LLM's _collective_rpc.

    Args:
        config: Parsed TRT-LLM worker configuration.
        setup_trtllm_engine: Callable that constructs and returns a
            TensorRTLLMEngine from config, without registering it with the
            Dynamo runtime. Provided by the refactored worker init (see
            components/src/dynamo/trtllm/workers/llm_worker.py).

    Returns:
        None when not in checkpoint mode (DYN_SNAPSHOT_CONTROL_DIR unset).
        An EngineSnapshotController wrapping the restored engine when restore
        completed and the caller should proceed with normal worker startup
        using the pre-built engine.

        Exits the process with status 0 if checkpointing completed
        successfully (checkpoint path, not restore path).
    """
    checkpoint_config = CheckpointConfig.from_env()
    if checkpoint_config is None:
        return None

    logger.info("Checkpoint mode enabled (watcher-driven signals)")

    start_time = time.time()
    engine = setup_trtllm_engine(config)
    logger.info(
        "TRT-LLM engine loaded in %.2fs (checkpoint mode)",
        time.time() - start_time,
    )

    gc.collect()

    snapshot_controller = EngineSnapshotController(
        engine=engine,
        quiesce_controller=TRTLLMEngineQuiesceController(engine),
        checkpoint_config=checkpoint_config,
        # Snapshot scope: KV cache only. Weight quiesce requires GMS and is
        # tracked separately. Passing ["kv_cache"] skips _release_gms_weights
        # in TRTLLMEngineQuiesceController.quiesce().
        quiesce_args=(["kv_cache"],),
    )
    if not await snapshot_controller.wait_for_restore():
        raise SystemExit(0)

    return snapshot_controller
