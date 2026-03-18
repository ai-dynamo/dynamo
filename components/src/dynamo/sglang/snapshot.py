# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo Snapshot integration for SGLang workers."""


import ctypes
import ctypes.util
import gc
import logging
import os
import time

import sglang as sgl

from dynamo.common.utils.snapshot import CheckpointConfig, EngineSnapshotController

from .request_handlers.handler_base import SGLangEngineQuiesceController

logger = logging.getLogger(__name__)


def _try_release_memory(label: str) -> None:
    """Force Python GC and glibc malloc_trim to return freed memory to the OS.

    Logs RSS before/after so you can see how much memory was actually reclaimable.
    """
    pid = os.getpid()

    def _get_rss_kb() -> int:
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except Exception:
            pass
        return 0

    rss_before = _get_rss_kb()

    collected = gc.collect()
    rss_after_gc = _get_rss_kb()

    try:
        libc_name = ctypes.util.find_library("c")
        if libc_name:
            libc = ctypes.CDLL(libc_name)
            libc.malloc_trim(0)
    except Exception as e:
        logger.debug("[MemRelease:%s] malloc_trim failed: %s", label, e)

    rss_after_trim = _get_rss_kb()

    logger.info(
        "[MemRelease:%s] gc.collect freed %d objects, "
        "RSS: %.2f MiB -> %.2f MiB (gc) -> %.2f MiB (malloc_trim), "
        "reclaimed=%.2f MiB",
        label,
        collected,
        rss_before / 1024,
        rss_after_gc / 1024,
        rss_after_trim / 1024,
        (rss_before - rss_after_trim) / 1024,
    )


_SLEEP_MODE_LEVEL = 1

# Memory tags to release/resume for CRIU checkpoint/restore.
# All GPU resources must be released so CRIU can snapshot the process cleanly.
_MEMORY_TAGS = ["kv_cache", "weights", "cuda_graph"]


class SGLangCheckpointAdapter:
    """Adapts an sgl.Engine to the sleep/wake_up interface expected by
    CheckpointConfig.run_lifecycle (matching vLLM's AsyncLLM API).

    sleep():   pause generation -> release GPU memory
    wake_up(): resume GPU memory -> continue generation
    """

    def __init__(self, engine: sgl.Engine):
        self._engine = engine

    async def sleep(self, level: int = 1) -> None:
        from sglang.srt.managers.io_struct import (
            PauseGenerationReqInput,
            ReleaseMemoryOccupationReqInput,
        )

        # Drain in-flight requests before touching GPU memory
        await self._engine.tokenizer_manager.pause_generation(PauseGenerationReqInput())
        await self._engine.tokenizer_manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=_MEMORY_TAGS), None
        )

    async def wake_up(self) -> None:
        from sglang.srt.managers.io_struct import (
            ContinueGenerationReqInput,
            ResumeMemoryOccupationReqInput,
        )

        # Synchronize the CUDA context in this (tokenizer-manager) process so
        # any pending ops complete before we message the scheduler.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning("CUDA sync before resume_memory_occupation failed: %s", e)

        await self._engine.tokenizer_manager.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=_MEMORY_TAGS), None
        )
        await self._engine.tokenizer_manager.continue_generation(
            ContinueGenerationReqInput()
        )


async def prepare_snapshot_engine(
    server_args,
) -> EngineSnapshotController[sgl.Engine] | None:
    """Single entry point for Dynamo Snapshot integration.

    Must be called BEFORE runtime creation so the engine can be checkpointed
    without active NATS/etcd connections.

    Returns:
        None when not in checkpoint mode.
        A snapshot controller when restore completed and the caller should use
        the restored engine.

        If checkpointing completed successfully, this function exits the
        process with status 0.
    """
    checkpoint_cfg = CheckpointConfig.from_env()
    if checkpoint_cfg is None:
        return None

    logger.info("Checkpoint mode enabled (watcher-driven signals)")

    # Enable memory_saver so GPU memory can be released for CRIU.
    # When using GMS, weights use VA-stable unmap/remap (no CPU backup); GMS
    # forbids enable_weights_cpu_backup. Otherwise use CPU backup for weights.
    server_args.enable_memory_saver = True
    _using_gms = getattr(server_args, "load_format", None) == "gms" or (
        isinstance(getattr(server_args, "load_format", None), type)
        and getattr(server_args.load_format, "__name__", "") == "GMSModelLoader"
    )
    if not _using_gms:
        server_args.enable_weights_cpu_backup = True

    start_time = time.time()
    engine = sgl.Engine(server_args=server_args)
    logger.info(
        f"SGLang engine loaded in {time.time() - start_time:.2f}s (checkpoint mode)"
    )

    _try_release_memory("after_engine_load")

    snapshot_controller = EngineSnapshotController(
        engine=engine,
        quiesce_controller=SGLangEngineQuiesceController(engine),
        checkpoint_config=checkpoint_cfg,
    )
    if not await snapshot_controller.wait_for_restore():
        raise SystemExit(0)

    return snapshot_controller
