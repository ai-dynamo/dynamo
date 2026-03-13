# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo Snapshot integration for SGLang workers."""

import logging
import time

import sglang as sgl

from dynamo.common.utils.snapshot import get_checkpoint_config

logger = logging.getLogger(__name__)

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

        await self._engine.tokenizer_manager.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=_MEMORY_TAGS), None
        )
        await self._engine.tokenizer_manager.continue_generation(
            ContinueGenerationReqInput()
        )


async def handle_checkpoint_mode(
    server_args, dynamo_args
) -> tuple[bool, sgl.Engine | None]:
    """Single entry point for Dynamo Snapshot integration.

    Must be called BEFORE runtime creation so the engine can be checkpointed
    without active NATS/etcd connections.

    Returns:
        (should_exit, engine) where:
        - (True, None): caller should return immediately (checkpoint already
          exists, or checkpoint completed successfully).
        - (False, None): not in checkpoint mode — cold-start normally.
        - (False, engine): restore completed — caller should use this engine.
    """
    should_exit, checkpoint_cfg = get_checkpoint_config()
    if should_exit:
        return True, None

    if checkpoint_cfg is None:
        return False, None

    unsupported_roles = [
        role
        for enabled, role in [
            (dynamo_args.embedding_worker, "embedding"),
            (dynamo_args.multimodal_processor, "multimodal-processor"),
            (dynamo_args.multimodal_encode_worker, "multimodal-encode-worker"),
            (dynamo_args.multimodal_worker, "multimodal-worker"),
            (dynamo_args.image_diffusion_worker, "image-diffusion"),
            (dynamo_args.video_generation_worker, "video-generation"),
            (dynamo_args.diffusion_worker, "llm-diffusion"),
        ]
        if enabled
    ]
    if unsupported_roles:
        raise ValueError(
            "checkpoint mode only supports standard, decode, and prefill LLM SGLang workers; "
            f"got {', '.join(unsupported_roles)}"
        )

    logger.info("Checkpoint mode enabled (watcher-driven signals)")

    # Enable memory_saver + weights CPU backup so weights survive CRIU
    # (mirrors vLLM's enable_sleep_mode = True)
    server_args.enable_memory_saver = True
    server_args.enable_weights_cpu_backup = True

    start_time = time.time()
    engine = sgl.Engine(server_args=server_args)
    logger.info(
        f"SGLang engine loaded in {time.time() - start_time:.2f}s (checkpoint mode)"
    )

    adapter = SGLangCheckpointAdapter(engine)
    if not await checkpoint_cfg.run_lifecycle(adapter, _SLEEP_MODE_LEVEL):
        return True, None

    return False, engine
