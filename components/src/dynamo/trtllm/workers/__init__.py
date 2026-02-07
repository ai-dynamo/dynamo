# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker initialization modules for TensorRT-LLM backend.

This package contains worker initialization functions for different modalities:
- llm_worker: Text and multimodal LLM inference using TensorRT-LLM
- video_diffusion_worker: Video generation using diffusion models
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from dynamo.trtllm.constants import Modality

if TYPE_CHECKING:
    from dynamo.runtime import DistributedRuntime
    from dynamo.trtllm.utils.trtllm_utils import Config


async def init_worker(
    runtime: DistributedRuntime, config: Config, shutdown_event: asyncio.Event
) -> None:
    """Dispatch to the appropriate worker based on modality.

    Uses lazy imports to avoid eagerly loading heavy dependencies
    (tensorrt_llm for LLM, visual_gen for diffusion) when only one
    modality is needed.

    Args:
        runtime: The Dynamo distributed runtime.
        config: Configuration parsed from command line.
        shutdown_event: Event to signal shutdown.
    """
    logging.info(f"Initializing the worker with config: {config}")

    modality = Modality(config.modality)

    if Modality.is_diffusion(modality):
        if modality == Modality.VIDEO_DIFFUSION:
            # Lazy import: visual_gen is only needed for diffusion
            from dynamo.trtllm.workers.video_diffusion_worker import (
                init_video_diffusion_worker,
            )

            await init_video_diffusion_worker(runtime, config, shutdown_event)
            return
        # TODO(nv-yna): Add IMAGE_DIFFUSION support in follow-up PR

    # Lazy import: tensorrt_llm is only needed for LLM modalities
    from dynamo.trtllm.workers.llm_worker import init_llm_worker

    await init_llm_worker(runtime, config, shutdown_event)
