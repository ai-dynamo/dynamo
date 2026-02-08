# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker initialization modules for TensorRT-LLM backend.

This package contains worker initialization functions for different modalities:
- llm_worker: Text and multimodal LLM inference
- video_diffusion_worker: Video generation using diffusion models

The init_worker() function dispatches to the appropriate worker based on modality.

Note on import strategy:
- llm_worker is imported eagerly (standard dependency, always available)
- video_diffusion_worker is imported lazily because it depends on visual_gen,
  an optional package only available on TensorRT-LLM's feat/visual_gen branch.
  Eager import would break text/multimodal users who don't have it installed.
"""

import asyncio
import logging

from dynamo.runtime import DistributedRuntime
from dynamo.trtllm.constants import Modality
from dynamo.trtllm.utils.trtllm_utils import Config
from dynamo.trtllm.workers.llm_worker import init_llm_worker


async def init_worker(
    runtime: DistributedRuntime, config: Config, shutdown_event: asyncio.Event
) -> None:
    """Initialize the appropriate worker based on modality.

    Dispatches to the correct worker initialization function based on the
    configured modality (text, multimodal, video_diffusion, etc.).

    Args:
        runtime: The Dynamo distributed runtime.
        config: Configuration parsed from command line.
        shutdown_event: Event to signal shutdown.
    """
    logging.info(f"Initializing worker with modality={config.modality}")

    modality = Modality(config.modality)

    if Modality.is_diffusion(modality):
        if modality == Modality.VIDEO_DIFFUSION:
            from dynamo.trtllm.workers.video_diffusion_worker import (
                init_video_diffusion_worker,
            )

            await init_video_diffusion_worker(runtime, config, shutdown_event)
            return
        # TODO(nv-yna): Add IMAGE_DIFFUSION support in follow-up PR

    # LLM modalities (text, multimodal)
    await init_llm_worker(runtime, config, shutdown_event)


__all__ = ["init_worker"]
