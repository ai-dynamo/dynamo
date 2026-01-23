# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal Streamline Encoder Worker Handler

Architecture: Frontend → PrefillWorker → Encoder → (PrefillWorker) → DecodeWorker

Streamline design philosophy:
- Frontend always talks to P/PD worker (single entry point)
- No separate Processor component
- P/PD worker orchestrates encoding internally
- Simpler deployment with fewer components
"""

import logging

from dynamo.runtime import Component, DistributedRuntime

from ..multimodal_utils import ImageLoader

logger = logging.getLogger(__name__)


class MultimodalStreamlineEncoderWorkerHandler:
    """
    Encoder worker for Multimodal Streamline path.

    Responsibilities:
    - Receive requests (URL or data) from Prefill/PD worker
    - Load and process media through vision model
    - Return embeddings to the calling worker

    This handler does NOT register as a model endpoint - it's called
    internally by MultimodalStreamlinePrefillWorkerHandler or MultimodalStreamlinePdWorkerHandler.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        component: Component,
        config,
    ):
        self.runtime = runtime
        self.component = component
        self.config = config

        # Vision model components (lazy initialized)
        self._vision_model = None
        self._image_processor = None
        self._vision_encoder = None
        self._projector = None

        self.image_loader = ImageLoader()

        logger.info("MultimodalStreamlineEncoderWorkerHandler initialized")

    async def encode(self, request, context):
        """
        Encode multimodal inputs and return embeddings.

        Args:
            request: Request containing multimodal URLs to encode
            context: Request context for cancellation handling

        Yields:
            Encoded embeddings for each multimodal input
        """
        # TODO: Implement encoding logic
        # 1. Parse request to extract image/video URLs
        # 2. Load media using self.image_loader
        # 3. Process through vision model
        # 4. Return embeddings
        raise NotImplementedError("MultimodalStreamlineEncoderWorkerHandler.encode")
        yield
