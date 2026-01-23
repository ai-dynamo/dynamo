# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal Streamline PD (Prefill+Decode) Worker Handler

Architecture: Frontend → PDWorker → Encoder

Streamline design philosophy:
- Frontend always talks to P/PD worker (single entry point)
- No separate Processor component
- P/PD worker orchestrates encoding internally
- Simpler deployment with fewer components
"""

import logging

from vllm.v1.engine.async_llm import AsyncLLM

from dynamo.runtime import Client, Component, DistributedRuntime

from ..handlers import BaseWorkerHandler
from ..multimodal_utils import ImageLoader

logger = logging.getLogger(__name__)


class MultimodalStreamlinePdWorkerHandler(BaseWorkerHandler):
    """
    Aggregated Prefill+Decode worker for Multimodal Streamline path.

    Responsibilities:
    - Receive requests from Frontend (registers as model endpoint)
    - Orchestrate encoding by calling encoder worker
    - Run both prefill and decode (no separate decode worker)

    Architecture:
        Frontend → [This Worker] <-> Encoder
                         ↓
            (orchestrate + prefill + decode)

    Use this for simpler deployments where P/D disaggregation is not needed.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        component: Component,
        engine_client: AsyncLLM,
        config,
        encoder_worker_client: Client = None,
    ):
        # Get default_sampling_params from config
        default_sampling_params = (
            config.engine_args.create_model_config().get_diff_sampling_param()
        )

        super().__init__(
            runtime,
            component,
            engine_client,
            default_sampling_params,
            enable_multimodal=True,
        )
        self.config = config
        self.encoder_worker_client = encoder_worker_client

        self.image_loader = ImageLoader()

        logger.info("MultimodalStreamlinePdWorkerHandler initialized")

    async def generate(self, request, context):
        """
        Process multimodal request: encode → prefill + decode.

        Args:
            request: Preprocessed request with token_ids and multimodal URLs
            context: Request context for cancellation handling

        Yields:
            Token outputs from inference
        """
        # TODO: Implement Streamline PD logic
        # 1. Extract multimodal URLs from request
        # 2. If has URLs, dispatch to encoder worker (async)
        # 3. Gather embeddings
        # 4. Run prefill + decode with embeddings
        # 5. Stream responses back
        raise NotImplementedError("MultimodalStreamlinePdWorkerHandler.generate")
        yield
