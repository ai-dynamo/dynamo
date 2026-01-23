# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal Streamline Prefill Worker Handler

Architecture: Frontend → PrefillWorker → Encoder → (PrefillWorker) → DecodeWorker

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

logger = logging.getLogger(__name__)


class MultimodalStreamlinePrefillWorkerHandler(BaseWorkerHandler):
    """
    Prefill worker for Multimodal Streamline path (disaggregated mode).

    Responsibilities:
    - Receive requests from Frontend (registers as model endpoint)
    - Orchestrate encoding by calling encoder worker
    - Run prefill with embeddings
    - Forward to decode worker for token generation

    Architecture:
        Frontend → [This Worker] → Encoder → [This Worker] → Decode Worker
                         ↓                         ↓
                    (orchestrate)              (prefill)
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        component: Component,
        engine_client: AsyncLLM,
        config,
        encoder_worker_client: Client = None,
        decode_worker_client: Client = None,
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
            enable_multimodal=config.enable_multimodal,
        )

        self.config = config
        self.encoder_worker_client = encoder_worker_client
        self.decode_worker_client = decode_worker_client

        logger.info("MultimodalStreamlinePrefillWorkerHandler initialized")

    async def generate(self, request, context):
        """
        Process multimodal request: encode → prefill → forward to decode.

        Args:
            request: Preprocessed request with token_ids and multimodal URLs
            context: Request context for cancellation handling

        Yields:
            Token outputs from decode worker
        """
        # TODO: Implement Streamline prefill logic
        # 1. Extract multimodal URLs from request
        # 2. If has URLs, dispatch to encoder worker (async)
        # 3. Gather embeddings
        # 4. Run prefill with embeddings
        # 5. Forward to decode worker with kv_transfer_params
        # 6. Stream decode responses back
        raise NotImplementedError("MultimodalStreamlinePrefillWorkerHandler.generate")
        yield
