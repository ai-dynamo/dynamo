# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo worker entry point for the remote custom encoder example."""

from __future__ import annotations

import asyncio

from dynamo.experimental.endpoint import serve_unary_endpoint
from dynamo.experimental.llm import LLMUnaryClient
from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime, dynamo_worker

from .config import RemoteEncoderConfig
from .encoder import InlineEncoder
from .orchestrator import ExternalEncoderOrchestrator
from .request import EncoderResultRequestBuilder


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    """Serve an inline custom encoder backed by a remote stock vLLM worker."""

    config = RemoteEncoderConfig.from_env()

    decoder_client = await runtime.endpoint(config.generator_endpoint).client()
    await decoder_client.wait_for_instances()

    backend_class = config.resolve_backend_class()
    encoder = InlineEncoder.from_backend(backend_class(), model=config.model)
    try:
        endpoint = runtime.endpoint(config.orchestrator_endpoint)
        await register_model(
            ModelInput.Tokens,
            ModelType.Chat,
            endpoint,
            config.model,
            model_name=config.public_model_name,
            custom_template_path=str(config.chat_template_path),
            worker_type=WorkerType.Aggregated,
            ignore_weights=True,
        )
        orchestrator = ExternalEncoderOrchestrator(
            encoder,
            LLMUnaryClient(decoder_client),
            config.decoder_model_name,
            EncoderResultRequestBuilder(),
        )
        await serve_unary_endpoint(endpoint, orchestrator)
    finally:
        encoder.close()


def main() -> None:
    asyncio.run(worker())


if __name__ == "__main__":
    main()
