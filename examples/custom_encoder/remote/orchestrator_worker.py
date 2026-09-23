# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo worker entry point for the remote custom encoder example."""

from __future__ import annotations

import asyncio

from dynamo.llm import LLMUnaryClient, ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime, dynamo_worker, serve_unary_endpoint
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.multimodal_utils.custom_encoder import ExternalEncoderHandoff

from .config import RemoteEncoderConfig
from .orchestrator import ExternalEncoderOrchestrator

configure_dynamo_logging()


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    """Serve an inline custom encoder backed by a remote stock vLLM worker."""

    config = RemoteEncoderConfig.from_env()

    generator_client = await runtime.endpoint(config.generator_endpoint).client()
    await generator_client.wait_for_instances()

    backend_class = config.resolve_backend_class()
    handoff = ExternalEncoderHandoff(
        backend_class(),
        name="remote-custom-encoder",
    )
    try:
        handoff.load(config.model)
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
            handoff,
            LLMUnaryClient(generator_client),
            config.generator_model_name,
        )
        await serve_unary_endpoint(endpoint, orchestrator)
    finally:
        handoff.shutdown()


def main() -> None:
    asyncio.run(worker())


if __name__ == "__main__":
    main()
