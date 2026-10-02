# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo worker entry point for the remote custom encoder example."""

from __future__ import annotations

import asyncio

from dynamo.llm import LLMUnaryClient, UnaryChatModel
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.multimodal_utils.custom_encoder import ExternalEncoderHandoff

from .config import RemoteEncoderConfig
from .orchestrator import DummyClassifier, ExternalEncoderOrchestrator

configure_dynamo_logging()


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    """Serve an inline custom encoder backed by a remote stock vLLM worker."""

    config = RemoteEncoderConfig.from_env()

    generator_client = await runtime.endpoint(config.generator_endpoint).client()
    await generator_client.wait_for_instances()

    handoff = ExternalEncoderHandoff(
        config.encoder_class(),
        name=config.service_name,
    )
    try:
        handoff.load(config.model)
        orchestrator = ExternalEncoderOrchestrator(
            handoff,
            LLMUnaryClient(generator_client),
            DummyClassifier(),
            config.generator_model_name,
        )
        await UnaryChatModel(
            model_path=config.model,
            service_name=config.service_name,
            public_model_name=config.public_model_name,
            chat_template=config.chat_template_path,
        ).serve(runtime, orchestrator)
    finally:
        handoff.shutdown()


def main() -> None:
    asyncio.run(worker())


if __name__ == "__main__":
    main()
