# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the declarative user ensemble as one discovered model worker."""

import asyncio

from dynamo.experimental.workflow import WorkflowEndpointHandler, WorkflowOrchestrator
from dynamo.experimental.workflow.vllm import ExternalEncoderRequestStage
from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime, dynamo_worker
from examples.experimental.workflow.user_ensemble.common.config import (
    CHAT_TEMPLATE,
    MODEL,
    ORCHESTRATOR_ENDPOINT,
    PUBLIC_MODEL_NAME,
    build_encoder_stage,
)
from examples.experimental.workflow.user_ensemble.common.stages import (
    DummyClassifier,
    EnsembleResponseStage,
)
from examples.experimental.workflow.user_ensemble.workflow.workflow import (
    compile_user_ensemble,
)


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    encoder = build_encoder_stage()
    try:
        orchestrator = await WorkflowOrchestrator.bind(
            compile_user_ensemble(),
            runtime=runtime,
            inline_runners={
                "encoder": encoder,
                "classifier": DummyClassifier(),
                "request_adapter": ExternalEncoderRequestStage(),
                "response": EnsembleResponseStage(),
            },
        )
        endpoint = runtime.endpoint(ORCHESTRATOR_ENDPOINT)
        await register_model(
            ModelInput.Tokens,
            ModelType.Chat,
            endpoint,
            MODEL,
            model_name=PUBLIC_MODEL_NAME,
            worker_type=WorkerType.Aggregated,
            custom_template_path=str(CHAT_TEMPLATE),
            ignore_weights=True,
        )
        await endpoint.serve_endpoint(WorkflowEndpointHandler(orchestrator).generate)
    finally:
        encoder.close()


def main() -> None:
    asyncio.run(worker())


if __name__ == "__main__":
    main()
