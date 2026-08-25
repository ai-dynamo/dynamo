# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the user ensemble through one discovered orchestrator worker."""

from __future__ import annotations

import asyncio
import os

from dynamo.experimental.workflow import WorkflowEndpointHandler, WorkflowOrchestrator
from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime, dynamo_worker
from examples.experimental.workflow.user_ensemble.config import DEFAULT_MODEL
from examples.experimental.workflow.user_ensemble.remote.bindings import (
    compile_remote_workflow,
)
from examples.experimental.workflow.user_ensemble.stages import EnsembleResponseStage

ORCHESTRATOR_ENDPOINT = "user-ensemble.orchestrator.generate"


@dynamo_worker()
async def orchestrator_worker(runtime: DistributedRuntime) -> None:
    """Bind the graph and advertise its endpoint as the user-facing model."""

    orchestrator = await WorkflowOrchestrator.bind(
        compile_remote_workflow(),
        runtime=runtime,
        inline_runners={"response": EnsembleResponseStage()},
    )
    endpoint = runtime.endpoint(ORCHESTRATOR_ENDPOINT)
    model = os.environ.get("DYN_MODEL", DEFAULT_MODEL)
    served_model_name = os.environ.get("DYN_SERVED_MODEL_NAME", model)
    custom_template_path = os.environ.get("DYN_CUSTOM_JINJA_TEMPLATE") or None
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat,
        endpoint,
        model,
        model_name=served_model_name,
        custom_template_path=custom_template_path,
        worker_type=WorkerType.Aggregated,
        ignore_weights=True,
    )
    await endpoint.serve_endpoint(WorkflowEndpointHandler(orchestrator).generate)


def main() -> None:
    asyncio.run(orchestrator_worker())


if __name__ == "__main__":
    main()
