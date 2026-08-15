# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend provider for the integrated-encoder user ensemble."""

from __future__ import annotations

from typing import Any

from dynamo.workflow import WorkflowOrchestrator
from examples.custom_backend.user_ensemble.remote.bindings import (
    compile_remote_workflow,
)
from examples.custom_backend.user_ensemble.stages import EnsembleResponseStage


async def provide_workflow(runtime: Any) -> WorkflowOrchestrator:
    """Bind remote model stages and the inline frontend response stage."""

    return await WorkflowOrchestrator.bind(
        compile_remote_workflow(),
        runtime=runtime,
        inline_runners={"response": EnsembleResponseStage()},
    )
