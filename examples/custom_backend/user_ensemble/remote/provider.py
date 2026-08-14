# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend provider for the remote user-ensemble workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dynamo.workflow import WorkflowExecutor, WorkflowFrontendApplication
from examples.custom_backend.user_ensemble.config import DEFAULT_MODEL
from examples.custom_backend.user_ensemble.remote.bindings import (
    compile_remote_workflow,
)
from examples.custom_backend.user_ensemble.workflow import adapt_workflow_result


async def provide_workflow(
    runtime: Any, frontend_config: Any
) -> WorkflowFrontendApplication:
    executor = await WorkflowExecutor.bind(
        compile_remote_workflow(),
        runtime=runtime,
    )
    model_path = (
        getattr(frontend_config, "model_path", None)
        or os.environ.get("DYN_MODEL")
        or DEFAULT_MODEL
    )
    model_name = (
        getattr(frontend_config, "model_name", None)
        or os.environ.get("DYN_SERVED_MODEL_NAME")
        or model_path
    )
    template_path = os.environ.get("DYN_CUSTOM_JINJA_TEMPLATE")
    return WorkflowFrontendApplication(
        executor=executor,
        model_path=model_path,
        model_name=model_name,
        custom_template_path=None if template_path is None else Path(template_path),
        result_adapter=adapt_workflow_result,
    )
