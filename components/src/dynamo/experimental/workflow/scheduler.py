# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency scheduling for declarative workflow graphs."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from typing import Any, cast

from dynamo.experimental.workflow.dispatcher import StageDispatcher
from dynamo.experimental.workflow.ir import StageIR, WorkflowIR
from dynamo.experimental.workflow.perf import WORKFLOW_PERF_TRACE
from dynamo.experimental.workflow.runtime import StageContext
from dynamo.experimental.workflow.types import ValueRef

logger = logging.getLogger(__name__)


class GraphScheduler:
    """Resolve static dependencies and run ready graph stages concurrently."""

    def __init__(self, workflow: WorkflowIR, dispatcher: StageDispatcher) -> None:
        self._workflow = workflow
        self._dispatcher = dispatcher

    async def run(
        self,
        inputs: Mapping[str, Any],
        attempt_id: str,
        request_context: Any = None,
    ) -> dict[str, Any]:
        tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}

        async def run_stage(stage: StageIR) -> dict[str, Any]:
            started_ns = time.perf_counter_ns()
            stage_inputs = {}
            for name, reference in stage.inputs.items():
                value = await resolve_raw(reference)
                stage_inputs[name] = await self._dispatcher.resolve_edge(
                    reference,
                    stage.id,
                    name,
                    value,
                )
            dependencies_ready_ns = time.perf_counter_ns()
            result = await self._dispatcher.call(
                stage.id,
                stage_inputs,
                StageContext(
                    workflow_name=self._workflow.name,
                    stage_id=stage.id,
                    attempt_id=attempt_id,
                ),
                request_context=request_context,
            )
            WORKFLOW_PERF_TRACE.emit(
                logger,
                "workflow.stage",
                attempt_id,
                dependency_wait_ms=(dependencies_ready_ns - started_ns) / 1_000_000,
                elapsed_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
                stage=stage.id,
            )
            return result

        async def resolve_raw(reference: ValueRef) -> Any:
            if reference.input_name is not None:
                return inputs[reference.input_name]
            stage_id = cast(str, reference.stage_id)
            output_name = cast(str, reference.output_name)
            stage_outputs = await asyncio.shield(tasks[stage_id])
            return stage_outputs[output_name]

        async def resolve_output(name: str, reference: ValueRef) -> Any:
            value = await resolve_raw(reference)
            return self._dispatcher.resolve_workflow_output(name, value)

        for stage in self._workflow.stages:
            tasks[stage.id] = asyncio.create_task(
                run_stage(stage), name=f"workflow:{stage.id}"
            )

        try:
            output_values = await asyncio.gather(
                *(
                    resolve_output(name, reference)
                    for name, reference in self._workflow.outputs.items()
                )
            )
            return dict(zip(self._workflow.outputs, output_values))
        except BaseException:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            raise
