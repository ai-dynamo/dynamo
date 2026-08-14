# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency scheduling for declarative workflow graphs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, cast

from dynamo.workflow.dispatcher import StageDispatcher
from dynamo.workflow.ir import StageIR, WorkflowIR
from dynamo.workflow.runtime import StageContext, WorkflowAttempt
from dynamo.workflow.types import ValueRef


class GraphScheduler:
    """Resolve static dependencies and run ready graph stages concurrently."""

    def __init__(self, workflow: WorkflowIR, dispatcher: StageDispatcher) -> None:
        self._workflow = workflow
        self._dispatcher = dispatcher

    async def run(
        self, inputs: Mapping[str, Any], attempt: WorkflowAttempt
    ) -> dict[str, Any]:
        tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}

        async def run_stage(stage: StageIR) -> dict[str, Any]:
            stage_inputs = {}
            for name, reference in stage.inputs.items():
                value = await resolve_raw(reference)
                stage_inputs[name] = await self._dispatcher.resolve_edge(
                    reference,
                    stage.id,
                    name,
                    value,
                )
            return await self._dispatcher.call(
                stage.id,
                stage.contract,
                stage_inputs,
                StageContext(
                    workflow_name=self._workflow.name,
                    stage_id=stage.id,
                    attempt_id=attempt.attempt_id,
                    invocation_id=f"{attempt.attempt_id}:{stage.id}",
                    deadline=attempt.deadline,
                    _cancelled=attempt.cancelled,
                    request_context=attempt.request_context,
                ),
            )

        async def resolve_raw(reference: ValueRef) -> Any:
            if reference.input_name is not None:
                return inputs[reference.input_name]
            stage_id = cast(str, reference.stage_id)
            output_name = cast(str, reference.output_name)
            stage_outputs = await asyncio.shield(tasks[stage_id])
            return stage_outputs[output_name]

        for stage in self._workflow.stages:
            tasks[stage.id] = asyncio.create_task(
                run_stage(stage), name=f"workflow:{stage.id}"
            )

        try:
            output_values = await asyncio.gather(
                *(
                    resolve_raw(reference)
                    for reference in self._workflow.outputs.values()
                )
            )
            return dict(zip(self._workflow.outputs, output_values))
        except BaseException:
            attempt.cancelled.set()
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            raise
