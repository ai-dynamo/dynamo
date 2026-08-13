# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency scheduling for declarative workflow graphs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, cast

from dynamo.experimental.workflow.dispatcher import StageDispatcher
from dynamo.experimental.workflow.ir import StageIR, WorkflowIR
from dynamo.experimental.workflow.runtime import StageContext
from dynamo.experimental.workflow.types import ValueRef


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
        tensor_exports: dict[
            ValueRef, asyncio.Task[Mapping[str, Mapping[str, Any]]]
        ] = {}

        async def run_stage(stage: StageIR) -> dict[str, Any]:
            stage_inputs = {}
            for name, reference in stage.inputs.items():
                value = await resolve_raw(reference)
                stage_inputs[name] = await self._dispatcher.resolve_edge(
                    reference,
                    stage.id,
                    name,
                    value,
                    tensor_exports,
                )
            return await self._dispatcher.call(
                stage.id,
                stage_inputs,
                StageContext(
                    workflow_name=self._workflow.name,
                    stage_id=stage.id,
                    attempt_id=attempt_id,
                ),
                request_context=request_context,
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
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            for export_task in tensor_exports.values():
                if not export_task.done():
                    export_task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            await asyncio.gather(*tensor_exports.values(), return_exceptions=True)
            raise
