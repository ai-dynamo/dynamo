# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-bound workflow stage dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from dynamo.workflow.plan import ExecutionPlan, InlineBinding
from dynamo.workflow.runtime import (
    StageContext,
    StageRunner,
    WorkflowExecutionError,
    _validate_value,
)
from dynamo.workflow.types import StageContract, WorkflowValidationError


class StageDispatcher:
    """Validate and invoke stages through their compiled bindings."""

    def __init__(
        self,
        plan: ExecutionPlan,
        inline_runners: Mapping[str, StageRunner],
    ) -> None:
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must use ExecutionPlan")
        if not isinstance(inline_runners, Mapping):
            raise TypeError("inline_runners must be a mapping")

        expected_keys = {
            binding.runner_key
            for binding in plan.bindings.values()
            if isinstance(binding, InlineBinding)
        }
        actual_keys = set(inline_runners)
        if actual_keys != expected_keys:
            raise WorkflowValidationError(
                "inline runners differ from execution plan; "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )

        runners = dict(inline_runners)
        for stage_id, contract in plan.stage_contracts.items():
            binding = plan.bindings[stage_id]
            if not isinstance(binding, InlineBinding):
                raise WorkflowValidationError(
                    f"dispatcher does not support binding for stage {stage_id!r}"
                )
            runner = runners[binding.runner_key]
            if not isinstance(runner, StageRunner):
                raise WorkflowValidationError(
                    f"runner {binding.runner_key!r} must implement StageRunner"
                )
            if runner.contract != contract:
                raise WorkflowValidationError(
                    f"runner {binding.runner_key!r} for stage {stage_id!r} "
                    "does not match its authored contract"
                )

        self._plan = plan
        self._inline_runners = MappingProxyType(runners)

    async def call(
        self,
        stage_id: str,
        contract: StageContract,
        inputs: Mapping[str, Any],
        context: StageContext,
    ) -> dict[str, Any]:
        """Invoke one stage and validate its complete input/output contract."""

        context.raise_if_cancelled()
        expected_inputs = set(contract.inputs)
        actual_inputs = set(inputs)
        if actual_inputs != expected_inputs:
            raise WorkflowExecutionError(
                f"stage {stage_id!r} inputs differ from its contract; "
                f"missing={sorted(expected_inputs - actual_inputs)}, "
                f"extra={sorted(actual_inputs - expected_inputs)}"
            )
        for input_name, spec in contract.inputs.items():
            _validate_value(
                spec, inputs[input_name], f"stage {stage_id!r} input {input_name!r}"
            )

        binding = self._plan.bindings[stage_id]
        if not isinstance(binding, InlineBinding):
            raise WorkflowExecutionError(f"unsupported binding for stage {stage_id!r}")
        result = await self._inline_runners[binding.runner_key].run(
            MappingProxyType(dict(inputs)), context
        )
        if not isinstance(result, Mapping):
            raise WorkflowExecutionError(
                f"stage {stage_id!r} returned a non-mapping result"
            )
        expected_outputs = set(contract.outputs)
        actual_outputs = set(result)
        if actual_outputs != expected_outputs:
            raise WorkflowExecutionError(
                f"stage {stage_id!r} outputs differ from its contract; "
                f"missing={sorted(expected_outputs - actual_outputs)}, "
                f"extra={sorted(actual_outputs - expected_outputs)}"
            )
        outputs = dict(result)
        for output_name, spec in contract.outputs.items():
            _validate_value(
                spec,
                outputs[output_name],
                f"stage {stage_id!r} output {output_name!r}",
            )
        return outputs
