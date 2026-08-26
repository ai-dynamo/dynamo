# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-bound workflow stage dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from dynamo.experimental.workflow.bindings import Binding, InlineBinding
from dynamo.experimental.workflow.ir import WorkflowIR
from dynamo.experimental.workflow.runtime import StageContext, WorkflowExecutionError
from dynamo.experimental.workflow.types import WorkflowValidationError, validate_name


class StageDispatcher:
    """Validate and invoke stages through their physical bindings."""

    def __init__(
        self,
        workflow: WorkflowIR,
        bindings: Mapping[str, Binding],
    ) -> None:
        if not isinstance(workflow, WorkflowIR):
            raise TypeError("workflow must use WorkflowIR")
        if not isinstance(bindings, Mapping):
            raise TypeError("bindings must be a mapping")

        bound_stages: dict[str, Binding] = {}
        for stage_id, binding in sorted(bindings.items()):
            validate_name(stage_id, "binding stage id")
            if not isinstance(binding, InlineBinding):
                raise WorkflowValidationError(
                    f"binding for stage {stage_id!r} uses an unsupported type"
                )
            bound_stages[stage_id] = binding

        contracts = {stage.id: stage.contract for stage in workflow.stages}
        expected_stages = set(contracts)
        actual_stages = set(bound_stages)
        if actual_stages != expected_stages:
            raise WorkflowValidationError(
                "bindings differ from workflow stages; "
                f"missing={sorted(expected_stages - actual_stages)}, "
                f"extra={sorted(actual_stages - expected_stages)}"
            )

        for stage_id, contract in contracts.items():
            if bound_stages[stage_id].runner.contract != contract:
                raise WorkflowValidationError(
                    f"inline runner for stage {stage_id!r} "
                    "does not match its authored contract"
                )

        self._contracts = MappingProxyType(contracts)
        self._bindings = MappingProxyType(bound_stages)

    async def call(
        self,
        stage_id: str,
        inputs: Mapping[str, Any],
        context: StageContext,
    ) -> dict[str, Any]:
        """Invoke one stage and validate its complete input/output contract."""

        contract = self._contracts[stage_id]
        expected_inputs = set(contract.inputs)
        actual_inputs = set(inputs)
        if actual_inputs != expected_inputs:
            raise WorkflowExecutionError(
                f"stage {stage_id!r} inputs differ from its contract; "
                f"missing={sorted(expected_inputs - actual_inputs)}, "
                f"extra={sorted(actual_inputs - expected_inputs)}"
            )
        binding = self._bindings[stage_id]
        if not isinstance(binding, InlineBinding):
            raise WorkflowExecutionError(f"unsupported binding for stage {stage_id!r}")
        result = await binding.runner.run(MappingProxyType(dict(inputs)), context)
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
        return outputs
