# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical invocation targets for authored workflow stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from dynamo.experimental.workflow.runtime import StageRunner
from dynamo.experimental.workflow.types import (
    StageContract,
    WorkflowValidationError,
    validate_name,
)


@dataclass(frozen=True)
class InlineBinding:
    """Bind one logical stage to an initialized in-process runner."""

    runner: StageRunner

    def __post_init__(self) -> None:
        if not isinstance(self.runner, StageRunner):
            raise WorkflowValidationError("inline runner must implement StageRunner")


def _validate_endpoint_id(endpoint_id: str) -> None:
    if not isinstance(endpoint_id, str):
        raise WorkflowValidationError("remote endpoint id must be a string")
    parts = endpoint_id.split(".")
    if len(parts) != 3:
        raise WorkflowValidationError(
            "remote endpoint id must use 'namespace.component.endpoint'"
        )
    for kind, part in zip(("namespace", "component", "endpoint"), parts):
        validate_name(part, f"remote {kind}")


@dataclass(frozen=True)
class RemoteBinding:
    """Bind one logical stage to a discovered Dynamo endpoint."""

    endpoint_id: str
    routing_policy: str = "round_robin"

    def __post_init__(self) -> None:
        _validate_endpoint_id(self.endpoint_id)
        if self.routing_policy != "round_robin":
            raise WorkflowValidationError(
                f"unsupported remote routing policy {self.routing_policy!r}"
            )


@dataclass(frozen=True)
class GenerateEndpointBinding(RemoteBinding):
    """Bind a contracted stage to Dynamo's stock token Generate endpoint."""


Binding = Union[InlineBinding, RemoteBinding]


def validate_binding_contract(binding: Binding, contract: StageContract) -> None:
    """Validate protocol-specific stage ports against one physical binding."""

    if not isinstance(binding, GenerateEndpointBinding):
        return
    expected_inputs = {"request", "encoder_features", "encoder_metadata"}
    if contract.inputs != expected_inputs:
        raise WorkflowValidationError(
            "Generate endpoint stage inputs must be request, encoder_features, "
            "and encoder_metadata"
        )
    if contract.outputs != {"chunk"}:
        raise WorkflowValidationError("Generate endpoint stage output must be chunk")
