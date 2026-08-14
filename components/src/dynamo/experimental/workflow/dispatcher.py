# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-bound workflow stage dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from dynamo.experimental.workflow.bindings import Binding, InlineBinding, RemoteBinding
from dynamo.experimental.workflow.ir import WorkflowIR
from dynamo.experimental.workflow.remote import RemoteStageClient
from dynamo.experimental.workflow.runtime import StageContext, WorkflowExecutionError
from dynamo.experimental.workflow.types import (
    StageContract,
    WorkflowValidationError,
    validate_name,
)


@runtime_checkable
class RemoteStageInvoker(Protocol):
    """Internal transport boundary used by the dispatcher."""

    async def run(
        self,
        stage_id: str,
        contract: StageContract,
        inputs: Mapping[str, Any],
        context: StageContext,
    ) -> Mapping[str, Any]:
        ...


class StageDispatcher:
    """Validate and invoke stages through their physical bindings."""

    def __init__(
        self,
        workflow: WorkflowIR,
        bindings: Mapping[str, Binding],
        remote_clients: Mapping[str, RemoteStageInvoker] = MappingProxyType({}),
    ) -> None:
        if not isinstance(workflow, WorkflowIR):
            raise TypeError("workflow must use WorkflowIR")
        if not isinstance(bindings, Mapping):
            raise TypeError("bindings must be a mapping")
        if not isinstance(remote_clients, Mapping):
            raise TypeError("remote_clients must be a mapping")

        bound_stages: dict[str, Binding] = {}
        for stage_id, binding in sorted(bindings.items()):
            validate_name(stage_id, "binding stage id")
            if not isinstance(binding, (InlineBinding, RemoteBinding)):
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

        expected_endpoints = {
            binding.endpoint_id
            for binding in bound_stages.values()
            if isinstance(binding, RemoteBinding)
        }
        actual_endpoints = set(remote_clients)
        if actual_endpoints != expected_endpoints:
            raise WorkflowValidationError(
                "remote clients differ from workflow bindings; "
                f"missing={sorted(expected_endpoints - actual_endpoints)}, "
                f"extra={sorted(actual_endpoints - expected_endpoints)}"
            )
        clients = dict(remote_clients)
        for endpoint_id, client in clients.items():
            if not isinstance(client, RemoteStageInvoker):
                raise WorkflowValidationError(
                    f"remote client {endpoint_id!r} does not implement stage invocation"
                )

        for stage_id, contract in contracts.items():
            binding = bound_stages[stage_id]
            if isinstance(binding, InlineBinding) and binding.runner.contract != contract:
                raise WorkflowValidationError(
                    f"inline runner for stage {stage_id!r} "
                    "does not match its authored contract"
                )

        self._contracts = MappingProxyType(contracts)
        self._bindings = MappingProxyType(bound_stages)
        self._remote_clients = MappingProxyType(clients)

    @classmethod
    async def bind(
        cls,
        workflow: WorkflowIR,
        bindings: Mapping[str, Binding],
        *,
        runtime: Any = None,
    ) -> "StageDispatcher":
        """Resolve remote endpoints and bind all physical stage targets."""

        endpoint_ids = {
            binding.endpoint_id
            for binding in bindings.values()
            if isinstance(binding, RemoteBinding)
        }
        if endpoint_ids and runtime is None:
            raise WorkflowValidationError(
                "runtime is required to bind remote workflow stages"
            )

        clients: dict[str, RemoteStageInvoker] = {}
        for endpoint_id in sorted(endpoint_ids):
            endpoint = runtime.endpoint(endpoint_id)
            client = await endpoint.client()
            await client.wait_for_instances()
            clients[endpoint_id] = RemoteStageClient(client)
        return cls(workflow, bindings, clients)

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
        frozen_inputs = MappingProxyType(dict(inputs))
        if isinstance(binding, InlineBinding):
            result = await binding.runner.run(frozen_inputs, context)
        else:
            result = await self._remote_clients[binding.endpoint_id].run(
                stage_id, contract, frozen_inputs, context
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
        return dict(result)
