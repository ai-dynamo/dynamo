# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-bound workflow stage dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from dynamo.workflow.plan import ExecutionPlan, InlineBinding, RemoteBinding
from dynamo.workflow.runtime import StageContext, StageRunner, WorkflowExecutionError
from dynamo.workflow.types import StageContract, WorkflowValidationError


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
    """Validate and invoke stages through their compiled bindings."""

    def __init__(
        self,
        plan: ExecutionPlan,
        inline_runners: Mapping[str, StageRunner],
        remote_clients: Mapping[str, RemoteStageInvoker] = MappingProxyType({}),
    ) -> None:
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must use ExecutionPlan")
        if not isinstance(inline_runners, Mapping):
            raise TypeError("inline_runners must be a mapping")
        if not isinstance(remote_clients, Mapping):
            raise TypeError("remote_clients must be a mapping")

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
        expected_endpoints = {
            binding.endpoint_id
            for binding in plan.bindings.values()
            if isinstance(binding, RemoteBinding)
        }
        actual_endpoints = set(remote_clients)
        if actual_endpoints != expected_endpoints:
            raise WorkflowValidationError(
                "remote clients differ from execution plan; "
                f"missing={sorted(expected_endpoints - actual_endpoints)}, "
                f"extra={sorted(actual_endpoints - expected_endpoints)}"
            )
        clients = dict(remote_clients)
        for endpoint_id, client in clients.items():
            if not isinstance(client, RemoteStageInvoker):
                raise WorkflowValidationError(
                    f"remote client {endpoint_id!r} does not implement stage invocation"
                )

        for stage_id, contract in plan.stage_contracts.items():
            binding = plan.bindings[stage_id]
            if isinstance(binding, RemoteBinding):
                continue
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
        self._remote_clients = MappingProxyType(clients)

    @classmethod
    async def bind(
        cls,
        plan: ExecutionPlan,
        *,
        runtime: Any = None,
        inline_runners: Mapping[str, StageRunner] = MappingProxyType({}),
    ) -> "StageDispatcher":
        """Resolve remote endpoints once and bind all physical stage targets."""

        from dynamo.workflow.remote import RemoteStageClient

        endpoint_ids = {
            binding.endpoint_id
            for binding in plan.bindings.values()
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
        return cls(plan, inline_runners, clients)

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
        binding = self._plan.bindings[stage_id]
        frozen_inputs = MappingProxyType(dict(inputs))
        if isinstance(binding, InlineBinding):
            result = await self._inline_runners[binding.runner_key].run(
                frozen_inputs, context
            )
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
        outputs = dict(result)
        return outputs
