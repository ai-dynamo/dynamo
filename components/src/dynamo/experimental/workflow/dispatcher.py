# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-bound workflow stage dispatch."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from dynamo.experimental.workflow.generate import GenerateEndpointInvoker
from dynamo.experimental.workflow.nixl import NixlTensorFanout, tensor_transfer_ref_from_dict
from dynamo.experimental.workflow.plan import (
    NIXL_CARRIER,
    ExecutionPlan,
    GenerateEndpointBinding,
    InlineBinding,
    RemoteBinding,
)
from dynamo.experimental.workflow.remote import NixlCarriedValue, RemoteStageClient
from dynamo.experimental.workflow.runtime import StageContext, StageRunner, WorkflowExecutionError
from dynamo.experimental.workflow.types import StageContract, ValueRef, WorkflowValidationError

logger = logging.getLogger(__name__)


@runtime_checkable
class RemoteStageInvoker(Protocol):
    """Internal transport boundary used by the dispatcher."""

    async def run(
        self,
        stage_id: str,
        contract: StageContract,
        inputs: Mapping[str, Any],
        context: StageContext,
        output_transfers: Mapping[str, tuple[str, ...]],
    ) -> Mapping[str, Any]:
        ...


class StageDispatcher:
    """Validate and invoke stages through their compiled bindings."""

    def __init__(
        self,
        plan: ExecutionPlan,
        inline_runners: Mapping[str, StageRunner],
        remote_clients: Mapping[str, Any] = MappingProxyType({}),
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

        runners = dict(inline_runners)
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

        output_transfers: dict[str, dict[str, list[str]]] = {}
        for target in plan.workflow.stages:
            target_binding = plan.bindings[target.id]
            if not self._nixl_remote(target_binding):
                continue
            for target_port, source in target.inputs.items():
                if source.stage_id is None:
                    continue
                source_binding = plan.bindings[source.stage_id]
                if not self._nixl_remote(source_binding):
                    continue
                source_output = source.output_name
                assert source_output is not None
                outputs = output_transfers.setdefault(source.stage_id, {})
                outputs.setdefault(source_output, []).append(
                    f"{target.id}.{target_port}"
                )

        self._plan = plan
        self._inline_runners = MappingProxyType(runners)
        self._remote_clients = MappingProxyType(dict(remote_clients))
        self._output_transfers = MappingProxyType(
            {
                stage_id: MappingProxyType(
                    {
                        output_name: tuple(sorted(transfer_ids))
                        for output_name, transfer_ids in outputs.items()
                    }
                )
                for stage_id, outputs in output_transfers.items()
            }
        )

    @staticmethod
    def _nixl_remote(binding: Any) -> bool:
        return (
            isinstance(binding, RemoteBinding)
            and binding.tensor_carrier == NIXL_CARRIER
        )

    @classmethod
    async def bind(
        cls,
        plan: ExecutionPlan,
        *,
        runtime: Any = None,
        inline_runners: Mapping[str, StageRunner] = MappingProxyType({}),
    ) -> "StageDispatcher":
        """Resolve remote endpoints once and bind all physical stage targets."""

        endpoint_ids = {
            binding.endpoint_id
            for binding in plan.bindings.values()
            if isinstance(binding, RemoteBinding)
        }
        if endpoint_ids and runtime is None:
            raise WorkflowValidationError(
                "runtime is required to bind remote workflow stages"
            )

        clients: dict[str, Any] = {}
        for endpoint_id in sorted(endpoint_ids):
            bindings = [
                binding
                for binding in plan.bindings.values()
                if isinstance(binding, RemoteBinding)
                and binding.endpoint_id == endpoint_id
            ]
            protocols = {
                GenerateEndpointBinding
                if isinstance(binding, GenerateEndpointBinding)
                else RemoteBinding
                for binding in bindings
            }
            if len(protocols) != 1:
                raise WorkflowValidationError(
                    f"remote endpoint {endpoint_id!r} cannot mix stage protocols"
                )
            logger.info("Binding workflow endpoint %r", endpoint_id)
            endpoint = runtime.endpoint(endpoint_id)
            client = await endpoint.client()
            logger.info("Waiting for workflow endpoint %r", endpoint_id)
            await client.wait_for_instances()
            logger.info("Workflow endpoint %r is available", endpoint_id)
            clients[endpoint_id] = (
                GenerateEndpointInvoker(client)
                if protocols == {GenerateEndpointBinding}
                else RemoteStageClient(client)
            )
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
        for input_name, spec in contract.inputs.items():
            value_spec = _require_value_spec(
                spec, f"stage {stage_id!r} input {input_name!r}"
            )
            value = inputs[input_name]
            location = f"stage {stage_id!r} input {input_name!r}"
            if isinstance(binding, RemoteBinding) and value_spec.type == "tensor":
                reference = tensor_transfer_ref_from_dict(value)
                _validate_value(
                    value_spec,
                    SimpleNamespace(dtype=reference.dtype, shape=reference.shape),
                    location,
                )
            else:
                _validate_value(value_spec, value, location)

        binding = self._plan.bindings[stage_id]
        frozen_inputs = MappingProxyType(dict(inputs))
        if isinstance(binding, InlineBinding):
            result = await self._inline_runners[binding.runner_key].run(
                frozen_inputs, context
            )
        elif isinstance(binding, GenerateEndpointBinding):
            result = await self._remote_clients[binding.endpoint_id].run(
                stage_id, contract, frozen_inputs, context
            )
        else:
            result = await self._remote_clients[binding.endpoint_id].run(
                stage_id,
                contract,
                frozen_inputs,
                context,
                self._output_transfers.get(stage_id, MappingProxyType({})),
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

    async def resolve_edge(
        self,
        reference: ValueRef,
        target_stage: str,
        target_port: str,
        value: Any,
    ) -> Any:
        """Select a consumer-specific NIXL reference when a value was carried."""

        if not isinstance(value, NixlCarriedValue):
            return value
        transfer_id = f"{target_stage}.{target_port}"
        source_stage = reference.stage_id
        if source_stage is None or not self._nixl_remote(
            self._plan.bindings[source_stage]
        ):
            raise WorkflowExecutionError(
                f"NIXL edge {transfer_id!r} must originate from a NIXL remote stage"
            )
        if not self._nixl_remote(self._plan.bindings[target_stage]):
            raise WorkflowExecutionError(
                f"NIXL edge {transfer_id!r} must target a NIXL remote stage"
            )
        reference_value = NixlTensorFanout.from_dict(value.value).for_transfer(
            transfer_id
        )
        return NixlCarriedValue(reference_value.to_dict())

    def resolve_workflow_output(self, output_name: str, value: Any) -> Any:
        """Reject carried tensors from the current unary workflow boundary."""

        if isinstance(value, NixlCarriedValue):
            raise WorkflowExecutionError(
                f"workflow output {output_name!r} cannot expose a NIXL tensor"
            )
        return value
