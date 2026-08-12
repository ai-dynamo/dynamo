# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-bound workflow stage dispatch."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Any, Protocol, runtime_checkable

from dynamo.workflow.plan import (
    NIXL_CARRIER,
    EdgePlan,
    ExecutionPlan,
    InlineBinding,
    RemoteBinding,
)
from dynamo.workflow.remote import RemoteStageClient
from dynamo.workflow.runtime import (
    StageContext,
    StageRunner,
    TensorCarrier,
    WorkflowExecutionError,
    _validate_value,
)
from dynamo.workflow.types import StageContract, ValueRef, WorkflowValidationError


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
        remote_clients: Mapping[str, RemoteStageInvoker] = MappingProxyType({}),
        tensor_carrier: TensorCarrier | None = None,
    ) -> None:
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must use ExecutionPlan")
        if not isinstance(inline_runners, Mapping):
            raise TypeError("inline_runners must be a mapping")
        if not isinstance(remote_clients, Mapping):
            raise TypeError("remote_clients must be a mapping")
        if tensor_carrier is not None and not isinstance(tensor_carrier, TensorCarrier):
            raise TypeError("tensor_carrier must implement TensorCarrier")

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

        local_nixl_edges = [
            edge
            for edge in plan.edges
            if edge.carrier == NIXL_CARRIER
            and self._edge_touches_local_process(plan, edge)
        ]
        if local_nixl_edges and tensor_carrier is None:
            raise WorkflowValidationError(
                "execution plan has NIXL edges touching local stages but no "
                "tensor_carrier was bound"
            )

        self._plan = plan
        self._inline_runners = MappingProxyType(runners)
        self._remote_clients = MappingProxyType(clients)
        self._tensor_carrier = tensor_carrier
        self._edges_by_target = MappingProxyType(
            {(edge.target_stage, edge.target_port): edge for edge in plan.edges}
        )

        output_transfers: dict[str, dict[str, list[str]]] = {}
        source_transfers: dict[ValueRef, list[str]] = {}
        for edge in plan.edges:
            if edge.carrier != NIXL_CARRIER:
                continue
            source_transfers.setdefault(edge.source, []).append(edge.transfer_id)
            source_stage_id = edge.source.stage_id
            if source_stage_id is None:
                continue
            source_output_name = edge.source.output_name
            assert source_output_name is not None
            stage_outputs = output_transfers.setdefault(source_stage_id, {})
            stage_outputs.setdefault(source_output_name, []).append(edge.transfer_id)
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
        self._source_transfers = MappingProxyType(
            {
                source: tuple(sorted(transfer_ids))
                for source, transfer_ids in source_transfers.items()
            }
        )

    @staticmethod
    def _edge_touches_local_process(plan: ExecutionPlan, edge: EdgePlan) -> bool:
        target_is_inline = isinstance(plan.bindings[edge.target_stage], InlineBinding)
        if edge.source.input_name is not None:
            source_is_inline = True
        else:
            source_stage_id = edge.source.stage_id
            assert source_stage_id is not None
            source_is_inline = isinstance(
                plan.bindings[source_stage_id], InlineBinding
            )
        return source_is_inline or target_is_inline

    @classmethod
    async def bind(
        cls,
        plan: ExecutionPlan,
        *,
        runtime: Any = None,
        inline_runners: Mapping[str, StageRunner] = MappingProxyType({}),
        tensor_carrier: TensorCarrier | None = None,
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

        clients: dict[str, RemoteStageInvoker] = {}
        for endpoint_id in sorted(endpoint_ids):
            endpoint = runtime.endpoint(endpoint_id)
            client = await endpoint.client()
            await client.wait_for_instances()
            clients[endpoint_id] = RemoteStageClient(client)
        return cls(plan, inline_runners, clients, tensor_carrier)

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
            value = inputs[input_name]
            location = f"stage {stage_id!r} input {input_name!r}"
            if isinstance(binding, RemoteBinding) and spec.type == "tensor":
                from dynamo.workflow.nixl import NixlTensorRef

                reference = NixlTensorRef.from_dict(value)
                _validate_value(
                    spec,
                    SimpleNamespace(dtype=reference.dtype, shape=reference.shape),
                    location,
                )
            else:
                _validate_value(spec, value, location)

        frozen_inputs = MappingProxyType(dict(inputs))
        if isinstance(binding, InlineBinding):
            result = await self._inline_runners[binding.runner_key].run(
                frozen_inputs, context
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
        outputs = dict(result)
        for output_name, spec in contract.outputs.items():
            if isinstance(binding, RemoteBinding) and spec.type == "tensor":
                from dynamo.workflow.nixl import NixlTensorFanout

                fanout = NixlTensorFanout.from_dict(outputs[output_name])
                expected_transfers = set(
                    self._output_transfers.get(stage_id, {}).get(output_name, ())
                )
                if set(fanout.transfers) != expected_transfers:
                    raise WorkflowExecutionError(
                        f"remote stage {stage_id!r} tensor output "
                        f"{output_name!r} transfers differ from execution plan"
                    )
            else:
                _validate_value(
                    spec,
                    outputs[output_name],
                    f"stage {stage_id!r} output {output_name!r}",
                )
        return outputs

    async def resolve_edge(
        self,
        reference: ValueRef,
        target_stage: str,
        target_port: str,
        value: Any,
        tensor_exports: dict[
            ValueRef, asyncio.Task[Mapping[str, Mapping[str, Any]]]
        ],
    ) -> Any:
        """Materialize one compiled graph edge for its target placement."""

        edge = self._edges_by_target[(target_stage, target_port)]
        if edge.source != reference:
            raise WorkflowExecutionError(
                f"edge targeting {target_stage!r}.{target_port!r} changed after compile"
            )
        if edge.carrier != NIXL_CARRIER:
            return value

        source_is_remote = reference.stage_id is not None and isinstance(
            self._plan.bindings[reference.stage_id], RemoteBinding
        )
        wire_reference: Mapping[str, Any]
        if source_is_remote:
            from dynamo.workflow.nixl import NixlTensorFanout

            wire_reference = (
                NixlTensorFanout.from_dict(value)
                .for_transfer(edge.transfer_id)
                .to_dict()
            )
        else:
            if self._tensor_carrier is None:
                raise WorkflowExecutionError(
                    f"NIXL edge {edge.transfer_id!r} has no bound tensor carrier"
                )
            export_task = tensor_exports.get(reference)
            if export_task is None:
                export_task = asyncio.create_task(
                    self._tensor_carrier.export_tensor_fanout(
                        value, self._source_transfers[reference]
                    ),
                    name=f"workflow-nixl-export:{edge.transfer_id}",
                )
                tensor_exports[reference] = export_task
            references = await asyncio.shield(export_task)
            try:
                wire_reference = references[edge.transfer_id]
            except KeyError as error:
                raise WorkflowExecutionError(
                    f"NIXL export has no transfer {edge.transfer_id!r}"
                ) from error

        if isinstance(self._plan.bindings[target_stage], RemoteBinding):
            return wire_reference
        if self._tensor_carrier is None:
            raise WorkflowExecutionError(
                f"NIXL edge {edge.transfer_id!r} has no bound tensor carrier"
            )
        return await self._tensor_carrier.import_tensor(wire_reference)
