# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters between unary workflow stages and streaming Dynamo endpoints."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, AsyncIterator, Mapping, Optional, Protocol

from dynamo.experimental.workflow.nixl import (
    NixlTensorFanout,
    tensor_transfer_ref_from_dict,
)
from dynamo.experimental.workflow.perf import WORKFLOW_PERF_TRACE
from dynamo.experimental.workflow.plan import NIXL_CARRIER
from dynamo.experimental.workflow.runtime import (
    ReleasableTensorCarrier,
    StageContext,
    StageRunner,
    TensorCarrier,
    WorkflowExecutionError,
)
from dynamo.experimental.workflow.types import (
    StageContract,
    WorkflowValidationError,
    validate_name,
)

STAGE_REQUEST_SCHEMA = "dynamo.experimental.workflow.carrier_request"
STAGE_RESPONSE_SCHEMA = "dynamo.experimental.workflow.carrier_response"
STAGE_WIRE_VERSION = 1

logger = logging.getLogger(__name__)


def _check_keys(data: Mapping[str, Any], required: set[str]) -> None:
    actual = set(data)
    missing = required - actual
    unknown = actual - required
    if missing:
        raise WorkflowExecutionError(
            f"remote carrier envelope missing fields: {sorted(missing)}"
        )
    if unknown:
        raise WorkflowExecutionError(
            f"remote carrier envelope has unknown fields: {sorted(unknown)}"
        )


def _validate_header(data: Mapping[str, Any], schema: str) -> None:
    if data["schema"] != schema:
        raise WorkflowExecutionError(
            f"unsupported remote carrier schema {data['schema']!r}"
        )
    version = data["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise WorkflowExecutionError("remote carrier version must be an integer")
    if version != STAGE_WIRE_VERSION:
        raise WorkflowExecutionError(f"unsupported remote carrier version {version!r}")


def _freeze_carriers(
    carriers: Mapping[str, str], values: Mapping[str, Any]
) -> Mapping[str, str]:
    if not isinstance(carriers, Mapping):
        raise WorkflowExecutionError("remote carrier tags must be an object")
    unknown = set(carriers) - set(values)
    if unknown:
        raise WorkflowExecutionError(
            f"remote carrier tags reference unknown ports {sorted(unknown)}"
        )
    normalized: dict[str, str] = {}
    for name, carrier in carriers.items():
        validate_name(name, "remote carried port")
        if carrier != NIXL_CARRIER:
            raise WorkflowExecutionError(
                f"unsupported remote value carrier {carrier!r}"
            )
        normalized[name] = carrier
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class NixlCarriedValue:
    """Internal marker that keeps NIXL metadata distinct from ordinary mappings."""

    value: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.value, Mapping):
            raise WorkflowExecutionError("NIXL carried value must be an object")
        object.__setattr__(self, "value", MappingProxyType(dict(self.value)))


@dataclass(frozen=True)
class StageRequestEnvelope:
    """Internal carrier metadata sent to one remote stage."""

    inputs: Mapping[str, Any]
    input_carriers: Mapping[str, str]
    output_transfers: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, Mapping):
            raise WorkflowExecutionError("remote stage inputs must be an object")
        inputs = MappingProxyType(dict(self.inputs))
        carriers = _freeze_carriers(self.input_carriers, inputs)
        if not isinstance(self.output_transfers, Mapping):
            raise WorkflowExecutionError("remote output transfers must be an object")
        transfers: dict[str, tuple[str, ...]] = {}
        for output_name, transfer_ids in self.output_transfers.items():
            validate_name(output_name, "remote output transfer port")
            if not isinstance(transfer_ids, (list, tuple)):
                raise WorkflowExecutionError(
                    "remote output transfer ids must be a list or tuple"
                )
            normalized = tuple(transfer_ids)
            if any(
                not isinstance(transfer_id, str) or not transfer_id
                for transfer_id in normalized
            ):
                raise WorkflowExecutionError(
                    "remote output transfer ids must be non-empty strings"
                )
            if len(set(normalized)) != len(normalized):
                raise WorkflowExecutionError(
                    f"remote output {output_name!r} has duplicate transfer ids"
                )
            transfers[output_name] = normalized
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "input_carriers", carriers)
        object.__setattr__(self, "output_transfers", MappingProxyType(transfers))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STAGE_REQUEST_SCHEMA,
            "version": STAGE_WIRE_VERSION,
            "inputs": dict(self.inputs),
            "input_carriers": dict(self.input_carriers),
            "output_transfers": {
                name: list(transfer_ids)
                for name, transfer_ids in self.output_transfers.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageRequestEnvelope":
        if not isinstance(data, Mapping):
            raise WorkflowExecutionError("remote stage request must be an object")
        _check_keys(
            data,
            {
                "schema",
                "version",
                "inputs",
                "input_carriers",
                "output_transfers",
            },
        )
        _validate_header(data, STAGE_REQUEST_SCHEMA)
        return cls(
            inputs=data["inputs"],
            input_carriers=data["input_carriers"],
            output_transfers=data["output_transfers"],
        )


@dataclass(frozen=True)
class StageResponseEnvelope:
    """Internal carrier metadata returned by one remote stage."""

    outputs: Mapping[str, Any]
    output_carriers: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.outputs, Mapping):
            raise WorkflowExecutionError("remote stage outputs must be an object")
        outputs = MappingProxyType(dict(self.outputs))
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(
            self,
            "output_carriers",
            _freeze_carriers(self.output_carriers, outputs),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STAGE_RESPONSE_SCHEMA,
            "version": STAGE_WIRE_VERSION,
            "outputs": dict(self.outputs),
            "output_carriers": dict(self.output_carriers),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageResponseEnvelope":
        if not isinstance(data, Mapping):
            raise WorkflowExecutionError("remote stage response must be an object")
        _check_keys(
            data,
            {"schema", "version", "outputs", "output_carriers"},
        )
        _validate_header(data, STAGE_RESPONSE_SCHEMA)
        return cls(
            outputs=data["outputs"],
            output_carriers=data["output_carriers"],
        )


class _DynamoClient(Protocol):
    async def round_robin(
        self,
        request: Mapping[str, Any],
        *,
        annotated: bool,
        context: Any = None,
    ) -> AsyncIterator[Any]:
        ...


class RemoteStageClient:
    """Adapt a unary workflow call to Dynamo's streaming endpoint API."""

    def __init__(self, client: _DynamoClient) -> None:
        self._client = client

    async def run(
        self,
        stage_id: str,
        contract: StageContract,
        inputs: Mapping[str, Any],
        context: StageContext,
        output_transfers: Mapping[str, tuple[str, ...]],
        *,
        request_context: Any = None,
    ) -> Mapping[str, Any]:
        started_ns = time.perf_counter_ns()
        stage_label = f"remote stage {stage_id!r} with contract {contract.id!r}"
        wire_inputs: dict[str, Any] = {}
        input_carriers: dict[str, str] = {}
        for name, value in inputs.items():
            if isinstance(value, NixlCarriedValue):
                wire_inputs[name] = dict(value.value)
                input_carriers[name] = NIXL_CARRIER
            else:
                wire_inputs[name] = value
        request = StageRequestEnvelope(
            inputs=wire_inputs,
            input_carriers=input_carriers,
            output_transfers=output_transfers,
        )

        transport_context = None
        if request_context is not None:
            detach = getattr(request_context, "detached", None)
            if not callable(detach):
                raise WorkflowExecutionError(
                    "request context cannot create a detached child context"
                )
            transport_context = detach(f"{context.attempt_id}:{context.stage_id}")

        try:
            stream = await self._client.round_robin(
                request.to_dict(), annotated=False, context=transport_context
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise WorkflowExecutionError(
                f"{stage_label} request failed at the transport boundary"
            ) from error

        try:
            try:
                response = await stream.__anext__()
            except StopAsyncIteration as error:
                raise WorkflowExecutionError(
                    f"{stage_label} returned no response mapping"
                ) from error
            try:
                await stream.__anext__()
            except StopAsyncIteration:
                pass
            else:
                raise WorkflowExecutionError(
                    f"{stage_label} returned multiple response mappings"
                )
            envelope = StageResponseEnvelope.from_dict(response)
        except BaseException as error:
            if transport_context is not None:
                transport_context.stop_generating()
            close = getattr(stream, "aclose", None)
            if callable(close):
                await close()
            if isinstance(error, (asyncio.CancelledError, WorkflowExecutionError)):
                raise
            if isinstance(error, Exception):
                raise WorkflowExecutionError(
                    f"{stage_label} response failed at the transport boundary"
                ) from error
            raise
        WORKFLOW_PERF_TRACE.emit(
            logger,
            "workflow.remote_call",
            context.attempt_id,
            elapsed_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
            stage=stage_id,
        )
        outputs: dict[str, Any] = dict(envelope.outputs)
        for name, carrier in envelope.output_carriers.items():
            if carrier == NIXL_CARRIER:
                outputs[name] = NixlCarriedValue(envelope.outputs[name])
        return outputs


class RemoteStageServer:
    """Adapt a unary ``StageRunner`` to Dynamo's streaming endpoint API."""

    def __init__(
        self,
        stage_id: str,
        runner: StageRunner,
        tensor_carrier: Optional[TensorCarrier] = None,
    ) -> None:
        validate_name(stage_id, "remote stage id")
        if not isinstance(runner, StageRunner):
            raise WorkflowValidationError("remote runner must implement StageRunner")
        if tensor_carrier is not None and not isinstance(tensor_carrier, TensorCarrier):
            raise WorkflowValidationError(
                "remote tensor_carrier must implement TensorCarrier"
            )
        self._stage_id = stage_id
        self._runner = runner
        self._tensor_carrier = tensor_carrier

    async def generate(
        self, request: Mapping[str, Any], context: Any = None
    ) -> AsyncIterator[dict[str, Any]]:
        envelope = StageRequestEnvelope.from_dict(request)
        expected_inputs = set(self._runner.contract.inputs)
        actual_inputs = set(envelope.inputs)
        if actual_inputs != expected_inputs:
            raise WorkflowExecutionError(
                f"remote stage {self._stage_id!r} inputs differ from its contract; "
                f"missing={sorted(expected_inputs - actual_inputs)}, "
                f"extra={sorted(actual_inputs - expected_inputs)}"
            )
        unknown_transfer_outputs = set(envelope.output_transfers) - set(
            self._runner.contract.outputs
        )
        if unknown_transfer_outputs:
            raise WorkflowExecutionError(
                f"remote stage {self._stage_id!r} has transfer requests for "
                f"unknown outputs {sorted(unknown_transfer_outputs)}"
            )

        request_id = uuid.uuid4().hex
        if context is not None:
            get_request_id = getattr(context, "id", None)
            if callable(get_request_id):
                candidate = get_request_id()
                if isinstance(candidate, str) and candidate:
                    request_id = candidate
        stage_context = StageContext(
            workflow_name=None,
            stage_id=self._stage_id,
            attempt_id=request_id,
        )

        async def invoke() -> tuple[dict[str, Any], dict[str, str]]:
            started_ns = time.perf_counter_ns()
            imported_tensors: list[Any] = []
            try:
                runner_inputs = dict(envelope.inputs)
                for name in envelope.input_carriers:
                    if self._tensor_carrier is None:
                        raise WorkflowExecutionError(
                            f"remote stage {self._stage_id!r} has no NIXL "
                            "tensor carrier"
                        )
                    imported = await self._tensor_carrier.import_tensor(
                        envelope.inputs[name]
                    )
                    imported_tensors.append(imported)
                    runner_inputs[name] = imported
                inputs_ready_ns = time.perf_counter_ns()
                result = await self._runner.run(
                    MappingProxyType(runner_inputs), stage_context
                )
                runner_finished_ns = time.perf_counter_ns()
                if not isinstance(result, Mapping):
                    raise WorkflowExecutionError(
                        f"remote stage {self._stage_id!r} returned a non-mapping result"
                    )
                expected_outputs = set(self._runner.contract.outputs)
                actual_outputs = set(result)
                if actual_outputs != expected_outputs:
                    raise WorkflowExecutionError(
                        f"remote stage {self._stage_id!r} outputs differ from its "
                        f"contract; missing={sorted(expected_outputs - actual_outputs)}, "
                        f"extra={sorted(actual_outputs - expected_outputs)}"
                    )
                wire_outputs = dict(result)
                output_carriers: dict[str, str] = {}
                for name, value in result.items():
                    if (
                        self._tensor_carrier is None
                        or not self._tensor_carrier.can_export(value)
                    ):
                        continue
                    transfer_ids = envelope.output_transfers.get(name, ())
                    if not transfer_ids:
                        raise WorkflowExecutionError(
                            f"remote tensor output {name!r} has no NIXL "
                            "consumer transfers"
                        )
                    references = await self._tensor_carrier.export_tensor_fanout(
                        value, transfer_ids
                    )
                    if set(references) != set(transfer_ids):
                        raise WorkflowExecutionError(
                            f"remote tensor output {name!r} NIXL references differ "
                            "from requested consumer transfers"
                        )
                    wire_outputs[name] = NixlTensorFanout(
                        {
                            transfer_id: tensor_transfer_ref_from_dict(
                                reference, transfer_id=transfer_id
                            )
                            for transfer_id, reference in references.items()
                        }
                    ).to_dict()
                    output_carriers[name] = NIXL_CARRIER
                WORKFLOW_PERF_TRACE.emit(
                    logger,
                    "workflow.remote_server",
                    request_id,
                    elapsed_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
                    export_ms=(time.perf_counter_ns() - runner_finished_ns) / 1_000_000,
                    import_ms=(inputs_ready_ns - started_ns) / 1_000_000,
                    runner_ms=(runner_finished_ns - inputs_ready_ns) / 1_000_000,
                    stage=self._stage_id,
                    tensor_transfers=sum(
                        len(transfer_ids)
                        for transfer_ids in envelope.output_transfers.values()
                    ),
                )
                return wire_outputs, output_carriers
            finally:
                if isinstance(self._tensor_carrier, ReleasableTensorCarrier):
                    for tensor in reversed(imported_tensors):
                        self._tensor_carrier.release_imported_tensor(tensor)

        invocation = asyncio.create_task(invoke(), name=f"workflow-remote:{request_id}")
        transport_task: asyncio.Future[Any] | None = None
        if context is not None:
            wait_for_stop = getattr(context, "async_killed_or_stopped", None)
            if callable(wait_for_stop):
                transport_task = asyncio.ensure_future(wait_for_stop())
        try:
            if transport_task is None:
                wire_outputs, output_carriers = await invocation
            else:
                done, _ = await asyncio.wait(
                    {invocation, transport_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if invocation in done:
                    wire_outputs, output_carriers = invocation.result()
                else:
                    raise asyncio.CancelledError()
        except BaseException:
            if not invocation.done():
                invocation.cancel()
            await asyncio.gather(invocation, return_exceptions=True)
            raise
        finally:
            if transport_task is not None and not transport_task.done():
                transport_task.cancel()
                await asyncio.gather(transport_task, return_exceptions=True)

        if context is not None and (
            bool(getattr(context, "is_stopped", lambda: False)())
            or bool(getattr(context, "is_killed", lambda: False)())
        ):
            raise asyncio.CancelledError()
        yield StageResponseEnvelope(
            outputs=wire_outputs,
            output_carriers=output_carriers,
        ).to_dict()
