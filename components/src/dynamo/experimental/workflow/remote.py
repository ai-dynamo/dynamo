# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters between unary workflow stages and streaming Dynamo endpoints."""

from __future__ import annotations

import asyncio
import uuid
from types import MappingProxyType
from typing import Any, AsyncIterator, Awaitable, Mapping, Optional, Protocol, TypeVar

from dynamo.experimental.workflow.nixl import NixlTensorFanout, NixlTensorRef
from dynamo.experimental.workflow.plan import INLINE_VALUE_TYPES
from dynamo.experimental.workflow.runtime import (
    StageContext,
    StageRunner,
    TensorCarrier,
    WorkflowExecutionError,
    _validate_value,
)
from dynamo.experimental.workflow.types import (
    StageContract,
    WorkflowValidationError,
    _require_value_spec,
    validate_name,
)

STAGE_REQUEST_SCHEMA = "dynamo.experimental.workflow.stage_request"
STAGE_RESPONSE_SCHEMA = "dynamo.experimental.workflow.stage_response"
STAGE_WIRE_VERSION = 2

_T = TypeVar("_T")


def _check_keys(data: Mapping[str, Any], required: set[str]) -> None:
    keys = set(data)
    missing = required - keys
    unknown = keys - required
    if missing:
        raise WorkflowExecutionError(
            f"remote envelope missing fields: {sorted(missing)}"
        )
    if unknown:
        raise WorkflowExecutionError(
            f"remote envelope has unknown fields: {sorted(unknown)}"
        )


def _validate_attempt_id(attempt_id: str) -> None:
    if not isinstance(attempt_id, str) or not attempt_id:
        raise WorkflowExecutionError("remote attempt id must be a non-empty string")
    try:
        attempt_id.encode("utf-8")
    except UnicodeEncodeError as error:
        raise WorkflowExecutionError("remote attempt id must be valid UTF-8") from error


async def _run_with_stage_lifecycle(
    operation: Awaitable[_T],
    context: StageContext,
    transport_context: Any,
) -> _T:
    """Bound all remote-stage work by its deadline and transport lifetime."""

    execution = asyncio.ensure_future(operation)
    wait_for_transport_stop = (
        None
        if transport_context is None
        else getattr(transport_context, "async_killed_or_stopped", None)
    )
    transport_stopped = (
        asyncio.ensure_future(wait_for_transport_stop())
        if callable(wait_for_transport_stop)
        else None
    )
    waiters: set[asyncio.Future[Any]] = {execution}
    if transport_stopped is not None:
        waiters.add(transport_stopped)

    try:
        done, _ = await asyncio.wait(
            waiters,
            timeout=context.remaining_time(),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            context._cancelled.set()
            raise asyncio.TimeoutError
        if execution in done:
            return await execution
        context._cancelled.set()
        raise asyncio.CancelledError
    except BaseException:
        context._cancelled.set()
        raise
    finally:
        for task in waiters:
            if not task.done():
                task.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)


@dataclass(frozen=True)
class StageRequestEnvelope:
    """Versioned request sent from the orchestrator to one stage endpoint."""

    workflow_name: str
    stage_id: str
    contract_id: str
    attempt_id: str
    invocation_id: str
    timeout_seconds: Optional[float]
    inputs: Mapping[str, Any]
    output_transfers: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        validate_name(self.workflow_name, "remote workflow name")
        validate_name(self.stage_id, "remote stage id")
        validate_name(self.contract_id, "remote contract id")
        _validate_attempt_id(self.attempt_id)
        _validate_attempt_id(self.invocation_id)
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise WorkflowExecutionError(
                "remote timeout_seconds must be a finite positive number"
            )
        if not isinstance(self.inputs, Mapping):
            raise WorkflowExecutionError("remote stage inputs must be an object")
        if not isinstance(self.output_transfers, Mapping):
            raise WorkflowExecutionError("remote output_transfers must be an object")
        output_transfers: dict[str, tuple[str, ...]] = {}
        for output_name, transfer_ids in self.output_transfers.items():
            validate_name(output_name, "remote output transfer port")
            if not isinstance(transfer_ids, (list, tuple)) or any(
                not isinstance(transfer_id, str) or not transfer_id
                for transfer_id in transfer_ids
            ):
                raise WorkflowExecutionError(
                    "remote output transfer ids must be non-empty strings"
                )
            if len(set(transfer_ids)) != len(transfer_ids):
                raise WorkflowExecutionError(
                    f"remote output {output_name!r} has duplicate transfer ids"
                )
            output_transfers[output_name] = tuple(transfer_ids)
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "output_transfers", MappingProxyType(output_transfers))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STAGE_REQUEST_SCHEMA,
            "version": STAGE_WIRE_VERSION,
            "workflow": self.workflow_name,
            "stage": self.stage_id,
            "contract": self.contract_id,
            "attempt": self.attempt_id,
            "invocation": self.invocation_id,
            "timeout_seconds": self.timeout_seconds,
            "inputs": dict(self.inputs),
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
                "workflow",
                "stage",
                "contract",
                "attempt",
                "invocation",
                "timeout_seconds",
                "inputs",
                "output_transfers",
            },
        )
        if data["schema"] != STAGE_REQUEST_SCHEMA:
            raise WorkflowExecutionError(
                f"unsupported remote request schema {data['schema']!r}"
            )
        if (
            not isinstance(data["version"], int)
            or isinstance(data["version"], bool)
            or data["version"] != STAGE_WIRE_VERSION
        ):
            raise WorkflowExecutionError(
                f"unsupported remote request version {data['version']!r}"
            )
        return cls(
            workflow_name=data["workflow"],
            stage_id=data["stage"],
            contract_id=data["contract"],
            attempt_id=data["attempt"],
            invocation_id=data["invocation"],
            timeout_seconds=data["timeout_seconds"],
            inputs=data["inputs"],
            output_transfers=data["output_transfers"],
        )


@dataclass(frozen=True)
class StageResponseEnvelope:
    """Versioned terminal response returned by one stage endpoint."""

    stage_id: str
    contract_id: str
    attempt_id: str
    invocation_id: str
    outputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        validate_name(self.stage_id, "remote stage id")
        validate_name(self.contract_id, "remote contract id")
        _validate_attempt_id(self.attempt_id)
        _validate_attempt_id(self.invocation_id)
        if not isinstance(self.outputs, Mapping):
            raise WorkflowExecutionError("remote stage outputs must be an object")
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STAGE_RESPONSE_SCHEMA,
            "version": STAGE_WIRE_VERSION,
            "stage": self.stage_id,
            "contract": self.contract_id,
            "attempt": self.attempt_id,
            "invocation": self.invocation_id,
            "outputs": dict(self.outputs),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageResponseEnvelope":
        if not isinstance(data, Mapping):
            raise WorkflowExecutionError("remote stage response must be an object")
        _check_keys(
            data,
            {
                "schema",
                "version",
                "stage",
                "contract",
                "attempt",
                "invocation",
                "outputs",
            },
        )
        if data["schema"] != STAGE_RESPONSE_SCHEMA:
            raise WorkflowExecutionError(
                f"unsupported remote response schema {data['schema']!r}"
            )
        if (
            not isinstance(data["version"], int)
            or isinstance(data["version"], bool)
            or data["version"] != STAGE_WIRE_VERSION
        ):
            raise WorkflowExecutionError(
                f"unsupported remote response version {data['version']!r}"
            )
        return cls(
            stage_id=data["stage"],
            contract_id=data["contract"],
            attempt_id=data["attempt"],
            invocation_id=data["invocation"],
            outputs=data["outputs"],
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
    """Adapt one unary workflow stage call to Dynamo's streaming client API.

    ``StageDispatcher`` owns this adapter for remote bindings. It sends the
    stage's contract-shaped input mapping directly and requires exactly one
    output mapping from the endpoint stream, keeping transport iteration opaque
    to ``StageRunner`` implementations.
    """

    def __init__(self, client: _DynamoClient) -> None:
        self._client = client

    async def run(
        self,
        stage_id: str,
        contract: StageContract,
        inputs: Mapping[str, Any],
        context: StageContext,
        output_transfers: Mapping[str, tuple[str, ...]],
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        request = StageRequestEnvelope(
            workflow_name=context.workflow_name,
            stage_id=stage_id,
            contract_id=contract.id,
            attempt_id=context.attempt_id,
            invocation_id=context.invocation_id,
            timeout_seconds=context.remaining_time(),
            inputs=inputs,
            output_transfers=output_transfers,
        )
        transport_context = None
        if context.request_context is not None:
            detach = getattr(context.request_context, "detached", None)
            if not callable(detach):
                raise WorkflowExecutionError(
                    "request context cannot create a detached child context"
                )
            transport_context = detach(context.invocation_id)

        try:
            stream = await self._client.round_robin(
                dict(inputs), annotated=False, context=transport_context
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
            if not isinstance(response, Mapping):
                raise WorkflowExecutionError(
                    f"{stage_label} returned a non-mapping response"
                )
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
        return dict(response)


class RemoteStageServer:
    """Adapt a unary ``StageRunner`` to Dynamo's streaming endpoint API.

    Dynamo ``serve_endpoint`` handlers must return an async iterator. This
    adapter validates the direct input mapping, awaits ``StageRunner.run`` once,
    validates its output mapping, and yields that mapping as one response frame
    so stage implementations never write transport-level ``yield``.
    """

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
        unsupported_ports = sorted(
            f"{direction}.{name}:{value_spec.type}"
            for direction, ports in (
                ("inputs", runner.contract.inputs),
                ("outputs", runner.contract.outputs),
            )
            for name, spec in ports.items()
            if (
                value_spec := _require_value_spec(
                    spec, f"remote stage {stage_id!r} {direction}.{name}"
                )
            ).type
            not in INLINE_VALUE_TYPES
            and not (value_spec.type == "tensor" and tensor_carrier is not None)
        )
        if unsupported_ports:
            raise WorkflowValidationError(
                "inline remote server does not support ports " f"{unsupported_ports}"
            )
        self._stage_id = stage_id
        self._runner = runner
        self._tensor_carrier = tensor_carrier

    async def generate(
        self, request: Mapping[str, Any], context: Any = None
    ) -> AsyncIterator[dict[str, Any]]:
        if not isinstance(request, Mapping):
            raise WorkflowExecutionError("remote stage request must be a mapping")
        inputs = MappingProxyType(dict(request))
        transport_context = context

        loop = asyncio.get_running_loop()
        deadline = (
            None
            if envelope.timeout_seconds is None
            else loop.time() + envelope.timeout_seconds
        )
        cancelled = asyncio.Event()
        stage_context = StageContext(
            workflow_name=None,
            stage_id=self._stage_id,
            attempt_id=request_id,
            invocation_id=request_id,
            deadline=None,
            _cancelled=cancelled,
            request_context=transport_context,
        )

        async def invoke() -> dict[str, Any]:
            expected_inputs = set(self._runner.contract.inputs)
            actual_inputs = set(envelope.inputs)
            if actual_inputs != expected_inputs:
                raise WorkflowExecutionError(
                    f"remote stage {self._stage_id!r} inputs differ from its contract; "
                    f"missing={sorted(expected_inputs - actual_inputs)}, "
                    f"extra={sorted(actual_inputs - expected_inputs)}"
                )

            runner_inputs = dict(envelope.inputs)
            for name, spec in self._runner.contract.inputs.items():
                if spec.type == "tensor":
                    if self._tensor_carrier is None:
                        raise WorkflowExecutionError(
                            f"remote stage {self._stage_id!r} has no NIXL tensor carrier"
                        )
                    runner_inputs[name] = await self._tensor_carrier.import_tensor(
                        envelope.inputs[name]
                    )
                _validate_value(
                    spec,
                    runner_inputs[name],
                    f"remote stage {self._stage_id!r} input {name!r}",
                )

            unknown_transfer_outputs = set(envelope.output_transfers) - set(
                self._runner.contract.outputs
            )
            if unknown_transfer_outputs:
                raise WorkflowExecutionError(
                    f"remote stage {self._stage_id!r} has transfer requests for "
                    f"unknown outputs {sorted(unknown_transfer_outputs)}"
                )

            result = await self._runner.run(MappingProxyType(runner_inputs), context)
            if not isinstance(result, Mapping):
                raise WorkflowExecutionError(
                    f"remote stage {self._stage_id!r} returned a non-mapping result"
                )
            expected_outputs = set(self._runner.contract.outputs)
            actual_outputs = set(result)
            if actual_outputs != expected_outputs:
                raise WorkflowExecutionError(
                    f"remote stage {self._stage_id!r} outputs differ from its contract; "
                    f"missing={sorted(expected_outputs - actual_outputs)}, "
                    f"extra={sorted(actual_outputs - expected_outputs)}"
                )
            outputs = dict(result)
            for name, spec in self._runner.contract.outputs.items():
                _validate_value(
                    spec,
                    outputs[name],
                    f"remote stage {self._stage_id!r} output {name!r}",
                )
            wire_outputs: dict[str, Any] = dict(outputs)
            for name, spec in self._runner.contract.outputs.items():
                transfer_ids = envelope.output_transfers.get(name, ())
                if spec.type != "tensor":
                    if transfer_ids:
                        raise WorkflowExecutionError(
                            f"remote stage {self._stage_id!r} received NIXL transfers "
                            f"for non-tensor output {name!r}"
                        )
                    continue
                if self._tensor_carrier is None:
                    raise WorkflowExecutionError(
                        f"remote stage {self._stage_id!r} has no NIXL tensor carrier"
                    )
                if not transfer_ids:
                    raise WorkflowExecutionError(
                        f"remote tensor output {name!r} has no consumer transfers"
                    )
                transfers = {}
                references = await self._tensor_carrier.export_tensor_fanout(
                    outputs[name], transfer_ids
                )
                if set(references) != set(transfer_ids):
                    raise WorkflowExecutionError(
                        f"remote tensor output {name!r} NIXL references differ "
                        "from requested consumer transfers"
                    )
                for transfer_id, reference in references.items():
                    transfers[transfer_id] = NixlTensorRef.from_dict(reference)
                wire_outputs[name] = NixlTensorFanout(transfers).to_dict()
            return wire_outputs

        wire_outputs = await _run_with_stage_lifecycle(
            invoke(), context, transport_context
        )

        yield StageResponseEnvelope(
            stage_id=self._stage_id,
            contract_id=self._runner.contract.id,
            attempt_id=envelope.attempt_id,
            invocation_id=envelope.invocation_id,
            outputs=wire_outputs,
        ).to_dict()
