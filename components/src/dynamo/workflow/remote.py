# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery-backed remote transport for workflow stage attempts."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, AsyncIterator, Mapping, Optional, Protocol

from dynamo.workflow.plan import INLINE_VALUE_TYPES
from dynamo.workflow.runtime import (
    StageContext,
    StageRunner,
    WorkflowExecutionError,
    _validate_value,
)
from dynamo.workflow.types import (
    StageContract,
    WorkflowValidationError,
    _require_value_spec,
    validate_name,
)

STAGE_REQUEST_SCHEMA = "dynamo.workflow.stage_request"
STAGE_RESPONSE_SCHEMA = "dynamo.workflow.stage_response"
STAGE_WIRE_VERSION = 1


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
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))

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
    """Invoke a workflow stage through a discovered Dynamo endpoint client."""

    def __init__(self, client: _DynamoClient) -> None:
        self._client = client

    async def run(
        self,
        stage_id: str,
        contract: StageContract,
        inputs: Mapping[str, Any],
        context: StageContext,
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
        )
        transport_context = None
        if context.request_context is not None:
            detach = getattr(context.request_context, "detached", None)
            if not callable(detach):
                raise WorkflowExecutionError(
                    "request context cannot create a detached child context"
                )
            transport_context = detach(context.invocation_id)

        stream = await self._client.round_robin(
            request.to_dict(), annotated=False, context=transport_context
        )
        responses = []
        try:
            async for response in stream:
                responses.append(response)
                if len(responses) > 1:
                    raise WorkflowExecutionError(
                        f"remote stage {stage_id!r} returned multiple terminal responses"
                    )
        except BaseException:
            if transport_context is not None:
                transport_context.stop_generating()
            close = getattr(stream, "aclose", None)
            if callable(close):
                await close()
            raise
        if not responses:
            raise WorkflowExecutionError(
                f"remote stage {stage_id!r} returned no terminal response"
            )
        envelope = StageResponseEnvelope.from_dict(responses[0])
        if (
            envelope.stage_id != stage_id
            or envelope.contract_id != contract.id
            or envelope.attempt_id != context.attempt_id
            or envelope.invocation_id != context.invocation_id
        ):
            raise WorkflowExecutionError(
                f"remote stage {stage_id!r} response identity does not match request"
            )
        return envelope.outputs


class RemoteStageServer:
    """Expose one StageRunner through a Dynamo streaming endpoint."""

    def __init__(self, stage_id: str, runner: StageRunner) -> None:
        validate_name(stage_id, "remote stage id")
        if not isinstance(runner, StageRunner):
            raise WorkflowValidationError("remote runner must implement StageRunner")
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
        )
        if unsupported_ports:
            raise WorkflowValidationError(
                "inline remote server does not support ports " f"{unsupported_ports}"
            )
        self._stage_id = stage_id
        self._runner = runner

    async def generate(
        self, request: Mapping[str, Any], transport_context: Any = None
    ) -> AsyncIterator[dict[str, Any]]:
        envelope = StageRequestEnvelope.from_dict(request)
        if envelope.stage_id != self._stage_id:
            raise WorkflowExecutionError(
                f"remote endpoint for {self._stage_id!r} received stage "
                f"{envelope.stage_id!r}"
            )
        if envelope.contract_id != self._runner.contract.id:
            raise WorkflowExecutionError(
                f"remote stage {self._stage_id!r} contract does not match endpoint"
            )

        expected_inputs = set(self._runner.contract.inputs)
        actual_inputs = set(envelope.inputs)
        if actual_inputs != expected_inputs:
            raise WorkflowExecutionError(
                f"remote stage {self._stage_id!r} inputs differ from its contract; "
                f"missing={sorted(expected_inputs - actual_inputs)}, "
                f"extra={sorted(actual_inputs - expected_inputs)}"
            )
        for name, spec in self._runner.contract.inputs.items():
            _validate_value(
                spec,
                envelope.inputs[name],
                f"remote stage {self._stage_id!r} input {name!r}",
            )

        loop = asyncio.get_running_loop()
        deadline = (
            None
            if envelope.timeout_seconds is None
            else loop.time() + envelope.timeout_seconds
        )
        cancelled = asyncio.Event()
        context = StageContext(
            workflow_name=envelope.workflow_name,
            stage_id=envelope.stage_id,
            attempt_id=envelope.attempt_id,
            invocation_id=envelope.invocation_id,
            deadline=deadline,
            _cancelled=cancelled,
            request_context=transport_context,
        )

        async def invoke() -> Mapping[str, Any]:
            return await self._runner.run(envelope.inputs, context)

        invoke_task = asyncio.create_task(
            invoke(), name=f"workflow-remote:{envelope.invocation_id}"
        )
        transport_task: asyncio.Future[Any] | None = None
        if transport_context is not None:
            wait_for_stop = getattr(transport_context, "async_killed_or_stopped", None)
            if callable(wait_for_stop):
                transport_task = asyncio.ensure_future(wait_for_stop())

        try:
            if transport_task is None:
                if envelope.timeout_seconds is None:
                    result = await invoke_task
                else:
                    result = await asyncio.wait_for(
                        invoke_task, timeout=envelope.timeout_seconds
                    )
            else:
                done, _ = await asyncio.wait(
                    {invoke_task, transport_task},
                    timeout=envelope.timeout_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if invoke_task in done:
                    result = invoke_task.result()
                elif transport_task in done:
                    raise asyncio.CancelledError()
                else:
                    raise asyncio.TimeoutError()
        except BaseException:
            cancelled.set()
            if not invoke_task.done():
                invoke_task.cancel()
            await asyncio.gather(invoke_task, return_exceptions=True)
            raise
        finally:
            if transport_task is not None and not transport_task.done():
                transport_task.cancel()
                await asyncio.gather(transport_task, return_exceptions=True)

        transport_stopped = transport_context is not None and bool(
            getattr(transport_context, "is_stopped", lambda: False)()
        )
        transport_killed = transport_context is not None and bool(
            getattr(transport_context, "is_killed", lambda: False)()
        )
        if transport_stopped or transport_killed:
            cancelled.set()
            raise asyncio.CancelledError()
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
        yield StageResponseEnvelope(
            stage_id=self._stage_id,
            contract_id=self._runner.contract.id,
            attempt_id=envelope.attempt_id,
            invocation_id=envelope.invocation_id,
            outputs=outputs,
        ).to_dict()
