# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration-driven SGLang ``/engine/*`` route registration."""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import inspect
import re
import threading
import types
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Literal,
    TypeAlias,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

_ROUTE_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
EngineRouteTarget: TypeAlias = Literal["engine", "tm"]
_TARGETS: frozenset[EngineRouteTarget] = frozenset({"engine", "tm"})


@dataclass(frozen=True)
class EngineRouteDescriptor:
    """A configured public path bound to one SGLang method."""

    path: str
    method: str
    target: EngineRouteTarget = "engine"


def _invalid_descriptor(descriptor: str, message: str) -> ValueError:
    return ValueError(
        f"Invalid SGLang engine route descriptor {descriptor!r}: {message}"
    )


def parse_engine_route_descriptors(
    values: str | Sequence[str] | None,
) -> list[EngineRouteDescriptor]:
    """Parse ``<path>[=<method>][:engine|tm]`` route descriptors."""

    if values is None:
        return []
    if isinstance(values, str):
        if not values.strip():
            return []
        raw_descriptors = values.split()
    else:
        raw_descriptors = list(values)

    descriptors: list[EngineRouteDescriptor] = []
    paths: set[str] = set()
    for position, raw_descriptor in enumerate(raw_descriptors, start=1):
        if not isinstance(raw_descriptor, str):
            raise ValueError(
                "Invalid SGLang engine route descriptor at position "
                f"{position}: expected a string"
            )

        descriptor = raw_descriptor.strip()
        if not descriptor:
            raise ValueError(
                "Invalid SGLang engine route descriptor at position "
                f"{position}: descriptor is empty"
            )
        if descriptor.count(":") > 1:
            raise _invalid_descriptor(
                descriptor, "expected at most one ':' target separator"
            )

        route_and_method, separator, raw_target = descriptor.rpartition(":")
        if separator:
            if not route_and_method or not raw_target:
                raise _invalid_descriptor(
                    descriptor, "both the route and target are required"
                )
            if raw_target not in _TARGETS:
                raise _invalid_descriptor(
                    descriptor,
                    f"unknown target {raw_target!r}; expected 'engine' or 'tm'",
                )
            target = cast(EngineRouteTarget, raw_target)
        else:
            route_and_method = descriptor
            target = "engine"

        if route_and_method.count("=") > 1:
            raise _invalid_descriptor(
                descriptor, "expected at most one '=' method separator"
            )

        path, method_separator, method = route_and_method.partition("=")
        if not method_separator:
            method = path
        if not path or not method:
            raise _invalid_descriptor(
                descriptor, "both the route path and method are required"
            )

        if any(not _ROUTE_SEGMENT_RE.fullmatch(segment) for segment in path.split("/")):
            raise _invalid_descriptor(
                descriptor,
                "the route path must contain non-empty '/'-separated segments "
                "using only letters, digits, '.', '_', and '-'",
            )
        if not method.isidentifier():
            raise _invalid_descriptor(
                descriptor, f"method {method!r} is not a Python identifier"
            )
        if method.startswith("_"):
            raise _invalid_descriptor(descriptor, "private methods cannot be exposed")
        if path in paths:
            raise _invalid_descriptor(
                descriptor, f"route path {path!r} is configured more than once"
            )

        paths.add(path)
        descriptors.append(EngineRouteDescriptor(path, method, target))

    return descriptors


def _unwrap_annotation(annotation: Any) -> Any:
    while True:
        origin = get_origin(annotation)
        if origin is Annotated:
            annotation = get_args(annotation)[0]
            continue
        if origin in (Union, types.UnionType):
            non_none_args = [
                argument
                for argument in get_args(annotation)
                if argument is not type(None)
            ]
            if len(non_none_args) == 1:
                annotation = non_none_args[0]
                continue
        return annotation


def _is_http_request_annotation(annotation: Any) -> bool:
    annotation = _unwrap_annotation(annotation)
    return (
        inspect.isclass(annotation)
        and annotation.__name__ == "Request"
        and annotation.__module__ in {"fastapi", "starlette.requests"}
    )


def _is_typed_request_annotation(annotation: Any) -> bool:
    annotation = _unwrap_annotation(annotation)
    if not inspect.isclass(annotation):
        return False
    return (
        dataclasses.is_dataclass(annotation)
        or hasattr(annotation, "__struct_fields__")
        or annotation.__module__ == "sglang.srt.managers.io_struct"
    )


def _callable_name(method: Callable[..., Any]) -> str:
    return getattr(method, "__name__", type(method).__name__)


def _resolve_type_hints(method: Callable[..., Any], target: Any) -> dict[str, Any]:
    function = getattr(method, "__func__", method)
    globalns = getattr(function, "__globals__", None)
    localns = {
        target_type.__name__: target_type for target_type in type(target).__mro__
    }
    try:
        return get_type_hints(
            function,
            globalns=globalns,
            localns=localns,
            include_extras=True,
        )
    except (NameError, TypeError) as error:
        raise ValueError(
            f"could not resolve type hints for method {_callable_name(method)!r}: "
            f"{error}"
        ) from error


@dataclass(frozen=True)
class _TokenizerManagerCallPlan:
    signature: inspect.Signature
    request_parameter: str | None
    request_type: type[Any] | None
    http_parameters: tuple[str, ...]

    @classmethod
    def from_method(
        cls, method: Callable[..., Any], target: Any
    ) -> _TokenizerManagerCallPlan:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "could not inspect tokenizer manager method "
                f"{_callable_name(method)!r}: {error}"
            ) from error

        type_hints = _resolve_type_hints(method, target)
        typed_parameters: list[tuple[str, type[Any]]] = []
        http_parameters: list[str] = []
        for parameter in signature.parameters.values():
            annotation = type_hints.get(parameter.name, parameter.annotation)
            if _is_http_request_annotation(annotation):
                http_parameters.append(parameter.name)
            elif _is_typed_request_annotation(annotation):
                typed_parameters.append(
                    (parameter.name, _unwrap_annotation(annotation))
                )

        if len(typed_parameters) > 1:
            names = ", ".join(name for name, _ in typed_parameters)
            raise ValueError(
                f"tokenizer manager method {_callable_name(method)!r} has multiple "
                f"typed request parameters: {names}"
            )

        if typed_parameters:
            request_parameter, request_type = typed_parameters[0]
            unsupported_required = [
                parameter.name
                for parameter in signature.parameters.values()
                if parameter.name not in {request_parameter, *http_parameters}
                and parameter.kind
                not in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
                and parameter.default is inspect.Parameter.empty
            ]
            if unsupported_required:
                raise ValueError(
                    f"tokenizer manager method {_callable_name(method)!r} has a typed "
                    "request plus unsupported required parameters: "
                    + ", ".join(unsupported_required)
                )
        else:
            request_parameter = None
            request_type = None

        return cls(
            signature=signature,
            request_parameter=request_parameter,
            request_type=request_type,
            http_parameters=tuple(http_parameters),
        )

    def prepare(self, body: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        call_kwargs: dict[str, Any]
        request_type = self.request_type
        if request_type is None:
            call_kwargs = dict(body)
            for parameter_name in self.http_parameters:
                call_kwargs[parameter_name] = None
            return [], call_kwargs

        request_parameter = self.request_parameter
        if request_parameter is None:
            raise RuntimeError("typed request plan has no request parameter")
        values: dict[str, Any] = {
            request_parameter: request_type(**body),
            **{parameter_name: None for parameter_name in self.http_parameters},
        }
        args: list[Any] = []
        call_kwargs = {}
        for parameter in self.signature.parameters.values():
            if parameter.name not in values:
                continue
            value = values[parameter.name]
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(value)
            else:
                call_kwargs[parameter.name] = value
        return args, call_kwargs


class _RunningLoopBridge:
    """Let a sync SGLang Engine wrapper submit work to its owner loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def run_until_complete(self, awaitable: Awaitable[Any]) -> Any:
        async def await_result() -> Any:
            return await awaitable

        return asyncio.run_coroutine_threadsafe(await_result(), self._loop).result()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loop, name)


async def _run_in_thread(
    function: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Run one blocking call without tying it to the event loop's executor."""

    result: concurrent.futures.Future[Any] = concurrent.futures.Future()

    def call() -> None:
        try:
            result.set_result(function(*args, **kwargs))
        except Exception as error:
            result.set_exception(error)

    thread = threading.Thread(
        target=call,
        name="dynamo-sglang-engine-route",
        daemon=True,
    )
    thread.start()
    cancellation: asyncio.CancelledError | None = None
    while thread.is_alive():
        try:
            await asyncio.sleep(0.01)
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
    thread.join()
    if not result.done():
        raise RuntimeError("SGLang engine route thread terminated without a result")
    if cancellation is not None:
        raise cancellation
    return result.result()


def _is_msgspec_struct(value: Any) -> bool:
    return not isinstance(value, type) and hasattr(type(value), "__struct_fields__")


def normalize_engine_route_value(value: Any, active_ids: set[int] | None = None) -> Any:
    """Recursively convert an SGLang result to transport-safe JSON data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if active_ids is None:
        active_ids = set()

    is_dataclass = dataclasses.is_dataclass(value) and not isinstance(value, type)
    is_msgspec_struct = _is_msgspec_struct(value)
    if (
        not is_dataclass
        and not is_msgspec_struct
        and not isinstance(value, (dict, list, tuple, set, frozenset))
    ):
        return str(value)

    value_id = id(value)
    if value_id in active_ids:
        return "<recursive reference>"

    active_ids.add(value_id)
    try:
        if is_dataclass:
            return {
                field.name: normalize_engine_route_value(
                    getattr(value, field.name), active_ids
                )
                for field in dataclasses.fields(value)
            }
        if is_msgspec_struct:
            return {
                field_name: normalize_engine_route_value(
                    getattr(value, field_name), active_ids
                )
                for field_name in type(value).__struct_fields__
            }
        if isinstance(value, dict):
            return {
                str(key): normalize_engine_route_value(item, active_ids)
                for key, item in value.items()
            }
        return [normalize_engine_route_value(item, active_ids) for item in value]
    finally:
        active_ids.remove(value_id)


def normalize_engine_route_result(result: Any) -> dict[str, Any]:
    """Convert a configured Engine or tokenizer-manager result to a JSON object."""

    if result is None:
        return {"status": "ok"}
    if isinstance(result, tuple) and len(result) in {2, 3}:
        keys = ("success", "message", "num_paused_requests")
        return {
            key: normalize_engine_route_value(value)
            for key, value in zip(keys, result, strict=False)
        }
    normalized = normalize_engine_route_value(result)
    if isinstance(normalized, dict):
        return normalized
    return {"result": normalized}


class _ConfiguredEngineRoute:
    def __init__(
        self,
        *,
        descriptor: EngineRouteDescriptor,
        engine: Any,
        target: Any,
        method: Callable[..., Any],
        engine_state_lock: asyncio.Lock,
    ) -> None:
        self.descriptor = descriptor
        self._engine = engine
        self._target = target
        self._method = method
        self._engine_state_lock = engine_state_lock
        self._tm_call_plan = (
            _TokenizerManagerCallPlan.from_method(method, target)
            if descriptor.target == "tm"
            else None
        )

    async def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise ValueError(
                f"/engine/{self.descriptor.path} requires a JSON object body"
            )

        if self._tm_call_plan is None:
            args: list[Any] = []
            kwargs = dict(body)
        else:
            auto_create_handle_loop = getattr(
                self._target, "auto_create_handle_loop", None
            )
            if auto_create_handle_loop is not None:
                auto_create_handle_loop()
            args, kwargs = self._tm_call_plan.prepare(body)

        if self.descriptor.target == "engine":
            async with self._engine_state_lock:
                result = await self._invoke(args, kwargs)
        else:
            result = await self._invoke(args, kwargs)

        return normalize_engine_route_result(result)

    async def _invoke(self, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(self._method):
            result = await self._method(*args, **kwargs)
        elif self.descriptor.target == "engine":
            result = await self._call_sync_engine(args, kwargs)
        else:
            result = await _run_in_thread(self._method, *args, **kwargs)

        if inspect.isawaitable(result):
            result = await result
        return result

    async def _call_sync_engine(self, args: list[Any], kwargs: dict[str, Any]) -> Any:
        owner_loop = asyncio.get_running_loop()

        def call() -> Any:
            original_loop = getattr(self._engine, "loop", None)
            bridge_required = original_loop is owner_loop or (
                original_loop is not None
                and callable(getattr(original_loop, "is_running", None))
                and original_loop.is_running()
            )
            if bridge_required:
                assert original_loop is not None
                self._engine.loop = _RunningLoopBridge(original_loop)
            try:
                return self._method(*args, **kwargs)
            finally:
                if bridge_required:
                    self._engine.loop = original_loop

        return await _run_in_thread(call)


def resolve_configured_engine_routes(
    engine: Any,
    values: str | Sequence[str] | None,
) -> list[tuple[str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]]]:
    """Resolve every configured route and callable without registering any."""

    descriptors = parse_engine_route_descriptors(values)
    engine_state_lock = asyncio.Lock()
    resolved: list[
        tuple[str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]]
    ] = []
    for descriptor in descriptors:
        target = (
            engine
            if descriptor.target == "engine"
            else getattr(engine, "tokenizer_manager", None)
        )
        if target is None:
            raise ValueError(
                f"Invalid SGLang engine route descriptor for "
                f"/engine/{descriptor.path}: the Engine has no tokenizer_manager"
            )

        try:
            method = getattr(target, descriptor.method)
        except AttributeError as error:
            raise ValueError(
                f"Invalid SGLang engine route descriptor for "
                f"/engine/{descriptor.path}: {descriptor.target} target has no "
                f"method {descriptor.method!r}"
            ) from error
        if not callable(method):
            raise ValueError(
                f"Invalid SGLang engine route descriptor for "
                f"/engine/{descriptor.path}: {descriptor.target} attribute "
                f"{descriptor.method!r} is not callable"
            )

        try:
            handler = _ConfiguredEngineRoute(
                descriptor=descriptor,
                engine=engine,
                target=target,
                method=method,
                engine_state_lock=engine_state_lock,
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid SGLang engine route descriptor for "
                f"/engine/{descriptor.path}: {error}"
            ) from error
        resolved.append((descriptor.path, handler))

    return resolved
