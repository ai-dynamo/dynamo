# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental trusted-code route plugins for :mod:`dynamo.frontend`."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import re
import sys
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, TypeAlias

from dynamo._core import _CustomHttpRequest
from dynamo._core import _CustomHttpResponse as Response
from dynamo._core import _CustomHttpRoute
from dynamo.runtime import Context, DistributedRuntime

_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
_PARAMETER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Headers(Mapping[str, list[str]]):
    """Immutable case-insensitive, multi-value HTTP headers."""

    def __init__(self, values: Mapping[str, list[str]]) -> None:
        self._values = {name.lower(): list(items) for name, items in values.items()}

    def __getitem__(self, name: str) -> list[str]:
        return list(self._values[name.lower()])

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def getall(self, name: str) -> list[str]:
        """Return all values for ``name``, or an empty list when absent."""
        return list(self._values.get(name.lower(), ()))


class Request:
    """A buffered request delivered to a custom route handler."""

    __slots__ = ("_inner", "_headers")

    def __init__(self, inner: _CustomHttpRequest) -> None:
        self._inner = inner
        self._headers = Headers(inner.headers)

    @property
    def method(self) -> str:
        return self._inner.method

    @property
    def path(self) -> str:
        return self._inner.path

    @property
    def path_params(self) -> dict[str, str]:
        return self._inner.path_params

    @property
    def query_string(self) -> str:
        """The raw query string without the leading ``?``."""
        return self._inner.query_string

    @property
    def query_params(self) -> dict[str, list[str]]:
        """Percent-decoded query parameters, preserving repeated values."""
        return self._inner.query_params

    @property
    def query(self) -> dict[str, list[str]]:
        """Alias for :attr:`query_params`."""
        return self.query_params

    @property
    def headers(self) -> Headers:
        return self._headers

    @property
    def body(self) -> bytes:
        return self._inner.body

    @property
    def context(self) -> Context:
        return self._inner.context

    def json(self) -> Any:
        """Decode the buffered body as JSON."""
        return self._inner.json()


Handler: TypeAlias = Callable[[Request], Awaitable[Response]]


def _parse_path(path: str) -> str:
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"route path must be an absolute string: {path!r}")
    if "?" in path or "#" in path:
        raise ValueError(
            f"route path must not contain a query string or fragment: {path!r}"
        )
    if path == "/":
        return "/"

    parameters: set[str] = set()
    canonical: list[str] = []
    for segment in path[1:].split("/"):
        if not segment:
            raise ValueError(f"route path contains an empty segment: {path!r}")
        if segment.startswith("{") or segment.endswith("}"):
            if not (segment.startswith("{") and segment.endswith("}")):
                raise ValueError(f"malformed route parameter in {path!r}")
            name = segment[1:-1]
            if name.startswith("*"):
                raise ValueError(f"catch-all routes are not supported: {path!r}")
            if not _PARAMETER.fullmatch(name):
                raise ValueError(f"invalid route parameter name {name!r} in {path!r}")
            if name in parameters:
                raise ValueError(f"duplicate route parameter {name!r} in {path!r}")
            parameters.add(name)
            canonical.append("{}")
        else:
            if "{" in segment or "}" in segment:
                raise ValueError(f"malformed route parameter in {path!r}")
            canonical.append(segment)
    return "/" + "/".join(canonical)


class Router:
    """Startup-only registry for trusted Python route handlers."""

    def __init__(self) -> None:
        self._routes: list[_CustomHttpRoute] = []
        self._registered: dict[tuple[str, str], str] = {}
        self._source = "<direct>"
        self._frozen = False

    def route(self, path: str, methods: Iterable[str]) -> Callable[[Handler], Handler]:
        """Register an async handler for one or more HTTP methods."""
        if self._frozen:
            raise RuntimeError("custom route registry is frozen")
        if isinstance(methods, str):
            raise TypeError("methods must be an iterable of method names, not a string")
        normalized = [method.upper() for method in methods]
        if not normalized:
            raise ValueError("custom route must specify at least one method")
        invalid = sorted(set(normalized) - _METHODS)
        if invalid:
            raise ValueError(f"unsupported custom route methods: {', '.join(invalid)}")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"duplicate method for custom route {path!r}")
        canonical = _parse_path(path)

        def decorator(handler: Handler) -> Handler:
            if self._frozen:
                raise RuntimeError("custom route registry is frozen")
            if not inspect.iscoroutinefunction(handler):
                raise TypeError(
                    f"custom route handler for {path!r} must be declared with async def"
                )

            async def invoke(inner: _CustomHttpRequest) -> Response:
                return await handler(Request(inner))

            for method in normalized:
                key = (method, canonical)
                if key in self._registered:
                    raise ValueError(
                        f"duplicate custom route {method} {path} from {self._source!r}; "
                        f"first registered by {self._registered[key]!r}"
                    )
                self._routes.append(
                    _CustomHttpRoute(self._source, method, path, invoke)
                )
                self._registered[key] = self._source
            return handler

        return decorator

    def get(self, path: str) -> Callable[[Handler], Handler]:
        return self.route(path, methods=["GET"])

    def post(self, path: str) -> Callable[[Handler], Handler]:
        return self.route(path, methods=["POST"])

    def put(self, path: str) -> Callable[[Handler], Handler]:
        return self.route(path, methods=["PUT"])

    def patch(self, path: str) -> Callable[[Handler], Handler]:
        return self.route(path, methods=["PATCH"])

    def delete(self, path: str) -> Callable[[Handler], Handler]:
        return self.route(path, methods=["DELETE"])

    def head(self, path: str) -> Callable[[Handler], Handler]:
        return self.route(path, methods=["HEAD"])

    def options(self, path: str) -> Callable[[Handler], Handler]:
        return self.route(path, methods=["OPTIONS"])

    def freeze(self) -> None:
        self._frozen = True

    def _set_source(self, source: str) -> None:
        self._source = source

    def _core_routes(self) -> list[_CustomHttpRoute]:
        if not self._frozen:
            raise RuntimeError("custom route registry must be frozen before use")
        return list(self._routes)


def _load_module(spec: str) -> ModuleType:
    path = Path(spec).expanduser()
    if path.is_file():
        if path.suffix != ".py":
            raise ValueError(f"custom route file must end in .py: {spec!r}")
        resolved = path.resolve()
        module_name = (
            "_dynamo_custom_routes_"
            + hashlib.sha256(str(resolved).encode()).hexdigest()
        )
        module_spec = importlib.util.spec_from_file_location(module_name, resolved)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"could not create an import spec for {resolved}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        try:
            module_spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        return module

    if path.suffix == ".py" or "/" in spec or spec.startswith("."):
        raise FileNotFoundError(f"custom route file does not exist: {spec}")
    return importlib.import_module(spec)


async def load_custom_routes(
    specs: Iterable[str], runtime: DistributedRuntime
) -> list[_CustomHttpRoute]:
    """Load route modules in order, run their hooks, and freeze the registry."""
    router = Router()
    for spec in specs:
        router._set_source(spec)
        try:
            module = _load_module(spec)
            hook = getattr(module, "register_routes", None)
            if not callable(hook):
                raise TypeError(
                    "module must define callable register_routes(router, runtime)"
                )
            result = hook(router, runtime)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                raise TypeError("register_routes(router, runtime) must return None")
        except Exception as error:
            raise RuntimeError(f"custom routes from {spec!r}: {error}") from error
    router.freeze()
    return router._core_routes()


__all__ = ["Headers", "Request", "Response", "Router"]
