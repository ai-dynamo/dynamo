# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Lightweight NVTX wrappers for Dynamo profiling.

Set DYN_NVTX to 1/true/on/yes to enable markers; default is disabled
(zero overhead). Values are trimmed and case-insensitive. The same variable
switches on the Rust runtime's markers, so one setting covers a whole capture.
An unrecognized value raises rather than silently disabling.

The `nvtx` package is an optional dependency (`pip install ai-dynamo[profiling]`,
already present in the container images). It is imported only when
DYN_NVTX is on, so call sites below cost nothing and require nothing when
profiling is off. Enabling it without the package installed logs a warning
and falls back to no-ops, so a launch preset that sets DYN_NVTX still serves.

Usage — same syntax as the bare nvtx module:

    from dynamo.common.utils import nvtx_utils as _nvtx

    # Manual range (needed when the range spans async yields or has conditional end)
    rng = _nvtx.start_range("my:range", color="blue")
    ...
    _nvtx.end_range(rng)

    # Decorator — annotates an entire function or async generator
    @_nvtx.annotate("my:func", color="green")
    def my_func(): ...

    @_nvtx.range_decorator("my:async_gen", color="green")
    async def my_async_gen():
        yield ...

    # Context manager — annotates a block (works with await and yield inside)
    with _nvtx.annotate("my:block", color="cyan"):
        result = await some_coroutine()

Request-scoped helpers (request_mark, start_request_range/end_request_range,
request_range, handler_stages) tag a range with the request id so it can be joined
with the same request's other processes and OpenTelemetry spans. They are
emitted by dynamo._core, never through the nvtx package (see the section at the
bottom of this module).

When enabled, uses a named nvtx.Domain and pre-allocated EventAttributes
objects (cached lazily by (message, color)) so that repeated calls to
start_range incur only a single dict lookup — no object allocation
or domain cache lookups on the hot path. Color names used by Dynamo are resolved
here without requiring matplotlib.
"""
import contextlib
import functools
import importlib
import inspect
import logging
import os

ENV_VAR = "DYN_NVTX"

# Parsed to match the Rust half of the same switch, which goes through
# dynamo_truthy::is_truthy (lib/truthy/src/lib.rs): 1/true/on/yes to enable,
# 0/false/off/no to disable, case-insensitive and whitespace-trimmed.
#
# Deliberately not the shared env_bool helper: it treats "on" as false by design
# (asserted in utils/tests/test_env.py), so DYN_NVTX=on would enable the Rust
# markers and silently disable these.
_TRUTHY = frozenset({"1", "true", "on", "yes"})
_FALSEY = frozenset({"0", "false", "off", "no"})


def _parse_enabled(name: str) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw or raw in _FALSEY:
        return False
    if raw in _TRUTHY:
        return True
    raise ValueError(
        f"{name}={os.getenv(name)!r} is not a recognized boolean. Use one of "
        f"{sorted(_TRUTHY)} to enable NVTX markers, or one of {sorted(_FALSEY)} "
        f"(or leave it unset) to disable them."
    )


ENABLED: bool = _parse_enabled(ENV_VAR)

# Request-scoped markers (bottom of this module) are emitted by dynamo._core, not
# the nvtx package, so they follow the switch alone: an image without the
# package still gets them.
REQUEST_ENABLED: bool = ENABLED

if ENABLED:
    try:
        import nvtx as _nvtx_lib
        from nvtx.colors import color_to_hex  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        # Only a missing nvtx package degrades to the no-op path: profiling presets
        # set DYN_NVTX for every launch and an image without the package must still
        # serve. A broken installation stays visible.
        if exc.name != "nvtx":
            raise
        logging.getLogger(__name__).warning(
            f"{ENV_VAR} is set but the nvtx package is not installed; NVTX markers are disabled"
        )
        ENABLED = False

if ENABLED:
    # Named domain + pre-allocated EventAttributes: no per-call object
    # allocation or domain cache lookups on the hot path.
    _domain = _nvtx_lib.get_domain("dynamo")
    _attr_cache: dict = {}

    # Reuse nvtx's color resolver, supplying only the extra names Dynamo uses
    # that would otherwise require matplotlib.
    _EXTRA_COLORS = {
        "lime": 0x00FF00,
        "magenta": 0xFF00FF,
        "pink": 0xFFC0CB,
    }

    def _resolve_color(color):
        if color is None or isinstance(color, int):
            return color
        try:
            return color_to_hex(_EXTRA_COLORS.get(color, color))
        except (TypeError, ValueError):
            # Missing matplotlib or an invalid color: use nvtx's default.
            return None

    def _get_attr(message: str, color: str):
        try:
            return _attr_cache[message, color]
        except KeyError:
            attr = _domain.get_event_attributes(
                message=message, color=_resolve_color(color)
            )
            _attr_cache[message, color] = attr
            return attr

    def start_range(message: str, color: str = "white"):
        return _domain.start_range(_get_attr(message, color))

    def end_range(rng) -> None:
        _domain.end_range(rng)

    def annotate(message=None, color=None, **kwargs):
        """nvtx.annotate in the "dynamo" domain with color fallback.

        Works as a decorator and as a context manager, like nvtx.annotate.
        Keeping every marker in one domain keeps them in one nsys row.
        """
        kwargs.setdefault("domain", "dynamo")
        return _nvtx_lib.annotate(message, color=_resolve_color(color), **kwargs)

    def range_decorator(message: str, color: str = "white"):
        """Decorator that wraps an async generator function with an NVTX range.

        Unlike annotate(), which only covers the synchronous setup before the
        first yield, this wraps the full generator iteration in a single range.
        """

        def decorator(func):
            if inspect.isasyncgenfunction(func):

                @functools.wraps(func)
                async def wrapper(*args, **kwargs):
                    rng = start_range(message, color)
                    try:
                        async for item in func(*args, **kwargs):
                            yield item
                    finally:
                        end_range(rng)

                return wrapper
            else:

                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    rng = start_range(message, color)
                    try:
                        return func(*args, **kwargs)
                    finally:
                        end_range(rng)

                return wrapper

        return decorator

else:
    # Pure Python no-ops: no C extension calls, no string allocations.
    # The ENV var is read once at import time — no per-call branch overhead.

    def start_range(message: str, color: str = "white"):  # type: ignore[misc]
        return None

    def end_range(rng) -> None:  # type: ignore[misc]
        pass

    class _NoOpAnnotate:
        """No-op that works as both a decorator and a context manager."""

        __slots__ = ()

        def __call__(self, func):
            return func

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    _noop_annotate = _NoOpAnnotate()

    def annotate(message: str = "", color: str = "white"):  # type: ignore[misc]
        return _noop_annotate

    def range_decorator(message: str = "", color: str = "white"):  # type: ignore[misc]
        """No-op decorator: returns the wrapped function unchanged."""

        def decorator(func):
            return func

        return decorator


# --------------------------------------------------------------------------- #
# Request-scoped markers and ranges
# --------------------------------------------------------------------------- #
#
# Each range or marker is named by its static stage and carries the request as a
# 64-bit NVTX payload key; ``attrs`` (such as ``engine_rid``) go into a
# ``dynamo.request rid=<id> ...`` identity marker with the same key. The Rust
# runtime (``dynamo._core``) owns that encoding, shared with the tracing mirror
# layer. These calls bypass the ``nvtx`` package and ``_attr_cache`` above: both
# register every distinct message in an unbounded cache, and identity markers
# are unique per request.
#
# Everything here is gated on ``REQUEST_ENABLED`` (DYN_NVTX alone; the ``nvtx``
# package is not required), and builds no strings or dicts while disabled.

_logger = logging.getLogger(__name__)

# None: not resolved yet. False: unavailable (warned once). Otherwise the
# (nvtx_request_mark, nvtx_request_range) pair from dynamo._core.
_core_api: "tuple | bool | None" = None


_warned_rust_off = False


def _request_api():
    """Resolve the Rust request-scoped bindings once; None if unavailable."""
    global _core_api, _warned_rust_off
    api = _core_api
    if api is None:
        try:
            core = importlib.import_module("dynamo._core")
            api = (core.nvtx_request_mark, core.nvtx_request_range)
            rust_enabled = core.nvtx_enabled()
        except (ImportError, AttributeError) as exc:
            _logger.warning(
                f"{ENV_VAR} is set but dynamo._core does not provide request-scoped "
                f"NVTX ({exc}); request-scoped NVTX markers are disabled"
            )
            _core_api = False
            return None
        if not rust_enabled:
            # Built without the `nvtx` feature, or asked before the runtime read
            # DYN_NVTX at startup: not cached, so a later call can still succeed.
            if not _warned_rust_off:
                _warned_rust_off = True
                _logger.warning(
                    f"{ENV_VAR} is set but the Rust runtime reports NVTX off (built "
                    f"without the nvtx feature, or not initialized yet); "
                    f"request-scoped NVTX markers are skipped"
                )
            return None
        _core_api = api
    return api or None


def _attrs_or_none(attrs: dict) -> "dict[str, str] | None":
    # Drop None so call sites can pass optional values (engine_rid=None) as-is.
    clean = {k: str(v) for k, v in attrs.items() if v is not None}
    return clean or None


def request_mark(stage: str, request_id: str, **attrs: str) -> None:
    """Emit an instantaneous request-scoped marker (no-op unless enabled)."""
    if not REQUEST_ENABLED:
        return
    api = _request_api()
    if api is not None:
        api[0](stage, request_id, _attrs_or_none(attrs))


def start_request_range(stage: str, request_id: str, **attrs: str):
    """Open a request-scoped range; returns a handle for end_request_range.

    Returns None when disabled or unavailable. The range also ends when the
    handle is garbage collected, but callers should end it in ``finally``.
    """
    if not REQUEST_ENABLED:
        return None
    api = _request_api()
    if api is None:
        return None
    return api[1](stage, request_id, _attrs_or_none(attrs))


def end_request_range(handle) -> None:
    """End a range from start_request_range. Idempotent; accepts None."""
    if handle is not None:
        handle.end()


@contextlib.contextmanager
def request_range(stage: str, request_id: str, **attrs: str):
    """Context manager form of start_request_range / end_request_range."""
    handle = start_request_range(stage, request_id, **attrs)
    try:
        yield handle
    finally:
        end_request_range(handle)


class _HandlerStages:
    """The two per-request handler ranges every backend emits.

    ``worker.handler.prepare:<role>`` runs from construction (handler body
    entry) to ``engine_start``; ``worker.engine.first_output:<role>`` runs from
    ``engine_start`` (the engine generate call) to ``first_output`` or, if the
    engine yields nothing, to ``close``. Callers MUST call ``close`` in a
    ``finally`` that covers the whole handler body so that exceptions, asyncio
    cancellation and early async-generator close still end the ranges.
    """

    __slots__ = (
        "_role",
        "_rid",
        "_engine_rid",
        "_prepare",
        "_first",
        "first_output_pending",
    )

    def __init__(self, role: str, request_id: str) -> None:
        self._role = role
        self._rid = request_id
        self._engine_rid: "str | None" = None
        self._first = None
        self.first_output_pending = False
        self._prepare = start_request_range(
            f"worker.handler.prepare:{role}", request_id
        )

    def end_prepare(self, engine_rid: "str | None" = None) -> None:
        """The handler hands the request to the engine: ends prepare.

        ``engine_rid`` is remembered for ``start_first_output``, for engines whose
        generate call is lazy and only submits once its stream is consumed.
        """
        end_request_range(self._prepare)
        self._prepare = None
        if engine_rid is not None:
            self._engine_rid = engine_rid

    def start_first_output(self, engine_rid: "str | None" = None) -> None:
        """Open first_output. ``engine_rid`` is omitted when equal to rid."""
        if engine_rid is None:
            engine_rid = self._engine_rid
        if self._first is None:
            self._first = start_request_range(
                f"worker.engine.first_output:{self._role}",
                self._rid,
                engine_rid=engine_rid if engine_rid != self._rid else None,
            )
            self.first_output_pending = self._first is not None

    def engine_start(self, engine_rid: "str | None" = None) -> None:
        """Mark the engine generate call: ends prepare, opens first_output."""
        self.end_prepare()
        self.start_first_output(engine_rid)

    def first_output(self) -> None:
        """The engine produced its first output; ends first_output."""
        self.first_output_pending = False
        end_request_range(self._first)
        self._first = None

    def close(self) -> None:
        """End whatever is still open. Idempotent."""
        self.first_output_pending = False
        end_request_range(self._prepare)
        end_request_range(self._first)
        self._prepare = None
        self._first = None


class _NoStages:
    """Shared inert stand-in used while request-scoped NVTX is off."""

    __slots__ = ()
    first_output_pending = False

    def end_prepare(self, engine_rid: "str | None" = None) -> None:
        pass

    def start_first_output(self, engine_rid: "str | None" = None) -> None:
        pass

    def engine_start(self, engine_rid: "str | None" = None) -> None:
        pass

    def first_output(self) -> None:
        pass

    def close(self) -> None:
        pass


NO_STAGES = _NoStages()


def handler_stages(role: str, request_id: str) -> "_HandlerStages | _NoStages":
    """Begin the handler prepare/first-output ranges for one request.

    ``role`` is one of prefill, decode, aggregated. Returns a shared no-op
    object (no allocation, no string building) when NVTX is off. Hot loops
    should guard ``first_output()`` with ``stages.first_output_pending``.
    """
    if not REQUEST_ENABLED or _request_api() is None:
        return NO_STAGES
    return _HandlerStages(role, request_id)
