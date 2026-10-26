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

When enabled, uses a named nvtx.Domain and pre-allocated EventAttributes
objects (cached lazily by (message, color)) so that repeated calls to
start_range incur only a single dict lookup — no object allocation
or domain cache lookups on the hot path. Color names used by Dynamo are resolved
here without requiring matplotlib.
"""
import functools
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


# `nvtx` is not a dependency of ai-dynamo — it ships in the optional
# `ai-dynamo[profiling]` extra and in the container images. It is imported only
# when markers are enabled, so the call sites carry no dependency on it.
ENABLED: bool = _parse_enabled(ENV_VAR)

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
