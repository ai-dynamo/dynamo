# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Lightweight NVTX wrappers for Dynamo profiling.

Set DYN_NVTX=1 to enable markers; default is disabled (zero overhead).

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
or domain cache lookups on the hot path. Color names are resolved here,
so markers never need matplotlib.
"""
import functools
import inspect
import logging
import os

ENABLED: bool = bool(int(os.getenv("DYN_NVTX", "0")))

if ENABLED:
    try:
        import nvtx as _nvtx_lib
    except ModuleNotFoundError as exc:
        # Only a missing nvtx package degrades to the no-op path: profiling presets
        # set DYN_NVTX for every launch and an image without the package must still
        # serve. A broken installation stays visible.
        if exc.name != "nvtx":
            raise
        logging.getLogger(__name__).warning(
            "DYN_NVTX=1 but the nvtx package is not installed; NVTX markers are disabled"
        )
        ENABLED = False

if ENABLED:
    # Named domain + pre-allocated EventAttributes: no per-call object
    # allocation or domain cache lookups on the hot path.
    _domain = _nvtx_lib.get_domain("dynamo")
    _attr_cache: dict = {}

    # The nvtx package resolves only a few color names itself and imports
    # matplotlib for any other, which runtime images do not ship; an unresolved
    # name then raises TypeError at the annotation site, which for a decorator
    # means at import. Dynamo therefore passes nvtx an RGB integer. The names nvtx
    # knows keep its values; any unknown name gets the nvtx default color.
    _COLORS = {
        "black": 0x000000,
        "blue": 0x0000FF,
        "brown": 0xA52A2A,
        "cyan": 0x00FFFF,
        "darkgreen": 0x006400,
        "gray": 0x808080,
        "green": 0x008000,
        "grey": 0x808080,
        "lime": 0x00FF00,
        "magenta": 0xFF00FF,
        "navy": 0x000080,
        "olive": 0x808000,
        "orange": 0xFFA500,
        "pink": 0xFFC0CB,
        "purple": 0x800080,
        "rapids": 0x7400FF,
        "red": 0xFF0000,
        "teal": 0x008080,
        "white": 0xFFFFFF,
        "yellow": 0xFFFF00,
    }

    def _resolve_color(color):
        if color is None or isinstance(color, int):
            return color
        return _COLORS.get(color)

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
        """nvtx.annotate in the "dynamo" domain with a matplotlib-free color.

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
