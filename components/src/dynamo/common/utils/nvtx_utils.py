# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Lightweight NVTX wrappers for Dynamo profiling.

Set DYN_NVTX to 1/true/yes to enable markers; default is disabled (zero overhead).

The `nvtx` package is an optional dependency (`pip install ai-dynamo[profiling]`,
already present in the container images). It is imported only when DYN_NVTX=1,
so call sites below cost nothing and require nothing when profiling is off.
Setting DYN_NVTX=1 without the package installed raises at import rather than
silently recording nothing.

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
or domain cache lookups on the hot path.
"""
import functools
import inspect

from dynamo.common.utils.env import env_bool

# env_bool, not bool(int(...)): DYN_NVTX's Rust-side twin DYN_ENABLE_RUST_NVTX
# goes through config::env_is_truthy, which accepts 1/true/yes. Parsing this one
# as an int made `DYN_NVTX=true` raise a bare ValueError at import — the exact
# confusing failure the fail-fast ImportError below exists to avoid.
#
# `nvtx` is not a dependency of ai-dynamo — it ships in the optional
# `ai-dynamo[profiling]` extra and in the container images. It is imported only
# under DYN_NVTX=1, so the marker call sites carry no dependency on it.
ENABLED: bool = env_bool("DYN_NVTX")

if ENABLED:
    # Fail fast and loud. Silently degrading to no-ops would mean discovering at
    # the end of a profiling run that nothing was recorded; raising at import
    # tells the user immediately that either DYN_NVTX is set by mistake or the
    # profiling extra is missing.
    try:
        import nvtx as _nvtx_lib
    except ImportError as exc:
        raise ImportError(
            "DYN_NVTX=1 requires the `nvtx` package, which is not installed. "
            "Install it with `pip install ai-dynamo[profiling]`, or unset "
            "DYN_NVTX to run without NVTX markers."
        ) from exc

    # Named domain + pre-allocated EventAttributes: no per-call object
    # allocation or domain cache lookups on the hot path.
    _domain = _nvtx_lib.get_domain("dynamo")
    _attr_cache: dict = {}

    def _get_attr(message: str, color: str):
        try:
            return _attr_cache[message, color]
        except KeyError:
            attr = _domain.get_event_attributes(message=message, color=color)
            _attr_cache[message, color] = attr
            return attr

    def start_range(message: str, color: str = "white"):
        return _domain.start_range(_get_attr(message, color))

    def end_range(rng) -> None:
        _domain.end_range(rng)

    # functools.partial so decorator and context-manager usage both land
    # in the "dynamo" domain, keeping all markers in one nsys row.
    annotate = functools.partial(_nvtx_lib.annotate, domain="dynamo")

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
