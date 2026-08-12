# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ergonomic NVTX timeline annotations for Python, shared across Dynamo workers.

This wraps the correlated-range primitives exposed by the Rust core
(``dynamo._core``) so Python hot paths land on the same Nsight Systems timeline
as the Rust runtime, under one shared gate:

* build the ``dynamo._core`` extension with ``--features nvtx``, and
* set ``DYN_ENABLE_RUST_NVTX=1`` at runtime.

When either is off, :func:`range` is a cheap no-op.

Correlated ranges (start/end) are used rather than the thread-local push/pop
stack because the workers are ``asyncio`` coroutines that interleave on one
event-loop thread: a start/end pair stays correct across ``await`` points and
overlapping requests, whereas push/pop would mis-nest.

Example::

    from dynamo import nvtx

    with nvtx.range("worker.trtllm.preprocess"):
        await prepare_inputs(request)
"""

from collections.abc import Iterator
from contextlib import contextmanager

from dynamo._core import nvtx_enabled as _nvtx_enabled
from dynamo._core import nvtx_range_end as _nvtx_range_end
from dynamo._core import nvtx_range_start as _nvtx_range_start

__all__ = ["enabled", "range"]


def enabled() -> bool:
    """Return ``True`` when NVTX annotations are active.

    Active means the ``dynamo._core`` extension was built with the ``nvtx``
    feature *and* ``DYN_ENABLE_RUST_NVTX`` is set. Use it to skip building an
    expensive span name when profiling is off.
    """
    return _nvtx_enabled()


@contextmanager
def range(message: str) -> Iterator[None]:
    """Annotate the wrapped block as a named NVTX range on the timeline.

    Safe in both synchronous and ``async`` code — the block may ``await`` or
    ``yield`` while the range is open. A no-op unless NVTX is :func:`enabled`.
    """
    range_id = _nvtx_range_start(message)
    try:
        yield
    finally:
        if range_id:
            _nvtx_range_end(range_id)
