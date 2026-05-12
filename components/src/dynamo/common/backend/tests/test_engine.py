# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``LLMEngine`` ABC.

Only tests that exercise actual logic: the ABC's abstract-method enforcement
and the concrete default implementation of ``abort()``.  Pure dataclass /
TypedDict mechanics are not tested — those are Python-language guarantees.
"""

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
)

from dynamo.common.backend.engine import (  # noqa: E402
    EngineConfig,
    GenerateChunk,
    LLMEngine,
)

# Framework-agnostic: routed to sample-unified-test via
# `pre_merge and gpu_0 and unified` (see test_engine.py module docstring).
pytestmark = [
    pytest.mark.unit,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _MissingCleanup(LLMEngine):
    """Every abstract method implemented except ``cleanup()`` — must remain
    abstract."""

    @classmethod
    async def from_args(cls, argv=None):  # type: ignore[override]
        raise NotImplementedError

    async def start(self):
        raise NotImplementedError

    async def generate(self, request, context):  # type: ignore[override]
        chunk: GenerateChunk = {"token_ids": []}
        yield chunk


class _Complete(LLMEngine):
    @classmethod
    async def from_args(cls, argv=None):  # type: ignore[override]
        return cls(), None

    async def start(self):
        return EngineConfig(model="fake")

    async def generate(self, request, context):  # type: ignore[override]
        chunk: GenerateChunk = {"token_ids": []}
        yield chunk

    async def cleanup(self) -> None:
        pass


def test_llm_engine_is_abstract():
    """The ABC itself cannot be instantiated, nor can a subclass that misses
    any of the four required methods.  Engine authors who forget one get a
    TypeError at instantiation, not a silent NotImplementedError at runtime."""
    with pytest.raises(TypeError):
        LLMEngine()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        _MissingCleanup()  # type: ignore[abstract]


async def test_default_abort_is_noop():
    """``abort()`` has a concrete default so engines without a cancellation
    mechanism are not forced to override it.  Awaiting it must complete
    without raising and without touching the (None) context."""
    engine = _Complete()
    await engine.abort(None)  # type: ignore[arg-type]
