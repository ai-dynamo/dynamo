# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``LLMEngine`` ABC and retained SDK contracts.

Only tests that exercise actual logic: the ABC's abstract-method enforcement
and concrete default methods, generation-stage gating, and the serialized
logits-processor entry boundary. Pure dataclass and TypedDict mechanics are
not tested because those are Python-language guarantees.
"""

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
    exc_type=ImportError,
)

from dynamo.common.backend.engine import (  # noqa: E402
    EngineConfig,
    ForcedTokenSequenceSpec,
    GenerateChunk,
    LLMEngine,
    LogitsProcessorSpec,
    PythonProcessorSpec,
    deserialize_logits_processor_entries,
    is_generation_stage,
    logits_processors_for_request,
    serialize_logits_processor_entries,
)
from dynamo.common.constants import DisaggregationMode  # noqa: E402

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


async def test_default_logits_processor_spec_is_none():
    engine = _Complete()
    assert await engine.logits_processor_spec() is None


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (DisaggregationMode.AGGREGATED, True),
        (DisaggregationMode.DECODE, True),
        (DisaggregationMode.PREFILL, False),
        (DisaggregationMode.ENCODE, False),
    ],
)
def test_is_generation_stage(mode, expected):
    assert is_generation_stage(mode) is expected


@pytest.mark.parametrize(
    ("mode", "generation_only", "expected_count"),
    [
        (DisaggregationMode.DECODE, True, 1),
        (DisaggregationMode.PREFILL, True, 0),
        (DisaggregationMode.PREFILL, False, 1),
    ],
)
def test_logits_processors_for_request(mode, generation_only, expected_count):
    spec = LogitsProcessorSpec(
        entries=(ForcedTokenSequenceSpec(token_ids=(1, 2), eos_token_id=0),),
        generation_only=generation_only,
    )
    assert (
        len(logits_processors_for_request(spec, disaggregation_mode=mode))
        == expected_count
    )


def test_logits_processors_for_request_without_spec():
    assert (
        logits_processors_for_request(
            None, disaggregation_mode=DisaggregationMode.DECODE
        )
        == []
    )


def test_logits_processor_entries_round_trip():
    entries = [ForcedTokenSequenceSpec(token_ids=(7, 8, 9), eos_token_id=2)]
    payload = serialize_logits_processor_entries(entries)
    assert payload == [
        {"kind": "forced_sequence", "token_ids": [7, 8, 9], "eos_token_id": 2}
    ]
    assert deserialize_logits_processor_entries(payload) == entries


def test_logits_processor_entries_reject_unsupported_serialization():
    with pytest.raises(TypeError):
        serialize_logits_processor_entries([PythonProcessorSpec(factory=lambda: None)])

    with pytest.raises(ValueError):
        deserialize_logits_processor_entries([{"kind": "unknown"}])


async def test_default_engine_controls_are_empty():
    """Engines opt into management controls explicitly."""
    engine = _Complete()
    assert engine.supported_controls() == set()
    assert await engine.engine_control("pause", {}) == {
        "status": "error",
        "message": "unsupported engine control: pause",
    }
