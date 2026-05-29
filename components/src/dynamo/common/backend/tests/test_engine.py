# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``LLMEngine`` ABC and the shared logits-processor
spec entry layer that lives next to it in ``engine.py``.

Only tests that exercise actual logic: the ABC's abstract-method enforcement
and the concrete default implementation of ``abort()``.  Pure dataclass /
TypedDict mechanics are not tested — those are Python-language guarantees.

The shared logits-processor layer (spec entry model, PREFILL skip,
env-gated smoke resolver, lazy-tokenizer regression) is tested below
without any backend imports."""

from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
    exc_type=ImportError,
)

from dynamo.common.backend.engine import (  # noqa: E402
    TEST_LOGITS_PROCESSOR_ENV,
    EngineConfig,
    ForcedTokenSequenceSpec,
    GenerateChunk,
    LLMEngine,
    LogitsProcessorSpec,
    PythonProcessorSpec,
    logits_processors_for_request,
    resolve_test_logits_processor_spec,
)
from dynamo.logits_processing.examples import (  # noqa: E402
    ForcedSequenceLogitsProcessor,
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


async def test_default_logits_processor_spec_is_none():
    """``logits_processor_spec()`` has a concrete default that opts the
    engine out of custom logits processors. Mirrors `kv_event_sources`
    returning `[]` by default."""
    engine = _Complete()
    assert await engine.logits_processor_spec() is None


# ---------------------------------------------------------------------------
# Shared logits-processor spec entry layer
# ---------------------------------------------------------------------------


def _logits_processor_set_env(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(TEST_LOGITS_PROCESSOR_ENV, raising=False)
    else:
        monkeypatch.setenv(TEST_LOGITS_PROCESSOR_ENV, value)


def test_build_returns_empty_when_plan_none():
    assert logits_processors_for_request(None, is_prefill=False) == []
    assert logits_processors_for_request(None, is_prefill=True) == []


def test_build_returns_empty_in_prefill_when_skip_prefill():
    """PREFILL skip is the disaggregated policy: prefill emits a
    visible token before decode resumes, and a fresh stateful processor
    on each side would corrupt the leading characters."""
    spec = LogitsProcessorSpec(
        entries=(ForcedTokenSequenceSpec(token_ids=(1, 2), eos_token_id=0),),
        skip_prefill=True,
    )
    assert logits_processors_for_request(spec, is_prefill=True) == []
    assert logits_processors_for_request(spec, is_prefill=False) == list(spec.entries)


def test_build_returns_entries_in_prefill_when_skip_prefill_false():
    """A spec with `skip_prefill=False` (e.g. a stateless future spec entry)
    flows through PREFILL too. Pins that the skip is spec-driven, not
    hard-coded to PREFILL."""
    spec = LogitsProcessorSpec(
        entries=(ForcedTokenSequenceSpec(token_ids=(1, 2), eos_token_id=0),),
        skip_prefill=False,
    )
    assert logits_processors_for_request(spec, is_prefill=True) == list(spec.entries)


def test_build_entries_are_same_objects_across_calls():
    """Entries are immutable data; the same instances flow through
    every call. Per-request state isolation is the backend realizer's
    job (each realizer constructs fresh processors per call)."""
    desc = ForcedTokenSequenceSpec(token_ids=(1, 2), eos_token_id=0)
    spec = LogitsProcessorSpec(entries=(desc,))
    first = logits_processors_for_request(spec, is_prefill=False)
    second = logits_processors_for_request(spec, is_prefill=False)
    assert first == [desc]
    assert second == [desc]
    assert first[0] is second[0]  # same spec entry object, by design


class _MockTokenizer:
    """Minimal stand-in satisfying `resolve_test_logits_processor_spec`."""

    eos_token_id = 2

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False):
        return [ord(c) for c in text]


@pytest.mark.parametrize("env", [None, "0", ""])
def test_resolve_test_logits_processor_spec_returns_none_when_env_off(env, monkeypatch):
    """Env off: returns None AND does not invoke the tokenizer factory.
    Pins the HIGH-severity guarantee that an engine with
    `skip_tokenizer_init=True` and the env hook off won't crash at
    spec-resolution time."""
    _logits_processor_set_env(monkeypatch, env)
    invocations = []

    def _tokenizer_factory():
        invocations.append("invoked")
        return _MockTokenizer()

    assert resolve_test_logits_processor_spec(_tokenizer_factory) is None
    assert invocations == []  # factory must not be called when env is off


def test_resolve_test_logits_processor_spec_resolves_token_ids_at_startup(monkeypatch):
    """Env on returns a `LogitsProcessorSpec` whose single spec entry is a
    `ForcedTokenSequenceSpec` with PRE-RESOLVED token IDs (the mock
    tokenizer's `encode` is called here, not at per-request time)."""
    _logits_processor_set_env(monkeypatch, "1")
    invocations = []

    class _OneShotTokenizer:
        eos_token_id = 2

        def encode(self, text, add_special_tokens=False):
            invocations.append(text)
            return [ord(c) for c in text]

    spec = resolve_test_logits_processor_spec(_OneShotTokenizer)
    assert spec is not None
    assert spec.needs_tokenizer is True
    assert spec.skip_prefill is True
    assert len(spec.entries) == 1
    desc = spec.entries[0]
    assert isinstance(desc, ForcedTokenSequenceSpec)
    assert desc.token_ids == tuple(ord(c) for c in "Hello world!")
    assert desc.eos_token_id == 2
    # Tokenizer was consulted exactly once at spec-resolution time.
    assert invocations == ["Hello world!"]


def test_resolve_test_logits_processor_spec_raises_when_eos_missing(monkeypatch):
    """Forced-sequence needs an EOS to use after the response. A
    tokenizer without one is a config error, not a silent corruption."""
    _logits_processor_set_env(monkeypatch, "1")

    class _NoEosTokenizer:
        eos_token_id = None

        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]

    with pytest.raises(ValueError, match="eos_token_id"):
        resolve_test_logits_processor_spec(_NoEosTokenizer)


def test_forced_sequence_processor_advances_through_sequence():
    """Pins the realizer-target contract: per-request state isolation
    via fresh instances, plus the forced-sequence behaviour itself.
    Backend realizers (TRT-LLM today; vLLM/SGLang later) all rely on
    this class for the actual mask logic."""
    import torch

    a = ForcedSequenceLogitsProcessor(token_ids=(5, 7), eos_token_id=2)
    b = ForcedSequenceLogitsProcessor(token_ids=(5, 7), eos_token_id=2)
    assert a is not b
    assert a.state == 0 and b.state == 0

    scores = torch.zeros(16)
    a([1], scores)
    # After the call, all positions except token_ids[0]=5 are -inf.
    assert scores[5].item() == 0.0
    finite = (scores > float("-inf")).nonzero().flatten().tolist()
    assert finite == [5]
    assert a.state == 1
    # b is unaffected.
    assert b.state == 0


def test_engine_usage_pattern(monkeypatch):
    """Mirrors how a unified engine threads the spec entry layer:

    1. In `from_args`: inline env check to flip the backend-specific
       skip-tokenizer-init flag (backend-shaped; no shared helper).
    2. In `start()`: resolve the spec once via the ABC override.
    3. In `generate()`: build per-request activation (entries),
       then realize via the backend-specific function (mocked here).
    """
    import os as _os

    _logits_processor_set_env(monkeypatch, "1")
    engine_args: dict[str, Any] = {"skip_tokenizer_init": True}

    # Step 1: env-on triggers the backend-specific flag flip. Each
    # backend (TRT-LLM dict, SGLang ServerArgs attribute, vLLM
    # EngineArgs) writes this differently; the env-var name is the
    # only thing that's shared.
    if _os.getenv(TEST_LOGITS_PROCESSOR_ENV) == "1":
        engine_args["skip_tokenizer_init"] = False
    assert engine_args["skip_tokenizer_init"] is False

    # Step 2: resolve.
    spec = resolve_test_logits_processor_spec(lambda: _MockTokenizer())
    assert spec is not None

    # Step 3a: aggregated / decode request activates.
    entries = logits_processors_for_request(spec, is_prefill=False)
    assert len(entries) == 1
    assert isinstance(entries[0], ForcedTokenSequenceSpec)

    # Step 3b: prefill request skips.
    entries_pf = logits_processors_for_request(spec, is_prefill=True)
    assert entries_pf == []

    # The backend-specific realizer is duck-typed; verify the
    # contract shape by mocking.
    fake_sampling_params = MagicMock()
    fake_attach = MagicMock()
    fake_attach(fake_sampling_params, entries)
    fake_attach.assert_called_once_with(fake_sampling_params, entries)


def test_python_processor_spec_is_carried_through():
    """PythonProcessorSpec is an escape hatch for in-process callers
    that bypass serialization. The shared layer just carries it through;
    the realizer decides whether to accept it (TRT-LLM does; vLLM and
    SGLang adapters will reject)."""

    def _factory():
        return MagicMock()

    desc = PythonProcessorSpec(factory=_factory)
    spec = LogitsProcessorSpec(entries=(desc,))
    activation = logits_processors_for_request(spec, is_prefill=False)
    assert activation == [desc]
