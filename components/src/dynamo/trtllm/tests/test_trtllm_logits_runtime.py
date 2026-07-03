# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TRT-LLM slice of the
DYN_ENABLE_TEST_LOGITS_PROCESSOR hook: the unified `from_args`
tokenizer-init flip, the unified `generate` attach/skip matrix
threaded through the shared spec entry layer in
`dynamo.common.backend.engine`, the TRT-LLM realizer (spec entry →
live `BaseLogitsProcessor` → `TrtllmDynamoLogitsAdapter`), the
adapter's shape and CUDA-stream behavior, and the realizer's
no-op-on-empty contract.

Shared-layer policy itself (generation-stage gating, spec entry
composition, env-gated spec resolver) is tested in
`dynamo.common.backend.tests.test_engine` without GPU or tensorrt_llm.
These tests exercise the same policy through the unified TRT-LLM
engine to confirm the wiring."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_1'. "
        "tensorrt_llm import requires CUDA/GPU.",
        allow_module_level=True,
    )

from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.llm_engine import TrtllmLLMEngine
from dynamo.trtllm.logits_processing.adapter import (
    TrtllmDynamoLogitsAdapter,
    attach_logits_processors,
)

ENV = "DYN_ENABLE_TEST_LOGITS_PROCESSOR"

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(0),
]


def _set_env(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)


class _MockTokenizer:
    """Minimal stand-in that satisfies `resolve_test_logits_processor_spec`."""

    eos_token_id = 2

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False):
        return [ord(c) for c in text]


# ---------------------------------------------------------------------------
# Unified from_args: tokenizer-init override after engine_args merge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("override_skip", [None, True])
def test_unified_from_args_forces_tokenizer_init(override_skip, monkeypatch):
    """With the env on, `engine.engine_args["skip_tokenizer_init"]` is
    `False` after all overrides, including the case where
    `--override-engine-args` set it to `True`."""
    _set_env(monkeypatch, "1")
    argv = [
        "--model-path",
        "Qwen/Qwen3-0.6B",
        "--free-gpu-memory-fraction",
        "0.3",
        "--max-seq-len",
        "1024",
        "--max-batch-size",
        "2",
    ]
    if override_skip is True:
        argv += ["--override-engine-args", '{"skip_tokenizer_init": true}']
    engine, _ = asyncio.run(TrtllmLLMEngine.from_args(argv))
    assert engine.engine_args["skip_tokenizer_init"] is False


def test_unified_from_args_does_not_force_tokenizer_init_for_prefill(monkeypatch):
    """A PREFILL worker never attaches the hook (generate() gates it out),
    so the env flip must NOT override an explicit `skip_tokenizer_init=True`
    the way it does for generation roles — the prefill worker keeps the
    tokenizer it was told to skip."""
    _set_env(monkeypatch, "1")
    argv = [
        "--model-path",
        "Qwen/Qwen3-0.6B",
        "--free-gpu-memory-fraction",
        "0.3",
        "--max-seq-len",
        "1024",
        "--max-batch-size",
        "2",
        "--disaggregation-mode",
        "prefill",
        "--override-engine-args",
        '{"skip_tokenizer_init": true}',
    ]
    engine, _ = asyncio.run(TrtllmLLMEngine.from_args(argv))
    assert engine.engine_args["skip_tokenizer_init"] is True


# ---------------------------------------------------------------------------
# Unified generate(): attach in AGG/DECODE, skip in PREFILL, no tokenizer
# access when env is off.
# ---------------------------------------------------------------------------


class _EmptyAsyncIter:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _NoTokenizerLLM:
    """Raises if `.tokenizer` is read; proves the env check guards
    tokenizer access at the call site, not only in the helper."""

    def __init__(self):
        self.captured_kwargs: dict[str, Any] | None = None

    @property
    def tokenizer(self):
        raise AssertionError("tokenizer must not be accessed when env is off")

    def generate_async(self, **kwargs):
        self.captured_kwargs = kwargs
        return _EmptyAsyncIter()


class _OKLLM:
    def __init__(self):
        self.tokenizer = _MockTokenizer()
        self.captured_kwargs: dict[str, Any] | None = None

    def generate_async(self, **kwargs):
        self.captured_kwargs = kwargs
        return _EmptyAsyncIter()


class _FakeContext:
    def id(self):
        return "test-req"

    def trace_headers(self):
        return None


class _FakeEngine:
    def __init__(self, llm):
        self._llm = object()
        self.llm = llm


def _make_engine(mode: DisaggregationMode, llm) -> TrtllmLLMEngine:
    """Construct an engine and stub `_engine` to a fake. Spec resolution
    is deferred to `_resolve_spec_like_start` below so the test driver
    can `await` the override the same way `start()` does."""
    engine = TrtllmLLMEngine(
        engine_args={},
        model_name="test/model",
        disaggregation_mode=mode,
    )
    engine._engine = _FakeEngine(llm)  # type: ignore[assignment]
    return engine


async def _resolve_plan_like_start(engine: TrtllmLLMEngine) -> None:
    """Mirror what `start()` does after engine init: unconditionally
    await the engine's `logits_processor_spec()` override and cache the
    result on the instance.

    Production calls this without an env guard. The override only touches
    the tokenizer for generation roles with the env on, so `_NoTokenizerLLM`
    (env-off, or any role gated out) never has its `.tokenizer` invoked."""
    engine._logits_processor_spec = await engine.logits_processor_spec()


def test_unified_prefill_resolves_no_spec_without_tokenizer(monkeypatch):
    """PREFILL with the env on resolves no spec and never touches the
    tokenizer: the role is gated out before resolution, so a prefill worker
    started without a tokenizer does not pay for one. `_NoTokenizerLLM`
    raises if `.tokenizer` is read."""
    _set_env(monkeypatch, "1")
    engine = _make_engine(DisaggregationMode.PREFILL, _NoTokenizerLLM())
    asyncio.run(_resolve_plan_like_start(engine))
    assert engine._logits_processor_spec is None


@pytest.mark.parametrize(
    "mode, env, llm_factory, expect_attached",
    [
        (DisaggregationMode.AGGREGATED, None, _NoTokenizerLLM, False),
        (DisaggregationMode.AGGREGATED, "1", _OKLLM, True),
        (DisaggregationMode.DECODE, "1", _OKLLM, True),
        (DisaggregationMode.PREFILL, "1", _NoTokenizerLLM, False),
    ],
)
def test_unified_generate_attachment_matrix(
    mode, env, llm_factory, expect_attached, monkeypatch
):
    """End-to-end pin through the shared spec entry layer:

    * AGG+env-off: no attach, no tokenizer access. `_NoTokenizerLLM`
      raises if `.tokenizer` is read; `resolve_test_logits_processor_spec` returns
      None without invoking the lazy tokenizer factory.
    * AGG+env-on, DECODE+env-on: attach. The shared
      `logits_processors_for_request` returns the spec's entries; the
      TRT-LLM realizer instantiates a `ForcedSequenceLogitsProcessor`
      from each and wraps it in `TrtllmDynamoLogitsAdapter`.
    * PREFILL+env-on: skip, with NO tokenizer access. The role is gated
      out before spec resolution, so `logits_processor_spec()` returns
      None (the `_NoTokenizerLLM` would raise if `.tokenizer` were read)
      and `sampling_params.logits_processor` stays None."""
    _set_env(monkeypatch, env)
    request: dict[str, Any] = {"token_ids": [1, 2, 3]}

    if mode == DisaggregationMode.PREFILL:
        # PREFILL builds an `LlmDisaggregatedParams` with a freshly minted
        # disagg_request_id; pin it to a constant so the test does not
        # depend on TRT-LLM's global counter state.
        monkeypatch.setattr(
            "dynamo.trtllm.llm_engine.get_global_disagg_request_id",
            lambda _machine_id: 0,
        )

    if mode == DisaggregationMode.DECODE:
        # `require_prefill_result` raises if `prefill_result` is missing,
        # and `_decode_prefill_handoff` is heavy. Provide a non-empty
        # `prefill_result` and stub the handoff to a `MagicMock` so we
        # reach `generate_async` without touching disagg codecs.
        from unittest.mock import MagicMock

        request["prefill_result"] = {"disaggregated_params": {}}
        monkeypatch.setattr(
            TrtllmLLMEngine,
            "_decode_prefill_handoff",
            staticmethod(lambda _result: MagicMock()),
        )

    llm = llm_factory()
    engine = _make_engine(mode, llm)

    async def _drive():
        await _resolve_plan_like_start(engine)
        async for _ in engine.generate(request, _FakeContext()):
            pass

    asyncio.run(_drive())

    assert llm.captured_kwargs is not None
    sp = llm.captured_kwargs["sampling_params"]
    if expect_attached:
        assert isinstance(sp.logits_processor, list)
        assert len(sp.logits_processor) == 1
    else:
        assert sp.logits_processor is None


# ---------------------------------------------------------------------------
# TRT-LLM attach_logits_processors contract
# ---------------------------------------------------------------------------


def test_attach_logits_processors_no_op_on_empty():
    """The unified engine calls `attach_logits_processors` unconditionally
    once `logits_processors_for_request` returns its (possibly empty) list of
    entries. Empty input must not touch
    `sampling_params.logits_processor`."""
    from unittest.mock import MagicMock

    sampling_params = MagicMock()
    sampling_params.logits_processor = None
    attach_logits_processors(sampling_params, [])
    assert sampling_params.logits_processor is None


def test_attach_logits_processors_realizes_forced_token_sequence_spec():
    """A `ForcedTokenSequenceSpec` realizes into a
    `ForcedSequenceLogitsProcessor` wrapped in
    `TrtllmDynamoLogitsAdapter`. Pins the spec entry-to-live-processor
    realization that the TRT-LLM realizer owns."""
    from unittest.mock import MagicMock

    from dynamo.common.backend.engine import ForcedTokenSequenceSpec

    sampling_params = MagicMock()
    entries = [ForcedTokenSequenceSpec(token_ids=(1, 2, 3), eos_token_id=2)]
    attach_logits_processors(sampling_params, entries)
    assert isinstance(sampling_params.logits_processor, list)
    assert len(sampling_params.logits_processor) == 1
    assert isinstance(sampling_params.logits_processor[0], TrtllmDynamoLogitsAdapter)


def test_attach_logits_processors_realizes_python_processor_spec():
    """A `PythonProcessorSpec` realizes by calling its factory
    fresh and wrapping the result. The TRT-LLM escape hatch for
    arbitrary `BaseLogitsProcessor` callables that don't fit a
    serializable spec entry."""
    from unittest.mock import MagicMock

    from dynamo.common.backend.engine import PythonProcessorSpec

    sampling_params = MagicMock()

    class _MinimalProcessor:
        def __call__(self, input_ids, scores):
            return None

    factory_calls = []

    def _factory():
        factory_calls.append("invoked")
        return _MinimalProcessor()

    entries = [PythonProcessorSpec(factory=_factory)]
    attach_logits_processors(sampling_params, entries)
    assert factory_calls == ["invoked"]
    assert isinstance(sampling_params.logits_processor, list)
    assert len(sampling_params.logits_processor) == 1
    assert isinstance(sampling_params.logits_processor[0], TrtllmDynamoLogitsAdapter)


# ---------------------------------------------------------------------------
# TrtllmDynamoLogitsAdapter behavior (no existing adapter unit coverage).
# ---------------------------------------------------------------------------


class _RecordingProcessor:
    def __init__(self):
        self.calls: list[tuple[list[int], torch.Tensor]] = []

    def __call__(self, input_ids, scores):
        self.calls.append((list(input_ids), scores.clone()))


@pytest.mark.parametrize(
    "shape, expect_invoke",
    [
        ((1, 1, 8), True),
        ((2, 1, 8), False),  # batch > 1
        ((1, 2, 8), False),  # beam > 1
    ],
)
def test_adapter_invokes_or_logs_on_bad_shape(shape, expect_invoke, caplog):
    """Supported `(1, 1, V)` shape invokes the processor with `ids[0]`
    and `logits[0, 0, :]`. Unsupported shapes log an error and leave
    logits unchanged (pinned via `caplog`, not `pytest.raises`)."""
    proc = _RecordingProcessor()
    adapter = TrtllmDynamoLogitsAdapter(proc)
    logits = torch.zeros(shape)
    logits_before = logits.clone()

    with caplog.at_level("ERROR", logger="dynamo.trtllm.logits_processing.adapter"):
        adapter(
            req_ids=0,
            logits=logits,
            ids=[[1, 2, 3]],
            stream_ptr=None,
        )

    if expect_invoke:
        assert len(proc.calls) == 1
        assert proc.calls[0][0] == [1, 2, 3]
        assert proc.calls[0][1].shape == (shape[2],)
        assert caplog.records == []
    else:
        assert proc.calls == []
        assert torch.equal(logits, logits_before)
        assert any(
            "logits processor" in record.message.lower() for record in caplog.records
        )


def test_adapter_enters_engine_cuda_stream():
    """When `stream_ptr` is non-null, the adapter wraps the processor
    call in `torch.cuda.stream(ExternalStream(stream_ptr))` so the
    processor runs on the engine's stream rather than the default
    stream. Capture the current stream inside the processor and confirm
    its raw pointer matches the engine stream we passed in."""
    captured: dict[str, int] = {}

    class _StreamCaptureProcessor:
        def __call__(self, input_ids, scores):
            captured["cuda_stream"] = torch.cuda.current_stream().cuda_stream

    engine_stream = torch.cuda.Stream()
    adapter = TrtllmDynamoLogitsAdapter(_StreamCaptureProcessor())
    logits = torch.zeros((1, 1, 8), device="cuda")
    adapter(
        req_ids=0,
        logits=logits,
        ids=[[1, 2, 3]],
        stream_ptr=engine_stream.cuda_stream,
    )
    assert captured.get("cuda_stream") == engine_stream.cuda_stream
