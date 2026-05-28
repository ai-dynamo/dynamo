# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified TRT-LLM backend's inline
DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR hook plus the existing
TrtllmDynamoLogitsAdapter."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_0'. "
        "tensorrt_llm import requires CUDA/GPU.",
        allow_module_level=True,
    )

from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.llm_engine import TrtllmLLMEngine
from dynamo.trtllm.logits_processing.adapter import TrtllmDynamoLogitsAdapter

ENV = "DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR"

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
    """Minimal stand-in that satisfies HelloWorldLogitsProcessor."""

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
    engine = TrtllmLLMEngine(
        engine_args={},
        model_name="test/model",
        disaggregation_mode=mode,
    )
    engine._engine = _FakeEngine(llm)  # type: ignore[assignment]
    return engine


@pytest.mark.parametrize(
    "mode, env, llm_factory, expect_attached",
    [
        (DisaggregationMode.AGGREGATED, None, _NoTokenizerLLM, False),
        (DisaggregationMode.AGGREGATED, "1", _OKLLM, True),
        (DisaggregationMode.DECODE, "1", _OKLLM, True),
        (DisaggregationMode.PREFILL, "1", _OKLLM, False),
    ],
)
def test_unified_generate_attachment_matrix(
    mode, env, llm_factory, expect_attached, monkeypatch
):
    """AGG+env-off: no attach, no tokenizer access (`_NoTokenizerLLM`
    raises if `.tokenizer` is read). AGG+env-on and DECODE+env-on: attach.
    PREFILL+env-on: skip (the disaggregated-mode policy)."""
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
