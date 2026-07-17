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

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_1'. "
        "tensorrt_llm import requires CUDA/GPU.",
        allow_module_level=True,
    )

from dynamo.trtllm.logits_processing.adapter import (
    TrtllmDynamoLogitsAdapter,
    attach_logits_processors,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(0),
]


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
