# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TRT-LLM logits-processor adapter."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_1'. "
        "tensorrt_llm import requires CUDA/GPU.",
        allow_module_level=True,
    )

from dynamo.trtllm.logits_processing.adapter import TrtllmDynamoLogitsAdapter

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_1,
]

# Intentionally unprofiled: the zero-VRAM tests run in the sequential GPU stage
# so TensorRT-LLM initialization is shared. The CUDA test overrides this below.


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


# Unlike the rest of this module (unprofiled CPU-only mocks), this test
# initializes a real CUDA context, so the GPU-parallel scheduler must reserve
# VRAM for it.
@pytest.mark.profiled_vram_gib(2.0)
@pytest.mark.requested_trtllm_vram_gib(2.0)
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
