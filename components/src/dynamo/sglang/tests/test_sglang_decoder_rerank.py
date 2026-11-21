# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder rerank protocol and lifecycle tests; no model is loaded."""

import asyncio
import math
from types import SimpleNamespace

import pytest

pytest.importorskip("sglang.srt.managers.io_struct")
pytest.importorskip("sglang.srt.speculative.spec_info")

from fastapi import HTTPException  # noqa: E402

from dynamo.llm.exceptions import (  # noqa: E402
    EngineShutdown,
    HttpError,
    InvalidArgument,
)
from dynamo.sglang.request_handlers.rerank.decoder_handler import (  # noqa: E402
    DecoderRerankWorkerHandler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]

_TEMPLATE = """Answer can only be "yes" or "no".
<Query>: {{ messages[0]['content'] }}
<Document>: {{ messages[1]['content'] }}"""


class _Context:
    def __init__(self):
        self.stop = asyncio.Event()

    def id(self):
        return "context"

    async def async_killed_or_stopped(self):
        await self.stop.wait()


class _Manager:
    def __init__(self, scores, *, block=False):
        self.scores = scores
        self.block = block
        self.requests = []
        self.aborts = []
        self.rid_to_state = {}
        self.started = asyncio.Event()
        self.tokenizer = SimpleNamespace(
            chat_template=_TEMPLATE,
            encode=lambda label, **kwargs: [{"yes": 7, "no": 8}[label]],
        )
        self.model_config = SimpleNamespace(
            model_path="Qwen/Qwen3-Reranker-0.6B",
            is_generation=True,
            hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
        )

    async def generate_request(self, request, raw_request):
        index = len(self.requests)
        self.requests.append(request)
        self.rid_to_state[request.rid] = object()
        if len(self.requests) == len(self.scores):
            self.started.set()
        try:
            if self.block:
                await asyncio.Event().wait()
            score = self.scores[index]
            if score is None:
                await asyncio.Event().wait()
            if isinstance(score, Exception):
                raise score
            yield {
                "meta_info": {
                    "id": request.rid,
                    "finish_reason": score
                    if isinstance(score, dict)
                    else {"type": "length"},
                    "output_token_ids_logprobs": [score],
                }
            }
        finally:
            self.rid_to_state.pop(request.rid, None)


def _handler(scores, *, block=False):
    manager = _Manager(scores, block=block)
    handler = DecoderRerankWorkerHandler.__new__(DecoderRerankWorkerHandler)
    handler.engine = SimpleNamespace(tokenizer_manager=manager)
    handler.enable_trace = False
    handler.shutdown_event = asyncio.Event()

    async def abort(manager, request_id, registry, context_id):
        # Older SGLang drops the registration when its waiter is cancelled.
        # Prove the adapter aborts before cancelling that waiter.
        if request_id in registry:
            manager.aborts.append(request_id)

    handler._abort_sglang_request = abort
    return handler, manager


async def _generate(handler, context, **kwargs):
    return [
        item
        async for item in handler.generate(
            {
                "model": "reranker",
                "query": "capital of France",
                "documents": ["Berlin", "Paris"],
                **kwargs,
            },
            context,
        )
    ]


@pytest.mark.asyncio
async def test_native_template_label_scores_and_top_n():
    handler, manager = _handler(
        [
            [(math.log(0.1), 7, None), (math.log(0.9), 8, None)],
            [(math.log(0.8), 7, None), (math.log(0.2), 8, None)],
        ]
    )
    [ranked] = await _generate(handler, _Context(), top_n=1, return_documents=False)
    assert len(ranked) == 1
    assert ranked[0]["index"] == 1
    assert ranked[0]["score"] == pytest.approx(0.8)
    assert "document" not in ranked[0]
    assert len({request.rid for request in manager.requests}) == 2
    assert "capital of France" in manager.requests[0].text
    assert "Berlin" in manager.requests[0].text
    assert manager.requests[0].sampling_params == {"max_new_tokens": 0}
    assert manager.requests[0].token_ids_logprob == [7, 8]
    assert manager.aborts == []
    assert manager.rid_to_state == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("signal", ["stop", "shutdown", "cancel_task"])
async def test_aborts_all_candidates_before_waiters_are_cancelled(signal):
    handler, manager = _handler([[], []], block=True)
    context = _Context()
    task = asyncio.create_task(_generate(handler, context))
    await asyncio.wait_for(manager.started.wait(), timeout=2)
    if signal == "stop":
        context.stop.set()
    elif signal == "shutdown":
        handler.shutdown_event.set()
    else:
        task.cancel()
    error = EngineShutdown if signal == "shutdown" else asyncio.CancelledError
    with pytest.raises(error):
        await asyncio.wait_for(task, timeout=2)
    assert set(manager.aborts) == {request.rid for request in manager.requests}
    assert manager.rid_to_state == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "logprobs",
    [[], [(0.0, 7, None)], [(float("nan"), 7, None), (0.0, 8, None)]],
)
async def test_missing_or_invalid_scores_are_errors(logprobs):
    handler, _ = _handler([logprobs, logprobs])
    with pytest.raises(ValueError, match="logprob"):
        await _generate(handler, _Context())


@pytest.mark.asyncio
async def test_extreme_logprobs_do_not_underflow():
    handler, _ = _handler([[(-1000.0, 7, None), (-1001.0, 8, None)]] * 2)
    [ranked] = await _generate(handler, _Context())
    assert ranked[0]["score"] == pytest.approx(1 / (1 + math.exp(-1)))


@pytest.mark.asyncio
async def test_zero_probability_label_is_valid():
    handler, _ = _handler(
        [[(-math.inf, 7, None), (0.0, 8, None)], [(0.0, 7, None), (-math.inf, 8, None)]]
    )
    [ranked] = await _generate(handler, _Context())
    assert [(row["index"], row["score"]) for row in ranked] == [(1, 1.0), (0, 0.0)]


@pytest.mark.parametrize(
    "architecture,is_generation",
    [
        ("Qwen3VLForConditionalGeneration", True),
        ("XLMRobertaForSequenceClassification", True),
        ("Qwen3ForCausalLM", False),
    ],
)
def test_rejects_unsupported_model_or_engine(architecture, is_generation):
    handler, manager = _handler([])
    manager.model_config.hf_config.architectures = [architecture]
    manager.model_config.is_generation = is_generation
    with pytest.raises(ValueError):
        handler.validate_engine(handler.engine)


def test_rejects_tokenizer_without_single_token_labels():
    handler, manager = _handler([])
    manager.tokenizer.encode = lambda *args, **kwargs: [7, 8]
    with pytest.raises(ValueError, match="one token"):
        handler.validate_engine(handler.engine)


def test_rejects_identical_label_token_ids():
    handler, manager = _handler([])
    manager.tokenizer.encode = lambda *args, **kwargs: [0]
    with pytest.raises(ValueError, match="distinct token IDs"):
        handler.validate_engine(handler.engine)


@pytest.mark.parametrize(
    "template",
    [
        "{{ messages[0]['content'] }}",
        'Answer can only be "yes" or "no". {{ query }} {{ document }}',
        'Answer can only be "yes" or "no". <|vision_start|>',
    ],
)
def test_rejects_incompatible_templates(template):
    handler, manager = _handler([])
    manager.tokenizer.chat_template = template
    with pytest.raises(ValueError):
        handler.validate_engine(handler.engine)


def test_accepts_supported_architecture_from_local_model_path():
    handler, manager = _handler([])
    manager.model_config.model_path = "/models/local-reranker"
    handler.validate_engine(handler.engine)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "native_error,status,error",
    [
        (
            {"type": "abort", "status_code": 400, "message": "engine error"},
            400,
            InvalidArgument,
        ),
        (
            {"type": "abort", "status_code": 503, "message": "engine error"},
            503,
            HttpError,
        ),
        (ValueError("engine error"), 400, InvalidArgument),
        (HTTPException(status_code=400, detail="engine error"), 400, InvalidArgument),
        (HTTPException(status_code=503, detail="engine error"), 503, HttpError),
    ],
)
async def test_preserves_native_error_status(native_error, status, error):
    handler, _ = _handler([native_error, native_error])
    with pytest.raises(error, match="engine error") as caught:
        await _generate(handler, _Context())
    if status == 503:
        assert caught.value.code == 503


@pytest.mark.asyncio
async def test_plain_native_abort_is_cancellation():
    handler, _ = _handler([{"type": "abort"}] * 2)
    with pytest.raises(asyncio.CancelledError):
        await _generate(handler, _Context())


@pytest.mark.asyncio
async def test_scoring_failure_aborts_remaining_candidate():
    handler, manager = _handler([ValueError("engine failed"), None])
    with pytest.raises(InvalidArgument, match="engine failed"):
        await asyncio.wait_for(_generate(handler, _Context()), timeout=2)
    assert manager.aborts == [manager.requests[1].rid]
    assert manager.rid_to_state == {}
