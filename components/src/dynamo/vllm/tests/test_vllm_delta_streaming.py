# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from dynamo.common.constants import DisaggregationMode
from dynamo.vllm.handlers import BaseWorkerHandler, build_sampling_params

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _FakeEngineClient:
    tokenizer = None

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        async def _stream():
            for response in self.responses:
                yield response

        return _stream()


class _FakeContext:
    def id(self):
        return "req-1"

    def trace_headers(self):
        return None


def _output(
    token_ids,
    *,
    index=0,
    finish_reason=None,
    stop_reason=None,
    logprobs=None,
):
    return SimpleNamespace(
        index=index,
        token_ids=token_ids,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
        logprobs=logprobs,
    )


def _request_output(outputs, *, prompt_token_ids=None, num_cached_tokens=0):
    return SimpleNamespace(
        outputs=outputs,
        prompt_token_ids=prompt_token_ids if prompt_token_ids is not None else [101],
        num_cached_tokens=num_cached_tokens,
        kv_transfer_params=None,
    )


def _handler_with_responses(responses):
    def _ignore_log(*args, **kwargs):
        pass

    handler = SimpleNamespace()
    handler.engine_client = _FakeEngineClient(responses)
    handler.runtime = SimpleNamespace(shutdown=lambda: None)
    handler._extract_logprobs = BaseWorkerHandler._extract_logprobs
    handler._log_with_lora_context = _ignore_log
    return handler


async def _collect_handler_chunks(responses):
    handler = _handler_with_responses(responses)
    chunks = []
    async for chunk in BaseWorkerHandler.generate_tokens(
        handler,
        prompt=None,
        sampling_params=SamplingParams(),
        request_id="req-1",
    ):
        chunks.append(chunk)
    return chunks, handler


def test_build_sampling_params_forces_delta_token_mode():
    request = {
        "token_ids": [1, 2, 3],
        "sampling_options": {"output_kind": RequestOutputKind.CUMULATIVE},
        "stop_conditions": {},
        "output_options": {},
    }

    sampling_params = build_sampling_params(
        request,
        default_sampling_params={
            "detokenize": True,
            "output_kind": RequestOutputKind.CUMULATIVE,
        },
    )

    assert sampling_params.detokenize is False
    assert sampling_params.output_kind == RequestOutputKind.DELTA


@pytest.mark.asyncio
async def test_generate_tokens_passes_delta_chunks_without_cumulative_slicing():
    responses = [
        _request_output([_output([1])], prompt_token_ids=[10, 11]),
        _request_output([_output([2, 3])], prompt_token_ids=[10, 11]),
        _request_output(
            [_output([4], finish_reason="length")], prompt_token_ids=[10, 11]
        ),
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert [chunk["token_ids"] for chunk in chunks] == [[1], [2, 3], [4]]
    assert chunks[-1]["finish_reason"] == "length"
    assert chunks[-1]["completion_usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 4,
        "total_tokens": 6,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


@pytest.mark.asyncio
async def test_generate_tokens_keeps_final_empty_delta_chunk_for_usage():
    responses = [
        _request_output([_output([1, 2])], prompt_token_ids=[10]),
        _request_output([_output([], finish_reason="length")], prompt_token_ids=[10]),
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert [chunk["token_ids"] for chunk in chunks] == [[1, 2], []]
    assert chunks[-1]["finish_reason"] == "length"
    assert chunks[-1]["completion_usage"]["completion_tokens"] == 2
    assert chunks[-1]["completion_usage"]["total_tokens"] == 3


@pytest.mark.asyncio
async def test_generate_tokens_ignores_logprobs_on_empty_final_delta_chunk():
    logprobs = [
        {7: SimpleNamespace(logprob=-0.7, rank=1, decoded_token="a")},
    ]
    responses = [
        _request_output([_output([1, 2])], prompt_token_ids=[10]),
        _request_output(
            [_output([], finish_reason="length", logprobs=logprobs)],
            prompt_token_ids=[10],
        ),
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert [chunk["token_ids"] for chunk in chunks] == [[1, 2], []]
    assert "log_probs" not in chunks[-1]
    assert "top_logprobs" not in chunks[-1]
    assert chunks[-1]["finish_reason"] == "length"
    assert chunks[-1]["completion_usage"]["completion_tokens"] == 2
    assert chunks[-1]["completion_usage"]["total_tokens"] == 3


@pytest.mark.asyncio
async def test_generate_tokens_tracks_interleaved_output_indexes_independently():
    responses = [
        _request_output([_output([1], index=0), _output([10, 11], index=1)]),
        _request_output(
            [
                _output([2], index=0, finish_reason="length"),
                _output([12], index=1),
            ]
        ),
        _request_output([_output([], index=1, finish_reason="length")]),
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert [(chunk["index"], chunk["token_ids"]) for chunk in chunks] == [
        (0, [1]),
        (1, [10, 11]),
        (0, [2]),
        (1, [12]),
        (1, []),
    ]
    assert chunks[2]["completion_usage"]["completion_tokens"] == 5
    assert chunks[-1]["completion_usage"]["completion_tokens"] == 5


@pytest.mark.asyncio
async def test_generate_tokens_reads_delta_aligned_logprobs_from_zero_offset():
    logprobs = [
        {7: SimpleNamespace(logprob=-0.7, rank=1, decoded_token="a")},
        {8: SimpleNamespace(logprob=-0.8, rank=1, decoded_token="b")},
    ]
    responses = [
        _request_output([_output([7, 8], finish_reason="length", logprobs=logprobs)])
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert chunks[0]["token_ids"] == [7, 8]
    assert chunks[0]["log_probs"] == [-0.7, -0.8]
    assert [entry[0]["token_id"] for entry in chunks[0]["top_logprobs"]] == [7, 8]


@pytest.mark.asyncio
async def test_generate_tokens_keeps_multichunk_delta_logprobs_aligned():
    first_logprobs = [
        {7: SimpleNamespace(logprob=-0.7, rank=1, decoded_token="a")},
    ]
    second_logprobs = [
        {8: SimpleNamespace(logprob=-0.8, rank=1, decoded_token="b")},
        {9: SimpleNamespace(logprob=-0.9, rank=1, decoded_token="c")},
    ]
    responses = [
        _request_output([_output([7], logprobs=first_logprobs)]),
        _request_output(
            [_output([8, 9], finish_reason="length", logprobs=second_logprobs)]
        ),
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert [chunk["token_ids"] for chunk in chunks] == [[7], [8, 9]]
    assert [chunk["log_probs"] for chunk in chunks] == [[-0.7], [-0.8, -0.9]]
    assert [
        [entry[0]["token_id"] for entry in chunk["top_logprobs"]] for chunk in chunks
    ] == [[7], [8, 9]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("num_cached_tokens", "expected_prompt_tokens_details"),
    [
        (None, None),
        (0, {"cached_tokens": 0}),
        (1, {"cached_tokens": 1}),
    ],
)
async def test_unified_llm_engine_passes_delta_chunks_and_counts_usage(
    num_cached_tokens,
    expected_prompt_tokens_details,
):
    pytest.importorskip("vllm.usage.usage_lib")
    from dynamo.vllm.llm_engine import VllmLLMEngine
    from dynamo.vllm.multimodal_utils.request_processor import (
        VllmMultimodalRequestProcessor,
    )

    responses = [
        _request_output([_output([1])], prompt_token_ids=[10, 11]),
        _request_output(
            [_output([2, 3], finish_reason="length")],
            prompt_token_ids=[10, 11],
            num_cached_tokens=num_cached_tokens,
        ),
    ]
    engine = VllmLLMEngine.__new__(VllmLLMEngine)
    engine.engine_client = _FakeEngineClient(responses)
    engine._default_sampling_params = {}
    engine._model_max_len = None
    engine.disaggregation_mode = DisaggregationMode.AGGREGATED
    engine.enable_rl = False
    engine._multimodal_request_processor = VllmMultimodalRequestProcessor(
        model="test-model",
        enable_multimodal=False,
    )
    engine._dp_range = None

    chunks = [
        chunk
        async for chunk in VllmLLMEngine.generate(
            engine,
            {
                "token_ids": [10, 11],
                "sampling_options": {},
                "stop_conditions": {},
                "output_options": {},
            },
            _FakeContext(),
        )
    ]

    sampling_params = engine.engine_client.calls[0][0][1]
    assert sampling_params.output_kind == RequestOutputKind.DELTA
    assert [chunk["token_ids"] for chunk in chunks] == [[1], [2, 3]]
    assert chunks[-1]["completion_usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
        "prompt_tokens_details": expected_prompt_tokens_details,
    }


@pytest.mark.asyncio
async def test_unified_decode_preserves_prefill_cached_token_details():
    pytest.importorskip("vllm.usage.usage_lib")
    from dynamo.vllm.llm_engine import VllmLLMEngine
    from dynamo.vllm.multimodal_utils.request_processor import (
        VllmMultimodalRequestProcessor,
    )

    responses = [
        _request_output(
            [_output([1], finish_reason="length")],
            prompt_token_ids=[10, 11],
            num_cached_tokens=0,
        )
    ]
    engine = VllmLLMEngine.__new__(VllmLLMEngine)
    engine.engine_client = _FakeEngineClient(responses)
    engine._default_sampling_params = {}
    engine._model_max_len = None
    engine.disaggregation_mode = DisaggregationMode.DECODE
    engine.enable_rl = False
    engine._multimodal_request_processor = VllmMultimodalRequestProcessor(
        model="test-model",
        enable_multimodal=False,
    )
    engine._dp_range = None

    chunks = [
        chunk
        async for chunk in VllmLLMEngine.generate(
            engine,
            {
                "token_ids": [10, 11],
                "sampling_options": {},
                "stop_conditions": {},
                "output_options": {},
                "prefill_result": {
                    "disaggregated_params": {
                        "kv_transfer_params": {"remote_engine_id": "prefill-1"}
                    },
                    "prompt_tokens_details": {"cached_tokens": 8},
                },
            },
            _FakeContext(),
        )
    ]

    assert chunks[-1]["completion_usage"]["prompt_tokens_details"] == {
        "cached_tokens": 8
    }


@pytest.mark.asyncio
async def test_unified_llm_engine_forwards_cache_salt_to_prompt():
    pytest.importorskip("vllm.usage.usage_lib")
    from dynamo.vllm.llm_engine import VllmLLMEngine
    from dynamo.vllm.multimodal_utils.request_processor import (
        VllmMultimodalRequestProcessor,
    )

    engine = VllmLLMEngine.__new__(VllmLLMEngine)
    engine.engine_client = _FakeEngineClient([])
    engine._default_sampling_params = {}
    engine._model_max_len = None
    engine.disaggregation_mode = DisaggregationMode.AGGREGATED
    engine.enable_rl = False
    engine._multimodal_request_processor = VllmMultimodalRequestProcessor(
        model="test-model",
        enable_multimodal=False,
    )
    engine._dp_range = None

    request = {
        "token_ids": [10, 11],
        "sampling_options": {},
        "stop_conditions": {},
        "output_options": {},
        "extra_args": {"nvext": {"cache_salt": "tenant-a"}},
    }
    async for _ in VllmLLMEngine.generate(engine, request, _FakeContext()):
        pass

    prompt = engine.engine_client.calls[0][0][0]
    assert prompt["cache_salt"] == "dynamo-cache-salt:tenant-a"
