# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

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
    routed_experts=None,
):
    return SimpleNamespace(
        index=index,
        token_ids=token_ids,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
        logprobs=logprobs,
        routed_experts=routed_experts,
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
    # These delta-streaming tests exercise base-model requests only. Model the
    # no-LoRA branch without constructing the full legacy worker handler.
    handler._generate_with_lora_admission_lock = lambda lora_request, create_generator: (
        create_generator(lora_request)
    )
    return handler


async def _collect_handler_chunks(
    responses,
    generation_artifact_session=None,
    include_routed_experts_response=True,
):
    handler = _handler_with_responses(responses)
    chunks = []
    async for chunk in BaseWorkerHandler.generate_tokens(
        handler,
        prompt=None,
        sampling_params=SamplingParams(),
        request_id="req-1",
        generation_artifact_session=generation_artifact_session,
        include_routed_experts_response=include_routed_experts_response,
    ):
        chunks.append(chunk)
    return chunks, handler


class _RecordingArtifactSession:
    def __init__(self):
        self.records = []
        self.finalizations = []

    def record_chunk(
        self,
        *,
        choice_index,
        prompt_token_ids,
        completion_token_ids,
        selected_logprobs,
        routed_experts,
    ):
        self.records.append(
            {
                "choice_index": choice_index,
                "prompt_token_ids": prompt_token_ids,
                "completion_token_ids": completion_token_ids,
                "selected_logprobs": selected_logprobs,
                "routed_experts": routed_experts,
            }
        )

    async def finalize_choice(self, *, choice_index, token_start):
        self.finalizations.append(
            {"choice_index": choice_index, "token_start": token_start}
        )
        return {
            "format": "generation_artifact_v1",
            "contents": ["moe_routes", "selected_logprobs"],
            "state": "ready",
            "actual_bytes": 123,
            "sha256": "a" * 64,
            "object_id": "opaque-1",
        }


class _FailingArtifactSession(_RecordingArtifactSession):
    contents = frozenset({"selected_logprobs"})

    async def finalize_choice(self, *, choice_index, token_start):
        raise RuntimeError("provider detail must not escape")


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
async def test_generation_artifact_receives_raw_routes_and_terminal_receipt() -> None:
    routes = np.array([[[0]], [[1]]], dtype=np.int32)
    logprobs = [
        {7: SimpleNamespace(logprob=-0.7, rank=1, decoded_token="a")},
        {8: SimpleNamespace(logprob=-0.8, rank=1, decoded_token="b")},
    ]
    session = _RecordingArtifactSession()
    responses = [
        _request_output([_output([7], logprobs=logprobs[:1])], prompt_token_ids=[101]),
        _request_output(
            [
                _output(
                    [8],
                    finish_reason="length",
                    logprobs=logprobs[1:],
                    routed_experts=routes,
                )
            ],
            prompt_token_ids=[101],
        ),
    ]

    chunks, _ = await _collect_handler_chunks(responses, session)

    assert session.records[-1]["routed_experts"] is routes
    assert session.records[-1]["prompt_token_ids"] == [101]
    assert [record["completion_token_ids"] for record in session.records] == [[7], [8]]
    assert [record["selected_logprobs"] for record in session.records] == [
        [-0.7],
        [-0.8],
    ]
    assert session.finalizations == [{"choice_index": 0, "token_start": 0}]
    assert chunks[-1]["engine_data"]["generation_artifact"]["state"] == "ready"


@pytest.mark.asyncio
async def test_artifact_only_routes_skip_legacy_base64_projection() -> None:
    routes = np.array([[[0]]], dtype=np.int32)
    session = _RecordingArtifactSession()
    responses = [
        _request_output(
            [_output([7], finish_reason="length", routed_experts=routes)],
            prompt_token_ids=[101],
        )
    ]

    chunks, _ = await _collect_handler_chunks(
        responses, session, include_routed_experts_response=False
    )

    assert session.records[-1]["routed_experts"] is routes
    assert "routed_experts" not in chunks[-1]["engine_data"]
    assert chunks[-1]["engine_data"]["generation_artifact"]["state"] == "ready"


@pytest.mark.asyncio
async def test_artifact_delivery_failure_emits_sanitized_terminal_receipt() -> None:
    responses = [
        _request_output([_output([7], finish_reason="length")], prompt_token_ids=[101])
    ]

    chunks, _ = await _collect_handler_chunks(responses, _FailingArtifactSession())

    receipt = chunks[-1]["engine_data"]["generation_artifact"]
    assert receipt == {
        "format": "generation_artifact_v1",
        "contents": ["selected_logprobs"],
        "state": "failed",
        "error_code": "artifact_delivery_failed",
        "error": "generation artifact delivery failed",
    }


def test_generation_artifact_selected_logprobs_forces_capture_only() -> None:
    request = {
        "token_ids": [1, 2, 3],
        "sampling_options": {},
        "stop_conditions": {},
        "output_options": {},
        "extra_args": {
            "nvext": {
                "generation_artifact": {
                    "format": "generation_artifact_v1",
                    "contents": ["selected_logprobs"],
                    "delivery": {
                        "mode": "object_store",
                        "target": {
                            "kind": "managed_fsspec",
                            "profile": "training",
                            "object_key": "run/request.dynexp",
                        },
                    },
                }
            }
        },
    }

    sampling_params = build_sampling_params(request, {})

    assert sampling_params.logprobs == 0
