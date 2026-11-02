# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real causal TITO frame production, shared with the Rust format and collector."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind

from dynamo.common.constants import DisaggregationMode
from dynamo.vllm.handlers import DecodeWorkerHandler, _backend_metadata_upload_settings

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]

FIXTURE = (
    Path(__file__).resolve().parents[5]
    / "tests/fixtures/decisions/causal-producer-v1.json"
)


async def produce_frames(worker_request):
    """Fake only the engine boundary; retain both real handler generators."""
    calls = []

    async def generate(prompt, sampling, request_id, **kwargs):
        assert prompt == {
            "prompt_token_ids": [1, 2, 3],
            "cache_salt": "dynamo-cache-salt:tenant",
        }
        assert sampling.logprob_token_ids == [20, 10]
        assert sampling.logprobs is None
        assert sampling.output_kind is RequestOutputKind.DELTA
        assert sampling.max_tokens == sampling.n == 1
        assert sampling.detokenize is False
        calls.append(request_id)
        yield RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=[99],
                    cumulative_logprob=None,
                    logprobs=[
                        {
                            20: Logprob(-2.0, rank=2, decoded_token="twenty"),
                            10: Logprob(-3.0, rank=3, decoded_token="ten"),
                            99: Logprob(-0.5, rank=1, decoded_token="sample"),
                        }
                    ],
                    finish_reason="length",
                    stop_reason=None,
                )
            ],
            finished=True,
            num_cached_tokens=0,
        )

    handler = object.__new__(DecodeWorkerHandler)
    handler.config = SimpleNamespace(disaggregation_mode=DisaggregationMode.AGGREGATED)
    handler.engine_client = SimpleNamespace(
        generate=generate,
        tokenizer=None,
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=1),
            model_config=SimpleNamespace(max_model_len=128),
        ),
    )
    handler.default_sampling_params = {}
    handler._multimodal_request_processor = SimpleNamespace(enable_multimodal=False)
    handler._deferred_aborts = {}
    handler._resolve_lora_request = Mock(return_value=None)
    handler._abort_monitor = Mock(return_value=nullcontext())
    context = SimpleNamespace(trace_headers=lambda: {})
    frames = [
        frame
        async for frame in handler._generate_token_mode(
            worker_request, context, "causal-producer-1"
        )
    ]
    assert calls == ["causal-producer-1"]
    return frames


@pytest.mark.asyncio
async def test_actual_causal_worker_preserves_requested_scores_and_terminal_cache():
    fixture = json.loads(FIXTURE.read_text())
    request = fixture["worker_request"]
    assert request["output_options"]["logprobs"] is None
    assert _backend_metadata_upload_settings(request) is None
    assert request["extra_args"]["vllm_tito"] == {
        key: value for key, value in fixture["request"].items() if key != "token_ids"
    } | {"request_id": "causal-producer-1", "priority": 0}
    frames = await produce_frames(request)
    assert frames == fixture["frames"]
    assert len(frames) == 1
    terminal = frames[0]
    assert terminal["finish_reason"] == "length"
    assert terminal["token_ids"] == [99]
    row = terminal["top_logprobs"][0]
    assert [entry["token_id"] for entry in row] == [20, 10, 99]
    assert [entry["logprob"] for entry in row] == [-2.0, -3.0, -0.5]
    assert terminal["completion_usage"]["prompt_tokens"] == 3
    assert terminal["completion_usage"]["prompt_tokens_details"] == {"cached_tokens": 0}
