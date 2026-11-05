# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.sglang.tests.prefill_handler_stubs import drain, kv_hit_prefill_handler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def _native_prefill_handler(monkeypatch, responses):
    """Prefill handler whose native stream yields ``responses`` as SGLang does."""
    handler = kv_hit_prefill_handler(monkeypatch, [], cancelled=False)

    async def native_results(_engine, _request):
        for response in responses:
            yield {
                "token_ids": response.get("output_ids") or [],
                "engine_data": {"sglang_response": response},
            }

    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.prefill_handler.native_generate_stream",
        native_results,
    )
    return handler


def _native_prefill_request(annotations):
    request = {
        "token_ids": [1, 2, 3],
        "routing": {},
        "extra_args": {"sglang_tito": {}},
    }
    if annotations is not None:
        request["annotations"] = annotations
    return {"request": request, "sampling_params": {}}


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "annotations",
    [None, ["prefill_output"], ["prefill_output", "prefill_output_first"]],
)
@pytest.mark.parametrize(
    "prompt_output",
    [
        {"input_token_logprobs": [[None, 1, None], [-0.5, 2, None], [-1.5, 3, None]]},
        {"routed_experts": "AAECAwQFBgc="},
        {"indexer_topk": [[0, 1], [1, 2], [2, 0]]},
    ],
    ids=["prompt_logprobs", "routed_experts", "indexer_topk"],
)
async def test_native_prefill_forwards_final_response_when_annotated(
    monkeypatch, annotations, prompt_output
):
    final_response = {
        "text": "",
        "output_ids": [4],
        "meta_info": {
            "id": "request-id",
            "finish_reason": {"type": "length", "length": 1},
            "prompt_tokens": 3,
            "cached_tokens": 2,
            **prompt_output,
        },
    }
    handler = _native_prefill_handler(monkeypatch, [final_response])
    context = SimpleNamespace(
        id=lambda: "request-id", trace_id=None, trace_headers=lambda: {}
    )

    chunks = await drain(
        handler.generate(_native_prefill_request(annotations), context)
    )

    assert len(chunks) == 2
    assert "disaggregated_params" in chunks[0]
    engine_data = {"kv_cache_hit": {"prompt_tokens": 3, "reused_tokens": 2}}
    if annotations:
        # One chunk: the prefill router keeps only the last chunk's engine_data.
        engine_data["sglang_response"] = final_response
    assert chunks[1] == {"token_ids": [], "engine_data": engine_data}
    if annotations:
        assert chunks[1]["engine_data"]["sglang_response"] is final_response
