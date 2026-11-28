# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation-artifact hidden-state capture through the SGLang decode handler."""

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("zstandard")

from dynamo.artifacts.vllm import (  # noqa: E402 -- optional dependency guards
    capture as vllm_capture,
)
from dynamo.artifacts.vllm.capture import CaptureBudget  # noqa: E402
from dynamo.artifacts.vllm.format import decode_generation_artifact  # noqa: E402
from dynamo.artifacts.vllm.hidden_states import HiddenStateProfile  # noqa: E402
from dynamo.common.constants import DisaggregationMode  # noqa: E402
from dynamo.llm.exceptions import InvalidArgument, WorkerOverloaded  # noqa: E402
from dynamo.sglang.tests.test_sglang_decode_handler import (  # noqa: E402
    _new_decode_handler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]

WIDTH = 8
PROMPT = [10, 11, 12]


def _bf16_values(rows):
    values = np.random.default_rng(0).standard_normal((rows, WIDTH))
    bits = values.astype(np.float32).view(np.uint32) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def _handler(*, profile=True, **kwargs):
    handler = _new_decode_handler(**kwargs)
    handler._enable_frontend_decoding = False
    handler._mm_hashes_supported = False
    handler._engine_supports_priority = False
    handler._routed_experts_kwargs = {}
    handler.enable_trace = False
    handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
    handler._resolve_lora = lambda request: None
    handler.artifact_hidden_state_profile = (
        HiddenStateProfile((3, 6), 4, "bf16", storage_root="") if profile else None
    )
    return handler


def _request(options=None, output_options=None, contents=("hidden_states",)):
    settings = {
        "kind": "generation_artifact",
        "contents": list(contents),
        "mode": "inline_base64",
    }
    if options is not None:
        settings["content_options"] = {"hidden_states": options}
    request = {
        "token_ids": list(PROMPT),
        "stop_conditions": {"max_tokens": 2},
        "extra_args": {"nvext": {"metadata_upload": settings}},
    }
    if output_options is not None:
        request["output_options"] = output_options
    return request


def _engine_output(values, *, finish="stop", retractions=0):
    async def output():
        # SGLang re-sends every row so far with each output.
        yield {
            "output_ids": [13],
            "meta_info": {
                "id": "r",
                "finish_reason": None,
                "hidden_states": [values[:3].tolist()],
                "output_token_logprobs": [(-0.5, 13, None)],
            },
        }
        yield {
            "output_ids": [14],
            "meta_info": {
                "id": "r",
                "finish_reason": {"type": finish},
                "hidden_states": [values[:3].tolist(), values[3].tolist()],
                "num_retractions": retractions,
                "output_token_logprobs": [(-0.25, 14, None)],
            },
        }

    return output()


def _context():
    return SimpleNamespace(
        id=lambda: "request-id",
        trace_id="trace-id",
        is_stopped=lambda: False,
        notify_first_token=lambda: None,
    )


async def _collect(stream):
    return [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_capture_returns_the_rows_and_hides_forced_logprobs():
    handler = _handler()
    values = _bf16_values(4)
    handler.engine = SimpleNamespace(
        async_generate=AsyncMock(return_value=_engine_output(values))
    )

    chunks = await _collect(handler.generate(_request(), _context()))

    kwargs = handler.engine.async_generate.await_args.kwargs
    assert kwargs["return_hidden_states"] is True
    # Only the first token and the finish carry the rows.
    assert kwargs["sampling_params"]["stream_interval"] == 3
    # Caps the prefix-cache match so every requested row is computed.
    assert kwargs["return_logprob"] is True
    assert kwargs["logprob_start_len"] == 0

    assert [chunk["token_ids"] for chunk in chunks] == [[13], [14]]
    assert all("log_probs" not in chunk for chunk in chunks)
    receipt = chunks[-1]["engine_data"]["metadata_upload"]
    data = base64.b64decode(receipt["data_base64"])
    choice = decode_generation_artifact(data).choices[0]
    assert choice.sequence_token_ids.as_array().tolist() == [10, 11, 12, 13, 14]
    (component,) = choice.components
    rows = np.frombuffer(component.hidden_states.data, dtype=np.uint16)
    assert rows.tobytes() == (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()


@pytest.mark.asyncio
async def test_client_logprobs_and_token_start_keep_their_meaning():
    handler = _handler()
    handler.engine = SimpleNamespace(
        async_generate=AsyncMock(return_value=_engine_output(_bf16_values(4)))
    )

    chunks = await _collect(
        handler.generate(
            _request(options={"token_start": 3}, output_options={"logprobs": 0}),
            _context(),
        )
    )

    kwargs = handler.engine.async_generate.await_args.kwargs
    # Cache reuse up to token_start is kept.
    assert kwargs["logprob_start_len"] == 3
    assert [chunk["log_probs"] for chunk in chunks] == [[-0.5], [-0.25]]
    receipt = chunks[-1]["engine_data"]["metadata_upload"]
    (component,) = (
        decode_generation_artifact(base64.b64decode(receipt["data_base64"]))
        .choices[0]
        .components
    )
    assert component.token_start == 3


@pytest.mark.asyncio
async def test_retracted_request_reports_an_invalid_capture():
    handler = _handler()
    handler.engine = SimpleNamespace(
        async_generate=AsyncMock(
            return_value=_engine_output(_bf16_values(4), retractions=1)
        )
    )

    chunks = await _collect(handler.generate(_request(), _context()))

    receipt = chunks[-1]["engine_data"]["metadata_upload"]
    assert (receipt["state"], receipt["error_code"]) == ("failed", "capture_invalid")


@pytest.mark.asyncio
async def test_cancelled_request_releases_capture_without_a_receipt(monkeypatch):
    budget = CaptureBudget(1 << 24)
    monkeypatch.setattr(vllm_capture, "process_capture_budget", lambda: budget)
    handler = _handler()
    handler.engine = SimpleNamespace(
        async_generate=AsyncMock(
            return_value=_engine_output(_bf16_values(4), finish="abort")
        )
    )

    chunks = await _collect(handler.generate(_request(), _context()))

    assert "metadata_upload" not in chunks[-1].get("engine_data", {})
    assert budget.used == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler_kwargs, request_update, match",
    [
        (
            {"serving_mode": DisaggregationMode.DECODE},
            {},
            "aggregated token-mode",
        ),
        ({"use_sglang_tokenizer": True}, {}, "aggregated token-mode"),
        ({"profile": False}, {}, "--enable-return-aux-hidden-states"),
        ({}, {"stop_conditions": {}}, "max_tokens"),
        ({}, {"multi_modal_data": {"image_url": ["x"]}}, "exact token prompt"),
    ],
)
async def test_unsupported_requests_fail_before_inference(
    handler_kwargs, request_update, match
):
    handler = _handler(**handler_kwargs)
    handler.engine = SimpleNamespace(async_generate=AsyncMock())
    request = {**_request(), **request_update}
    if handler_kwargs.get("serving_mode") == DisaggregationMode.DECODE:
        request["bootstrap_info"] = {
            "bootstrap_host": "prefill.invalid",
            "bootstrap_port": 1234,
            "bootstrap_room": 1,
        }

    with pytest.raises(InvalidArgument, match=match):
        await _collect(handler.generate(request, _context()))
    handler.engine.async_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_other_contents_are_rejected_for_sglang():
    handler = _handler()
    handler.engine = SimpleNamespace(async_generate=AsyncMock())
    request = _request(contents=("hidden_states", "selected_logprobs"))

    with pytest.raises(InvalidArgument, match="hidden_states only"):
        await _collect(handler.generate(request, _context()))


@pytest.mark.asyncio
async def test_native_generate_rejects_artifact_capture():
    handler = _handler()
    handler.engine = SimpleNamespace(async_generate=AsyncMock())
    request = _request()
    request["extra_args"]["sglang_tito"] = {"sampling_params": {"max_new_tokens": 2}}

    with pytest.raises(InvalidArgument, match="native SGLang Generate"):
        await _collect(handler.generate(request, _context()))


@pytest.mark.asyncio
async def test_full_capture_budget_maps_to_overloaded(monkeypatch):
    monkeypatch.setattr(
        vllm_capture, "process_capture_budget", lambda: CaptureBudget(1)
    )
    handler = _handler()
    handler.engine = SimpleNamespace(async_generate=AsyncMock())

    with pytest.raises(WorkerOverloaded):
        await _collect(handler.generate(_request(), _context()))
