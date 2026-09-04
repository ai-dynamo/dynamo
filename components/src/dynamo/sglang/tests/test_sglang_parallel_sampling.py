# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``n > 1`` fan-out in SGLang disaggregated serving.

Regression coverage for ai-dynamo/dynamo#14098: a disaggregated request with
``n > 1`` must run as ``n`` single-sample sub-requests with one bootstrap room
each instead of hanging after prefill.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.llm import HttpError
from dynamo.sglang.parallel_sampling import (
    BOOTSTRAP_ROOMS_KEY,
    choice_request_ids,
    merge_choice_streams,
    raise_if_disagg_parallel_sampling,
    requested_parallel_samples,
    resolve_decode_bootstrap_rooms,
    resolve_prefill_bootstrap_rooms,
    single_sample_params,
)
from dynamo.sglang.protocol import (
    PreprocessedRequest,
    SamplingOptions,
    SglangMultimodalRequest,
    StopConditions,
)
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler
from dynamo.sglang.request_handlers.llm.prefill_handler import PrefillWorkerHandler
from dynamo.sglang.request_handlers.multimodal.worker_handler import (
    MultimodalWorkerHandler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.timeout(10),
]

BOOTSTRAP_HOST = "10.0.0.5"
BOOTSTRAP_PORT = 0  # never bound


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_requested_parallel_samples_defaults_to_one():
    assert requested_parallel_samples({}) == 1
    assert requested_parallel_samples({"n": 3}) == 3


def test_single_sample_params_copies_and_forces_one_sample():
    sampling_params = {"n": 3, "temperature": 0.7}

    single = single_sample_params(sampling_params)

    assert single == {"n": 1, "temperature": 0.7}
    assert sampling_params["n"] == 3


def test_choice_request_ids_keeps_sglang_ids_unique_per_choice():
    assert choice_request_ids(None, "ctx", 1) == [None]
    assert choice_request_ids("trace", "ctx", 1) == ["trace"]
    assert choice_request_ids("trace", "ctx", 2) == [
        "trace-choice-0",
        "trace-choice-1",
    ]
    # Siblings must be abortable by rid even when SGLang would assign one.
    assert choice_request_ids(None, "ctx", 2) == ["ctx-choice-0", "ctx-choice-1"]


def test_decode_rooms_single_sample_uses_bootstrap_room():
    assert resolve_decode_bootstrap_rooms({"bootstrap_room": 7}, 1) == [7]
    assert resolve_decode_bootstrap_rooms(
        {"bootstrap_room": 7, BOOTSTRAP_ROOMS_KEY: [7, 9]}, 1
    ) == [7]


def test_decode_rooms_fan_out_uses_per_choice_rooms():
    bootstrap_info = {"bootstrap_room": 8, BOOTSTRAP_ROOMS_KEY: [8, 16, 24]}

    assert resolve_decode_bootstrap_rooms(bootstrap_info, 3) == [8, 16, 24]


@pytest.mark.parametrize(
    "bootstrap_info",
    [
        # Older frontend: a single room only.
        {"bootstrap_room": 8},
        # Wrong cardinality, not integers, not distinct.
        {"bootstrap_room": 8, BOOTSTRAP_ROOMS_KEY: [8]},
        {"bootstrap_room": 8, BOOTSTRAP_ROOMS_KEY: [8, "16"]},
        {"bootstrap_room": 8, BOOTSTRAP_ROOMS_KEY: [8, 8]},
    ],
)
def test_decode_rooms_fan_out_rejects_unusable_room_lists(bootstrap_info):
    with pytest.raises(HttpError) as excinfo:
        resolve_decode_bootstrap_rooms(bootstrap_info, 2)

    assert excinfo.value.code == 400
    assert "n=2" in excinfo.value.message
    assert "Upgrade the Dynamo frontend" in excinfo.value.message


def test_prefill_rooms_single_sample_prefers_router_room():
    generate = iter([11, 12])

    assert resolve_prefill_bootstrap_rooms({"bootstrap_room": 5}, 1, lambda: 0) == [5]
    assert resolve_prefill_bootstrap_rooms(None, 1, lambda: next(generate)) == [11]
    assert resolve_prefill_bootstrap_rooms({}, 1, lambda: next(generate)) == [12]


def test_prefill_rooms_fan_out_uses_router_rooms():
    bootstrap_info = {"bootstrap_room": 8, BOOTSTRAP_ROOMS_KEY: [8, 16]}

    assert resolve_prefill_bootstrap_rooms(bootstrap_info, 2, lambda: 0) == [8, 16]


def test_prefill_rooms_fan_out_rejects_single_router_room():
    with pytest.raises(HttpError) as excinfo:
        resolve_prefill_bootstrap_rooms({"bootstrap_room": 8}, 2, lambda: 0)

    assert excinfo.value.code == 400
    assert "single bootstrap room" in excinfo.value.message


def test_prefill_rooms_fan_out_draws_distinct_local_rooms_without_router_room():
    # The worker owns room generation when the frontend sent none; a repeated
    # draw must not produce two choices on the same room.
    draws = iter([7, 7, 9])

    rooms = resolve_prefill_bootstrap_rooms(None, 2, lambda: next(draws))

    assert rooms == [7, 9]


def test_raise_if_disagg_parallel_sampling_only_rejects_multiple_samples():
    raise_if_disagg_parallel_sampling({})
    raise_if_disagg_parallel_sampling({"n": 1})

    with pytest.raises(HttpError) as excinfo:
        raise_if_disagg_parallel_sampling({"n": 2})

    assert excinfo.value.code == 400
    assert "n=2" in excinfo.value.message


@pytest.mark.asyncio
async def test_multimodal_decode_rejects_parallel_sampling_before_dispatch():
    handler = MultimodalWorkerHandler.__new__(MultimodalWorkerHandler)
    handler.serving_mode = DisaggregationMode.DECODE
    request = SglangMultimodalRequest(
        request=PreprocessedRequest(
            token_ids=[1, 2, 3],
            stop_conditions=StopConditions(max_tokens=8),
            sampling_options=SamplingOptions(n=2),
        )
    )

    with pytest.raises(HttpError) as excinfo:
        await _collect(handler.generate(request, _context()))

    assert excinfo.value.code == 400


# ---------------------------------------------------------------------------
# merge_choice_streams
# ---------------------------------------------------------------------------


async def _collect(stream):
    return [item async for item in stream]


@pytest.mark.asyncio
async def test_merge_choice_streams_tags_items_with_their_choice_index():
    async def choice(items, pauses):
        for item in items:
            for _ in range(pauses):
                await asyncio.sleep(0)
            yield item

    aborted: list[int] = []
    merged = await _collect(
        merge_choice_streams(
            [choice(["a0", "a1", "a2"], 3), choice(["b0", "b1"], 1)], aborted.append
        )
    )

    assert aborted == []

    assert sorted(merged) == [
        (0, "a0"),
        (0, "a1"),
        (0, "a2"),
        (1, "b0"),
        (1, "b1"),
    ]
    # Per-choice order is preserved, and the faster stream is not held back
    # behind the slower one.
    assert [item for index, item in merged if index == 0] == ["a0", "a1", "a2"]
    assert [item for index, item in merged if index == 1] == ["b0", "b1"]
    assert merged.index((1, "b1")) < merged.index((0, "a2"))


@pytest.mark.asyncio
async def test_merge_choice_streams_propagates_failure_and_aborts_siblings():
    aborted: list[int] = []
    aborted_before_sibling_closed: bool | None = None

    async def endless():
        nonlocal aborted_before_sibling_closed
        try:
            while True:
                await asyncio.sleep(0)
                yield "tick"
        finally:
            aborted_before_sibling_closed = 0 in aborted

    async def failing():
        yield "first"
        raise RuntimeError("choice 1 failed")

    received = []
    with pytest.raises(RuntimeError, match="choice 1 failed"):
        async for item in merge_choice_streams([endless(), failing()], aborted.append):
            received.append(item)

    assert (1, "first") in received
    # Every unfinished sub-request is aborted, the failed one included, and the
    # abort lands before the sibling is torn down (SGLang forgets a cancelled
    # request, so a later abort would be a no-op).
    assert sorted(aborted) == [0, 1]
    assert aborted_before_sibling_closed is True


@pytest.mark.asyncio
async def test_merge_choice_streams_close_cancels_every_sibling():
    closed = [asyncio.Event(), asyncio.Event()]
    started = [asyncio.Event(), asyncio.Event()]

    async def choice(index):
        started[index].set()
        try:
            while True:
                await asyncio.sleep(0)
                yield index
        finally:
            closed[index].set()

    aborted: list[int] = []
    merged = merge_choice_streams([choice(0), choice(1)], aborted.append)
    first = await merged.__anext__()
    assert first[1] in (0, 1)
    await merged.aclose()

    assert all(event.is_set() for event in started)
    assert sorted(aborted) == [0, 1]
    assert all(event.is_set() for event in closed)


# ---------------------------------------------------------------------------
# Handler fan-out
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Records ``async_generate`` calls and replays canned output per room."""

    def __init__(self, outputs_by_room: dict[int, list[Any]]):
        self.outputs_by_room = outputs_by_room
        self.calls: list[dict[str, Any]] = []
        self.aborted: list[str] = []
        self.tokenizer_manager = SimpleNamespace(
            abort_request=lambda rid, abort_all=False: self.aborted.append(rid)
        )

    async def async_generate(self, **kwargs):
        self.calls.append(kwargs)
        return self._replay(kwargs["bootstrap_room"])

    async def _replay(self, room: int):
        for output in self.outputs_by_room[room]:
            await asyncio.sleep(0)
            if isinstance(output, Exception):
                raise output
            yield dict(output, meta_info=dict(output["meta_info"]))


def _context(trace_id: str | None = "trace"):
    return SimpleNamespace(
        id=lambda: "context-id",
        trace_id=trace_id,
        is_stopped=lambda: False,
        notify_first_token=lambda: None,
        trace_headers=lambda: None,
    )


@asynccontextmanager
async def _no_cancellation_monitor(*_args, **_kwargs):
    yield None


def _new_disagg_decode_handler(engine, *, use_sglang_tokenizer: bool = False):
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.engine = engine
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(served_model_name="test-model"),
        dynamo_args=SimpleNamespace(enable_rl=False),
    )
    handler.serving_mode = DisaggregationMode.DECODE
    handler.use_sglang_tokenizer = use_sglang_tokenizer
    handler.enable_trace = False
    handler.shutdown_event = None
    handler._first_token_source = None
    handler._routed_experts_kwargs = {}
    handler._engine_supports_priority = False
    handler.lora_id_for_name = {}
    handler._cancellation_monitor = _no_cancellation_monitor
    handler._get_input_param = lambda request: (
        {"prompt": "hi"}
        if use_sglang_tokenizer
        else {"input_ids": request["token_ids"]}
    )
    return handler


def _new_prefill_handler(engine, room_draws=None):
    handler = PrefillWorkerHandler.__new__(PrefillWorkerHandler)
    handler.engine = engine
    handler.bootstrap_host = BOOTSTRAP_HOST
    handler.bootstrap_port = BOOTSTRAP_PORT
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(served_model_name="test-model"),
        dynamo_args=SimpleNamespace(enable_rl=False),
    )
    handler.serving_mode = DisaggregationMode.PREFILL
    handler.use_sglang_tokenizer = False
    handler.enable_trace = False
    handler.shutdown_event = None
    handler._engine_supports_priority = False
    handler.lora_id_for_name = {}
    handler._consume_tasks = set()
    handler._cancellation_monitor = _no_cancellation_monitor
    handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
    if room_draws is not None:
        draws = iter(room_draws)
        handler._generate_bootstrap_room = lambda: next(draws)
    return handler


def _token_outputs(rid: str, tokens: list[int]) -> list[dict[str, Any]]:
    outputs = []
    for position, token in enumerate(tokens):
        final = position == len(tokens) - 1
        outputs.append(
            {
                "output_ids": [token],
                "meta_info": {
                    "id": rid,
                    "finish_reason": {"type": "stop"} if final else None,
                    **(
                        {"prompt_tokens": 3, "completion_tokens": len(tokens)}
                        if final
                        else {}
                    ),
                },
            }
        )
    return outputs


def _text_outputs(rid: str, texts: list[str]) -> list[dict[str, Any]]:
    outputs = []
    for position, text in enumerate(texts):
        final = position == len(texts) - 1
        outputs.append(
            {
                "text": text,
                "output_ids": [position],
                "meta_info": {
                    "id": rid,
                    "finish_reason": {"type": "stop"} if final else None,
                },
            }
        )
    return outputs


def _token_request(n: int | None, bootstrap_info: dict[str, Any]) -> dict[str, Any]:
    sampling_options: dict[str, Any] = {"temperature": 0.5}
    if n is not None:
        sampling_options["n"] = n
    return {
        "token_ids": [1, 2, 3],
        "sampling_options": sampling_options,
        "stop_conditions": {"max_tokens": 8},
        "output_options": {},
        "routing": {"dp_rank": 2},
        "bootstrap_info": bootstrap_info,
    }


def _fan_out_bootstrap_info(rooms: list[int]) -> dict[str, Any]:
    return {
        "bootstrap_host": BOOTSTRAP_HOST,
        "bootstrap_port": BOOTSTRAP_PORT,
        "bootstrap_room": rooms[0],
        BOOTSTRAP_ROOMS_KEY: rooms,
    }


@pytest.mark.asyncio
async def test_decode_fans_out_parallel_sampling_per_bootstrap_room():
    engine = _FakeEngine(
        {
            100: _token_outputs("trace-choice-0", [11, 12, 13]),
            200: _token_outputs("trace-choice-1", [21, 22]),
        }
    )
    handler = _new_disagg_decode_handler(engine)

    outputs = await _collect(
        handler.generate(
            _token_request(2, _fan_out_bootstrap_info([100, 200])), _context()
        )
    )

    # One n=1 sub-request per room, each with its own SGLang request id.
    assert [call["bootstrap_room"] for call in engine.calls] == [100, 200]
    assert [call["rid"] for call in engine.calls] == [
        "trace-choice-0",
        "trace-choice-1",
    ]
    for call in engine.calls:
        assert call["sampling_params"]["n"] == 1
        assert call["sampling_params"]["temperature"] == 0.5
        assert call["bootstrap_host"] == BOOTSTRAP_HOST
        assert call["bootstrap_port"] == BOOTSTRAP_PORT
        assert call["data_parallel_rank"] == 2
        assert call["input_ids"] == [1, 2, 3]

    # Sub-streams are merged back by choice index with per-choice order kept.
    by_choice: dict[int, list[int]] = {}
    for out in outputs:
        by_choice.setdefault(out["index"], []).extend(out["token_ids"])
    assert by_choice == {0: [11, 12, 13], 1: [21, 22]}

    finishing = [out for out in outputs if out.get("finish_reason")]
    assert {out["index"] for out in finishing} == {0, 1}
    # Usage reports the running request-wide total, like the single-stream
    # n > 1 path, so the last finishing chunk carries all completion tokens.
    assert finishing[-1]["completion_usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }
    assert finishing[0]["completion_usage"]["completion_tokens"] in (2, 3)
    assert engine.aborted == []


@pytest.mark.asyncio
async def test_decode_fan_out_aborts_siblings_when_one_choice_fails():
    engine = _FakeEngine(
        {
            100: _token_outputs("trace-choice-0", list(range(100))),
            200: [
                _token_outputs("trace-choice-1", [21, 22])[0],
                RuntimeError("choice 1 failed"),
            ],
        }
    )
    handler = _new_disagg_decode_handler(engine)

    with pytest.raises(RuntimeError, match="choice 1 failed"):
        await _collect(
            handler.generate(
                _token_request(2, _fan_out_bootstrap_info([100, 200])), _context()
            )
        )

    # The surviving sub-request would otherwise decode to max_tokens with no
    # consumer; both are aborted in SGLang by the rids the handler assigned.
    assert sorted(engine.aborted) == ["trace-choice-0", "trace-choice-1"]


@pytest.mark.asyncio
async def test_decode_fan_out_merges_text_chunks_under_one_response_id():
    engine = _FakeEngine(
        {
            100: _text_outputs("sglang-0", ["Hel", "Hello"]),
            200: _text_outputs("sglang-1", ["Bye"]),
        }
    )
    handler = _new_disagg_decode_handler(engine, use_sglang_tokenizer=True)
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "n": 2,
        "max_tokens": 8,
        "bootstrap_info": _fan_out_bootstrap_info([100, 200]),
    }

    outputs = await _collect(handler.generate(request, _context()))

    assert [call["bootstrap_room"] for call in engine.calls] == [100, 200]
    assert all(call["sampling_params"]["n"] == 1 for call in engine.calls)
    assert {out["id"] for out in outputs} == {"trace"}
    deltas: dict[int, str] = {}
    for out in outputs:
        (choice,) = out["choices"]
        deltas[choice["index"]] = (
            deltas.get(choice["index"], "") + choice["delta"]["content"]
        )
    assert deltas == {0: "Hello", 1: "Bye"}


@pytest.mark.asyncio
async def test_decode_rejects_parallel_sampling_from_older_frontend():
    engine = _FakeEngine({})
    handler = _new_disagg_decode_handler(engine)
    bootstrap_info = {
        "bootstrap_host": BOOTSTRAP_HOST,
        "bootstrap_port": BOOTSTRAP_PORT,
        "bootstrap_room": 100,
    }

    with pytest.raises(HttpError) as excinfo:
        await _collect(handler.generate(_token_request(2, bootstrap_info), _context()))

    assert excinfo.value.code == 400
    assert engine.calls == []


@pytest.mark.asyncio
async def test_decode_single_sample_keeps_pre_fan_out_wire_shape():
    engine = _FakeEngine({100: _token_outputs("trace", [11])})
    handler = _new_disagg_decode_handler(engine)
    bootstrap_info = {
        "bootstrap_host": BOOTSTRAP_HOST,
        "bootstrap_port": BOOTSTRAP_PORT,
        "bootstrap_room": 100,
    }

    outputs = await _collect(
        handler.generate(_token_request(None, bootstrap_info), _context())
    )

    (call,) = engine.calls
    assert call["bootstrap_room"] == 100
    assert call["rid"] == "trace"
    assert "n" not in call["sampling_params"]
    assert outputs == [
        {
            "index": 0,
            "finish_reason": "stop",
            "token_ids": [11],
            "completion_usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
            },
        }
    ]


@pytest.mark.asyncio
async def test_prefill_fans_out_parallel_sampling_with_router_rooms():
    engine = _FakeEngine(
        {
            100: _token_outputs("trace-choice-0", [11]),
            200: _token_outputs("trace-choice-1", [21]),
        }
    )
    handler = _new_prefill_handler(engine)

    outputs = await _collect(
        handler.generate(
            _token_request(2, _fan_out_bootstrap_info([100, 200])), _context()
        )
    )

    assert outputs == [
        {
            "token_ids": [],
            "text": None,
            "finish_reason": None,
            "disaggregated_params": _fan_out_bootstrap_info([100, 200]),
        }
    ]
    assert [call["bootstrap_room"] for call in engine.calls] == [100, 200]
    assert [call["rid"] for call in engine.calls] == [
        "trace-choice-0",
        "trace-choice-1",
    ]
    for call in engine.calls:
        assert call["sampling_params"]["n"] == 1
        assert call["sampling_params"]["max_new_tokens"] == 1
        assert call["sampling_params"]["temperature"] == 0.5
    # Every sub-request was consumed so its handoff completed.
    assert handler._consume_tasks == set()


@pytest.mark.asyncio
async def test_prefill_rejects_parallel_sampling_from_older_frontend():
    engine = _FakeEngine({})
    handler = _new_prefill_handler(engine)
    bootstrap_info = {
        "bootstrap_host": BOOTSTRAP_HOST,
        "bootstrap_port": BOOTSTRAP_PORT,
        "bootstrap_room": 100,
    }

    with pytest.raises(HttpError) as excinfo:
        await _collect(handler.generate(_token_request(2, bootstrap_info), _context()))

    assert excinfo.value.code == 400
    assert engine.calls == []


@pytest.mark.asyncio
async def test_prefill_draws_local_rooms_when_frontend_sent_none():
    engine = _FakeEngine(
        {
            7: _token_outputs("trace-choice-0", [11]),
            9: _token_outputs("trace-choice-1", [21]),
        }
    )
    handler = _new_prefill_handler(engine, room_draws=[7, 7, 9])
    request = _token_request(2, {})
    del request["bootstrap_info"]

    outputs = await _collect(handler.generate(request, _context()))

    assert outputs[0]["disaggregated_params"] == {
        "bootstrap_host": BOOTSTRAP_HOST,
        "bootstrap_port": BOOTSTRAP_PORT,
        "bootstrap_room": 7,
        BOOTSTRAP_ROOMS_KEY: [7, 9],
    }
    assert [call["bootstrap_room"] for call in engine.calls] == [7, 9]


@pytest.mark.asyncio
async def test_prefill_single_sample_keeps_pre_fan_out_wire_shape():
    engine = _FakeEngine({100: _token_outputs("trace", [11])})
    handler = _new_prefill_handler(engine)
    bootstrap_info = {
        "bootstrap_host": BOOTSTRAP_HOST,
        "bootstrap_port": BOOTSTRAP_PORT,
        "bootstrap_room": 100,
    }

    outputs = await _collect(
        handler.generate(_token_request(None, bootstrap_info), _context())
    )

    assert outputs[0]["disaggregated_params"] == bootstrap_info
    assert BOOTSTRAP_ROOMS_KEY not in outputs[0]["disaggregated_params"]
    (call,) = engine.calls
    assert call["rid"] == "trace"
    assert call["sampling_params"] == {"n": 1, "max_new_tokens": 1, "temperature": 0.5}


@pytest.mark.asyncio
async def test_prefill_wrapped_request_reads_parallel_sampling_from_sampling_params():
    engine = _FakeEngine(
        {
            100: _token_outputs("trace-choice-0", [11]),
            200: _token_outputs("trace-choice-1", [21]),
        }
    )
    handler = _new_prefill_handler(engine)
    request = {
        "request": {
            "token_ids": [1, 2, 3],
            "bootstrap_info": _fan_out_bootstrap_info([100, 200]),
        },
        "sampling_params": {"n": 2, "max_new_tokens": 8},
    }

    outputs = await _collect(handler.generate(request, _context()))

    assert outputs[0]["disaggregated_params"][BOOTSTRAP_ROOMS_KEY] == [100, 200]
    assert [call["bootstrap_room"] for call in engine.calls] == [100, 200]
    assert all(call["sampling_params"]["n"] == 1 for call in engine.calls)
