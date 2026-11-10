# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-scoped NVTX ranges around the SGLang engine call.

``worker.handler.prepare:<role>`` must close right before ``async_generate`` and
``worker.engine.first_output:<role>`` must open there, carry the SGLang ``rid`` as
``engine_rid`` and close on the first output or on any early exit. The NVTX
binding is replaced by a recorder, so no profiler or built ``dynamo._core`` is
needed.
"""

from contextlib import aclosing
from types import SimpleNamespace
from typing import Any

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils import nvtx_utils
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler
from dynamo.sglang.request_handlers.llm.prefill_handler import PrefillWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]

_ENGINE_RID = "engine-rid"


class _Recorder:
    def __init__(self) -> None:
        self.log: list[tuple] = []

    def mark(self, stage, request_id, attrs=None):
        self.log.append(("mark", stage, request_id, attrs))

    def range(self, stage, request_id, attrs=None):
        self.log.append(("start", stage, request_id, attrs))
        log = self.log

        class _Range:
            def end(self_inner):  # noqa: N805
                log.append(("end", stage, request_id))

        return _Range()


@pytest.fixture
def nvtx_log(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(nvtx_utils, "ENABLED", True)
    monkeypatch.setattr(nvtx_utils, "_core_api", (rec.mark, rec.range))
    return rec.log


def _context():
    return SimpleNamespace(
        id=lambda: "request-id",
        trace_id="trace-id",
        trace_headers=lambda: None,
        is_stopped=lambda: False,
        notify_first_token=lambda: None,
    )


class _Engine:
    """async_generate snapshots the NVTX log at call time."""

    def __init__(self, log: list, chunks: list[Any]) -> None:
        self.log = log
        self.chunks = chunks
        self.log_at_call: list[tuple] | None = None

    async def async_generate(self, **kwargs: Any):
        self.log_at_call = list(self.log)

        async def stream():
            for chunk in self.chunks:
                yield chunk

        return stream()


def _stub_common(handler: Any) -> None:
    handler._get_input_param = lambda request: {"input_ids": [1, 2, 3]}
    handler._resolve_lora = lambda request: None
    handler._priority_kwargs = lambda priority: {}
    handler.enable_trace = False


def _decode_handler(engine: Any, mode: DisaggregationMode, monkeypatch):
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.decode_handler.new_sglang_request_id",
        lambda: _ENGINE_RID,
    )
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.engine = engine
    handler.serving_mode = mode
    handler.use_sglang_tokenizer = False
    handler._first_token_source = None
    handler._enable_frontend_decoding = False
    handler._mm_hashes_supported = False
    handler._routed_experts_kwargs = {}
    _stub_common(handler)
    handler._build_sampling_params = lambda request: {"max_new_tokens": 1}
    handler._build_logprob_kwargs = lambda request: {}
    handler._metadata_uploader_from_request = lambda request: None

    async def passthrough(stream, context, *args, **kwargs):
        async for chunk in stream:
            yield chunk

    handler._process_token_stream = passthrough
    return handler


_DISAGG_REQUEST = {
    "bootstrap_info": {
        "bootstrap_host": "127.0.0.1",
        "bootstrap_port": 8998,
        "bootstrap_room": 42,
    }
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,role,request_body",
    [
        (DisaggregationMode.AGGREGATED, "aggregated", {}),
        (DisaggregationMode.DECODE, "decode", _DISAGG_REQUEST),
    ],
)
async def test_decode_handler_ranges_wrap_engine_call(
    nvtx_log, monkeypatch, mode, role, request_body
):
    engine = _Engine(nvtx_log, [{"token_ids": [1]}, {"token_ids": [2]}])
    handler = _decode_handler(engine, mode, monkeypatch)
    prepare = f"worker.handler.prepare:{role}"
    first = f"worker.engine.first_output:{role}"

    stream = handler.generate(dict(request_body), _context())
    async with aclosing(stream):
        await anext(stream)
        # prepare has closed and first_output has opened by the engine call.
        assert engine.log_at_call == [
            ("start", prepare, "request-id", None),
            ("end", prepare, "request-id"),
            ("start", first, "request-id", {"engine_rid": _ENGINE_RID}),
        ]
        # By the first yielded chunk, first_output has ended exactly once.
        assert nvtx_log.count(("end", first, "request-id")) == 1

    assert nvtx_log == [
        ("start", prepare, "request-id", None),
        ("end", prepare, "request-id"),
        ("start", first, "request-id", {"engine_rid": _ENGINE_RID}),
        ("end", first, "request-id"),
    ]


@pytest.mark.asyncio
async def test_decode_handler_closes_prepare_when_setup_fails(nvtx_log, monkeypatch):
    engine = _Engine(nvtx_log, [])
    handler = _decode_handler(engine, DisaggregationMode.DECODE, monkeypatch)

    # Disaggregated decode without bootstrap_info raises before the engine call.
    with pytest.raises(RuntimeError, match="bootstrap_info"):
        async for _ in handler.generate({}, _context()):
            pass

    assert engine.log_at_call is None
    assert nvtx_log == [
        ("start", "worker.handler.prepare:decode", "request-id", None),
        ("end", "worker.handler.prepare:decode", "request-id"),
    ]


@pytest.mark.asyncio
async def test_prefill_handler_ranges_end_on_early_close(nvtx_log, monkeypatch):
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.prefill_handler.new_sglang_request_id",
        lambda: _ENGINE_RID,
    )
    engine = _Engine(nvtx_log, [])
    handler = PrefillWorkerHandler.__new__(PrefillWorkerHandler)
    handler.engine = engine
    handler.config = SimpleNamespace(server_args=SimpleNamespace())
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 8998
    handler._generate_bootstrap_room = lambda: 42
    _stub_common(handler)

    stream = handler.generate(
        {"request": {}, "sampling_params": {"max_new_tokens": 16}}, _context()
    )
    async with aclosing(stream):
        # Bootstrap info is yielded before any engine output exists. The engine stream
        # is lazy, so first_output opens only once consumption starts, which an early
        # close after the bootstrap never reaches.
        assert "disaggregated_params" in await anext(stream)

    prepare = "worker.handler.prepare:prefill"
    assert nvtx_log == [
        ("start", prepare, "request-id", None),
        ("end", prepare, "request-id"),
    ]
