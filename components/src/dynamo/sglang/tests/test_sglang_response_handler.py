# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the SGLang engine response handler."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]

_COMPONENTS_SRC = Path(__file__).resolve().parents[3]


def _load_response_handler():
    path = (
        _COMPONENTS_SRC
        / "dynamo"
        / "sglang"
        / "request_handlers"
        / "llm"
        / "response_handler.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_sglang_response_handler_impl", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


response_handler_module = _load_response_handler()


class FakeContext:
    def __init__(self, rid: str, *, stopped: bool = False) -> None:
        self.trace_id = rid
        self._rid = rid
        self._stopped = stopped
        self.stop_count = 0

    def id(self) -> str:
        return self._rid

    def is_stopped(self) -> bool:
        return self._stopped

    def stop_generating(self) -> None:
        self._stopped = True
        self.stop_count += 1


class FakeSender:
    batches: ClassVar[list[list[tuple[object, dict]]]] = []
    failed: ClassVar[list[int]] = []

    def __init__(self) -> None:
        self.sent = []
        self.close_count = 0
        self.errors = []

    @staticmethod
    def send_batch(items):
        FakeSender.batches.append(items)
        return FakeSender.failed

    def send(self, response) -> None:
        self.sent.append(response)

    def close(self) -> None:
        self.close_count += 1

    def close_with_error(self, message: str) -> None:
        self.errors.append(message)


class FakeDecodeHandler:
    def __init__(self, response_handler, *, text: bool = False) -> None:
        self.engine = SimpleNamespace(response_handler=response_handler)
        self.use_sglang_tokenizer = text
        self.config = SimpleNamespace(
            server_args=SimpleNamespace(
                incremental_streaming_output=True,
                served_model_name="test-model",
            ),
            dynamo_args=SimpleNamespace(enable_rl=False),
        )

    @staticmethod
    def _extract_logprobs(*args, **kwargs):
        return None, None


@dataclass
class FakeOutput:
    rid: str
    output: dict


def _entry(
    response_handler,
    rid: str,
    mode: str,
    *,
    sender: FakeSender | None = None,
    context: FakeContext | None = None,
):
    sender = sender or FakeSender()
    context = context or FakeContext(rid)
    handler = FakeDecodeHandler(response_handler, text=mode == "text")
    entry = response_handler_module._ResponseEntry(
        rid=rid,
        handler=handler,
        context=context,
        response_sender=sender,
        mode=mode,
    )
    response_handler.add(entry)
    return entry


@pytest.fixture(autouse=True)
def reset_sender_state():
    FakeSender.batches = []
    FakeSender.failed = []


def test_one_native_batch_call_converts_token_text_and_native_outputs():
    response_handler = response_handler_module.DynamoResponseHandler()
    token = _entry(response_handler, "token", "token")
    text = _entry(response_handler, "text", "text")
    native = _entry(response_handler, "native", "native")

    response_handler.handle_batch(
        [
            FakeOutput(
                "token",
                {
                    "output_ids": [7, 8],
                    "meta_info": {"id": "token", "finish_reason": None},
                },
            ),
            FakeOutput(
                "text",
                {
                    "text": "hello",
                    "meta_info": {"id": "text", "finish_reason": None},
                },
            ),
            FakeOutput(
                "native",
                {"output_ids": [9], "meta_info": {"id": "native"}},
            ),
        ]
    )

    assert len(FakeSender.batches) == 1
    assert len(FakeSender.batches[0]) == 3
    assert FakeSender.batches[0][0] == (
        token.response_sender,
        {"index": 0, "token_ids": [7, 8]},
    )
    assert FakeSender.batches[0][1][0] is text.response_sender
    assert FakeSender.batches[0][1][1]["choices"][0]["delta"]["content"] == "hello"
    assert FakeSender.batches[0][2] == (
        native.response_sender,
        {
            "token_ids": [],
            "engine_data": {
                "sglang_response": {
                    "output_ids": [9],
                    "meta_info": {"id": "native"},
                }
            },
        },
    )


def test_internal_probe_output_is_ignored():
    response_handler = response_handler_module.DynamoResponseHandler()

    response_handler.handle_batch([FakeOutput("internal-probe", {"output_ids": [1]})])

    assert FakeSender.batches == []


def test_failed_sender_is_removed_without_stopping_other_requests():
    response_handler = response_handler_module.DynamoResponseHandler()
    failed = _entry(response_handler, "failed", "token")
    active = _entry(response_handler, "active", "token")
    FakeSender.failed = [0]

    response_handler.handle_batch(
        [
            FakeOutput(
                "failed",
                {
                    "output_ids": [1],
                    "meta_info": {"id": "failed", "finish_reason": None},
                },
            ),
            FakeOutput(
                "active",
                {
                    "output_ids": [2],
                    "meta_info": {"id": "active", "finish_reason": None},
                },
            ),
        ]
    )

    assert response_handler.get("failed") is None
    assert response_handler.get("active") is active
    assert failed.context.stop_count == 1


def test_conversion_failure_closes_and_stops_only_that_request():
    response_handler = response_handler_module.DynamoResponseHandler()
    entry = _entry(response_handler, "bad", "token")

    response_handler.handle_batch(
        [
            FakeOutput(
                "bad",
                {
                    "output_ids": [1],
                    "meta_info": {"id": "bad", "finish_reason": {"type": "length"}},
                },
            )
        ]
    )

    assert response_handler.get("bad") is None
    assert entry.response_sender.errors
    assert entry.context.stop_count == 1
    assert FakeSender.batches == []


def test_terminal_iterator_closes_sender_and_removes_registry_entry():
    response_handler = response_handler_module.DynamoResponseHandler()
    entry = _entry(response_handler, "terminal", "token")
    terminal = {"token_ids": [3], "finish_reason": "length"}

    async def drive():
        async def stream():
            assert response_handler.get("terminal") is entry
            yield terminal

        driven = response_handler_module._drive_response_handler_stream(
            stream(), response_handler, entry
        )
        assert [item async for item in driven] == []

    asyncio.run(drive())

    assert entry.response_sender.sent == [terminal]
    assert entry.response_sender.close_count == 1
    assert response_handler.get("terminal") is None
