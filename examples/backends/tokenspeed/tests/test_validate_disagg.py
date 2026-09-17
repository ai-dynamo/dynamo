# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline validator regressions with HTTP, tokenization, and discovery mocked."""

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge, pytest.mark.timeout(10)]


@pytest.fixture(params=[0, 2], ids=["normal", "optimized"])
def validator(request, monkeypatch):
    """Execute the actual script with normal and python -OO compilation semantics."""
    path = Path(__file__).parents[1] / "validate_disagg.py"
    module = ModuleType("tokenspeed_disagg_validator")
    exec(compile(path.read_text(), str(path), "exec", optimize=request.param), module.__dict__)
    clock = 0.0

    async def sleep(delay):
        nonlocal clock
        clock += delay

    # Advance readiness deadlines without sleeping or changing shared modules.
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock))
    monkeypatch.setattr(module, "asyncio", SimpleNamespace(sleep=sleep, get_running_loop=module.asyncio.get_running_loop))
    return module


async def test_readiness_retries_closed_port_and_unregistered_model(validator):
    """A closed port and temporary HTTP failure do not skip the readiness budget."""
    calls = 0

    async def respond(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("port closed", request=request)
        if calls == 2:
            return httpx.Response(503, text="starting")
        return httpx.Response(200, json={"data": [{"id": "longcat-flash"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        await validator.wait_for_frontend(http, "http://frontend", "longcat-flash", timeout=5)
    assert calls == 3


async def test_readiness_timeout_retains_transport_error(validator):
    """An unavailable frontend fails with the last connection error at the deadline."""
    async def respond(request):
        raise httpx.ConnectError("port closed", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises(RuntimeError, match="ConnectError: port closed"):
            await validator.wait_for_frontend(http, "http://frontend", "longcat-flash", timeout=0.01)


@pytest.mark.parametrize(
    "fault,message",
    [
        (None, None),
        ("wrong-text", "incorrect generation"),
        ("missing-done", "Incomplete"),
        ("unknown-prefill", "Unexpected prefill"),
        ("missing-decode", "Missing decode"),
        ("forced-prefill", "Forced prefill"),
        ("cold-overlap", "Cold prefix"),
        ("wrong-reuse", "Cached-prefix owner"),
    ],
)
async def test_full_validation_cannot_pass_bad_deployment(
    validator, monkeypatch, tmp_path, fault, message
):
    """Every safety check remains active under optimization, with a passing control."""
    topics = ["ORANGE", "PURPLE", "SILVER"]
    shutdown = Mock()

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            content = messages[0]["content"]
            return [next(i for i, word in enumerate(topics) if word in content)] * 128

    async def client():
        return SimpleNamespace(instance_ids=lambda: [1, 2])

    runtime = SimpleNamespace(
        endpoint=lambda _: SimpleNamespace(client=client), shutdown=shutdown
    )

    class Router:
        async def get_overlap_scores(self, tokens):
            owner = tokens[0] + 1
            return {
                "workers": [
                    {
                        "worker_id": worker,
                        "device_blocks": 2 if worker == owner or (owner == 3 and fault == "cold-overlap") else 0,
                    }
                    for worker in [1, 2]
                ]
            }

    class Stream:
        status_code = 200

        def __init__(self, payload, headers):
            index = payload["prompt"][0]
            forced = headers.get("x-dynamo-prefill-instance-id")
            owner = int(forced) if forced else min(index + 1, 2)
            if fault == "unknown-prefill":
                owner = 99
            elif fault == "forced-prefill" and forced:
                owner = 3 - int(forced)
            elif fault == "wrong-reuse" and not forced:
                owner = 3 - owner
            self.chunk = {
                "choices": [{"text": "WRONG" if fault == "wrong-text" else topics[index]}],
                "nvext": {
                    "worker_id": {
                        "prefill_worker_id": owner,
                        "decode_worker_id": None if fault == "missing-decode" else 3,
                    }
                },
            }

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def aiter_lines(self):
            yield "data: " + json.dumps(self.chunk)
            if fault != "missing-done":
                yield "data: [DONE]"

    class Http:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, *args, **kwargs):
            return httpx.Response(200, json={"data": [{"id": "longcat-flash"}]})

        def stream(self, method, url, *, json, headers):
            return Stream(json, headers)

    monkeypatch.setattr(validator, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()))
    monkeypatch.setattr(validator, "DistributedRuntime", lambda *a, **k: runtime)
    monkeypatch.setattr(validator, "KvRouter", lambda *a, **k: Router())
    monkeypatch.setattr(validator, "KvRouterConfig", lambda **k: k)
    monkeypatch.setattr(validator, "httpx", SimpleNamespace(AsyncClient=lambda **k: Http(), TransportError=httpx.TransportError))
    args = SimpleNamespace(tokenizer="mock", namespace="test", model="longcat-flash", url="http://frontend", output=tmp_path / "result.json")
    if fault is None:
        await validator.run(args)
    else:
        with pytest.raises(RuntimeError, match=message):
            await validator.run(args)
    report = json.loads(args.output.read_text())
    assert report["passed"] is (fault is None)
    if fault is None:
        assert len(report["requests"]) == 7
    else:
        assert message in report["error"]
    shutdown.assert_called_once_with()
