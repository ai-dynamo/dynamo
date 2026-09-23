# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline validator regressions with HTTP, tokenization, and discovery mocked."""

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(60),  # Allow cold imports of transformers and Dynamo bindings.
]


@pytest.fixture
def optimization_level():
    return 0


@pytest.fixture
def validator(optimization_level, monkeypatch):
    """Load the script, optionally using python -OO compilation semantics."""
    path = Path(__file__).parents[1] / "validate_disagg.py"
    module = ModuleType("tokenspeed_disagg_validator")
    exec(
        compile(path.read_text(), str(path), "exec", optimize=optimization_level),
        module.__dict__,
    )
    clock = 0.0

    async def sleep(delay):
        nonlocal clock
        clock += delay

    # Advance readiness deadlines without sleeping or changing shared modules.
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock))
    monkeypatch.setattr(
        module,
        "asyncio",
        SimpleNamespace(sleep=sleep, get_running_loop=module.asyncio.get_running_loop),
    )
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
        await validator.wait_for_frontend(
            http, "http://frontend", "longcat-flash", timeout=5
        )
    assert calls == 3


async def test_readiness_timeout_retains_transport_error(validator):
    """An unavailable frontend fails with the last connection error at the deadline."""

    async def respond(request):
        raise httpx.ConnectError("port closed", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises(RuntimeError, match="ConnectError: port closed"):
            await validator.wait_for_frontend(
                http, "http://frontend", "longcat-flash", timeout=0.01
            )


@pytest.mark.parametrize("optimization_level", [0, 2], ids=["normal", "optimized"])
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
        ("round-robin-0", "Cached-prefix owner"),
        ("round-robin-1", "Cached-prefix owner"),
        ("cache-timeout", "Expected 2 cached blocks"),
        ("http-error", "HTTP 503"),
        ("empty-http-error", "HTTP 503"),
        ("invalid-utf8-error", "HTTP 503"),
    ],
)
async def test_full_validation_cannot_pass_bad_deployment(
    validator, monkeypatch, tmp_path, fault, message
):
    """Every safety check remains active under optimization, with a passing control."""
    topics = ["ORANGE", "PURPLE", "SILVER"]
    shutdown = Mock()

    class ErrorBody(httpx.AsyncByteStream):
        def __init__(self):
            self.chunks_read = 0
            self.closed = False

        async def __aiter__(self):
            if fault == "empty-http-error":
                return
            for _ in range(8192):
                self.chunks_read += 1
                yield b"x" * 999 + (b"\xe2" if fault == "invalid-utf8-error" else b"x")

        async def aclose(self):
            self.closed = True

    error_body = ErrorBody()

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
                        "device_blocks": 2
                        if fault != "cache-timeout"
                        and (worker == owner or (owner == 3 and fault == "cold-overlap"))
                        else 0,
                    }
                    for worker in [1, 2]
                ]
            }

    unforced_requests = 0

    async def respond(request):
        nonlocal unforced_requests
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "longcat-flash"}]})
        if fault in ("http-error", "empty-http-error", "invalid-utf8-error"):
            return httpx.Response(503, stream=error_body)
        payload = json.loads(request.content)
        headers = request.headers
        index = payload["prompt"][0]
        forced = headers.get("x-dynamo-prefill-instance-id")
        owner = int(forced) if forced else min(index + 1, 2)
        if not forced and fault in ("round-robin-0", "round-robin-1"):
            offset = int(fault[-1])
            owner = (unforced_requests + offset) % 2 + 1
            unforced_requests += 1
        if fault == "unknown-prefill":
            owner = 99
        elif fault == "forced-prefill" and forced:
            owner = 3 - int(forced)
        elif fault == "wrong-reuse" and not forced:
            owner = 3 - owner
        chunk = {
            "choices": [{"text": "WRONG" if fault == "wrong-text" else topics[index]}],
            "nvext": {
                "worker_id": {
                    "prefill_worker_id": owner,
                    "decode_worker_id": None if fault == "missing-decode" else 3,
                }
            },
        }
        body = "data: " + json.dumps(chunk) + "\n\n"
        if fault != "missing-done":
            body += "data: [DONE]\n\n"
        return httpx.Response(200, text=body)

    monkeypatch.setattr(
        validator,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
    )
    monkeypatch.setattr(validator, "DistributedRuntime", lambda *a, **k: runtime)
    monkeypatch.setattr(validator, "KvRouter", lambda *a, **k: Router())
    monkeypatch.setattr(validator, "KvRouterConfig", lambda **k: k)
    monkeypatch.setattr(
        validator,
        "httpx",
        SimpleNamespace(
            AsyncClient=lambda **k: httpx.AsyncClient(
                transport=httpx.MockTransport(respond), **k
            ),
            TransportError=httpx.TransportError,
        ),
    )
    args = SimpleNamespace(
        tokenizer="mock",
        namespace="test",
        model="longcat-flash",
        url="http://frontend",
        output=tmp_path / "result.json",
    )
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
    if fault in ("http-error", "empty-http-error", "invalid-utf8-error"):
        assert error_body.closed
        assert error_body.chunks_read == (0 if fault == "empty-http-error" else 1)
        expected_body = ""
        if fault != "empty-http-error":
            expected_body = "x" * 999 + (
                "\ufffd" if fault == "invalid-utf8-error" else "x"
            )
        assert report["error"] == "warm-0: HTTP 503: " + expected_body
    shutdown.assert_called_once_with()
