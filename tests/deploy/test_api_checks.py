# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from tests.deploy import api_checks
from tests.deploy.conftest import deployment_spec
from tests.deploy.dgd_utils import validate_chat_response
from tests.deploy.response_checks import (
    validate_embedding,
    validate_stop_response,
    validate_stream,
)
from tests.utils import client

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
    pytest.mark.parallel,
]


def response(content="hello", finish="stop", tokens=1):
    result = requests.Response()
    result.status_code = 200
    result._content_consumed = True
    result._content = json.dumps(
        {
            "model": "model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"completion_tokens": tokens},
        }
    ).encode()
    return result


@pytest.mark.parametrize(
    "content,stop", [(None, "hello"), ("", "hello"), ("The", "The ")]
)
def test_stop_accepts_shortened_content_and_rejects_stop_sequence(content, stop):
    validate_chat_response(response(content), "model", max_tokens=30, stop=stop)
    with pytest.raises(AssertionError):
        validate_chat_response(response(stop), "model", max_tokens=30, stop=stop)


def test_chat_keeps_default_minimum_length_without_stop():
    with pytest.raises(AssertionError, match="Response content too short"):
        validate_chat_response(response("hello"), "model", max_tokens=30)


def test_token_limit_and_finish_reason():
    validate_chat_response(response("h", "length"), "model", 0, max_tokens=1)
    with pytest.raises(AssertionError):
        validate_chat_response(response(tokens=2), "model", 0, max_tokens=1)
    with pytest.raises(AssertionError):
        validate_chat_response(response(finish="tool_calls"), "model", 0, max_tokens=30)


def test_stream_requires_one_finish_and_done():
    content = 'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}'
    finish = 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
    validate_stream([content, finish, "data: [DONE]"])
    for lines in (
        ["event:error", content, finish, "data: [DONE]"],
        ["event: error", content, finish, "data: [DONE]"],
        [content, finish],
        [content, finish, finish, "data: [DONE]"],
        [content, finish, content, "data: [DONE]"],
    ):
        with pytest.raises(AssertionError):
            validate_stream(lines)


def test_embedding_contract_rejects_wire_base64_and_nonfinite_vectors():
    body = {
        "object": "list",
        "data": [{"index": 0, "object": "embedding", "embedding": [0.1, 0.2]}],
    }
    validate_embedding(body, 1, 2)
    for vector in ("base64", [float("nan"), 0.2], [0.1], [True, 0.2]):
        body["data"][0]["embedding"] = vector
        with pytest.raises(AssertionError):
            validate_embedding(body, 1, 2)


@pytest.mark.parametrize("endpoint", [None, "/custom/chat/completions"])
def test_api_cases_use_shared_client_and_preserve_invalid_response(
    monkeypatch, tmp_path, endpoint
):
    calls = []

    def send(url, payload, **kwargs):
        calls.append((url, payload))
        if payload.get("stream"):
            result = response()
            result._content = b'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}\n\ndata: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n'
            return result
        if payload.get("max_tokens") == 1:
            return response("The", "length", tokens=1)
        if "stop" in payload:
            return response("The", tokens=2)
        return response(
            "The air in the ruins was thick with the scent of damp earth. " * 2,
            tokens=30,
        )

    monkeypatch.setattr(api_checks, "send_request", send)
    api_checks.check_deployment_api(
        "http://test", "model", "chat", tmp_path, endpoint=endpoint
    )
    assert len(calls) == 4
    assert all(
        url == "http://test" + (endpoint or "/v1/chat/completions") for url, _ in calls
    )
    assert calls[-1][1]["stop"] == "air"
    assert (
        json.loads((tmp_path / "stop.json").read_text())["response"]
        == response("The", tokens=2).text
    )
    monkeypatch.setattr(
        api_checks, "send_request", lambda *a, **k: response(tokens=100)
    )
    with pytest.raises(AssertionError):
        api_checks.check_deployment_api(
            "http://test", "model", "chat", tmp_path, endpoint=endpoint
        )
    assert "100" in json.loads((tmp_path / "unary.json").read_text())["response"]


def test_embedding_api_uses_default_endpoint(monkeypatch, tmp_path):
    calls = []

    def send(url, payload, **kwargs):
        calls.append((url, payload))
        count = len(payload["input"]) if isinstance(payload["input"], list) else 1
        result = response()
        result._content = json.dumps(
            {
                "model": "model",
                "object": "list",
                "data": [
                    {"index": i, "object": "embedding", "embedding": [0.1] * 1024}
                    for i in range(count)
                ],
            }
        ).encode()
        return result

    monkeypatch.setattr(api_checks, "send_request", send)
    api_checks.check_deployment_api("http://test", "model", "embedding", tmp_path)
    assert len(calls) == 3
    assert all(url == "http://test/v1/embeddings" for url, _ in calls)
    assert "encoding_format" not in calls[0][1]
    assert calls[1][1]["encoding_format"] == "float"
    assert calls[2][1]["input"] == ["Hello", "World"]


def test_embedding_readiness_uses_embedding_payload(monkeypatch):
    post = Mock(return_value=response())
    monkeypatch.setattr(client.requests, "post", post)
    monkeypatch.setattr(client.time, "sleep", lambda _: None)
    payload = {"model": "model", "input": "test"}
    assert client.wait_for_model_availability(
        "http://test", "/v1/embeddings", "model", client.logger, payload=payload
    )
    assert post.call_args.kwargs["json"] == payload


def test_deploy_fixture_overrides_frontend_separately():
    root = Path(__file__).resolve().parents[2]
    options = {"--frontend-image": "frontend:candidate", "--model-cache-pvc": ""}
    request = SimpleNamespace(config=SimpleNamespace(getoption=options.__getitem__))
    spec = deployment_spec.__wrapped__(
        root / "examples/backends/sglang/deploy/agg_embed.yaml",
        "worker:candidate",
        "test",
        request,
    )
    assert spec["Frontend"].image == "frontend:candidate"
    assert spec["decode"].image == "worker:candidate"
    assert spec["decode"].model == "Qwen/Qwen3-Embedding-0.6B"
    assert "--embedding-worker" in spec["decode"]._get_args()
    assert "--use-sglang-tokenizer" in spec["decode"]._get_args()


@pytest.mark.parametrize("content", ["The", "The "])
def test_stop_allows_text_before_stop(content):
    baseline = response("The air in the ruins", tokens=30).json()
    stopped = response(content, tokens=2)
    body = validate_chat_response(stopped, "model", max_tokens=30, stop="air")
    validate_stop_response(body, baseline, "air")


@pytest.mark.parametrize(
    "content,finish,tokens",
    [
        (None, "stop", 2),
        ("", "stop", 2),
        ("The air", "stop", 2),
        ("Other", "stop", 2),
        ("The ", "length", 2),
        ("The ", "stop", 30),
        (False, "stop", 2),
    ],
)
def test_stop_rejects_leaked_stop_divergence_and_no_early_termination(
    content, finish, tokens
):
    baseline = response("The air in the ruins", tokens=30).json()
    with pytest.raises(AssertionError):
        body = validate_chat_response(
            response(content, finish, tokens), "model", max_tokens=30, stop="air"
        )
        validate_stop_response(body, baseline, "air")
