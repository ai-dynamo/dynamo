# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from pydantic import ValidationError

from tests.deploy import api_checks
from tests.deploy.conftest import deployment_spec
from tests.deploy.dgd_utils import validate_chat_response
from tests.deploy.response_checks import (
    validate_chat,
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
            "id": "chat-test",
            "created": 1,
            "object": "chat.completion",
            "model": "model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "completion_tokens": tokens,
                "prompt_tokens": 10,
                "total_tokens": 10 + tokens,
            },
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


def stream_chunk(content=None, finish=None, **fields):
    return {
        "id": "chat-test",
        "created": 1,
        "model": "model",
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": content}, "finish_reason": finish}
        ],
        **fields,
    }


def stream_lines(*chunks):
    return ["data: " + json.dumps(chunk) for chunk in chunks] + ["data: [DONE]"]


def test_stream_requires_one_finish_and_done():
    content, finish, done = stream_lines(
        stream_chunk("hello"), stream_chunk(finish="stop")
    )
    validate_stream([content, finish, done], "model")
    for lines in (
        ["event:error", content, finish, done],
        ["event: error", content, finish, done],
        [content, finish],
        [content, finish, finish, done],
        [content, finish, content, done],
        [content, finish, done, content],
    ):
        with pytest.raises(AssertionError):
            validate_stream(lines, "model")


@pytest.mark.parametrize("fault", ["model", "id", "role"])
def test_stream_rejects_wrong_model_changed_id_and_nonassistant_role(fault):
    first = stream_chunk("hello")
    last = stream_chunk(finish="stop")
    if fault == "role":
        first["choices"][0]["delta"]["role"] = "user"
    else:
        last[fault] = "wrong"
    with pytest.raises(AssertionError):
        validate_stream(stream_lines(first, last), "model")


def embedding_body(vectors):
    return {
        "model": "model",
        "object": "list",
        "usage": {"prompt_tokens": len(vectors), "total_tokens": len(vectors)},
        "data": [
            {"index": i, "object": "embedding", "embedding": vector}
            for i, vector in enumerate(vectors)
        ],
    }


def encode_embedding(vector):
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


@pytest.mark.parametrize(
    "vector,encoding",
    [
        ("base64", "float"),
        ([float("nan"), 0.2], "float"),
        ([0.1], "float"),
        ([True, 0.2], "float"),
        ("!!!", "base64"),
        (encode_embedding([0.1]), "base64"),
        (encode_embedding([float("inf"), 0.2]), "base64"),
        ([0.1, 0.2], "base64"),
    ],
)
def test_embedding_rejects_invalid_vectors_and_encoding(vector, encoding):
    with pytest.raises((AssertionError, ValueError)):
        validate_embedding(embedding_body([vector]), 1, 2, "model", encoding)


@pytest.mark.parametrize("kind", ["chat", "stream", "embedding"])
def test_response_schemas_reject_wrong_field_types(kind):
    if kind == "chat":
        body = response().json()
        body["usage"]["completion_tokens"] = "1"
        with pytest.raises(ValidationError):
            validate_chat(body, 30)
    elif kind == "stream":
        chunk = stream_chunk("hello", finish="stop")
        chunk["created"] = "1"
        with pytest.raises(ValidationError):
            validate_stream(stream_lines(chunk), "model")
    else:
        body = embedding_body([[0.1, 0.2]])
        body["usage"]["prompt_tokens"] = "1"
        with pytest.raises(ValidationError):
            validate_embedding(body, 1, 2, "model")


@pytest.fixture
def chat_sender():
    def send(url, payload, **kwargs):
        if payload.get("stream"):
            result = response()
            result._content = "\n\n".join(
                stream_lines(stream_chunk("hello"), stream_chunk(finish="stop"))
            ).encode()
            return result
        if payload.get("max_tokens") == 1:
            return response("The", "length", tokens=1)
        if "stop" in payload:
            return response("The", tokens=2)
        return response(
            "The air in the ruins was thick with the scent of damp earth. " * 2,
            tokens=30,
        )

    return send


@pytest.mark.parametrize("inject_sender", [False, True])
@pytest.mark.parametrize(
    "endpoint", ["/v1/chat/completions", "/custom/chat/completions"]
)
def test_api_cases_use_shared_client_and_preserve_invalid_response(
    monkeypatch, tmp_path, endpoint, inject_sender, chat_sender
):
    send = Mock(side_effect=chat_sender)
    options = {"request_sender": send} if inject_sender else {}
    if inject_sender:
        monkeypatch.setattr(
            api_checks, "send_request", Mock(side_effect=AssertionError)
        )
    else:
        monkeypatch.setattr(api_checks, "send_request", send)
    url = "http://test" + endpoint
    api_checks.check_chat_api(url, "model", tmp_path, **options)
    assert send.call_count == 4
    assert all(call.args[0] == url for call in send.call_args_list)
    assert send.call_args.args[1]["stop"] == "air"
    assert (
        json.loads((tmp_path / "stop.json").read_text())["response"]
        == response("The", tokens=2).text
    )
    monkeypatch.setattr(
        api_checks, "send_request", lambda *a, **k: response(tokens=100)
    )
    with pytest.raises(AssertionError):
        api_checks.check_chat_api(url, "model", tmp_path)
    assert "100" in json.loads((tmp_path / "unary.json").read_text())["response"]


@pytest.mark.parametrize("tokens,finish", [(0, "length"), (1, "stop")])
def test_limited_case_rejects_zero_tokens_and_wrong_finish(
    tmp_path, chat_sender, tokens, finish
):
    def send(url, payload, **kwargs):
        if payload.get("max_tokens") == 1:
            return response("", finish, tokens=tokens)
        return chat_sender(url, payload, **kwargs)

    with pytest.raises(AssertionError):
        api_checks.check_chat_api(
            "http://test/chat", "model", tmp_path, request_sender=send
        )
    saved = json.loads((tmp_path / "limited.json").read_text())
    assert json.loads(saved["response"])["usage"]["completion_tokens"] == tokens


def test_stream_http_error_body_is_preserved(tmp_path):
    error = response()
    error.status_code = 500
    error._content = b'{"error":{"message":"backend exploded"}}'
    artifact = tmp_path / "stream.json"
    with pytest.raises(requests.HTTPError):
        api_checks._request(
            "http://test/chat",
            {"model": "model", "stream": True},
            artifact,
            request_sender=lambda *args, **kwargs: error,
        )
    saved = json.loads(artifact.read_text())
    assert saved["http_status"] == 500
    assert saved["response"] == error.text


@pytest.fixture
def embedding_sender():
    def send(url, payload, **kwargs):
        inputs = payload["input"]
        inputs = inputs if isinstance(inputs, list) else [inputs]
        dimensions = payload.get("dimensions", 1024)
        vectors = [[0.1 if text == "Hello" else 0.5] * dimensions for text in inputs]
        if len(inputs) > 1:
            vectors = [[x + 1e-5 for x in vector] for vector in vectors]
        if payload.get("encoding_format") == "base64":
            vectors = [encode_embedding(vector) for vector in vectors]
        result = response()
        result._content = json.dumps(embedding_body(vectors)).encode()
        return result

    return send


def test_embedding_api_checks_formats_dimensions_and_batch(tmp_path, embedding_sender):
    send = Mock(side_effect=embedding_sender)
    url = "http://test/custom/embeddings"
    api_checks.check_embedding_api(url, "model", tmp_path, request_sender=send)
    assert send.call_count == 6
    assert all(call.args[0] == url for call in send.call_args_list)
    payloads = [call.args[1] for call in send.call_args_list]
    assert "encoding_format" not in payloads[0]
    assert payloads[1]["encoding_format"] == "float"
    assert payloads[3]["input"] == ["Hello", "World"]
    assert payloads[4]["dimensions"] == payloads[5]["dimensions"] == 128
    saved = json.loads((tmp_path / "base64.json").read_text())
    assert saved["request"]["encoding_format"] == "base64"
    wire = json.loads(saved["response"])
    encoded = wire["data"][0]["embedding"]
    result = validate_embedding(wire, 1, 128, "model", "base64")
    assert result.data[0].embedding == pytest.approx([0.1] * 128)
    assert wire["data"][0]["embedding"] == encoded


@pytest.mark.parametrize(
    "fault",
    ["constant", "duplicate", "swap", "zero_usage", "total_usage", "batch_usage"],
)
def test_embedding_rejects_wrong_contents_and_usage(tmp_path, embedding_sender, fault):
    def send(url, payload, **kwargs):
        result = embedding_sender(url, payload, **kwargs)
        if fault == "constant":
            inputs = payload["input"]
            inputs = inputs if isinstance(inputs, list) else [inputs]
            dimensions = payload.get("dimensions", 1024)
            vectors = [[0.3] * dimensions for _ in inputs]
            if payload.get("encoding_format") == "base64":
                vectors = [encode_embedding(vector) for vector in vectors]
            result._content = json.dumps(embedding_body(vectors)).encode()
            return result
        if not isinstance(payload["input"], list):
            return result
        body = result.json()
        if fault == "duplicate":
            body["data"][1]["embedding"] = body["data"][0]["embedding"]
        elif fault == "swap":
            body["data"][0]["embedding"], body["data"][1]["embedding"] = (
                body["data"][1]["embedding"],
                body["data"][0]["embedding"],
            )
        elif fault == "zero_usage":
            body["usage"] = {"prompt_tokens": 0, "total_tokens": 0}
        elif fault == "total_usage":
            body["usage"]["total_tokens"] = 3
        else:
            body["usage"] = {"prompt_tokens": 3, "total_tokens": 3}
        result._content = json.dumps(body).encode()
        return result

    with pytest.raises(AssertionError):
        api_checks.check_embedding_api(
            "http://test/embed", "model", tmp_path, request_sender=send
        )
    assert (tmp_path / "batch.json").exists()


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
    baseline = validate_chat(response("The air in the ruins", tokens=30).json(), 30)
    stopped = response(content, tokens=2)
    body = validate_chat_response(stopped, "model", max_tokens=30, stop="air")
    validate_stop_response(body, baseline, "air")


@pytest.mark.parametrize(
    "content,finish,tokens",
    [
        ("", "stop", 2),
        ("T", "stop", 2),
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
    baseline = validate_chat(response("The air in the ruins", tokens=30).json(), 30)
    with pytest.raises((AssertionError, ValidationError)):
        body = validate_chat_response(
            response(content, finish, tokens), "model", max_tokens=30, stop="air"
        )
        validate_stop_response(body, baseline, "air")
