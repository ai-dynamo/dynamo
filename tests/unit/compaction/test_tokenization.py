# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import sys
from types import SimpleNamespace

import httpx
import pytest
from test_clients import packet

import dynamo.compaction.clients as clients
import dynamo.compaction.tokenization as tokenization
from dynamo.compaction.fixtures import sdk_response
from dynamo.compaction.helper import execute_model
from dynamo.compaction.protocol import Invalid, canonical, strict_json
from dynamo.compaction.tokenization import RenderedCounter, count_request

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "PREFIX" + canonical(messages) + "SUFFIX"

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return list(text.encode())


def test_exact_choice_rendering_and_largest_question_context():
    counter = RenderedCounter(Tokenizer(), "fixture")
    payload = {
        "model": "local",
        "input": "Evidence",
        "questions": [
            {
                "type": "choice",
                "name": "a",
                "instructions": "Keep?",
                "choices": [
                    {"value": "keep", "description": "original"},
                    {"value": "summarize", "description": "summary"},
                ],
            }
        ],
    }
    content = "Evidence\n\nQuestion: Keep?\nA: keep - original\nB: summarize - summary\nAnswer with the letter of one option only."
    expected = len(
        Tokenizer().apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    assert counter(canonical(payload)) == expected


def test_actor_count_preserves_actual_text_blocks_and_reserves_output():
    data = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "max_output_tokens": 20,
    }
    result = count_request(data, RenderedCounter(Tokenizer(), "fixture"))
    assert result["total_tokens"] == result["input_tokens"] + 20
    assert result["counter_name"] == "fixture"


def test_text_blocks_match_dynamo_newline_concatenation():
    counter = RenderedCounter(Tokenizer(), "fixture")
    split = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "one"},
                    {"type": "text", "text": "two"},
                ],
            }
        ]
    }
    joined = {"messages": [{"role": "user", "content": "one\ntwo"}]}
    assert counter(canonical(split)) == counter(canonical(joined))


@pytest.mark.parametrize(
    "extra",
    [
        {"tools": []},
        {"messages": [{"role": "tool", "content": "bad"}]},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "private"}}],
                }
            ]
        },
        {"max_output_tokens": True},
    ],
)
def test_unknown_rendering_shapes_fail_closed(extra):
    payload = {
        "messages": [{"role": "user", "content": "x"}],
        "max_output_tokens": 1,
        **extra,
    }
    with pytest.raises(Invalid):
        count_request(payload, RenderedCounter(Tokenizer(), "fixture"))


def test_model_helper_explicit_profile_uses_all_three_sdk_calls(monkeypatch):
    monkeypatch.setattr(
        tokenization,
        "load_counter",
        lambda _: RenderedCounter(Tokenizer(), "mock-rendered"),
    )
    original = clients.make_client
    seen = []

    def handler(req):
        seen.append(req.url.path)
        response = sdk_response(req)
        body = response.json()
        expected = RenderedCounter(Tokenizer(), "mock-rendered").usage(
            canonical(strict_json(req.content))
        )
        if req.url.path.endswith("decisions"):
            body["usage"]["input_tokens"] = expected
            body["usage"]["total_tokens"] = expected
        else:
            body["usage"]["prompt_tokens"] = expected
            body["usage"]["total_tokens"] = expected + 1
        return httpx.Response(200, json=body, headers=response.headers)

    monkeypatch.setattr(
        clients,
        "make_client",
        lambda base, transport, api_key: original(
            "http://fixture.invalid/v1", httpx.MockTransport(handler)
        ),
    )
    args = SimpleNamespace(
        tokenizer_path="/explicit/fixture",
        base_url="http://127.0.0.1:8000/v1",
        decision_model="Qwen/Qwen3-0.6B",
        generation_model="Qwen/Qwen3-0.6B",
        max_context_tokens=8192,
        min_probability=0.9,
        min_confidence=0.9,
        generation_max_tokens=512,
    )
    result = asyncio.run(execute_model(packet(), args))
    assert result["status"] == "accepted" and result["mode"] == "model"
    assert result["counter_name"] == "mock-rendered"
    assert seen == ["/v1/decisions", "/v1/chat/completions", "/v1/chat/completions"]


def test_tokenizer_artifacts_are_pinned_before_local_only_load(tmp_path, monkeypatch):
    path = tmp_path / tokenization.PROFILE_REVISION
    path.mkdir()
    artifacts = [b"tokenizer", b"config", b"template"]
    for name, data in zip(
        ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"), artifacts
    ):
        (path / name).write_bytes(data)
    digest = hashlib.sha256(b"".join(artifacts)).hexdigest()
    monkeypatch.setattr(tokenization, "PROFILE_ARTIFACT_SHA256", digest)
    calls = []

    def load(local_path, **kwargs):
        calls.append((local_path, kwargs))
        return SimpleNamespace(chat_template="pinned template")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)),
    )
    counter = tokenization.load_counter(str(path))
    assert digest in counter.name
    assert calls == [
        (str(path), {"local_files_only": True, "trust_remote_code": False})
    ]
    (path / "tokenizer.json").write_bytes(b"changed")
    with pytest.raises(Invalid, match="unqualified_tokenizer_profile"):
        tokenization.load_counter(str(path))
    assert len(calls) == 1


@pytest.mark.parametrize("path", ["relative/path", "/missing/profile"])
def test_invalid_tokenizer_paths_reject_before_loading(path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=None)
    )
    with pytest.raises(Invalid, match="invalid_tokenizer_path"):
        tokenization.load_counter(path)


def test_decision_usage_aggregates_questions_but_context_uses_largest():
    counter = RenderedCounter(Tokenizer(), "fixture")
    question = {
        "type": "choice",
        "instructions": "Keep?",
        "choices": [
            {"value": "keep", "description": "original"},
            {"value": "summarize", "description": "summary"},
        ],
    }
    serialized = canonical({"input": "evidence", "questions": [question, question]})
    assert counter.usage(serialized) == counter(serialized) * 2


def test_candidate_projection_count_and_invalid_shape():
    counter = RenderedCounter(Tokenizer(), "fixture")
    serialized = canonical({"summary": "x", "retained": []})
    assert counter(serialized) == len(serialized)
    with pytest.raises(Invalid, match="unsupported_counter_input"):
        counter(canonical({"unknown": "input"}))
