# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import importlib.util
import json
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from tests.utils import decision_api as client

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def _response(status, dialect="oai", code="rate_limit_exceeded", retry_after="1"):
    response = requests.Response()
    response.status_code = status
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    if dialect == "oai":
        body = {"error": {"code": code, "type": code}}
    else:
        body = {"detail": "capacity", "code": code}
    response._content = json.dumps(body).encode()
    return response


@pytest.mark.parametrize("dialect", ["oai", "systemone"])
@pytest.mark.parametrize("wrong_question", ["route", "urgent", "severity"])
def test_mocker_probability_oracle_rejects_uniform_scores(dialect, wrong_question):
    label_ids = {"route": [32, 33], "urgent": [0, 9], "severity": [0, 5, 9]}
    expected = {
        "route": [math.exp(-0.3), math.exp(-0.4)],
        "urgent": [math.exp(-0.1), math.exp(-1)],
        "severity": [math.exp(-0.1), math.exp(-0.6), math.exp(-1)],
    }
    probabilities = {
        key: [weight / sum(weights) for weight in weights]
        for key, weights in expected.items()
    }
    uniform = {
        **probabilities,
        wrong_question: [1 / len(probabilities[wrong_question])]
        * len(probabilities[wrong_question]),
    }
    client.assert_mocker_probabilities(
        _mocker_body(probabilities, dialect), dialect, label_ids
    )
    with pytest.raises(AssertionError):
        client.assert_mocker_probabilities(
            _mocker_body(uniform, dialect), dialect, label_ids
        )


def _mocker_body(probabilities, dialect):
    if dialect == "oai":
        return {
            "answers": [
                {"probability": probabilities[question][0]}
                if question == "urgent"
                else {
                    "probabilities": [
                        {"probability": p} for p in probabilities[question]
                    ]
                }
                for question in ("route", "urgent", "severity")
            ]
        }
    return {
        "answers": {
            "route": {"probabilities": dict(enumerate(probabilities["route"]))},
            "urgent": {"noul": probabilities["urgent"][0]},
            "severity": {"probabilities": dict(enumerate(probabilities["severity"]))},
        }
    }


def test_external_sdk_collection_does_not_import_native_backend(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name not in ("transformers", "tests.serve.test_systemone_sglang"), name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    path = Path(__file__).parents[1] / "serve" / "test_decisions_external.py"
    spec = importlib.util.spec_from_file_location("decision_external_import_test", path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_native_probability_reference_preserves_zero_decode_controls(monkeypatch):
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: "prompt",
        encode=lambda text: {
            "prompt": [1, 2],
            "promptA": [1, 2, 32],
            "promptB": [1, 2, 33],
        }[text],
    )
    frame = {
        "output_ids": [],
        "meta_info": {
            "completion_tokens": 0,
            "output_token_ids_logprobs": [[[-1, 32, None], [-2, 33, None]]],
        },
    }
    response = SimpleNamespace(
        status_code=200,
        text="",
        iter_lines=lambda: iter(
            [b"data: " + json.dumps(frame).encode(), b"data: [DONE]"]
        ),
    )

    def post(url, *, json, stream, timeout):
        assert url == "http://native/generate" and stream and timeout == 60
        assert json["input_ids"] == [1, 2]
        assert json["token_ids_logprob"] == [32, 33]
        assert json["return_logprob"] and not json["return_text_in_logprobs"]
        assert json["sampling_params"] == {
            "max_new_tokens": 0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "n": 1,
        }
        return nullcontext(response)

    monkeypatch.setattr(client.requests, "post", post)
    probabilities, mass, input_tokens = client._native_probabilities(
        "http://native", tokenizer, "question", ["A", "B"], "model"
    )
    assert probabilities == pytest.approx(
        [1 / (1 + math.exp(-1)), 1 / (1 + math.exp(1))]
    )
    assert mass == pytest.approx(math.exp(-1) + math.exp(-2))
    assert input_tokens == 2


@pytest.mark.parametrize("status", [429, 503, 529])
@pytest.mark.parametrize("dialect", ["oai", "systemone"])
def test_systemone_client_retries_only_bounded_capacity_errors(
    monkeypatch, status, dialect
):
    sleeps = []
    calls = []
    monkeypatch.setattr(client.time, "sleep", sleeps.append)

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return next(responses)

    monkeypatch.setattr(client.requests, "post", post)
    overloaded = _response(status, dialect)
    responses = iter([overloaded, overloaded, _response(200, dialect)])
    assert client.post_with_capacity_retry("unused", {}).status_code == 200
    assert sleeps == [1, 1]
    assert len(calls) == 3
    calls.clear()
    responses = iter([overloaded, overloaded, overloaded])
    assert client.post_with_capacity_retry("unused", {}).status_code == status
    assert sleeps == [1, 1, 1, 1]
    assert len(calls) == 3


@pytest.mark.parametrize("retry_after", [None, "0", "2", "invalid"])
@pytest.mark.parametrize("status", [429, 503, 529])
def test_capacity_retry_rejects_invalid_retry_metadata(
    monkeypatch, status, retry_after
):
    response = _response(status, retry_after=retry_after)
    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: response)
    monkeypatch.setattr(client.time, "sleep", lambda _: pytest.fail("must not sleep"))
    with pytest.raises(AssertionError, match="rate_limit_exceeded"):
        client.post_with_capacity_retry("unused", {})


@pytest.mark.parametrize("dialect", ["oai", "systemone"])
@pytest.mark.parametrize("status", [400, 429, 503, 529])
def test_non_capacity_errors_are_not_retried_even_with_retry_after(
    monkeypatch, dialect, status
):
    response = _response(status, dialect, code="server_error")
    calls = []

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    monkeypatch.setattr(client.requests, "post", post)
    monkeypatch.setattr(client.time, "sleep", lambda _: pytest.fail("must not sleep"))
    assert client.post_with_capacity_retry("unused", {}) is response
    assert len(calls) == 1


@pytest.mark.parametrize(
    "body", [b"not-json", b"[]", b'{"error":null}', b'{"error":"capacity"}']
)
def test_malformed_error_is_not_retried(monkeypatch, body):
    response = _response(503)
    response._content = body
    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: response)
    monkeypatch.setattr(client.time, "sleep", lambda _: pytest.fail("must not sleep"))
    assert client.post_with_capacity_retry("unused", {}) is response
