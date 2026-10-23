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

from tests.utils import decision_api as client

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_chat_decode_accepts_generated_tokens_and_normal_completion(finish_reason):
    client.assert_chat_decode(
        {
            "choices": [{"finish_reason": finish_reason}],
            "usage": {"completion_tokens": 1},
        }
    )


@pytest.mark.parametrize("tokens,finish_reason", [(0, "stop"), (1, None), (1, "error")])
def test_chat_decode_rejects_zero_decode_or_abnormal_completion(tokens, finish_reason):
    with pytest.raises(AssertionError):
        client.assert_chat_decode(
            {
                "choices": [{"finish_reason": finish_reason}],
                "usage": {"completion_tokens": tokens},
            }
        )


def test_native_parity_rejects_invalid_systemone_contract(monkeypatch):
    probabilities = {
        "route": [0.6, 0.4],
        "urgent": [0.6, 0.4],
        "severity": [0.4, 0.4, 0.2],
    }
    jev = {
        **_mocker_body(probabilities, "systemone"),
        "model": "wrong-model",
        "usage": {"input_tokens": 21, "output_tokens": 0},
    }
    oai = {
        "model": "model",
        "answers": [
            {
                "type": "choice",
                "name": "route",
                "choice": "billing",
                "confidence": 0.5,
                "probabilities": [
                    {"value": value, "probability": p}
                    for value, p in zip(
                        ("billing", "technical"), probabilities["route"]
                    )
                ],
            },
            {"type": "predicate", "name": "urgent", "probability": 0.6},
            {
                "type": "score",
                "name": "severity",
                "score": 0.8,
                "confidence": 0.5,
                "probabilities": [
                    {"value": i, "label": label, "probability": p}
                    for i, (label, p) in enumerate(
                        zip(("Low", "Moderate", "High"), probabilities["severity"])
                    )
                ],
            },
        ],
        "usage": {
            "input_tokens": 21,
            "output_tokens": 0,
            "total_tokens": 21,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    jev = {
        **jev,
        "answers": {
            **jev["answers"],
            "severity": {**jev["answers"]["severity"], "score": 0.8},
        },
    }
    chat = {"choices": [{"finish_reason": "stop"}], "usage": {"completion_tokens": 1}}

    def post(url, **kwargs):
        body = (
            jev
            if url.endswith("systemone")
            else oai
            if url.endswith("decisions")
            else chat
        )
        return SimpleNamespace(
            status_code=200,
            text="",
            headers={"x-dynamo-systemone-version": "1"},
            json=lambda: body,
        )

    monkeypatch.setattr(client.requests, "post", post)
    references = iter(
        (probabilities[question], 0.1, 7)
        for question in ("route", "urgent", "severity")
    )
    monkeypatch.setattr(client, "_native_probabilities", lambda *args: next(references))
    with pytest.raises(AssertionError):
        client.assert_native_sglang_parity(
            "http://dynamo", "http://native", "model", None
        )


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


@pytest.mark.parametrize("capacity_status", [429, 529])
def test_systemone_client_retries_only_bounded_capacity_errors(
    monkeypatch, capacity_status
):
    sleeps = []
    monkeypatch.setattr(client.time, "sleep", sleeps.append)
    responses = iter(
        SimpleNamespace(status_code=status, headers={"retry-after": "1"}, text="")
        for status in [capacity_status, capacity_status, 200]
    )
    monkeypatch.setattr(
        client.requests, "post", lambda *args, **kwargs: next(responses)
    )
    assert client.post_with_capacity_retry("unused", {}).status_code == 200
    assert sleeps == [1, 1]
    overloaded = SimpleNamespace(
        status_code=capacity_status, headers={"retry-after": "1"}, text=""
    )
    responses = iter([overloaded, overloaded, overloaded])
    monkeypatch.setattr(
        client.requests, "post", lambda *args, **kwargs: next(responses)
    )
    assert client.post_with_capacity_retry("unused", {}).status_code == capacity_status
    assert sleeps == [1, 1, 1, 1]
    unavailable = SimpleNamespace(
        status_code=capacity_status, headers={}, text="missing retry contract"
    )
    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: unavailable)
    with pytest.raises(AssertionError, match="missing retry contract"):
        client.post_with_capacity_retry("unused", {})
    unavailable = SimpleNamespace(
        status_code=503, headers={"retry-after": "1"}, text="worker unavailable"
    )
    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: unavailable)
    assert client.post_with_capacity_retry("unused", {}) is unavailable
    assert sleeps == [1, 1, 1, 1]
