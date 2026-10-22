# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Equivalent wire fixtures and assertions for decision API qualification."""

import json
import math
import time

import pytest
import requests


def post_with_capacity_retry(url, payload):
    for attempt in range(3):
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code != 429 or attempt == 2:
            return response
        assert response.headers.get("retry-after") == "1", response.text
        time.sleep(1)


def decision_payload(model, dialect="oai"):
    state = "A customer was charged twice and requests a refund today."
    if dialect == "systemone":
        return {
            "model": model,
            "state": state,
            "chat_template_kwargs": {"enable_thinking": False},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose the responsible team.",
                    "criteria": {"billing": "Payments", "technical": "Software"},
                },
                "urgent": {"type": "noul", "instructions": "Action is needed today."},
                "severity": {
                    "type": "score",
                    "instructions": "Rate the impact.",
                    "criteria": ["Low", "Moderate", "High"],
                },
            },
        }
    if dialect != "oai":
        raise ValueError(f"Unknown decision fixture dialect: {dialect}")
    return {
        "model": model,
        "input": state,
        "questions": [
            {
                "type": "choice",
                "name": "route",
                "instructions": "Choose the responsible team.",
                "choices": [
                    {"value": "billing", "description": "Payments"},
                    {"value": "technical", "description": "Software"},
                ],
            },
            {
                "type": "predicate",
                "name": "urgent",
                "instructions": "Action is needed today.",
            },
            {
                "type": "score",
                "name": "severity",
                "instructions": "Rate the impact.",
                "levels": [{"label": value} for value in ("Low", "Moderate", "High")],
            },
        ],
    }


def assert_distribution(values):
    assert values
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in values)
    assert sum(values) == pytest.approx(1.0, abs=1e-9)


def assert_decision_body(body, model, dialect):
    assert dialect in ("oai", "systemone")
    assert body["model"] == model
    if dialect == "oai":
        answers = body["answers"]
        assert [answer["name"] for answer in answers] == ["route", "urgent", "severity"]
        assert [answer["type"] for answer in answers] == [
            "choice",
            "predicate",
            "score",
        ]
        route, urgent, severity = answers
        assert [p["value"] for p in route["probabilities"]] == ["billing", "technical"]
        assert [p["value"] for p in severity["probabilities"]] == [0, 1, 2]
        assert [p["label"] for p in severity["probabilities"]] == [
            "Low",
            "Moderate",
            "High",
        ]
        probability = urgent["probability"]
        probabilities = [p["probability"] for p in severity["probabilities"]]
        for answer in (route, severity):
            assert_distribution([p["probability"] for p in answer["probabilities"]])
            assert (
                math.isfinite(answer["confidence"]) and 0 <= answer["confidence"] <= 1
            )
            assert "label_mass" not in answer and "x_label_mass" not in answer
        usage = body["usage"]
        assert usage["input_tokens"] > 0
        assert (
            0 <= usage["input_tokens_details"]["cached_tokens"] <= usage["input_tokens"]
        )
        assert usage["input_tokens_details"]["cache_write_tokens"] == 0
        assert (
            usage["output_tokens"]
            == usage["output_tokens_details"]["reasoning_tokens"]
            == 0
        )
        assert usage["total_tokens"] == usage["input_tokens"]
    else:
        assert list(body["answers"]) == ["route", "urgent", "severity"]
        route, urgent, severity = body["answers"].values()
        assert list(route["probabilities"]) == ["billing", "technical"]
        assert list(severity["probabilities"]) == ["0", "1", "2"]
        probabilities = list(severity["probabilities"].values())
        for answer in (route, severity):
            assert_distribution(list(answer["probabilities"].values()))
        if dialect == "systemone":
            probability = urgent["noul"]
            assert urgent["type"] == "noul"
            assert severity["legend"] == {"0": "Low", "1": "Moderate", "2": "High"}
            assert body["usage"]["input_tokens"] > 0
            assert body["usage"]["output_tokens"] == 0
            for answer in (route, urgent, severity):
                if "x_label_mass" in answer:
                    assert (
                        math.isfinite(answer["x_label_mass"])
                        and 0 <= answer["x_label_mass"] <= 1
                    )
            for answer in (route, severity):
                assert (
                    math.isfinite(answer["confidence"])
                    and 0 <= answer["confidence"] <= 1
                )
    assert route["choice"] in ("billing", "technical")
    assert math.isfinite(probability) and 0 <= probability <= 1
    assert severity["score"] == pytest.approx(
        sum(index * probability for index, probability in enumerate(probabilities)),
        abs=1e-9,
    )


def assert_request_headers(response):
    assert response.headers["x-request-id"]
    assert response.headers["x-typesafe-request-id"] == response.headers["x-request-id"]


def assert_mocker_probabilities(body, dialect, label_token_ids):
    """Check the mocker's exact deterministic requested-token probabilities."""
    assert dialect in ("oai", "systemone")
    for index, question in enumerate(("route", "urgent", "severity")):
        # The mocker adds a common -10 offset; normalization cancels it out.
        weights = [
            math.exp(-0.1 * ((token_id % 10) + 1))
            for token_id in label_token_ids[question]
        ]
        expected = [weight / sum(weights) for weight in weights]
        answer = body["answers"][index if dialect == "oai" else question]
        if question == "urgent":
            actual = answer["probability" if dialect == "oai" else "noul"]
            assert actual == pytest.approx(expected[0], abs=1e-9, rel=0)
        else:
            actual = (
                [entry["probability"] for entry in answer["probabilities"]]
                if dialect == "oai"
                else list(answer["probabilities"].values())
            )
            assert actual == pytest.approx(expected, abs=1e-9, rel=0)


def _native_probabilities(base_url, tokenizer, content, labels, model):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt)
    label_ids = []
    for label in labels:
        encoded = tokenizer.encode(prompt + label)
        assert encoded[:-1] == prompt_ids and len(encoded) == len(prompt_ids) + 1
        label_ids.append(encoded[-1])
    assert len(set(label_ids)) == len(labels)
    payload = {
        "model": model,
        "input_ids": prompt_ids,
        "sampling_params": {
            "max_new_tokens": 0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "n": 1,
        },
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "token_ids_logprob": label_ids,
        "cache_salt": "systemone-parity-reference",
        "stream": True,
    }
    frames = []
    with requests.post(
        f"{base_url}/generate", json=payload, stream=True, timeout=60
    ) as response:
        assert response.status_code == 200, response.text
        for line in response.iter_lines():
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                frames.append(json.loads(line[6:]))
    assert frames, "Native Generate returned no score frame"
    final = frames[-1]
    assert "error" not in final, final
    assert final.get("output_ids", []) == []
    assert final["meta_info"]["completion_tokens"] == 0
    rows = final["meta_info"]["output_token_ids_logprobs"]
    assert len(rows) == 1
    assert [row[1] for row in rows[0]] == label_ids
    logprobs = [row[0] for row in rows[0]]
    assert all(math.isfinite(value) for value in logprobs)
    weights = [math.exp(value - max(logprobs)) for value in logprobs]
    return (
        [weight / sum(weights) for weight in weights],
        sum(map(math.exp, logprobs)),
        len(prompt_ids),
    )


def assert_native_sglang_parity(base_url, native_url, model, tokenizer):
    payload = decision_payload(model, "systemone")
    state = payload["state"]
    response = requests.post(f"{base_url}/v1/systemone", json=payload, timeout=60)
    assert response.status_code == 200, response.text
    assert response.headers["x-dynamo-systemone-version"] == "1"
    body = response.json()
    assert list(body["answers"]) == ["route", "urgent", "severity"]
    oai_response = requests.post(
        f"{base_url}/v1/decisions", json=decision_payload(model), timeout=60
    )
    assert oai_response.status_code == 200, oai_response.text
    oai_body = oai_response.json()
    assert_decision_body(oai_body, model, "oai")
    references = (
        (
            "route",
            f"{state}\n\nQuestion: Choose the responsible team.\nA: billing - Payments\nB: technical - Software\nAnswer with the letter of one option only.",
            ["A", "B"],
        ),
        (
            "urgent",
            f"{state}\n\nIs the following true? Action is needed today.\nAnswer with yes or no only.",
            ["yes", "no"],
        ),
        (
            "severity",
            f"{state}\n\nQuestion: Rate the impact.\n0: Low\n1: Moderate\n2: High\nAnswer with the number of one level only.",
            ["0", "1", "2"],
        ),
    )
    total_input_tokens = 0
    for ordinal, (question_id, content, labels) in enumerate(references):
        probabilities, label_mass, input_tokens = _native_probabilities(
            native_url, tokenizer, content, labels, model
        )
        total_input_tokens += input_tokens
        answer = body["answers"][question_id]
        if "x_label_mass" in answer:
            assert answer["x_label_mass"] == pytest.approx(
                label_mass, abs=5e-4, rel=1e-3
            )
        oai_answer = oai_body["answers"][ordinal]
        if question_id == "urgent":
            assert oai_answer["probability"] == pytest.approx(
                probabilities[0], abs=5e-4, rel=1e-3
            )
        else:
            assert [
                p["probability"] for p in oai_answer["probabilities"]
            ] == pytest.approx(probabilities, abs=5e-4, rel=1e-3)
        if question_id == "urgent":
            assert answer["noul"] == pytest.approx(probabilities[0], abs=5e-4, rel=1e-3)
        else:
            assert list(answer["probabilities"].values()) == pytest.approx(
                probabilities, abs=5e-4, rel=1e-3
            )
            assert sum(answer["probabilities"].values()) == pytest.approx(
                1.0, abs=1e-12
            )
            if question_id == "severity":
                expected_score = sum(
                    index * value for index, value in enumerate(probabilities)
                )
                assert answer["score"] == pytest.approx(
                    expected_score, abs=5e-4, rel=1e-3
                )
    assert body["usage"] == {"input_tokens": total_input_tokens, "output_tokens": 0}
    chat = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60,
    )
    assert chat.status_code == 200, chat.text
    assert len(chat.json()["choices"]) == 1
