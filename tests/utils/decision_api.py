# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Equivalent wire fixtures and assertions for decision API qualification."""

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
