# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict P0 evaluation contracts shared by live adapters and offline auditing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class DecisionContractError(ValueError):
    """A response or request cannot establish successful decision evaluation."""


@dataclass(frozen=True)
class DecisionSummary:
    question_count: int
    refusal_count: int
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    label_mass: list[float] | None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DecisionContractError(message)


def integer(value: Any) -> int:
    require(type(value) is int and value >= 0, "Expected a nonnegative integer counter")
    return value


def probability(value: Any) -> float:
    require(
        type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
        "Expected a finite probability in [0,1]",
    )
    return float(value)


def typed(value: Any) -> tuple[type, Any]:
    require(type(value) in (str, bool), "Choices must preserve string or boolean type")
    return type(value), value


def distribution(values: list[Any]) -> list[float]:
    result = [probability(value) for value in values]
    require(
        bool(result) and abs(sum(result) - 1) <= 1e-9,
        "Candidate probabilities must sum to one",
    )
    return result


def validate_request(body: dict, dialect: str) -> None:
    require(isinstance(body, dict), "Request must be an object")
    if dialect == "native_score":
        require(
            body.get("stream") is True and body.get("return_logprob") is True,
            "Native scoring requires streaming log probabilities",
        )
        require(
            isinstance(body.get("sampling_params"), dict)
            and type(body["sampling_params"].get("max_new_tokens")) is int
            and body["sampling_params"]["max_new_tokens"] == 0,
            "Native scoring requires zero decode",
        )
        for key in ("input_ids", "token_ids_logprob"):
            require(
                isinstance(body.get(key), list) and bool(body[key]),
                "Native scoring requires token IDs",
            )
            for value in body[key]:
                integer(value)
        require(
            len(set(body["token_ids_logprob"])) == len(body["token_ids_logprob"]),
            "Duplicate candidate token IDs",
        )
        return
    require(
        isinstance(body.get("model"), str) and bool(body["model"]),
        "Explicit model is required",
    )
    require(dialect in ("oai", "systemone"), "Unknown decision dialect")
    allowed = (
        {"model", "state", "questions", "images", "chat_template_kwargs"}
        if dialect == "systemone"
        else {"model", "input", "questions"}
    )
    if dialect == "oai":
        allowed |= {"safety_identifier"}
    require(not set(body) - allowed, "Unsupported or mixed request fields")
    require(not body.get("images"), "Images are outside text qualification")
    controls = body.get("chat_template_kwargs", {})
    require(
        isinstance(controls, dict)
        and all(
            key in ("enable_thinking", "thinking") and value is False
            for key, value in controls.items()
        ),
        "Unsupported template controls",
    )
    if dialect == "systemone":
        require(
            "state" in body and isinstance(body["state"], (str, dict, list)),
            "Missing text state",
        )
        require(not body.get("images"), "Image inputs are outside text qualification")
        questions = body.get("questions")
        require(
            isinstance(questions, dict) and bool(questions),
            "Expected ordered question object",
        )
        for key, question in questions.items():
            require(isinstance(key, str) and bool(key), "Invalid question identifier")
            _question(question, dialect)
    else:
        require(
            "input" in body and isinstance(body["input"], (str, dict, list)),
            "Missing text input",
        )
        if dialect == "oai":
            _openai_text(body["input"])
        questions = body.get("questions")
        require(
            isinstance(questions, list) and bool(questions),
            "Expected ordered questions array",
        )
        for question in questions:
            _question(question, dialect)


def _question(question: Any, dialect: str) -> None:
    require(isinstance(question, dict), "Question must be an object")
    kind = question.get("type")
    predicate = {"oai": "predicate", "systemone": "noul"}[dialect]
    require(kind in (predicate, "choice", "score"), "Unsupported question type")
    if dialect == "oai":
        expected_fields = (
            {"choices"}
            if kind == "choice"
            else {"levels"}
            if kind == "score"
            else set()
        )
        require(
            not (set(question) & {"choices", "levels"}) - expected_fields,
            "Mixed question candidate fields",
        )
        require(
            isinstance(question.get("instructions"), str),
            "Missing question instructions",
        )
        require(
            "name" not in question or isinstance(question["name"], str),
            "Invalid question name",
        )
        require(
            not set(question) - {"type", "name", "instructions", "choices", "levels"},
            "Mixed question schema",
        )
    else:
        require(
            not set(question) - {"type", "instructions", "criteria"},
            "Mixed question schema",
        )
    if kind == "choice":
        options = question.get({"oai": "choices", "systemone": "criteria"}[dialect])
        require(
            isinstance(options, dict if dialect == "systemone" else list)
            and bool(options),
            "Missing choice candidates",
        )
        values = (
            list(options)
            if dialect == "systemone"
            else [o.get("value") if isinstance(o, dict) else None for o in options]
        )
        keys = [typed(value) for value in values]
        if dialect != "oai":
            require(
                all(isinstance(value, str) for value in values),
                "Jev choices must be strings",
            )
        require(len(set(keys)) == len(keys), "Duplicate choice values")
    if kind == "score":
        levels = question.get("criteria" if dialect == "systemone" else "levels")
        require(isinstance(levels, list) and bool(levels), "Missing score levels")
        if dialect == "oai":
            require(
                all(
                    isinstance(level, dict) and isinstance(level.get("label"), str)
                    for level in levels
                ),
                "Invalid score labels",
            )


def _openai_text(value: Any) -> None:
    if isinstance(value, str):
        return
    require(
        isinstance(value, list) and bool(value),
        "OpenAI input must be text or user messages",
    )
    for message in value:
        require(
            isinstance(message, dict)
            and message.get("role") == "user"
            and not set(message) - {"role", "content", "type"},
            "Invalid input message",
        )
        require(
            message.get("type", "message") == "message", "Invalid input message type"
        )
        content = message.get("content")
        if isinstance(content, str):
            continue
        require(isinstance(content, list) and bool(content), "Missing text content")
        require(
            all(
                isinstance(part, dict)
                and set(part) == {"type", "text"}
                and part["type"] == "input_text"
                and isinstance(part["text"], str)
                for part in content
            ),
            "Only text input parts are supported",
        )


def validate_response(
    body: dict, dialect: str, request: dict | None = None
) -> DecisionSummary:
    require(
        isinstance(body, dict) and "error" not in body,
        "Response is not successful decision JSON",
    )
    if request is not None:
        validate_request(request, dialect)
    if dialect == "native_score":
        return _native(body, request)
    require(dialect in ("oai", "systemone"), "Unknown decision dialect")
    root_fields = {"model", "answers"}
    require(
        set(body) - {"usage"} == root_fields,
        "Mixed or missing response envelope fields",
    )
    require(
        isinstance(body.get("model"), str) and bool(body["model"]),
        "Missing response model",
    )
    if request is not None:
        require(
            body["model"] == request["model"],
            "Response model differs from registered qualification model",
        )
    answers = body.get("answers")
    require(
        isinstance(answers, list if dialect == "oai" else dict) and bool(answers),
        "Missing answers",
    )
    questions = None if request is None else request["questions"]
    if questions is not None:
        require(len(answers) == len(questions), "Answer count differs from request")
        if dialect != "oai":
            require(
                list(answers) == list(questions),
                "Answer identifiers/order differ from request",
            )
    answer_list = answers if isinstance(answers, list) else list(answers.values())
    question_list = (
        list(questions.values()) if isinstance(questions, dict) else questions
    )
    refusals, masses = 0, []
    for index, answer in enumerate(answer_list):
        question = None if question_list is None else question_list[index]
        refusal, mass = _answer(answer, dialect, question)
        refusals += refusal
        if mass is not None:
            masses.append(mass)
    tokens, output, cached = _usage(body.get("usage"), dialect)
    return DecisionSummary(
        len(answers), refusals, tokens, output, cached, masses or None
    )


def _usage(usage: Any, dialect: str) -> tuple[int | None, int | None, int | None]:
    if usage is None:
        return None, None, None
    require(isinstance(usage, dict), "Missing measured usage")
    tokens = integer(usage.get("input_tokens"))
    output = integer(usage.get("output_tokens"))
    require(output == 0, "P0 decision scoring must not generate tokens")
    cached = None
    if dialect == "oai":
        details = usage.get("input_tokens_details")
        out = usage.get("output_tokens_details")
        require(
            isinstance(details, dict) and isinstance(out, dict),
            "Missing OpenAI usage details",
        )
        cached = integer(details.get("cached_tokens"))
        require(cached <= tokens, "Cache reads exceed input tokens")
        require(
            integer(details.get("cache_write_tokens")) == 0,
            "Unexpected P0 cache-write accounting",
        )
        require(
            integer(out.get("reasoning_tokens")) == 0, "Unexpected reasoning tokens"
        )
    if dialect != "systemone":
        require(
            integer(usage.get("total_tokens")) == tokens + output,
            "Usage total differs from measured counters",
        )
    return tokens, output, cached


def _answer(
    answer: Any, dialect: str, question: dict | None
) -> tuple[int, float | None]:
    require(isinstance(answer, dict), "Answer must be an object")
    kind = answer.get("type")
    if dialect == "oai":
        require(
            "name" in answer
            and (answer["name"] is None or isinstance(answer["name"], str)),
            "Invalid answer name",
        )
        if question is not None:
            require(
                answer["name"] == question.get("name"),
                "Answer name/order differs from request",
            )
        if kind == "refusal":
            require(
                set(answer) == {"type", "name"},
                "Refusal contains manufactured answer data",
            )
            return 1, None
    if question is not None:
        require(kind == question["type"], "Answer type differs from request")
    predicate = {"oai": "predicate", "systemone": "noul"}[dialect]
    require(kind in (predicate, "choice", "score"), "Invalid answer type")
    allowed = {"type"}
    if dialect == "oai":
        allowed.add("name")
    if dialect == "systemone":
        allowed.add("x_label_mass")
    if kind in ("choice", "score"):
        allowed |= {kind, "probabilities"}
        allowed.add("confidence")
        if kind == "score" and dialect == "systemone":
            allowed.add("legend")
    else:
        allowed.add(
            {
                "oai": "probability",
                "systemone": "noul",
            }[dialect]
        )
    require(not set(answer) - allowed, "Mixed or unknown answer fields")
    mass = None
    if dialect == "systemone" and "x_label_mass" in answer:
        mass = probability(answer["x_label_mass"])
    elif dialect == "oai":
        require(
            "label_mass" not in answer and "x_label_mass" not in answer,
            "Mixed OpenAI answer fields",
        )
    if kind == predicate:
        probability(answer.get("probability" if dialect == "oai" else "noul"))
        return 0, mass
    values = answer.get("probabilities")
    require(
        isinstance(values, list if dialect == "oai" else dict) and bool(values),
        "Missing candidate probabilities",
    )
    if dialect == "oai":
        require(
            all(isinstance(item, dict) for item in values),
            "Invalid probability entries",
        )
        keys = [item.get("value") for item in values]
        probs = distribution([item.get("probability") for item in values])
    else:
        keys = list(values)
        probs = distribution(list(values.values()))
    if kind == "choice":
        candidate_keys = [typed(value) for value in keys]
        require(
            len(set(candidate_keys)) == len(candidate_keys),
            "Duplicate probability choices",
        )
        winner = keys[max(range(len(probs)), key=probs.__getitem__)]
        require(
            typed(answer.get("choice")) == typed(winner),
            "Choice is not the ordered modal candidate",
        )
        if question is not None:
            options = question[{"oai": "choices", "systemone": "criteria"}[dialect]]
            expected = (
                list(options)
                if dialect == "systemone"
                else [o["value"] for o in options]
            )
            require(
                candidate_keys == [typed(v) for v in expected],
                "Candidate types/order differ from request",
            )
    elif kind == "score":
        if dialect == "oai":
            require(
                all(type(key) is int for key in keys), "Score ordinals must be integers"
            )
        require(
            keys
            == (
                list(range(len(keys)))
                if dialect == "oai"
                else [str(i) for i in range(len(keys))]
            ),
            "Score levels are not ordered ordinals",
        )
        score = answer.get("score")
        require(
            type(score) in (int, float)
            and math.isfinite(score)
            and abs(score - sum(i * p for i, p in enumerate(probs))) <= 1e-9,
            "Score is not the expected ordinal value",
        )
        if question is not None:
            levels = question["criteria" if dialect == "systemone" else "levels"]
            require(len(keys) == len(levels), "Score cardinality differs from request")
            if dialect == "oai":
                require(
                    [v.get("label") for v in values] == [v["label"] for v in levels],
                    "Score labels differ from request",
                )
            if dialect == "systemone":
                require(
                    answer.get("legend") == {str(i): v for i, v in enumerate(levels)},
                    "Score legend differs from request",
                )
    confidence = probability(answer.get("confidence"))
    require(
        abs(confidence - _confidence(probs, kind)) <= 1e-9,
        "Confidence does not match published reducer",
    )
    return 0, mass


def _confidence(probs: list[float], kind: str) -> float:
    if len(probs) == 1:
        return 1.0
    count = len(probs)
    mode = max(range(count), key=probs.__getitem__)
    if kind == "choice":
        return (count * probs[mode] - 1) / (count - 1)
    distance = sum(prob * abs(i - mode) for i, prob in enumerate(probs))
    uniform = sum(abs(i - (count - 1) / 2) for i in range(count)) / count
    return max(0.0, min(1.0, 1 - distance / uniform))


def _native(body: dict, request: dict | None) -> DecisionSummary:
    meta = body.get("meta_info")
    require(isinstance(meta, dict), "Missing native score metadata")
    finish = meta.get("finish_reason")
    require(
        isinstance(finish, dict)
        and finish.get("type") == "length"
        and finish.get("length") == 0,
        "Native score is not a terminal zero-decode result",
    )
    require(body.get("output_ids", []) == [], "Native scoring generated tokens")
    tokens, output = (
        integer(meta.get("prompt_tokens")),
        integer(meta.get("completion_tokens")),
    )
    require(output == 0, "Native scoring generated tokens")
    rows = meta.get("output_token_ids_logprobs")
    require(
        isinstance(rows, list)
        and len(rows) == 1
        and isinstance(rows[0], list)
        and bool(rows[0]),
        "Missing complete candidate row",
    )
    require(
        all(isinstance(row, list) and len(row) >= 2 for row in rows[0]),
        "Invalid native score row",
    )
    ids = [integer(row[1]) for row in rows[0]]
    require(len(set(ids)) == len(ids), "Duplicate native score candidates")
    scores = [row[0] for row in rows[0]]
    require(
        all(type(s) in (int, float) and not math.isnan(s) and s <= 0 for s in scores),
        "Invalid vocabulary log probabilities",
    )
    require(any(math.isfinite(s) for s in scores), "All native candidates impossible")
    mass = sum(math.exp(s) for s in scores)
    require(mass <= 1 + 1e-6, "Native vocabulary mass exceeds one")
    if request is not None:
        require(
            ids == request["token_ids_logprob"],
            "Native candidate order/cardinality differs from request",
        )
        require(
            tokens == len(request["input_ids"]),
            "Native prompt usage differs from input IDs",
        )
    cached = None if "cached_tokens" not in meta else integer(meta["cached_tokens"])
    require(cached is None or cached <= tokens, "Cache reads exceed prompt tokens")
    return DecisionSummary(1, 0, tokens, output, cached, [min(mass, 1.0)])
