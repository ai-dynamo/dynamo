# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import math

import pytest
from dynamo_decision_perf.contracts import (
    DecisionContractError,
    validate_request,
    validate_response,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def fixture(dialect="oai"):
    if dialect == "oai":
        request = {
            "model": "m",
            "input": "x",
            "questions": [
                {
                    "type": "choice",
                    "name": "q",
                    "instructions": "pick",
                    "choices": [{"value": True}, {"value": "true"}],
                }
            ],
        }
        answer = {
            "type": "choice",
            "name": "q",
            "choice": True,
            "probabilities": [
                {"value": True, "probability": 0.8},
                {"value": "true", "probability": 0.2},
            ],
            "confidence": 0.6,
        }
        body = {
            "model": "m",
            "answers": [answer],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 0,
                "total_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    elif dialect == "systemone":
        request = {
            "model": "m",
            "state": "x",
            "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}},
        }
        body = {
            "model": "m",
            "answers": {
                "q": {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": 0.8, "b": 0.2},
                    "confidence": 0.6,
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }
    else:
        request = {
            "model": "m",
            "input": "x",
            "nvext": {"format": "sglang_native"},
            "questions": [
                {
                    "type": "choice",
                    "id": "q",
                    "question": "pick",
                    "options": [{"name": "a"}, {"name": "b"}],
                }
            ],
        }
        body = {
            "model": "m",
            "object": "decisions",
            "prompt_format_version": 1,
            "answers": {
                "q": {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": 0.8, "b": 0.2},
                    "label_mass": 0.9,
                }
            },
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 0,
                "total_tokens": 10,
                "reasoning_tokens": 0,
            },
        }
    return request, body


@pytest.mark.parametrize("dialect", ["oai", "systemone", "sglang_native"])
def test_valid_dialects_preserve_question_and_usage(dialect):
    request, body = fixture(dialect)
    original = copy.deepcopy(body)
    summary = validate_response(body, dialect, request)
    assert summary.question_count == 1 and summary.refusal_count == 0
    assert summary.input_tokens == 10 and summary.output_tokens == 0
    assert summary.cached_tokens == (0 if dialect == "oai" else None)
    assert body == original


@pytest.mark.parametrize("dialect", ["oai", "systemone", "sglang_native"])
@pytest.mark.parametrize("explicit_null", [False, True])
def test_missing_usage_is_unknown_not_zero(dialect, explicit_null):
    request, body = fixture(dialect)
    body.pop("usage")
    if explicit_null:
        body["usage"] = None
    summary = validate_response(body, dialect, request)
    assert summary.input_tokens is None
    assert summary.output_tokens is None
    assert summary.cached_tokens is None


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.1, 1.1, True])
def test_invalid_probability_is_not_a_success(bad):
    request, body = fixture()
    body["answers"][0]["probabilities"][0]["probability"] = bad
    with pytest.raises(DecisionContractError):
        validate_response(body, "oai", request)


@pytest.mark.parametrize(
    "mutation",
    ["count", "type", "choice_type", "order", "usage", "error", "name", "output"],
)
def test_malformed_success_fails_closed(mutation):
    request, body = fixture()
    answer = body["answers"][0]
    if mutation == "count":
        body["answers"] = []
    if mutation == "type":
        answer["type"] = "noul"
    if mutation == "choice_type":
        answer["choice"] = 1
    if mutation == "order":
        answer["probabilities"].reverse()
    if mutation == "usage":
        body["usage"]["input_tokens_details"].pop("cached_tokens")
    if mutation == "error":
        body["error"] = {"message": "not successful"}
    if mutation == "name":
        answer["name"] = "different"
    if mutation == "output":
        body["usage"]["output_tokens"] = 1
    with pytest.raises(DecisionContractError):
        validate_response(body, "oai", request)


def test_actual_refusal_is_counted_not_manufactured_as_answer():
    request, body = fixture()
    body["answers"] = [{"type": "refusal", "name": "q"}]
    summary = validate_response(body, "oai", request)
    assert (summary.question_count, summary.refusal_count) == (1, 1)


def test_native_requires_mass_and_jev_preserves_key_order():
    request, body = fixture("sglang_native")
    body["answers"]["q"].pop("label_mass")
    with pytest.raises(DecisionContractError):
        validate_response(body, "sglang_native", request)
    request, body = fixture("systemone")
    body["answers"]["q"]["probabilities"] = {"b": 0.2, "a": 0.8}
    with pytest.raises(DecisionContractError):
        validate_response(body, "systemone", request)


def native_fixture():
    request = {
        "model": "m",
        "input_ids": [1, 2],
        "token_ids_logprob": [3, 4],
        "sampling_params": {"max_new_tokens": 0},
        "return_logprob": True,
        "stream": True,
    }
    body = {
        "output_ids": [],
        "meta_info": {
            "finish_reason": {"type": "length", "length": 0},
            "prompt_tokens": 2,
            "completion_tokens": 0,
            "output_token_ids_logprobs": [
                [[math.log(0.8), 3, None], [math.log(0.1), 4, None]]
            ],
        },
    }
    return request, body


@pytest.mark.parametrize("dialect", ["oai", "systemone", "sglang_native"])
@pytest.mark.parametrize("kind", ["score", "predicate"])
def test_scores_and_predicates_match_each_dialect(dialect, kind):
    request, body = fixture(dialect)
    predicate = {"oai": "predicate", "systemone": "noul", "sglang_native": "yes_no"}[
        dialect
    ]
    q = {"type": "score" if kind == "score" else predicate}
    if dialect == "oai":
        q.update(name="q", instructions="rate")
        answer = {"type": q["type"], "name": "q"}
        if kind == "score":
            q["levels"] = [{"label": "low"}, {"label": "high"}]
            answer.update(
                score=0.2,
                probabilities=[
                    {"value": 0, "label": "low", "probability": 0.8},
                    {"value": 1, "label": "high", "probability": 0.2},
                ],
                confidence=0.6,
            )
        else:
            answer["probability"] = 0.8
    else:
        answer = {"type": q["type"]}
        if kind == "score":
            q["criteria" if dialect == "systemone" else "levels"] = ["low", "high"]
            answer.update(score=0.2, probabilities={"0": 0.8, "1": 0.2})
            if dialect == "systemone":
                answer.update(legend={"0": "low", "1": "high"}, confidence=0.6)
        elif dialect == "systemone":
            answer["noul"] = 0.8
        else:
            answer["probabilities"] = {"yes": 0.8, "no": 0.2}
        if dialect == "sglang_native":
            q.update(id="q", question="rate")
            answer["label_mass"] = 0.9
    request["questions"] = {"q": q} if dialect == "systemone" else [q]
    body["answers"] = [answer] if dialect == "oai" else {"q": answer}
    assert validate_response(body, dialect, request).question_count == 1


def test_wrong_confidence_and_mixed_response_are_rejected():
    request, body = fixture()
    body["answers"][0]["confidence"] = 0.9
    with pytest.raises(DecisionContractError):
        validate_response(body, "oai", request)
    request, body = fixture()
    body["answers"][0]["noul"] = 0.8
    with pytest.raises(DecisionContractError):
        validate_response(body, "oai", request)


def test_native_model_optional_but_decoding_controls_strict():
    request, _ = native_fixture()
    request.pop("model")
    validate_request(request, "native_score")
    request["sampling_params"]["max_new_tokens"] = False
    with pytest.raises(DecisionContractError):
        validate_request(request, "native_score")


@pytest.mark.parametrize("dialect", ["oai", "systemone", "sglang_native"])
def test_request_dialect_rejects_mixed_fields(dialect):
    request, _ = fixture(dialect)
    request["messages"] = []
    with pytest.raises(DecisionContractError):
        validate_request(request, dialect)


@pytest.mark.parametrize(
    "change", ["images", "thinking", "mixed_question", "native_bool"]
)
def test_unsupported_request_controls_are_not_ignored(change):
    request, _ = fixture("sglang_native")
    if change == "images":
        request["images"] = ["image"]
    if change == "thinking":
        request["chat_template_kwargs"] = {"enable_thinking": True}
    if change == "mixed_question":
        request["questions"][0]["levels"] = ["low", "high"]
    if change == "native_bool":
        request["questions"][0]["options"][0]["name"] = True
    with pytest.raises(DecisionContractError):
        validate_request(request, "sglang_native")


def test_mixed_response_root_and_nontext_openai_input_fail_closed():
    request, body = fixture()
    body["choices"] = []
    with pytest.raises(DecisionContractError):
        validate_response(body, "oai", request)
    request["input"] = {"arbitrary": "object"}
    with pytest.raises(DecisionContractError):
        validate_request(request, "oai")


@pytest.mark.parametrize("mutation", [None, "nonterminal", "decode", "ids", "mass"])
def test_native_scoring_terminal_and_candidate_contract(mutation):
    request, body = native_fixture()
    meta = body["meta_info"]
    if mutation == "nonterminal":
        meta["finish_reason"] = None
    if mutation == "decode":
        body["output_ids"] = [42]
    if mutation == "ids":
        meta["output_token_ids_logprobs"][0].reverse()
    if mutation == "mass":
        meta["output_token_ids_logprobs"][0][0][0] = 0.0
    if mutation is None:
        result = validate_response(body, "native_score", request)
        assert result.question_count == 1 and result.label_mass == pytest.approx([0.9])
    else:
        with pytest.raises(DecisionContractError):
            validate_response(body, "native_score", request)
