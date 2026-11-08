# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract checks for the explicit external decision execution profiles."""

import copy

import pytest

from tests.utils.decision_api import assert_decision_body, decision_payload

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def response(dialect, output_tokens):
    route = {
        "type": "choice",
        "choice": "billing",
        "confidence": 0.6,
        "probabilities": {"billing": 0.6, "technical": 0.4},
    }
    urgent = {"type": "noul", "noul": 0.6}
    severity = {
        "type": "score",
        "score": 0.8,
        "confidence": 0.4,
        "probabilities": {"0": 0.4, "1": 0.4, "2": 0.2},
        "legend": {"0": "Low", "1": "Moderate", "2": "High"},
    }
    usage = {"input_tokens": 21, "output_tokens": output_tokens}
    answers = {"route": route, "urgent": urgent, "severity": severity}
    if dialect == "oai":
        answers = [
            {
                **{
                    key: value for key, value in route.items() if key != "probabilities"
                },
                "name": "route",
                "probabilities": [
                    {"value": value, "probability": probability}
                    for value, probability in route["probabilities"].items()
                ],
            },
            {"type": "predicate", "name": "urgent", "probability": 0.6},
            {
                "type": "score",
                "name": "severity",
                "score": 0.8,
                "confidence": 0.4,
                "probabilities": [
                    {"value": index, "label": label, "probability": probability}
                    for index, (label, probability) in enumerate(
                        zip(("Low", "Moderate", "High"), (0.4, 0.4, 0.2))
                    )
                ],
            },
        ]
        usage = {
            **usage,
            "total_tokens": 21 + output_tokens,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
    return {"model": "model", "answers": answers, "usage": usage}


@pytest.mark.parametrize("dialect", ("oai", "systemone"))
@pytest.mark.parametrize(
    "profile,output_tokens",
    (("sglang_serial", 0), ("vllm_causal", 3), ("clef_flash", 0)),
)
def test_execution_profiles_require_actual_output_accounting(
    dialect, profile, output_tokens
):
    body = response(dialect, output_tokens)
    assert_decision_body(body, "model", dialect, execution_profile=profile)
    invalid = copy.deepcopy(body)
    invalid["usage"]["output_tokens"] = 0 if output_tokens else 3
    if dialect == "oai":
        invalid["usage"]["total_tokens"] = 21 + invalid["usage"]["output_tokens"]
    with pytest.raises(AssertionError):
        assert_decision_body(invalid, "model", dialect, execution_profile=profile)


@pytest.mark.parametrize("profile", ("sglang_serial", "vllm_causal", "clef_flash"))
def test_profile_keeps_semantic_fixture_and_only_omits_unsupported_clef_controls(
    profile,
):
    payload = decision_payload("model", "systemone", execution_profile=profile)
    assert list(payload["questions"]) == ["route", "urgent", "severity"]
    assert ("chat_template_kwargs" in payload) == (profile != "clef_flash")
    assert decision_payload("model", execution_profile=profile) == decision_payload(
        "model"
    )


def test_default_profile_preserves_existing_sglang_contract():
    assert decision_payload("model", "systemone")["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    for dialect in ("oai", "systemone"):
        assert_decision_body(response(dialect, 0), "model", dialect)


@pytest.mark.parametrize("dialect", ("oai", "systemone"))
def test_unknown_execution_profile_is_rejected(dialect):
    with pytest.raises(ValueError, match="execution profile"):
        decision_payload("model", dialect, execution_profile="unknown")
    with pytest.raises(ValueError, match="execution profile"):
        assert_decision_body(
            response(dialect, 0), "model", dialect, execution_profile="unknown"
        )
