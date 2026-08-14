# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for the output-quality detector.

Needs no deployment and no GPU, so the detector itself is regression-tested in
pre-merge even though the tests that use it only run against a live endpoint.

The degenerate strings below are the real thing: captured verbatim from a
Kimi-K2.5 replica on GB300 that emitted token 0 (``!``) for every generated
token. The healthy strings are captured from the healthy replicas of the same
deployment, so the thresholds are calibrated against real output on both sides
rather than invented examples.
"""

import pytest

from tests.recipes.helpers import (
    KNOWN_ANSWER_PROBES,
    answer_text,
    assert_answers,
    assert_natural_language,
    degeneracy_reason,
    message_text,
    unique_prompt,
    worker_id_of,
    wrong_answer_reason,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
]

_HEALTHY_REASONING = (
    'The user is asking a simple factual question: "What is the capital of '
    'France?" This is straightforward general knowledge. The answer is Paris. '
    "I should give a clear, direct response."
)
_HEALTHY_CONTENT = (
    "I don't have access to real-time weather data or the internet, so I "
    "cannot tell you the current weather in Paris."
)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("!" * 100, id="token0_100"),
        pytest.param("!" * 3000, id="token0_3000"),
        pytest.param("The user is asking" + "!" * 784, id="collapse_midway"),
        pytest.param("", id="empty"),
        pytest.param("   \n  ", id="whitespace_only"),
        pytest.param("." * 26, id="punctuation_only"),
    ],
)
def test_degenerate_output_is_flagged(text):
    assert degeneracy_reason(text) is not None
    with pytest.raises(AssertionError):
        assert_natural_language(text)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(_HEALTHY_REASONING, id="real_reasoning"),
        pytest.param(_HEALTHY_CONTENT, id="real_content"),
        pytest.param("4", id="terse_answer"),
        pytest.param("Paris.", id="terse_with_punctuation"),
        pytest.param("ok ok", id="short_repetitive_but_valid"),
        pytest.param("def f(x):\n    return x*2\n\nprint(f(21))", id="code"),
        pytest.param(
            "**Paris** is the capital of France.\n\n- Population: 2.1M\n"
            "- Region: Ile-de-France",
            id="markdown",
        ),
        pytest.param("1, 2, 3, 4, 5, 6, 7, 8, 9, 10", id="repeated_separators"),
    ],
)
def test_healthy_output_is_not_flagged(text):
    """No false positives: terse, repetitive-but-valid and non-prose all pass.

    A false positive here is worse than a false negative -- it would fail real
    deployments on correct answers like "4".
    """
    assert degeneracy_reason(text) is None
    assert_natural_language(text)


def test_message_text_joins_reasoning_and_content():
    """Both fields count, because a stuck decoder leaves content empty."""
    body = {
        "choices": [
            {"message": {"reasoning_content": "thinking", "content": " answer"}}
        ]
    }
    assert message_text(body) == "thinking answer"


def test_message_text_tolerates_missing_fields():
    assert message_text({}) == ""
    assert message_text({"choices": [{"message": {}}]}) == ""
    assert message_text({"choices": [{"message": {"content": None}}]}) == ""


@pytest.mark.parametrize(
    "body,expected",
    [
        ({"worker_id": 123}, "123"),
        ({"choices": [{"nvext": {"worker_id": "abc"}}]}, "abc"),
        ({"nvext": {"prefill_worker_id": 7}}, "7"),
        ({"choices": [{"message": {}}]}, None),
    ],
)
def test_worker_id_is_found_at_any_depth(body, expected):
    assert worker_id_of(body) == expected


def test_unique_prompt_varies_the_prefix():
    """Two calls must differ, or KV routing pins them to the same replica."""
    a, b = unique_prompt("hello"), unique_prompt("hello")
    assert a != b
    assert a.endswith("hello") and b.endswith("hello")


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("Paris.", id="bare"),
        pytest.param("The capital of France is Paris.", id="sentence"),
        pytest.param("paris", id="lowercase"),
        pytest.param("It is **PARIS**, the largest city in France.", id="markdown"),
    ],
)
def test_correct_answer_accepted(text):
    assert wrong_answer_reason(text, ("paris",)) is None


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("The capital of France is Lyon.", id="wrong_city"),
        pytest.param("I don't know.", id="refusal"),
        pytest.param("The capital is Parisian architecture.", id="substring_only"),
    ],
)
def test_incorrect_answer_rejected(text):
    """Fluent, well-formed and wrong must fail -- that is the whole point.

    ``Parisian`` is included because a naive substring check would accept it
    while the model never actually named the city.
    """
    assert degeneracy_reason(text) is None, "these are all valid language"
    assert wrong_answer_reason(text, ("paris",)) is not None


@pytest.mark.parametrize(
    "text,accepted,should_match",
    [
        pytest.param("The answer is 4.", ("4",), True, id="digit_alone"),
        pytest.param("There are 24 hours.", ("4",), False, id="digit_inside_number"),
        pytest.param("It orbits the Sun.", ("sun",), True, id="word"),
        pytest.param("See you on Sunday.", ("sun",), False, id="word_inside_word"),
        pytest.param("The answer is four.", ("4", "four"), True, id="alternate_form"),
    ],
)
def test_matching_is_whole_word(text, accepted, should_match):
    """Word boundaries stop '4' matching '24' and 'sun' matching 'Sunday'."""
    assert (wrong_answer_reason(text, accepted) is None) is should_match


def test_degeneracy_is_reported_before_incorrectness():
    """A stuck decoder must report the decoder, not a missing keyword."""
    with pytest.raises(AssertionError, match="repeating one token"):
        assert_answers("The" + "!" * 400, ("paris",))


def test_answer_text_prefers_content_over_reasoning():
    """Reasoning explores wrong answers; only the conclusion is the claim."""
    body = {
        "choices": [
            {
                "message": {
                    "reasoning_content": "Maybe Lyon? No, it is Paris.",
                    "content": "Paris",
                }
            }
        ]
    }
    assert answer_text(body) == "Paris"


def test_answer_text_falls_back_when_content_empty():
    body = {"choices": [{"message": {"reasoning_content": "thinking", "content": ""}}]}
    assert answer_text(body) == "thinking"


def test_known_answer_probes_are_well_formed():
    """Each probe must ask a question and accept at least one answer."""
    assert KNOWN_ANSWER_PROBES
    for prompt, accepted in KNOWN_ANSWER_PROBES:
        assert prompt.strip()
        assert accepted, f"no accepted answers for {prompt!r}"
        assert all(
            a == a.lower() for a in accepted
        ), f"accepted answers should be lowercase for readability: {accepted}"
