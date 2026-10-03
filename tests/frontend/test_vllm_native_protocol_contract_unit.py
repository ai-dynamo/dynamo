# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the native HTTP differential oracle, without a model."""

import copy
import json
from types import SimpleNamespace

import pytest

from tests.frontend import test_vllm_mixed_release_http as mixed_release
from tests.frontend.test_vllm_mixed_release_http import assert_full_vocab_boundary
from tests.frontend.test_vllm_native_protocol_http import (
    _assert_native_prompt_count,
    _generated_logprobs,
    _matches_native,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.parametrize("direction", ["new-frontend", "old-frontend"])
def test_legacy_full_vocab_oracle_rejects_success_sse_and_missing_errors(direction):
    def body(message):
        error = {
            "message": message,
            "code": 400,
            "type": "BadRequestError",
            "param": "prompt_logprobs",
        }
        return json.dumps({"error": error} if direction == "new-frontend" else error)

    record = {
        "status": 400,
        "content_type": "application/json",
        "body": body("prompt_logprobs -1 unavailable: expected u32"),
    }
    assert_full_vocab_boundary(record, direction=direction)
    for field, value in (
        ("status", 200),
        ("content_type", "text/event-stream"),
        ("body", body("")),
    ):
        with pytest.raises(AssertionError):
            assert_full_vocab_boundary({**record, field: value}, direction=direction)
    unrelated_error = {
        **record,
        "body": body("model missing"),
    }
    with pytest.raises(AssertionError):
        assert_full_vocab_boundary(unrelated_error, direction="new-frontend")
    with pytest.raises(AssertionError):
        assert_full_vocab_boundary(unrelated_error, direction="old-frontend")


@pytest.mark.parametrize("endpoint", ["chat/completions", "completions"])
def test_full_vocab_oracle_requires_every_token_and_correct_placement(endpoint):
    payload = [
        None,
        {
            str(token): {"logprob": -1.0, "rank": token + 1, "decoded_token": "x"}
            for token in range(3)
        },
    ]
    body = {"choices": [{}], "usage": {"prompt_tokens": 2}}
    owner = body if endpoint == "chat/completions" else body["choices"][0]
    owner["prompt_logprobs"] = payload
    record = {
        "request": {"prompt_logprobs": -1, "stream": False},
        "status": 200,
        "content_type": "application/json",
        "body": json.dumps(body),
    }
    _assert_native_prompt_count(record, endpoint, max_logprobs=-1, vocab_size=3)
    for invalid in (
        [None, {"0": payload[1]["0"]}],
        [None, {**payload[1], "3": payload[1]["0"]}],
        [None, {"1": payload[1]["0"], "2": payload[1]["1"], "3": payload[1]["2"]}],
        [None, {**payload[1], "0": {**payload[1]["0"], "logprob": float("nan")}}],
        [None, {**payload[1], "0": {**payload[1]["0"], "rank": 0}}],
        None,
    ):
        owner["prompt_logprobs"] = invalid
        with pytest.raises(AssertionError):
            _assert_native_prompt_count(
                {**record, "body": json.dumps(body)},
                endpoint,
                max_logprobs=-1,
                vocab_size=3,
            )


@pytest.mark.parametrize("endpoint", ["chat/completions", "completions"])
@pytest.mark.parametrize("stream,max_logprobs", [(True, -1), (False, 20)])
def test_prompt_count_oracle_requires_pre_stream_error(endpoint, stream, max_logprobs):
    error = {
        "code": 400,
        "type": "BadRequestError",
        "param": "prompt_logprobs",
        "message": "unavailable",
    }
    record = {
        "request": {"prompt_logprobs": -1, "stream": stream},
        "status": 400,
        "content_type": "application/json",
        "body": json.dumps({"error": error}),
    }
    _assert_native_prompt_count(
        record, endpoint, max_logprobs=max_logprobs, vocab_size=3
    )
    for field, value in (
        ("status", 200),
        ("content_type", "text/event-stream"),
        (
            "body",
            json.dumps({"error": {**error, "param": "model"}}),
        ),
        ("body", json.dumps(error)),
        ("body", json.dumps({"error": {**error, "code": 500}})),
        ("body", json.dumps({"error": {**error, "type": "Bad Request"}})),
    ):
        with pytest.raises(AssertionError):
            _assert_native_prompt_count(
                {**record, field: value},
                endpoint,
                max_logprobs=max_logprobs,
                vocab_size=3,
            )
    if not stream:
        # A default-limit rejection cannot satisfy a full-vocab positive control.
        with pytest.raises(AssertionError):
            _assert_native_prompt_count(record, endpoint, max_logprobs=-1, vocab_size=3)


def _chunk(text, tokens, offsets, index=0):
    return {
        "choices": [
            {
                "index": index,
                "text": text,
                "logprobs": {
                    "tokens": tokens,
                    "text_offset": offsets,
                    "token_logprobs": [-0.5] * len(tokens),
                    "top_logprobs": [{token: -0.5} for token in tokens],
                },
            }
        ]
    }


def test_completion_offsets_allow_different_native_sse_boundaries():
    one_at_a_time = [
        _chunk("é", ["token_id:1"], [0]),
        _chunk("🦀", ["token_id:22"], [1]),
        _chunk("z", ["token_id:3"], [2]),
    ]
    coalesced = [
        _chunk("é", ["token_id:1"], [0]),
        _chunk("🦀z", ["token_id:22", "token_id:3"], [1, 12]),
    ]
    assert _generated_logprobs(
        one_at_a_time, "completions", stream=True
    ) == _generated_logprobs(coalesced, "completions", stream=True)

    for wrong_offsets in ([1, 2], [], [2, 13]):
        invalid = copy.deepcopy(coalesced)
        invalid[1]["choices"][0]["logprobs"]["text_offset"] = wrong_offsets
        with pytest.raises(AssertionError):
            _generated_logprobs(invalid, "completions", stream=True)


def test_completion_offsets_validate_unary_and_independent_choices():
    records = [
        _chunk("é🦀", ["é", "🦀"], [0, 1]),
        _chunk("x", ["x"], [0], index=1),
        _chunk("z", ["z"], [2]),
    ]
    _generated_logprobs(records, "completions", stream=True)
    unary = [_chunk("é🦀z", ["token_id:1", "token_id:22", "token_id:3"], [0, 10, 21])]
    _generated_logprobs(unary, "completions")
    unary[0]["choices"][0]["logprobs"]["text_offset"] = [0, 1, 2]
    with pytest.raises(AssertionError):
        _generated_logprobs(unary, "completions")


@pytest.mark.parametrize("endpoint", ["chat/completions", "completions"])
def test_choice_identity_survives_stream_interleaving(endpoint):
    records = [
        _chunk("a", ["a"], [0]),
        _chunk("x", ["x"], [0], index=1),
        _chunk("b", ["b"], [1]),
        _chunk("y", ["y"], [1], index=1),
    ]
    if endpoint == "chat/completions":
        for record in records:
            choice = record["choices"][0]
            choice["logprobs"] = {
                "content": [{"token": choice["text"], "logprob": -0.5}]
            }
    expected = _generated_logprobs(records, endpoint, stream=True)
    reordered = [records[1], records[3], records[0], records[2]]
    assert _matches_native(
        expected, _generated_logprobs(reordered, endpoint, stream=True)
    )
    swapped = copy.deepcopy(records)
    for record in swapped:
        record["choices"][0]["index"] = 1 - record["choices"][0]["index"]
    assert not _matches_native(
        expected, _generated_logprobs(swapped, endpoint, stream=True)
    )


@pytest.mark.parametrize("index", [-1, True, "0"])
def test_choice_index_requires_nonnegative_integer(index):
    with pytest.raises(AssertionError):
        _generated_logprobs([_chunk("a", ["a"], [0], index)], "completions")


def test_duplicate_choice_indices_are_not_silently_merged():
    record = _chunk("a", ["a"], [0])
    duplicate = copy.deepcopy(record)
    duplicate["choices"].extend(record["choices"])
    with pytest.raises(AssertionError):
        _generated_logprobs([duplicate], "completions", stream=True)
    with pytest.raises(AssertionError):
        _generated_logprobs([record, record], "completions")


@pytest.mark.parametrize("pending_checks", [0, 2])
def test_container_removal_wait_observes_eventual_absence(monkeypatch, pending_checks):
    responses = iter(
        [SimpleNamespace(returncode=0, stderr="")] * pending_checks
        + [SimpleNamespace(returncode=1, stderr="Error: No such object: owned-test")]
    )
    sleeps = []
    monkeypatch.setattr(
        mixed_release.subprocess, "run", lambda *a, **k: next(responses)
    )
    monkeypatch.setattr(mixed_release.time, "monotonic", lambda: 0)
    monkeypatch.setattr(mixed_release.time, "sleep", sleeps.append)
    mixed_release._wait_for_container_removal("owned-test")
    assert sleeps == [0.1] * pending_checks


@pytest.mark.parametrize("daemon_failure", [False, True])
def test_container_removal_wait_does_not_mask_leaks_or_daemon_failure(
    monkeypatch, daemon_failure
):
    clock = iter([0, 31])
    response = SimpleNamespace(
        returncode=1 if daemon_failure else 0,
        stderr="Cannot connect to the Docker daemon" if daemon_failure else "",
    )
    monkeypatch.setattr(mixed_release.subprocess, "run", lambda *a, **k: response)
    monkeypatch.setattr(mixed_release.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        mixed_release.time, "sleep", lambda _: pytest.fail("must fail without sleeping")
    )
    message = "cannot verify" if daemon_failure else "was not removed"
    with pytest.raises(RuntimeError, match=message):
        mixed_release._wait_for_container_removal("owned-test")
