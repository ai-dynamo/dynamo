# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json

import pytest

from dynamo.compaction.coordinator import Counter, Pipeline
from dynamo.compaction.protocol import Request

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def request(units=None, **budget):
    units = units or [
        {"id": "u", "role": "user", "text": "Do not delete files.", "protected": False},
        {
            "id": "a",
            "role": "assistant",
            "text": "Tests failed. More details.",
            "protected": False,
        },
    ]
    snapshot = {"units": units, "context_units": [], "session_id": "cpu-session"}
    digest = hashlib.sha256(
        json.dumps(
            snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    return Request.parse(
        {
            "version": 1,
            "snapshot_sha256": digest,
            **snapshot,
            "budget": {"max_output_tokens": 4096, **budget},
        }
    )


class Models:
    def __init__(self, selections=None, claims=None, supported=True):
        self.selections = (
            [{"id": "a", "action": "summarize", "score": 0.9}]
            if selections is None
            else selections
        )
        self.claims = (
            [{"text": "Tests failed.", "source_ids": ["a"]}]
            if claims is None
            else claims
        )
        self.supported = supported
        self.calls = []

    async def select(self, units, context, session_id):
        self.calls.append("select")
        return {"selections": self.selections}

    async def compact(self, units, context, session_id):
        self.calls.append("compact")
        return {"claims": self.claims}

    async def verify(self, units, claims, context, session_id):
        self.calls.append("verify")
        return {
            "supported": self.supported,
            "unsupported_claims": [] if self.supported else [0],
        }


def run(models, req=None, counter=None, mode="fixture"):
    return asyncio.run(
        Pipeline(
            models,
            counter or Counter("fixture_bytes", lambda text: len(text.encode()), False),
            mode=mode,
        ).run(req or request())
    )


def test_accepts_grounded_summary_and_protects_user_immutably():
    req = request()
    result = run(Models(), req)
    assert result.status == "accepted"
    assert result.retained_ids == ("u",)
    assert "Tests failed." in result.summary
    assert req.units[0].protected is False
    assert result.snapshot_sha256 == req.snapshot_sha256


def test_semantic_verifier_rejects_hallucination_without_repair():
    models = Models(
        claims=[{"text": "Tests passed.", "source_ids": ["a"]}], supported=False
    )
    result = run(models)
    assert result.status == "rejected"
    assert result.reason == "semantic_rejection"
    assert models.calls == ["select", "compact", "verify"]
    assert result.summary is None
    assert result.retained_ids == ("u", "a")


@pytest.mark.parametrize(
    "selections",
    [
        [],
        [{"id": "a", "action": "unknown", "score": None}],
        [{"id": "a", "action": "summarize", "score": None}],
    ],
)
def test_missing_or_unknown_selection_preserves_original(selections):
    result = run(Models(selections=selections))
    assert result.status == "rejected"
    assert result.retained_ids == ("u", "a")


@pytest.mark.parametrize("score", [True, float("nan"), float("inf"), -1, 2])
def test_invalid_scores_cannot_authorize_compaction(score):
    assert (
        run(
            Models(selections=[{"id": "a", "action": "summarize", "score": score}])
        ).status
        == "rejected"
    )


@pytest.mark.parametrize(
    "selections",
    [
        [{"id": "u", "action": "summarize", "score": 1}],
        [{"id": "unknown", "action": "summarize", "score": 1}],
        [{"id": "a", "action": "summarize", "score": 1}] * 2,
    ],
)
def test_protected_unknown_or_duplicate_selector_ids_rejected(selections):
    assert run(Models(selections=selections)).status == "rejected"


@pytest.mark.parametrize(
    "claims",
    [
        [{"text": "injected", "source_ids": ["missing"]}],
        [{"text": "injected", "source_ids": ["u"]}],
        [{"text": "injected", "source_ids": []}],
        [{"text": "a", "source_ids": ["a"], "path": "/tmp/generated"}],
        [],
    ],
)
def test_invalid_citations_and_schema_fail_before_verification(claims):
    models = Models(claims=claims)
    assert run(models).status == "rejected"
    assert "verify" not in models.calls


def test_model_execution_requires_qualified_counter():
    models = Models()
    assert run(models, mode="model").reason == "unqualified_counter"
    assert models.calls == []


def test_budget_failure_preserves_all_context():
    models = Models()
    result = run(models, request(max_output_tokens=1))
    assert result.reason == "protected_budget_exceeded"
    assert models.calls == []


def test_call_budget_preflight_and_invalid_counter():
    models = Models()
    assert run(models, request(max_model_calls=2)).reason == "call_budget_exceeded"
    assert models.calls == []
    assert (
        run(models, counter=Counter("invalid", lambda text: True, True)).reason
        == "invalid_counter"
    )


@pytest.mark.parametrize("bad", [True, 0, -1, 1000000])
def test_invalid_budget_rejected_before_execution(bad):
    with pytest.raises(ValueError):
        request(max_output_tokens=bad)


def test_low_selection_score_keeps_original():
    assert (
        run(
            Models(selections=[{"id": "a", "action": "summarize", "score": 0.1}])
        ).status
        == "rejected"
    )


def test_full_context_is_available_to_every_stage():
    class ContextModels(Models):
        async def select(self, units, context, session_id):
            assert context[0].role == "user" and units[0].id == "a"
            return await super().select(units, context, session_id)

        async def compact(self, units, context, session_id):
            assert context[0].text == "Do not delete files."
            return await super().compact(units, context, session_id)

        async def verify(self, units, claims, context, session_id):
            assert context[0].must_retain
            return await super().verify(units, claims, context, session_id)

    assert run(ContextModels()).status == "accepted"
