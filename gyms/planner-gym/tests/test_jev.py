# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real HTTP boundary with local responses, never a hosted model."""

import asyncio
import importlib
import json

import pytest
from autoscaling_arena.jev_report import summarize_decisions

planner_types = pytest.importorskip("dynamo.planner.core.types")
httpx = pytest.importorskip("httpx")
TickInput = planner_types.TickInput
WorkerCounts = planner_types.WorkerCounts
JevAutoscaler = importlib.import_module("autoscaling_arena.adapters.jev").JevAutoscaler


def snapshot(at=15):
    return TickInput(
        now_s=at,
        worker_counts=WorkerCounts(
            ready_num_prefill=1,
            expected_num_prefill=1,
            ready_num_decode=1,
            expected_num_decode=4,
            decode_scaling_in_progress=True,
        ),
    )


def response_for(payload, choices=None, confidence=0.9):
    choices = choices or {"prefill": "2", "decode": "4"}
    return {
        "model": "jev-1.13.0",
        "answers": {
            role: {
                "type": "choice",
                "choice": choices[role],
                "probabilities": {
                    key: float(key == choices[role]) for key in question["criteria"]
                },
                "confidence": confidence,
            }
            for role, question in payload["questions"].items()
        },
        "usage": {"input_tokens": 300, "output_tokens": 0},
    }


def make_engine(tmp_path, handler, **kwargs):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.typesafe.ai"
    )
    return JevAutoscaler(
        client=client, decision_log=tmp_path / "decisions.jsonl", **kwargs
    )


def test_model_choices_drive_targets_and_hold_preserves_pending_workers(tmp_path):
    requests = []

    def handler(request):
        assert request.url.path == "/v1/systemone"
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json=response_for(payload))

    engine = make_engine(tmp_path, handler, history_ticks=1)

    async def run():
        try:
            for at in (15, 30, 45):
                result = await engine.tick(engine.initial_tick(at), snapshot(at))
                assert result.scale_to.num_prefill == 2
                assert result.scale_to.num_decode == 4
                assert result.next_tick.at_s == at + 15
        finally:
            await engine.shutdown()

    asyncio.run(run())
    assert len(requests[0]["questions"]) == 2
    assert set(requests[0]["questions"]["decode"]["criteria"]) == {"3", "4", "5"}
    assert (
        requests[0]["state"]["current"]["pools"]["decode"]["waiting_requests"] is None
    )
    assert requests[0]["state"]["history"] == []
    assert [s["now_s"] for s in requests[2]["state"]["history"]] == [30]
    summary = summarize_decisions(tmp_path / "decisions.jsonl")
    assert summary["calls"] == 3
    assert summary["errors"] == 0
    assert summary["reported_input_tokens"] == 900
    assert summary["models"] == ["jev-1.13.0"]


@pytest.mark.parametrize(
    "fault", ["target", "missing", "nan", "wrong_type", "distribution", "model", "http"]
)
def test_invalid_response_aborts_and_logs_without_applying_partial_decision(
    tmp_path, fault
):
    def handler(request):
        if fault == "http":
            return httpx.Response(401, text="a-secret-that-must-not-be-logged")
        result = response_for(json.loads(request.content))
        answer = result["answers"]["decode"]
        if fault == "target":
            answer["choice"] = "999"
        elif fault == "missing":
            del result["answers"]["decode"]
        elif fault == "nan":
            answer["confidence"] = float("nan")
        elif fault == "wrong_type":
            answer["type"] = "score"
        elif fault == "model":
            result["model"] = "unexpected-model-version"
        else:
            answer["probabilities"]["3"] = 0.5
        return httpx.Response(200, content=json.dumps(result))

    engine = make_engine(tmp_path, handler)

    async def run():
        try:
            with pytest.raises(RuntimeError, match="Jev decision failed"):
                await engine.tick(engine.initial_tick(15), snapshot())
        finally:
            await engine.shutdown()

    asyncio.run(run())
    text = (tmp_path / "decisions.jsonl").read_text()
    assert "a-secret" not in text
    row = json.loads(text)
    assert row["status"] == "error"
    assert row["targets"] == {"prefill": 1, "decode": 4}


def test_timeout_hold_is_explicit_and_call_budget_is_hard(tmp_path):
    async def handler(request):
        await asyncio.sleep(1)
        raise AssertionError("deadline should cancel the call")

    engine = make_engine(
        tmp_path, handler, timeout_s=0.01, failure_mode="hold", max_calls=1
    )

    async def run():
        try:
            result = await engine.tick(engine.initial_tick(15), snapshot())
            assert result.scale_to.num_decode == 4
            with pytest.raises(RuntimeError, match="max_calls exhausted"):
                await engine.tick(engine.initial_tick(30), snapshot(30))
        finally:
            await engine.shutdown()

    asyncio.run(run())
    assert summarize_decisions(tmp_path / "decisions.jsonl")["errors"] == 1


def test_optional_confidence_gate_holds_absolute_target(tmp_path):
    def handler(request):
        return httpx.Response(
            200, json=response_for(json.loads(request.content), confidence=0.2)
        )

    engine = make_engine(tmp_path, handler, min_confidence=0.8)

    async def run():
        try:
            result = await engine.tick(engine.initial_tick(15), snapshot())
            assert result.scale_to.num_prefill == 1
            assert result.scale_to.num_decode == 4
        finally:
            await engine.shutdown()

    asyncio.run(run())
    assert (
        summarize_decisions(tmp_path / "decisions.jsonl")["confidence_gated_ticks"] == 1
    )


def test_aggregated_mode_and_upper_bound(tmp_path):
    def handler(request):
        payload = json.loads(request.content)
        assert set(payload["questions"]) == {"decode"}
        assert set(payload["questions"]["decode"]["criteria"]) == {"3", "4"}
        return httpx.Response(200, json=response_for(payload, {"decode": "3"}))

    engine = make_engine(tmp_path, handler, mode="agg", max_decode=4)

    async def run():
        try:
            result = await engine.tick(engine.initial_tick(15), snapshot())
            assert result.scale_to.num_prefill is None
            assert result.scale_to.num_decode == 3
        finally:
            await engine.shutdown()

    asyncio.run(run())


def test_missing_key_fails_before_http_client_creation(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        JevAutoscaler()
