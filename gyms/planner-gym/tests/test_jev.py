# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real HTTP boundary with local responses, never a hosted model."""

import asyncio
import hashlib
import importlib
import json

import pytest
from autoscaling_arena.jev_report import summarize_decisions
from feedback_fixtures import knowline_context, sample

planner_types = pytest.importorskip("dynamo.planner.core.types")
httpx = pytest.importorskip("httpx")
TickInput = planner_types.TickInput
WorkerCounts = planner_types.WorkerCounts
JevAutoscaler = importlib.import_module("autoscaling_arena.adapters.jev").JevAutoscaler
validate_answer = importlib.import_module(
    "autoscaling_arena.adapters.jev"
)._validate_answer
engine_protocol = importlib.import_module("dynamo.planner.core.engine_protocol")


@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.pre_merge
def test_jev_keeps_the_common_tick_contract_without_runtime_reconfiguration(tmp_path):
    engine = make_engine(tmp_path, lambda request: httpx.Response(200, json={}))
    try:
        assert isinstance(engine, engine_protocol.EngineProtocol)
        assert isinstance(engine.initial_tick(0.0), planner_types.ScheduledTick)
    finally:
        asyncio.run(engine.shutdown())


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


def test_self_hosted_fixed_pools_do_not_make_model_calls(tmp_path):
    def handler(request):
        pytest.fail("A fixed allocation must not call the model endpoint")

    engine = make_engine(
        tmp_path,
        handler,
        provider="self_hosted",
        base_url="http://127.0.0.1",
        model="PelaAI/KnowLine-31B-Gen1",
        min_prefill=1,
        max_prefill=1,
        min_decode=1,
        max_decode=1,
        failure_mode="raise",
        context=knowline_context(),
    )
    engine.on_telemetry(sample())
    tick = TickInput(
        now_s=0,
        worker_counts=WorkerCounts(
            ready_num_prefill=1,
            expected_num_prefill=1,
            ready_num_decode=1,
            expected_num_decode=1,
        ),
    )

    async def run():
        try:
            result = await engine.tick(engine.initial_tick(0), tick)
            assert result.scale_to.num_prefill == result.scale_to.num_decode == 1
        finally:
            await engine.shutdown()

    asyncio.run(run())
    record = json.loads((tmp_path / "decisions.jsonl").read_text())
    assert record["response_source"] == "forced_bounds"
    assert record["api_calls_used"] == 0
    assert record["status"] == "ok"
    summary = summarize_decisions(tmp_path / "decisions.jsonl")
    assert summary["calls"] == 0
    assert summary["forced_decisions"] == 1
    assert summary["total_wall_latency_s"] == 0


def test_self_hosted_uses_joint_choices_even_when_one_pool_is_fixed(tmp_path):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert set(payload["questions"]) == {"fleet"}
        assert all(
            fleet["prefill_workers"] == 1
            for fleet in payload["questions"]["fleet"]["criteria"].values()
        )
        result = response_for(payload, choices={"fleet": "p=1,d=4"})
        result["model"] = "PelaAI/KnowLine-31B-Gen1"
        return httpx.Response(200, json=result)

    engine = make_engine(
        tmp_path,
        handler,
        provider="self_hosted",
        base_url="http://127.0.0.1",
        model="PelaAI/KnowLine-31B-Gen1",
        min_prefill=1,
        max_prefill=1,
        min_decode=1,
        max_decode=8,
        failure_mode="raise",
        context=knowline_context(44),
    )
    engine.on_telemetry(sample(d=1, starting_d=3))
    engine.on_telemetry(sample(15, d=1, starting_d=3, gpu_seconds=540))

    async def run():
        try:
            result = await engine.tick(engine.initial_tick(15), snapshot())
            assert result.scale_to.num_prefill == 1
            assert result.scale_to.num_decode == 4
        finally:
            await engine.shutdown()

    asyncio.run(run())
    assert len(requests) == 1
    record = json.loads((tmp_path / "decisions.jsonl").read_text())
    assert "forced_pools" not in record
    assert record["response_source"] == "api"
    assert record["api_calls_used"] == 1


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
    assert row["error_stage"] == {
        "http": "http_or_json",
        "model": "model",
        "missing": "answer_keys",
    }.get(fault, "choice_decode")
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


@pytest.mark.parametrize(
    "probabilities",
    [
        {"3": 0.05, "4": 0.93, "5": 0.01},
        {"3": 0.06, "4": 0.94, "5": 0.01},
    ],
)
def test_hosted_rounded_probabilities_preserve_model_choice(tmp_path, probabilities):
    def handler(request):
        result = response_for(json.loads(request.content))
        result["answers"]["decode"]["probabilities"] = probabilities
        return httpx.Response(200, json=result)

    engine = make_engine(tmp_path, handler)

    async def run():
        try:
            result = await engine.tick(engine.initial_tick(15), snapshot())
            assert result.scale_to.num_decode == 4
        finally:
            await engine.shutdown()

    asyncio.run(run())
    row = json.loads((tmp_path / "decisions.jsonl").read_text())
    assert row["status"] == "ok"
    assert row["answers"]["decode"]["probabilities"] == probabilities


@pytest.mark.parametrize(
    "probabilities",
    [
        {"3": 0.05, "4": 0.91, "5": 0.01},
        {"3": 0.07, "4": 0.95, "5": 0.01},
        {"3": 0.051, "4": 0.93, "5": 0.01},
        {"3": -0.01, "4": 1, "5": 0.01},
    ],
)
def test_rounding_tolerance_does_not_accept_invalid_distributions(probabilities):
    with pytest.raises(ValueError):
        validate_answer(
            {
                "type": "choice",
                "choice": "4",
                "confidence": 0.9,
                "probabilities": probabilities,
            },
            {"3": {}, "4": {}, "5": {}},
        )


def test_recovery_reuses_exact_decisions_and_enforces_remaining_api_budget(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response_for(requests[-1]))

    first = make_engine(original, handler)

    async def initial():
        try:
            for at in (15, 30):
                await first.tick(first.initial_tick(at), snapshot(at))
        finally:
            await first.shutdown()

    asyncio.run(initial())
    source = original / "decisions.jsonl"
    engine = make_engine(
        recovery,
        handler,
        prior_calls=3,
        max_calls=4,
        replay_decisions=str(source),
        replay_decisions_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )

    async def recover():
        try:
            for at in (15, 30, 45):
                result = await engine.tick(engine.initial_tick(at), snapshot(at))
                assert result.scale_to.num_decode == 4
            with pytest.raises(RuntimeError, match="max_calls exhausted"):
                await engine.tick(engine.initial_tick(60), snapshot(60))
        finally:
            await engine.shutdown()

    asyncio.run(recover())
    assert len(requests) == 3  # Two original calls, then one new call.
    summary = summarize_decisions(recovery / "decisions.jsonl")
    assert summary["calls"] == 1
    assert summary["replayed_decisions"] == 2
    assert summary["reported_input_tokens"] == 300
    rows = [
        json.loads(line)
        for line in (recovery / "decisions.jsonl").read_text().splitlines()
    ]
    assert [row["api_calls_used"] for row in rows] == [3, 3, 4]


@pytest.mark.parametrize("fault", ["checksum", "request"])
def test_recovery_refuses_changed_log_or_observations(tmp_path, fault):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response_for(json.loads(request.content)))

    first = make_engine(tmp_path, handler)

    async def initial():
        try:
            await first.tick(first.initial_tick(15), snapshot())
        finally:
            await first.shutdown()

    asyncio.run(initial())
    source = tmp_path / "decisions.jsonl"
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    kwargs = dict(
        prior_calls=1, replay_decisions=str(source), replay_decisions_sha256=checksum
    )
    if fault == "checksum":
        source.write_text(source.read_text() + "\n")
        with pytest.raises(ValueError, match="checksum mismatch"):
            JevAutoscaler(**kwargs)
    else:
        engine = make_engine(recovery, handler, **kwargs)

        async def recover():
            try:
                with pytest.raises(ValueError, match="request differs"):
                    await engine.tick(engine.initial_tick(16), snapshot(16))
            finally:
                await engine.shutdown()

        asyncio.run(recover())
    assert len(requests) == 1
