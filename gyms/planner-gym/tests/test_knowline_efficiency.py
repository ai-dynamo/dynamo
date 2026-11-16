# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json

import httpx
import pytest
from feedback_fixtures import knowline_context, sample

types = pytest.importorskip("dynamo.planner.core.types")
from autoscaling_arena.adapters.jev import JevAutoscaler  # noqa: E402
from autoscaling_arena.adapters.knowline_policy import KnowLinePolicy  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]
MODEL = "PelaAI/KnowLine-31B-Gen1"


def policy(budget=24, mode="disagg", **limits):
    return KnowLinePolicy(
        mode,
        knowline_context(budget, **limits),
        {"prefill": (1, 16), "decode": (1, 16)},
        1,
    )


def test_heterogeneous_joint_choices_never_overspend_the_shared_budget():
    controller = policy()
    controller.feedback.observe(sample(p=2, d=1))
    snapshot = {"now_s": 0}
    options = controller.decision_inputs(snapshot, {"prefill": 2, "decode": 1}, [])[
        "fleet"
    ]["criteria"]
    assert "p=2,d=2" in options  # 8 + 16 = 24 GPUs.
    assert "p=3,d=1" in options  # 12 + 8 = 20 GPUs.
    assert "p=3,d=2" not in options  # Individually legal additions sum to 28.
    assert all(
        v["target_gpus"] == v["prefill_workers"] * 4 + v["decode_workers"] * 8 <= 24
        for v in options.values()
    )


def test_draining_gpus_do_not_fund_additions_or_same_tick_pool_swaps():
    controller = policy()
    controller.feedback.observe(sample(p=2, d=1, draining_d=1))
    snapshot = {"now_s": 0}
    options = controller.decision_inputs(snapshot, {"prefill": 2, "decode": 1}, [])[
        "fleet"
    ]["criteria"]
    assert set(options) == {"p=1,d=1", "p=2,d=1"}
    assert snapshot["allocation"]["resident_gpus"] == 24
    assert snapshot["allocation"]["free_gpus"] == 0


def test_aggregate_options_preserve_the_existing_total_worker_minimum():
    controller = policy(mode="agg", minimum=2)
    controller.feedback.observe(sample(p=0, d=2))
    options = controller.decision_inputs({"now_s": 0}, {"decode": 2}, [])["fleet"][
        "criteria"
    ]
    assert set(options) == {"p=0,d=2", "p=0,d=3"}


@pytest.mark.parametrize("fault", ["missing", "inconsistent", "overspent"])
def test_missing_or_inconsistent_resource_evidence_is_rejected(fault):
    controller = policy()
    if fault != "missing":
        controller.feedback.observe(
            sample(
                p=1 if fault == "inconsistent" else 2,
                d=1,
                draining_d=2 if fault == "overspent" else 0,
            )
        )
    with pytest.raises(ValueError):
        controller.decision_inputs({"now_s": 0}, {"prefill": 2, "decode": 1}, [])


def tick(at=0):
    return types.TickInput(
        now_s=at,
        worker_counts=types.WorkerCounts(
            ready_num_prefill=2,
            expected_num_prefill=2,
            ready_num_decode=1,
            expected_num_decode=1,
        ),
    )


def answer(payload, *, choice="p=2,d=2", confidence=1):
    return {
        "model": MODEL,
        "answers": {
            "fleet": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": {
                    k: int(k == choice)
                    for k in payload["questions"]["fleet"]["criteria"]
                },
            }
        },
    }


@pytest.mark.parametrize("confidence,expected", [(1, 2), (0.5, 1)])
def test_model_applies_a_joint_target_atomically_and_receives_measured_outcomes(
    tmp_path, confidence, expected
):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json=answer(payload, confidence=confidence))

    engine = JevAutoscaler(
        provider="self_hosted",
        base_url="http://localhost",
        model=MODEL,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="http://localhost"
        ),
        context=knowline_context(),
        min_confidence=0.8,
        decision_log=tmp_path / "decisions.jsonl",
    )

    async def run():
        try:
            engine.on_telemetry(sample(p=2, d=1))
            await engine.tick(engine.initial_tick(0), tick())
            # Keep observed fleet at the old allocation: the next prompt must
            # distinguish the previous request from its actual outcome.
            engine.on_telemetry(
                sample(5, p=2, d=1, terminal=10, completed=9, good=6, gpu_seconds=80)
            )
            result = await engine.tick(engine.initial_tick(5), tick(5))
            assert result.scale_to.num_prefill == 2
            assert result.scale_to.num_decode == expected
        finally:
            await engine.shutdown()

    asyncio.run(run())
    current = requests[1]["state"]["current"]
    assert current["outcomes"]["good_rate"] == 0.6
    assert current["outcomes"]["goodput_per_gpu"] == 6 / 80
    assert current["previous_action"]["requested_targets"]["decode"] == expected
    assert (
        requests[1]["state"]["policy_contract"] == "joint-fleet-measured-efficiency-v2"
    )
    assert "GPU-second" in requests[1]["questions"]["fleet"]["instructions"]


def test_new_joint_contract_rejects_an_old_saved_decision_before_http(tmp_path):
    prior = tmp_path / "prior.jsonl"
    prior.write_text(
        json.dumps({"status": "ok", "request_sha256": "old-independent-pools"}) + "\n"
    )

    def respond(request):
        pytest.fail("No model call is needed to reject an incompatible prefix")

    engine = JevAutoscaler(
        provider="self_hosted",
        base_url="http://localhost",
        model=MODEL,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="http://localhost"
        ),
        context=knowline_context(),
        prior_calls=1,
        replay_decisions=str(prior),
        replay_decisions_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
    )

    async def run():
        try:
            engine.on_telemetry(sample(p=2, d=1))
            with pytest.raises(ValueError, match="differs"):
                await engine.tick(engine.initial_tick(0), tick())
        finally:
            await engine.shutdown()

    asyncio.run(run())
