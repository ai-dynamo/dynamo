# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import batch_planner_control_loop as control_loop
import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]


@dataclasses.dataclass
class FakeObservation:
    job_demands: list[dict[str, Any]]
    pool_traffic: list[dict[str, Any]]
    dispatcher_feedback: list[dict[str, Any]]


@dataclasses.dataclass
class FakeDrainLimit:
    pool_id: str
    max_admission_rps: float
    valid_until_s: float
    decision_id: str


@dataclasses.dataclass
class FakeDiagnostics:
    safety_paused: bool = False
    required_batch_rps: float = 3.0
    minimum_deadline_slack_s: float = 10.0


@dataclasses.dataclass
class FakePlan:
    replica_floor: int
    drain_limit: FakeDrainLimit
    diagnostics: FakeDiagnostics


@dataclasses.dataclass
class FakeTick:
    now_s: float
    ready_replicas: int
    batch: FakeObservation


class SequenceCollector:
    def __init__(self, values: list[FakeObservation | BaseException]) -> None:
        self._values = list(values)
        self.calls = 0

    async def collect(self) -> FakeObservation:
        value = self._values[self.calls]
        self.calls += 1
        if isinstance(value, BaseException):
            raise value
        return value


class FakeActuator:
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.decisions: list[FakeDrainLimit] = []
        self.fail_on_call = fail_on_call

    async def apply_drain_limit(self, decision: FakeDrainLimit) -> None:
        self.decisions.append(decision)
        if self.fail_on_call == len(self.decisions):
            raise RuntimeError("Redis write failed")


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def __call__(self) -> float:
        if not self._values:
            raise AssertionError("clock called more times than expected")
        return self._values.pop(0)


def observation() -> FakeObservation:
    return FakeObservation(
        job_demands=[
            {
                "job_id": "batch-1",
                "remaining_requests": 100,
                "deadline_at_s": 200.0,
            }
        ],
        pool_traffic=[{"pool_id": "pool-a", "online_offered_rps": 7.0}],
        dispatcher_feedback=[{"pool_id": "pool-a", "queued_requests": 100}],
    )


def settings(
    *, apply: bool = False, iterations: int = 1
) -> control_loop.ControlLoopSettings:
    return control_loop.ControlLoopSettings(
        run_id="20260828T130000Z-planner-loop-abcdef",
        pool_id="pool-a",
        work_class="gsm8k-128",
        safe_rps_per_ready_replica=10.0,
        ready_replicas=2,
        online_offered_rps=7.0,
        iterations=iterations,
        interval_seconds=5.0,
        drain_lease_duration_s=20.0,
        apply_drain_limit=apply,
    )


def tick_builder(
    now_s: float,
    ready_replicas: int,
    batch: FakeObservation,
) -> FakeTick:
    return FakeTick(now_s, ready_replicas, batch)


def policy(
    tick: FakeTick,
    _config: object,
    *,
    decision_id: str,
) -> FakePlan:
    return FakePlan(
        replica_floor=tick.ready_replicas,
        drain_limit=FakeDrainLimit(
            pool_id="pool-a",
            max_admission_rps=13.0,
            valid_until_s=tick.now_s + 20.0,
            decision_id=decision_id,
        ),
        diagnostics=FakeDiagnostics(),
    )


def pause_builder(
    pool_id: str,
    rate: float,
    valid_until_s: float,
    decision_id: str,
) -> FakeDrainLimit:
    return FakeDrainLimit(pool_id, rate, valid_until_s, decision_id)


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_dry_run_records_two_decisions_without_actuation(tmp_path: Path) -> None:
    output = tmp_path / "decisions.jsonl"
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    with control_loop.JsonlDecisionRecorder(output) as recorder:
        asyncio.run(
            control_loop.run_control_loop(
                settings=settings(iterations=2),
                collector=SequenceCollector([observation(), observation()]),
                policy_config=object(),
                policy=policy,
                tick_input_builder=tick_builder,
                pause_decision_builder=pause_builder,
                recorder=recorder,
                clock=SequenceClock([100.0, 105.0]),
                sleeper=fake_sleep,
            )
        )

    records = read_records(output)
    assert len(records) == 2
    assert [record["status"] for record in records] == ["planned", "planned"]
    assert [record["mode"] for record in records] == ["dry_run", "dry_run"]
    assert all(not record["actuation"]["drain_limit_applied"] for record in records)
    assert all(not record["actuation"]["replica_scaling_applied"] for record in records)
    assert records[0]["observation"]["job_demands"][0]["job_id"] == "batch-1"
    assert records[0]["decision"]["drain_limit"]["max_admission_rps"] == 13.0
    assert records[0]["diagnostics"]["required_batch_rps"] == 3.0
    assert sleeps == [5.0]


def test_apply_mode_renews_unique_leased_decisions(tmp_path: Path) -> None:
    output = tmp_path / "decisions.jsonl"
    actuator = FakeActuator()

    async def fake_sleep(_delay: float) -> None:
        return None

    with control_loop.JsonlDecisionRecorder(output) as recorder:
        asyncio.run(
            control_loop.run_control_loop(
                settings=settings(apply=True, iterations=2),
                collector=SequenceCollector([observation(), observation()]),
                policy_config=object(),
                policy=policy,
                tick_input_builder=tick_builder,
                pause_decision_builder=pause_builder,
                recorder=recorder,
                actuator=actuator,
                clock=SequenceClock([100.0, 105.0]),
                sleeper=fake_sleep,
            )
        )

    assert len(actuator.decisions) == 2
    assert [decision.valid_until_s for decision in actuator.decisions] == [120.0, 125.0]
    assert len({decision.decision_id for decision in actuator.decisions}) == 2
    records = read_records(output)
    assert [record["status"] for record in records] == ["applied", "applied"]
    assert all(record["actuation"]["drain_limit_applied"] for record in records)


def test_apply_collection_failure_writes_best_effort_pause(tmp_path: Path) -> None:
    output = tmp_path / "decisions.jsonl"
    actuator = FakeActuator()

    with (
        control_loop.JsonlDecisionRecorder(output) as recorder,
        pytest.raises(control_loop.ControlLoopFailure, match="collection failed"),
    ):
        asyncio.run(
            control_loop.run_control_loop(
                settings=settings(apply=True),
                collector=SequenceCollector([RuntimeError("gateway unavailable")]),
                policy_config=object(),
                policy=policy,
                tick_input_builder=tick_builder,
                pause_decision_builder=pause_builder,
                recorder=recorder,
                actuator=actuator,
                clock=SequenceClock([100.0, 101.0]),
            )
        )

    assert len(actuator.decisions) == 1
    pause = actuator.decisions[0]
    assert pause.max_admission_rps == 0.0
    assert pause.valid_until_s == 121.0
    assert pause.decision_id.endswith("-fail-closed")
    record = read_records(output)[0]
    assert record["error"]["phase"] == "collection"
    assert record["fail_closed_pause"]["attempted"] is True
    assert record["fail_closed_pause"]["applied"] is True


def test_failed_best_effort_pause_preserves_original_failure(tmp_path: Path) -> None:
    output = tmp_path / "decisions.jsonl"
    actuator = FakeActuator(fail_on_call=1)

    with (
        control_loop.JsonlDecisionRecorder(output) as recorder,
        pytest.raises(control_loop.ControlLoopFailure, match="collection failed"),
    ):
        asyncio.run(
            control_loop.run_control_loop(
                settings=settings(apply=True),
                collector=SequenceCollector([RuntimeError("gateway unavailable")]),
                policy_config=object(),
                policy=policy,
                tick_input_builder=tick_builder,
                pause_decision_builder=pause_builder,
                recorder=recorder,
                actuator=actuator,
                clock=SequenceClock([100.0, 101.0]),
            )
        )

    record = read_records(output)[0]
    assert record["error"]["phase"] == "collection"
    assert record["fail_closed_pause"]["attempted"] is True
    assert record["fail_closed_pause"]["applied"] is False
    assert record["fail_closed_pause"]["error"]["type"] == "RuntimeError"


def test_dry_run_collection_failure_never_calls_actuator(tmp_path: Path) -> None:
    output = tmp_path / "decisions.jsonl"
    actuator = FakeActuator()

    with (
        control_loop.JsonlDecisionRecorder(output) as recorder,
        pytest.raises(control_loop.ControlLoopFailure),
    ):
        asyncio.run(
            control_loop.run_control_loop(
                settings=settings(),
                collector=SequenceCollector([RuntimeError("gateway unavailable")]),
                policy_config=object(),
                policy=policy,
                tick_input_builder=tick_builder,
                pause_decision_builder=pause_builder,
                recorder=recorder,
                actuator=actuator,
                clock=SequenceClock([100.0]),
            )
        )

    assert actuator.decisions == []
    assert read_records(output)[0]["fail_closed_pause"] is None


def test_apply_policy_failure_writes_pause_with_observation(tmp_path: Path) -> None:
    output = tmp_path / "decisions.jsonl"
    actuator = FakeActuator()

    def failing_policy(
        _tick: FakeTick,
        _config: object,
        *,
        decision_id: str,
    ) -> FakePlan:
        del decision_id
        raise ValueError("policy input rejected")

    with (
        control_loop.JsonlDecisionRecorder(output) as recorder,
        pytest.raises(control_loop.ControlLoopFailure, match="policy failed"),
    ):
        asyncio.run(
            control_loop.run_control_loop(
                settings=settings(apply=True),
                collector=SequenceCollector([observation()]),
                policy_config=object(),
                policy=failing_policy,
                tick_input_builder=tick_builder,
                pause_decision_builder=pause_builder,
                recorder=recorder,
                actuator=actuator,
                clock=SequenceClock([100.0, 101.0, 102.0]),
            )
        )

    assert len(actuator.decisions) == 1
    assert actuator.decisions[0].max_admission_rps == 0.0
    record = read_records(output)[0]
    assert record["observation"]["job_demands"][0]["job_id"] == "batch-1"
    assert record["decision"] is None
    assert record["error"]["phase"] == "policy"


def test_actuation_failure_does_not_issue_a_second_mutation(tmp_path: Path) -> None:
    output = tmp_path / "decisions.jsonl"
    actuator = FakeActuator(fail_on_call=1)

    with (
        control_loop.JsonlDecisionRecorder(output) as recorder,
        pytest.raises(control_loop.ControlLoopFailure, match="actuation failed"),
    ):
        asyncio.run(
            control_loop.run_control_loop(
                settings=settings(apply=True),
                collector=SequenceCollector([observation()]),
                policy_config=object(),
                policy=policy,
                tick_input_builder=tick_builder,
                pause_decision_builder=pause_builder,
                recorder=recorder,
                actuator=actuator,
                clock=SequenceClock([100.0, 101.0]),
            )
        )

    assert len(actuator.decisions) == 1
    record = read_records(output)[0]
    assert record["error"]["phase"] == "actuation"
    assert record["fail_closed_pause"] is None
    assert record["decision"]["drain_limit"]["max_admission_rps"] == 13.0


def test_invalid_policy_decision_fails_closed_before_applying_it(
    tmp_path: Path,
) -> None:
    output = tmp_path / "decisions.jsonl"
    actuator = FakeActuator()

    def mismatched_policy(
        tick: FakeTick,
        _config: object,
        *,
        decision_id: str,
    ) -> FakePlan:
        return FakePlan(
            replica_floor=1,
            drain_limit=FakeDrainLimit(
                pool_id="another-pool",
                max_admission_rps=1.0,
                valid_until_s=tick.now_s + 20.0,
                decision_id=decision_id,
            ),
            diagnostics=FakeDiagnostics(),
        )

    with (
        control_loop.JsonlDecisionRecorder(output) as recorder,
        pytest.raises(control_loop.ControlLoopFailure, match="wrong pool"),
    ):
        asyncio.run(
            control_loop.run_control_loop(
                settings=settings(apply=True),
                collector=SequenceCollector([observation()]),
                policy_config=object(),
                policy=mismatched_policy,
                tick_input_builder=tick_builder,
                pause_decision_builder=pause_builder,
                recorder=recorder,
                actuator=actuator,
                clock=SequenceClock([100.0, 101.0, 102.0]),
            )
        )

    assert len(actuator.decisions) == 1
    assert actuator.decisions[0].pool_id == "pool-a"
    assert actuator.decisions[0].max_admission_rps == 0.0


def test_records_are_strict_json_and_redact_credential_shapes(tmp_path: Path) -> None:
    output = tmp_path / "decisions.jsonl"
    secret = "top" + "secret"
    hf_value = "hf_" + ("x" * 24)
    with control_loop.JsonlDecisionRecorder(output) as recorder:
        recorder.append(
            {
                "error": (
                    f"redis://user:{secret}@host/0 Bearer {secret} HF_TOKEN={hf_value}"
                ),
                "positive": math.inf,
                "negative": -math.inf,
            }
        )

    raw = output.read_text()
    assert secret not in raw
    assert hf_value not in raw
    record = json.loads(raw)
    assert record["positive"] == "Infinity"
    assert record["negative"] == "-Infinity"


def required_cli() -> list[str]:
    return [
        "--pool",
        "pool-a",
        "--work-class",
        "gsm8k-128",
        "--safe-rps-per-ready-replica",
        "10",
        "--ready-replicas",
        "2",
        "--online-offered-rps",
        "7",
    ]


def test_cli_defaults_to_dry_run_and_bounds_max_replicas() -> None:
    args = control_loop.parse_args(required_cli())

    control_loop.validate_args(args)

    assert args.apply_drain_limit is False
    assert args.iterations == 1
    assert args.max_replicas == 2
    assert args.redis_url is None
    assert args.tenant == "planner-poc-baseline"


def test_cli_accepts_bounded_interval_flag() -> None:
    args = control_loop.parse_args([*required_cli(), "--interval", "4"])

    control_loop.validate_args(args)

    assert args.interval_seconds == 4.0


def test_apply_mode_requires_explicit_redis_target() -> None:
    args = control_loop.parse_args([*required_cli(), "--apply-drain-limit"])

    with pytest.raises(
        control_loop.ControlLoopConfigurationError,
        match="requires --redis-url and --redis-control-key",
    ):
        control_loop.validate_args(args)


def test_redis_options_are_rejected_without_apply_flag() -> None:
    args = control_loop.parse_args(
        [*required_cli(), "--redis-url", "redis://127.0.0.1:6379/0"]
    )

    with pytest.raises(
        control_loop.ControlLoopConfigurationError,
        match="explicit --apply-drain-limit",
    ):
        control_loop.validate_args(args)


def test_credential_bearing_http_url_is_rejected() -> None:
    args = control_loop.parse_args(
        [
            *required_cli(),
            "--batch-base-url",
            "https://user:password@example.test",
        ]
    )

    with pytest.raises(
        control_loop.ControlLoopConfigurationError,
        match="must not contain credentials",
    ):
        control_loop.validate_args(args)


def test_output_is_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "decisions.jsonl"
    with control_loop.JsonlDecisionRecorder(output):
        pass

    with pytest.raises(FileExistsError):
        control_loop.JsonlDecisionRecorder(output)


def test_main_records_startup_error_and_returns_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "decisions.jsonl"
    secret = "do-not-" + "record"

    async def failing_live_runner(
        _args: Any,
        _settings: control_loop.ControlLoopSettings,
        _recorder: control_loop.JsonlDecisionRecorder,
    ) -> None:
        raise RuntimeError(f"redis://user:{secret}@host/0")

    exit_code = control_loop.main(
        [*required_cli(), "--output", str(output)],
        live_runner=failing_live_runner,
    )

    assert exit_code == 1
    record = read_records(output)[0]
    assert record["status"] == "error"
    assert record["error"]["phase"] == "startup"
    assert secret not in output.read_text()
    assert secret not in capsys.readouterr().err


def test_settings_reject_lease_that_cannot_outlive_interval() -> None:
    unsafe = dataclasses.replace(
        settings(),
        interval_seconds=20.0,
        drain_lease_duration_s=20.0,
    )

    with pytest.raises(
        control_loop.ControlLoopConfigurationError,
        match="greater than the loop interval",
    ):
        control_loop.validate_settings(unsafe)
