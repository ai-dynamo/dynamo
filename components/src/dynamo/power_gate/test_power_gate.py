# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from . import gate
from .gate import GateConfig, GateContext, GateTimeout, run_gate

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

NOW = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)


class ExecInvoked(RuntimeError):
    pass


def _context(gpu_count: int = 1) -> GateContext:
    return GateContext(
        dgd_uid="dgd-uid",
        component="decode",
        expected_gpu_count=gpu_count,
        in_gate_bound_watts_per_gpu=350,
    )


def _gpu(uuid: str = "GPU-a", observed_at: datetime = NOW) -> dict:
    return {
        "uuid": uuid,
        "requestedWatts": 350,
        "targetWatts": 350,
        "constraintMinWatts": 100,
        "constraintMaxWatts": 700,
        "policyOutcome": "annotated",
        "writeOutcome": "succeeded",
        "readbackOutcome": "succeeded",
        "enforcedCapWatts": 350,
        "actuator": "nvml",
        "observedAt": observed_at.isoformat().replace("+00:00", "Z"),
    }


def _report(*gpus: dict) -> dict:
    selected = list(gpus) if gpus else [_gpu()]
    uuids = sorted(gpu["uuid"] for gpu in selected)
    return {
        "version": 1,
        "dgdUID": "dgd-uid",
        "component": "decode",
        "podUID": "pod-uid",
        "node": "node-a",
        "allocationID": f"pod-uid/main/{','.join(uuids)}",
        "gpus": selected,
    }


def _config(tmp_path: Path, report: object | None, gpu_count: int = 1) -> GateConfig:
    pod_uid_file = tmp_path / "pod-uid"
    pod_uid_file.write_text("pod-uid\n", encoding="utf-8")
    report_file = tmp_path / "report"
    if report is not None:
        report_file.write_text(json.dumps(report), encoding="utf-8")
    return GateConfig(
        context=_context(gpu_count),
        pod_uid_file=pod_uid_file,
        report_file=report_file,
        timeout_seconds=0,
    )


def _exec_once(executions: list[tuple[str, tuple[str, ...]]]):
    def invoke(executable: str, arguments) -> None:
        executions.append((executable, tuple(arguments)))
        raise ExecInvoked

    return invoke


def test_valid_fresh_exact_report_execs_original_command_once(tmp_path: Path):
    executions: list[tuple[str, tuple[str, ...]]] = []
    config = _config(tmp_path, _report())

    with pytest.raises(ExecInvoked):
        run_gate(
            ("python3", "-m", "dynamo.vllm"),
            config,
            exec_process=_exec_once(executions),
            now=lambda: NOW,
        )

    assert executions == [("python3", ("python3", "-m", "dynamo.vllm"))]


def test_python_distribution_exposes_power_gate_console_script():
    repository_root = Path(__file__).resolve().parents[4]
    with (repository_root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["scripts"]["dynamo-power-gate"] == "dynamo.power_gate:main"


def test_multi_gpu_report_requires_exact_allocation_identity(tmp_path: Path):
    executions: list[tuple[str, tuple[str, ...]]] = []
    report = _report(_gpu("GPU-b"), _gpu("GPU-a"))
    config = _config(tmp_path, report, gpu_count=2)

    with pytest.raises(ExecInvoked):
        run_gate(
            ("backend", "--arg"),
            config,
            exec_process=_exec_once(executions),
            now=lambda: NOW,
        )

    assert executions == [("backend", ("backend", "--arg"))]


@pytest.mark.parametrize(
    ("case", "reason"),
    (
        ("missing", "report_missing"),
        ("malformed", "report_malformed"),
        ("version", "report_version_mismatch"),
        ("dgd", "dgd_identity_mismatch"),
        ("component", "component_identity_mismatch"),
        ("pod", "pod_identity_mismatch"),
        ("allocation", "allocation_identity_mismatch"),
        ("count", "gpu_count_mismatch"),
        ("safe-default", "policy_outcome_not_annotated"),
        ("write", "write_not_succeeded"),
        ("readback", "readback_not_succeeded"),
        ("above-bound", "enforced_cap_above_bound"),
        ("stale", "report_not_fresh"),
        ("future", "report_not_fresh"),
    ),
)
def test_rejected_reports_never_start_backend(tmp_path: Path, case: str, reason: str):
    report = copy.deepcopy(_report())
    if case == "missing":
        candidate = None
    elif case == "malformed":
        candidate = ["not", "an", "object"]
    else:
        candidate = report
        if case == "version":
            report["version"] = 2
        elif case == "dgd":
            report["dgdUID"] = "other-dgd"
        elif case == "component":
            report["component"] = "prefill"
        elif case == "pod":
            report["podUID"] = "other-pod"
        elif case == "allocation":
            report["allocationID"] = "pod-uid/main/GPU-other"
        elif case == "count":
            report["gpus"] = []
        elif case == "safe-default":
            report["gpus"][0]["policyOutcome"] = "safe_default_conflict"
        elif case == "write":
            report["gpus"][0]["writeOutcome"] = "failed"
            report["gpus"][0]["readbackOutcome"] = "not_attempted"
            report["gpus"][0]["enforcedCapWatts"] = None
        elif case == "readback":
            report["gpus"][0]["readbackOutcome"] = "failed"
            report["gpus"][0]["enforcedCapWatts"] = None
        elif case == "above-bound":
            report["gpus"][0]["enforcedCapWatts"] = 351
        elif case == "stale":
            report["gpus"][0]["observedAt"] = (NOW - timedelta(seconds=61)).isoformat()
        elif case == "future":
            report["gpus"][0]["observedAt"] = (NOW + timedelta(seconds=1)).isoformat()

    executions: list[tuple[str, tuple[str, ...]]] = []
    config = _config(tmp_path, candidate)
    with pytest.raises(GateTimeout) as error:
        run_gate(
            ("backend",),
            config,
            exec_process=_exec_once(executions),
            now=lambda: NOW,
        )

    assert error.value.last_reason == reason
    assert executions == []


def test_freshness_boundary_is_accepted(tmp_path: Path):
    executions: list[tuple[str, tuple[str, ...]]] = []
    config = _config(tmp_path, _report(_gpu(observed_at=NOW - timedelta(minutes=1))))

    with pytest.raises(ExecInvoked):
        run_gate(
            ("backend",),
            config,
            exec_process=_exec_once(executions),
            now=lambda: NOW,
        )

    assert len(executions) == 1


def test_timeout_does_not_start_backend(tmp_path: Path):
    executions: list[tuple[str, tuple[str, ...]]] = []
    config = _config(tmp_path, None)

    with pytest.raises(GateTimeout, match="report_missing"):
        run_gate(
            ("backend",),
            config,
            exec_process=_exec_once(executions),
            now=lambda: NOW,
        )

    assert executions == []


def test_oversized_json_integer_is_stable_malformed_rejection(tmp_path: Path):
    config = _config(tmp_path, _report())
    encoded = json.dumps(_report()).replace('"version": 1', '"version": ' + "9" * 5000)
    config.report_file.write_text(encoded, encoding="utf-8")
    executions: list[tuple[str, tuple[str, ...]]] = []

    with pytest.raises(GateTimeout, match="report_malformed"):
        run_gate(
            ("backend",),
            config,
            exec_process=_exec_once(executions),
            now=lambda: NOW,
        )

    assert executions == []


def test_missing_original_executable_returns_127_after_gate(monkeypatch, capsys):
    monkeypatch.setenv(gate.DGD_UID_ENV, "dgd-uid")
    monkeypatch.setenv(gate.COMPONENT_ENV, "decode")
    monkeypatch.setenv(gate.EXPECTED_GPU_COUNT_ENV, "1")
    monkeypatch.setenv(gate.IN_GATE_BOUND_WATTS_ENV, "350")
    commands = []

    def missing_executable(command, _config):
        commands.append(tuple(command))
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(gate, "run_gate", missing_executable)

    assert gate.main(["--", "/missing/backend", "--flag"]) == 127
    assert commands == [("/missing/backend", "--flag")]
    assert "exec_failed" in capsys.readouterr().err


def test_timeout_writes_bounded_stable_termination_reason(
    tmp_path: Path, monkeypatch, capsys
):
    termination_message = tmp_path / "termination-log"
    monkeypatch.setattr(gate, "TERMINATION_MESSAGE_FILE", termination_message)
    monkeypatch.setenv(gate.DGD_UID_ENV, "dgd-uid")
    monkeypatch.setenv(gate.COMPONENT_ENV, "decode")
    monkeypatch.setenv(gate.EXPECTED_GPU_COUNT_ENV, "1")
    monkeypatch.setenv(gate.IN_GATE_BOUND_WATTS_ENV, "350")

    def rejected_report(_command, _config):
        raise GateTimeout("pod_identity_mismatch")

    monkeypatch.setattr(gate, "run_gate", rejected_report)

    assert gate.main(["--", "backend"]) == 1
    expected = "dynamo-power-gate: enforcement_timeout: pod_identity_mismatch\n"
    assert termination_message.read_text(encoding="utf-8") == expected
    assert capsys.readouterr().err == expected


@pytest.mark.skipif(os.name == "nt", reason="POSIX process semantics")
@pytest.mark.parametrize(
    ("backend_command", "expected_returncode"),
    (
        (("/bin/sh", "-c", "exit 23"), 23),
        (("/bin/sh", "-c", "kill -TERM $$"), -signal.SIGTERM),
    ),
)
def test_exec_preserves_backend_exit_and_signal(
    tmp_path: Path,
    backend_command: tuple[str, ...],
    expected_returncode: int,
):
    report = _report(_gpu(observed_at=datetime.now(timezone.utc)))
    config = _config(tmp_path, report)
    child_script = """
from datetime import timedelta
from pathlib import Path
import sys

from dynamo.power_gate.gate import GateConfig, GateContext, run_gate

config = GateConfig(
    context=GateContext(
        dgd_uid="dgd-uid",
        component="decode",
        expected_gpu_count=1,
        in_gate_bound_watts_per_gpu=350,
    ),
    pod_uid_file=Path(sys.argv[1]),
    report_file=Path(sys.argv[2]),
    freshness_limit=timedelta(days=1),
    timeout_seconds=0,
)
run_gate(tuple(sys.argv[3:]), config)
"""
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[2]
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), inherited_pythonpath) if part
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            child_script,
            str(config.pod_uid_file),
            str(config.report_file),
            *backend_command,
        ],
        env=environment,
        check=False,
    )

    assert completed.returncode == expected_returncode
