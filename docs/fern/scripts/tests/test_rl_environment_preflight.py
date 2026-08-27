# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the RL GPU-host environment preflight artifact."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
import rl_environment_preflight

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

TEMPLATE = Path(__file__).resolve().parents[1] / "rl_validation_record.template.json"
REPOSITORIES = {
    "dynamo": Path("/repos/dynamo"),
    "recipe": Path("/repos/verl-recipe"),
    "core": Path("/repos/verl"),
}


def _record() -> dict:
    record = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    record["record_id"] = "verl-dynamo-preflight-test"
    record["framework"].update(
        {
            "recipe_commit": "a" * 40,
            "core_commit": "b" * 40,
        }
    )
    record["environment"].update(
        {
            "dynamo_commit": "c" * 40,
            "container_image": "registry.example/dynamo:v1",
            "container_image_digest": "sha256:" + "d" * 64,
            "cuda_version": "13.0",
            "driver_version": "590.00",
        }
    )
    record["environment"]["backend"]["version"] = "0.27.1"
    record["hardware"].update(
        {
            "gpu_model": "H100 SXM",
            "gpus_per_node": 2,
            "interconnect": "NVLink",
            "network": "InfiniBand",
        }
    )
    return record


class FakeMachine:
    def __init__(self) -> None:
        self.commits = {
            "/repos/dynamo": "c" * 40,
            "/repos/verl-recipe": "a" * 40,
            "/repos/verl": "b" * 40,
        }
        self.dirty: set[str] = set()
        self.missing_binaries: set[str] = set()
        self.gpu_rows = (
            "0, NVIDIA H100 80GB HBM3, 590.00, 81559\n"
            "1, NVIDIA H100 80GB HBM3, 590.00, 81559\n"
        )
        self.topology_returncode = 0
        self.topology = "GPU0 GPU1 CPU Affinity\nGPU0 X NV18 0-31\nGPU1 NV18 X 0-31\n"
        self.software = {
            "backend_version": "0.27.1",
            "backend_error": None,
            "torch_version": "2.10.0",
            "torch_cuda_version": "13.0",
            "torch_error": None,
        }

    def run(self, args) -> subprocess.CompletedProcess[str]:
        argv = list(args)
        if argv[0] == "git" and argv[3:5] == ["rev-parse", "HEAD"]:
            path = argv[2]
            commit = self.commits.get(path)
            return subprocess.CompletedProcess(
                argv,
                0 if commit else 1,
                f"{commit}\n" if commit else "",
                "" if commit else "not a repository",
            )
        if argv[0] == "git" and argv[3] == "status":
            path = argv[2]
            output = " M modified.py\n" if path in self.dirty else ""
            return subprocess.CompletedProcess(argv, 0, output, "")
        if argv[:2] == [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
        ]:
            return subprocess.CompletedProcess(argv, 0, self.gpu_rows, "")
        if argv[:3] == ["nvidia-smi", "topo", "-m"]:
            return subprocess.CompletedProcess(
                argv,
                self.topology_returncode,
                self.topology if self.topology_returncode == 0 else "",
                "topology unavailable" if self.topology_returncode else "",
            )
        raise AssertionError(f"unexpected command: {argv}")

    def lookup(self, name: str) -> str | None:
        return None if name in self.missing_binaries else f"/usr/bin/{name}"

    def probe_software(self, _backend: str) -> dict[str, str | None]:
        return dict(self.software)


def _build(
    machine: FakeMachine,
    *,
    record: dict | None = None,
    host_system: str = "Linux",
    image_digest: str | None = "sha256:" + "d" * 64,
    required_binaries: tuple[str, ...] = ("etcd", "nats-server"),
):
    return rl_environment_preflight.build_report(
        record or _record(),
        record_path=Path("/artifacts/framework-validation.json"),
        record_sha256="e" * 64,
        repositories=REPOSITORIES,
        observed_image_digest=image_digest,
        required_binaries=required_binaries,
        runner=machine.run,
        binary_lookup=machine.lookup,
        software_probe=machine.probe_software,
        host_system=host_system,
        generated_at=datetime(2026, 8, 27, 18, 0, tzinfo=UTC),
    )


def test_matching_host_writes_a_passing_privacy_bounded_report() -> None:
    report, findings = _build(FakeMachine())
    assert findings == []
    assert report["schema"] == "dynamo.rl.environment-preflight.v1"
    assert report["strict_status"] == "passed"
    assert report["failed_check_count"] == 0
    assert report["not_checked_count"] == 2
    assert report["host"]["hostname_captured"] is False
    assert report["host"]["environment_variables_captured"] is False
    assert len(report["hardware"]["gpus"]) == 2
    assert report["container"]["observed_digest_provenance"] == "operator_argument"


@pytest.mark.parametrize(
    "case,expected_check",
    [
        ("commit", "repository.dynamo.commit"),
        ("dirty", "repository.recipe.clean"),
        ("binary", "binary.nats-server"),
        ("gpu_count", "hardware.gpu_count"),
        ("gpu_model", "hardware.gpu_model"),
        ("driver", "environment.driver_version"),
        ("backend", "environment.backend_version"),
        ("cuda", "environment.cuda_version"),
        ("image", "environment.container_image_digest"),
        ("operating_system", "host.operating_system"),
        ("topology", "hardware.topology_capture"),
    ],
)
def test_preflight_fails_each_machine_check(case: str, expected_check: str) -> None:
    machine = FakeMachine()
    host_system = "Linux"
    image_digest = "sha256:" + "d" * 64
    if case == "commit":
        machine.commits["/repos/dynamo"] = "f" * 40
    elif case == "dirty":
        machine.dirty.add("/repos/verl-recipe")
    elif case == "binary":
        machine.missing_binaries.add("nats-server")
    elif case == "gpu_count":
        machine.gpu_rows = "0, NVIDIA H100 80GB HBM3, 590.00, 81559\n"
    elif case == "gpu_model":
        machine.gpu_rows = (
            "0, NVIDIA A100-SXM4-80GB, 590.00, 81559\n"
            "1, NVIDIA A100-SXM4-80GB, 590.00, 81559\n"
        )
    elif case == "driver":
        machine.gpu_rows = (
            "0, NVIDIA H100 80GB HBM3, 591.00, 81559\n"
            "1, NVIDIA H100 80GB HBM3, 591.00, 81559\n"
        )
    elif case == "backend":
        machine.software["backend_version"] = "0.26.0"
    elif case == "cuda":
        machine.software["torch_cuda_version"] = "12.9"
    elif case == "image":
        image_digest = None
    elif case == "operating_system":
        host_system = "Darwin"
    elif case == "topology":
        machine.topology_returncode = 1
    report, findings = _build(
        machine, host_system=host_system, image_digest=image_digest
    )
    assert report["strict_status"] == "failed"
    assert any(finding.startswith(expected_check + ":") for finding in findings)


def test_malformed_gpu_query_is_preserved_as_a_failed_artifact() -> None:
    machine = FakeMachine()
    machine.gpu_rows = "not,a,valid,row,shape\n"
    report, findings = _build(machine)
    assert report["hardware"]["query_succeeded"] is False
    assert "must return index" in report["hardware"]["query_error"]
    assert any(finding.startswith("hardware.gpu_query:") for finding in findings)


def test_invalid_validation_record_is_rejected_before_host_probing() -> None:
    record = _record()
    del record["hardware"]["gpu_model"]
    with pytest.raises(
        rl_environment_preflight.PreflightError,
        match="validation record structure is invalid",
    ):
        _build(FakeMachine(), record=record)


def test_cli_writes_failed_report_and_strict_mode_returns_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record_path = tmp_path / "record.json"
    output = tmp_path / "preflight.json"
    record_path.write_text(json.dumps(_record()), encoding="utf-8")
    report = {
        "strict_status": "failed",
        "failed_check_count": 1,
        "not_checked_count": 2,
    }
    monkeypatch.setattr(
        rl_environment_preflight,
        "build_report",
        lambda *args, **kwargs: (report, ["hardware.gpu_count: mismatch"]),
    )
    args = [
        str(record_path),
        "--dynamo-repo",
        "/repos/dynamo",
        "--recipe-repo",
        "/repos/verl-recipe",
        "--core-repo",
        "/repos/verl",
        "--output-json",
        str(output),
        "--strict",
    ]
    assert rl_environment_preflight.main(args) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == report
    captured = capsys.readouterr()
    assert "hardware.gpu_count: mismatch" in captured.err
    assert "Wrote failed preflight report" in captured.err


def test_cli_non_strict_mode_preserves_findings_without_claiming_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_path = tmp_path / "record.json"
    output = tmp_path / "preflight.json"
    record_path.write_text(json.dumps(_record()), encoding="utf-8")
    report = {
        "strict_status": "failed",
        "failed_check_count": 1,
        "not_checked_count": 2,
    }
    monkeypatch.setattr(
        rl_environment_preflight,
        "build_report",
        lambda *args, **kwargs: (report, ["binary.etcd: unavailable"]),
    )
    args = [
        str(record_path),
        "--dynamo-repo",
        "/repos/dynamo",
        "--recipe-repo",
        "/repos/verl-recipe",
        "--core-repo",
        "/repos/verl",
        "--output-json",
        str(output),
    ]
    assert rl_environment_preflight.main(args) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["strict_status"] == "failed"


def test_cli_rejects_a_symlinked_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_path = tmp_path / "record.json"
    target = tmp_path / "target.json"
    output = tmp_path / "preflight.json"
    record_path.write_text(json.dumps(_record()), encoding="utf-8")
    target.write_text("{}", encoding="utf-8")
    output.symlink_to(target)
    report = {
        "strict_status": "passed",
        "failed_check_count": 0,
        "not_checked_count": 2,
    }
    monkeypatch.setattr(
        rl_environment_preflight,
        "build_report",
        lambda *args, **kwargs: (report, []),
    )
    args = [
        str(record_path),
        "--dynamo-repo",
        "/repos/dynamo",
        "--recipe-repo",
        "/repos/verl-recipe",
        "--core-repo",
        "/repos/verl",
        "--output-json",
        str(output),
    ]
    assert rl_environment_preflight.main(args) == 2
    assert target.read_text(encoding="utf-8") == "{}"
