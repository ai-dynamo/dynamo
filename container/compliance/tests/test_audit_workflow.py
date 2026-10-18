# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the audit's actual inventory selector and required status gate."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from compliance.audit_image_sbom import main as audit

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]
_ROOT = Path(__file__).resolve().parents[3]


def _workflow(name):
    return yaml.load(
        (_ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader
    )


@pytest.mark.parametrize("platform", ["amd64", "arm64"])
@pytest.mark.parametrize("case", ["current", "historical", "wrong-arch", "duplicate"])
@pytest.mark.timeout(10)
def test_inventory_selection_rejects_substitutes(tmp_path, platform, case):
    workflow = _workflow("shared-compliance-audit.yml")
    script = next(
        step["run"]
        for step in workflow["jobs"]["audit"]["steps"]
        if step.get("id") == "inventory"
    )
    script = script.replace("/tmp/compliance", str(tmp_path))
    script = script.replace("${{ matrix.platform }}", platform)
    paths = [f"linux_{platform}/baseline/osrb-old-12345678.csv"]
    paths.append(f"linux_{platform}/osrb-runtime-12345678.diff.csv")
    if case in {"current", "duplicate"}:
        paths.append(f"linux_{platform}/osrb-runtime-12345678.csv")
    if case == "duplicate":
        paths.append(f"linux_{platform}/osrb-other-12345678.csv")
    if case == "wrong-arch":
        other = "arm64" if platform == "amd64" else "amd64"
        paths.append(f"linux_{other}/osrb-runtime-12345678.csv")
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ecosystem,name,version,spdx\n")
    output = tmp_path / "outputs"
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "GITHUB_SHA": "12345678abcdef",
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert (result.returncode == 0) == (case == "current"), result.stderr
    if case == "current":
        assert output.read_text().strip() == (
            f"csv={tmp_path}/linux_{platform}/osrb-runtime-12345678.csv"
        )


@pytest.mark.parametrize("result", ["success", "skipped", "failure", "cancelled"])
@pytest.mark.timeout(10)
def test_audits_feed_required_backend_status(result):
    if shutil.which("jq") is None:
        pytest.skip("jq is required to exercise the workflow's status expression")
    jobs = _workflow("pr.yaml")["jobs"]
    gate = jobs["backend-status-check"]
    needs = {
        f"{framework}-compliance-audit": {"result": result}
        for framework in ["vllm", "sglang", "trtllm", "frontend"]
    }
    assert set(needs).issubset(gate["needs"])
    script = gate["steps"][0]["run"].replace("${{ toJson(needs) }}", json.dumps(needs))
    completed = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=5
    )
    assert (completed.returncode == 0) == (result in {"success", "skipped"})


def test_audits_fail_closed_and_retain_both_architecture_reports():
    job = _workflow("shared-compliance-audit.yml")["jobs"]["audit"]
    assert job["strategy"]["matrix"]["platform"] == ["amd64", "arm64"]
    assert all(
        step.get("continue-on-error", "false") == "false" for step in job["steps"]
    )
    steps = {step["name"]: step for step in job["steps"]}
    assert "--fail-on-findings" in steps["Audit final image vs osrb-deps.csv"]["run"]
    assert (
        "--fail-on-findings"
        in steps["Report final-image license-policy violations"]["run"]
    )
    upload = steps["Upload audit report"]
    assert upload["if"] == "always()"
    assert "${{ matrix.platform }}" in upload["with"]["name"]
    assert "/tmp/syft.json" in upload["with"]["path"]


@pytest.mark.parametrize("version", ["1", "2"])
def test_other_ecosystem_cannot_cover_python_dependency(tmp_path, version):
    sbom = tmp_path / "scan.json"
    sbom.write_text(
        json.dumps(
            {
                "components": [
                    {
                        "name": "shared-name",
                        "version": version,
                        "purl": f"pkg:pypi/shared-name@{version}",
                    }
                ]
            }
        )
    )
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("ecosystem,name,version,spdx\nrust,shared-name,1,MIT\n")
    assert (
        audit(
            [
                "--syft-sbom",
                str(sbom),
                "--our-csv",
                str(inventory),
                "--fail-on-findings",
            ]
        )
        == 1
    )
