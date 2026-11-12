# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real archive selection, baseline validation, and comparison failure modes."""

import io
import json
import subprocess
from zipfile import ZipFile

import pytest
from compliance import prepare_diff
from compliance.prepare_diff import (
    ComparisonError,
    inventory_name,
    prepare_comparison,
    select_osrb_csv,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]
CURRENT = "a" * 40
BASE = "b" * 40
CSV = "ecosystem,name,version,spdx,source_url,notes\npython,example,1.0,MIT,,\n"


def write_inventory(
    root, arch="amd64", sha=CURRENT, prefix="vllm-runtime", cuda="13.0"
):
    path = root / f"linux_{arch}" / inventory_name(prefix, arch, sha, cuda)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CSV)
    return path


def archive_with(*names):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        for name in names:
            archive.writestr(name, CSV)
    return stream.getvalue()


def compare(root, **kwargs):
    return prepare_comparison(
        root, "org/repo", "vllm-runtime", "linux/amd64", CURRENT, "13.0", **kwargs
    )


def test_selects_exact_current_inventory_excluding_history_and_variants(tmp_path):
    expected = write_inventory(tmp_path)
    write_inventory(tmp_path, arch="arm64")
    write_inventory(tmp_path, prefix="vllm-runtime-efa")
    write_inventory(tmp_path, cuda="12.9")
    write_inventory(tmp_path / "baseline", sha=BASE)
    expected.with_suffix(".diff.csv").write_text("wrong")
    assert (
        select_osrb_csv(tmp_path, "vllm-runtime", "amd64", CURRENT, "13.0") == expected
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arch": "arm64"},
        {"sha": BASE},
        {"prefix": "vllm-runtime-efa"},
        {"cuda": "12.9"},
    ],
)
def test_rejects_arch_sha_variant_and_cuda_mismatch(tmp_path, kwargs):
    write_inventory(tmp_path, **kwargs)
    with pytest.raises(ComparisonError, match="Expected one"):
        select_osrb_csv(tmp_path, "vllm-runtime", "amd64", CURRENT, "13.0")


def test_ambiguous_flat_and_arch_inventories_fail(tmp_path):
    path = write_inventory(tmp_path)
    (tmp_path / path.name).write_text(CSV)
    with pytest.raises(ComparisonError, match="found 2"):
        select_osrb_csv(tmp_path, "vllm-runtime", "amd64", CURRENT, "13.0")


def test_required_baseline_cannot_silently_emit_unavailable(tmp_path):
    write_inventory(tmp_path)
    with pytest.raises(ComparisonError, match="Expected baseline is missing"):
        compare(tmp_path)
    assert not list(tmp_path.rglob("*.diff.csv"))


def test_explicit_bootstrap_records_distinct_status(tmp_path):
    current = write_inventory(tmp_path)
    compare(tmp_path, allow_bootstrap=True)
    assert "baseline_bootstrap" in current.with_suffix(".diff.csv").read_text()
    assert (
        json.loads((current.parent / "baseline/baseline.json").read_text())["status"]
        == "bootstrap"
    )


def test_download_failure_cannot_bootstrap_or_write_successful_diff(
    tmp_path, monkeypatch
):
    current = write_inventory(tmp_path)

    def fail(*args):
        raise subprocess.CalledProcessError(1, "gh")

    monkeypatch.setattr(prepare_diff, "download_baseline", fail)
    with pytest.raises(subprocess.CalledProcessError):
        compare(tmp_path, base_sha=BASE, artifact_id="7", run_id="90")
    assert not current.with_suffix(".diff.csv").exists()
    with pytest.raises(ComparisonError, match="cannot bypass"):
        compare(
            tmp_path, base_sha=BASE, artifact_id="7", run_id="90", allow_bootstrap=True
        )


@pytest.mark.parametrize(
    "member",
    [
        f"linux_arm64/osrb-vllm-runtime-13.0-arm64-{BASE[:8]}.csv",
        f"linux_amd64/osrb-vllm-runtime-efa-13.0-amd64-{BASE[:8]}.csv",
        f"linux_amd64/osrb-vllm-runtime-12.9-amd64-{BASE[:8]}.csv",
        f"linux_amd64/baseline/osrb-vllm-runtime-13.0-amd64-{BASE[:8]}.csv",
        f"linux_amd64/osrb-vllm-runtime-13.0-amd64-{BASE[:8]}.diff.csv",
    ],
)
def test_baseline_archive_does_not_substitute_other_inventory(
    tmp_path, monkeypatch, member
):
    write_inventory(tmp_path)
    monkeypatch.setattr(
        prepare_diff, "download_baseline", lambda *a: archive_with(member)
    )
    with pytest.raises(ComparisonError, match="matching baseline"):
        compare(tmp_path, base_sha=BASE, artifact_id="7", run_id="90")
    assert not list(tmp_path.rglob("*.diff.csv"))


def test_valid_comparison_preserves_exact_provenance(tmp_path, monkeypatch):
    current = write_inventory(tmp_path)
    member = f"linux_amd64/{inventory_name('vllm-runtime', 'amd64', BASE, '13.0')}"
    monkeypatch.setattr(
        prepare_diff, "download_baseline", lambda *a: archive_with(member)
    )
    compare(tmp_path, base_sha=BASE, artifact_id="7", run_id="90")
    assert len(current.with_suffix(".diff.csv").read_text().splitlines()) == 1
    metadata = json.loads((current.parent / "baseline/baseline.json").read_text())
    assert metadata["source_sha"] == BASE
    assert metadata["artifact_id"] == "7"
    assert metadata["run_id"] == "90"
    assert (
        current.parent
        / "baseline"
        / inventory_name("vllm-runtime", "amd64", BASE, "13.0")
    ).read_text() == CSV


@pytest.mark.parametrize(
    "field,value",
    [("expired", True), ("name", "compliance-wrong"), ("id", 9), ("size_in_bytes", 0)],
)
def test_invalid_artifact_metadata_is_rejected(field, value):
    metadata = {
        "id": 7,
        "name": f"compliance-{BASE}-vllm-runtime",
        "expired": False,
        "size_in_bytes": 100,
        "workflow_run": {"id": 90, "head_sha": BASE},
    }
    metadata[field] = value
    with pytest.raises(ComparisonError, match="does not match"):
        prepare_diff.validate_artifact(metadata, "7", "90", BASE, "vllm-runtime")


def test_same_sha_other_run_artifact_is_rejected():
    metadata = {
        "id": 7,
        "name": f"compliance-{BASE}-vllm-runtime",
        "expired": False,
        "size_in_bytes": 100,
        "workflow_run": {"id": 91, "head_sha": BASE},
    }
    with pytest.raises(ComparisonError, match="does not match"):
        prepare_diff.validate_artifact(metadata, "7", "90", BASE, "vllm-runtime")


def test_empty_or_malformed_inventory_fails(tmp_path):
    current = write_inventory(tmp_path)
    current.write_text("ecosystem,name,version,spdx,source_url,notes\n")
    with pytest.raises(ComparisonError, match="Empty"):
        compare(tmp_path, allow_bootstrap=True)
    current.write_text("arbitrary,columns\n")
    with pytest.raises(ComparisonError, match="columns"):
        compare(tmp_path, allow_bootstrap=True)


@pytest.mark.parametrize("flag,modern", [(False, False), (True, False), (True, True)])
def test_operator_arm64_migration_is_explicit_and_legacy_only(
    tmp_path, monkeypatch, flag, modern
):
    amd64 = write_inventory(tmp_path, prefix="operator", cuda="")
    arm64 = write_inventory(tmp_path, arch="arm64", prefix="operator", cuda="")
    members = [inventory_name("operator", "amd64", BASE)]
    if modern:
        members.append("build-provenance.json")
    monkeypatch.setattr(
        prepare_diff, "download_baseline", lambda *a: archive_with(*members)
    )
    args = (tmp_path, "org/repo", "operator", "linux/amd64,linux/arm64", CURRENT)
    kwargs = dict(
        base_sha=BASE,
        artifact_id="7",
        run_id="90",
        bootstrap_legacy_operator_arm64=flag,
    )
    if not flag or modern:
        with pytest.raises(ComparisonError):
            prepare_comparison(*args, **kwargs)
        assert not list(tmp_path.rglob("*.diff.csv"))
    else:
        prepare_comparison(*args, **kwargs)
        assert len(amd64.with_suffix(".diff.csv").read_text().splitlines()) == 1
        assert "baseline_bootstrap" in arm64.with_suffix(".diff.csv").read_text()
        record = json.loads((arm64.parent / "baseline/baseline.json").read_text())
        assert record["source_sha"] == BASE
        assert "legacy AMD64-only" in record["bootstrap_reason"]


@pytest.mark.parametrize(
    "prefix,old_arch", [("operator", "arm64"), ("frontend", "amd64")]
)
def test_operator_migration_cannot_hide_bad_architecture_or_other_image(
    tmp_path, monkeypatch, prefix, old_arch
):
    write_inventory(tmp_path, arch="arm64", prefix=prefix, cuda="")
    member = inventory_name(prefix, old_arch, CURRENT)  # wrong baseline SHA
    monkeypatch.setattr(
        prepare_diff, "download_baseline", lambda *a: archive_with(member)
    )
    with pytest.raises(ComparisonError):
        prepare_comparison(
            tmp_path,
            "org/repo",
            prefix,
            "linux/arm64",
            CURRENT,
            base_sha=BASE,
            artifact_id="7",
            run_id="90",
            bootstrap_legacy_operator_arm64=True,
        )
