# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for final-image CycloneDX license-policy validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from compliance.policy.validate_sbom import main

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _component(
    name: str,
    version: str,
    license_id: str | None,
    *,
    license_as_name: bool = False,
) -> dict:
    component = {
        "type": "library",
        "name": name,
        "version": version,
        "purl": f"pkg:deb/ubuntu/{name}@{version}",
    }
    if license_id is not None:
        key = "name" if license_as_name else "id"
        component["licenses"] = [{"license": {key: license_id}}]
    return component


def _write_json(path: Path, components: list[dict]) -> Path:
    path.write_text(json.dumps({"components": components}), encoding="utf-8")
    return path


def _write_policy(path: Path) -> Path:
    path.write_text(
        """[licenses]
allow = ["MIT", "LicenseRef-NVIDIA-Proprietary"]
deny = ["GPL-3.0-only"]
unknown = "deny"
copyleft = "deny"
""",
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, rows: list[tuple[str, str, str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ecosystem", "name", "version", "spdx", "source_url", "notes"])
        for ecosystem, name, version, spdx in rows:
            writer.writerow([ecosystem, name, version, spdx, "", ""])
    return path


def test_reports_only_final_image_delta_violations(tmp_path, capsys):
    baseline = _write_json(
        tmp_path / "baseline.cdx.json",
        [_component("baseline-gpl", "1", "GPL-3.0-only")],
    )
    image = _write_json(
        tmp_path / "image.cdx.json",
        [
            _component("baseline-gpl", "1", "GPL-3.0-only"),
            _component("allowed", "2", "MIT"),
            _component("shipped-gpl", "3", "GPL-3.0-only"),
        ],
    )
    report = tmp_path / "policy-report.txt"

    rc = main(
        [
            "--syft-sbom",
            str(image),
            "--base-sbom",
            str(baseline),
            "--our-csv",
            str(
                _write_csv(
                    tmp_path / "osrb.csv",
                    [
                        ("dpkg", "baseline-gpl", "1", "GPL-3.0-only"),
                        ("dpkg", "allowed", "2", "MIT"),
                        ("dpkg", "shipped-gpl", "3", "GPL-3.0-only"),
                    ],
                )
            ),
            "--policy",
            str(_write_policy(tmp_path / "licenses.toml")),
            "--report",
            str(report),
            "--fail-on-findings",
        ]
    )

    assert rc == 1
    output = capsys.readouterr().out
    assert "POLICY VIOLATIONS: 1" in output
    assert "unique attributed components checked: 2" in output
    assert "shipped-gpl@3" in output
    assert "baseline-gpl" not in output
    assert report.read_text(encoding="utf-8") == output


def test_unknown_license_is_reported_without_forcing_failure(tmp_path, capsys):
    image = _write_json(tmp_path / "image.cdx.json", [_component("mystery", "1", None)])

    rc = main(
        [
            "--syft-sbom",
            str(image),
            "--our-csv",
            str(
                _write_csv(tmp_path / "osrb.csv", [("dpkg", "mystery", "1", "UNKNOWN")])
            ),
            "--policy",
            str(_write_policy(tmp_path / "licenses.toml")),
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "POLICY VIOLATIONS: 1" in output
    assert "license is UNKNOWN" in output


def test_uses_reviewed_attribution_instead_of_syft_license(tmp_path, capsys):
    image = _write_json(
        tmp_path / "image.cdx.json",
        [
            _component(
                "nvidia-runtime",
                "1",
                "LicenseRef-NVIDIA-Proprietary",
                license_as_name=True,
            )
        ],
    )

    rc = main(
        [
            "--syft-sbom",
            str(image),
            "--our-csv",
            str(
                _write_csv(
                    tmp_path / "osrb.csv",
                    [
                        (
                            "dpkg",
                            "nvidia-runtime",
                            "1",
                            "LicenseRef-NVIDIA-Proprietary",
                        )
                    ],
                )
            ),
            "--policy",
            str(_write_policy(tmp_path / "licenses.toml")),
            "--fail-on-findings",
        ]
    )

    assert rc == 0
    assert "POLICY VIOLATIONS: 0" in capsys.readouterr().out


def test_reports_unmatched_component_count_without_inventing_license(tmp_path, capsys):
    image = _write_json(
        tmp_path / "image.cdx.json", [_component("unattributed", "1", "GPL-3.0-only")]
    )

    rc = main(
        [
            "--syft-sbom",
            str(image),
            "--our-csv",
            str(_write_csv(tmp_path / "osrb.csv", [])),
            "--policy",
            str(_write_policy(tmp_path / "licenses.toml")),
            "--fail-on-findings",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "unique components without exact attribution match: 1" in output
    assert "POLICY VIOLATIONS: 0" in output
