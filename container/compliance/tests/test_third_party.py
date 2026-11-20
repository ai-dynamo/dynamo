# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for THIRD-PARTY.txt: the Dynamo NOTICES plus the base image's components.

Run from the repo root with the compliance package on the path:

    PYTHONPATH=container python -m pytest container/compliance/tests/test_third_party.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compliance.generators import dpkg
from compliance.generators.__main__ import main
from compliance.generators.common import (
    NOASSERTION,
    Component,
    load_base_components,
    spdx_license_text,
    write_third_party,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _write_base_sbom(directory: Path, name: str = "base@test-amd64.cdx.json") -> Path:
    """A baseline SBOM with two MIT libraries, an unlicensed one, a duplicate and a file."""
    directory.mkdir(parents=True, exist_ok=True)
    components = [
        {
            "type": "library",
            "name": "zlib1g",
            "version": "1.3",
            "purl": "pkg:deb/ubuntu/zlib1g@1.3",
            "licenses": [{"license": {"id": "Zlib"}}],
        },
        {
            "type": "library",
            "name": "libremoved",
            "version": "3.0",
            "purl": "pkg:deb/ubuntu/libremoved@3.0",
            "licenses": [{"license": {"id": "ISC"}}],
        },
        {
            "type": "library",
            "name": "libexpr",
            "version": "4.0",
            "licenses": [
                {"expression": "Apache-2.0 AND Apache-2.0 WITH LLVM-exception"},
                {"license": {"name": "Odd)License"}},
                {"license": {"name": "MIT OR Zlib"}},
            ],
        },
        {
            "type": "library",
            "name": "liba",
            "version": "1.0",
            "licenses": [{"license": {"id": "MIT"}}],
        },
        {
            "type": "library",
            "name": "libb",
            "version": "2.0",
            "licenses": [
                {"license": {"name": "MIT License"}},
                {"license": {"name": "Apache 2.0"}},
            ],
        },
        {"type": "library", "name": "libnolicense", "version": "0.1"},
        {
            "type": "library",
            "name": "libnolicense",
            "version": "0.1",
            "licenses": [{"license": {"name": "sha256:0123abcd"}}],
        },
        {"type": "file", "name": "/etc/skipped-file"},
        {"type": "operating-system", "name": "ubuntu", "version": "24.04"},
    ]
    path = directory / name
    path.write_text(json.dumps({"bomFormat": "CycloneDX", "components": components}))
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "baseline_sbom": name,
                        "baseline_image": "nvcr.io/nvidia/base/ubuntu",
                        "baseline_tag": "noble-test",
                    }
                ]
            }
        )
    )
    return path


def _write_go_sbom(path: Path) -> Path:
    component = {
        "type": "library",
        "name": "github.com/example/mod",
        "version": "v1.2.3",
        "purl": "pkg:golang/github.com/example/mod@v1.2.3",
        "licenses": [{"license": {"id": "BSD-3-Clause"}}],
    }
    path.write_text(json.dumps({"bomFormat": "CycloneDX", "components": [component]}))
    return path


def test_concatenates_notices_files(tmp_path: Path) -> None:
    (tmp_path / "NOTICES-Go.txt").write_text("GO NOTICES BODY\n")
    (tmp_path / "NOTICES-Rust.txt").write_text("RUST NOTICES BODY\n")
    (tmp_path / "NOTICES-Python.txt").write_text("STALE PYTHON NOTICES\n")
    (tmp_path / "osrb-deps.csv").write_text("not a notices file\n")

    text = write_third_party(tmp_path, ["rust", "go"]).read_text()

    assert text.startswith("THIRD-PARTY SOFTWARE NOTICES\n")
    assert text.index("GO NOTICES BODY") < text.index("RUST NOTICES BODY")
    assert "not a notices file" not in text
    assert "STALE PYTHON NOTICES" not in text
    assert "BASE IMAGE" not in text


def test_source_location_only_when_given(tmp_path: Path) -> None:
    url = "https://example.invalid/dynamo-sources.zip"

    assert url not in write_third_party(tmp_path, []).read_text()
    text = write_third_party(tmp_path, [], source_location=url).read_text()
    assert f"available at:\n{url}\n" in text


def test_base_components_parsed(tmp_path: Path) -> None:
    sbom = _write_base_sbom(tmp_path / "base_sboms")

    comps = {c.name: c for c in load_base_components(sbom)}

    assert sorted(comps) == [
        "liba",
        "libb",
        "libexpr",
        "libnolicense",
        "libremoved",
        "zlib1g",
    ]
    assert comps["libb"].spdx == "MIT License AND Apache 2.0"
    assert comps["libnolicense"].spdx == NOASSERTION


def test_base_section_renders_each_license_text_once(tmp_path: Path) -> None:
    sbom = _write_base_sbom(tmp_path / "base_sboms")
    out = tmp_path / "legal"
    out.mkdir()

    text = write_third_party(out, [], base_sbom=sbom).read_text()

    assert "BASE IMAGE — nvcr.io/nvidia/base/ubuntu:noble-test" in text
    assert "## liba 1.0\n\nLicenses: MIT\n" in text
    assert "## libnolicense 0.1\n\nLicenses: NOASSERTION\n" in text
    assert text.count("## libnolicense 0.1") == 1
    assert "/etc/skipped-file" not in text
    mit = spdx_license_text("MIT")
    assert mit and text.count(mit) == 1
    zlib = spdx_license_text("Zlib")
    assert zlib and text.count(zlib) == 1
    apache = spdx_license_text("Apache-2.0")
    assert apache and text.count(apache) == 1
    missing = text.split("is the reference:\n", 1)[1]
    assert missing.splitlines() == ["Apache 2.0", "Odd)License"]


def test_base_debs_limited_to_shipped_versions(tmp_path: Path) -> None:
    sbom = _write_base_sbom(tmp_path / "base_sboms")
    root = tmp_path / "root"
    for pkg in ("zlib1g", "liba"):
        (root / "usr/share/doc" / pkg).mkdir(parents=True)
        (root / "usr/share/doc" / pkg / "copyright").write_text(f"{pkg} COPYRIGHT\n")

    comps = {
        c.name: c
        for c in load_base_components(sbom, {("zlib1g", "1.3")}, dpkg_root=root)
    }

    assert comps["zlib1g"].license_text == "zlib1g COPYRIGHT\n"
    assert comps["liba"].license_text is None
    assert "libremoved" not in comps


def _run_main(tmp_path: Path, ecosystem: str) -> str:
    sbom = _write_base_sbom(tmp_path / "base_sboms")
    go_sbom = _write_go_sbom(tmp_path / "sbom-go.cdx.json")
    out = tmp_path / "legal"
    rc = main(
        [
            "--ecosystem",
            ecosystem,
            "--go-sbom",
            str(go_sbom),
            "--output-dir",
            str(out),
            "--subtract-sbom",
            str(sbom),
            "--source-location",
            "https://example.invalid/sources.zip",
            "--dpkg-root",
            str(tmp_path / "root"),
        ]
    )
    assert rc == 0
    return (out / "THIRD-PARTY.txt").read_text()


def test_main_go_only_keeps_every_base_deb(tmp_path: Path) -> None:
    text = _run_main(tmp_path, "go")

    assert "https://example.invalid/sources.zip" in text
    assert text.index("## github.com/example/mod v1.2.3") < text.index("BASE IMAGE")
    assert "## zlib1g 1.3" in text
    assert "## libremoved 3.0" in text


def test_main_dpkg_limits_base_debs_to_shipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped = [Component(ecosystem="dpkg", name="zlib1g", version="1.3", spdx="Zlib")]
    monkeypatch.setattr(dpkg, "collect_components", lambda root=None: shipped)

    text = _run_main(tmp_path, "go,dpkg")

    assert "## zlib1g 1.3" in text
    assert "## libremoved 3.0" not in text


def test_main_dpkg_without_packages_keeps_every_base_deb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dpkg, "collect_components", lambda root=None: [])

    assert "## libremoved 3.0" in _run_main(tmp_path, "go,dpkg")


def test_base_label_falls_back_to_filename(tmp_path: Path) -> None:
    sbom = _write_base_sbom(tmp_path / "base_sboms")
    (sbom.parent / "manifest.json").write_text(
        json.dumps({"entries": [{"baseline_sbom": sbom.name}]})
    )

    text = write_third_party(tmp_path, [], base_sbom=sbom).read_text()

    assert f"BASE IMAGE — {sbom.name}" in text
