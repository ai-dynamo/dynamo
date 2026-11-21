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
from compliance.generators.__main__ import main, source_location
from compliance.generators.common import (
    NOASSERTION,
    Component,
    load_base_components,
    spdx_license_text,
    write_third_party,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

# A free-text name from the cuda baselines: its lowercase "and" is not an operator.
NVIDIA_LICENSE_NAME = "NVIDIA Software License Agreement and CUDA Supplement to Software License Agreement"


def _write_base_sbom(directory: Path, name: str = "base@test-amd64.cdx.json") -> Path:
    """A baseline SBOM with two MIT libraries, an unlicensed one, a duplicate and a file."""
    directory.mkdir(parents=True, exist_ok=True)
    components = [
        {
            "type": "library",
            "name": "zlib1g",
            "version": "1.3",
            "purl": "pkg:deb/ubuntu/zlib1g@1.3",
            "licenses": [
                {"license": {"id": "Zlib"}},
                {"license": {"name": "Expat"}},
                {"license": {"name": "Artistic"}},
                {"license": {"name": "public-domain"}},
            ],
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
                {"license": {"name": "UNKNOWN"}},
                {"license": {"name": "(MIT"}},
                {"license": {"name": "Late)License"}},
                {"license": {"name": "Apache-2.0 with LLVM exceptions"}},
            ],
        },
        {
            "type": "library",
            "name": "liba",
            "version": "1.0",
            "licenses": [
                {"license": {"id": "MIT"}},
                {"license": {"id": "LGPL-3.0+"}},
                {"license": {"id": "GPL-2.0-or-later"}},
            ],
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
        {
            "type": "library",
            "name": "cuda-libs",
            "version": "13.0",
            "licenses": [{"license": {"name": NVIDIA_LICENSE_NAME}}],
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

    text = write_third_party(tmp_path, ["rust", "go"]).read_text(encoding="utf-8")

    assert text.startswith("THIRD-PARTY SOFTWARE NOTICES\n")
    assert text.index("GO NOTICES BODY") < text.index("RUST NOTICES BODY")
    assert "not a notices file" not in text
    assert "STALE PYTHON NOTICES" not in text
    assert "BASE IMAGE" not in text


def test_source_location_only_when_given(tmp_path: Path) -> None:
    url = "https://example.invalid/dynamo-sources.zip"

    omitted = write_third_party(tmp_path, []).read_text(encoding="utf-8")
    assert "Source code for the open-source software" not in omitted
    text = write_third_party(tmp_path, [], source_location=url).read_text(
        encoding="utf-8"
    )
    assert f"available at:\n{url}\n" in text


def test_base_components_parsed(tmp_path: Path) -> None:
    sbom = _write_base_sbom(tmp_path / "base_sboms")

    comps = {c.name: c for c in load_base_components(sbom)}

    assert sorted(comps) == [
        "cuda-libs",
        "liba",
        "libb",
        "libexpr",
        "libnolicense",
        "libremoved",
        "zlib1g",
    ]
    assert comps["libb"].spdx == "MIT License AND Apache 2.0"
    assert comps["libexpr"].spdx == (
        "(Apache-2.0 AND Apache-2.0 WITH LLVM-exception) AND Odd)License"
        " AND (MIT OR Zlib) AND UNKNOWN AND (MIT AND Late)License"
        " AND Apache-2.0 with LLVM exceptions"
    )
    assert comps["libexpr"].license_ids == (
        "Apache-2.0",
        "LLVM-exception",
        "Odd)License",
        "MIT",
        "Zlib",
        "UNKNOWN",
        "MIT",
        "Late)License",
        "Apache-2.0 with LLVM exceptions",
    )
    assert comps["libnolicense"].spdx == NOASSERTION
    assert comps["zlib1g"].spdx == "Zlib AND MIT AND Artistic AND public-domain"
    assert "libremoved" in {c.name for c in load_base_components(sbom, set())}


def test_base_section_renders_each_license_text_once(tmp_path: Path) -> None:
    sbom = _write_base_sbom(tmp_path / "base_sboms")
    out = tmp_path / "legal"
    out.mkdir()

    text = write_third_party(out, [], base_sbom=sbom).read_text(encoding="utf-8")

    assert "BASE IMAGE — nvcr.io/nvidia/base/ubuntu:noble-test" in text
    assert "## liba 1.0\n\nLicenses: MIT AND LGPL-3.0+ AND GPL-2.0-or-later\n" in text
    assert "## libnolicense 0.1\n\nLicenses: NOASSERTION\n" in text
    assert text.count("## libnolicense 0.1") == 1
    assert "/etc/skipped-file" not in text
    mit = spdx_license_text("MIT")
    assert mit and text.count(mit) == 1
    lgpl3 = spdx_license_text("LGPL-3.0")
    gpl2 = spdx_license_text("GPL-2.0-or-later")
    assert gpl2 and text.count(gpl2) == 1
    assert lgpl3 and text.count(lgpl3) == 1
    zlib = spdx_license_text("Zlib")
    assert zlib and text.count(zlib) == 1
    apache = spdx_license_text("Apache-2.0")
    assert apache and text.count(apache) == 1
    missing = text.split("is the reference:\n", 1)[1]
    assert missing.splitlines() == [
        "Apache 2.0",
        "Apache-2.0 with LLVM exceptions",
        "Artistic",
        "Late)License",
        NVIDIA_LICENSE_NAME,
        "Odd)License",
        "public-domain",
    ]
    assert "\nCUDA\n" not in missing
    artistic2 = spdx_license_text("Artistic-2.0")
    assert artistic2 and artistic2 not in text


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


@pytest.mark.parametrize("version", [None, "1.7.0"])
def test_main_points_to_the_versioned_ngc_source_archive(
    tmp_path: Path, version: str | None
) -> None:
    out = tmp_path / "legal"
    go_sbom = _write_go_sbom(tmp_path / "sbom-go.cdx.json")
    args = ["--ecosystem", "go", "--go-sbom", str(go_sbom), "--output-dir", str(out)]

    assert main(args + (["--source-version", version] if version else [])) == 0

    text = (out / "THIRD-PARTY.txt").read_text(encoding="utf-8")
    assert f"available at:\n{source_location(version)}\n" in text
    if version:
        assert (
            "https://catalog.ngc.nvidia.com/orgs/nvidia/ai-dynamo/resources/"
            "dynamo-dependency-source-archive/1.7.0\n"
        ) in text
        assert (
            "download-version nvidia/ai-dynamo/dynamo-dependency-source-archive:1.7.0"
            in text
        )


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
    return (out / "THIRD-PARTY.txt").read_text(encoding="utf-8")


def test_main_go_only_keeps_every_base_deb(tmp_path: Path) -> None:
    text = _run_main(tmp_path, "go")

    assert "https://example.invalid/sources.zip" in text
    assert text.index("## github.com/example/mod v1.2.3") < text.index("BASE IMAGE")
    assert "## zlib1g 1.3" in text
    assert "## libremoved 3.0" in text
    assert "limited to those still installed" not in text


def test_main_dpkg_limits_base_debs_to_shipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped = [Component(ecosystem="dpkg", name="zlib1g", version="1.3", spdx="Zlib")]
    calls: list[Path] = []

    def collect(root: Path) -> list[Component]:
        calls.append(root)
        return shipped

    monkeypatch.setattr(dpkg, "collect_components", collect)

    text = _run_main(tmp_path, "go,dpkg")

    assert len(calls) == 1
    assert "## zlib1g 1.3" in text
    assert "## libremoved 3.0" not in text
    assert "limited to those still installed" in text


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

    text = write_third_party(tmp_path, [], base_sbom=sbom).read_text(encoding="utf-8")

    assert f"BASE IMAGE — {sbom.name}" in text


@pytest.mark.parametrize("package_name", ["base-pkg", "Base_Pkg"])
@pytest.mark.parametrize("bundled_license", [False, True])
def test_main_retains_baseline_python_license_text(
    tmp_path: Path, package_name: str, bundled_license: bool
) -> None:
    site = tmp_path / "site-packages"
    dist = site / "base_pkg-1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: {package_name}\nVersion: 1.0\n"
        "License-Expression: MIT\n"
    )
    copyright_text = "Copyright (c) 2026 The Actual Package Authors"
    if bundled_license:
        (dist / "licenses").mkdir()
        (dist / "licenses" / "LICENSE.txt").write_text(copyright_text)
    sbom = tmp_path / "base.cdx.json"
    sbom.write_text(
        json.dumps(
            {
                "components": [
                    {
                        "type": "library",
                        "name": "base-pkg",
                        "version": "1.0",
                        "purl": "pkg:pypi/base-pkg@1.0",
                        "licenses": [{"license": {"id": "MIT"}}],
                    }
                ]
            }
        )
    )
    out = tmp_path / "legal"

    assert (
        main(
            [
                "--ecosystem",
                "python",
                "--site-packages",
                str(site),
                "--subtract-sbom",
                str(sbom),
                "--output-dir",
                str(out),
            ]
        )
        == 0
    )

    text = (out / "THIRD-PARTY.txt").read_text(encoding="utf-8")
    assert "## base-pkg 1.0" in text
    assert (copyright_text in text) == bundled_license
    assert "MIT License" in text  # Canonical text remains available as fallback.
    # The retained base license must not turn the delta reports into full inventories.
    assert "base-pkg" not in (out / "osrb-deps.csv").read_text(encoding="utf-8")
    assert "Base_Pkg" not in (out / "osrb-deps.csv").read_text(encoding="utf-8")
    assert (
        json.loads((out / "osrb.cdx.json").read_text(encoding="utf-8"))["components"]
        == []
    )


def test_base_python_license_matches_version_and_ecosystem(tmp_path: Path) -> None:
    sbom = tmp_path / "base.cdx.json"
    sbom.write_text(
        json.dumps(
            {
                "components": [
                    {
                        "type": "library",
                        "name": "example",
                        "version": version,
                        "purl": f"pkg:{ecosystem}/example@{version}",
                        "licenses": [{"license": {"id": "MIT"}}],
                    }
                    for version, ecosystem in [
                        ("1.0", "pypi"),
                        ("2.0", "pypi"),
                        ("3.0", "deb"),
                    ]
                ]
            }
        )
    )
    collected = [
        Component(
            ecosystem="python",
            name="example",
            version=version,
            spdx="MIT",
            license_text="ACTUAL LICENSE",
        )
        for version in ("1.0", "3.0")
    ]

    components = load_base_components(sbom, python_components=collected)

    assert {c.version: c.license_text for c in components} == {
        "1.0": "ACTUAL LICENSE",
        "2.0": None,
        "3.0": None,
    }


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "other_licenses",
    [
        [],
        [{"license": {"name": "The Apache Software License, Version 2.0"}}],
        [{"license": {"id": "Apache-2.0"}}, {"license": {"name": "Opaque License"}}],
    ],
)
def test_base_duplicates_prefer_resolvable_spdx(
    tmp_path: Path, reverse: bool, other_licenses: list[dict]
) -> None:
    components = [
        {
            "type": "library",
            "name": "jackson-core",
            "version": "2.16.0",
            "licenses": licenses,
        }
        for licenses in [other_licenses, [{"license": {"id": "Apache-2.0"}}]]
    ]
    sbom = tmp_path / "base.cdx.json"
    sbom.write_text(
        json.dumps({"components": components[::-1] if reverse else components})
    )

    (component,) = load_base_components(sbom)

    assert component.spdx == "Apache-2.0"
    assert component.license_ids == ("Apache-2.0",)
