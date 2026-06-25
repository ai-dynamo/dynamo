# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for full-license-text inclusion in NOTICES.

Run from the repo root with the compliance package on the path:

    PYTHONPATH=container python -m pytest container/compliance/tests/test_license_text.py
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from compliance.generators import go as go_gen
from compliance.generators import rust as rust_gen
from compliance.generators.common import (
    Component,
    read_harvested_license,
    render_notices,
    spdx_license_text,
    subtract_baseline,
    write_cyclonedx,
    write_merged_csv,
)

_REPO = Path(__file__).resolve().parents[3]
_POLICY = _REPO / "container/compliance/policy/licenses.toml"


def _allowed_spdx_ids() -> set[str]:
    allow = tomllib.loads(_POLICY.read_text())["licenses"]["allow"]
    ids: set[str] = set()
    for entry in allow:
        e = str(entry)
        if e.startswith("LicenseRef"):
            continue
        if " WITH " in e:
            base, exc = e.split(" WITH ", 1)
            ids.add(base.strip())
            ids.add(exc.strip())
        else:
            ids.add(e.strip())
    return ids


def test_every_allowed_spdx_id_has_text():
    """Each allowed SPDX id resolves to canonical text — no allow-list gaps."""
    for lid in _allowed_spdx_ids():
        assert spdx_license_text(lid), f"missing canonical text for {lid}"


def test_spdx_lookup_behaviour():
    assert "MIT License" in spdx_license_text("MIT")
    both = spdx_license_text("MIT OR Apache-2.0")
    assert "--- MIT ---" in both and "--- Apache-2.0 ---" in both
    assert spdx_license_text("UNKNOWN") is None
    assert spdx_license_text("LicenseRef-Proprietary") is None


def test_canonical_text_carries_disclaimer():
    canon = Component(
        ecosystem="rust",
        name="serde",
        version="1.0.0",
        spdx="MIT",
        source_url="pkg:cargo/serde@1.0.0",
        license_text=spdx_license_text("MIT"),
        license_text_is_canonical=True,
    )
    out = render_notices("rust", [canon])
    assert "No license file was distributed" in out
    assert "MIT License" in out


def test_bundled_text_has_no_disclaimer():
    actual = Component(
        ecosystem="python",
        name="foo",
        version="1.0.0",
        spdx="MIT",
        source_url="https://pypi.org/project/foo/1.0.0/",
        license_text="MIT License\n\nCopyright (c) 2024 Real Author\n...",
        license_text_is_canonical=False,
    )
    out = render_notices("python", [actual])
    assert "No license file was distributed" not in out
    assert "Real Author" in out


def test_go_module_path_escaping():
    assert (
        go_gen._escape_go_module_path("github.com/Azure/Foo")
        == "github.com/!azure/!foo"
    )
    assert go_gen._escape_go_module_path("golang.org/x/sys") == "golang.org/x/sys"


def test_rust_prefers_harvested_over_canonical(tmp_path):
    crate = tmp_path / "serde-1.0.0"
    crate.mkdir()
    (crate / "LICENSE-MIT").write_text(
        "MIT License\n\nCopyright (c) 2014 Serde Authors"
    )
    comp = rust_gen._component_from_sbom_entry(
        {
            "name": "serde",
            "version": "1.0.0",
            "licenses": [{"license": {"id": "MIT"}}],
            "purl": "pkg:cargo/serde@1.0.0",
        },
        licenses_dir=tmp_path,
    )
    assert "Serde Authors" in comp.license_text
    assert comp.license_text_is_canonical is False
    # un-harvested crate falls back to canonical
    comp2 = rust_gen._component_from_sbom_entry(
        {
            "name": "tokio",
            "version": "1.0.0",
            "licenses": [{"license": {"id": "MIT"}}],
            "purl": "pkg:cargo/tokio@1.0.0",
        },
        licenses_dir=tmp_path,
    )
    assert comp2.license_text and comp2.license_text_is_canonical is True


def test_go_prefers_harvested_via_escaped_path(tmp_path):
    mod_dir = tmp_path / "github.com" / "!azure" / "foo@v1.0.0"
    mod_dir.mkdir(parents=True)
    (mod_dir / "LICENSE").write_text("Apache License 2.0\n\nCopyright Azure")
    comp = go_gen._component_from_sbom_entry(
        {
            "name": "github.com/Azure/foo",
            "version": "v1.0.0",
            "licenses": [{"license": {"id": "Apache-2.0"}}],
            "purl": "pkg:golang/github.com/Azure/foo@v1.0.0",
        },
        licenses_dir=tmp_path,
    )
    assert "Copyright Azure" in comp.license_text
    assert comp.license_text_is_canonical is False


def test_read_harvested_license_none_when_absent(tmp_path):
    assert read_harvested_license(None, "x-1.0") is None
    assert read_harvested_license(tmp_path, "missing-9.9") is None


def _make_wheel(path, dist_info, extra_members):
    import json
    import zipfile

    members = {
        f"{dist_info[:-10]}/__init__.py": b"",
        f"{dist_info}/METADATA": b"Metadata-Version: 2.1\nName: x\nVersion: 1.0\n",
        **extra_members,
    }
    record = "".join(f"{n},,{len(d)}\n" for n, d in members.items())
    record += f"{dist_info}/RECORD,,\n"
    with zipfile.ZipFile(path, "w") as z:
        for n, d in members.items():
            z.writestr(n, d)
        z.writestr(f"{dist_info}/RECORD", record)
    return json  # unused; keeps import local


def test_bundle_wheel_notices_injects_and_keeps_record_valid(tmp_path):
    import base64
    import hashlib
    import json
    import zipfile

    from compliance import bundle_wheel_notices

    di = "foo-1.0.dist-info"
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "components": [
            {
                "type": "library",
                "name": "serde",
                "version": "1.0.0",
                "purl": "pkg:cargo/serde@1.0.0",
                "licenses": [{"expression": "MIT OR Apache-2.0"}],
            }
        ],
    }
    whl = tmp_path / "foo-1.0-py3-none-any.whl"
    _make_wheel(whl, di, {f"{di}/sboms/foo.cyclonedx.json": json.dumps(sbom).encode()})

    assert bundle_wheel_notices.process(whl, None) == 0

    arc = f"{di}/licenses/THIRD-PARTY-RUST-LICENSES.txt"
    with zipfile.ZipFile(whl) as z:
        assert arc in z.namelist()
        data = z.read(arc)
        assert b"serde" in data and b"MIT" in data
        want = "sha256=" + base64.urlsafe_b64encode(
            hashlib.sha256(data).digest()
        ).decode().rstrip("=")
        record = z.read(f"{di}/RECORD").decode()
        assert any(
            ln.startswith(arc) and want in ln and str(len(data)) in ln
            for ln in record.splitlines()
        )


def test_bundle_wheel_notices_noop_without_sbom(tmp_path):
    import zipfile

    from compliance import bundle_wheel_notices

    di = "bar-1.0.dist-info"
    whl = tmp_path / "bar-1.0-py3-none-any.whl"
    _make_wheel(whl, di, {})
    assert bundle_wheel_notices.process(whl, None) == 0
    with zipfile.ZipFile(whl) as z:
        assert not any("THIRD-PARTY" in n for n in z.namelist())


def _sample_components() -> list[Component]:
    return [
        Component(
            "python",
            "accelerate",
            "1.0",
            "Apache-2.0",
            "https://pypi.org/project/accelerate/1.0/",
        ),
        Component(
            "python",
            "nvidia-cublas",
            "13.1",
            "LicenseRef-NVIDIA-Proprietary",
            "https://pypi.org/project/nvidia-cublas/13.1/",
        ),
        Component("rust", "serde", "1.0", "MIT", "pkg:cargo/serde@1.0"),
        Component(
            "dpkg", "libstdc++6", "14", "GPL-3.0-or-later WITH GCC-exception-3.1", None
        ),
    ]


def test_write_merged_csv_notes_only_for_non_permissive(tmp_path):
    import csv as _csv

    comps = _sample_components()
    # Only non-permissive packages (those with a matching exception) get a note.
    exc = {
        ("dpkg", "libstdc++6"): "GNU toolchain: GPL-3.0 with GCC runtime exception.",
        ("python", "nvidia-cublas"): "NVIDIA proprietary CUDA wheel; redistributable.",
    }
    out = tmp_path / "osrb-deps.csv"
    write_merged_csv(comps, exc, out)
    rows = {(r["ecosystem"], r["name"]): r for r in _csv.DictReader(out.open())}
    # 6-column schema incl. notes
    assert list(next(iter(rows.values())).keys()) == [
        "ecosystem",
        "name",
        "version",
        "spdx",
        "source_url",
        "notes",
    ]
    assert rows[("dpkg", "libstdc++6")]["notes"].startswith("GNU toolchain")
    assert rows[("python", "nvidia-cublas")]["notes"]
    # Permissive packages carry no note.
    assert rows[("python", "accelerate")]["notes"] == ""
    assert rows[("rust", "serde")]["notes"] == ""


def test_subtract_baseline_normalizes_python_name_separators():
    comps = [
        Component("python", "arctic_inference", "0.1.1", "UNKNOWN", None),
        Component("python", "vllm_test_utils", "0.1", "UNKNOWN", None),
        Component("python", "keep_me", "1.0.0", "MIT", None),
    ]
    baseline = {
        ("arctic-inference", "0.1.1"),
        ("vllm-test-utils", "0.1"),
    }

    kept = subtract_baseline(comps, baseline)

    assert [(c.name, c.version) for c in kept] == [("keep_me", "1.0.0")]


def test_write_cyclonedx_is_valid_delta_sbom(tmp_path):
    import json

    out = tmp_path / "osrb.cdx.json"
    write_cyclonedx(_sample_components(), out)
    bom = json.loads(out.read_text())
    assert bom["bomFormat"] == "CycloneDX" and bom["specVersion"] == "1.5"
    by_name = {c["name"]: c for c in bom["components"]}
    assert len(by_name) == 4
    # SPDX id form, LicenseRef name form, compound expression form, purls.
    assert by_name["serde"]["licenses"] == [{"license": {"id": "MIT"}}]
    assert by_name["nvidia-cublas"]["licenses"] == [
        {"license": {"name": "LicenseRef-NVIDIA-Proprietary"}}
    ]
    assert by_name["libstdc++6"]["licenses"] == [
        {"expression": "GPL-3.0-or-later WITH GCC-exception-3.1"}
    ]
    assert by_name["serde"]["purl"] == "pkg:cargo/serde@1.0"
    assert by_name["accelerate"]["purl"] == "pkg:pypi/accelerate@1.0"


def test_nvidia_proprietary_licenseref_resolves_text():
    # Part C: the proprietary CUDA wheels (LicenseRef-NVIDIA-Proprietary) now
    # carry text via the canonical fallback; a generic LicenseRef still does not.
    assert spdx_license_text("LicenseRef-NVIDIA-Proprietary")
    assert spdx_license_text("LicenseRef-Proprietary") is None
