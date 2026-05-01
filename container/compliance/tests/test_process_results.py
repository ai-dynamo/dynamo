# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for container/compliance/process_results.py.

These tests focus on the SPDX-driven ATTRIBUTIONS renderers, the native overlay,
and backwards-compatible CSV behavior. Rust/Go paths are exercised in Group B's
own test suites; here we only verify that process_results.py does not require
those modules to run the SPDX + CSV paths.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE_SPDX = HERE / "fixtures" / "sample.spdx.json"
SCRIPT = HERE.parent / "process_results.py"


def _load_process_results():
    """Load process_results.py as a module without requiring a parent package."""
    spec = importlib.util.spec_from_file_location("process_results_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


pr = _load_process_results()


# ---- Helpers ------------------------------------------------------------------


def _write_tsvs(target_dir: Path) -> None:
    (target_dir / "dpkg.tsv").write_text(
        "libfoo\t1.0\tApache-2.0\nlibbar\t2.0\tMIT\n", encoding="utf-8"
    )
    (target_dir / "python.tsv").write_text(
        "requests\t2.31.0\tApache-2.0\nnumpy\t1.26\tBSD-3-Clause\n", encoding="utf-8"
    )


def _copy_spdx(target_dir: Path) -> None:
    shutil.copy(FIXTURE_SPDX, target_dir / "sbom.spdx.json")


def _run_script(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=True,
        capture_output=True,
        text=True,
    )


# ---- Tests --------------------------------------------------------------------


def test_spdx_license_fallback_order():
    pkg_concluded = {"licenseConcluded": "Apache-2.0", "licenseDeclared": "MIT"}
    pkg_declared_only = {"licenseConcluded": "NOASSERTION", "licenseDeclared": "MIT"}
    pkg_none = {"licenseConcluded": "NOASSERTION", "licenseDeclared": "NOASSERTION"}
    assert pr._spdx_license_for_pkg(pkg_concluded) == "Apache-2.0"
    assert pr._spdx_license_for_pkg(pkg_declared_only) == "MIT"
    assert pr._spdx_license_for_pkg(pkg_none) == "UNKNOWN"


def test_first_party_filter():
    assert pr._is_first_party_python("ai-dynamo-runtime")
    assert pr._is_first_party_python("AI-Dynamo-Runtime")
    assert pr._is_first_party_python("dynamo-foo")
    assert not pr._is_first_party_python("requests")
    assert not pr._is_first_party_python("numpy")


def test_iter_spdx_packages_by_purl(tmp_path: Path):
    spdx = json.loads(FIXTURE_SPDX.read_text())
    debs = pr._iter_spdx_packages_by_purl(spdx, "deb")
    pypis = pr._iter_spdx_packages_by_purl(spdx, "pypi")
    deb_names = {p["name"] for p in debs}
    pypi_names = {p["name"] for p in pypis}
    assert deb_names == {"libfoo", "libbar"}
    # ai-dynamo-runtime is PyPI at this stage (filter applied at writer layer)
    assert pypi_names == {"requests", "numpy", "ai-dynamo-runtime"}
    # Fallback: numpy has no concluded/declared license → UNKNOWN
    numpy_pkg = next(p for p in pypis if p["name"] == "numpy")
    assert numpy_pkg["license"] == "UNKNOWN"
    # libbar has NOASSERTION concluded but MIT declared → MIT
    libbar_pkg = next(p for p in debs if p["name"] == "libbar")
    assert libbar_pkg["license"] == "MIT"


def test_write_attributions_apt_and_python(tmp_path: Path):
    spdx = json.loads(FIXTURE_SPDX.read_text())
    apt_path = tmp_path / "ATTRIBUTIONS-Apt.md"
    py_path = tmp_path / "ATTRIBUTIONS-Python.md"

    assert pr.write_attributions_apt(spdx, apt_path) == 2
    assert pr.write_attributions_python(spdx, py_path) == 2  # ai-dynamo filtered

    apt = apt_path.read_text()
    py = py_path.read_text()

    # Header present
    assert "Third-Party Software Attributions — Apt (Debian/Ubuntu)" in apt
    assert "Third-Party Software Attributions — Python (PyPI)" in py

    # Sorted case-insensitively by name
    assert apt.index("## libbar") < apt.index("## libfoo")
    assert py.index("## numpy") < py.index("## requests")

    # First-party filter drops ai-dynamo-runtime
    assert "ai-dynamo-runtime" not in py

    # UNKNOWN fallback shows up (numpy has NOASSERTION on both fields)
    assert "**License:** `UNKNOWN`" in py

    # Ends with exactly one trailing newline
    assert apt.endswith("\n") and not apt.endswith("\n\n")
    assert py.endswith("\n") and not py.endswith("\n\n")


def test_native_md_from_overlay(tmp_path: Path):
    yaml_path = tmp_path / "native.yaml"
    yaml_path.write_text(
        """
packages:
  - name: criu
    repo: https://github.com/checkpoint-restore/criu
    dockerfile: deploy/snapshot/Dockerfile
    artifacts:
      - { name: criu, license: GPL-2.0-only }
      - { name: libcriu, license: LGPL-2.1-only }
  - name: ucx
    repo: https://github.com/openucx/ucx
    dockerfile: container/templates/wheel_builder.Dockerfile
    artifacts:
      - { name: ucx, license: BSD-3-Clause }
""",
        encoding="utf-8",
    )
    out = tmp_path / "ATTRIBUTIONS-Native.md"
    count = pr.write_attributions_native(yaml_path, out)
    assert count == 3
    text = out.read_text()
    assert "Third-Party Software Attributions — Native" in text
    # Case-insensitive sort: criu, libcriu, ucx
    assert text.index("## criu ") < text.index("## libcriu ")
    assert text.index("## libcriu ") < text.index("## ucx ")
    assert "Source: https://github.com/checkpoint-restore/criu" in text
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_native_md_missing_file(tmp_path: Path):
    out = tmp_path / "ATTRIBUTIONS-Native.md"
    count = pr.write_attributions_native(tmp_path / "nope.yaml", out)
    assert count == 0
    assert not out.exists()


def test_missing_spdx_does_not_crash_csv(tmp_path: Path):
    """CSV pipeline must succeed even when sbom.spdx.json is absent."""
    _write_tsvs(tmp_path)
    out_csv = tmp_path / "out.csv"
    result = _run_script(["--target-dir", str(tmp_path), "--output", str(out_csv)])
    assert out_csv.is_file()
    body = out_csv.read_text()
    assert "package_name,version,type,spdx_license" in body
    assert "libfoo,1.0,dpkg,Apache-2.0" in body
    # Apt/Python attributions are skipped when SPDX is missing
    assert "sbom.spdx.json" in result.stderr
    assert not (tmp_path / "ATTRIBUTIONS-Apt.md").exists()


def test_end_to_end_roundtrip(tmp_path: Path):
    _write_tsvs(tmp_path)
    _copy_spdx(tmp_path)
    out_csv = tmp_path / "out.csv"
    _run_script(["--target-dir", str(tmp_path), "--output", str(out_csv)])

    assert out_csv.is_file() and out_csv.stat().st_size > 0
    assert (tmp_path / "ATTRIBUTIONS-Apt.md").is_file()
    assert (tmp_path / "ATTRIBUTIONS-Python.md").is_file()

    py_text = (tmp_path / "ATTRIBUTIONS-Python.md").read_text()
    assert "ai-dynamo-runtime" not in py_text


def test_csv_is_deterministic(tmp_path: Path):
    _write_tsvs(tmp_path)
    _copy_spdx(tmp_path)
    out1 = tmp_path / "out1.csv"
    out2 = tmp_path / "out2.csv"
    _run_script(["--target-dir", str(tmp_path), "--output", str(out1)])
    _run_script(["--target-dir", str(tmp_path), "--output", str(out2)])
    assert out1.read_bytes() == out2.read_bytes()


def test_csv_backwards_compat_bytes(tmp_path: Path):
    """With no new flags, CSV output must match the legacy two-column sort order.

    The csv module emits \\r\\n line terminators; read as bytes to verify.
    """
    _write_tsvs(tmp_path)
    # Intentionally NO spdx file, so no attributions-dir side effects.
    out_csv = tmp_path / "out.csv"
    _run_script(["--target-dir", str(tmp_path), "--output", str(out_csv)])
    expected = (
        b"package_name,version,type,spdx_license\r\n"
        b"libbar,2.0,dpkg,MIT\r\n"
        b"libfoo,1.0,dpkg,Apache-2.0\r\n"
        b"numpy,1.26,python,BSD-3-Clause\r\n"
        b"requests,2.31.0,python,Apache-2.0\r\n"
    )
    assert out_csv.read_bytes() == expected


def test_dedupe_csv_rows_collapses_same_key_prefer_known_license():
    rows = [
        {
            "package_name": "pyyaml",
            "version": "6.0.1",
            "type": "python",
            "spdx_license": "UNKNOWN",
        },
        {
            "package_name": "pyyaml",
            "version": "6.0.1",
            "type": "python",
            "spdx_license": "MIT",
        },
        {
            "package_name": "pyyaml",
            "version": "6.0.1",
            "type": "dpkg",
            "spdx_license": "MIT",
        },
    ]
    out = pr._dedupe_csv_rows(rows)
    assert len(out) == 2  # python keeps MIT, dpkg row preserved separately
    by_type = {r["type"]: r for r in out}
    assert by_type["python"]["spdx_license"] == "MIT"
    assert by_type["dpkg"]["spdx_license"] == "MIT"


def test_dedupe_flag_drops_duplicate_rows_in_csv(tmp_path: Path):
    """End-to-end: --dedupe collapses a duplicate (name, version, type) in the CSV."""
    (tmp_path / "dpkg.tsv").write_text("libfoo\t1.0\tApache-2.0\n", encoding="utf-8")
    (tmp_path / "python.tsv").write_text(
        "pyyaml\t6.0.1\tUNKNOWN\npyyaml\t6.0.1\tMIT\n", encoding="utf-8"
    )
    out_csv = tmp_path / "out.csv"
    _run_script(["--target-dir", str(tmp_path), "--output", str(out_csv), "--dedupe"])
    body = out_csv.read_bytes()
    assert body.count(b"pyyaml") == 1
    assert b"pyyaml,6.0.1,python,MIT" in body


def test_apply_overrides_to_csv_rewrites_unknown_rows(tmp_path: Path):
    overrides = tmp_path / "license_overrides.yaml"
    overrides.write_text(
        "overrides:\n"
        "  - {ecosystem: dpkg, name: openssl, license: Apache-2.0}\n"
        "  - {ecosystem: python, name: torchao, license: BSD-3-Clause}\n",
        encoding="utf-8",
    )
    rows = [
        {"package_name": "openssl", "version": "3.0.13", "type": "dpkg", "spdx_license": "UNKNOWN"},
        {"package_name": "torchao", "version": "0.15.0", "type": "python", "spdx_license": "UNKNOWN"},
        {"package_name": "torchao", "version": "0.16.0", "type": "python", "spdx_license": "MIT"},
        {"package_name": "requests", "version": "2.31.0", "type": "python", "spdx_license": "UNKNOWN"},
    ]
    rewritten = pr._apply_overrides_to_csv(rows, str(overrides))
    assert rewritten == 2
    by_name = {(r["package_name"], r["version"]): r for r in rows}
    assert by_name[("openssl", "3.0.13")]["spdx_license"] == "Apache-2.0"
    assert by_name[("torchao", "0.15.0")]["spdx_license"] == "BSD-3-Clause"
    assert by_name[("torchao", "0.16.0")]["spdx_license"] == "MIT"  # not overridden
    assert by_name[("requests", "2.31.0")]["spdx_license"] == "UNKNOWN"  # no entry


def test_apply_overrides_to_csv_handles_missing_yaml(tmp_path: Path):
    rows = [{"package_name": "foo", "version": "1", "type": "dpkg", "spdx_license": "UNKNOWN"}]
    assert pr._apply_overrides_to_csv(rows, str(tmp_path / "missing.yaml")) == 0
    assert rows[0]["spdx_license"] == "UNKNOWN"


def test_apply_overrides_to_csv_disabled_when_path_blank():
    rows = [{"package_name": "foo", "version": "1", "type": "dpkg", "spdx_license": "UNKNOWN"}]
    assert pr._apply_overrides_to_csv(rows, "") == 0


def test_apply_overrides_to_pkgs_rewrites_markdown_schema(tmp_path: Path):
    overrides = tmp_path / "license_overrides.yaml"
    overrides.write_text(
        "overrides:\n"
        "  - {ecosystem: dpkg, name: cuda-libraries-12-9, license: LicenseRef-NVIDIA-Software-License-Agreement}\n",
        encoding="utf-8",
    )
    pkgs = [
        {"name": "cuda-libraries-12-9", "version": "12.9.1-1", "license": "UNKNOWN"},
        {"name": "libfoo", "version": "1.0", "license": "Apache-2.0"},
    ]
    rewritten = pr._apply_overrides_to_pkgs(pkgs, "dpkg", str(overrides))
    assert rewritten == 1
    assert pkgs[0]["license"] == "LicenseRef-NVIDIA-Software-License-Agreement"
    assert pkgs[1]["license"] == "Apache-2.0"


def test_load_overrides_skips_blank_or_partial_rows(tmp_path: Path):
    overrides = tmp_path / "license_overrides.yaml"
    overrides.write_text(
        "overrides:\n"
        "  - {ecosystem: dpkg, name: openssl, license: Apache-2.0}\n"
        "  - {ecosystem: dpkg, name: incomplete}\n"
        "  - {name: missing-eco, license: MIT}\n"
        "  - {ecosystem: python, name: '', license: GPL-2.0-only}\n",
        encoding="utf-8",
    )
    pr._load_overrides.cache_clear()
    exact, patterns = pr._load_overrides(str(overrides))
    assert exact == {("dpkg", "openssl"): "Apache-2.0"}
    assert patterns == ()


def test_load_overrides_supports_name_patterns(tmp_path: Path):
    overrides = tmp_path / "license_overrides.yaml"
    overrides.write_text(
        "overrides:\n"
        "  - {ecosystem: dpkg, name_pattern: 'libpython3*', license: Python-2.0}\n"
        "  - {ecosystem: dpkg, name_pattern: 'cuda-*-12-9', license: LicenseRef-NVIDIA-Software-License-Agreement}\n"
        "  - {ecosystem: python, name: openssl, license: Apache-2.0}\n",
        encoding="utf-8",
    )
    pr._load_overrides.cache_clear()
    exact, patterns = pr._load_overrides(str(overrides))
    assert exact == {("python", "openssl"): "Apache-2.0"}
    assert patterns == (
        ("dpkg", "libpython3*", "Python-2.0"),
        (
            "dpkg",
            "cuda-*-12-9",
            "LicenseRef-NVIDIA-Software-License-Agreement",
        ),
    )


def test_apply_overrides_to_csv_uses_name_pattern(tmp_path: Path):
    overrides = tmp_path / "license_overrides.yaml"
    overrides.write_text(
        "overrides:\n"
        "  - {ecosystem: dpkg, name_pattern: 'libpython3*', license: Python-2.0}\n"
        "  - {ecosystem: dpkg, name_pattern: 'cuda-*-12-9', license: LicenseRef-NVIDIA-Software-License-Agreement}\n",
        encoding="utf-8",
    )
    pr._load_overrides.cache_clear()
    packages = [
        {
            "package_name": "libpython3.12-dev",
            "version": "3.12.3",
            "type": "dpkg",
            "spdx_license": "UNKNOWN",
        },
        {
            "package_name": "libpython3-stdlib",
            "version": "3.12.3",
            "type": "dpkg",
            "spdx_license": "UNKNOWN",
        },
        {
            "package_name": "cuda-libraries-12-9",
            "version": "12.9.0",
            "type": "dpkg",
            "spdx_license": "UNKNOWN",
        },
        {
            "package_name": "openssl",  # not in any override entry
            "version": "3.0",
            "type": "dpkg",
            "spdx_license": "UNKNOWN",
        },
    ]
    rewritten = pr._apply_overrides_to_csv(packages, str(overrides))
    assert rewritten == 3
    licenses = {p["package_name"]: p["spdx_license"] for p in packages}
    assert licenses["libpython3.12-dev"] == "Python-2.0"
    assert licenses["libpython3-stdlib"] == "Python-2.0"
    assert (
        licenses["cuda-libraries-12-9"]
        == "LicenseRef-NVIDIA-Software-License-Agreement"
    )
    assert licenses["openssl"] == "UNKNOWN"


def test_apply_overrides_to_csv_exact_wins_over_pattern(tmp_path: Path):
    """Exact name overrides take precedence over a glob pattern that matches."""
    overrides = tmp_path / "license_overrides.yaml"
    overrides.write_text(
        "overrides:\n"
        "  - {ecosystem: dpkg, name_pattern: 'lib*', license: GPL-2.0-only}\n"
        "  - {ecosystem: dpkg, name: libfoo, license: Apache-2.0}\n",
        encoding="utf-8",
    )
    pr._load_overrides.cache_clear()
    packages = [
        {
            "package_name": "libfoo",
            "version": "1.0",
            "type": "dpkg",
            "spdx_license": "UNKNOWN",
        },
        {
            "package_name": "libbar",
            "version": "1.0",
            "type": "dpkg",
            "spdx_license": "UNKNOWN",
        },
    ]
    pr._apply_overrides_to_csv(packages, str(overrides))
    licenses = {p["package_name"]: p["spdx_license"] for p in packages}
    assert licenses["libfoo"] == "Apache-2.0"
    assert licenses["libbar"] == "GPL-2.0-only"


def test_attributions_dir_override(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    _write_tsvs(src)
    _copy_spdx(src)
    attr_dir = tmp_path / "attr"
    out_csv = tmp_path / "out.csv"
    _run_script(
        [
            "--target-dir",
            str(src),
            "--output",
            str(out_csv),
            "--attributions-dir",
            str(attr_dir),
        ]
    )
    assert (attr_dir / "ATTRIBUTIONS-Apt.md").is_file()
    assert (attr_dir / "ATTRIBUTIONS-Python.md").is_file()
    assert not (src / "ATTRIBUTIONS-Apt.md").exists()


# ---- LicenseRef-* normalization (audit followup) -------------------------------


def test_normalize_license_refs_apache_variants():
    cases = [
        "LicenseRef-Apache",
        "LicenseRef-Apache-2",
        "LicenseRef-Apache-2.0",
        "LicenseRef-Apache-2.0-License",
        "LicenseRef-Apache-License-2.0",
        "LicenseRef-Apache-License-v2.0",
        "LicenseRef-Apache-License-Version-2.0",
        "LicenseRef-Apache-License--Version-2.0",
        "LicenseRef-Apache-Software-License",
    ]
    for value in cases:
        assert pr._normalize_license_refs(value) == "Apache-2.0", value


def test_normalize_license_refs_mit_variants():
    assert pr._normalize_license_refs("LicenseRef-Expat") == "MIT"
    assert pr._normalize_license_refs("LicenseRef-MIT") == "MIT"
    assert pr._normalize_license_refs("LicenseRef-MIT-License") == "MIT"


def test_normalize_license_refs_bsd_explicit_versions_only():
    assert pr._normalize_license_refs("LicenseRef-BSD-2") == "BSD-2-Clause"
    assert pr._normalize_license_refs("LicenseRef-BSD-3") == "BSD-3-Clause"
    assert pr._normalize_license_refs("LicenseRef-BSD-4") == "BSD-4-Clause"
    assert (
        pr._normalize_license_refs("LicenseRef-BSD-3-Clause-License") == "BSD-3-Clause"
    )


def test_normalize_license_refs_preserves_ambiguous_and_eulas():
    # Bare LicenseRef-BSD is ambiguous (could be 2/3/4-Clause); leave alone.
    assert pr._normalize_license_refs("LicenseRef-BSD") == "LicenseRef-BSD"
    # NVIDIA EULA strings are real LicenseRef- entries (not on SPDX list).
    nvidia = "LicenseRef-NVIDIA-Software-License-Agreement"
    assert pr._normalize_license_refs(nvidia) == nvidia
    # Hash-based syft identifiers (LicenseRef-<sha256>) are technically valid
    # SPDX 2.3 - they reference real license text in hasExtractedLicensingInfos.
    # We preserve them; UX cleanup is a P2 followup tracked in AUDIT.md.
    sha = "LicenseRef-5a5e7ca0e9f3f9679977e3a3e9ede45ad92885a3297ea78e766979f9866c5a16"
    assert pr._normalize_license_refs(sha) == sha


def test_normalize_license_refs_compound_expression():
    # Operator chunks pass through unchanged; each token is normalized.
    assert (
        pr._normalize_license_refs(
            "LicenseRef-Apache-2.0 AND LicenseRef-Expat OR BSD-3-Clause"
        )
        == "Apache-2.0 AND MIT OR BSD-3-Clause"
    )
    # Parenthesized compound: parens preserved, inner tokens rewritten.
    assert (
        pr._normalize_license_refs("(LicenseRef-Expat AND Apache-2.0)")
        == "(MIT AND Apache-2.0)"
    )


def test_normalize_license_refs_unknown_passthrough():
    # The literal string "UNKNOWN" must round-trip identically.
    assert pr._normalize_license_refs("UNKNOWN") == "UNKNOWN"
    assert pr._normalize_license_refs("") == ""
    # LicenseRef-UNKNOWN collapses to UNKNOWN so downstream counters don't
    # treat the row as resolved.
    assert pr._normalize_license_refs("LicenseRef-UNKNOWN") == "UNKNOWN"


def test_normalize_license_refs_does_not_touch_canonical_spdx():
    # Pure SPDX expressions must round-trip unchanged.
    for value in ["Apache-2.0", "MIT", "BSD-3-Clause", "GPL-2.0-only OR MIT"]:
        assert pr._normalize_license_refs(value) == value


def test_read_tsv_normalizes_license_refs(tmp_path: Path):
    tsv = tmp_path / "python.tsv"
    tsv.write_text(
        "responses\t0.26.0\tLicenseRef-Apache-2.0\n"
        "pillow\t8.3.1\tHPND\n"
        "weird\t1.0\tLicenseRef-Expat AND LicenseRef-BSD-3\n",
        encoding="utf-8",
    )
    rows = pr.read_tsv(tsv, "python")
    by_name = {r["package_name"]: r["spdx_license"] for r in rows}
    assert by_name["responses"] == "Apache-2.0"
    assert by_name["pillow"] == "HPND"
    assert by_name["weird"] == "MIT AND BSD-3-Clause"


def test_spdx_license_for_pkg_normalizes_license_refs():
    pkg = {"licenseConcluded": "LicenseRef-Apache-License-2.0"}
    assert pr._spdx_license_for_pkg(pkg) == "Apache-2.0"
    pkg = {"licenseConcluded": "NOASSERTION", "licenseDeclared": "LicenseRef-Expat"}
    assert pr._spdx_license_for_pkg(pkg) == "MIT"


def test_normalize_license_refs_collapses_noise_tokens_to_unknown():
    # libpython3.* pattern from syft tokenizing the PSF license preamble.
    expr = (
        "LicenseRef-By AND GPL-2.0-only AND LicenseRef-Permission "
        "AND LicenseRef-Redistribution AND LicenseRef-This"
    )
    assert pr._normalize_license_refs(expr) == "UNKNOWN"
    # Two noise tokens is enough to trigger the collapse.
    assert (
        pr._normalize_license_refs("LicenseRef-By AND LicenseRef-This") == "UNKNOWN"
    )


def test_normalize_license_refs_keeps_single_noise_token_compounds():
    # A single noise token alongside real licenses is most likely incidental
    # and should NOT collapse the expression.
    expr = "Apache-2.0 AND LicenseRef-By"
    out = pr._normalize_license_refs(expr)
    assert "Apache-2.0" in out
    assert out != "UNKNOWN"


def test_csv_pipeline_filters_first_party_dynamo_python(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "dpkg.tsv").write_text("openssl\t1.0\tApache-2.0\n", encoding="utf-8")
    (src / "python.tsv").write_text(
        "ai-dynamo\t1.1.0\tApache-2.0\n"
        "ai-dynamo-runtime\t1.1.0\tApache-2.0\n"
        "dynamo-something\t1.0\tApache-2.0\n"
        "requests\t2.32.0\tApache-2.0\n",
        encoding="utf-8",
    )
    out_csv = tmp_path / "out.csv"
    _run_script(["--target-dir", str(src), "--output", str(out_csv)])
    rows = list(csv.DictReader(out_csv.open()))
    names = {r["package_name"] for r in rows}
    assert "ai-dynamo" not in names
    assert "ai-dynamo-runtime" not in names
    assert "dynamo-something" not in names
    assert names == {"openssl", "requests"}
