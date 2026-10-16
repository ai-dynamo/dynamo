# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Cargo binary SBOM emitter.

Run from the repo root with the compliance package on the path:

    PYTHONPATH=container python -m pytest container/compliance/tests/test_cargo_sbom.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compliance.cargo_sbom import (
    _licenses_field,
    build_sbom,
    packages_by_key,
    parse_tree,
)
from compliance.generators.rust import collect_components

# CPU-only unit tests; markers are required by .ai/pytest-guidelines.md
# (lifecycle / test-type / hardware categories).
pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _metadata(*packages: dict) -> dict:
    return {"packages": list(packages)}


def _pkg(name: str, version: str, **extra) -> dict:
    return {"name": name, "version": version, **extra}


class TestParseTree:
    def test_parses_plain_lines(self):
        assert parse_tree("tokio v1.52.4\nserde v1.0.0\n") == {
            ("tokio", "1.52.4"),
            ("serde", "1.0.0"),
        }

    def test_strips_dedupe_and_kind_markers(self):
        # cargo tree appends " (*)" to an elided subtree and " (proc-macro)"
        # to proc-macro crates; both must resolve to the same package.
        out = "async-trait v0.1.89 (proc-macro)\nasync-trait v0.1.89 (*)\n"
        assert parse_tree(out) == {("async-trait", "0.1.89")}

    def test_keeps_distinct_versions_of_one_crate(self):
        # The whole point of a per-binary SBOM: two versions of a crate are two
        # components, because dedupe downstream keys on (name, version).
        assert parse_tree("prost v0.14.3\nprost v0.13.5\n") == {
            ("prost", "0.14.3"),
            ("prost", "0.13.5"),
        }

    def test_ignores_blank_and_unparseable_lines(self):
        assert parse_tree("\ntokio v1.52.4\n   \n[build-dependencies]\n") == {
            ("tokio", "1.52.4")
        }

    def test_empty_output_is_an_error(self):
        # Silently emitting an empty SBOM would under-attribute a shipped binary.
        with pytest.raises(RuntimeError):
            parse_tree("")


class TestLicensesField:
    def test_spdx_expression_passes_through(self):
        assert _licenses_field(_pkg("a", "1", license="MIT OR Apache-2.0")) == [
            {"expression": "MIT OR Apache-2.0"}
        ]

    def test_legacy_slash_form_is_normalized(self):
        # Crates predating SPDX support in cargo write "MIT/Apache-2.0".
        assert _licenses_field(_pkg("a", "1", license="MIT/Apache-2.0")) == [
            {"expression": "MIT OR Apache-2.0"}
        ]

    def test_missing_license_yields_empty(self):
        # Routed to license_overrides.yaml downstream rather than guessed at.
        assert _licenses_field(_pkg("a", "1")) == []
        assert _licenses_field(_pkg("a", "1", license="  ")) == []


class TestBuildSbom:
    def test_emits_cargo_purls_and_licenses(self):
        meta = _metadata(
            _pkg("tokio", "1.52.4", license="MIT", repository="https://x/tokio")
        )
        sbom = build_sbom({("tokio", "1.52.4")}, packages_by_key(meta), "root-crate")
        (comp,) = sbom["components"]
        assert comp["purl"] == "pkg:cargo/tokio@1.52.4"
        assert comp["licenses"] == [{"expression": "MIT"}]
        assert comp["externalReferences"] == [{"type": "vcs", "url": "https://x/tokio"}]
        assert sbom["bomFormat"] == "CycloneDX"

    def test_crate_absent_from_metadata_still_attributed(self):
        # Dropping it would be a silent attribution hole; emit it license-less.
        sbom = build_sbom({("ghost", "0.1.0")}, {}, "root-crate")
        (comp,) = sbom["components"]
        assert comp["name"] == "ghost"
        assert "licenses" not in comp

    def test_components_are_sorted(self):
        sbom = build_sbom(
            {("b", "1.0.0"), ("a", "2.0.0"), ("a", "1.0.0")}, {}, "root-crate"
        )
        assert [(c["name"], c["version"]) for c in sbom["components"]] == [
            ("a", "1.0.0"),
            ("a", "2.0.0"),
            ("b", "1.0.0"),
        ]


class TestConsumedByRustGenerator:
    """The output only matters if generators/rust.py can read it."""

    def test_round_trips_through_collect_components(self, tmp_path: Path):
        meta = _metadata(
            _pkg("rcgen", "0.13.2", license="MIT OR Apache-2.0"),
            _pkg("dynamo-ext-proc", "1.5.0", license="Apache-2.0"),
        )
        sbom = build_sbom(
            {("rcgen", "0.13.2"), ("dynamo-ext-proc", "1.5.0")},
            packages_by_key(meta),
            "dynamo-ext-proc",
        )
        sbom_path = tmp_path / "sbom-rust-epp.cdx.json"
        sbom_path.write_text(json.dumps(sbom), encoding="utf-8")

        # No wheels anywhere: everything here must come from the extra SBOM.
        components = collect_components([tmp_path], extra_sboms=[sbom_path])
        by_name = {c.name: c for c in components}
        assert by_name["rcgen"].version == "0.13.2"
        assert by_name["rcgen"].spdx == "MIT OR Apache-2.0"
        # First-party crates are kept on purpose (see generators/rust.py), so
        # the binary itself appears in NOTICES.
        assert "dynamo-ext-proc" in by_name

    def test_missing_extra_sbom_is_fatal(self, tmp_path: Path):
        # A broken build wiring must fail loudly rather than quietly emit
        # NOTICES that omit the binary's crates.
        with pytest.raises(FileNotFoundError):
            collect_components([tmp_path], extra_sboms=[tmp_path / "absent.json"])
