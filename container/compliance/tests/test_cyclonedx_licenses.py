# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CycloneDX licenses[] normalization and its generator callers.

Run from the repo root with the compliance package on the path:

    PYTHONPATH=container python -m pytest container/compliance/tests/test_cyclonedx_licenses.py
"""

from __future__ import annotations

import pytest
from compliance.generators import go as go_gen
from compliance.generators import rust as rust_gen
from compliance.generators.common import UNKNOWN, normalize_cyclonedx_licenses

# CPU-only unit tests; markers are required by .ai/pytest-guidelines.md
# (lifecycle / test-type / hardware categories).
pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

_MIT = {"license": {"id": "MIT"}}
_APACHE = {"license": {"id": "Apache-2.0"}}


@pytest.mark.parametrize(
    "licenses_field, expected",
    [
        (None, UNKNOWN),
        ([], UNKNOWN),
        ([_MIT], "MIT"),
        (
            [{"license": {"name": "Acme Custom License"}}],
            "LicenseRef-Acme-Custom-License",
        ),
        ([_MIT, _APACHE], "MIT AND Apache-2.0"),
        ([_APACHE, _MIT], "Apache-2.0 AND MIT"),
        (
            [{"expression": "MIT OR Apache-2.0"}, {"license": {"id": "BSD-3-Clause"}}],
            "(MIT OR Apache-2.0) AND BSD-3-Clause",
        ),
        # Spaces are replaced before the parenthesization check, so a custom
        # name in a multi-entry list is never wrapped.
        (
            [_MIT, {"license": {"name": "Acme Custom License"}}],
            "MIT AND LicenseRef-Acme-Custom-License",
        ),
    ],
)
def test_normalize_cyclonedx_licenses(licenses_field, expected):
    """Check license IDs, custom names, missing data, and combinations."""
    assert normalize_cyclonedx_licenses(licenses_field) == expected


def _go_entry(**extra) -> dict:
    """Build a Go SBOM entry with the fields required by the generator."""
    return {"name": "github.com/acme/mod", "version": "v1.0.0", **extra}


@pytest.mark.parametrize("top_level", [{}, {"licenses": []}])
def test_go_falls_back_to_evidence_licenses(top_level):
    """Use evidence licenses when top-level licenses are absent or empty."""
    entry = _go_entry(**top_level, evidence={"licenses": [_APACHE]})
    assert go_gen._component_from_sbom_entry(entry).spdx == "Apache-2.0"


def test_go_prefers_top_level_licenses_over_evidence():
    """Keep the top-level license when evidence declares a different license."""
    entry = _go_entry(licenses=[_MIT], evidence={"licenses": [_APACHE]})
    assert go_gen._component_from_sbom_entry(entry).spdx == "MIT"


def _rust_entry(**extra) -> dict:
    """Build a Rust SBOM entry with the fields required by the generator."""
    return {"name": "acme-crate", "version": "1.0.0", **extra}


def test_rust_override_applies_only_when_sbom_is_unknown(monkeypatch):
    """Look up a Rust license override when the SBOM has no license."""
    calls: list[tuple] = []

    def fake_lookup(ecosystem, name, version=""):
        """Record the requested package and return its license override."""
        calls.append((ecosystem, name, version))
        return "BSD-2-Clause"

    monkeypatch.setattr(rust_gen.license_overrides, "lookup", fake_lookup)
    comp = rust_gen._component_from_sbom_entry(_rust_entry())
    assert comp.spdx == "BSD-2-Clause"
    assert calls == [("rust", "acme-crate", "1.0.0")]


def test_rust_sbom_license_wins_over_override(monkeypatch):
    """Preserve a known SBOM license even when an override would differ."""
    monkeypatch.setattr(
        rust_gen.license_overrides, "lookup", lambda *a, **k: "BSD-2-Clause"
    )
    comp = rust_gen._component_from_sbom_entry(_rust_entry(licenses=[_MIT]))
    assert comp.spdx == "MIT"
