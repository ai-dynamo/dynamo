# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for inject_source_binaries.py."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inject_source_binaries import (
    _deterministic_serial,
    build_component,
    inject,
    load_context_versions,
)


class TestVcsUrlBranches:
    """Tests for VCS URL branching based on purl_type."""

    def test_inject_vcs_url_github(self) -> None:
        """GitHub purl_type uses github.com."""
        entry = {
            "name": "test-pkg",
            "repo": "org/repo",
            "purl_type": "github",
            "license": "MIT",
        }
        comp = build_component("test_key", entry, "1.0.0")
        ext_refs = comp["externalReferences"]
        vcs_ref = next(r for r in ext_refs if r["type"] == "vcs")
        assert vcs_ref["url"] == "https://github.com/org/repo"

    def test_inject_vcs_url_gitlab(self) -> None:
        """GitLab purl_type uses gitlab.com."""
        entry = {
            "name": "test-pkg",
            "repo": "org/repo",
            "purl_type": "gitlab",
            "license": "MIT",
        }
        comp = build_component("test_key", entry, "1.0.0")
        ext_refs = comp["externalReferences"]
        vcs_ref = next(r for r in ext_refs if r["type"] == "vcs")
        assert vcs_ref["url"] == "https://gitlab.com/org/repo"

    def test_inject_vcs_url_bitbucket(self) -> None:
        """Bitbucket purl_type uses bitbucket.org."""
        entry = {
            "name": "test-pkg",
            "repo": "org/repo",
            "purl_type": "bitbucket",
            "license": "MIT",
        }
        comp = build_component("test_key", entry, "1.0.0")
        ext_refs = comp["externalReferences"]
        vcs_ref = next(r for r in ext_refs if r["type"] == "vcs")
        assert vcs_ref["url"] == "https://bitbucket.org/org/repo"

    def test_inject_vcs_url_unknown_raises(self) -> None:
        """Unknown purl_type raises ValueError."""
        entry = {
            "name": "test-pkg",
            "repo": "org/repo",
            "purl_type": "unknown",
            "license": "MIT",
        }
        with pytest.raises(ValueError, match="unsupported purl_type: unknown"):
            build_component("test_key", entry, "1.0.0")


class TestLoadContextVersions:
    """Tests for load_context_versions function."""

    def test_loads_all_scalars(self) -> None:
        """Extracts all str/int/float values, not just _version/_ref suffixed."""
        context = {
            "dynamo": {
                "nixl_ref": "v1.0.0",
                "nixl_version": "1.0.0",
                "some_count": 42,
                "ratio": 3.14,
                "nested": {"ignored": "value"},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(context, f)
            f.flush()
            versions = load_context_versions(Path(f.name), "dynamo")

        assert versions["nixl_ref"] == "v1.0.0"
        assert versions["nixl_version"] == "1.0.0"
        assert versions["some_count"] == "42"
        assert versions["ratio"] == "3.14"
        assert "nested" not in versions


class TestInjectFailsOnMissing:
    """Tests for inject() RuntimeError on missing versions."""

    def test_inject_load_context_fails_on_missing(self) -> None:
        """Entry in binary_refs whose applies_to includes framework but version absent -> RuntimeError."""
        bom = {"components": []}
        binary_refs = {
            "missing_key": {
                "name": "missing-pkg",
                "repo": "org/missing",
                "purl_type": "github",
                "applies_to": ["dynamo"],
            }
        }
        context = {"dynamo": {"other_key": "v1.0.0"}}  # missing_key not present

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as ctx_f:
            yaml.dump(context, ctx_f)
            ctx_path = Path(ctx_f.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as refs_f:
            yaml.dump(binary_refs, refs_f)
            refs_path = Path(refs_f.name)

        with pytest.raises(
            RuntimeError, match="context.yaml.*has no entry for 'missing_key'"
        ):
            inject(bom, ctx_path, refs_path, "dynamo")


class TestDeterministicSerial:
    """Tests for deterministic UUID generation."""

    def test_deterministic_serial_reproducible(self) -> None:
        """Same inputs produce same serial number."""
        bom = {
            "components": [
                {"purl": "pkg:cargo/a@1.0"},
                {"purl": "pkg:cargo/b@2.0"},
            ]
        }
        serial1 = _deterministic_serial("dynamo", "abc123", bom)
        serial2 = _deterministic_serial("dynamo", "abc123", bom)
        assert serial1 == serial2
        assert serial1.startswith("urn:uuid:")

    def test_deterministic_serial_varies_with_inputs(self) -> None:
        """Different inputs produce different serial numbers."""
        bom = {"components": [{"purl": "pkg:cargo/a@1.0"}]}
        serial1 = _deterministic_serial("dynamo", "abc123", bom)
        serial2 = _deterministic_serial("vllm", "abc123", bom)
        serial3 = _deterministic_serial("dynamo", "def456", bom)
        assert serial1 != serial2
        assert serial1 != serial3

    def test_deterministic_serial_sorted_purls(self) -> None:
        """PURL order doesn't affect serial (sorted internally)."""
        bom1 = {
            "components": [
                {"purl": "pkg:cargo/a@1.0"},
                {"purl": "pkg:cargo/b@2.0"},
            ]
        }
        bom2 = {
            "components": [
                {"purl": "pkg:cargo/b@2.0"},
                {"purl": "pkg:cargo/a@1.0"},
            ]
        }
        assert _deterministic_serial("dynamo", "abc", bom1) == _deterministic_serial(
            "dynamo", "abc", bom2
        )
