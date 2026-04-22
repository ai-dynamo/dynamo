# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for render_attributions.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render_attributions import (
    ECOSYSTEMS,
    _code_fence,
    _extract_copyright,
    _license_spdx,
    _render_section,
    _select,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestLicenseSpdx:
    """Tests for _license_spdx function."""

    def test_license_spdx_single_id(self) -> None:
        """Single license id passes through unchanged."""
        comp = {"licenses": [{"license": {"id": "MIT"}}]}
        assert _license_spdx(comp) == "MIT"

    def test_license_spdx_single_name(self) -> None:
        """Single license name passes through when no id."""
        comp = {"licenses": [{"license": {"name": "Custom License"}}]}
        assert _license_spdx(comp) == "Custom License"

    def test_license_spdx_array_uses_or(self) -> None:
        """Multiple licenses[] entries are joined with OR, not AND."""
        comp = {
            "licenses": [
                {"license": {"id": "MIT"}},
                {"license": {"id": "Apache-2.0"}},
            ]
        }
        result = _license_spdx(comp)
        assert result == "MIT OR Apache-2.0"
        assert " AND " not in result

    def test_license_spdx_expression_passthrough(self) -> None:
        """A single expression is returned verbatim without wrapping."""
        comp = {"licenses": [{"expression": "MIT OR Apache-2.0"}]}
        assert _license_spdx(comp) == "MIT OR Apache-2.0"

    def test_license_spdx_mixed_id_and_expression(self) -> None:
        """Mixed id and expression entries joined with OR."""
        comp = {
            "licenses": [
                {"license": {"id": "GPL-3.0-only"}},
                {"expression": "MIT OR Apache-2.0"},
            ]
        }
        result = _license_spdx(comp)
        assert "GPL-3.0-only" in result
        assert "MIT OR Apache-2.0" in result
        assert " OR " in result

    def test_license_spdx_empty(self) -> None:
        """Empty licenses array returns NOASSERTION."""
        assert _license_spdx({}) == "NOASSERTION"
        assert _license_spdx({"licenses": []}) == "NOASSERTION"
        assert _license_spdx({"licenses": None}) == "NOASSERTION"


class TestCodeFence:
    """Tests for _code_fence function."""

    def test_code_fence_no_backticks(self) -> None:
        """Text without backticks gets standard triple backtick fence."""
        assert _code_fence("plain text") == "```"

    def test_code_fence_escapes_backticks(self) -> None:
        """License text with ``` produces a longer fence."""
        text = "Example:\n```\ncode\n```\nMore text"
        fence = _code_fence(text)
        assert len(fence) >= 4
        assert fence == "````"

    def test_code_fence_longer_run(self) -> None:
        """Fence is one longer than the longest backtick run."""
        text = "Some `code` and ````longer```` fence"
        fence = _code_fence(text)
        assert fence == "`````"

    def test_code_fence_single_backtick(self) -> None:
        """Single backticks don't affect the fence."""
        text = "Use `code` inline"
        assert _code_fence(text) == "```"


class TestRenderSection:
    """Tests for _render_section function."""

    def test_render_section_uses_dynamic_fence(self) -> None:
        """Rendered section uses appropriate fence for license text."""
        comp = {
            "name": "test-pkg",
            "version": "1.0.0",
            "purl": "pkg:cargo/test-pkg@1.0.0",
            "licenses": [
                {
                    "license": {
                        "id": "MIT",
                        "text": {
                            "contentType": "text/plain",
                            "content": "Example:\n```\ncode\n```",
                        },
                    }
                }
            ],
        }
        section = _render_section(comp)
        # Should use ```` instead of ``` since text contains ```
        assert "````" in section
        assert section.count("````") == 2  # opening and closing


class TestSelectDeduplication:
    """Tests for _select function."""

    def test_select_deduplicates_by_purl(self) -> None:
        """Duplicate PURL entries collapse to one (last-wins)."""
        components = [
            {
                "purl": "pkg:cargo/foo@1.0",
                "name": "foo",
                "version": "1.0",
                "note": "first",
            },
            {
                "purl": "pkg:cargo/foo@1.0",
                "name": "foo",
                "version": "1.0",
                "note": "second",
            },
            {"purl": "pkg:cargo/bar@2.0", "name": "bar", "version": "2.0"},
        ]
        selected = _select(components, "pkg:cargo/")
        assert len(selected) == 2
        # Find the foo component
        foo = next(c for c in selected if c["name"] == "foo")
        assert foo["note"] == "second"  # last-wins

    def test_select_filters_by_prefix(self) -> None:
        """Only components matching prefix are selected."""
        components = [
            {"purl": "pkg:cargo/foo@1.0", "name": "foo", "version": "1.0"},
            {"purl": "pkg:pypi/bar@2.0", "name": "bar", "version": "2.0"},
        ]
        cargo = _select(components, "pkg:cargo/")
        pypi = _select(components, "pkg:pypi/")
        assert len(cargo) == 1
        assert cargo[0]["name"] == "foo"
        assert len(pypi) == 1
        assert pypi[0]["name"] == "bar"


class TestEcosystemDispatch:
    """Tests for ecosystem routing."""

    def test_ecosystem_dispatch_cargo(self) -> None:
        """pkg:cargo/ routes to Rust ecosystem."""
        eco = next(
            (e for p, e, _ in ECOSYSTEMS if "pkg:cargo/foo@1.0".startswith(p)), None
        )
        assert eco == "Rust"

    def test_ecosystem_dispatch_unknown(self) -> None:
        """Unknown PURL prefix doesn't match any ecosystem."""
        purl = "pkg:conan/foo@1.0"
        matched = [e for p, e, _ in ECOSYSTEMS if purl.startswith(p)]
        assert matched == []


class TestExtractCopyright:
    """Tests for _extract_copyright function."""

    def test_extract_copyright_from_license_text(self) -> None:
        """Copyright extracted from license text."""
        comp = {
            "licenses": [
                {
                    "license": {
                        "id": "MIT",
                        "text": {
                            "content": "MIT License\n\nCopyright (c) 2024 NVIDIA\n\nPermission..."
                        },
                    }
                }
            ]
        }
        copyright_line = _extract_copyright(comp)
        assert "2024" in copyright_line
        assert "NVIDIA" in copyright_line

    def test_extract_copyright_from_field(self) -> None:
        """Falls back to component.copyright field."""
        comp = {"copyright": "Copyright 2023 Example Corp", "licenses": []}
        assert _extract_copyright(comp) == "Copyright 2023 Example Corp"

    def test_extract_copyright_empty(self) -> None:
        """Returns empty string when no copyright found."""
        comp = {"licenses": [{"license": {"id": "MIT"}}]}
        assert _extract_copyright(comp) == ""


class TestIntegration:
    """Integration tests using fixture files."""

    def test_multi_license_fixture(self) -> None:
        """Test multi_license.cdx.json fixture processing."""
        with open(FIXTURES / "multi_license.cdx.json") as f:
            bom = json.load(f)
        components = bom["components"]

        # Single ID
        single = next(c for c in components if c["name"] == "single-license")
        assert _license_spdx(single) == "MIT"

        # Multiple IDs -> OR
        multi = next(c for c in components if c["name"] == "multi-license")
        result = _license_spdx(multi)
        assert result == "MIT OR Apache-2.0"

        # Expression passthrough
        expr = next(c for c in components if c["name"] == "expr-license")
        assert _license_spdx(expr) == "MIT OR Apache-2.0"

        # Mixed
        mixed = next(c for c in components if c["name"] == "mixed-license")
        result = _license_spdx(mixed)
        assert "GPL-3.0-only" in result
        assert "MIT OR Apache-2.0" in result

    def test_backtick_fixture(self) -> None:
        """Test backtick_license.cdx.json renders without corruption."""
        with open(FIXTURES / "backtick_license.cdx.json") as f:
            bom = json.load(f)
        comp = bom["components"][0]
        section = _render_section(comp)
        # Should escape the backticks in license text
        assert "`````" in section  # 5 backticks because text has ````

    def test_duplicate_purl_fixture(self) -> None:
        """Test duplicate_purl.cdx.json deduplication."""
        with open(FIXTURES / "duplicate_purl.cdx.json") as f:
            bom = json.load(f)
        components = bom["components"]
        selected = _select(components, "pkg:cargo/")
        # 3 components, 2 have same PURL -> 2 unique
        assert len(selected) == 2
        purls = [c["purl"] for c in selected]
        assert len(set(purls)) == 2
