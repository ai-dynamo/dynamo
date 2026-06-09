# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for full-license-text inclusion in NOTICES.

Run from the repo root with the compliance package on the path:

    PYTHONPATH=container python -m pytest container/compliance/tests/test_license_text.py
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from compliance.generators.common import (
    Component,
    render_notices,
    spdx_license_text,
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
