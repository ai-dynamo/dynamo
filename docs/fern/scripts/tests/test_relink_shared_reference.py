# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for relinking release snapshots to the shared Reference pages."""

from __future__ import annotations

from pathlib import Path

import pytest
from relink_shared_reference import relink

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

SUBDIR = Path("reference/general")


@pytest.fixture
def tree(tmp_path: Path) -> tuple[Path, Path]:
    snapshot = tmp_path / "pages-v1.0.0"
    shared = tmp_path / "pages-dev"
    # kubernetes/ sits outside --subdir, so the [Other] case below fails if the
    # relink stops being scoped to reference/general/.
    for rel in (
        "reference/general/compatibility.mdx",
        "reference/general/support.md",
        "kubernetes/compatibility.mdx",
    ):
        (shared / rel).parent.mkdir(parents=True, exist_ok=True)
        (shared / rel).write_text("shared\n", encoding="utf-8")
    (snapshot / "reference/general").mkdir(parents=True)
    (snapshot / "reference/general/tag-only.mdx").write_text("kept\n", encoding="utf-8")
    (snapshot / "cli").mkdir()
    return snapshot, shared


def _relink(tree: tuple[Path, Path], text: str, page: str = "cli/page.mdx") -> str:
    snapshot, shared = tree
    return relink(text, snapshot / page, snapshot, shared, SUBDIR)


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "[Matrix](../reference/general/compatibility.mdx#vllm)",
            "[Matrix](../../pages-dev/reference/general/compatibility.mdx#vllm)",
        ),
        (
            '<Card href="../reference/general/support.md">',
            '<Card href="../../pages-dev/reference/general/support.md">',
        ),
        (
            "[support]: ../reference/general/support.md",
            "[support]: ../../pages-dev/reference/general/support.md",
        ),
    ],
)
def test_dropped_shared_page_points_at_pages_dev(
    tree: tuple[Path, Path], text: str, expected: str
) -> None:
    assert _relink(tree, text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "[Kept](../reference/general/tag-only.mdx)",
        "[Gone](../reference/general/removed-on-main.mdx)",
        "[Other](../kubernetes/compatibility.mdx)",
        "[Abs](/dynamo/reference/general/compatibility.mdx)",
        "[Web](https://example.com/reference/general/compatibility.mdx)",
    ],
)
def test_other_links_are_unchanged(tree: tuple[Path, Path], text: str) -> None:
    assert _relink(tree, text) == text


def test_sibling_inside_shared_dir(tree: tuple[Path, Path]) -> None:
    text = "[Matrix](compatibility.mdx)"
    page = "reference/general/tag-only.mdx"
    assert _relink(tree, text, page) == (
        "[Matrix](../../../pages-dev/reference/general/compatibility.mdx)"
    )
