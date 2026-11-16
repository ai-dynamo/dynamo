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
"""Tests for resolving relative links in translated pages to site URLs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import resolve_translation_links

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

PAGE = Path("translations/zh-CN/pages-dev/cli/getting-started/quickstart.mdx")
NAV = """navigation:
  - page: Intro
    path: pages/kubernetes/intro.mdx
  - page: Routing
    path: pages/router/routing.md
"""


def _resolve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> str:
    (tmp_path / "index.yml").write_text(NAV, encoding="utf-8")
    for rel in ("pages/kubernetes/intro.mdx", "pages/router/routing.md"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("base\n", encoding="utf-8")
    mirror = tmp_path / "translations/zh-CN/pages-dev/router/routing.md"
    mirror.parent.mkdir(parents=True)
    mirror.write_text("translated\n", encoding="utf-8")
    page = tmp_path / PAGE
    page.parent.mkdir(parents=True)
    page.write_text(text, encoding="utf-8")
    argv = [
        "resolve_translation_links.py",
        "--nav",
        str(tmp_path / "index.yml"),
        "--translations-root",
        str(tmp_path / "translations"),
        "--site-root",
        "/dynamo",
        "--version-slug",
        "dev",
        "--pages-dir",
        "pages-dev",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert resolve_translation_links.main() == 0
    return page.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            '<Card href="../../../../../pages/kubernetes/intro.mdx">',
            '<Card href="/dynamo/dev/intro">',
        ),
        (
            '<Card href="../../../../../pages/router/routing.md#kv">',
            '<Card href="/dynamo/zh-CN/dev/routing#kv">',
        ),
        (
            "[Intro](../../../../../pages/kubernetes/intro.mdx)",
            "[Intro](/dynamo/dev/intro)",
        ),
    ],
)
def test_rewrites_relative_page_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, expected: str
) -> None:
    assert _resolve(tmp_path, monkeypatch, text) == expected


@pytest.mark.parametrize(
    "text",
    [
        '<Card href="/dynamo/dev/intro">',
        '<Card href="https://example.com/a.md">',
        '`<Card href="../../../../../pages/kubernetes/intro.mdx">`',
        '<Card href="../../../../../pages/missing.mdx">',
    ],
)
def test_leaves_other_hrefs_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    assert _resolve(tmp_path, monkeypatch, text) == text
