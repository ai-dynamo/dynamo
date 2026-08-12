# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for hosted-page SiteStyles injection."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import inject_site_styles as site_styles  # noqa: E402

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


def page(path: Path, body: str = "## Heading\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ntitle: Test\n---\n\n{body}", encoding="utf-8")
    return path


def test_discovers_all_published_page_roots(tmp_path: Path):
    expected = {
        page(tmp_path / "index.mdx"),
        page(tmp_path / "pages-dev" / "guide.md"),
        page(tmp_path / "pages-v1.0.0" / "reference.mdx"),
        page(tmp_path / "pages" / "legacy.md"),
        page(tmp_path / "digest" / "post.mdx"),
        page(tmp_path / "blogs" / "archive.md"),
        page(tmp_path / "translations" / "zh-CN" / "pages-dev" / "guide.md"),
    }
    page(tmp_path / "blogs" / "README.md")
    page(tmp_path / "components" / "README.md")

    assert set(site_styles.discover_pages(tmp_path)) == expected


def test_injects_after_frontmatter_and_is_idempotent(tmp_path: Path):
    target = page(tmp_path / "pages-dev" / "guide.md")

    assert site_styles.inject(target)
    rendered = target.read_text(encoding="utf-8")
    assert rendered.startswith("---\ntitle: Test\n---\n\nimport { SiteStyles }")
    assert rendered.count(site_styles.IMPORT) == 1
    assert rendered.count(site_styles.COMPONENT) == 1
    assert not site_styles.inject(target)


def test_injects_before_legacy_content_without_frontmatter(tmp_path: Path):
    target = tmp_path / "pages-v0.8.0" / "generated.md"
    target.parent.mkdir(parents=True)
    target.write_text("## Generated reference\n", encoding="utf-8")

    assert site_styles.inject(target)
    rendered = target.read_text(encoding="utf-8")
    assert rendered.startswith(f"{site_styles.IMPORT}\n\n{site_styles.COMPONENT}")
    assert rendered.endswith("## Generated reference\n")
