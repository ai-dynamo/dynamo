# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR = REPO_ROOT / "tests/parity/generate_parity_table.py"


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value is not None:
                self.hrefs.append(value)


@pytest.mark.timeout(60)
def test_generate_parser_parity_table_html() -> None:
    html = _render_html()
    links = LinkCollector()
    links.feed(html)
    fixture_links = [href for href in links.hrefs if href.startswith("fixtures/")]
    fixture_families = {href.split("/")[1] for href in fixture_links}

    assert "<html" in html.lower()
    assert "<table" in html.lower()
    assert "Dynamo Tool Call Parser - Parity Table" in html
    assert "generate_parity_table.py parser --html" in html
    assert "PARSER.batch.*" in html
    assert "PARSER.stream.*" in html
    assert fixture_links
    assert len(fixture_families) > 10
    assert "deepseek_v3" in fixture_families
    assert "qwen3_coder" in fixture_families
    assert any("/PARSER.batch" in href for href in fixture_links)
    assert any("/PARSER.stream" in href for href in fixture_links)


@pytest.mark.timeout(60)
def test_generate_parser_parity_table_batch_mode_excludes_stream_links() -> None:
    html = _render_html("--mode", "batch")
    links = LinkCollector()
    links.feed(html)
    fixture_links = [href for href in links.hrefs if href.startswith("fixtures/")]

    assert fixture_links
    assert all("/PARSER.batch" in href for href in fixture_links)
    assert all("PARSER.stream." not in href for href in fixture_links)


def _render_html(*extra_args: str) -> str:
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR.relative_to(REPO_ROOT)),
            "parser",
            "--html",
            *extra_args,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout
