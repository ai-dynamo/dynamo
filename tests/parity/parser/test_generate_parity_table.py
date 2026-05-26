# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
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
    assert "Dynamo Tool Calling Parser - Parity Table" in html
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


@pytest.mark.timeout(60)
def test_generate_reasoning_parity_table_leak_markers_are_parser_specific() -> None:
    html = _render_html_for("reasoning")

    assert "Dynamo Reasoning - Parity Table" in html
    assert re.search(
        r'<td class="cell na[^"]*"[^>]*><a href="fixtures/qwen3/REASONING\.batch\.yaml">n/a</a>'
        r'<div class="ttip"><div class="ttip-head">REASONING\.batch\.3\.b — qwen3',
        html,
    )
    assert re.search(
        r'<td class="cell leak[^"]*"[^>]*><a href="fixtures/gpt_oss/REASONING\.stream\.yaml">↯D</a>'
        r'<div class="ttip"><div class="ttip-head">REASONING\.stream\.3\.b — gpt_oss',
        html,
    )
    assert re.search(
        r'<td class="cell ok[^"]*"[^>]*><a href="fixtures/kimi_k25/REASONING\.batch\.yaml">=</a>'
        r'<div class="ttip"><div class="ttip-head">REASONING\.batch\.3\.b — kimi_k25',
        html,
    )
    assert re.search(
        r'<td class="cell research[^"]*"[^>]*><a href="fixtures/granite/REASONING\.batch\.yaml">V\?</a>'
        r'<div class="ttip"><div class="ttip-head">REASONING\.batch\.2\.a — granite',
        html,
    )
    assert re.search(
        r'<td class="cell ok[^"]*"[^>]*><a href="fixtures/minimax_append_think/REASONING\.batch\.yaml">=</a>'
        r'<div class="ttip"><div class="ttip-head">REASONING\.batch\.3\.a — minimax_append_think',
        html,
    )


def _render_html(*extra_args: str) -> str:
    return _render_html_for("parser", *extra_args)


def _render_html_for(table: str, *extra_args: str) -> str:
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR.relative_to(REPO_ROOT)),
            table,
            "--html",
            *extra_args,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout
