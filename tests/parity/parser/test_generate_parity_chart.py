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
GENERATOR = REPO_ROOT / "tests/parity/parser/generate_parity_chart.py"


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


def test_generate_parser_parity_table_html() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR.relative_to(REPO_ROOT)),
            "--html",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    html = result.stdout
    links = LinkCollector()
    links.feed(html)
    fixture_links = [href for href in links.hrefs if href.startswith("fixtures/")]
    fixture_families = {href.split("/")[1] for href in fixture_links}

    assert "<html" in html.lower()
    assert "<table" in html.lower()
    assert "Dynamo Parser Parity Table" in html
    assert "generate_parity_chart.py --html" in html
    assert fixture_links
    assert len(fixture_families) > 10
    assert "deepseek_v3" in fixture_families
    assert "qwen3_coder" in fixture_families
    assert all("/PARSER.batch" in href for href in fixture_links)
    assert all("PARSER.stream." not in href for href in fixture_links)
