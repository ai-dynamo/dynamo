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
    assert re.search(
        r'data-col-toggle="model"[^>]+data-default-visible="true"[^>]+aria-pressed="true"',
        html,
    )
    assert '<body class="view-overview parser-dynamo">' in html
    assert 'value="overview" data-view-toggle checked> Overview' in html
    assert 'value="details" data-view-toggle> Details' in html
    assert 'value="dynamo" data-parser-toggle checked> Dynamo' in html
    assert 'value="vllm" data-parser-toggle> vLLM' in html
    assert 'value="sglang" data-parser-toggle> SGLang' in html
    assert (
        '<label class="checkbox-option parity-option"><input type="checkbox" data-parity-toggle> Parity</label>'
        in html
    )
    assert ".view-overview .parity-option { display: none; }" in html
    assert "td.cell { text-align: center; min-width: 24px; font-weight: 400; }" in html
    assert 'td.cell[data-status-dynamo="na"]::before' in html
    assert "data-marker-parity-vllm" in html
    assert "content: attr(data-marker-parity-vllm)" in html
    assert 'data-marker-dynamo="' in html
    assert 'data-marker-dynamo="="' not in html
    assert 'data-marker-vllm="="' not in html
    assert 'data-marker-sglang="="' not in html
    assert "color: #aeb6bf;" in html
    assert "background: #e4e8ec;" in html
    assert 'data-marker-parity-dynamo="D"' not in html
    assert 'data-marker-parity-vllm="V"' not in html
    assert 'data-marker-parity-sglang="S"' not in html
    assert 'data-marker-parity-dynamo="VS"' in html
    assert 'data-marker-parity-vllm="DS"' in html
    assert 'data-marker-parity-sglang="DV"' in html
    assert 'data-marker-parity-dynamo="↯="' in html
    assert 'data-marker-parity-dynamo="↯VS"' in html
    assert 'data-marker-dynamo="D!"' not in html
    assert ".view-details.parity-mode .parity-explainer { display: block; }" in html
    assert "<strong>Parity:</strong>" in html
    assert "color: #8b949e;" in html
    assert '<span style="color:#8b949e">·</span> Dynamo-only fixture' in html
    # `·` came only from the now-hidden nemotron_deci / nemotron_nano rows.
    assert 'data-marker-dynamo="·"' not in html
    assert '<span style="color:#555">D</span> Dynamo-only fixture' not in html
    assert "green = selected implementation output is clean" in html
    assert "red = selected implementation leaks parser markup" in html
    assert "Why not applicable" in html
    assert "Why n/a" not in html
    assert "get('view')" in html
    assert "get('parser')" in html
    assert "get('parity')" in html
    assert "url.searchParams.set('view', view)" in html
    assert "url.searchParams.set('parser', parser)" in html
    assert "url.searchParams.set('parity', '1')" in html
    assert "Tool calling family" in html
    assert "generate_parity_table.py toolcalling --html" in html
    assert "TOOLCALLING.batch.*" in html
    assert "TOOLCALLING.stream.*" in html
    assert 'id="case-descriptions-batch"' in html
    assert 'id="case-descriptions-stream"' in html
    assert "TOOLCALLING.batch.1</td><td>Single tool call" in html
    assert "TOOLCALLING.stream.1.a</td><td>Single complete tool-call payload" in html
    assert re.search(
        r'data-status-dynamo="ok" data-status-vllm="problem" data-status-sglang="na" '
        r'data-marker-dynamo="" data-marker-vllm="↯" data-marker-sglang="n/a" '
        r'data-marker-parity-dynamo="" data-marker-parity-vllm="↯" '
        r'data-marker-parity-sglang="n/a"><a href="fixtures/deepseek_v4/TOOLCALLING\.batch\.4\.yaml">V</a>'
        r'<div class="ttip"><div class="ttip-head">TOOLCALLING\.batch\.4\.a — deepseek_v4',
        html,
    )
    assert 'data-marker-vllm="!"' in html
    assert 'data-marker-parity-vllm="!"' in html
    assert re.search(
        r'data-status-dynamo="ok" data-status-vllm="ok" data-status-sglang="problem" '
        r'data-marker-dynamo="" data-marker-vllm="" data-marker-sglang="↯" '
        r'data-marker-parity-dynamo="S" data-marker-parity-vllm="S" '
        r'data-marker-parity-sglang="↯DV"><a href="fixtures/harmony/TOOLCALLING\.batch\.yaml">S</a>'
        r'<div class="ttip"><div class="ttip-head">TOOLCALLING\.batch\.1 — harmony',
        html,
    )
    assert re.search(
        r'data-status-dynamo="ok" data-status-vllm="problem" data-status-sglang="ok" '
        r'data-marker-dynamo="" data-marker-vllm="↯" data-marker-sglang="" '
        r'data-marker-parity-dynamo="VS" data-marker-parity-vllm="↯DS" '
        r'data-marker-parity-sglang="DV"><a href="fixtures/llama3_json/TOOLCALLING\.batch\.4\.yaml">VS</a>'
        r'<div class="ttip"><div class="ttip-head">TOOLCALLING\.batch\.4\.a — llama3_json',
        html,
    )
    assert fixture_links
    assert len(fixture_families) > 10
    assert "deepseek_v3" in fixture_families
    assert "qwen3_coder" in fixture_families
    assert any("/TOOLCALLING.batch" in href for href in fixture_links)
    assert any("/TOOLCALLING.stream" in href for href in fixture_links)


@pytest.mark.timeout(60)
def test_generate_parser_parity_table_batch_mode_excludes_stream_links() -> None:
    html = _render_html("--mode", "batch")
    links = LinkCollector()
    links.feed(html)
    fixture_links = [href for href in links.hrefs if href.startswith("fixtures/")]

    assert fixture_links
    assert all("/TOOLCALLING.batch" in href for href in fixture_links)
    assert all("TOOLCALLING.stream." not in href for href in fixture_links)


@pytest.mark.timeout(60)
def test_generate_combined_parity_table_html() -> None:
    html = _render_html_for("all")
    links = LinkCollector()
    links.feed(html)

    assert "Dynamo Parser Parity Table" in html
    assert "generate_parity_table.py all --html" in html
    assert 'data-tab-target="tab-toolcalling-batch">TC batch</button>' in html
    assert 'title="Tool Calling batch"' in html
    assert 'aria-label="Tool Calling stream"' in html
    assert 'data-tab-target="tab-reasoning-batch">Reasoning batch</button>' in html
    assert 'data-tab-target="tab-reasoning-stream">Reasoning stream</button>' in html
    assert "params.get('tab')" in html
    assert "validTargets.has(requested)" in html
    assert "document.getElementById(requested)" not in html
    assert "url.searchParams.set('tab', id)" in html
    assert "TOOLCALLING.batch." in html
    assert "TOOLCALLING.stream." in html
    assert 'id="case-descriptions-toolcalling-batch"' in html
    assert 'id="case-descriptions-toolcalling-stream"' in html
    assert "TOOLCALLING.batch.1</td><td>Single tool call" in html
    assert "TOOLCALLING.stream.1.a</td><td>Single complete tool-call payload" in html
    assert "REASONING.batch." in html
    assert "REASONING.stream." in html
    assert "Why not applicable" in html
    assert "Why n/a" not in html
    assert "toolcalling/fixtures/" in "".join(links.hrefs)
    assert "reasoning/fixtures/" in "".join(links.hrefs)
    assert "../../lib/parsers/TOOLCALLING_CASES.md" in links.hrefs
    assert "../../lib/parsers/REASONING_CASES.md" in links.hrefs
    assert "../../pyproject.toml" in links.hrefs


@pytest.mark.timeout(60)
def test_generate_reasoning_parity_table_leak_markers_are_parser_specific() -> None:
    html = _render_html_for("reasoning")

    assert "Dynamo Reasoning Parser - Parity Table" in html
    assert "Reasoning family" in html
    assert re.search(
        r'data-col-toggle="model"[^>]+data-default-visible="true"[^>]+aria-pressed="true"',
        html,
    )
    assert re.search(
        r'<td class="cell na[^"]*"[^>]*><a href="fixtures/qwen3/REASONING\.batch\.yaml">n/a</a>'
        r'<div class="ttip"><div class="ttip-head">REASONING\.batch\.3\.b — qwen3',
        html,
    )
    split_end_cell = _cell_for(html, "REASONING.stream.3.b — gpt_oss")
    assert re.search(
        r'<td class="cell documented[^"]*"[^>]*><a href="fixtures/gpt_oss/REASONING\.stream\.yaml">S</a>',
        split_end_cell,
    )
    handoff_cell = _cell_for(html, "REASONING.stream.4.b — gpt_oss")
    assert re.search(
        r'<td class="cell documented[^"]*"[^>]*><a href="fixtures/gpt_oss/REASONING\.stream\.yaml">V</a>',
        handoff_cell,
    )
    assert "↯ Dynamo leaks" not in handoff_cell
    assert "Dynamo: Unlike vLLM streaming reasoning" in handoff_cell
    assert "Divergent reasons" not in handoff_cell
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
    return _render_html_for("toolcalling", *extra_args)


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


def _cell_for(html: str, heading: str) -> str:
    idx = html.find(heading)
    assert idx != -1
    start = html.rfind("<td", 0, idx)
    end = html.find("</td>", idx)
    assert start != -1 and end != -1
    return html[start:end]
