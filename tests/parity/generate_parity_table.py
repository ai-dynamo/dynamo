#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate parity tables for each parser stage from one common entrypoint.

Examples:
    python3 tests/parity/generate_parity_table.py all --html > tests/parity/PARITY.html
    python3 tests/parity/generate_parity_table.py toolcalling --html > tests/parity/toolcalling/PARITY.html
    python3 tests/parity/generate_parity_table.py toolcalling --mode stream > tests/parity/toolcalling/PARITY.stream.md
    python3 tests/parity/generate_parity_table.py reasoning --html > tests/parity/reasoning/PARITY.html
"""

from __future__ import annotations

import argparse
import datetime
import html as html_lib
import sys
import zoneinfo
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.parity.reasoning import table as reasoning_table  # noqa: E402
from tests.parity.toolcalling import table as toolcalling_table  # noqa: E402


def _rewrite_panel_paths(panel: dict[str, Any], stage_dir: str) -> dict[str, Any]:
    """Adjust links from stage-local PARITY.html paths to tests/parity/PARITY.html."""
    rewritten = dict(panel)

    def rewrite(text: str) -> str:
        return (
            text.replace('href="fixtures/', f'href="{stage_dir}/fixtures/')
            .replace('href="../../../lib/', 'href="../../lib/')
            .replace('href="../../../pyproject.toml"', 'href="../../pyproject.toml"')
        )

    rewritten["group_headers"] = rewrite(str(rewritten["group_headers"]))
    rewritten["sub_headers"] = rewrite(str(rewritten["sub_headers"]))
    rewritten["body_rows"] = [rewrite(str(row)) for row in rewritten["body_rows"]]
    return rewritten


def _tab_button(panel: dict[str, Any]) -> str:
    active = " active" if panel["active"] else ""
    selected = "true" if panel["active"] else "false"
    panel_id = html_lib.escape(str(panel["id"]))
    label = html_lib.escape(str(panel["label"]))
    title = html_lib.escape(str(panel.get("tab_title", panel["label"])))
    return (
        f'<button class="tab-button{active}" id="{panel_id}-button" '
        f'type="button" role="tab" aria-selected="{selected}" '
        f'aria-label="{title}" title="{title}" '
        f'data-tab-target="{panel_id}">{label}</button>'
    )


def _combined_toolcalling_panels() -> list[dict[str, Any]]:
    panels = []
    for mode in ("batch", "stream"):
        _mode, panel, _has_cases = toolcalling_table._load_html_panel(mode)
        panel = _rewrite_panel_paths(panel, "toolcalling")
        panel.update(
            {
                "id": f"tab-toolcalling-{mode}",
                "label": f"TC {mode}",
                "tab_title": f"Tool Calling {mode}",
                "active": False,
                "case_docs_href": "../../lib/parsers/TOOLCALLING_CASES.md",
                "case_docs_label": "lib/parsers/TOOLCALLING_CASES.md",
                "case_prefix": f"TOOLCALLING.{mode}.",
                "case_section_id": f"toolcalling-{mode}",
            }
        )
        panels.append(panel)
    return panels


def _combined_reasoning_panels() -> list[dict[str, Any]]:
    rows, columns, refs = reasoning_table._load()
    no_vllm, no_sglang = reasoning_table._derive_no_peer_sets(rows)
    panels = []
    for mode in ("batch", "stream"):
        mode_columns = reasoning_table._columns_for_mode(columns, mode)
        panel = reasoning_table._html_panel(
            rows,
            mode_columns,
            refs,
            no_vllm,
            no_sglang,
            mode=mode,
            active=False,
        )
        panel = _rewrite_panel_paths(panel, "reasoning")
        panel.update(
            {
                "id": f"tab-reasoning-{mode}",
                "label": f"Reasoning {mode}",
                "active": False,
                "case_docs_href": "../../lib/parsers/REASONING_CASES.md",
                "case_docs_label": "lib/parsers/REASONING_CASES.md",
                "case_prefix": "REASONING.",
                "case_section_id": f"reasoning-{mode}",
                "legend_html": reasoning_table._legend_html(rows, mode_columns),
            }
        )
        panels.append(panel)
    return panels


def render_combined_html() -> str:
    panels = [*_combined_toolcalling_panels(), *_combined_reasoning_panels()]
    panels[0]["active"] = True

    now = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
    stamp = now.strftime("%Y-%m-%d %H:%M %Z")
    sha = toolcalling_table._commit_sha()

    return (
        toolcalling_table._make_jinja_env()
        .get_template("parity_table.html.j2")
        .render(
            title="Dynamo Parser Parity Table",
            stamp=stamp,
            sha=sha,
            short_sha=sha[:12] if sha else "",
            command="python3 tests/parity/generate_parity_table.py all --html",
            output="tests/parity/PARITY.html",
            tabs=[_tab_button(panel) for panel in panels],
            panels=panels,
            peer_versions=toolcalling_table._peer_version_items(
                toolcalling_table._peer_versions()
            ),
            peer_versions_href="../../pyproject.toml",
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Dynamo parity tables.",
    )
    parser.add_argument(
        "stage",
        choices=("all", "toolcalling", "reasoning"),
        help="Parity stage to render.",
    )
    args, rest = parser.parse_known_args(argv)

    if args.stage == "all":
        stage_parser = argparse.ArgumentParser(
            description="Generate the combined Dynamo parser parity HTML page.",
        )
        stage_parser.add_argument(
            "--html",
            action="store_true",
            help="Emit the combined HTML page.",
        )
        stage_args = stage_parser.parse_args(rest)
        if not stage_args.html:
            parser.error("stage 'all' currently supports --html only")
        print(render_combined_html())
        return

    stage_table = toolcalling_table if args.stage == "toolcalling" else reasoning_table
    stage_table.main(rest)


if __name__ == "__main__":
    main()
