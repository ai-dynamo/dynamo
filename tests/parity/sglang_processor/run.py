#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare Dynamo's SGLang chat processor with native SGLang on the Unified corpus.

Both sides run on identical frozen jobs built by frontend-crates ``build_job``.
The native side is the unmodified frontend-crates ``capture_sglang_unified.main()``,
called the way ``capture_peer_unified_versions.worker`` calls it in stream mode.
The Dynamo side is ``capture_dynamo_sglang_unified.main()``. Assembly uses
frontend-crates ``assemble_stream``. Nothing in frontend-crates is modified.

Usage:
    FRONTEND_CRATES_DIR=/path/to/frontend-crates \\
        python3 tests/parity/sglang_processor/run.py --output /tmp/sglang-replay

Writes per-family captures, ``comparison.json``, ``summary.md`` and
``report.html``. Exits 1 with ``--strict`` if any final output differs, or
on an integrity failure (see ``--require-published-match``).
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import subprocess
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run(module: Any, job: dict) -> dict:
    original_in, original_out = sys.stdin, sys.stdout
    try:
        sys.stdin, sys.stdout = io.StringIO(json.dumps(job)), io.StringIO()
        module.main()
        return yaml.safe_load(sys.stdout.getvalue())
    finally:
        sys.stdin, sys.stdout = original_in, original_out


def _state(result: dict) -> str:
    if "chunks" in result:
        return "value"
    return "error" if "error" in result else "unavailable"


def _calls(events: list[dict]) -> list[tuple]:
    return [
        (e["name"], json.dumps(e["arguments"], sort_keys=True, ensure_ascii=False))
        for e in events
        if e["kind"] == "tool_call"
    ]


def _text(events: list[dict], kind: str) -> str:
    return "".join(e["text"] for e in events if e["kind"] == kind)


def classify(native: list[dict], dynamo: list[dict]) -> str:
    """Name the first kind of difference between two assembled outputs."""
    native_calls, dynamo_calls = _calls(native), _calls(dynamo)
    if len(dynamo_calls) > len(native_calls):
        return "dynamo_more_tool_calls"
    if len(dynamo_calls) < len(native_calls):
        return "dynamo_fewer_tool_calls"
    if native_calls != dynamo_calls:
        return "tool_call_name_or_arguments"
    if any(_text(native, k) != _text(dynamo, k) for k in ("text", "reasoning")):
        return "text_or_reasoning"
    return "event_order"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--families", nargs="*", help="Limit to these families (default: all)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any executed case's final output differs",
    )
    parser.add_argument(
        "--require-published-match",
        action="store_true",
        help="Exit 1 unless the native run reproduces every committed "
        "sglang_python capture it is checked against (same SGLang version only)",
    )
    args = parser.parse_args()

    fc_root = Path(os.environ["FRONTEND_CRATES_DIR"]).resolve()
    utils = fc_root / "conformance/utils/src"
    families_dir = fc_root / "conformance/fixtures-unified-v2/families"
    sys.path.insert(0, str(utils))
    sys.path.insert(0, str(HERE))

    import capture_dynamo_sglang_unified as dynamo  # noqa: PLC0415
    import capture_sglang_unified as native  # noqa: PLC0415
    import normalized_yaml  # noqa: PLC0415
    from capture_stimulus import assemble_stream  # noqa: PLC0415
    from capture_unified_job import build_job  # noqa: PLC0415

    sglang_version = version("sglang")

    def published(family: str) -> dict:
        path = families_dir / family / "sglang_python.yaml"
        if not path.exists():
            return {}
        authored = yaml.safe_load(
            (families_dir / family / "inputs_and_golden.yaml").read_text()
        )
        snapshot = normalized_yaml.expand_snapshot(
            yaml.safe_load(path.read_text()), authored
        )
        selected = (snapshot.get("capture_selection") or {}).get("version")
        if selected != sglang_version:
            return {}
        return {
            record["value"]["case_key"]: record["value"]["observation"]
            for record in snapshot["records"].values()
        }

    args.output.mkdir(parents=True, exist_ok=True)
    captures = args.output / "captures"
    captures.mkdir(exist_ok=True)
    families = sorted(
        p.name
        for p in families_dir.iterdir()
        if (p / "inputs_and_golden.yaml").exists()
    )
    if args.families:
        families = [f for f in families if f in set(args.families)]

    report: dict[str, Any] = {
        "provenance": {
            "frontend_crates_revision": _git_revision(fc_root),
            "dynamo_revision": _git_revision(HERE),
            "sglang_version": sglang_version,
            "boundary": "text injection into SglangStreamingPostProcessor._process_text",
            "bypassed": [
                "incremental detokenization",
                "stop-token and stop-string filtering",
                "token-id delimiter gating",
                "logprobs",
                "Rust HTTP/SSE projection",
            ],
        },
        "families": {},
        "cases": [],
    }
    for family in families:
        if family not in native.FAMILY_PARSERS:
            report["families"][family] = {
                "skipped": "no registered SGLang stream pipeline"
            }
            continue
        job = build_job(families_dir, family, mode="unified")
        golden = {
            case["id"]: (case.get("golden") or {}).get("assembled")
            for case in build_job(families_dir, family, mode="stream")["cases"]
        }
        native_out, dynamo_out = _run(native, job), _run(dynamo, job)
        for name, document in (("native", native_out), ("dynamo", dynamo_out)):
            (captures / f"{family}.{name}.yaml").write_text(
                yaml.safe_dump(
                    document, allow_unicode=True, sort_keys=False, width=4096
                )
            )
        recorded = published(family)
        counts: Counter = Counter()
        for case in job["cases"]:
            native_result = native_out["results"][case["id"]]
            dynamo_result = dynamo_out["results"][case["id"]]
            row: dict[str, Any] = {
                "family": family,
                "id": case["id"],
                "scenario": case["scenario"],
                "native_state": _state(native_result),
                "dynamo_state": _state(dynamo_result),
            }
            if row["native_state"] != row["dynamo_state"]:
                counts["state_mismatch"] += 1
            if row["native_state"] == row["dynamo_state"] == "value":
                n_asm = assemble_stream(native_result["chunks"])
                d_asm = assemble_stream(dynamo_result["chunks"])
                row.update(
                    native_assembled=n_asm,
                    dynamo_assembled=d_asm,
                    chunks_equal=native_result["chunks"] == dynamo_result["chunks"],
                    assembled_equal=n_asm == d_asm,
                    native_equals_golden=n_asm == golden.get(case["id"]),
                    dynamo_equals_golden=d_asm == golden.get(case["id"]),
                )
                if not row["assembled_equal"]:
                    row["difference"] = classify(n_asm, d_asm)
                    counts["diff:" + row["difference"]] += 1
                observation = recorded.get(case["scenario"])
                if observation and "value" in observation:
                    row["native_matches_published"] = (
                        n_asm == observation["value"]["assembled"]
                    )
                    counts["published_checked"] += 1
                    counts["published_equal"] += row["native_matches_published"]
                counts["executed"] += 1
                counts["chunks_equal"] += row["chunks_equal"]
                counts["assembled_equal"] += row["assembled_equal"]
            else:
                counts["not_executed"] += 1
            report["cases"].append(row)
        report["families"][family] = dict(counts)

    totals: Counter = Counter()
    for counts in report["families"].values():
        totals.update({k: v for k, v in counts.items() if isinstance(v, int)})
    report["totals"] = dict(totals)
    (args.output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1)
    )
    (args.output / "summary.md").write_text(_summary_markdown(report))
    (args.output / "report.html").write_text(_report_html(report))
    print(_summary_markdown(report))

    failed = False
    if totals["state_mismatch"]:
        print("FAIL: native and Dynamo executed different case sets", file=sys.stderr)
        failed = True
    if (
        args.require_published_match
        and totals["published_checked"] != totals["published_equal"]
    ):
        print("FAIL: native run does not reproduce committed captures", file=sys.stderr)
        failed = True
    if args.strict and totals["executed"] != totals["assembled_equal"]:
        print("FAIL: final output differs from native SGLang", file=sys.stderr)
        failed = True
    return 1 if failed else 0


def _summary_markdown(report: dict) -> str:
    p, t = report["provenance"], report["totals"]
    lines = [
        "# Dynamo SGLang chat processor vs native SGLang (Unified corpus)",
        "",
        f"- frontend-crates `{p['frontend_crates_revision']}`, Dynamo `{p['dynamo_revision']}`, SGLang `{p['sglang_version']}`",
        f"- Boundary: {p['boundary']}; bypassed: {', '.join(p['bypassed'])}",
        f"- Native run reproduces committed captures: {t.get('published_equal', 0)}/{t.get('published_checked', 0)}",
        "",
        "| Family | Executed | Same final output | Different | Same per-chunk stream |",
        "|---|---|---|---|---|",
    ]
    for family, c in sorted(report["families"].items()):
        if "skipped" in c:
            lines.append(f"| {family} | skipped: {c['skipped']} | | | |")
            continue
        lines.append(
            f"| {family} | {c.get('executed', 0)} | {c.get('assembled_equal', 0)} | "
            f"{c.get('executed', 0) - c.get('assembled_equal', 0)} | {c.get('chunks_equal', 0)} |"
        )
    lines.append(
        f"| **total** | {t.get('executed', 0)} | {t.get('assembled_equal', 0)} | "
        f"{t.get('executed', 0) - t.get('assembled_equal', 0)} | {t.get('chunks_equal', 0)} |"
    )
    lines += ["", "| Difference | Cases |", "|---|---|"]
    for key, value in sorted(t.items(), key=lambda kv: -kv[1]):
        if key.startswith("diff:"):
            lines.append(f"| {key[5:]} | {value} |")
    return "\n".join(lines) + "\n"


def _events_html(events: list[dict] | None) -> str:
    if events is None:
        return "<i>n/a</i>"
    parts = []
    for e in events:
        if e["kind"] == "tool_call":
            body = (
                f"<b>{html.escape(str(e['name']))}</b>"
                f"({html.escape(json.dumps(e['arguments'], ensure_ascii=False))})"
            )
        else:
            body = html.escape(repr(e["text"]))
        parts.append(f'<div class="{e["kind"]}"><span>{e["kind"]}</span> {body}</div>')
    return "".join(parts) or "<i>(empty)</i>"


def _report_html(report: dict) -> str:
    rows = []
    for c in report["cases"]:
        if c.get("assembled_equal", True):
            continue
        closest = (
            "Dynamo"
            if c["dynamo_equals_golden"]
            else "native"
            if c["native_equals_golden"]
            else "neither"
        )
        rows.append(
            f"<tr><td>{html.escape(c['family'])}</td><td><code>{html.escape(c['scenario'])}</code></td>"
            f"<td>{html.escape(c['difference'])}</td><td>{closest}</td>"
            f"<td>{_events_html(c['native_assembled'])}</td><td>{_events_html(c['dynamo_assembled'])}</td></tr>"
        )
    summary = html.escape(_summary_markdown(report))
    return (
        '<!doctype html><meta charset="utf-8"><title>Dynamo SGLang processor replay</title>'
        "<style>body{font:13px system-ui,sans-serif;margin:20px}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:4px 6px;vertical-align:top}th{background:#f3f3f3}"
        "span{display:inline-block;min-width:68px;color:#666}.tool_call span{color:#0a5}"
        ".reasoning span{color:#a50}</style>"
        f"<pre>{summary}</pre>"
        f"<h2>Cases whose final output differs ({len(rows)})</h2>"
        "<p>Closest to GOLDEN uses exact equality, which is stricter than the frontend-crates viewer.</p>"
        "<table><tr><th>Family</th><th>Case</th><th>Difference</th><th>Equals GOLDEN</th>"
        "<th>Native SGLang</th><th>Dynamo processor</th></tr>"
        + "".join(rows)
        + "</table>"
    )


if __name__ == "__main__":
    sys.exit(main())
