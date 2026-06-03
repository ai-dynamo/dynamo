#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Generate the FE.process_output behavioral parity matrix.

Merges per-backend pytest JUnit (from the shared YAML fixtures:
FE.process_output.4 assembly, .6 detok, .9 reasoning) into a fixture-case x
backend grid: pass / xfail (documented gap) / FAIL / n/a.

This is the behavioral analog of the parser PARITY.html grid -- but NOT a
Dynamo-vs-reference divergence grid: vLLM/SGLang expose no callable frontend on
the same input, so a cell means "does Dynamo-on-that-engine satisfy the shared
contract." An xfail cell is a vllm<->sglang divergence on Dynamo's own behavior.

The other frontend stages aren't expressible as a single shared input->output
replay (backend-specific request shaping, per-backend functions, mock-based
error paths), so they live as the `# FE.preprocess.N` / `# FE.response_misc.N`
annotated unit tests in this directory rather than in this matrix.

Usage:
  python frontend_matrix.py cases --junit vllm=vllm.xml sglang=sglang.xml
  python frontend_matrix.py html  --junit vllm=vllm.xml sglang=sglang.xml --out PARITY.html
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

BACKENDS = ["vllm", "sglang"]

_CASE_COLOR = {
    "pass": "#bfe3c6",
    "xfail": "#fde9a9",
    "FAIL": "#efb3b3",
    "skip": "#e4e8ec",
    "n/a": "#e4e8ec",
}
_CASES_DOC_HREF = "../../../components/src/dynamo/frontend/tests/FRONTEND_CASES.md"
_CASES_DOC_LABEL = "components/src/dynamo/frontend/tests/FRONTEND_CASES.md"
# (fixture file, glossary label, FE.process_output stage prefix)
_FIXTURES = [
    ("frontend_assembly", "assembly (FE.process_output.4)", "FE.process_output.4"),
    ("frontend_detok", "detok (FE.process_output.6)", "FE.process_output.6"),
    ("frontend_reasoning", "reasoning (FE.process_output.9)", "FE.process_output.9"),
]


# --------------------------------------------------------------------------- #
# JUnit -> case status
# --------------------------------------------------------------------------- #


def _case_id(name: str) -> str:
    m = re.search(r"\[(.+)\]", name)
    return m.group(1) if m else name


def _parse_junit_full(path: Path) -> dict[str, tuple[str, str]]:
    """param-id -> (status, message). status in {pass, xfail, FAIL, skip}."""
    out: dict[str, tuple[str, str]] = {}
    root = ET.parse(path).getroot()
    for tc in root.iter("testcase"):
        cid = _case_id(tc.get("name", ""))
        status, message = "pass", ""
        for child in tc:
            if child.tag in ("failure", "error"):
                status, message = "FAIL", child.get("message", "")
            elif child.tag == "skipped":
                blob = f"{child.get('type', '')} {child.get('message', '')}".lower()
                status = "xfail" if "xfail" in blob else "skip"
                message = child.get("message", "")
        out[cid] = (status, message)
    return out


def parse_junit(path: Path) -> dict[str, str]:
    """param-id -> status in {pass, xfail, FAIL, skip}."""
    return {cid: status for cid, (status, _) in _parse_junit_full(path).items()}


def render_cases(junits: dict[str, Path]) -> str:
    per_backend = {b: parse_junit(p) for b, p in junits.items()}
    backends = [b for b in BACKENDS if b in per_backend] + [
        b for b in per_backend if b not in BACKENDS
    ]
    case_ids = sorted({c for res in per_backend.values() for c in res})
    rows = [
        "| Case | " + " | ".join(backends) + " |",
        "|" + "---|" * (len(backends) + 1),
    ]
    for cid in case_ids:
        cells = [per_backend[b].get(cid, "n/a") for b in backends]
        rows.append(f"| {cid} | " + " | ".join(cells) + " |")
    legend = "\nLegend: pass; xfail = known, documented gap; FAIL = unexpected; n/a = not run on that backend."
    return "\n".join(rows) + "\n" + legend


# --------------------------------------------------------------------------- #
# HTML dashboard — TOOLCALLING / REASONING PARITY.html style. Local artifact.
# --------------------------------------------------------------------------- #

_CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 1.5em; }
h1 { font-size: 22px; } h2 { font-size: 16px; margin-top: 1.2em; }
.generated { color: #888; font-size: 12px; }
table { border-collapse: collapse; font-family: ui-monospace, monospace; font-size: 13px; margin: 0.6em 0; }
th, td { border: 1px solid #ccc; padding: 3px 8px; }
th { background: #f5f5f5; }
th.rowhdr { text-align: left; font-weight: 600; font-family: -apple-system, system-ui, sans-serif; }
td.cell { text-align: center; min-width: 64px; position: relative; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.04); }
td.cell:hover { z-index: 2000; filter: brightness(0.97); }
.legend { font-size: 12px; color: #444; max-width: 72em; margin: 0.5em 0; }
.swatch { display: inline-block; width: 12px; height: 12px; border: 1px solid rgba(0,0,0,0.22); vertical-align: -2px; margin: 0 3px 0 12px; }
.swatch:first-of-type { margin-left: 0; }
.table-toolbar { display: flex; gap: 12px; flex-wrap: wrap; margin: 0.8em 0 0.4em; }
.radio-group { display: inline-flex; align-items: center; gap: 8px; border: 1px solid #d0d7de; background: #f8fafc; border-radius: 6px; padding: 6px 10px; }
.radio-group-label { font-size: 13px; font-weight: 600; }
.radio-option { display: inline-flex; align-items: center; gap: 4px; font-size: 13px; cursor: pointer; }
.radio-option input { margin: 0; }
body.view-overview td.cell .cell-label { visibility: hidden; }
body.view-details td.cell .cell-label { visibility: visible; }
table.glossary { margin-top: 0.5em; }
table.glossary td.sub { white-space: nowrap; font-weight: bold; }
table.glossary tr.category td { background: #eef; font-family: -apple-system, system-ui, sans-serif; font-weight: bold; }
.info-panel { border: 1px solid #d0d7de; background: #f8fafc; padding: 12px 14px; margin: 1em 0; border-radius: 6px; max-width: 1100px; }
.info-panel h2 { margin: 0 0 0.35em 0; font-size: 17px; }
.info-panel p { margin: 0 0 0.7em 0; font-size: 13px; color: #333; }
/* hover tooltip — same look as the TOOLCALLING / REASONING dashboards */
.ttip { visibility: hidden; opacity: 0; position: absolute; left: 0; top: 100%; z-index: 3000; background: #1f2937; color: #e5e7eb; padding: 8px 10px; border-radius: 6px; font-family: ui-monospace, monospace; font-size: 12px; line-height: 1.4; min-width: 280px; max-width: 70vw; width: max-content; box-shadow: 0 4px 14px rgba(0,0,0,0.35); text-align: left; white-space: normal; transition: opacity 120ms ease; }
td.cell:hover .ttip { visibility: visible; opacity: 1; }
.ttip-head { font-weight: bold; margin-bottom: 4px; color: #fbbf24; }
.ttip-section { font-weight: bold; margin-top: 6px; color: #93c5fd; }
.ttip-pre { margin: 2px 0 0 0; white-space: pre-wrap; word-break: break-word; color: #e5e7eb; }
"""

_VIEW_JS = """
(function(){
  var radios = Array.prototype.slice.call(document.querySelectorAll('input[name="fe-view"]'));
  function setView(v){
    document.body.classList.remove('view-overview', 'view-details');
    document.body.classList.add('view-' + v);
    try { var u = new URL(window.location); u.searchParams.set('view', v); history.replaceState(null, '', u); } catch (e) {}
  }
  var want = new URLSearchParams(window.location.search).get('view');
  var v0 = (want === 'details' || want === 'overview') ? want : 'overview';
  radios.forEach(function(r){ r.checked = (r.value === v0); r.addEventListener('change', function(){ if (r.checked) setView(r.value); }); });
  setView(v0);
})();
"""


def _esc(s) -> str:
    return html.escape(str(s))


def _cell(text: str, color: str, head: str, lines: list[tuple[str, str]]) -> str:
    tip = '<span class="ttip"><span class="ttip-head">' + _esc(head) + "</span>"
    for section, body in lines:
        if section:
            tip += '<div class="ttip-section">' + _esc(section) + "</div>"
        tip += '<div class="ttip-pre">' + _esc(body) + "</div>"
    tip += "</span>"
    return f'<td class="cell" style="background:{color}"><span class="cell-label">{_esc(text)}</span>{tip}</td>'


def _legend(items: list[tuple[str, str]], tail: str) -> str:
    spans = "".join(
        f'<span class="swatch" style="background:{_CASE_COLOR[key]}"></span>{_esc(label)}'
        for key, label in items
    )
    return f'<p class="legend">{spans} · {tail}</p>'


def _load_fixture_cases(tests_dir: Path) -> dict[str, dict]:
    """case_id -> {description, model_text, expected, fixture} from the fixture YAMLs."""
    out: dict[str, dict] = {}
    for name, label, stage in _FIXTURES:
        path = tests_dir / "fixtures" / f"{name}.yaml"
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text()) or {}
        for case_id, body in (data.get("cases") or {}).items():
            out[case_id] = {
                "description": body.get("description", ""),
                "model_text": body.get("model_text", ""),
                "expected": body.get("expected", {}) or {},
                "fixture": label,
                "stage": stage,
            }
    return out


def _format_expected(expected: dict) -> str:
    """Render a fixture `expected` block as the contract both backends must meet."""
    if not expected:
        return "(see fixture)"
    parts: list[str] = []
    calls = expected.get("tool_calls")
    if calls is not None:
        if calls:
            parts.append(
                "tool_calls: "
                + "; ".join(
                    f"{c.get('name')}({json.dumps(c.get('arguments', {}), ensure_ascii=False)})"
                    for c in calls
                )
            )
        else:
            parts.append("tool_calls: []")
    if "reasoning_contains" in expected:
        parts.append(f"reasoning contains: {expected['reasoning_contains']!r}")
    if "content" in expected:
        parts.append(f"content: {expected['content']!r}")
    if "content_contains" in expected:
        parts.append(f"content contains: {expected['content_contains']!r}")
    if "finish_reason" in expected:
        parts.append(f"finish_reason: {expected['finish_reason']}")
    return "\n".join(parts)


def _matrix(tests_dir: Path, junits: dict[str, Path]) -> str:
    full = {b: _parse_junit_full(p) for b, p in junits.items()}
    backends = [b for b in BACKENDS if b in full] + [
        b for b in full if b not in BACKENDS
    ]
    cases = _load_fixture_cases(tests_dir)
    case_ids = sorted({c for res in full.values() for c in res})

    rows = (
        "<tr><th>Case</th>" + "".join(f"<th>{_esc(b)}</th>" for b in backends) + "</tr>"
    )
    for cid in case_ids:
        case_base, _, chunk = cid.rpartition("-")
        info = cases.get(case_base, {})
        stage = info.get("stage", "FRONTEND.?")
        label = f"{stage}.{case_base} · chunk={chunk}"
        model_text = info.get("model_text", "")
        expected = _format_expected(info.get("expected", {}))
        cells = ""
        for b in backends:
            status, message = full[b].get(cid, ("n/a", ""))
            lines = []
            if info.get("description"):
                lines.append(("", info["description"]))
            if model_text:
                lines.append(("input (model_text)", model_text))
            lines.append(("expected output (contract)", expected))
            if message:
                # xfail/FAIL reasons carry the observed output after a "||" marker.
                why, sep, actual = message.partition("||")
                if sep:
                    lines.append(("ACTUAL output (this backend)", actual.strip()))
                    lines.append(("why", why.strip()))
                else:
                    lines.append(("status reason", why.strip()))
            cells += _cell(
                status,
                _CASE_COLOR.get(status, "#e4e8ec"),
                f"{label} · {b} · {status}",
                lines,
            )
        rows += f'<tr><th class="rowhdr">{_esc(label)}</th>{cells}</tr>'

    legend = _legend(
        [
            ("pass", "pass"),
            ("xfail", "xfail (documented gap; hover for reason)"),
            ("FAIL", "FAIL"),
            ("n/a", "n/a (not run on that backend)"),
        ],
        "View: Overview = color blocks, Details = labels. Hover any cell for the case input (model_text), expected output, and status reason.",
    )

    groups: dict[str, list[tuple[str, str]]] = {}
    for cid, info in cases.items():
        groups.setdefault(info["fixture"], []).append((cid, info["description"]))
    glossary = ""
    for label in sorted(groups):
        glossary += f'<tr class="category"><td colspan="2">{_esc(label)}</td></tr>'
        for cid, d in sorted(groups[label]):
            glossary += f'<tr><td class="sub">{_esc(cid)}</td><td>{_esc(d)}</td></tr>'
    return (
        "<section>"
        f"<table>{rows}</table>{legend}"
        f'<h2>Fixture case descriptions</h2><p class="legend">Cases live in '
        f"<code>fixtures/frontend_*.yaml</code>; taxonomy in "
        f'<a href="{_CASES_DOC_HREF}">{_CASES_DOC_LABEL}</a>.</p>'
        f'<table class="glossary"><tbody>{glossary}</tbody></table>'
        "</section>"
    )


def _info_panel() -> str:
    return (
        '<div class="info-panel">'
        "<h2>FE.process_output — behavioral parity (chat-processor)</h2>"
        "<p>The frontend turns OpenAI requests into engine input and re-assembles the engine "
        "stream into OpenAI chunks. This grid replays the shared YAML fixtures through both "
        "backends' <code>StreamingPostProcessor.process_output</code> and checks the assembled "
        "deltas: <strong>FE.process_output.4</strong> tool-call assembly, <strong>.6</strong> "
        "incremental detok, <strong>.9</strong> reasoning&harr;tool orchestration. Write a case "
        "once; both engines must satisfy it.</p>"
        "<p><strong>Not a Dynamo-vs-reference divergence grid.</strong> vLLM/SGLang expose no "
        "callable frontend on the same input, so a cell means &ldquo;does Dynamo-on-that-engine "
        "satisfy the shared contract.&rdquo; An <code>xfail</code> cell is a vllm-vs-sglang "
        "divergence on Dynamo's own behavior (hover for the ACTUAL output + reason).</p>"
        "<p>The frontend's other stages aren't a single shared input&rarr;output replay, so they "
        "are per-backend annotated unit tests (not this grid): <code>FE.preprocess.*</code> "
        "(1 chat-template, 2 dispatch, 3 request shaping, 7 worker boundary) and "
        "<code>FE.response_misc.*</code> (5 finish-reason, 8 error surface).</p>"
        "</div>"
    )


def render_html(tests_dir: Path, junits: dict[str, Path], generated_at: str) -> str:
    toolbar = (
        '<div class="table-toolbar"><div class="radio-group" role="radiogroup" aria-label="View">'
        '<span class="radio-group-label">View:</span>'
        '<label class="radio-option"><input type="radio" name="fe-view" value="overview" checked> Overview</label>'
        '<label class="radio-option"><input type="radio" name="fe-view" value="details"> Details</label>'
        "</div></div>"
    )
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        "<title>FE.process_output Parity</title><style>" + _CSS + "</style></head>"
        '<body class="view-overview">'
        "<h1>FE.process_output parity — Dynamo chat-processor</h1>"
        f'<p class="generated">Auto-generated {_esc(generated_at)} · local artifact (not committed) · '
        "<code>frontend_matrix.py html</code></p>"
        + _info_panel()
        + toolbar
        + _matrix(tests_dir, junits)
        + "<script>"
        + _VIEW_JS
        + "</script>"
        "</body></html>"
    )


def _parse_junit_args(pairs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for pair in pairs:
        backend, _, path = pair.partition("=")
        out[backend] = Path(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate the FRONTEND.* behavioral parity matrix."
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    cases = sub.add_parser("cases", help="case x backend, merge JUnit XML (markdown)")
    cases.add_argument("--junit", nargs="+", required=True, metavar="backend=path.xml")

    htmlp = sub.add_parser("html", help="render PARITY.html")
    htmlp.add_argument("--tests-dir", type=Path, default=Path(__file__).parent)
    htmlp.add_argument("--junit", nargs="+", required=True, metavar="backend=path.xml")
    htmlp.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    if args.mode == "cases":
        print(render_cases(_parse_junit_args(args.junit)))
    else:
        generated_at = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
            "%Y-%m-%d %H:%M %Z"
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            render_html(args.tests_dir, _parse_junit_args(args.junit), generated_at)
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
