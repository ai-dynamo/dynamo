#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Generate the FRONTEND.* parity matrices.

Two distinct matrices, two sources:

  coverage  — stage x backend, from a static scan of `# FRONTEND.N`
              annotations across frontend/tests/test_*.py. Cell states:
              covered (own-backend test), shared (only a shared-utils test),
              GAP, n/a. This is the canonical FRONTEND coverage-parity artifact;
              it needs no test run.

  cases     — fixture-case x backend, from merging per-backend pytest JUnit XML
              (run test_{vllm,sglang}_frontend_assembly.py with --junit-xml).
              Cell states: pass / xfail (known gap) / FAIL / n/a. This is the
              behavioral analog of the parser PARITY.html grid; xfail cells are
              the vllm<->sglang divergences. NOTE: unlike the parser grid, there
              is no peer-captured expected.{vllm,sglang} -- cells say whether
              Dynamo-on-that-engine satisfies the shared contract, not how the
              reference engines diverge.

Usage:
  python frontend_matrix.py coverage
  python frontend_matrix.py cases --junit vllm=vllm.xml sglang=sglang.xml
"""

from __future__ import annotations

import argparse
import datetime
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zoneinfo import ZoneInfo

STAGES = [str(i) for i in range(1, 10)]
STAGE_NAMES = {
    "1": "chat-template preprocessing",
    "2": "parser construction/dispatch",
    "3": "request shaping",
    "4": "tool-call assembly",
    "5": "finish-reason mapping",
    "6": "incremental detok",
    "7": "worker subprocess boundary",
    "8": "error surface",
    "9": "reasoning<->tool orchestration",
}
# Settled n/a cells (documented in FRONTEND_CASES.md / DIS-2064 notes).
NA_CELLS = {("7", "vllm")}
BACKENDS = ["vllm", "sglang"]


def _backend_of(filename: str) -> str:
    if "vllm" in filename:
        return "vllm"
    if "sglang" in filename:
        return "sglang"
    return "shared"


def scan_coverage(tests_dir: Path) -> dict[str, dict[str, set[str]]]:
    """stage -> {vllm,sglang,shared} -> set of files carrying that annotation."""
    cov = {s: {"vllm": set(), "sglang": set(), "shared": set()} for s in STAGES}
    for path in sorted(tests_dir.glob("test_*.py")):
        backend = _backend_of(path.name)
        for line in path.read_text().splitlines():
            if "#" not in line or "FRONTEND." not in line:
                continue
            comment = line.split("#", 1)[1]  # annotations live in the comment
            for stage in re.findall(r"FRONTEND\.([0-9])", comment):
                cov[stage][backend].add(path.name)
    return cov


def coverage_cell(cov, stage: str, backend: str) -> str:
    if (stage, backend) in NA_CELLS:
        return "n/a"
    if cov[stage][backend]:
        return "covered"
    if cov[stage]["shared"]:
        return "shared"
    return "GAP"


def render_coverage(tests_dir: Path) -> str:
    cov = scan_coverage(tests_dir)
    rows = ["| Stage | vllm | sglang |", "|---|---|---|"]
    for s in STAGES:
        rows.append(
            f"| {s} {STAGE_NAMES[s]} | {coverage_cell(cov, s, 'vllm')} | {coverage_cell(cov, s, 'sglang')} |"
        )
    legend = (
        "\nLegend: covered = own-backend test; shared = covered only by a shared-utils test "
        "(backend-specific wiring still untested); GAP = no test; n/a = stage does not apply "
        "to this backend."
    )
    return "\n".join(rows) + "\n" + legend


# --------------------------------------------------------------------------- #
# Behavioral case-matrix (merge per-backend JUnit XML)
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
# HTML dashboard (PARITY.html-style; local artifact, never committed)
# --------------------------------------------------------------------------- #

_COV_COLOR = {
    "covered": "#d4edda",
    "shared": "#fff3cd",
    "GAP": "#f8d7da",
    "n/a": "#e9ecef",
}
_CASE_COLOR = {
    "pass": "#d4edda",
    "xfail": "#fff3cd",
    "FAIL": "#f8d7da",
    "skip": "#e9ecef",
}


def _td(text: str, color: str, title: str = "") -> str:
    tip = f' title="{html.escape(title)}"' if title else ""
    return f'<td class="cell"{tip} style="background:{color}">{html.escape(str(text))}</td>'


def _coverage_html(tests_dir: Path) -> str:
    cov = scan_coverage(tests_dir)
    rows = ["<tr><th>Stage</th><th>vllm</th><th>sglang</th></tr>"]
    for s in STAGES:
        cells = "".join(
            _td(v := coverage_cell(cov, s, b), _COV_COLOR.get(v, "#fff"))
            for b in BACKENDS
        )
        rows.append(
            f"<tr><th class='rowhdr'>{s} {html.escape(STAGE_NAMES[s])}</th>{cells}</tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _cases_html(junits: dict[str, Path]) -> str:
    full = {b: _parse_junit_full(p) for b, p in junits.items()}
    backends = [b for b in BACKENDS if b in full] + [
        b for b in full if b not in BACKENDS
    ]
    case_ids = sorted({c for res in full.values() for c in res})
    rows = ["<tr><th>Case</th>" + "".join(f"<th>{b}</th>" for b in backends) + "</tr>"]
    for cid in case_ids:
        cells = ""
        for b in backends:
            status, message = full[b].get(cid, ("n/a", ""))
            cells += _td(status, _CASE_COLOR.get(status, "#e9ecef"), title=message)
        rows.append(f"<tr><th class='rowhdr'>{html.escape(cid)}</th>{cells}</tr>")
    return "<table>" + "".join(rows) + "</table>"


def render_html(tests_dir: Path, junits: dict[str, Path], generated_at: str) -> str:
    cases_section = (
        _cases_html(junits)
        if junits
        else "<p><em>No JUnit provided — run the fixture suites with --junit-xml on each backend.</em></p>"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>FRONTEND.* Parity</title>
<style>
 body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2rem; color: #222; }}
 h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 1.8rem; }}
 table {{ border-collapse: collapse; margin: 0.6rem 0; }}
 td, th {{ border: 1px solid #cfcfcf; padding: 4px 10px; font-size: 0.86rem; }}
 th {{ background: #f4f4f4; }} th.rowhdr {{ text-align: left; font-weight: 500; }}
 td.cell {{ text-align: center; }}
 .legend {{ font-size: 0.82rem; color: #444; max-width: 60rem; }}
 .swatch {{ display: inline-block; width: 0.9rem; height: 0.9rem; border: 1px solid #aaa; vertical-align: middle; margin-right: 3px; }}
 .note {{ background: #f6f8fa; border-left: 3px solid #999; padding: 0.5rem 0.9rem; max-width: 60rem; font-size: 0.85rem; }}
</style></head><body>
<h1>FRONTEND.* Parity — Dynamo chat-processor</h1>
<p class="legend">Generated {html.escape(generated_at)}. Local artifact — not committed.</p>

<h2>1. Coverage matrix (stage &times; backend)</h2>
<p class="legend">Static scan of <code># FRONTEND.N</code> annotations.
 <span class="swatch" style="background:{_COV_COLOR['covered']}"></span>covered (own-backend test)
 <span class="swatch" style="background:{_COV_COLOR['shared']}"></span>shared (only a shared-utils test)
 <span class="swatch" style="background:{_COV_COLOR['GAP']}"></span>GAP
 <span class="swatch" style="background:{_COV_COLOR['n/a']}"></span>n/a</p>
{_coverage_html(tests_dir)}

<h2>2. Behavioral case-matrix (fixture case &times; backend)</h2>
<p class="legend">Merged from per-backend pytest JUnit (FRONTEND.4 assembly + FRONTEND.6 detok).
 <span class="swatch" style="background:{_CASE_COLOR['pass']}"></span>pass
 <span class="swatch" style="background:{_CASE_COLOR['xfail']}"></span>xfail (known gap; hover for reason)
 <span class="swatch" style="background:{_CASE_COLOR['FAIL']}"></span>FAIL</p>
{cases_section}

<div class="note"><strong>Not a behavioral divergence grid.</strong> Unlike the parser
<code>PARITY.html</code> (which captures <code>expected.{{dynamo,vllm,sglang}}</code> from each
engine's parser on the same model text), FRONTEND has no callable peer frontend to capture, so a
cell means &ldquo;does Dynamo-on-that-engine satisfy the shared contract.&rdquo; The one xfail cell is a
<strong>vllm-vs-sglang</strong> divergence on Dynamo's own behavior, not Dynamo-vs-reference.</div>
</body></html>
"""


def _parse_junit_args(pairs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for pair in pairs:
        backend, _, path = pair.partition("=")
        out[backend] = Path(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate FRONTEND.* parity matrices.")
    sub = ap.add_subparsers(dest="mode", required=True)

    cov = sub.add_parser("coverage", help="stage x backend, static annotation scan")
    cov.add_argument("--tests-dir", type=Path, default=Path(__file__).parent)

    cases = sub.add_parser("cases", help="case x backend, merge JUnit XML")
    cases.add_argument("--junit", nargs="+", required=True, metavar="backend=path.xml")

    htmlp = sub.add_parser("html", help="render PARITY.html (both matrices)")
    htmlp.add_argument("--tests-dir", type=Path, default=Path(__file__).parent)
    htmlp.add_argument("--junit", nargs="*", default=[], metavar="backend=path.xml")
    htmlp.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    if args.mode == "coverage":
        print(render_coverage(args.tests_dir))
    elif args.mode == "cases":
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
