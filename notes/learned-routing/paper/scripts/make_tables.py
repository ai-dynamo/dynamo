#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the final report's generated markdown tables into LaTeX fragments for the paper.

The campaign's report stage writes every results table as a markdown fragment
(``<campaign root>/report/data/tables/*.md``) from the single test pass, with no replay. This script
copies selected columns and rows of those fragments into ``generated/*.tex`` so the paper's numbers
are the report's numbers, character for character. It computes nothing.

Usage (stdlib only):
    python3 scripts/make_tables.py --tables <campaign root>/report/data/tables --out generated
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

UNICODE = {
    "−": "$-$",  # minus sign
    "–": "--",
    "—": "---",
    "≥": "$\\geq$",
    "≤": "$\\leq$",
    "†": "$^{\\dagger}$",
    "θ": "$\\theta$",
    "τ": "$\\tau$",
    "×": "$\\times$",
    "√": "$\\surd$",
    "σ": "$\\sigma$",
    "·": "$\\cdot$",
    "|": "$|$",
    "*": "$^{*}$",
}


def tex(cell: str) -> str:
    """Escape one markdown cell for LaTeX; backticked names become \\code{}."""
    parts = re.split(r"(`[^`]*`)", cell.strip())
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) >= 2:
            out.append("\\code{" + escape(part[1:-1]) + "}")
        else:
            out.append(escape(part))
    return "".join(out)


def escape(text: str) -> str:
    # Checkpoint labels such as "g20" (CMA-ES generation 20) are written "gen 20".
    text = re.sub(r"(?<![\w.-])g(\d{1,2})(?!\w)", r"gen \1", text)
    text = text.replace("\\", "\\textbackslash{}")
    for char in "&%#_{}":
        text = text.replace(char, "\\" + char)
    text = text.replace("~", "\\textasciitilde{}")
    for char, repl in UNICODE.items():
        text = text.replace(char, repl)
    # Signed numbers keep a typographic minus.
    text = re.sub(r"(?<![\w$])-(?=\d)", "$-$", text)
    text = re.sub(r"(?<![\w$])\+(?=\d)", "$+$", text)
    # Adjacent inline-math pieces ("$\\cdot$$\\sigma$") would open display math; merge them.
    return text.replace("$$", "")


# Reader-facing names for the report's internal row labels (amendment numbers, tier and ablation
# tags). Applied by substring to every rendered cell AFTER rows are filtered by their raw report
# names, so KEY_ROWS and the rows= lists keep matching. Order matters: longer keys first.
# sections/appendix-provenance.tex maps each reader-facing name back to its campaign identifier.
RELABEL = [
    ("M1 (= A12 M1-default-init)", "M1"),
    (
        "M1 unconstrained (tier A, gaming-flagged)",
        "M1 unconstrained (flagged for gaming)",
    ),
    ("M1 unconstrained (gaming-flagged)", "M1 unconstrained (flagged for gaming)"),
    ("M1-noaff (A15)", "M1-noaff"),
    ("M1-v2 (A17.1)", "M1-v2"),
    ("(ablation a)", "(ablation)"),
    ("(abl. a)", "(ablation)"),
    ("A14 itl_only", "SLO: ITL bound only"),
    ("A14 e2e_only", "SLO: slowdown bound only"),
    ("A14 ttft_len_itl", "SLO: ITL + length-scaled TTFT"),
    ("A14 abs_e2e_itl", "SLO: ITL + absolute E2E bound"),
    ("A19 good tokens (secondary)", "token-weighted goodput (secondary)"),
    # One name per policy in every table and figure: "<policy>" when tuned, "<policy>@defaults" at
    # shipped parameters, the TTFT source in parentheses, and the prose spelling of round-robin.
    # Every baseline is a port, so lmetric carries no "(port)" tag of its own.
    (
        "llm-d-optimized-baseline@throughput",
        "llm-d-optimized-baseline@defaults (throughput)",
    ),
    ("sticky-bounded@defaults", "sticky-session bounded@defaults"),
    ("sticky-hard@defaults", "sticky-session hard@defaults"),
    ("lmetric (port)", "lmetric"),
    ("round_robin", "round-robin"),
]

# A selected configuration "<run>-s<restart> g<generation> <mean|best_so_far>" (the report's run key).
RUN_KEY = re.compile(r"^(\S+)-s(\d+) g(\d+) (mean|best_so_far)$")
CANDIDATE = {"mean": "CMA-ES mean", "best_so_far": "best-so-far sample"}


def relabel(cell: str) -> str:
    for old, new in RELABEL:
        cell = cell.replace(old, new)
    return cell


def run_key(cell: str) -> str | None:
    """Rewrite a run key as "restart r, gen. g, <candidate>"; None if the cell is not a run key."""
    m = RUN_KEY.match(cell.strip())
    if m is None:
        return None
    _, restart, generation, which = m.groups()
    return f"restart {restart}, gen.\\ {generation}, {CANDIDATE[which]}"


def split_row(line: str) -> list[str]:
    """Split one markdown table row on unescaped pipes and undo the markdown escapes \\| and \\*."""
    cells = re.split(r"(?<!\\)\|", line.strip()[1:-1])
    return [cell.strip().replace("\\|", "|").replace("\\*", "*") for cell in cells]


def read_md(path: Path, table: int = 0) -> tuple[list[str], list[list[str]]]:
    """Read the ``table``-th markdown table (0-based) of a fragment; most fragments hold one."""
    blocks: list[list[str]] = [[]]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            blocks[-1].append(line)
        elif blocks[-1]:
            blocks.append([])
    blocks = [block for block in blocks if block]
    if table >= len(blocks):
        raise SystemExit(
            f"{path.name}: has {len(blocks)} tables, asked for index {table}"
        )
    lines = blocks[table]
    header = split_row(lines[0])
    rows = []
    for line in lines[2:]:
        rows.append(split_row(line))
    for row in rows:
        if len(row) != len(header):
            raise SystemExit(
                f"{path.name}: row has {len(row)} cells, header {len(header)}: {row}"
            )
    return header, rows


def emit(
    tables: Path,
    out: Path,
    source: str,
    target: str,
    columns: list[tuple[str, str]],
    colspec: str,
    rows: list[str] | None = None,
    rename_rows: dict[str, str] | None = None,
    longtable_caption: str | None = None,
    label: str | None = None,
    size: str = "\\scriptsize",
    table: int = 0,
    colsep: str | None = None,
    stack: int | None = None,
    shrink: bool = True,
    cell_replace: dict[str, str] | None = None,
) -> None:
    """Write generated/<target>.tex: the chosen columns (source header -> printed header) and rows.

    ``rows`` filters and orders by the first column; ``None`` keeps every row in report order.
    ``table`` picks one table of a fragment that holds several (the live fragment).
    ``colsep`` overrides \\tabcolsep inside the table's group (wide live tables).
    ``stack`` splits the columns after the first into blocks of that many and stacks the blocks
    vertically, each with its own header row and every row repeated, so a wide table keeps its
    font size. The last block may be narrower. ``colspec`` then describes the first block.
    ``shrink`` (default) wraps a non-long table in adjustbox, which shrinks it only when it is wider
    than the text block; ``False`` sets it at ``size`` exactly (the layout must fit the width).
    ``cell_replace`` maps substrings of the rendered (LaTeX) cells to replacements, for typesetting
    fixes such as a non-breaking hyphen; it never touches a number.
    A longtable ends its second-to-last row with ``\\*``, so its last row never sits alone on a page.
    """
    header, body = read_md(tables / source, table)
    # A repeated header name is addressed as "name#2", "name#3", ... in the order it appears.
    index: dict[str, int] = {}
    for i, name in enumerate(header):
        key, n = name, 1
        while key in index:
            n += 1
            key = f"{name}#{n}"
        index[key] = i
    missing = [name for name, _ in columns if name not in index]
    if missing:
        raise SystemExit(f"{source}: missing columns {missing}; have {list(index)}")
    by_first = {row[0]: row for row in body}
    if rows is not None:
        absent = [name for name in rows if name not in by_first]
        if absent:
            raise SystemExit(f"{source}: missing rows {absent}")
        body = [by_first[name] for name in rows]
    rename_rows = rename_rows or {}
    if stack is None:
        blocks = [columns]
    else:
        rest = columns[1:]
        blocks = [
            [columns[0], *rest[i : i + stack]] for i in range(0, len(rest), stack)
        ]
        if longtable_caption is not None:
            raise SystemExit(f"{target}: stack={stack} needs a short table")
    heads = [" & ".join(printed for _, printed in block) + " \\\\" for block in blocks]
    head = heads[0]
    lines = [
        f"% GENERATED by scripts/make_tables.py from the report fragment report/data/tables/{source}"
        + (f", table {table + 1}." if table else "."),
        "% Do not edit; regenerate.",
    ]

    def render(block: list[tuple[str, str]]) -> list[str]:
        out = []
        for row in body:
            cells = []
            keys = []
            for k, (name, _) in enumerate(block):
                value = row[index[name]]
                if k == 0 and value in rename_rows:
                    cells.append(rename_rows[value])
                elif (step := run_key(value)) is not None:
                    cells.append(step)
                    keys.append(value.strip())
                else:
                    cells.append(tex(relabel(value)))
            for old, new in (cell_replace or {}).items():
                cells = [cell.replace(old, new) for cell in cells]
            line = " & ".join(cells) + " \\\\"
            if keys:
                line += " % run key: " + "; ".join(keys)
            out.append(line)
        return out

    rendered = render(columns)
    if longtable_caption is None:
        stacked = []
        for k, block in enumerate(blocks):
            stacked += [
                *(["\\midrule[\\heavyrulewidth]"] if k else []),
                heads[k],
                "\\midrule",
                *render(block),
            ]
        tabular = [
            "\\begin{tabular}{" + colspec + "}",
            "\\toprule",
            *stacked,
            "\\bottomrule",
            "\\end{tabular}",
        ]
        lines += [
            "{" + size,
            *([f"\\setlength{{\\tabcolsep}}{{{colsep}}}"] if colsep else []),
            # adjustbox shrinks a table only when it is wider than the text block.
            *(
                [
                    "\\begin{adjustbox}{max width=\\linewidth}",
                    *tabular,
                    "\\end{adjustbox}}",
                ]
                if shrink
                else [*tabular[:-1], tabular[-1] + "}"]
            ),
        ]
    else:
        lines += [
            "{" + size,
            "\\setlength{\\LTcapwidth}{\\linewidth}",
            "\\begin{longtable}{" + colspec + "}",
            "\\caption{"
            + longtable_caption
            + "}"
            + (f"\\label{{{label}}}" if label else "")
            + "\\\\",
            "\\toprule",
            head,
            "\\midrule",
            "\\endfirsthead",
            "\\toprule",
            head,
            "\\midrule",
            "\\endhead",
            "\\bottomrule",
            "\\endfoot",
            *keep_last_row(rendered),
            "\\end{longtable}}",
        ]
    (out / f"{target}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def keep_last_row(rows: list[str]) -> list[str]:
    """End the second-to-last longtable row with \\* so the last row cannot open a page alone."""
    if len(rows) < 2:
        return rows
    # The row end is the last " \\"; a "% run key:" comment may follow it.
    head, sep, tail = rows[-2].rpartition(" \\\\")
    if not sep:
        raise SystemExit(f"row without a row end: {rows[-2]}")
    return [*rows[:-2], head + " \\\\*" + tail, rows[-1]]


KEY_ROWS = [
    "M1-v2 (headline learned arm)",
    "M2-ais",
    "M1 (= A12 M1-default-init)",
    "M1 unconstrained (gaming-flagged)",
    "llm-d-precise-prefix",
    "ramjet (val-best baseline)",
    "lmetric (port)",
    "M0 (default cost fn, tuned)",
    "round_robin",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--tables", type=Path, required=True, help="report/data/tables directory"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="paper generated/ directory"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t, o = args.tables, args.out

    emit(
        t,
        o,
        "headline_strata.md",
        "strata",
        [
            ("Stratum", "Stratum"),
            ("Cells", "Cells"),
            ("Segments", "Seg."),
            (
                "M1-v2 minus ramjet, segment mean [95% CI]",
                "M1-v2 $-$ ramjet [95\\% CI]",
            ),
            ("Segments ahead", "Ahead"),
            ("One-sided exact Wilcoxon p (descriptive except 'all')", "One-sided $p$"),
        ],
        "@{}lrrlrr@{}",
        rename_rows={
            "all": "\\textbf{all (pre-registered test)}",
            "mode:open_speedup": "load mode: open loop",
            "mode:closed_concurrency": "load mode: closed loop",
            "mode:agentic_lanes": "load mode: agentic lanes",
            "N:2": "$\\Nw = 2$ (unseen)",
            "N:4": "$\\Nw = 4$ (train count)",
            "N:6": "$\\Nw = 6$ (selection-exposed)",
            "N:8": "$\\Nw = 8$ (train count)",
            "N:16": "$\\Nw = 16$ (unseen)",
            "N:32": "$\\Nw = 32$ (unseen)",
            "unseen_N": "unseen $\\Nw$ (2, 16, 32)",
            "seen_N": "seen $\\Nw$ (4, 8)",
            "transform_extrapolation": "transforms beyond train ranges",
            "family:mooncake": "family: Mooncake",
            "family:fast25_conversation": "family: FAST25 conversation",
            "family:fast25_synthetic": "family: FAST25 synthetic",
            "family:synthetic_sessions": "family: sessions",
            "family:agentx": "family: AgentX",
        },
    )
    emit(
        t,
        o,
        "ladder.md",
        "ladder",
        [
            ("Rung", "Rung"),
            ("Val k0-2 (selection)", "Val $k_{0..2}$"),
            ("Val k3-10 (fresh)", "Val $k_{3..10}$"),
            ("Test vs default (cell mean)", "Test vs $\\piref$"),
            ("Test vs ramjet, segment mean [95% CI]", "Test vs ramjet [95\\% CI]"),
            ("One-sided p vs ramjet", "$p$"),
        ],
        "@{}P{0.25\\linewidth}rrrlr@{}",
    )
    emit(
        t,
        o,
        "ladder_pairs.md",
        "ladder_pairs",
        [
            ("Comparison (A minus B)", "Comparison ($A - B$)"),
            ("Segment mean [95% CI]", "Segment mean [95\\% CI]"),
            ("Segments ahead", "Ahead"),
            ("One-sided p (A > B)", "$p$ ($A > B$)"),
            ("Cell mean", "Cell mean"),
        ],
        "@{}P{0.46\\linewidth}lrrr@{}",
    )
    emit(
        t,
        o,
        "coef_m1v2.md",
        "coef_m1v2",
        [
            ("#", "$f$"),
            ("Feature", "Feature"),
            ("θ", "$\\theta_f$"),
            ("Within-set sd (train)", "sd$_f$"),
            ("θ × sd (utility per sd)", "$\\theta_f\\,\\mathrm{sd}_f$"),
            ("Sign constraint", "Constraint"),
        ],
        "@{}rlrrrl@{}",
    )
    emit(
        t,
        o,
        "coef_v1.md",
        "coef_v1",
        [
            ("Arm", "Arm"),
            ("θ1 `overlap_frac`", "$\\theta_1$"),
            ("θ2 `new_prefill_tokens_k`", "$\\theta_2$"),
            ("θ3 `active_prefill_tokens_k`", "$\\theta_3$"),
            ("θ4 `kv_load_frac`", "$\\theta_4$"),
            ("θ5 `active_requests_s`", "$\\theta_5$"),
            ("θ6 `session_affinity`", "$\\theta_6$"),
            ("θ7 `isl_x_prefill_load`", "$\\theta_7$"),
        ],
        "@{}P{0.24\\linewidth}rrrrrrr@{}",
    )
    emit(
        t,
        o,
        "robustness.md",
        "robustness",
        [
            ("Condition", "Condition"),
            (
                "M1-v2 minus ramjet, segment mean [95% CI]",
                "M1-v2 $-$ ramjet [95\\% CI]",
            ),
            ("Segments ahead", "Ahead"),
            ("One-sided p", "$p$"),
            ("Kendall τ-b vs nominal ranking", "$\\kendall$-b"),
            ("Rank of M1-v2 / ramjet", "Rank M1-v2 / ramjet"),
        ],
        "@{}llrrrl@{}",
    )
    emit(
        t,
        o,
        "tau_hashhome.md",
        "tau_hashhome",
        [
            ("Variant (lag build, lag 0)", "Variant of M1"),
            ("vs M1: segment mean [95% CI], all cells", "vs M1, all cells"),
            ("N=2", "$\\Nw = 2$"),
            ("N=16", "$\\Nw = 16$"),
            ("N=32", "$\\Nw = 32$"),
        ],
        "@{}lllll@{}",
        rename_rows={
            "m1-tau0.747": "$\\temp = 0.747$",
            "m1-tau2.75": "$\\temp = 2.75$",
            "m1-hashhome": "\\code{hash\\_home} pair only",
        },
    )
    emit(
        t,
        o,
        "isl_buckets.md",
        "isl_buckets",
        [
            ("ISL bucket (tokens)", "Prompt tokens"),
            (
                "Good fraction: M1-v2 − ramjet (pp)",
                "\\shortstack[r]{Good frac.\\\\M1-v2 $-$ ramjet (pp)}",
            ),
            ("segments higher", "\\shortstack[r]{Seg.\\\\higher}"),
            (
                "Good fraction: ramjet − default (pp)",
                "\\shortstack[r]{Good frac.\\\\ramjet $-$ $\\piref$ (pp)}",
            ),
            (
                "TTFT p90: M1-v2 vs ramjet (log-ratio)",
                "\\shortstack[r]{TTFT p90 log-ratio,\\\\M1-v2 vs ramjet}",
            ),
            ("segments higher#2", "\\shortstack[r]{Seg.\\\\higher}"),
        ],
        "@{}lrrrrr@{}",
        # Ranges take an en dash, as in the ISL figure; the open bucket reads "at least".
        rename_rows={
            "0-2048": "0--2048",
            "2048-8192": "2048--8192",
            "8192-32768": "8192--32768",
            "32768-65536": "32768--65536",
            "65536-": "$\\geq$\\,65536",
        },
        size="\\footnotesize",
        shrink=False,
    )
    emit(
        t,
        o,
        "lr13_flags.md",
        "lr13_flags",
        [
            ("Policy", "Policy"),
            (
                "Worker share above cap (cells of 60)",
                "\\shortstack[r]{Share\\\\$>$ cap}",
            ),
            ("NMI(worker; ISL quartile) > 0.10", "\\shortstack[r]{NMI\\\\$> 0.10$}"),
            ("Long-request sacrifice: ≥32K good < default − 0.05", "$\\geq$32K"),
            ("≥64K", "$\\geq$64K"),
            ("Top-ISL-decile", "\\shortstack[r]{Top\\\\decile}"),
            (
                "Cells with a dump worker",
                "\\shortstack[r]{Worker with\\\\no good request}",
            ),
            (
                "Mean of min per-worker good fraction",
                "\\shortstack[r]{Min worker\\\\good}",
            ),
        ],
        "@{}P{0.33\\linewidth}rrrrrrr@{}",
        size="\\footnotesize",
        shrink=False,
    )
    emit(
        t,
        o,
        "secondary_vs_ramjet.md",
        "secondary_vs_ramjet",
        [
            ("Policy", "Policy"),
            ("TTFT p50", "TTFT p50"),
            ("TTFT p90", "TTFT p90"),
            ("TTFT p90 32K-64K (n≥20)", "TTFT p90, 32--64K"),
            ("E2E p90 (in window)", "E2E p90"),
            ("mean-ITL p90", "ITL p90"),
            ("prefix reuse", "Prefix reuse"),
        ],
        "@{}P{0.27\\linewidth}llllll@{}",
        rows=[r for r in KEY_ROWS[:-1] if not r.startswith("ramjet")]
        + ["default@defaults (reference)"],
    )
    for side in ("ramjet", "default"):
        emit(
            t,
            o,
            f"N_vs_{side}.md",
            f"N_vs_{side}",
            [
                ("Policy", "Policy"),
                ("N=2 unseen N (8 cells, 6 seg)", "$\\Nw = 2$"),
                ("N=4 train N (13 cells, 7 seg)", "$\\Nw = 4$"),
                ("N=6 selection-exposed (7 cells, 6 seg)", "$\\Nw = 6$"),
                ("N=8 train N (17 cells, 8 seg)", "$\\Nw = 8$"),
                ("N=16 unseen N (7 cells, 6 seg)", "$\\Nw = 16$"),
                ("N=32 unseen N (8 cells, 6 seg)", "$\\Nw = 32$"),
            ],
            # Two stacked blocks (N = 2, 4, 6 over N = 8, 16, 32) keep the six intervals legible.
            "@{}P{0.25\\linewidth}lll@{}",
            rows=[
                r for r in KEY_ROWS if not (side == "ramjet" and r.startswith("ramjet"))
            ]
            + (["default@defaults (reference)"] if side == "ramjet" else []),
            size="\\scriptsize",
            stack=3,
            shrink=False,
        )
        emit(
            t,
            o,
            f"family_vs_{side}.md",
            f"family_vs_{side}",
            [
                ("Policy", "Policy"),
                ("mooncake (25 cells, 2 seg)", "Mooncake"),
                ("fast25_conversation (3 cells, 2 seg)", "FAST25 conv."),
                ("fast25_synthetic (3 cells, 2 seg)", "FAST25 synth."),
                ("synthetic_sessions (15 cells, 4 seg)", "Sessions"),
                ("agentx (14 cells, 4 seg)", "AgentX"),
            ],
            # Two stacked blocks (Mooncake and FAST25 over sessions and AgentX) keep the five
            # intervals legible.
            "@{}P{0.24\\linewidth}lll@{}",
            rows=[
                r for r in KEY_ROWS if not (side == "ramjet" and r.startswith("ramjet"))
            ]
            + (["default@defaults (reference)"] if side == "ramjet" else []),
            size="\\scriptsize",
            stack=3,
            shrink=False,
        )
        emit(
            t,
            o,
            f"mode_vs_{side}.md",
            f"mode_vs_{side}",
            [
                ("Policy", "Policy"),
                ("open_speedup (31 cells, 8 seg)", "Open loop (31 cells, 8 seg.)"),
                ("closed_concurrency (15 cells, 6 seg)", "Closed loop (15, 6)"),
                ("agentic_lanes (14 cells, 4 seg)", "Agentic lanes (14, 4)"),
            ],
            "@{}P{0.3\\linewidth}lll@{}",
            rows=[
                r for r in KEY_ROWS if not (side == "ramjet" and r.startswith("ramjet"))
            ]
            + (["default@defaults (reference)"] if side == "ramjet" else []),
            size="\\scriptsize",
        )
        emit(
            t,
            o,
            f"holdout_vs_{side}.md",
            f"holdout_vs_{side}",
            [
                ("Policy", "Policy"),
                ("unseen N 2/16/32 (23 cells)", "Unseen $\\Nw$ 2, 16, 32 (23 cells)"),
                ("seen N 4/8 (30 cells)", "Seen $\\Nw$ 4, 8 (30)"),
                ("transform extrapolation (10 cells)", "Transforms (10)"),
            ],
            "@{}P{0.3\\linewidth}lll@{}",
            rows=[
                r for r in KEY_ROWS if not (side == "ramjet" and r.startswith("ramjet"))
            ]
            + (["default@defaults (reference)"] if side == "ramjet" else []),
            size="\\scriptsize",
        )
    emit(
        t,
        o,
        "leaderboard.md",
        "leaderboard_full",
        [
            ("Policy", "Policy"),
            ("Selected config (val k0-2)", "Selected config"),
            ("Val k3-10", "Val $k_{3..10}$"),
            ("Test vs default: segment mean [95% CI]", "Test vs $\\piref$ [95\\% CI]"),
            ("Test vs ramjet: segment mean [95% CI]", "Test vs ramjet [95\\% CI]"),
            ("Segments ahead of ramjet", "Ahead"),
            ("One-sided Wilcoxon p vs ramjet", "$p$"),
        ],
        "@{}P{0.18\\linewidth}P{0.12\\linewidth}rllrr@{}",
        longtable_caption=(
            "Every test policy against $\\piref$ and against the validation-selected best baseline "
            "(tuned ramjet), 60 cells in 12 segments, segment means with 95\\% segment-bootstrap "
            "intervals. Selected config: restart, generation and candidate (best-so-far sample or CMA-ES mean) "
            "chosen on validation replicates $k_{0..2}$. Val $k_{3..10}$: mean clipped log-ratio against "
            "$\\piref$ on fresh validation replicates, the cross-policy selection score; policies at "
            "shipped defaults have none. "
            "The one-sided $p$ is a test only in the headline row; elsewhere it is descriptive."
        ),
        label="tab:leaderboard-full",
        size="\\tiny",
    )
    secondary_caption = (
        "Secondary metrics against ramjet on test, part {part} (descriptive, computed after the test "
        "pass from its records, no replay). Each entry is the segment-mean log-ratio, with the number "
        "of segments on which the policy's value is higher. Lower is better for latencies, higher for "
        "throughput and prefix reuse."
    )
    emit(
        t,
        o,
        "secondary_vs_ramjet.md",
        "secondary_full_a",
        [
            ("Policy", "Policy"),
            ("TTFT p50", "TTFT p50"),
            ("TTFT p90", "TTFT p90"),
            ("TTFT p99", "TTFT p99"),
            ("TTFT p90 32K-64K (n≥20)", "TTFT p90, 32--64K"),
            ("E2E p90 (in window)", "E2E p90 (in window)"),
        ],
        "@{}P{0.27\\linewidth}lllll@{}",
        longtable_caption=secondary_caption.format(part="1 (latency)"),
        label="tab:secondary-full",
        size="\\tiny",
    )
    emit(
        t,
        o,
        "secondary_vs_ramjet.md",
        "secondary_full_b",
        [
            ("Policy", "Policy"),
            ("mean-ITL p90", "Mean-ITL p90"),
            ("per-token ITL p99", "Per-token ITL p99"),
            ("output throughput", "Output throughput"),
            ("prefix reuse", "Prefix reuse"),
            ("AgentX trajectory mean", "AgentX trajectory"),
        ],
        "@{}P{0.27\\linewidth}lllll@{}",
        longtable_caption=secondary_caption.format(
            part="2 (decode, throughput, reuse)"
        ),
        label="tab:secondary-full-b",
        size="\\tiny",
    )

    live_tables(t, o)


LIVE_CELLS = [
    ("Mooncake open L2", "\\shortstack[l]{Mooncake\\\\open L2}"),
    ("Mooncake open L3", "\\shortstack[l]{Mooncake\\\\open L3}"),
    ("Mooncake closed L3", "\\shortstack[l]{Mooncake\\\\closed L3}"),
    ("FAST25 conv. open L2", "\\shortstack[l]{FAST25 conv.\\\\open L2}"),
    ("FAST25 synth. open L2", "\\shortstack[l]{FAST25 synth.\\\\open L2}"),
    ("Sessions open L2", "\\shortstack[l]{Sessions\\\\open L2}"),
]
LIVE_POLICIES = {
    "default@defaults": "$\\piref$",
    "ramjet (tuned)": "tuned \\policy{ramjet}",
    "M1-v2": "M1-v2",
    "M1": "M1",
    "M0": "M0",
    "round_robin": "round-robin",
}


def live_tables(t: Path, o: Path) -> None:
    """The live GPU validation tables (report section 15, fragment live_validation.md)."""
    src = "live_validation.md"
    emit(
        t,
        o,
        src,
        "live_goodput",
        [("Policy", "Policy"), *LIVE_CELLS],
        "@{}lllllll@{}",
        rename_rows=LIVE_POLICIES,
        table=0,
    )
    for k, (target, first) in enumerate(
        [
            ("live_vs_default", "Δ vs default@defaults"),
            ("live_vs_ramjet", "Δ vs ramjet"),
        ]
    ):
        emit(
            t,
            o,
            src,
            target,
            [
                (first, "Policy"),
                *LIVE_CELLS,
                ("Mean live [range]", "Mean live [range]"),
                ("Mean sim", "Mean sim"),
            ],
            "@{}lllllllll@{}",
            rename_rows=LIVE_POLICIES,
            table=1 + k,
            colsep="4pt",
        )
    pairs = {}
    for a, a_tex in LIVE_POLICIES.items():
        for b, b_tex in (
            ("default@defaults", "$\\piref$"),
            ("ramjet", "\\policy{ramjet}"),
        ):
            pairs[f"{a} vs {b}"] = f"{a_tex} vs {b_tex}"
    emit(
        t,
        o,
        src,
        "live_minus_sim",
        [("Live − sim", "Comparison"), *LIVE_CELLS],
        "@{}lllllll@{}",
        rename_rows=pairs,
        table=3,
    )
    emit(
        t,
        o,
        src,
        "live_tau",
        # Paragraph columns wrap these headers themselves.
        [("", ""), *[(name, name) for name, _ in LIVE_CELLS]],
        "@{}P{0.2\\linewidth}*{6}{P{0.115\\linewidth}}@{}",
        rename_rows={
            "τ_b, sim k = 0 (registered)": "$\\kendall$-b, sim $\\krep = 0$ (registered)",
            "τ_b, sim k0–2 mean": "$\\kendall$-b, sim mean of $\\krep = 0..2$",
            "τ_b, sim lag 10 ms": "$\\kendall$-b, sim lag \\SI{10}{ms}",
            "τ_b, sim lag 50 ms": "$\\kendall$-b, sim lag \\SI{50}{ms}",
            "Discordant pairs (k = 0)": "Discordant pairs ($\\krep = 0$)",
        },
        # The narrow paragraph columns would otherwise break "M1-v2" after its hyphen.
        cell_replace={"M1-v2": "M1\\nobreakdash-v2"},
        table=4,
    )
    emit(
        t,
        o,
        src,
        "live_sign",
        [
            ("Cell", "Cell"),
            ("Sim (k = 0)", "Sim ($\\krep = 0$)"),
            ("Live", "Live"),
            ("Same sign", "Same sign"),
            (
                "|live| > 2√2·σ_live",
                "$|\\text{live}| > 2\\sqrt{2}\\,\\sigma_{\\mathrm{live}}$",
            ),
        ],
        "@{}lrrll@{}",
        rename_rows={
            "Sessions open L2 (not informative: |sim| ≤ 0.01)": (
                "Sessions open L2 (not informative: $|\\text{sim}| \\leq 0.01$)"
            ),
            "FAST25 conv. open L2": "FAST25 conv.\\ open L2",
            "FAST25 synth. open L2": "FAST25 synth.\\ open L2",
        },
        table=5,
    )
    emit(
        t,
        o,
        src,
        "live_engine",
        [
            ("Policy", "Policy"),
            ("TTFT p50", "TTFT p50"),
            ("TTFT p90", "TTFT p90"),
            ("Mean-ITL p50", "Mean-ITL p50"),
            ("Mean-ITL p90", "Mean-ITL p90"),
            ("Max worker share live / sim", "Max worker share, live / sim"),
        ],
        "@{}lrrrrl@{}",
        rename_rows=LIVE_POLICIES,
        table=6,
    )


if __name__ == "__main__":
    main()
