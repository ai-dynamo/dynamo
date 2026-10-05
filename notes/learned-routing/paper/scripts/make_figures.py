#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Redraw the final report's figures as pgfplots code with reader-facing labels and the paper's fonts.

The campaign's report stage drew its figures as PNGs with internal labels (amendment numbers,
ablation tags, raw segment and feature identifiers) and small text. This script reads the numbers
the report plotted from ``<campaign root>/report/data/report_data.json`` and writes TikZ/pgfplots
pictures that the paper inputs inside its figure environments:

  generated/fig_leaderboard.tex        report F3   generated/fig_ladder.tex         report F4
  generated/fig_robustness.tex         report F5   generated/fig_coefficients.tex   report F6
  generated/fig_headline_segments.tex  report F1   generated/fig_val_vs_test.tex    report F7
  generated/fig_n_extrapolation.tex    report F2   generated/fig_profiles.tex       report F8
  generated/fig_cache_pressure.tex     report F9   generated/fig_isl.tex            report F10

It computes no statistic: every plotted value is a value from report_data.json, written with full
float precision. Two figures draw a function of those values exactly as the report did: the
performance profiles (F8) step through each policy's sorted per-cell scores with heights counted
over the 60 cells, and the prompt-length figure (F10) prints its good-fraction axis in percentage
points by labelling the ticks, not by rescaling the data. Policy names pass through the same RELABEL
map as the tables (scripts/make_tables.py). The data selection, order and colours follow
report/scripts/build_report_data.py; titles that repeated the captions are dropped.

Checks before writing (the script exits non-zero if any fails):
  * the leaderboard and robustness rows (order, names, groups) equal the report's own table
    fragments (leaderboard.md, robustness.md), and each plotted mean and interval rounds to the
    value printed there;
  * re-reading every written file recovers each coordinate exactly (value equality with the JSON),
    and the segment counts printed as tick labels (F2) are present verbatim.

Usage (stdlib only):
    python3 scripts/make_figures.py --cr <campaign root> --out generated
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from make_tables import read_md, relabel

# Display names and groups, copied from report/scripts/build_report_data.py (GROUPS, NAMES,
# GROUP_LABEL, COLOR); the checks below compare them with the report's leaderboard fragment.
GROUPS = {
    "learned": ["m1v2", "m1noaff", "ablam1", "m2r2", "m1", "m1cont", "m1u"],
    "tuned": [
        "ramjet",
        "ablabase",
        "llmdpp",
        "twotier",
        "stickybounded",
        "m0",
        "lmetric",
        "stickyhard",
        "llmdob",
        "dualmap",
        "chwbl",
    ],
    "ais": ["m2ais", "m1ais", "m0ais", "llmdobm"],
    "defaults": [
        "llm-d-precise-prefix@defaults",
        "lmetric@defaults",
        "two-tier@defaults",
        "ramjet@defaults",
        "llm-d-optimized-baseline@throughput",
        "sticky-bounded@defaults",
        "sticky-hard@defaults",
        "dualmap@defaults",
        "chwbl@defaults",
        "round_robin",
    ],
}
NAMES = {
    "m1v2": "M1-v2 (headline learned arm)",
    "m1noaff": "M1-noaff",
    "ablam1": "M1 + queue threshold (abl. a)",
    "m2r2": "M2 rank 2",
    "m1": "M1 (= A12 M1-default-init)",
    "m1cont": "M1 continued (secondary)",
    "m1u": "M1 unconstrained (gaming-flagged)",
    "ramjet": "ramjet (val-best baseline)",
    "ablabase": "ramjet + queue threshold (abl. a)",
    "llmdpp": "llm-d-precise-prefix",
    "twotier": "two-tier",
    "stickybounded": "sticky-session bounded",
    "m0": "M0 (default cost fn, tuned)",
    "lmetric": "lmetric (port)",
    "stickyhard": "sticky-session hard",
    "llmdob": "llm-d-optimized-baseline (throughput)",
    "dualmap": "dualmap",
    "chwbl": "chwbl",
    "m2ais": "M2-ais",
    "m1ais": "M1-ais",
    "m0ais": "M0-ais",
    "llmdobm": "llm-d-optimized-baseline (modeled)",
    "round_robin": "round_robin",
}
GROUP_LABEL = {
    "learned": "learned (router-observable)",
    "tuned": "tuned baseline",
    "ais": "AIS league",
    "defaults": "shipped defaults",
}
GROUP_TEX = {
    "learned": "learned (router-observable)",
    "tuned": "tuned baseline",
    "ais": "\\ais{}-informed league",
    "defaults": "shipped defaults",
}
COLOR = {
    "learned": "2A78D6",
    "tuned": "EB6834",
    "ais": "1BAF7A",
    "defaults": "4A3AA7",
    "muted": "898781",
    "neg": "E34948",
}
REF = "default"
HEAD, BASE = "m1v2", "ramjet"
# The dashed MDE line of F5 (facts/gate.json context_not_gate.mde_vs_best_heuristic, 0.03815...).
MDE_LINE = 0.03815
LAG_TIMING = ("lag", "s0", "s1", "d0", "d1")


def group_of(key: str) -> str:
    for group, keys in GROUPS.items():
        if key in keys:
            return group
    raise SystemExit(f"no group for {key}")


def fmt(x: float, nd: int = 4) -> str:
    """The report's signed fixed-point format (build_report_data.py fmt with sign=True)."""
    return f"{x:+.{nd}f}"


def fmt_ci(s: dict) -> str:
    return f"{fmt(s['segment_mean'])} [{fmt(s['ci95'][0])}, {fmt(s['ci95'][1])}]"


def esc(text: str) -> str:
    return (
        text.replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def num(x: float) -> str:
    """Full precision: float(num(x)) == x."""
    return repr(float(x))


def rows_leaderboard(data: dict, tables: Path) -> list[dict]:
    comp = data["comparisons"][REF]
    policies = [k for g in ("learned", "tuned", "ais", "defaults") for k in GROUPS[g]]
    if set(policies) != set(comp):
        raise SystemExit(
            f"policy set differs from report_data: {sorted(set(policies) ^ set(comp))}"
        )
    # F3's order: descending test cell mean against the default (stable over the GROUPS order).
    order = sorted(policies, key=lambda k: -comp[k]["all"]["cell_mean"])
    header, body = read_md(tables / "leaderboard.md")
    col = {name: i for i, name in enumerate(header)}
    if [r[0] for r in body] != [NAMES.get(k, k) for k in order]:
        raise SystemExit(
            "leaderboard: order or names differ from report/data/tables/leaderboard.md"
        )
    rows = []
    for k, r in zip(order, body):
        s = comp[k]["all"]
        if r[col["Group"]] != GROUP_LABEL[group_of(k)]:
            raise SystemExit(f"leaderboard: group of {k} differs from the fragment")
        if r[col["Test vs default: segment mean [95% CI]"]] != fmt_ci(s):
            raise SystemExit(f"leaderboard: {k} interval differs from the fragment")
        rows.append(
            {
                "key": k,
                "label": esc(relabel(NAMES.get(k, k))),
                "group": "muted" if k == "round_robin" else group_of(k),
                "mean": s["segment_mean"],
                "lo": s["ci95"][0],
                "hi": s["ci95"][1],
            }
        )
    return rows


def rows_robustness(data: dict, tables: Path) -> list[dict]:
    rob = data["robustness"]
    header, body = read_md(tables / "robustness.md")
    col = {name: i for i, name in enumerate(header)}
    if [r[0] for r in body] != list(rob):
        raise SystemExit(
            "robustness: conditions differ from report/data/tables/robustness.md"
        )
    rows = []
    for c, r in zip(rob, body):
        s = rob[c]
        if r[col["M1-v2 minus ramjet, segment mean [95% CI]"]] != fmt_ci(s):
            raise SystemExit(f"robustness: {c} interval differs from the fragment")
        p = s.get("wilcoxon_one_sided_p")
        if c == "nominal":
            group = "learned"
        elif c.startswith(LAG_TIMING):
            group = "tuned"
        elif c.startswith("A19"):
            group = "defaults"
        else:
            group = "ais"
        rows.append(
            {
                "key": c,
                "label": esc(relabel(c)),
                "group": group,
                "mean": s["segment_mean"],
                "lo": s["ci95"][0],
                "hi": s["ci95"][1],
                "p": p,
                "ptxt": ""
                if p is None
                else ("$p < 0.001$" if p < 0.001 else f"$p$ {p:.3f}"),
            }
        )
    return rows


def picture(
    rows: list[dict],
    *,
    xlabel: str,
    legend: list[tuple[str, str]],
    height: str,
    width: str,
    legend_pos: str,
    mde: bool,
    pvalues: bool,
    xmax: float | None = None,
) -> str:
    """One horizontal interval plot: a dot at the segment mean and a bar over the 95% CI per row.

    ``xmax`` fixes the right end of the axis (layout only) so the widest interval ends inside it.
    ``pvalues`` prints each row's one-sided p in a column just right of the axis, clear of the data
    and of the dashed MDE line.
    """
    ticks = ",".join(str(i) for i in range(len(rows)))
    labels = ",".join("{" + r["label"] + "}" for r in rows)
    out = [
        "% GENERATED by scripts/make_figures.py from report/data/report_data.json. Do not edit; regenerate.",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        f"  scale only axis, width={width}, height={height},",
        "  y dir=reverse, ymin=-0.7, ymax=" + f"{len(rows) - 0.3},",
        f"  ytick={{{ticks}}}, yticklabels={{{labels}}},",
        "  yticklabel style={font=\\scriptsize, text=black!75}, xticklabel style={font=\\scriptsize},",
        "  scaled x ticks=false, xticklabel style={/pgf/number format/fixed},",
        "  tick align=outside, tick style={black!40}, axis line style={black!35},",
        "  axis x line*=bottom, axis y line*=left,",
        "  xmajorgrids, ymajorgrids, grid style={black!12, line width=0.4pt},",
        "  enlarge x limits={abs=0.012},",
        *([f"  xmax={num(xmax)},"] if xmax is not None else []),
        f"  xlabel={{{xlabel}}}, xlabel style={{font=\\footnotesize, align=center}},",
        f"  legend style={{font=\\scriptsize, draw=none, fill=white, fill opacity=0.85, text opacity=1, {legend_pos}}},",
        "  legend cell align=left,",
        "  clip=false,",
        "]",
    ]
    for group, text in legend:
        out.append(
            f"\\addlegendimage{{color={{rgb,255:red,{int(COLOR[group][0:2], 16)};green,{int(COLOR[group][2:4], 16)};"
            f"blue,{int(COLOR[group][4:6], 16)}}}, line width=1.2pt, mark=*, mark size=1.6pt}}"
        )
        out.append(f"\\addlegendentry{{{text}}}")
    out.append(
        "\\draw[black!70, line width=0.6pt] (axis cs:0,-0.7) -- (axis cs:0,"
        + f"{len(rows) - 0.3});"
    )
    if mde:
        out.append(
            f"\\draw[black!45, line width=0.5pt, dashed] (axis cs:{num(MDE_LINE)},-0.7) -- (axis cs:{num(MDE_LINE)},"
            + f"{len(rows) - 0.3});"
        )
    for i, r in enumerate(rows):
        c = COLOR[r["group"]]
        rgb = f"{{rgb,255:red,{int(c[0:2], 16)};green,{int(c[2:4], 16)};blue,{int(c[4:6], 16)}}}"
        out.append(f"% row {i}: {r['key']}")
        out.append(
            f"\\addplot[color={rgb}, line width=1.2pt, forget plot] coordinates {{({num(r['lo'])},{i}) ({num(r['hi'])},{i})}};"
        )
        out.append(
            f"\\addplot[color={rgb}, only marks, mark=*, mark size=1.7pt, mark options={{draw=white, line width=0.5pt}}, forget plot] "
            f"coordinates {{({num(r['mean'])},{i})}};"
        )
        if pvalues and r["ptxt"]:
            out.append(
                f"\\node[anchor=west, font=\\scriptsize, text=black!70, xshift=4pt] at ({{rel axis cs:1,0}} |- {{axis cs:0,{i}}}) {{{r['ptxt']}}};"
            )
    out += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(out) + "\n"


COORD = re.compile(
    r"% row (\d+): (.+)\n\\addplot\[[^\n]*coordinates \{\(([^,]+),\1\) \(([^,]+),\1\)\};\n\\addplot\[[^\n]*coordinates \{\(([^,]+),\1\)\};"
)


def check_written(path: Path, rows: list[dict]) -> None:
    """Re-read the written picture and require every coordinate to equal the JSON value exactly."""
    found = COORD.findall(path.read_text(encoding="utf-8"))
    if len(found) != len(rows):
        raise SystemExit(
            f"{path.name}: {len(found)} rows written, {len(rows)} expected"
        )
    for (i, key, lo, hi, mean), r in zip(found, rows):
        if key != r["key"] or (float(lo), float(hi), float(mean)) != (
            r["lo"],
            r["hi"],
            r["mean"],
        ):
            raise SystemExit(
                f"{path.name}: row {i} ({key}) does not equal report_data.json"
            )


# ---------------------------------------------------------------------------------------------
# The report's other figures (F1, F2, F4, F6-F10), redrawn as pgfplots at the paper's font sizes.
# Every plotted value is written from report_data.json at full precision inside an
# "% data: <id>" block and re-read after writing; categorical positions (rows, rungs, buckets)
# and offsets are layout. No title repeats the caption; raw identifiers become paper names.
# ---------------------------------------------------------------------------------------------

HEADER = "% GENERATED by scripts/make_figures.py from report/data/report_data.json. Do not edit; regenerate."
NS = [2, 4, 6, 8, 16, 32]
TRAIN_NS = (4, 8)
# Common axis style: 8 pt ticks and legends, 9 pt axis labels (the paper is set at 11 pt).
AXIS = [
    "  scale only axis,",
    "  tick label style={font=\\scriptsize}, label style={font=\\footnotesize, align=center},",
    "  scaled ticks=false, tick label style={/pgf/number format/fixed},",
    "  tick align=outside, tick style={black!40}, axis line style={black!35},",
    "  axis x line*=bottom, axis y line*=left,",
    "  grid style={black!12, line width=0.4pt},",
    "  legend style={font=\\scriptsize, draw=none, fill=white, fill opacity=0.85, text opacity=1},",
    "  legend cell align=left,",
]
SEG_FAMILY = {
    "agentx": "AgentX",
    "fast25_synthetic": "FAST25 synth.",
    "mooncake": "Mooncake",
    "sessions": "sessions",
}
PRESSURE_FAMILY = [
    ("mooncake", "Mooncake", "*"),
    ("fast25_conversation", "FAST25 conversation", "square*"),
    ("fast25_synthetic", "FAST25 synthetic", "triangle*"),
]
SERIES_N = [
    ("m1v2", "M1-v2", "*", "solid"),
    ("ramjet", "tuned \\policy{ramjet}", "square*", "dashed"),
    ("llmdpp", "\\policy{llm-d-precise-prefix}", "triangle*", "densely dotted"),
    ("m1", "M1", "diamond*", "dashdotted"),
]
SERIES_COLOR = {"m1v2": "learned", "ramjet": "tuned", "llmdpp": "ais", "m1": "defaults"}
LADDER = [
    ("M0", "m0"),
    ("M1", "m1"),
    ("M1 cont.", "m1cont"),
    ("M2 r2", "m2r2"),
    ("M1-noaff", "m1noaff"),
    ("M1+q", "ablam1"),
    ("M1-v2", "m1v2"),
    ("M0-ais", "m0ais"),
    ("M1-ais", "m1ais"),
    ("M2-ais", "m2ais"),
]
VAL_TEST_LABELS = (
    "m1v2",
    "ramjet",
    "m1",
    "m2ais",
    "llmdpp",
    "chwbl",
    "dualmap",
    "llmdob",
    "stickyhard",
    "m0",
)
ISL_TICKS = ["$<$2K", "2--8K", "8--32K", "32--64K", "$\\geq$64K"]
# The window the report plotted its performance profiles over (np.linspace(-0.1, 0.6, 281)).
PROFILE_WINDOW = (-0.1, 0.6)


def rgb(color: str) -> str:
    c = COLOR.get(color, color)
    return f"{{rgb,255:red,{int(c[0:2], 16)};green,{int(c[2:4], 16)};blue,{int(c[4:6], 16)}}}"


def signed(x: float, nd: int) -> str:
    """The report's signed label format with a typographic minus."""
    return f"{x:+.{nd}f}".replace("-", "$-$").replace("+", "$+$")


class Picture:
    """A pgfplots picture whose data coordinates are re-read and compared after writing."""

    def __init__(self) -> None:
        self.lines = [HEADER, "\\begin{tikzpicture}"]
        self.expected: dict[str, list[tuple[float, float]]] = {}
        self.texts: list[str] = []

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def data(self, ident: str, options: str, points: list[tuple[float, float]]) -> None:
        if ident in self.expected:
            raise SystemExit(f"duplicate data id {ident}")
        self.expected[ident] = [(float(x), float(y)) for x, y in points]
        coords = " ".join(f"({num(x)},{num(y)})" for x, y in points)
        self.lines.append(f"% data: {ident}")
        self.lines.append(f"\\addplot[{options}] coordinates {{{coords}}};")

    def text(self, line: str) -> None:
        """A line that carries data as text (tick labels); checked verbatim after writing."""
        self.texts.append(line)
        self.lines.append(line)

    def write(self, path: Path) -> None:
        path.write_text(
            "\n".join(self.lines + ["\\end{tikzpicture}"]) + "\n", encoding="utf-8"
        )
        check_data(path, self.expected, self.texts)


DATA = re.compile(r"% data: (\S+)\n\\addplot\[[^\n]*\] coordinates \{([^\n]*)\};")
PAIR = re.compile(r"\(([^,()]+),([^,()]+)\)")


def check_data(
    path: Path, expected: dict[str, list[tuple[float, float]]], texts: list[str]
) -> None:
    """Every written coordinate equals the value taken from report_data.json, exactly."""
    text = path.read_text(encoding="utf-8")
    found = {
        ident: [(float(x), float(y)) for x, y in PAIR.findall(body)]
        for ident, body in DATA.findall(text)
    }
    if found != expected:
        bad = sorted(
            k for k in set(found) | set(expected) if found.get(k) != expected.get(k)
        )
        raise SystemExit(
            f"{path.name}: written data differ from report_data.json: {bad}"
        )
    lines = text.splitlines()
    for line in texts:
        if line not in lines:
            raise SystemExit(
                f"{path.name}: text line missing after writing: {line[:60]}"
            )


def band(x_lo: float, x_hi: float) -> str:
    """A full-height shaded band between two x positions (layout)."""
    return (
        f"\\fill[black!7] ({{axis cs:{num(x_lo)},0}} |- {{rel axis cs:0,0}}) rectangle "
        f"({{axis cs:{num(x_hi)},0}} |- {{rel axis cs:0,1}});"
    )


def hline(y: float, style: str) -> str:
    return f"\\draw[{style}] ({{rel axis cs:0,0}} |- {{axis cs:0,{num(y)}}}) -- ({{rel axis cs:1,0}} |- {{axis cs:0,{num(y)}}});"


def vline(x: float, style: str) -> str:
    return f"\\draw[{style}] ({{axis cs:{num(x)},0}} |- {{rel axis cs:0,0}}) -- ({{axis cs:{num(x)},0}} |- {{rel axis cs:0,1}});"


def fig_n_extrapolation(data: dict) -> Picture:
    """F2: against the reference by worker count (left); M1-v2 minus ramjet with 95% CIs (right)."""
    comp = data["comparisons"]
    pic = Picture()
    xaxis = [
        "  xmode=log, log basis x=2, xmin=1.65, xmax=38,",
        "  xtick={2,4,6,8,16,32}, xticklabels={2,4,6,8,16,32}, xminorticks=false,",
        "  xlabel={Worker count $\\Nw$ (log scale)},",
        "  ymajorgrids,",
    ]
    bands = [band(n * 2**-0.12, n * 2**0.12) for n in TRAIN_NS]
    pic.add(
        "\\begin{axis}[",
        "  name=nleft, width=0.34\\linewidth, height=5.2cm,",
        *AXIS,
        *xaxis,
        "  ylabel={Against $\\piref$: segment-mean\\\\clipped log-ratio},",
        "  legend style={at={(0.5,1.03)}, anchor=south, legend columns=2, /tikz/every even column/.append style={column sep=6pt}},",
        "]",
        *bands,
    )
    for key, label, mark, dash in SERIES_N:
        pts = [(n, comp[REF][key][f"N:{n}"]["segment_mean"]) for n in NS]
        pic.data(
            f"vs_default.{key}",
            f"color={rgb(SERIES_COLOR[key])}, {dash}, line width=1.1pt, mark={mark}, mark size=1.8pt, mark options={{solid}}",
            pts,
        )
        pic.add(f"\\addlegendentry{{{label}}}")
    pic.add("\\end{axis}")

    head = comp[BASE][HEAD]
    ahead = ",".join(
        f"{head[f'N:{n}']['segments_ahead']}/{head[f'N:{n}']['segments']}" for n in NS
    )
    geometry = [
        "  at={(nleft.south east)}, anchor=south west, xshift=2.1cm,",
        "  width=0.34\\linewidth, height=5.2cm, ymin=-0.012, ymax=0.12,",
    ]
    pic.add(
        "\\begin{axis}[",
        *geometry,
        *AXIS,
        *xaxis,
        "  ylabel={M1-v2 $-$ tuned \\policy{ramjet}:\\\\segment-mean clipped log-ratio},",
        "  ytick={0,0.03,0.06,0.09,0.12}, clip=false,",
        "]",
        *bands,
        hline(0.0, "black!70, line width=0.6pt"),
        hline(MDE_LINE, "black!45, line width=0.5pt, dashed"),
        f"\\node[anchor=west, font=\\scriptsize, text=black!55] at ({{rel axis cs:1,0}} |- {{axis cs:0,{num(MDE_LINE)}}}) {{MDE}};",
    )
    for n in NS:
        lo, hi = head[f"N:{n}"]["ci95"]
        pic.data(
            f"vs_ramjet.ci.N{n}",
            f"color={rgb('learned')}, line width=0.9pt, mark=-, mark size=2.5pt, forget plot",
            [(n, lo), (n, hi)],
        )
    pic.data(
        "vs_ramjet.mean",
        f"color={rgb('learned')}, line width=1.1pt, mark=*, mark size=1.8pt, mark options={{draw=white, line width=0.4pt}}",
        [(n, head[f"N:{n}"]["segment_mean"]) for n in NS],
    )
    pic.add("\\end{axis}")
    # The segments M1-v2 is ahead on, as a top axis over the right panel.
    pic.add(
        "\\begin{axis}[",
        *geometry,
        "  scale only axis, xmode=log, log basis x=2, xmin=1.65, xmax=38,",
        "  axis x line*=top, axis y line=none, tick align=outside, tick style={black!40},",
        "  axis line style={black!35}, xminorticks=false,",
        "  tick label style={font=\\scriptsize}, label style={font=\\footnotesize},",
        "  xtick={2,4,6,8,16,32},",
    )
    pic.text(f"  xticklabels={{{ahead}}},")
    pic.add(
        "  xlabel={Segments on which M1-v2 is ahead},",
        "]",
        "\\addplot[draw=none, forget plot] coordinates {(2,0)};",
        "\\end{axis}",
    )
    return pic


def fig_headline_segments(data: dict) -> Picture:
    """F1: M1-v2 minus ramjet per test segment, with the value printed beside each bar."""
    st = data["comparisons"][BASE][HEAD]["all"]
    segs = sorted(st["by_segment"], key=lambda s: (s.split(":")[0], s))
    labels = ",".join(
        "{" + f"{SEG_FAMILY[s.split(':')[0]]} {s.split(':')[1]}" + "}" for s in segs
    )
    pic = Picture()
    pic.add(
        "\\begin{axis}[",
        "  width=0.6\\linewidth, height=6.2cm,",
        *AXIS,
        "  y dir=reverse, ymin=-0.6, ymax=" + f"{len(segs) - 0.4},",
        "  ytick={" + ",".join(str(i) for i in range(len(segs))) + "},",
        f"  yticklabels={{{labels}}},",
        "  xmin=-0.004, xmax=0.112, xtick={0,0.02,0.04,0.06,0.08,0.1}, xmajorgrids,",
        "  xlabel={M1-v2 $-$ tuned \\policy{ramjet}: clipped log-ratio of windowed goodput\\\\"
        "(mean over the segment's cells, 3 CRN replicates each)},",
        "  clip=false,",
        "]",
    )
    for i, seg in enumerate(segs):
        v = st["by_segment"][seg]
        color = "learned" if v > 0 else "neg"
        pic.data(
            f"segment.{seg}",
            f"xbar, bar shift=0pt, bar width=0.6, fill={rgb(color)}, draw=none, forget plot",
            [(v, i)],
        )
        note = " (tie at ceiling)" if seg.startswith("sessions") else ""
        pic.add(
            f"\\node[anchor=west, font=\\scriptsize, text=black!70] at (axis cs:{num(max(v, 0.0) + 0.002)},{i}) {{{signed(v, 4)}{note}}};"
        )
    pic.add(vline(0.0, "black!70, line width=0.6pt"), "\\end{axis}")
    return pic


def fig_val_vs_test(data: dict) -> Picture:
    """F7: fresh-validation score against test, per policy; selected policies labelled in a column."""
    comp, val = data["comparisons"][REF], data["val_k3_10"]
    ymin, ymax, height_cm = -0.02, 0.2, 7.0
    pic = Picture()
    pic.add(
        "\\begin{axis}[",
        f"  width=0.5\\linewidth, height={height_cm}cm, xmin=0, xmax=0.2, ymin={ymin}, ymax={ymax},",
        *AXIS,
        "  xmajorgrids, ymajorgrids,",
        "  xlabel={Validation, fresh replicates $k_{3..10}$:\\\\mean over cells of the clipped log-ratio against $\\piref$},",
        "  ylabel={Test (single pass): mean over cells\\\\of the clipped log-ratio against $\\piref$},",
        "  legend style={at={(0.03,0.97)}, anchor=north west},",
        "  clip=false,",
        "]",
        "\\draw[black!45, line width=0.5pt, dashed] (axis cs:0,0) -- (axis cs:0.2,0.2);",
    )
    for group in ("learned", "tuned", "ais"):
        keys = [k for k in val if group_of(k) == group]
        pic.data(
            f"points.{group}",
            f"only marks, mark=*, mark size=2pt, color={rgb(group)}, mark options={{draw=white, line width=0.4pt}}",
            [(val[k], comp[k]["all"]["cell_mean"]) for k in keys],
        )
        pic.add(f"\\addlegendentry{{{GROUP_TEX[group]}}}")
    # Labels in a column right of the axis, at least one label height apart, with leader lines.
    sep = (
        (ymax - ymin) * (9.5 / 28.4527) / height_cm
    )  # 9.5 pt in data units (28.45 pt per cm)
    placed, last = [], None
    for k in sorted(VAL_TEST_LABELS, key=lambda k: -comp[k]["all"]["cell_mean"]):
        y = comp[k]["all"]["cell_mean"]
        ly = y if last is None else min(y, last - sep)
        placed.append((k, y, ly))
        last = ly
    for k, y, ly in placed:
        label = esc(relabel(NAMES[k])).split(" (")[0]
        pic.add(
            f"\\draw[black!35, line width=0.3pt] (axis cs:{num(val[k])},{num(y)}) -- ({{rel axis cs:1,0}} |- {{axis cs:0,{num(ly)}}})"
            f" node[anchor=west, xshift=1pt, font=\\scriptsize, text=black!75, inner sep=1pt] {{{label}}};"
        )
    pic.add("\\end{axis}")
    return pic


def fig_cache_pressure(data: dict) -> Picture:
    """F9: per-cell gain against cache pressure, against the reference (left) and ramjet (right)."""
    cells = data["cache_pressure"]["cells"]
    pic = Picture()
    panels = [
        (
            "m1v2_vs_default",
            "M1-v2 against $\\piref$:\\\\per-cell clipped log-ratio",
            "name=cpleft,",
        ),
        (
            "m1v2_vs_ramjet",
            "M1-v2 against tuned \\policy{ramjet}:\\\\per-cell clipped log-ratio",
            "at={(cpleft.south east)}, anchor=south west, xshift=2.1cm,",
        ),
    ]
    for key, ylabel, place in panels:
        pic.add(
            "\\begin{axis}[",
            f"  {place} width=0.36\\linewidth, height=5cm,",
            *AXIS,
            "  xmajorgrids, ymajorgrids,",
            "  xlabel={Cache pressure: distinct prefix tokens\\\\per \\SI{100}{s} over cluster KV tokens},",
            f"  ylabel={{{ylabel}}},",
            "  legend style={at={(0.03,0.97)}, anchor=north west},",
            "]",
        )
        for fam, name, mark in PRESSURE_FAMILY:
            pts = [(c["pressure"], c[key]) for c in cells if c["family"] == fam]
            pic.data(
                f"{key}.{fam}",
                f"only marks, mark={mark}, mark size=2pt, color={rgb('learned')}, mark options={{draw=white, line width=0.3pt}}",
                pts,
            )
            if key == "m1v2_vs_ramjet":
                pic.add(f"\\addlegendentry{{{name}}}")
        pic.add(hline(0.0, "black!70, line width=0.6pt"), "\\end{axis}")
    fams = {c["family"] for c in cells}
    if fams != {f for f, _, _ in PRESSURE_FAMILY}:
        raise SystemExit(f"cache pressure families differ: {sorted(fams)}")
    return pic


def fig_ladder(data: dict) -> Picture:
    """F4: each rung on the selection replicates, the fresh validation replicates and test."""
    comp, v02, v310 = data["comparisons"][REF], data["val_k0_2"], data["val_k3_10"]
    pic = Picture()
    ticks = ",".join("{" + lab + "}" for lab, _ in LADDER)
    pic.add(
        "\\begin{axis}[",
        "  width=0.82\\linewidth, height=5.2cm,",
        *AXIS,
        f"  xmin=-0.6, xmax={len(LADDER) - 0.4}, xtick={{{','.join(str(i) for i in range(len(LADDER)))}}}, xticklabels={{{ticks}}},",
        "  ymajorgrids, ylabel={Mean over cells of the\\\\clipped log-ratio against $\\piref$},",
        "  legend style={at={(0.5,1.03)}, anchor=south, legend columns=3, /tikz/every even column/.append style={column sep=8pt}},",
        "]",
    )
    series = [
        ("selection", v02, -0.18, "*", "learned", "validation $k_{0..2}$ (selection)"),
        ("fresh", v310, 0.0, "square*", "tuned", "validation $k_{3..10}$ (fresh)"),
        (
            "test",
            {k: comp[k]["all"]["cell_mean"] for _, k in LADDER},
            0.18,
            "diamond*",
            "ais",
            "test (single pass)",
        ),
    ]
    size = {"*": "2.2pt", "square*": "2pt", "diamond*": "2.8pt"}
    for ident, values, off, mark, color, label in series:
        pic.data(
            f"rungs.{ident}",
            f"only marks, mark={mark}, mark size={size[mark]}, color={rgb(color)}, mark options={{draw=white, line width=0.4pt}}",
            [(i + off, values[k]) for i, (_, k) in enumerate(LADDER)],
        )
        pic.add(f"\\addlegendentry{{{label}}}")
    refs = [
        (
            "selection",
            v02["ramjet"],
            "densely dotted",
            "tuned \\policy{ramjet}: selection",
        ),
        ("fresh", v310["ramjet"], "dashed", "tuned \\policy{ramjet}: fresh"),
        (
            "test",
            comp["ramjet"]["all"]["cell_mean"],
            "solid",
            "tuned \\policy{ramjet}: test",
        ),
    ]
    for ident, y, dash, label in refs:
        pic.data(
            f"ramjet.{ident}",
            f"color={rgb('muted')}, {dash}, line width=0.8pt, no marks",
            [(-0.6, y), (len(LADDER) - 0.4, y)],
        )
        pic.add(f"\\addlegendentry{{{label}}}")
    pic.add(vline(6.5, "black!25, line width=0.8pt"), "\\end{axis}")
    return pic


def fig_coefficients(data: dict) -> Picture:
    """F6: M1-v2's nonzero coefficients times their within-set sd, with the default-cost anchor."""
    coef = data["coefficients_m1v2"]
    anchor = next(c for c in coef if c["index"] == 0)
    rows = sorted(
        (c for c in coef if c["theta"] != 0 and c["index"] != 0),
        key=lambda c: c["theta_x_sd"],
    )
    labels = ",".join(
        "{" + f"{c['index']}\\enspace\\code{{{esc(c['feature'])}}}" + "}" for c in rows
    )
    pic = Picture()
    pic.add(
        "\\begin{axis}[",
        "  width=0.5\\linewidth, height=6.4cm,",
        *AXIS,
        f"  ymin=-0.6, ymax={len(rows) - 0.4}, ytick={{{','.join(str(i) for i in range(len(rows)))}}},",
        f"  yticklabels={{{labels}}},",
        "  xmin=-7, xmax=1.6, xmajorgrids,",
        "  xlabel={$\\theta_f\\,\\mathrm{sd}_f$: utility change per within-set sd of feature $f$},",
        "  clip=false,",
        "]",
    )
    for i, c in enumerate(rows):
        color = "learned" if c["theta_x_sd"] > 0 else "neg"
        pic.data(
            f"coef.{c['index']}",
            f"xbar, bar shift=0pt, bar width=0.6, fill={rgb(color)}, draw=none, forget plot",
            [(c["theta_x_sd"], i)],
        )
    pic.data(
        "anchor",
        f"color={rgb('muted')}, dashed, line width=0.6pt, no marks, forget plot",
        [(anchor["theta_x_sd"], -0.6), (anchor["theta_x_sd"], len(rows) - 0.4)],
    )
    pic.add(
        f"\\node[anchor=north west, font=\\scriptsize, text=black!70, align=left] at (axis cs:{num(anchor['theta_x_sd'])},{len(rows) - 0.4})"
        f" {{default-cost anchor\\\\$\\theta_0\\,\\mathrm{{sd}}_0 = {anchor['theta_x_sd']:.2f}$}};",
        vline(0.0, "black!70, line width=0.6pt"),
        "\\end{axis}",
    )
    return pic


def fig_isl(data: dict) -> Picture:
    """F10: good-fraction differences (left, in percentage points) and TTFT p90 (right) by prompt length."""
    isl = data["isl_buckets"]
    buckets = list(isl)
    if len(buckets) != len(ISL_TICKS):
        raise SystemExit(f"ISL buckets differ: {buckets}")
    ticks = ",".join("{" + t + "}" for t in ISL_TICKS)
    common = [
        "  width=0.36\\linewidth, height=5cm,",
        *AXIS,
        f"  xmin=-0.6, xmax={len(buckets) - 0.4}, xtick={{{','.join(str(i) for i in range(len(buckets)))}}}, xticklabels={{{ticks}}},",
        "  xlabel={Prompt length (tokens)}, ymajorgrids,",
    ]
    pic = Picture()
    pic.add(
        "\\begin{axis}[",
        "  name=islleft,",
        *common,
        # Data are fractions; the tick labels print them in percentage points.
        "  yticklabel={\\pgfmathparse{100*\\tick}\\pgfmathprintnumber[fixed, precision=0]{\\pgfmathresult}},",
        "  ylabel={In-window good fraction,\\\\difference (percentage points)},",
        "  legend style={at={(0.5,1.03)}, anchor=south, legend columns=2, /tikz/every even column/.append style={column sep=8pt}},",
        "]",
    )
    for ident, key, off, color, label in (
        (
            "ramjet_default",
            "good_ramjet_default",
            -0.2,
            "tuned",
            "tuned \\policy{ramjet} $-$ $\\piref$",
        ),
        (
            "m1v2_ramjet",
            "good_m1v2_ramjet",
            0.2,
            "learned",
            "M1-v2 $-$ tuned \\policy{ramjet}",
        ),
    ):
        pic.data(
            f"good.{ident}",
            f"ybar, bar shift=0pt, bar width=0.38, fill={rgb(color)}, draw=none, area legend",
            [(i + off, isl[b][key]["segment_mean"]) for i, b in enumerate(buckets)],
        )
        pic.add(f"\\addlegendentry{{{label}}}")
    pic.add(hline(0.0, "black!70, line width=0.6pt"), "\\end{axis}")
    pic.add(
        "\\begin{axis}[",
        "  at={(islleft.south east)}, anchor=south west, xshift=2.1cm,",
        *common,
        "  ylabel={TTFT p90 log-ratio, M1-v2 against\\\\tuned \\policy{ramjet} ($>0$: M1-v2 slower)},",
        "]",
    )
    pic.data(
        "ttft.m1v2_ramjet",
        f"ybar, bar shift=0pt, bar width=0.55, fill={rgb('learned')}, draw=none, forget plot",
        [
            (i, isl[b]["ttft_m1v2_ramjet"]["segment_mean"])
            for i, b in enumerate(buckets)
        ],
    )
    pic.add(hline(0.0, "black!70, line width=0.6pt"), "\\end{axis}")
    return pic


def fig_profiles(data: dict) -> Picture:
    """F8: the fraction of the 60 test cells whose score against the reference exceeds tau.

    Each curve is the step function of the sorted per-cell scores (the report evaluated the same
    function on a grid); its x coordinates are the report's per-cell values, its heights counts/60.
    """
    prof = data["performance_profiles"]
    lo, hi = PROFILE_WINDOW
    pic = Picture()
    pic.add(
        "\\begin{axis}[",
        "  width=0.5\\linewidth, height=5.2cm,",
        *AXIS,
        f"  xmin={num(lo)}, xmax={num(hi)}, ymin=0, ymax=1.02, xmajorgrids, ymajorgrids,",
        "  xlabel={$\\tau$: per-cell clipped log-ratio against $\\piref$},",
        "  ylabel={Fraction of the 60 test cells\\\\with a score above $\\tau$},",
        "  legend style={at={(0.97,0.97)}, anchor=north east},",
        "]",
    )
    for key, label, _, dash in SERIES_N:
        v = sorted(prof[key])
        n = len(v)
        if n != 60:
            raise SystemExit(f"performance profile {key}: {n} cells, expected 60")
        pts = [(lo, 1.0)] + [(x, (n - i - 1) / n) for i, x in enumerate(v)]
        pic.data(
            f"profile.{key}",
            f"const plot, color={rgb(SERIES_COLOR[key])}, {dash}, line width=1.1pt, no marks",
            pts,
        )
        pic.add(f"\\addlegendentry{{{label}}}")
    pic.add(vline(0.0, "black!70, line width=0.6pt"), "\\end{axis}")
    return pic


FIGURES = {
    "fig_n_extrapolation": fig_n_extrapolation,
    "fig_headline_segments": fig_headline_segments,
    "fig_val_vs_test": fig_val_vs_test,
    "fig_cache_pressure": fig_cache_pressure,
    "fig_ladder": fig_ladder,
    "fig_coefficients": fig_coefficients,
    "fig_isl": fig_isl,
    "fig_profiles": fig_profiles,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cr", type=Path, required=True, help="campaign root")
    parser.add_argument(
        "--out", type=Path, required=True, help="paper generated/ directory"
    )
    args = parser.parse_args()
    data = json.loads(
        (args.cr / "report/data/report_data.json").read_text(encoding="utf-8")
    )
    tables = args.cr / "report/data/tables"
    gate = json.loads((args.cr / "facts/gate.json").read_text(encoding="utf-8"))
    if round(gate["context_not_gate"]["mde_vs_best_heuristic"], 5) != MDE_LINE:
        raise SystemExit("MDE line differs from facts/gate.json")

    lb = rows_leaderboard(data, tables)
    path = args.out / "fig_leaderboard.tex"
    path.write_text(
        picture(
            lb,
            xlabel="Test: segment-mean clipped log-ratio against $\\piref$\\\\(95\\% segment-bootstrap CI)",
            legend=[(g, GROUP_TEX[g]) for g in ("learned", "tuned", "ais", "defaults")],
            height="11.5cm",
            width="0.5\\linewidth",
            legend_pos="at={(0.5,1.02)}, anchor=south, legend columns=2, /tikz/every even column/.append style={column sep=8pt}",
            mde=False,
            pvalues=False,
            # Room to the right of the widest interval (M1-v2's, upper end 0.2218).
            xmax=0.3,
        ),
        encoding="utf-8",
    )
    check_written(path, lb)

    rb = rows_robustness(data, tables)
    path = args.out / "fig_robustness.tex"
    path.write_text(
        picture(
            rb,
            xlabel="M1-v2 minus ramjet on test, segment-mean clipped log-ratio\\\\(95\\% CI; dashed: the MDE)",
            legend=[
                ("learned", "nominal (pre-registered)"),
                ("tuned", "lag or timing perturbation (replayed)"),
                ("ais", "SLO definition (rescored, no replay)"),
                ("defaults", "token-weighted goodput (secondary)"),
            ],
            height="9.5cm",
            width="0.55\\linewidth",
            legend_pos="at={(0.5,1.02)}, anchor=south, legend columns=2, /tikz/every even column/.append style={column sep=8pt}",
            mde=True,
            pvalues=True,
        ),
        encoding="utf-8",
    )
    check_written(path, rb)
    for name, draw in FIGURES.items():
        draw(data).write(args.out / f"{name}.tex")
    print(
        f"wrote fig_leaderboard.tex ({len(lb)} rows), fig_robustness.tex ({len(rb)} rows) and "
        f"{', '.join(f'{n}.tex' for n in FIGURES)}; values equal report_data.json"
    )


if __name__ == "__main__":
    main()
