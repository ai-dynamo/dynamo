#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Dynamo Dark figure linter and design scorer.

Statically checks the figure-generator sources of the blog-figures skill
(Python generators + HTML/CSS templates) against the canonical design
tokens, and computes a 0-100 design score from the mechanically measurable
signals.

Checks (see RATINGS.md for the rubric):
  - raw-hex        (ERROR): a color literal not present in design_tokens.yaml
                            and not an annotated override (`# lint-allow-hex`).
  - forbidden-font (ERROR): a font family outside the Dynamo Dark set
                            (Geist, Inter, a display serif, Comic Sans, ...).
  - unknown-font   (WARN):  a family that is neither allowed nor explicitly
                            forbidden.
  - contrast       (WARN):  a token text-on-surface pair below WCAG AA 4.5:1.
  - font-weight    (INFO):  a weight outside {300,400,500,600,700}.

The scorer reports ONLY what a static raster pipeline can measure honestly
(palette compliance, typography compliance, contrast pass-rate, palette
variety, structural label presence). Dimensions that need eyes on the
rendered pixels (true data-ink, single-accent semantics, composition,
before/after honesty) are NOT scored here; they stay in RATINGS.md as the
manual LLM-critique checklist.

Usage:
    python3 lint_figures.py [PATH ...]        # lint (default: ../examples)
    python3 lint_figures.py --score           # + 0-100 score breakdown
    python3 lint_figures.py --json            # machine-readable output
    python3 lint_figures.py --strict          # also fail on WARN

Exit code is non-zero when any ERROR is found (or any WARN under --strict),
so it can gate a build.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET = SCRIPT_DIR.parent / "examples"
DEFAULT_TOKENS = SCRIPT_DIR.parent / "examples" / "design_tokens.yaml"

SCANNED_SUFFIXES = {".py", ".html", ".css"}
SEVERITY_RANK = {"ERROR": 3, "WARN": 2, "INFO": 1}

# Generic CSS keywords and non-color families that are always acceptable.
GENERIC_FAMILIES = {"sans-serif", "serif", "monospace", "system-ui", "ui-monospace"}
# The documented display title font is not in the machine tokens (which cover
# the compact/body scale); it is part of the one Dynamo Dark family.
EXTRA_ALLOWED_FAMILIES = {"helvetica neue"}
# Families explicitly outside Dynamo Dark (fail hard).
FORBIDDEN_FAMILIES = {
    "geist",
    "geist mono",
    "inter",
    "comic sans",
    "comic sans ms",
    "impact",
    "georgia",
    "times",
    "times new roman",
    "iowan old style",
    "iowan",
    "courier",
    "courier new",
}

HEX_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b")
PY_FAMILY_RE = re.compile(r"""family\s*=\s*["']([^"']+)["']""")
CSS_FAMILY_RE = re.compile(r"""(?:font-family|--font-[\w-]+)\s*:\s*([^;]+);""")
WEIGHT_RE = re.compile(r"""weight\s*=\s*["']?(\d{3})["']?""")
VALID_WEIGHTS = {"300", "400", "500", "600", "700"}

# Token text-on-surface pairs the aesthetic actually uses. "#..." literals are
# taken as-is; dotted names resolve through the tokens' colors tree.
CONTRAST_PAIRS = [
    ("text.primary", "background.primary", "body text on ground"),
    ("text.secondary", "background.primary", "secondary text on ground"),
    ("text.medium", "background.primary", "axis / tick text on ground"),
    ("text.muted", "background.primary", "muted sub-meta on ground"),
    ("text.primary", "background.surface", "text on card surface"),
    ("text.secondary", "background.surface", "secondary text on card"),
    ("#000000", "accent.dynamo_green", "in-bar label on green fill"),
]
WCAG_AA = 4.5


@dataclass
class LintMessage:
    """A single finding, mirroring the dsvg LintMessage shape."""

    severity: str
    rule_id: str
    path: str
    line: int
    message: str


# ─── Color parsing + WCAG contrast (ported, figure-agnostic) ─────────────────


def parse_color(value: str) -> tuple[int, int, int] | None:
    """Parse a hex or basic named color to an (r, g, b) tuple in 0-255."""
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if value == "white":
        return (255, 255, 255)
    if value == "black":
        return (0, 0, 0)
    if value.startswith("#"):
        h = value.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) == 6 and all(c in "0123456789abcdef" for c in h):
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    return None


def relative_luminance(r: float, g: float, b: float) -> float:
    """WCAG 2.1 relative luminance from sRGB values (0-255)."""

    def linearize(v: float) -> float:
        v = v / 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


def wcag_contrast_ratio(fg: str, bg: str) -> float | None:
    """WCAG contrast ratio between two colors, or None if unparseable."""
    fg_rgb = parse_color(fg)
    bg_rgb = parse_color(bg)
    if fg_rgb is None or bg_rgb is None:
        return None
    l1 = relative_luminance(*fg_rgb)
    l2 = relative_luminance(*bg_rgb)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


# ─── Token loading ───────────────────────────────────────────────────────────


def load_tokens(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _normalize_hex(value: str) -> str | None:
    m = HEX_RE.fullmatch(value.strip())
    if not m:
        return None
    h = value.strip().lower().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return f"#{h}" if len(h) in (6, 8) else None


def collect_token_hexes(node: Any, acc: set[str]) -> set[str]:
    """Recursively collect every hex string value in the tokens tree."""
    if isinstance(node, str):
        norm = _normalize_hex(node)
        if norm:
            acc.add(norm)
    elif isinstance(node, dict):
        for v in node.values():
            collect_token_hexes(v, acc)
    elif isinstance(node, list):
        for v in node:
            collect_token_hexes(v, acc)
    return acc


def collect_neutral_hexes(tokens: dict[str, Any]) -> set[str]:
    """Structural neutrals (backgrounds, text, borders, greys), not semantic
    accents. Used so per-figure accent variety isn't inflated by the neutral
    palette (a token :root declaration is not four 'accents')."""
    colors = tokens.get("colors", {})
    acc: set[str] = set()
    for group in ("background", "text"):
        for v in (colors.get(group) or {}).values():
            n = _normalize_hex(v) if isinstance(v, str) else None
            if n:
                acc.add(n)
    subtle = (colors.get("border") or {}).get("subtle")
    if isinstance(subtle, str) and (n := _normalize_hex(subtle)):
        acc.add(n)
    for key in ("rich_black", "white"):
        v = (colors.get("brand") or {}).get(key)
        if isinstance(v, str) and (n := _normalize_hex(v)):
            acc.add(n)
    acc.add("#555555")  # grey chart fill for non-accent bars
    return acc


def allowed_font_families(tokens: dict[str, Any]) -> set[str]:
    typo = tokens.get("typography", {})
    families: set[str] = set(GENERIC_FAMILIES) | set(EXTRA_ALLOWED_FAMILIES)
    for key in ("font_family", "font_family_mono"):
        stack = typo.get(key, "")
        for part in stack.split(","):
            name = part.strip().strip("'\"").lower()
            if name:
                families.add(name)
    return families


def resolve_token_color(tokens: dict[str, Any], ref: str) -> str | None:
    """Resolve a literal hex or a dotted token path (e.g. text.primary)."""
    if ref.startswith("#"):
        return ref
    node: Any = tokens.get("colors", {})
    for part in ref.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, str) else None


# ─── Source scanning ─────────────────────────────────────────────────────────


def _iter_families(field_value: str) -> list[str]:
    return [p.strip().strip("'\"").lower() for p in field_value.split(",") if p.strip()]


def scan_file(
    path: Path,
    token_hexes: set[str],
    allowed_fonts: set[str],
    neutral_hexes: set[str],
) -> tuple[list[LintMessage], dict[str, Any]]:
    """Scan one source file; return findings plus per-file stats for scoring."""
    messages: list[LintMessage] = []
    stats = {
        "hex_total": 0,
        "hex_ok": 0,
        "family_total": 0,
        "family_ok": 0,
        "accent_hexes": set(),
    }
    text = path.read_text(encoding="utf-8")
    rel = str(path)

    for lineno, line in enumerate(text.splitlines(), 1):
        allow_hex = "lint-allow-hex" in line
        for m in HEX_RE.finditer(line):
            norm = _normalize_hex(m.group())
            if norm is None:
                continue
            stats["hex_total"] += 1
            if norm in token_hexes or allow_hex:
                stats["hex_ok"] += 1
                if norm not in neutral_hexes:
                    stats["accent_hexes"].add(norm)
            else:
                messages.append(
                    LintMessage(
                        "ERROR",
                        "raw-hex",
                        rel,
                        lineno,
                        f"color {m.group()!r} is not a design token "
                        f"(annotate with `# lint-allow-hex` if it is a "
                        f"documented per-script override)",
                    )
                )

        family_fields: list[str] = []
        if path.suffix == ".py":
            family_fields += PY_FAMILY_RE.findall(line)
        if path.suffix in (".html", ".css"):
            family_fields += CSS_FAMILY_RE.findall(line)
        for field_value in family_fields:
            for fam in _iter_families(field_value):
                if fam.startswith("var("):
                    continue  # CSS custom property; resolves to a token stack
                stats["family_total"] += 1
                if fam in FORBIDDEN_FAMILIES:
                    messages.append(
                        LintMessage(
                            "ERROR",
                            "forbidden-font",
                            rel,
                            lineno,
                            f"font family {fam!r} is outside the Dynamo Dark "
                            f"set (Helvetica Neue / Helvetica / Arial + Roboto "
                            f"Mono); no second family",
                        )
                    )
                elif fam in allowed_fonts:
                    stats["family_ok"] += 1
                else:
                    messages.append(
                        LintMessage(
                            "WARN",
                            "unknown-font",
                            rel,
                            lineno,
                            f"font family {fam!r} is not a known Dynamo Dark "
                            f"family; use the token sans/mono stacks",
                        )
                    )

        for w in WEIGHT_RE.findall(line):
            if w not in VALID_WEIGHTS:
                messages.append(
                    LintMessage(
                        "INFO",
                        "font-weight",
                        rel,
                        lineno,
                        f"font weight {w} is outside the usual "
                        f"{sorted(VALID_WEIGHTS)} steps",
                    )
                )

    return messages, stats


def check_contrast(tokens: dict[str, Any]) -> tuple[list[LintMessage], list[dict]]:
    """Validate the token text-on-surface pairs against WCAG AA."""
    messages: list[LintMessage] = []
    results: list[dict] = []
    for fg_ref, bg_ref, label in CONTRAST_PAIRS:
        fg = resolve_token_color(tokens, fg_ref)
        bg = resolve_token_color(tokens, bg_ref)
        ratio = wcag_contrast_ratio(fg, bg) if fg and bg else None
        passed = ratio is not None and ratio >= WCAG_AA
        results.append(
            {"pair": label, "fg": fg, "bg": bg, "ratio": ratio, "pass": passed}
        )
        if ratio is not None and not passed:
            messages.append(
                LintMessage(
                    "WARN",
                    "contrast",
                    "design_tokens.yaml",
                    0,
                    f"{label}: {fg} on {bg} is {ratio:.1f}:1 "
                    f"(WCAG AA requires >= {WCAG_AA}:1)",
                )
            )
    return messages, results


# ─── Scoring ─────────────────────────────────────────────────────────────────


def _pct(ok: int, total: int) -> float:
    return 100.0 if total == 0 else round(100.0 * ok / total, 1)


def compute_score(
    file_stats: dict[str, dict], contrast_results: list[dict], scripts: list[Path]
) -> dict[str, Any]:
    """Compute the measured 0-100 score. Judged dims are NOT included here."""
    hex_total = sum(s["hex_total"] for s in file_stats.values())
    hex_ok = sum(s["hex_ok"] for s in file_stats.values())
    fam_total = sum(s["family_total"] for s in file_stats.values())
    fam_ok = sum(s["family_ok"] for s in file_stats.values())

    palette = _pct(hex_ok, hex_total)
    typography = _pct(fam_ok, fam_total)
    contrast = _pct(
        sum(1 for r in contrast_results if r["pass"]), len(contrast_results)
    )

    # Variety: distinct token colors referenced per generator. A single chart
    # overloaded with >4 accents loses points; diagrams legitimately use more
    # role colors, so this is a light signal, not a hard gate.
    max_colors = max((len(s["accent_hexes"]) for s in file_stats.values()), default=0)
    variety = 100.0 if max_colors <= 4 else max(60.0, 100.0 - (max_colors - 4) * 10)

    # Structural label presence (proxy for label coverage): does each figure
    # generator declare a title AND labels/axes? Scripts that render an HTML
    # template carry the title in the HTML, so a `.html` reference counts.
    gen_scripts = [
        p for p in scripts if p.name.startswith("gen_") and p.suffix == ".py"
    ]
    labelled = 0
    for p in gen_scripts:
        src = p.read_text(encoding="utf-8")
        has_title = "title=" in src or "<h1>" in src or ".html" in src
        has_labels = any(
            k in src
            for k in (
                "title_text",
                "xaxis",
                "annotation",
                "add_annotation",
                "label",
                "text=",
                ".html",
            )
        )
        if has_title and has_labels:
            labelled += 1
    label_structure = _pct(labelled, len(gen_scripts))

    weights = {
        "palette_compliance": 30,
        "typography_compliance": 25,
        "contrast": 25,
        "variety": 10,
        "label_structure": 10,
    }
    dims = {
        "palette_compliance": palette,
        "typography_compliance": typography,
        "contrast": contrast,
        "variety": variety,
        "label_structure": label_structure,
    }
    overall = round(sum(dims[k] * weights[k] / 100 for k in weights), 1)
    return {
        "overall_measured": overall,
        "ship_threshold": 85,
        "ship_ready_measured": overall >= 85,
        "dimensions": {k: {"score": dims[k], "weight": weights[k]} for k in weights},
        "max_colors_in_one_figure": max_colors,
        "judged_dimensions": [
            "data_ink_discipline",
            "single_accent_semantics",
            "composition_and_geometry",
            "before_after_honesty",
            "title_carries_takeaway",
        ],
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────


def gather_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.suffix in SCANNED_SUFFIXES]
        elif p.suffix in SCANNED_SUFFIXES:
            files.append(p)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="files/dirs to lint")
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--score", action="store_true", help="print 0-100 score")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--strict", action="store_true", help="fail on WARN too")
    args = parser.parse_args()

    targets = args.paths or [DEFAULT_TARGET]
    tokens = load_tokens(args.tokens)
    token_hexes = collect_token_hexes(tokens.get("colors", {}), set())
    neutral_hexes = collect_neutral_hexes(tokens)
    allowed_fonts = allowed_font_families(tokens)

    files = gather_files(targets)
    messages: list[LintMessage] = []
    file_stats: dict[str, dict] = {}
    for f in files:
        msgs, stats = scan_file(f, token_hexes, allowed_fonts, neutral_hexes)
        messages += msgs
        file_stats[str(f)] = stats

    contrast_msgs, contrast_results = check_contrast(tokens)
    messages += contrast_msgs

    score = compute_score(file_stats, contrast_results, files) if args.score else None

    errors = sum(1 for m in messages if m.severity == "ERROR")
    warns = sum(1 for m in messages if m.severity == "WARN")

    if args.json:
        payload: dict[str, Any] = {
            "files_scanned": len(files),
            "errors": errors,
            "warnings": warns,
            "messages": [asdict(m) for m in messages],
            "contrast": contrast_results,
        }
        if score is not None:
            payload["score"] = score
        print(json.dumps(payload, indent=2))
    else:
        _print_report(files, messages, contrast_results, score, errors, warns)

    if errors or (args.strict and warns):
        return 1
    return 0


def _print_report(
    files: list[Path],
    messages: list[LintMessage],
    contrast_results: list[dict],
    score: dict | None,
    errors: int,
    warns: int,
) -> None:
    print(f"Dynamo Dark figure lint — {len(files)} file(s) scanned\n")
    if messages:
        for m in sorted(messages, key=lambda x: -SEVERITY_RANK[x.severity]):
            loc = f"{m.path}:{m.line}" if m.line else m.path
            print(f"  [{m.severity}] {m.rule_id}: {loc}\n           {m.message}")
    else:
        print("  No findings.")
    print(f"\n  {errors} error(s), {warns} warning(s)")

    if score is not None:
        print("\nDesign score (measured dimensions only)")
        for name, d in score["dimensions"].items():
            print(f"  {name:<24} {d['score']:>5.1f} / 100   (weight {d['weight']}%)")
        print(f"  {'-' * 44}")
        verdict = "SHIP-READY" if score["ship_ready_measured"] else "BELOW THRESHOLD"
        print(
            f"  {'OVERALL (measured)':<24} {score['overall_measured']:>5.1f} / 100"
            f"   [{verdict}, threshold {score['ship_threshold']}]"
        )
        print("\nContrast pairs (WCAG AA >= 4.5:1):")
        for r in contrast_results:
            ratio = f"{r['ratio']:.1f}:1" if r["ratio"] is not None else "n/a"
            mark = "PASS" if r["pass"] else "FAIL"
            print(f"  [{mark}] {r['pair']:<32} {ratio}")
        print(
            "\nJudged dimensions (NOT measured here — walk them in RATINGS.md "
            "against the rendered PNG):"
        )
        for j in score["judged_dimensions"]:
            print(f"  - {j}")


if __name__ == "__main__":
    sys.exit(main())
