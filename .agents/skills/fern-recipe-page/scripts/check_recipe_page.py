#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a Fern model-recipe page against its recipe and the catalog.

Run from the repo root:

    python3 .agents/skills/fern-recipe-page/scripts/check_recipe_page.py <slug>
    python3 .agents/skills/fern-recipe-page/scripts/check_recipe_page.py --all
    python3 .agents/skills/fern-recipe-page/scripts/check_recipe_page.py --all --pr

<slug> is the page basename without .mdx, e.g. `kimi-k3`.

Checks, in the order they tend to bite:

  structure   MDX that renders wrong or breaks the build - <div> balance, code
              fences, frontmatter/SPDX, and any `##` heading trapped inside a
              data-* wrapper (invisible for every other selection).
  vocabulary  data-* and picker values are CSS-coupled. A value absent from
              RecipeStyles.tsx renders a control that silently filters nothing.
  picker      Reachable combinations must equal the catalog's target list, and
              every reachable combination must render content in every section
              that has data-* blocks. A dead end looks fine until a reader
              picks that combination and the deploy step is blank.
  paths       Every kubectl path, GitHub blob URL and relative .mdx link must
              resolve.
  landing     The page needs a card on overview.mdx, in catalog order, and the
              two header counts must match the catalog.

Exit code is non-zero if any check fails, so it can gate a PR.
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import sys
from pathlib import Path

PDIR = Path("docs/fern/pages/recipes/model-recipes")
CATALOG = Path("docs/fern/pages/recipes/_catalog")
STYLES = Path("docs/fern/components/RecipeStyles.tsx")
OVERVIEW = PDIR / "overview.mdx"

# Rows in the order every recipe page must present them.
ROW_ORDER = ["sku", "usecase", "engine", "variant"]
ROW_LABEL = {"sku": "GPU", "usecase": "Workload", "engine": "Backend", "variant": "Topology"}


# --------------------------------------------------------------------------- util
class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, check: str, ok: bool, detail: str = "") -> None:
        self.rows.append((check, "PASS" if ok else "FAIL", detail))

    @property
    def failed(self) -> bool:
        return any(s == "FAIL" for _, s, _ in self.rows)

    def render(self, title: str) -> str:
        out = [f"### {title}", ""]
        for check, status, detail in self.rows:
            mark = "x" if status == "PASS" else " "
            out.append(f"- [{mark}] **{check}** {detail}".rstrip())
        return "\n".join(out)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def strip_fences(text: str):
    """Yield (line, in_fence) so checks can ignore code blocks."""
    fence = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fence = not fence
            yield line, True
            continue
        yield line, fence


def css_vocabulary() -> dict[str, set[str]]:
    """The values RecipeStyles.tsx actually implements, per picker dimension."""
    if not STYLES.is_file():
        return {}
    css = read(STYLES)
    vocab: dict[str, set[str]] = {}
    for dim, value in re.findall(r'name="recipe-(\w+)"\]\[value="([\w-]+)"', css):
        vocab.setdefault(dim, set()).add(value)
    return vocab


# ----------------------------------------------------------------------- parsing
def parse_picker(text: str):
    """Return (dims, needs, row_labels) for the page's target picker."""
    dims: dict[str, list[str]] = {}
    for dim, value in re.findall(
        r'<input type="radio"[^>]*name="recipe-(\w+)"[^>]*value="([^"]+)"', text
    ):
        dims.setdefault(dim, []).append(value)

    needs: dict[tuple[str, str], dict[str, str]] = {}
    for m in re.finditer(r'<label htmlFor="recipe-(\w+)-([\w.-]+)"([^>]*)>', text):
        req = dict(re.findall(r'data-needs-(\w+)="([^"]*)"', m.group(3)))
        if req:
            needs[(m.group(1), m.group(2))] = req

    labels = re.findall(r'<span className="dynamo-target-picker-dim">([^<]+)</span>', text)
    return dims, needs, labels


def sections(text: str) -> dict[str, str]:
    """Split at `##` and `###`, ignoring headings inside code fences."""
    out: dict[str, list[str]] = {}
    current = "(intro)"
    for line, in_fence in strip_fences(text):
        if not in_fence and re.match(r"^#{2,3} ", line):
            current = line.strip()
        out.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def wrapper_blocks(body: str) -> list[dict[str, str]]:
    """data-* wrapper divs in a section, excluding picker summary panels."""
    return [
        dict(re.findall(r'data-(\w+)="([^"]*)"', raw))
        for raw in re.findall(r'<div ((?:data-\w+="[^"]*"\s*)+)>', body)
        if "picker-summary" not in raw
    ]


def visible(block: dict[str, str], selection: dict[str, str]) -> bool:
    """A block shows when, for every dimension, it omits the attribute or lists
    the selected value. Mirrors the `[data-x]:not([data-x~="v"])` CSS."""
    return all(
        selection[dim] in block.get(dim, selection[dim]).split() for dim in selection
    )


def reachable(dims, needs) -> list[dict[str, str]]:
    """Combinations a reader can actually click to, honouring data-needs-*."""
    keys = sorted(dims)
    out = []
    for combo in itertools.product(*[dims[k] for k in keys]):
        sel = dict(zip(keys, combo))
        if all(
            all(sel.get(rk) in rv.split() for rk, rv in needs.get((k, sel[k]), {}).items())
            for k in keys
        ):
            out.append(sel)
    return out


# Catalog values are free-text; these map them onto the picker vocabulary.
ENGINE_ALIAS = {"vllm": "vllm", "sglang": "sglang", "trtllm": "trtllm",
                "tensorrt-llm": "trtllm", "tensorrtllm": "trtllm"}
TOPOLOGY_ALIAS = {"aggregated": "agg", "disaggregated": "disagg"}
WORKLOAD_ALIAS = {"chat": "chat", "agentic": "agentic",
                  "agentic-trace-replay": "agentic", "trace-replay": "agentic",
                  "static-synthetic": "static", "static-generation": "static",
                  "multimodal": "multimodal"}


def catalog_groups(targets):
    """Distinct (gpu, workload, engine, topology) groups behind the target list.

    Several catalog targets routinely collapse onto one picker combination -
    Nemotron-3.5-Lightning ships 30 targets across 12 groups because it
    enumerates spec-decode variants the picker does not expose as a dimension.
    Compare groups, not raw target counts. Returns None if any value falls
    outside the known vocabulary, so the caller can skip rather than false-fail.
    """
    groups = set()
    for t in targets:
        gpu, eng, top, wl = t["gpu"], t["engine"], t["topology"], t["workload"]
        if not all((gpu, eng, top, wl)):
            return None
        eng = ENGINE_ALIAS.get(eng.lower())
        top = TOPOLOGY_ALIAS.get(top.lower())
        wl = WORKLOAD_ALIAS.get(wl.lower())
        if not all((eng, top, wl)):
            return None
        for one in gpu.replace("/", " ").split():
            groups.add((one.strip(",").lower(), wl, eng, top))
    return groups


def catalog_targets(slug: str):
    """(gpu, framework, topology, workload) per shipped target, from the catalog."""
    for path in sorted((CATALOG / "recipes").glob("*.yaml")):
        text = read(path)
        page = re.search(r"^page:\s*(\S+)", text, re.M)
        if not page or Path(page.group(1)).stem != slug:
            continue
        rows = []
        for block in re.split(r"\n- id: ", text)[1:]:
            grab = lambda pat: (re.search(pat, block).group(1) if re.search(pat, block) else None)
            rows.append(
                {
                    "gpu": grab(r"gpu:\s*(\S+)"),
                    "engine": grab(r"framework:\s*(\S+)"),
                    "topology": grab(r"topology:\s*(\S+)"),
                    "workload": grab(r"type:\s*(\S+)"),
                    "asset": grab(r"asset:\s*(\S+)"),
                }
            )
        return path, rows
    return None, None


# ------------------------------------------------------------------------ checks
def check_structure(text: str, rep: Report) -> None:
    opens = len(re.findall(r"<div\b", text))
    closes = len(re.findall(r"</div>", text))
    rep.add("div balance", opens == closes, "" if opens == closes else f"{opens} open / {closes} close")
    rep.add("code fences", text.count("\n```") % 2 == 0)

    body = text.split("---", 2)[2] if text.startswith("---") and text.count("---") >= 2 else text
    # `# ...` inside a bash fence is a comment, not a heading.
    h1 = [l for l, in_fence in strip_fences(body) if not in_fence and re.match(r"^# ", l)]
    rep.add("no body H1", not h1, "Fern renders the title from frontmatter; found " + h1[0][:40] if h1 else "")
    rep.add("SPDX header", "SPDX-License-Identifier" in text[:400])

    depth, nested, indented = 0, [], []
    for line, in_fence in strip_fences(text):
        if in_fence:
            if re.match(r"^\s+```", line):
                indented.append(line.strip()[:30])
            continue
        if re.match(r"^<div\b", line):
            depth += 1
        elif re.match(r"^</div>", line):
            depth = max(0, depth - 1)
        elif line.startswith("## ") and depth > 0:
            nested.append(line.strip())
    rep.add("headings at top level", not nested, "; ".join(nested[:3]))
    rep.add("fences at column 0", not indented, "; ".join(indented[:3]))

    # A wrapper div must be surrounded by blank lines or the markdown inside it
    # is not parsed as markdown.
    lines = text.split("\n")
    tight = [
        f"L{i+1}"
        for i, line in enumerate(lines)
        if re.match(r"^<div data-[^>]*>$", line) and i + 1 < len(lines) and lines[i + 1].strip()
    ]
    rep.add("blank line after wrapper divs", not tight, " ".join(tight[:5]))


def check_vocabulary(text: str, dims, rep: Report) -> None:
    vocab = css_vocabulary()
    if not vocab:
        rep.add("CSS vocabulary", True, "(RecipeStyles.tsx not found - skipped)")
        return
    bad = []
    for dim, values in dims.items():
        for value in values:
            if value not in vocab.get(dim, set()):
                bad.append(f"recipe-{dim}={value}")
    for dim, value in re.findall(r'data-(sku|usecase|engine|variant)="([^"]*)"', text):
        for token in value.split():
            if token not in vocab.get(dim, set()):
                bad.append(f"data-{dim}={token}")
    rep.add(
        "CSS-coupled values implemented",
        not bad,
        "unimplemented: " + ", ".join(sorted(set(bad))[:6]) if bad else "",
    )


def check_picker(slug: str, text: str, rep: Report) -> None:  # noqa: C901
    dims, needs, labels = parse_picker(text)
    if not dims:
        rep.add("picker present", False, "no recipe-* radios found")
        return

    expected = [ROW_LABEL[d] for d in ROW_ORDER if d in dims]
    rep.add(
        "row order GPU/Workload/Backend/Topology",
        labels[: len(expected)] == expected,
        f"got {labels}" if labels[: len(expected)] != expected else "",
    )

    for dim, values in dims.items():
        checked = len(re.findall(rf'name="recipe-{dim}"[^>]*defaultChecked', text))
        rep.add(f"one default in {ROW_LABEL.get(dim, dim)}", checked == 1, f"{checked} defaultChecked")

    combos = reachable(dims, needs)
    rep.add("reachable combinations", bool(combos), f"{len(combos)} selectable")

    # Dead ends: a section with data-* blocks that renders nothing, and has no
    # static prose to carry it.
    dead = []
    for name, body in sections(text).items():
        blocks = wrapper_blocks(body)
        if not blocks:
            continue
        stripped = re.sub(r"<[^>]+>", "", re.sub(r"^#{1,6} .*$", "", body, flags=re.M))
        has_static = len(stripped.strip()) > 40
        for sel in combos:
            if not any(visible(b, sel) for b in blocks) and not has_static:
                dead.append(f"{'+'.join(sel[k] for k in sorted(sel))} -> {name}")
    rep.add("no dead-end selections", not dead, "; ".join(dead[:3]) + (f" (+{len(dead)-3})" if len(dead) > 3 else ""))

    path, targets = catalog_targets(slug)
    if targets is None:
        rep.add("catalog entry", False, f"no _catalog/recipes/*.yaml with page {slug}.mdx")
    else:
        rep.add("catalog entry", True, f"{path.name}, {len(targets)} target(s)")
        groups = catalog_groups(targets)
        # Advisory only. A picker chip often groups several catalog rows (one
        # `hopper` chip covers h100/h200/a100), and one catalog target can be
        # split across chips, so the counts legitimately differ. The reliable
        # signal is asset coverage, below.
        detail = f"{len(combos)} selectable"
        if groups is not None:
            detail += f" vs {len(groups)} catalog group(s) from {len(targets)} target(s)"
            if len(combos) != len(groups):
                detail += " - confirm by hand that each maps to a real target"
        rep.add("combinations vs catalog (advisory)", True, detail)

        # Hard check: every shipped target must be reachable from the page.
        # Pages reference a target three ways: the full asset path, a
        # recipe-relative path (`vllm/agg-b200-agentic/deploy.yaml`), or a
        # RECIPE= variable holding the directory. Matching the last two path
        # components of the asset's directory covers all three.
        missing = []
        for t in targets:
            asset = t.get("asset")
            if not asset:
                continue
            tail = "/".join(Path(asset).parent.parts[-2:])
            if tail and tail not in text:
                missing.append(asset)
        rep.add(
            "every catalog target referenced",
            not missing,
            "page never mentions: " + ", ".join(missing[:3]) if missing else f"{len(targets)} target(s)",
        )


def check_paths(page: Path, text: str, rep: Report) -> None:
    missing = []
    for rel in set(re.findall(r"github\.com/ai-dynamo/dynamo/(?:blob|tree)/main/([A-Za-z0-9_./\-]+)", text)):
        if not Path(rel.rstrip(".,);")).exists():
            missing.append(rel)
    for rel in set(re.findall(r"-f\s+(recipes/[A-Za-z0-9_./\-]+)", text)):
        if not Path(rel.rstrip(".,);")).exists():
            missing.append(rel)
    for rel in re.findall(r"\]\((\.\.?/[^)#]*\.mdx)", text):
        if not (page.parent / rel).resolve().exists():
            missing.append(rel)
    rep.add("paths and links resolve", not missing, "missing: " + ", ".join(missing[:4]) if missing else "")


def check_landing(slug: str, rep: Report) -> None:
    if not OVERVIEW.is_file():
        rep.add("landing page", False, "overview.mdx not found")
        return
    text = read(OVERVIEW)
    hrefs = [h[:-4] for h in re.findall(r'href="([a-z0-9.\-]+\.mdx)"', text)]
    rep.add("card on landing page", slug in hrefs, "" if slug in hrefs else f"no card links to {slug}.mdx")

    index = CATALOG / "index.yaml"
    if index.is_file():
        body = read(index)
        active = re.split(r"^deferred_recipes:", body, flags=re.M)[0]
        ids = re.findall(r"^-\s+([a-z0-9-]+)\s*$", active, re.M)
        # The catalog id is not always the page filename (inkling -> inkling-nvfp4),
        # so resolve each id through its entry's `page:` field.
        order = []
        for rid in ids:
            entry = CATALOG / "recipes" / f"{rid}.yaml"
            page_ref = re.search(r"^page:\s*(\S+)", read(entry), re.M) if entry.is_file() else None
            order.append(Path(page_ref.group(1)).stem if page_ref else rid)
        mismatch = [f"{a}!={b}" for a, b in zip(hrefs, order) if a != b]
        rep.add(
            "card order matches catalog",
            hrefs == order,
            "; ".join(mismatch[:3]) if mismatch else "",
        )

        families = re.search(r"<h3>(\d+) model families</h3>", text)
        configs = re.search(r"<strong>(\d+)</strong>\s*\n?\s*<span>Deployable configurations", text)
        want_families = len(ids)
        want_configs = 0
        for rid in ids:
            entry = CATALOG / "recipes" / f"{rid}.yaml"
            if entry.is_file():
                want_configs += len(re.findall(r"^- id: ", read(entry), re.M))
        rep.add(
            "landing counts",
            bool(families and configs)
            and int(families.group(1)) == want_families
            and int(configs.group(1)) == want_configs,
            f"page says {families.group(1) if families else '?'} families / "
            f"{configs.group(1) if configs else '?'} configs; "
            f"catalog says {want_families} / {want_configs}",
        )


def check_nav(slug: str, rep: Report) -> None:
    nav = Path("docs/fern/index.yml")
    if not nav.is_file():
        rep.add("navigation", True, "(index.yml not found - skipped)")
        return
    rep.add(
        "wired into index.yml",
        f"model-recipes/{slug}.mdx" in read(nav),
        "" if f"model-recipes/{slug}.mdx" in read(nav) else "add a `- page:` entry under the Recipes tab",
    )


# -------------------------------------------------------------------------- main
def check_page(slug: str) -> Report:
    rep = Report()
    page = PDIR / f"{slug}.mdx"
    if not page.is_file():
        rep.add("page exists", False, str(page))
        return rep
    text = read(page)
    check_structure(text, rep)
    dims, _, _ = parse_picker(text)
    check_vocabulary(text, dims, rep)
    check_picker(slug, text, rep)
    check_paths(page, text, rep)
    check_landing(slug, rep)
    check_nav(slug, rep)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug", nargs="?", help="page basename without .mdx, e.g. kimi-k3")
    ap.add_argument("--all", action="store_true", help="check every recipe page")
    ap.add_argument("--pr", action="store_true", help="emit a markdown block for the PR description")
    args = ap.parse_args()

    if not PDIR.is_dir():
        print("run this from the repo root (docs/fern/pages/recipes/model-recipes not found)", file=sys.stderr)
        return 2

    slugs = (
        sorted(p.stem for p in PDIR.glob("*.mdx") if p.stem != "overview")
        if args.all
        else ([args.slug] if args.slug else [])
    )
    if not slugs:
        ap.print_usage(sys.stderr)
        return 2

    blocks, failed = [], False
    for slug in slugs:
        rep = check_page(slug)
        failed |= rep.failed
        blocks.append(rep.render(slug))

    if args.pr:
        print("<!-- generated by .agents/skills/fern-recipe-page/scripts/check_recipe_page.py -->")
        print("## Recipe page validation\n")
    print("\n\n".join(blocks))
    if failed:
        print("\nFAILED - fix the items above before requesting review.", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
