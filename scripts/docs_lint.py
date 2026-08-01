#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic linter for the structural docs rules.

Stdlib only, no network. Checks docs/examples/recipes for the must-fix subset of
docs/fern/pages/community/contributing/documentation/documentation-style-guide.md:

  SPDX       header present + correct form for the file type (and not as a body H1)
  FRONTMATTER docs .md/.mdx have a real YAML key; no duplicate body `# H1`
  LINK       relative links resolve; docs/ links must not escape docs/ (use a GitHub URL)
  NAV        docs/fern/index.yml `path:` entries resolve (no dangling refs)
  INTERNAL   no NVBug/JIRA-style IDs, internal hosts, or TODO/FIXME in shipped docs

Usage:
  python3 scripts/docs_lint.py [--scan docs,examples,recipes] [--json]
  python3 scripts/docs_lint.py file1.md file2.md      # lint specific files (pre-commit mode)

Exit code: 1 if any error-severity findings, else 0.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass

# Repo root, derived from this file's location (scripts/docs_lint.py).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fern content root. Nav `path:` values are relative to this directory, not to docs/.
FERN_DIR = os.path.join("docs", "fern")
NAV_FILE = os.path.join(FERN_DIR, "index.yml")

PUBLIC_NV_HOSTS = (
    "docs.nvidia.com",
    "developer.nvidia.com",
    "catalog.ngc.nvidia.com",
    "build.nvidia.com",
    "ngc.nvidia.com",
    "helm.ngc.nvidia.com",
    "pypi.nvidia.com",
    "docs.dynamo.nvidia.com",
    "research.nvidia.com",
    "www.nvidia.com",
    "nvidia.com",
    "nvevents.nvidia.com",
)
JIRA_RE = re.compile(r"\b(DYN|DYNAMO|DIS|DEP|NSPECT|SCANNERAU|PLC)-\d+\b")
NVBUG_RE = re.compile(r"(?i)\bnvbugs?\b[\s:#]*\d")  # require an actual bug number
TODO_RE = re.compile(r"\b(TODO|FIXME|XXX):")  # real markers, not prose mentions
NV_HOST_RE = re.compile(r"https?://([a-z0-9.-]+\.nvidia\.com)", re.I)
LINK_RE = re.compile(
    r"(?<!!)\[[^\]]*\]\(\s*([^)\s]+)"
)  # markdown links (skip ! images)
FENCE_RE = re.compile(r"```.*?```", re.S)
FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
PATH_RE = re.compile(r"^\s*path:\s*(\S+)\s*$", re.M)

# Not Fern pages, even though they sit under docs/: agent instructions and authoring READMEs
# carry an HTML-comment SPDX block and a body H1, and none of them appear in the nav. SPDX and
# link rules still apply to them; only the frontmatter rules are skipped.
NON_PAGE_FILES = ("AGENTS.md", "CLAUDE.md", "README.md")

# Docs on how to fix each rule, surfaced in the CI failure output.
RULE_HELP = {
    "SPDX": "add the SPDX header (frontmatter `#` lines for Fern pages, HTML comment otherwise)",
    "FRONTMATTER": "frontmatter needs SPDX + one real YAML key; start the body at `##`",
    "LINK": "relative links stay inside docs/; link outside it with a github.com/ai-dynamo/dynamo URL",
    "NAV": "every `path:` in docs/fern/index.yml must resolve to a real file",
    "INTERNAL": "remove tracker IDs, internal hosts, and TODO/FIXME from shipped docs",
}
STYLE_GUIDE = (
    "docs/fern/pages/community/contributing/documentation/documentation-style-guide.md"
)


@dataclass
class Finding:
    path: str
    line: int
    rule: str
    severity: str  # "error" | "warn"
    message: str


def blank_code(text: str) -> str:
    """Blank fenced + inline code, preserving line numbers, so links/headings/SPDX inside code
    examples aren't matched."""
    text = FENCE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    return re.sub(r"`[^`\n]*`", lambda m: " " * len(m.group(0)), text)


def frontmatter(text: str):
    m = FM_RE.match(text)
    if not m:
        return None, text
    return m.group(1), text[m.end() :]


def check_spdx(rel: str, text: str, out: list) -> None:
    ext = os.path.splitext(rel)[1].lower()
    if ext in (".md", ".mdx"):
        fm, body = frontmatter(text)
        if fm is not None:
            if (
                "SPDX-License-Identifier" not in fm
                or "SPDX-FileCopyrightText" not in fm
            ):
                out.append(
                    Finding(
                        rel,
                        1,
                        "SPDX",
                        "error",
                        "missing SPDX header in frontmatter (2 `#` lines inside ---)",
                    )
                )
        else:
            head = "\n".join(text.splitlines()[:12])
            if "SPDX-License-Identifier" not in head:
                out.append(
                    Finding(
                        rel,
                        1,
                        "SPDX",
                        "error",
                        "missing SPDX header (HTML-comment block for frontmatter-less markdown)",
                    )
                )
        # SPDX accidentally in the body renders as an H1
        for i, ln in (
            enumerate(blank_code(body).splitlines(), 1) if fm is not None else []
        ):
            if re.match(r"^#\s+SPDX", ln):
                out.append(
                    Finding(
                        rel,
                        i,
                        "SPDX",
                        "error",
                        "SPDX line in body renders as an H1 — move it into frontmatter",
                    )
                )
    else:  # code / config
        head = "\n".join(text.splitlines()[:15])
        if (
            "SPDX-License-Identifier" not in head
            or "SPDX-FileCopyrightText" not in head
        ):
            out.append(Finding(rel, 1, "SPDX", "error", "missing SPDX header block"))


def check_frontmatter(rel: str, text: str, out: list) -> None:
    fm, body = frontmatter(text)
    if fm is None:
        out.append(
            Finding(
                rel,
                1,
                "FRONTMATTER",
                "warn",
                "no `---` frontmatter (Fern pages need SPDX + a key)",
            )
        )
        return
    # The frontmatter must carry at least one real YAML key. A comment-only block (just the SPDX
    # `#` lines) isn't parsed as frontmatter, so the SPDX lines render as H1s. Any real YAML key
    # fixes it (`title`, `subtitle`, or `sidebar-title`).
    if not re.search(r"^\s*[A-Za-z][\w-]*\s*:", fm, re.M):
        out.append(
            Finding(
                rel,
                1,
                "FRONTMATTER",
                "error",
                "frontmatter has only comments, no YAML key — SPDX will render as an H1; "
                "add `subtitle:` or `sidebar-title:`",
            )
        )
    # Fern generates the page H1 from the nav `page:` value, so a body `# H1` renders a second,
    # duplicate title (start the body at `##`). Locale mirrors under translations/ are paired to
    # the base page's nav entry, so the rule applies to them too.
    m = re.search(r"^#\s+\S", blank_code(body), re.M)
    if m:
        body_offset = text[: len(text) - len(body)].count("\n")
        line = body_offset + blank_code(body)[: m.start()].count("\n") + 1
        out.append(
            Finding(
                rel,
                line,
                "FRONTMATTER",
                "warn",
                "body `# H1` duplicates the Fern nav-generated title — start the body at `##`",
            )
        )


def check_links(rel: str, abspath: str, text: str, repo: str, out: list) -> None:
    docs_root = os.path.join(repo, "docs")
    in_docs = os.path.abspath(abspath).startswith(os.path.abspath(docs_root) + os.sep)
    body = blank_code(text)
    for m in LINK_RE.finditer(body):
        url = m.group(1).split("#", 1)[0]
        if not url or url.startswith(("http://", "https://", "mailto:", "tel:", "/")):
            continue
        line = body[: m.start()].count("\n") + 1
        target = os.path.normpath(os.path.join(os.path.dirname(abspath), url))
        if in_docs and not os.path.abspath(target).startswith(
            os.path.abspath(docs_root) + os.sep
        ):
            out.append(
                Finding(
                    rel,
                    line,
                    "LINK",
                    "error",
                    f"relative link escapes docs/ ({url}) — use an absolute github.com/ai-dynamo/dynamo URL",
                )
            )
        elif not os.path.exists(target):
            out.append(
                Finding(rel, line, "LINK", "error", f"broken relative link: {url}")
            )


def check_internal(rel: str, text: str, out: list) -> None:
    for i, ln in enumerate(text.splitlines(), 1):
        if NVBUG_RE.search(ln):
            out.append(
                Finding(rel, i, "INTERNAL", "error", "NVBug reference in shipped docs")
            )
        if JIRA_RE.search(ln):
            out.append(
                Finding(
                    rel,
                    i,
                    "INTERNAL",
                    "error",
                    f"tracker ID in shipped docs: {JIRA_RE.search(ln).group(0)}",
                )
            )
        if TODO_RE.search(blank_code(ln)):
            out.append(
                Finding(rel, i, "INTERNAL", "warn", "TODO/FIXME in shipped docs")
            )
        for h in NV_HOST_RE.findall(ln):
            if h.lower() not in PUBLIC_NV_HOSTS:
                out.append(
                    Finding(rel, i, "INTERNAL", "warn", f"internal-looking host: {h}")
                )


def check_nav(repo: str, out: list) -> None:
    """Every nav `path:` must resolve. Paths are relative to docs/fern/, not to docs/."""
    index = os.path.join(repo, NAV_FILE)
    if not os.path.exists(index):
        out.append(Finding(NAV_FILE, 1, "NAV", "error", "navigation file not found"))
        return
    with open(index, encoding="utf-8") as f:
        content = f.read()
    for m in PATH_RE.finditer(content):
        p = m.group(1).strip().strip("\"'")
        target = os.path.join(repo, FERN_DIR, p)
        if not os.path.exists(target):
            line = content[: m.start()].count("\n") + 1
            out.append(
                Finding(NAV_FILE, line, "NAV", "error", f"nav path has no file: {p}")
            )


def gather(repo: str, scan: list) -> list:
    exts = (".md", ".mdx", ".py", ".sh", ".yaml", ".yml")
    files = []
    for tree in scan:
        root = os.path.join(repo, tree)
        for dirpath, _, names in os.walk(root):
            if "/.git" in dirpath or "/node_modules" in dirpath:
                continue
            for n in names:
                if n.endswith(exts):
                    files.append(os.path.join(dirpath, n))
    return sorted(files)


def emit_github(out: list, errors: list, scanned: int) -> None:
    """Emit GitHub Actions annotations plus a job summary.

    Annotations render inline on the offending line in the pull request diff, which needs no
    write permission — `pre-merge.yml` runs on `pull_request`, so its token cannot post a
    comment, least of all from a fork.
    """
    for f in sorted(out, key=lambda x: (x.path, x.line)):
        level = "error" if f.severity == "error" else "warning"
        msg = f.message.replace("\n", " ")
        print(f"::{level} file={f.path},line={f.line},title=docs-lint {f.rule}::{msg}")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    lines = ["## Docs lint", ""]
    if errors:
        lines += [
            f"**{len(errors)} error(s)** across {scanned} scanned files. "
            "Every one must be fixed before this pull request can merge.",
            "",
            "| Rule | File | Line | Problem |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| `{f.rule}` | `{f.path}` | {f.line} | {f.message} |"
            for f in sorted(errors, key=lambda x: (x.path, x.line))
        ]
        rules = sorted({f.rule for f in errors})
        lines += ["", "### How to fix", ""]
        lines += [f"- **{r}**: {RULE_HELP.get(r, '')}" for r in rules]
        lines += [
            "",
            f"Full standard: [`{os.path.basename(STYLE_GUIDE)}`]({STYLE_GUIDE}). "
            "Reproduce locally with `python3 scripts/docs_lint.py --scan docs`.",
        ]
    else:
        lines.append(f"No errors across {scanned} scanned files.")
    with open(summary, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Dynamo docs deterministic linter")
    ap.add_argument("--repo", default=REPO_ROOT)
    ap.add_argument(
        "--scan", default="docs,examples,recipes", help="comma-separated trees to scan"
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--github",
        action="store_true",
        help="emit GitHub Actions annotations and a job summary",
    )
    ap.add_argument("--no-nav", action="store_true", help="skip the navigation check")
    ap.add_argument("files", nargs="*", help="specific files to lint (pre-commit mode)")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)

    files = (
        [os.path.abspath(f) for f in args.files]
        if args.files
        else gather(repo, [t.strip() for t in args.scan.split(",") if t.strip()])
    )

    out: list = []
    if not args.no_nav:
        check_nav(repo, out)
    for abspath in files:
        rel = os.path.relpath(abspath, repo)
        try:
            with open(abspath, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        ext = os.path.splitext(abspath)[1].lower()
        check_spdx(rel, text, out)
        if ext in (".md", ".mdx"):
            check_internal(rel, text, out)
            check_links(rel, abspath, text, repo, out)
            # Frontmatter rules apply to Fern pages only, not to AGENTS.md/CLAUDE.md/README.md.
            if (
                os.path.abspath(abspath).startswith(os.path.join(repo, "docs") + os.sep)
                and os.path.basename(rel) not in NON_PAGE_FILES
            ):
                check_frontmatter(rel, text, out)

    errors = [f for f in out if f.severity == "error"]
    if args.github:
        emit_github(out, errors, len(files))
    if args.json:
        print(json.dumps([asdict(f) for f in out], indent=2))
    else:
        by_rule: dict = {}
        for f in out:
            by_rule.setdefault((f.rule, f.severity), 0)
            by_rule[(f.rule, f.severity)] += 1
        for f in sorted(out, key=lambda x: (x.path, x.line)):
            print(f"{f.severity.upper():5} {f.rule:11} {f.path}:{f.line}  {f.message}")
        print("\n--- summary ---")
        for (rule, sev), n in sorted(by_rule.items()):
            print(f"  {sev:5} {rule:11} {n}")
        print(
            f"  files scanned: {len(files)} | findings: {len(out)} | errors: {len(errors)}"
        )
        if errors:
            bar = "=" * 72
            rules = sorted({f.rule for f in errors})
            print(f"\n{bar}")
            print(
                f"DOCS LINT FAILED: {len(errors)} error(s) must be fixed before merge."
            )
            print(bar)
            for r in rules:
                print(f"  {r:11} {RULE_HELP.get(r, '')}")
            print(f"\n  Standard:  {STYLE_GUIDE}")
            print("  Reproduce: python3 scripts/docs_lint.py --scan docs")
            print(f"{bar}\n")
        else:
            print("\nDocs lint passed: 0 errors.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
