#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic linter for the structural docs rules.

Stdlib only, no network. Checks docs/examples/recipes for the must-fix subset of
docs/fern/pages/community/contributing/documentation/documentation-style-guide.md:

  SPDX       header present + correct form for the file type (and not as a body H1)
  FRONTMATTER docs .md/.mdx have a real YAML key; no duplicate body `# H1`
  LINK       relative links resolve; docs/ links must not escape docs/ (use a GitHub URL);
             no hardcoded docs.nvidia.com self-links
  NAV        docs/fern/index.yml `path:` entries resolve, and every page file is in the nav
  INTERNAL   no NVBug/JIRA-style IDs, internal hosts, or TODO/FIXME in shipped docs

Usage:
  python3 docs/fern/scripts/docs_lint.py [--scan docs,examples,recipes] [--json]
  python3 docs/fern/scripts/docs_lint.py file1.md file2.md   # lint specific files

Exit code: 1 if any error-severity findings, else 0.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass

# Repo root, derived from this file's location (docs/fern/scripts/docs_lint.py): four levels up.
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

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
JIRA_RE = re.compile(r"\b(DYN|DYNAMO|DIS|DEP|OPS|NSPECT|SCANNERAU|PLC)-\d+\b")
# Hardcoded links back to this site. The style guide requires a relative path with the extension
# for doc → doc links; an absolute URL also pins a release snapshot at whatever version it names.
SELF_LINK_RE = re.compile(r"https?://docs\.nvidia\.com/dynamo/\S", re.I)
# Dated archives — release notes and blog posts describe the site as it stood on a date, so their
# absolute links are pinned deliberately and must not follow a version rewrite.
SELF_LINK_EXEMPT = ("/blog/", "/reference/general/releases")
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
    "NAV": "every `path:` in docs/fern/index.yml must resolve to a real file, and every page "
    "file needs a `path:` entry",
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
            if (
                "SPDX-License-Identifier" not in head
                or "SPDX-FileCopyrightText" not in head
            ):
                out.append(
                    Finding(
                        rel,
                        1,
                        "SPDX",
                        "error",
                        "missing SPDX header (HTML-comment block for frontmatter-less markdown)",
                    )
                )
        # SPDX accidentally in the body renders as an H1. Line numbers are file-relative, so the
        # frontmatter block the body starts after has to be added back.
        if fm is not None:
            body_offset = text[: len(text) - len(body)].count("\n")
            for i, ln in enumerate(blank_code(body).splitlines(), 1):
                if re.match(r"^#\s+SPDX", ln):
                    out.append(
                        Finding(
                            rel,
                            body_offset + i,
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
        line = body[: m.start()].count("\n") + 1
        # Checked before the external-URL skip: a link to our own published site is not an
        # external reference, it is a doc → doc link written the wrong way.
        if (
            in_docs
            and not any(x in rel.replace(os.sep, "/") for x in SELF_LINK_EXEMPT)
            and SELF_LINK_RE.match(m.group(1))
        ):
            out.append(
                Finding(
                    rel,
                    line,
                    "LINK",
                    "warn",
                    f"hardcoded docs.nvidia.com self-link ({url}) — use a relative path with the "
                    "extension so version rewrites follow the reader",
                )
            )
            continue
        if not url or url.startswith(("http://", "https://", "mailto:", "tel:", "/")):
            continue
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
    # Blank the whole file once: a fence spans several lines, so blanking a single line in
    # isolation never sees the opening and closing markers and cannot exempt a code example.
    # Every rule below reads the blanked line, not the raw one. Previously only
    # TODO_RE did, so the two error rules and the host warning fired inside code
    # fences -- a troubleshooting page quoting a real log line, or contributing
    # docs showing an example tracker ID, failed a required check for content
    # that is deliberately verbatim.
    for i, blanked_ln in enumerate(blank_code(text).splitlines(), 1):
        if NVBUG_RE.search(blanked_ln):
            out.append(
                Finding(rel, i, "INTERNAL", "error", "NVBug reference in shipped docs")
            )
        jira = JIRA_RE.search(blanked_ln)
        if jira:
            out.append(
                Finding(
                    rel,
                    i,
                    "INTERNAL",
                    "error",
                    f"tracker ID in shipped docs: {jira.group(0)}",
                )
            )
        if TODO_RE.search(blanked_ln):
            out.append(
                Finding(rel, i, "INTERNAL", "warn", "TODO/FIXME in shipped docs")
            )
        for h in NV_HOST_RE.findall(blanked_ln):
            if h.lower() not in PUBLIC_NV_HOSTS:
                out.append(
                    Finding(rel, i, "INTERNAL", "warn", f"internal-looking host: {h}")
                )


def check_nav(repo: str, out: list) -> None:
    """Nav and content must agree in both directions.

    Forward: every `path:` resolves to a file (error — a dangling path breaks the build).
    Reverse: every page file is referenced by some `path:` (warn — an unreferenced page is
    unreachable, but the tree also holds legitimate non-page files).

    Paths are relative to docs/fern/, not to docs/.
    """
    index = os.path.join(repo, NAV_FILE)
    if not os.path.exists(index):
        out.append(Finding(NAV_FILE, 1, "NAV", "error", "navigation file not found"))
        return
    with open(index, encoding="utf-8") as f:
        content = f.read()
    referenced = set()
    for m in PATH_RE.finditer(content):
        p = m.group(1).strip().strip("\"'")
        referenced.add(p)
        target = os.path.join(repo, FERN_DIR, p)
        if not os.path.exists(target):
            line = content[: m.start()].count("\n") + 1
            out.append(
                Finding(NAV_FILE, line, "NAV", "error", f"nav path has no file: {p}")
            )

    pages_root = os.path.join(repo, FERN_DIR, "pages")
    for dirpath, dirnames, names in os.walk(pages_root):
        # `_catalog/`, `_assets/`, and the like hold source data and fragments, not pages.
        dirnames[:] = [d for d in dirnames if not d.startswith("_")]
        for n in names:
            if not n.endswith((".md", ".mdx")) or n in NON_PAGE_FILES:
                continue
            rel = os.path.relpath(
                os.path.join(dirpath, n), os.path.join(repo, FERN_DIR)
            )
            if rel not in referenced:
                out.append(
                    Finding(
                        # `rel` is relative to docs/fern because that is what
                        # index.yml `path:` entries are measured against, and
                        # the membership test above needs it in that form. What
                        # gets reported has to be repo-relative like every other
                        # rule: Finding.file feeds emit_github, and GitHub
                        # silently drops an annotation whose path it cannot
                        # resolve, so the warning never reached the file.
                        os.path.join(FERN_DIR, rel),
                        1,
                        "NAV",
                        "warn",
                        "page file has no `path:` entry in docs/fern/index.yml — unreachable "
                        "on the site",
                    )
                )


class MissingScanTree(Exception):
    """A requested scan tree does not exist."""


def gather(repo: str, scan: list) -> list:
    exts = (".md", ".mdx", ".py", ".sh", ".yaml", ".yml")
    files = []
    for tree in scan:
        root = os.path.join(repo, tree)
        # os.walk on a nonexistent root yields nothing and raises nothing, so a
        # moved directory or a typo in the workflow argument produced a passing
        # required check that verified zero files.
        if not os.path.isdir(root):
            raise MissingScanTree(tree)
        for dirpath, dirnames, names in os.walk(root):
            # Prune rather than test the joined path: `"/.git" in dirpath` also
            # matches `.github`, so `--scan .github` skipped its whole tree.
            # Pruning is also cheaper, since os.walk never descends.
            dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules"}]
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
            "Reproduce locally with `python3 docs/fern/scripts/docs_lint.py --scan docs`.",
        ]
    else:
        lines.append(f"No errors across {scanned} scanned files.")
    with open(summary, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Dynamo docs deterministic linter")
    ap.add_argument("--repo", default=REPO_ROOT)
    # `docs` only, matching the `Docs Lint` job. Widening this to examples and
    # recipes surfaces pre-existing violations there that no job gates, so a
    # local run would fail on work the author never touched.
    ap.add_argument("--scan", default="docs", help="comma-separated trees to scan")
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

    try:
        files = (
            [os.path.abspath(f) for f in args.files]
            if args.files
            else gather(repo, [t.strip() for t in args.scan.split(",") if t.strip()])
        )
    except MissingScanTree as missing:
        print(
            f"--scan names a tree that does not exist under {repo}: {missing}. "
            f"Refusing to report a clean run over nothing.",
            file=sys.stderr,
        )
        return 2

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
            print("  Reproduce: python3 docs/fern/scripts/docs_lint.py --scan docs")
            print(f"{bar}\n")
        else:
            print("\nDocs lint passed: 0 errors.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
