# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Citation-resolution linter for a Dynamo skill.

The skills are distillations of the Dynamo documentation and source, so every
load-bearing fact should cite the public doc or source file it came from, and
that citation must resolve against a real Dynamo checkout. This linter extracts
every cited Dynamo repo path from a skill's SKILL.md + references/*.md and
classifies each one:

  RESOLVED  the path exists in the Dynamo checkout (a real doc/source citation)
  MISSING   cited but not present in the checkout (broken or invented citation)
  INTERNAL  cites a non-public authoring artifact (citations.md, the repo
            survey, HANDOFF.md, the dynamo-skills/ corpus, SKILL_AUTHORING.md)
            instead of the doc itself -- the distilled fact is not linked back
            to public documentation
  TEMPLATE  a `<placeholder>` pattern path (e.g. recipes/<model>/...); a valid
            shape citation, reported but not resolved

MISSING and INTERNAL are failures: those claims are not linked to public Dynamo
documentation. RESOLVED and TEMPLATE pass.

Usage:
    python3 check-citations.py --repo /path/to/dynamo <skill-dir>
    python3 check-citations.py --repo /path/to/dynamo --json <skill-dir>

Exit code: 1 if any MISSING or INTERNAL citation, else 0. Output uses
PASS / FAIL / WARN lines on stdout for the agent to parse.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Top-level Dynamo trees a skill cites as documentation/source, plus the two
# root manifest files the skills pin per release.
DOC_TREES = (
    "components",
    "lib",
    "deploy",
    "docs",
    "examples",
    "recipes",
    "benchmarks",
    "container",
    "launch",
    "tests",
)
ROOT_FILES = ("Cargo.toml", "pyproject.toml")

_TREES = "|".join(DOC_TREES)
_ROOTS = "|".join(re.escape(f) for f in ROOT_FILES)
_CHARS = r"[A-Za-z0-9._/<>+-]+"

# A cited Dynamo path is either:
#   1. a relative-prefixed link to a repo tree   (../../../docs/dgdr.md)
#   2. a repo-relative tree path at a clean token boundary  (`docs/dgdr.md`)
#   3. a bare root manifest                       (container/context.yaml is
#      covered by the tree form; Cargo.toml / pyproject.toml are bare)
# The clean-boundary lookbehind excludes `\w`, `/`, `.` and `-` so that a
# cross-skill reference like `dynamo-deploy/SKILL.md` is NOT mistaken for the
# Dynamo `deploy/` tree.
PATH_RE = re.compile(
    r"(?:\.{1,2}/)+(?:"
    + _TREES
    + r")/"
    + _CHARS
    + r"|(?<![\w/.-])(?:"
    + _TREES
    + r")/"
    + _CHARS
    + r"|(?<![\w/.-])(?:"
    + _ROOTS
    + r")\b"
)

# Non-public authoring artifacts: citing these means the fact is linked to an
# internal manifest, not to the documentation it distills.
INTERNAL_RE = re.compile(
    r"citations\.md|DYNAMO_REPO_SURVEY|HANDOFF\.md|dynamo-skills/|SKILL_AUTHORING"
)

# Blank triple-fenced code blocks (example manifests etc.) so their contents
# are not mistaken for citations, preserving line numbers.
FENCE_RE = re.compile(r"```.*?```", re.S)

_REL_PREFIX = re.compile(r"^(?:\.{1,2}/)+")


def is_citation_like(raw: str) -> bool:
    """Filter out tokens that look like a repo path but are not citations.

    A bare single segment with no file extension is almost always a kubectl
    resource selector (``kubectl logs deploy/dynamo-operator``) or a rhetorical
    ``X/Y`` phrase (``the deploy/validate sequence``), not a doc/source path. A
    real citation has a file extension, a nested path, or a trailing slash.
    """
    stripped = _REL_PREFIX.sub("", raw)
    if "/" not in stripped:  # bare root manifest (Cargo.toml, pyproject.toml)
        return True
    if stripped.endswith("/"):  # explicit directory citation
        return True
    rest = stripped.split("/", 1)[1]
    if "." in rest.split("/")[-1]:  # has a file extension
        return True
    return "/" in rest  # nested path (>= 2 segments under the tree)


def classify(repo: Path, raw: str) -> str:
    if "<" in raw or ">" in raw:
        return "TEMPLATE"
    rel = _REL_PREFIX.sub("", raw)
    target = repo / rel
    if raw.endswith("/"):
        return "RESOLVED" if target.is_dir() else "MISSING"
    if target.exists() or target.is_dir():
        return "RESOLVED"
    return "MISSING"


def check(repo: Path, skill_dir: Path):
    md_files = [skill_dir / "SKILL.md"]
    refs = skill_dir / "references"
    if refs.is_dir():
        md_files += sorted(refs.glob("*.md"))

    findings = []
    counts = {"RESOLVED": 0, "MISSING": 0, "INTERNAL": 0, "TEMPLATE": 0}
    seen = set()
    for f in md_files:
        if not f.is_file():
            continue
        rel = str(f.relative_to(skill_dir))
        blanked = FENCE_RE.sub(
            lambda m: "\n" * m.group(0).count("\n"),
            f.read_text(encoding="utf-8", errors="replace"),
        )

        # INTERNAL: one finding per line that cites a non-public artifact.
        for i, ln in enumerate(blanked.splitlines(), 1):
            m = INTERNAL_RE.search(ln)
            if m:
                counts["INTERNAL"] += 1
                findings.append((rel, i, "INTERNAL", m.group(0)))

        # Resolvable Dynamo doc/source citations.
        for m in PATH_RE.finditer(blanked):
            raw = m.group(0).rstrip(").,:;`'\"")
            if not is_citation_like(raw):
                continue
            if INTERNAL_RE.search(raw):
                continue  # already counted by the INTERNAL line scan
            lineno = blanked[: m.start()].count("\n") + 1
            verdict = classify(repo, raw)
            counts[verdict] += 1
            key = (verdict, raw)
            if verdict == "MISSING" or key not in seen:
                findings.append((rel, lineno, verdict, raw))
            seen.add(key)
    return findings, counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Dynamo skill citation-resolution linter")
    ap.add_argument(
        "--repo",
        default=os.environ.get("DYNAMO", os.path.expanduser("~/dynamo")),
        help="Path to a Dynamo checkout to resolve citations against",
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "skill_dir", help="Path to the skill directory under .agents/skills/"
    )
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    skill_dir = Path(args.skill_dir).resolve()
    if not (repo / "deploy").is_dir() and not (repo / "components").is_dir():
        print(f"WARN|repo|{repo} does not look like a Dynamo checkout", flush=True)
    if not skill_dir.is_dir():
        print(f"FAIL|skill-dir|{skill_dir} is not a directory", flush=True)
        return 1

    findings, counts = check(repo, skill_dir)

    if args.json:
        print(
            json.dumps(
                {
                    "counts": counts,
                    "findings": [
                        {"file": f, "line": ln, "verdict": v, "path": p}
                        for (f, ln, v, p) in findings
                    ],
                },
                indent=2,
            )
        )
    else:
        for f, ln, verdict, raw in sorted(findings, key=lambda x: (x[2], x[0], x[1])):
            tag = "FAIL" if verdict in ("MISSING", "INTERNAL") else "PASS"
            print(f"{tag}|{verdict}|{f}:{ln} {raw}", flush=True)
        print(
            f"\nCitations: {counts['RESOLVED']} resolved, "
            f"{counts['MISSING']} missing, {counts['INTERNAL']} internal, "
            f"{counts['TEMPLATE']} template",
            flush=True,
        )
    return 1 if (counts["MISSING"] or counts["INTERNAL"]) else 0


if __name__ == "__main__":
    sys.exit(main())
