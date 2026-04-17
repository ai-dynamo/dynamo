#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo release version bump script.

A single rule-driven engine that finds every Dynamo version reference in the
repo and updates it to the new release version. Used by both Phase 1 (release
branch prep) and Phase 2 (port to main) workflows.

Design:
    - A single :data:`RULES` table declares every version reference type
      (regex + replacement + target format + scope + file filter). The engine
      walks the repo once, applying every rule that matches. Adding a new
      reference type = one row in :data:`RULES`.
    - A :class:`Version` type owns format conversion (Python PEP 440 form vs.
      Cargo/Helm semver form vs. URL-safe dashed form).
    - Per-file opt-out via the ``bump-version: ignore`` comment marker.
    - ``--check`` is just a dry-run against the expected version: any change
      the engine would make = a stale reference.
    - ``--dry-run`` previews without writing.
    - ``--skip-core/containers/helm/docs`` gate which rule categories fire.

Usage:
    # Full release: bump all versions to 1.0.0
    python3 .github/scripts/bump_version.py --new-version 1.0.0 \\
        --vllm-version 0.19.0 --sglang-version 0.5.7 \\
        --trtllm-version 1.3.0rc1 --nixl-version 0.10.1 \\
        --release-date "Feb 15, 2026"

    # Post-release: Helm-only patch
    python3 .github/scripts/bump_version.py --new-version 0.9.0.post1 \\
        --skip-core --skip-containers --skip-docs

    # Dry run
    python3 .github/scripts/bump_version.py --new-version 1.0.0 --dry-run

    # Check for stale versions (CI)
    python3 .github/scripts/bump_version.py --check

Version format conventions:
    Python / container tags / git:  0.9.0.post1  (PEP 440, dot separator)
    Rust crates / Helm charts:      0.9.0-post1  (semver, hyphen separator)
    URL slugs:                      0-9-0-post1  (all dashes)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator

# Per-file opt-out: any file containing this marker is skipped entirely.
# Use it in test fixtures / historical artifacts that must stay on an old version.
IGNORE_MARKER = "bump-version: ignore"


# ---------------------------------------------------------------------------
# Version type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Version:
    """A Dynamo release version.

    Knows its three canonical string forms:
      - :meth:`python`:  ``0.9.0`` / ``0.9.0.post1`` (PEP 440, for Python/images/git)
      - :meth:`semver`:  ``0.9.0`` / ``0.9.0-post1`` (for Cargo/Helm)
      - :meth:`dashed`:  ``0-9-0`` / ``0-9-0-post1`` (for URL slugs)
    """

    major: int
    minor: int
    patch: int
    post: int | None = None

    _PARSE_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[.-]post(\d+))?$")

    @classmethod
    def parse(cls, s: str) -> Version:
        m = cls._PARSE_RE.match(s)
        if not m:
            raise argparse.ArgumentTypeError(
                f"Not a valid version: {s!r}. Expected X.Y.Z or X.Y.Z.postN."
            )
        maj, min_, pat, post = m.groups()
        return cls(
            int(maj), int(min_), int(pat), int(post) if post is not None else None
        )

    def python(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}.post{self.post}" if self.post is not None else base

    def semver(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}-post{self.post}" if self.post is not None else base

    def dashed(self) -> str:
        return self.python().replace(".", "-")

    @property
    def is_post(self) -> bool:
        return self.post is not None

    def _sort_key(self) -> tuple[int, int, int, int]:
        # PEP 440: a base release sorts before any of its post-releases, so
        # treat ``post=None`` as ``-1`` (less than any real post number). The
        # default dataclass ordering would compare ``None`` to ``int`` and
        # raise ``TypeError`` at runtime.
        return (
            self.major,
            self.minor,
            self.patch,
            self.post if self.post is not None else -1,
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._sort_key() < other._sort_key()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._sort_key() <= other._sort_key()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._sort_key() > other._sort_key()

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._sort_key() >= other._sort_key()

    def __str__(self) -> str:
        return self.python()


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class Scope(Enum):
    CORE = "core"  # pyproject, Cargo, setup.py
    CONTAINERS = "containers"  # image tags, git refs, wheel pins, operator samples
    HELM = "helm"  # Chart.yaml, values.yaml
    DOCS = "docs"  # support-matrix, feature-matrix, release-artifacts


# ---------------------------------------------------------------------------
# Rule table
# ---------------------------------------------------------------------------

# Version token: X.Y.Z with optional .postN or -postN suffix.
_VER = r"\d+\.\d+\.\d+(?:[.-]post\d+)?"


def _always(_: Path) -> bool:
    return True


def _name_is(*names: str) -> Callable[[Path], bool]:
    allowed = set(names)
    return lambda p: p.name in allowed


def _endswith(*suffixes: str) -> Callable[[Path], bool]:
    return lambda p: any(p.as_posix().endswith(s) for s in suffixes)


@dataclass(frozen=True)
class Rule:
    """A single version-reference update rule.

    ``replacement`` may use the placeholders ``{python}``, ``{semver}``, and
    ``{dashed}`` — these expand from the :class:`Version` being applied.
    Standard regex backreferences (``\\1``, ``\\g<1>``) are preserved.
    """

    name: str
    scope: Scope
    pattern: re.Pattern[str]
    replacement: str
    file_filter: Callable[[Path], bool] = _always

    def apply(self, content: str, version: Version) -> str:
        repl = (
            self.replacement.replace("{python}", version.python())
            .replace("{semver}", version.semver())
            .replace("{dashed}", version.dashed())
        )
        return self.pattern.sub(repl, content)


RULES: tuple[Rule, ...] = (
    # -- Core version files ---------------------------------------------------
    Rule(
        "pyproject_project_version",
        Scope.CORE,
        re.compile(rf'(^version\s*=\s*"){_VER}(")', re.MULTILINE),
        r"\g<1>{python}\g<2>",
        _name_is("pyproject.toml"),
    ),
    Rule(
        "pyproject_ai_dynamo_pin",
        Scope.CORE,
        # ai-dynamo / ai_dynamo / ai-dynamo-runtime / ai-dynamo[vllm] == X.Y.Z
        re.compile(rf"(ai[_-]dynamo(?:[_-]runtime)?(?:\[[^\]]*\])?==){_VER}"),
        r"\g<1>{python}",
        _name_is("pyproject.toml"),
    ),
    Rule(
        "cargo_package_version",
        Scope.CORE,
        re.compile(rf'(^version\s*=\s*"){_VER}(")', re.MULTILINE),
        r"\g<1>{semver}\g<2>",
        _name_is("Cargo.toml"),
    ),
    Rule(
        "gpu_memory_setup_version",
        Scope.CORE,
        re.compile(rf'(version\s*=\s*"){_VER}(")'),
        r"\g<1>{python}\g<2>",
        _endswith("lib/gpu_memory_service/setup.py"),
    ),
    # -- Helm charts ----------------------------------------------------------
    Rule(
        "helm_chart_version",
        Scope.HELM,
        re.compile(rf"(^version:\s*){_VER}", re.MULTILINE),
        r"\g<1>{semver}",
        _name_is("Chart.yaml"),
    ),
    Rule(
        "helm_chart_appVersion",
        Scope.HELM,
        re.compile(rf'(^appVersion:\s*"){_VER}(")', re.MULTILINE),
        r"\g<1>{semver}\g<2>",
        _name_is("Chart.yaml"),
    ),
    Rule(
        "helm_dependency_dynamo_operator",
        Scope.HELM,
        re.compile(rf"(- name: dynamo-operator\n\s+version:\s*){_VER}"),
        r"\g<1>{semver}",
        _name_is("Chart.yaml"),
    ),
    Rule(
        "helm_snapshot_values_tag",
        Scope.HELM,
        re.compile(
            rf"(repository:\s*nvcr\.io/nvidia/ai-dynamo/snapshot-agent\n\s*tag:\s*){_VER}"
        ),
        r"\g<1>{semver}",
        _endswith("deploy/helm/charts/snapshot/values.yaml"),
    ),
    # -- Container image tags (broad, self-healing) ---------------------------
    Rule(
        "image_tag_ai_dynamo_ns",
        Scope.CONTAINERS,
        re.compile(rf"((?:nvcr\.io/nvidia/)?ai-dynamo/[a-z][a-z0-9-]*):{_VER}"),
        r"\g<1>:{python}",
    ),
    Rule(
        "image_tag_short_dynamo",
        Scope.CONTAINERS,
        re.compile(
            r"((?:vllm|sglang|tensorrtllm)-runtime"
            r"|dynamo-frontend|kubernetes-operator|frontend"
            rf"|epp-image|snapshot-agent):{_VER}"
        ),
        r"\g<1>:{python}",
    ),
    # Wheel filenames and pip install specs in docs/Dockerfiles/shell.
    # pyproject.toml pins are owned by the CORE rule (pyproject_ai_dynamo_pin);
    # excluding pyproject.toml here keeps it consistent when --skip-core is set.
    Rule(
        "pip_wheel_or_pin",
        Scope.CONTAINERS,
        re.compile(rf"(ai[_-]dynamo(?:[_-]runtime)?(?:\[[^\]]*\])?)(==|-){_VER}"),
        r"\g<1>\g<2>{python}",
        lambda p: p.name != "pyproject.toml",
    ),
    # Operator sample YAML dynamoVersion: "X.Y.Z"
    Rule(
        "operator_dynamoVersion_field",
        Scope.CONTAINERS,
        re.compile(rf'(dynamoVersion:\s*"){_VER}(")'),
        r"\g<1>{python}\g<2>",
    ),
    # git checkout release/X.Y.Z
    Rule(
        "git_checkout_release_branch",
        Scope.CONTAINERS,
        re.compile(rf"(git checkout release/){_VER}"),
        r"\g<1>{python}",
    ),
    # pip git URLs: git+https://...@release/X.Y.Z
    Rule(
        "git_url_release_ref",
        Scope.CONTAINERS,
        re.compile(rf"(@release/){_VER}"),
        r"\g<1>{python}",
    ),
    # DYNAMO_VERSION=X.Y.Z env-var assignment
    Rule(
        "env_dynamo_version",
        Scope.CONTAINERS,
        re.compile(rf"(DYNAMO_VERSION=){_VER}"),
        r"\g<1>{python}",
    ),
)


# Files excluded from scanning entirely. These are either auto-generated
# (regenerated by CRD/helm-docs), binary, or lockfiles. Per-file opt-out via
# the IGNORE_MARKER comment is preferred for any new cases.
EXCLUDE_GLOBS: tuple[str, ...] = (
    # Lockfiles / generated / vendored
    "**/*.lock",
    "**/go.sum",
    "**/go.mod",
    ".git/**",
    "**/__pycache__/**",
    "**/node_modules/**",
    "**/.venv/**",
    "**/*.pyc",
    # Binary assets
    "**/*.png",
    "**/*.jpg",
    "**/*.gif",
    "**/*.woff*",
    "**/*.ttf",
    "**/*.ico",
    "**/*.pdf",
    # Auto-generated by regen steps in the workflow
    "deploy/operator/config/crd/bases/**",
    "deploy/helm/charts/crds/templates/**",
    "deploy/helm/charts/platform/README.md",
    "docs/kubernetes/api-reference.md",
    "deploy/operator/api/v1alpha1/zz_generated.deepcopy.go",
    # Separately-versioned sub-package
    "error_classification/**",
    # The bump script itself (it contains version patterns)
    ".github/scripts/bump_version.py",
)


# ---------------------------------------------------------------------------
# File iteration
# ---------------------------------------------------------------------------


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(8192)
    except OSError:
        return True


def _excluded(rel: Path) -> bool:
    for pat in EXCLUDE_GLOBS:
        if rel.match(pat):
            return True
        # Also accept "**/foo" → match "foo" anywhere in path.
        if pat.startswith("**/") and rel.match(pat[3:]):
            return True
    return False


def iter_repo_files(repo: Path) -> Iterator[tuple[Path, Path, str]]:
    """Yield (abspath, relpath, content) for every text file to consider.

    Skips binary files, excluded paths, non-UTF-8 files, and files carrying
    the :data:`IGNORE_MARKER` opt-out comment.
    """
    for dirpath, dirnames, filenames in os.walk(repo):
        # Prune hidden / noise directories in-place so we don't descend.
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not d.startswith(".")
            and d not in {"node_modules", "__pycache__", ".venv", "target"}
        )
        for filename in filenames:
            path = Path(dirpath) / filename
            rel = path.relative_to(repo)
            if _excluded(rel):
                continue
            if _is_binary(path):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                # Not valid UTF-8 => definitely not a version-relevant text file.
                continue
            if IGNORE_MARKER in content:
                continue
            yield path, rel, content


# ---------------------------------------------------------------------------
# Engine: apply rules
# ---------------------------------------------------------------------------


@dataclass
class Change:
    path: Path  # relative to repo root
    scope: Scope
    rules: list[str] = field(default_factory=list)


def apply_rules(
    repo: Path,
    version: Version,
    active_scopes: set[Scope],
    dry_run: bool = False,
) -> list[Change]:
    """Walk the repo, apply every matching rule, collect :class:`Change` records.

    When ``dry_run`` is True, files are not written — used by both ``--dry-run``
    and ``--check`` modes.
    """
    changes: list[Change] = []
    for abs_path, rel_path, content in iter_repo_files(repo):
        new_content = content
        hit_by_scope: dict[Scope, list[str]] = {}
        for rule in RULES:
            if rule.scope not in active_scopes:
                continue
            if not rule.file_filter(rel_path):
                continue
            updated = rule.apply(new_content, version)
            if updated != new_content:
                hit_by_scope.setdefault(rule.scope, []).append(rule.name)
                new_content = updated
        if new_content != content:
            for scope, rule_names in hit_by_scope.items():
                changes.append(Change(rel_path, scope, rule_names))
            if not dry_run:
                abs_path.write_text(new_content, encoding="utf-8")
    return changes


# ---------------------------------------------------------------------------
# Backend metadata (consumed by DOCS specialised functions)
# ---------------------------------------------------------------------------


@dataclass
class BackendVersions:
    vllm: str | None = None
    sglang: str | None = None
    trtllm: str | None = None
    nixl: str | None = None

    def all_set(self) -> bool:
        return all([self.vllm, self.sglang, self.trtllm, self.nixl])

    def any_set(self) -> bool:
        return any([self.vllm, self.sglang, self.trtllm, self.nixl])


# ---------------------------------------------------------------------------
# DOCS specialised updates (table-row insertion / section-scoped edits)
# ---------------------------------------------------------------------------


def update_feature_matrix(repo: Path, version: Version, dry_run: bool) -> list[Change]:
    path = repo / "docs/reference/feature-matrix.md"
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    updated = re.sub(
        r"\*Updated for Dynamo v[\d.]+(?:\.post\d+)?\*",
        f"*Updated for Dynamo v{version.python()}*",
        content,
    )
    if updated == content:
        return []
    if not dry_run:
        path.write_text(updated, encoding="utf-8")
    return [
        Change(
            Path("docs/reference/feature-matrix.md"),
            Scope.DOCS,
            ["feature_matrix_tag"],
        )
    ]


def update_support_matrix(
    repo: Path,
    version: Version,
    backends: BackendVersions,
    dry_run: bool,
) -> list[Change]:
    path = repo / "docs/reference/support-matrix.md"
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    original = content
    rules_hit: list[str] = []

    # Remove "*(in progress)*" from the version row being released.
    before = content
    content = re.sub(
        rf"(\*\*v{re.escape(version.python())}\*\*)\s*\*\(in progress\)\*",
        r"\g<1>",
        content,
    )
    if content != before:
        rules_hit.append("support_matrix_in_progress")

    if backends.all_set():
        before = content
        content = re.sub(
            r"(\*\*Latest stable release:\*\* \[v)[\d.]+\S*"
            r"(\]\(https://github\.com/ai-dynamo/dynamo/releases/tag/v)[\d.]+\S*"
            r"(\) --).*",
            (
                rf"\g<1>{version.python()}\g<2>{version.python()}\g<3>"
                rf" SGLang `{backends.sglang}` |"
                rf" TensorRT-LLM `{backends.trtllm}` |"
                rf" vLLM `{backends.vllm}` |"
                rf" NIXL `{backends.nixl}`"
            ),
            content,
        )
        if content != before:
            rules_hit.append("support_matrix_at_a_glance")

        row_marker = f"| **v{version.python()}**"
        if row_marker not in content:
            new_row = (
                f"| **v{version.python()}** "
                f"| `{backends.sglang}` "
                f"| `{backends.trtllm}` "
                f"| `{backends.vllm}` "
                f"| `{backends.nixl}` |"
            )
            before = content
            content = re.sub(
                r"(\| \*\*main \(ToT\)\*\* \|[^\n]+\n)",
                rf"\g<1>{new_row}\n",
                content,
            )
            if content != before:
                rules_hit.append("support_matrix_backend_row")

    if content == original:
        return []
    if not dry_run:
        path.write_text(content, encoding="utf-8")
    return [Change(Path("docs/reference/support-matrix.md"), Scope.DOCS, rules_hit)]


def update_release_artifacts(
    repo: Path,
    version: Version,
    release_date: str | None,
    backends: BackendVersions,
    dry_run: bool,
) -> list[Change]:
    path = repo / "docs/reference/release-artifacts.md"
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    original = content
    rules_hit: list[str] = []

    # Header
    before = content
    content = re.sub(
        r"(## Current Release: Dynamo v)[\d.]+\S*",
        rf"\g<1>{version.python()}",
        content,
    )
    if content != before:
        rules_hit.append("release_artifacts_header")

    # GitHub Release link
    content = re.sub(
        r"(\*\*GitHub Release:\*\* \[v)[\d.]+\S*"
        r"(\]\(https://github\.com/ai-dynamo/dynamo/releases/tag/v)[\d.]+\S*(\))",
        rf"\g<1>{version.python()}\g<2>{version.python()}\g<3>",
        content,
    )

    # Docs link
    content = re.sub(
        r"(\*\*Docs:\*\* \[v)[\d.]+\S*"
        r"(\]\(https://docs\.nvidia\.com/dynamo/v-)[\d-]+\S*(/?\))",
        rf"\g<1>{version.python()}\g<2>{version.dashed()}\g<3>",
        content,
    )

    # New row in the GitHub Releases table
    row_marker = f"| `v{version.python()}`"
    if row_marker not in content:
        date_str = release_date or datetime.now().strftime("%b %d, %Y")
        new_row = (
            f"| `v{version.python()}` | {date_str} "
            f"| [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v{version.python()}) "
            f"| [Docs](https://docs.nvidia.com/dynamo/v-{version.dashed()}/) |"
        )
        table_pattern = r"(### GitHub Releases\n\n\| Version \|.*\n\|[-| ]+\n)"
        if re.search(table_pattern, content):
            content = re.sub(table_pattern, rf"\g<1>{new_row}\n", content)
            rules_hit.append("release_artifacts_table_row")
        else:
            # Fail loud: the doc structure changed and we'd silently skip the row.
            raise RuntimeError(
                "release-artifacts.md: could not find the GitHub Releases table "
                "header. The doc structure changed — update this script."
            )

    # Backend versions inside the Current Release section only.
    backend_labels = {"vllm": "vLLM", "sglang": "SGLang", "trtllm": "TRT-LLM"}
    cr_start = content.find("## Current Release")
    if cr_start >= 0 and backends.any_set():
        next_h2 = content.find("\n## ", cr_start + 1)
        if next_h2 < 0:
            next_h2 = len(content)
        section = content[cr_start:next_h2]
        before = section
        for key, label in backend_labels.items():
            val = getattr(backends, key)
            if val:
                section = re.sub(
                    rf"({re.escape(label)} `)v[^`]+(`)",
                    rf"\g<1>v{val}\g<2>",
                    section,
                )
        if section != before:
            content = content[:cr_start] + section + content[next_h2:]
            rules_hit.append("release_artifacts_backend_versions")

    if content == original:
        return []
    if not dry_run:
        path.write_text(content, encoding="utf-8")
    return [Change(Path("docs/reference/release-artifacts.md"), Scope.DOCS, rules_hit)]


# ---------------------------------------------------------------------------
# Detect current version
# ---------------------------------------------------------------------------


def detect_current_version(repo: Path) -> Version:
    pyproject = repo / "pyproject.toml"
    m = re.search(
        r'^version\s*=\s*"([^"]+)"',
        pyproject.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not m:
        raise SystemExit("ERROR: could not detect current version from pyproject.toml")
    return Version.parse(m.group(1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--new-version",
        type=Version.parse,
        help="Target version (e.g. 1.0.0 or 0.9.0.post1)",
    )
    p.add_argument(
        "--old-version",
        type=Version.parse,
        help="Current version to replace (auto-detected from pyproject.toml if omitted)",
    )
    p.add_argument("--repo-root", default=".", help="Repository root (default: cwd)")

    # Backend metadata consumed by DOCS functions
    p.add_argument("--vllm-version", help="vLLM backend version (e.g. 0.19.0)")
    p.add_argument("--sglang-version", help="SGLang backend version")
    p.add_argument("--trtllm-version", help="TensorRT-LLM backend version")
    p.add_argument("--nixl-version", help="NIXL backend version")
    p.add_argument("--release-date", help="Release date (e.g. 'Feb 15, 2026')")

    # Modes
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without writing",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Check for stale version references (exit 1 if any). "
        "Uses --expected-version or auto-detects from pyproject.toml.",
    )
    p.add_argument(
        "--expected-version",
        type=Version.parse,
        help="Expected current version for --check mode",
    )

    # Scope gating
    p.add_argument(
        "--skip-core",
        action="store_true",
        help="Skip core version files (pyproject.toml, Cargo.toml, setup.py)",
    )
    p.add_argument(
        "--skip-containers",
        action="store_true",
        help="Skip container image tags, git refs, operator samples, wheel pins",
    )
    p.add_argument(
        "--skip-helm",
        action="store_true",
        help="Skip Helm Chart.yaml and values.yaml updates",
    )
    p.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip reference docs (support-matrix, feature-matrix, release-artifacts)",
    )

    # Verbosity
    p.add_argument("-v", "--verbose", action="store_true", help="Log every rule hit")
    p.add_argument(
        "-q", "--quiet", action="store_true", help="Log only the final summary"
    )

    # Machine-readable output
    p.add_argument(
        "--summary-file",
        help="Write a markdown summary of changes to this path (e.g. $GITHUB_STEP_SUMMARY)",
    )
    return p


def _active_scopes(args: argparse.Namespace) -> set[Scope]:
    scopes = set(Scope)
    if args.skip_core:
        scopes.discard(Scope.CORE)
    if args.skip_containers:
        scopes.discard(Scope.CONTAINERS)
    if args.skip_helm:
        scopes.discard(Scope.HELM)
    if args.skip_docs:
        scopes.discard(Scope.DOCS)
    return scopes


def _log(args: argparse.Namespace, level: str, msg: str) -> None:
    if args.quiet and level != "summary":
        return
    if level == "verbose" and not args.verbose:
        return
    print(msg)


def _format_summary(
    version: Version,
    old: Version | None,
    changes: list[Change],
    active_scopes: set[Scope],
    dry_run: bool,
) -> str:
    lines = []
    header = f"Bumped {old} -> {version}" if old else f"Bumped to {version}"
    if dry_run:
        header = f"[dry-run] {header}"
    lines.append(f"## Version bump: {header}")
    lines.append("")
    lines.append(f"**Scopes:** {', '.join(sorted(s.value for s in active_scopes))}")
    lines.append("")
    lines.append(f"**Changed files:** {len(changes)}")
    lines.append("")
    if changes:
        lines.append("| File | Scope | Rules |")
        lines.append("|---|---|---|")
        for ch in sorted(changes, key=lambda c: (c.scope.value, c.path.as_posix())):
            lines.append(
                f"| `{ch.path.as_posix()}` | {ch.scope.value} | {', '.join(ch.rules)} |"
            )
    return "\n".join(lines) + "\n"


def _run_check(args: argparse.Namespace, repo: Path) -> int:
    expected = args.expected_version or detect_current_version(repo)
    _log(
        args, "info", f"Checking for stale version references (expected: {expected})..."
    )
    active = _active_scopes(args)
    # Rerun the engine with the expected version — anything that would change is stale.
    stale = apply_rules(repo, expected, active, dry_run=True)
    # The DOCS specialised functions own table-row insertion and section-scoped
    # edits in support-matrix.md / release-artifacts.md / feature-matrix.md.
    # Those files now carry the bump-version: ignore marker, so apply_rules() no
    # longer touches them; without this dry-run pass --check would have a blind
    # spot for stale "Current Release" headers, "Latest stable release" lines,
    # missing release-row entries, and stale "Updated for vX.Y.Z" tags.
    # Backend versions / release date are usually not passed to --check; the
    # all_set/any_set guards inside the DOCS helpers keep the
    # backend-dependent rewrites no-ops in that case, while still catching the
    # version-only staleness signals.
    if Scope.DOCS in active:
        backends = BackendVersions(
            vllm=args.vllm_version,
            sglang=args.sglang_version,
            trtllm=args.trtllm_version,
            nixl=args.nixl_version,
        )
        stale.extend(update_feature_matrix(repo, expected, dry_run=True))
        stale.extend(update_support_matrix(repo, expected, backends, dry_run=True))
        stale.extend(
            update_release_artifacts(
                repo, expected, args.release_date, backends, dry_run=True
            )
        )
    if not stale:
        _log(args, "summary", "All version references are up to date.")
        return 0
    for ch in stale:
        _log(
            args,
            "info",
            f"STALE: {ch.path.as_posix()} ({ch.scope.value}: {', '.join(ch.rules)})",
        )
    _log(args, "summary", f"Found {len(stale)} stale version reference(s).")
    return 1


def _run_bump(args: argparse.Namespace, repo: Path) -> int:
    if not args.new_version:
        raise SystemExit("ERROR: --new-version is required (unless using --check)")

    new_ver: Version = args.new_version
    old_ver: Version = args.old_version or detect_current_version(repo)
    if old_ver == new_ver:
        _log(
            args,
            "summary",
            f"WARNING: old ({old_ver}) == new ({new_ver}). Nothing to bump.",
        )
        return 0

    active = _active_scopes(args)
    _log(
        args,
        "info",
        f"{'DRY RUN: Would bump' if args.dry_run else 'Bumping'} "
        f"{old_ver} -> {new_ver}  (scopes: {', '.join(sorted(s.value for s in active))})",
    )
    if new_ver.is_post:
        _log(
            args,
            "info",
            f"  Post-release formats: python={new_ver.python()} semver={new_ver.semver()}",
        )

    backends = BackendVersions(
        vllm=args.vllm_version,
        sglang=args.sglang_version,
        trtllm=args.trtllm_version,
        nixl=args.nixl_version,
    )

    changes: list[Change] = []
    # Rule-based scopes (core, containers, helm, + any text-pattern DOCS rules)
    changes.extend(apply_rules(repo, new_ver, active, dry_run=args.dry_run))
    # Specialised DOCS functions (table-row insertion / sectioned edits)
    if Scope.DOCS in active:
        changes.extend(update_feature_matrix(repo, new_ver, args.dry_run))
        changes.extend(update_support_matrix(repo, new_ver, backends, args.dry_run))
        changes.extend(
            update_release_artifacts(
                repo, new_ver, args.release_date, backends, args.dry_run
            )
        )

    _log(
        args,
        "summary",
        f"{'[dry-run] ' if args.dry_run else ''}Changed {len(changes)} file(s)",
    )
    for ch in sorted(changes, key=lambda c: (c.scope.value, c.path.as_posix())):
        _log(
            args,
            "info",
            f"  {ch.scope.value}: {ch.path.as_posix()}  ({', '.join(ch.rules)})",
        )

    if args.summary_file:
        Path(args.summary_file).write_text(
            _format_summary(new_ver, old_ver, changes, active, args.dry_run),
            encoding="utf-8",
        )

    if not changes:
        _log(
            args,
            "summary",
            f"WARNING: no changes produced. The old version ({old_ver}) may not "
            "appear in any files, or every scope was skipped.",
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo = Path(args.repo_root).resolve()
    if args.check:
        return _run_check(args, repo)
    return _run_bump(args, repo)


if __name__ == "__main__":
    sys.exit(main())
