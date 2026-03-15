#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Dynamo Release Version Bump Script.

Discovery-based version bumper that scans the entire repo for Dynamo version
references and updates them to the new release version. Used by both Phase 1
(release branch prep) and Phase 2 (port to main) workflows.

Usage:
    # Full release: bump all versions to 1.0.0 with backend metadata
    python3 .github/scripts/bump_version.py \\
        --new-version 1.0.0 \\
        --vllm-version 0.15.1 \\
        --sglang-version 0.5.7 \\
        --trtllm-version 1.3.0rc1 \\
        --nixl-version 0.9.0 \\
        --cuda-versions-vllm 12.9,13.0 \\
        --cuda-versions-sglang 12.9,13.0 \\
        --cuda-versions-trtllm 13.0 \\
        --release-date "Feb 15, 2026"

    # Post-release: Helm-only patch (skip containers, wheels, etc.)
    python3 .github/scripts/bump_version.py \\
        --new-version 0.9.0.post1 \\
        --skip-core --skip-containers --skip-docs

    # Dry run (print changes without writing)
    python3 .github/scripts/bump_version.py --new-version 1.0.0 --dry-run

    # Check for stale versions (CI mode)
    python3 .github/scripts/bump_version.py --check --expected-version 0.9.0

Version format conventions:
    Python / container tags / git:  0.9.0.post1  (PEP 440, dot separator)
    Rust crates / Helm charts:      0.9.0-post1  (semver, hyphen separator)
    The script accepts the Python format and auto-converts for Cargo/Helm.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Version format helpers
# ---------------------------------------------------------------------------


def to_semver(version: str) -> str:
    """Convert Python PEP 440 post-release to semver format for Cargo/Helm.

    0.9.0.post1 -> 0.9.0-post1
    1.0.0       -> 1.0.0       (no change for non-post versions)
    """
    return re.sub(r"\.post(\d+)$", r"-post\1", version)


def is_post_release(version: str) -> bool:
    """Check if version is a post-release (e.g., 0.9.0.post1)."""
    return bool(re.search(r"\.post\d+$", version))


# ---------------------------------------------------------------------------
# Regex patterns that identify Dynamo version references
# ---------------------------------------------------------------------------
# Version number pattern: X.Y.Z with optional .postN or -postN suffix
_VER = r"\d+\.\d+\.\d+(?:[.-]post\d+)?"

# Container image tags: ai-dynamo/<any-image>:X.Y.Z
# Broad match — catches any image name under ai-dynamo/ so new images
# (epp-image, snapshot-agent, etc.) are picked up automatically.
IMAGE_TAG_RE = re.compile(
    r"((?:nvcr\.io/nvidia/)?ai-dynamo/[a-z][a-z0-9-]*)" rf":({_VER})"
)

# Wheel filenames and pip install specs:
#   ai_dynamo_runtime-0.7.0-cp310-..., ai-dynamo==0.8.1, ai-dynamo[vllm]==0.8.1.post1
WHEEL_FILE_RE = re.compile(
    r"(ai[_-]dynamo(?:[_-]runtime)?(?:\[[^\]]*\])?)" rf"(==|-)({_VER})"
)

# dynamoVersion: "0.6.0" or "0.9.0.post1" in operator samples
DYNAMO_VERSION_FIELD_RE = re.compile(rf'(dynamoVersion:\s*")({_VER})(")')

# git checkout release/X.Y.Z or release/X.Y.Z.post1
GIT_CHECKOUT_RE = re.compile(rf"(git checkout release/)({_VER})")

# git+https://...@release/X.Y.Z (pip git install URLs)
GIT_REF_RE = re.compile(rf"(@release/)({_VER})")

# DYNAMO_VERSION=X.Y.Z in shell snippets and docs
DYNAMO_VERSION_ENV_RE = re.compile(rf"(DYNAMO_VERSION=)({_VER})")

# Standalone Dynamo image tags without full registry path (e.g., vllm-runtime:0.8.0)
# Kept as an explicit list (unlike IMAGE_TAG_RE) to avoid false positives
# without the ai-dynamo/ prefix as a disambiguator.
SHORT_IMAGE_TAG_RE = re.compile(
    r"((?:vllm|sglang|tensorrtllm)-runtime"
    r"|dynamo-frontend|kubernetes-operator|frontend"
    r"|epp-image|snapshot-agent)"
    rf":({_VER})"
)

# Unified pattern table used by both scan_and_replace and check_stale_versions.
# (regex, replacement_template, version_group_index, track_spans_for_dedup)
_VERSION_PATTERNS: list[tuple[re.Pattern, str, int, bool]] = [
    (IMAGE_TAG_RE, r"\1:{ver}", 2, True),
    (SHORT_IMAGE_TAG_RE, r"\1:{ver}", 2, True),
    (WHEEL_FILE_RE, r"\1\g<2>{ver}", 3, False),
    (DYNAMO_VERSION_FIELD_RE, r"\g<1>{ver}\3", 2, False),
    (GIT_CHECKOUT_RE, r"\g<1>{ver}", 2, False),
    (GIT_REF_RE, r"\g<1>{ver}", 2, False),
    (DYNAMO_VERSION_ENV_RE, r"\g<1>{ver}", 2, False),
]

# ---------------------------------------------------------------------------
# Paths excluded from the catch-all discovery scan
# ---------------------------------------------------------------------------
EXCLUDE_PATTERNS = [
    "**/*.lock",
    "**/go.sum",
    "**/go.mod",
    ".git/**",
    "**/__pycache__/**",
    "**/node_modules/**",
    "**/.venv/**",
    "**/*.pyc",
    "**/*.png",
    "**/*.jpg",
    "**/*.gif",
    "**/*.woff*",
    "**/*.ttf",
    "**/*.ico",
    "**/*.pdf",
    # Auto-generated files handled by regen steps
    "deploy/operator/config/crd/bases/**",
    "deploy/helm/charts/crds/templates/**",
    "deploy/helm/charts/platform/README.md",
    "docs/kubernetes/api-reference.md",
    "deploy/operator/api/v1alpha1/zz_generated.deepcopy.go",
    # Test fixtures with intentional old versions
    "deploy/operator/internal/**/*_test.go",
    "deploy/operator/internal/checkpoint/**",
    "deploy/operator/internal/dynamo/graph_test.go",
    "tests/**",
    # The bump script itself
    ".github/scripts/bump_version.py",
    # Error classification (separate package, has its own version)
    "error_classification/**",
]

# Files that get targeted edits (not catch-all replacement).
# The catch-all scanner skips these; dedicated functions handle them.
TARGETED_EDIT_FILES = frozenset(
    {
        "docs/reference/release-artifacts.md",
        "docs/reference/support-matrix.md",
        "docs/reference/feature-matrix.md",
        "pyproject.toml",
        "Cargo.toml",
        "lib/bindings/python/Cargo.toml",
        "lib/gpu_memory_service/setup.py",
        "deploy/helm/charts/crds/Chart.yaml",
        "deploy/helm/charts/platform/Chart.yaml",
        "deploy/helm/charts/platform/components/operator/Chart.yaml",
        "deploy/helm/charts/snapshot/Chart.yaml",
        "deploy/helm/charts/snapshot/values.yaml",
        # Operator Go source and samples are handled by update_operator_source()
        "deploy/operator/api/v1alpha1/dynamographdeploymentrequest_types.go",
        "deploy/operator/config/samples/nvidia.com_v1alpha1_dynamographdeploymentrequest.yaml",
        "deploy/operator/config/samples/nvidia.com_v1alpha1_dynamocheckpoint.yaml",
    }
)


def _matches_exclude(rel_path: str) -> bool:
    """Robust exclusion check using pathlib-style matching."""
    p = Path(rel_path)
    for pattern in EXCLUDE_PATTERNS:
        if p.match(pattern):
            return True
        # Handle ** prefix patterns
        if pattern.startswith("**/"):
            suffix = pattern[3:]
            if fnmatch.fnmatch(rel_path, f"*{suffix}") or fnmatch.fnmatch(
                p.name, suffix
            ):
                return True
        if fnmatch.fnmatch(rel_path, pattern):
            return True
    return False


def is_binary(filepath: Path) -> bool:
    """Quick check if a file is binary."""
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except (OSError, PermissionError):
        return True


def _iter_version_files(
    repo: Path,
    extra_skip: frozenset[str] = frozenset(),
) -> Iterator[tuple[str, Path, str]]:
    """Yield (rel_path, filepath, content) for all version-relevant files."""
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and d not in ("node_modules", "__pycache__", ".venv")
        ]
        for filename in filenames:
            filepath = Path(dirpath) / filename
            rel_path = str(filepath.relative_to(repo))
            if _matches_exclude(rel_path) or rel_path in extra_skip:
                continue
            if is_binary(filepath):
                continue
            try:
                content = filepath.read_text(encoding="utf-8", errors="surrogateescape")
            except (OSError, UnicodeDecodeError):
                continue
            yield rel_path, filepath, content


# ===================================================================
# Category 1: Core version files (targeted edits)
# ===================================================================


def update_pyproject_toml(
    repo: Path, old_ver: str, new_ver: str, dry_run: bool
) -> list[str]:
    """Update version in pyproject.toml."""
    filepath = repo / "pyproject.toml"
    changes = []
    content = filepath.read_text()
    new_content = content

    # version = "X.Y.Z"
    new_content = re.sub(
        rf'(version\s*=\s*"){re.escape(old_ver)}"',
        rf'\g<1>{new_ver}"',
        new_content,
    )
    # ai-dynamo-runtime==X.Y.Z
    new_content = re.sub(
        rf"(ai-dynamo-runtime==){re.escape(old_ver)}",
        rf"\g<1>{new_ver}",
        new_content,
    )

    if new_content != content:
        changes.append(f"pyproject.toml: version {old_ver} -> {new_ver}")
        if not dry_run:
            filepath.write_text(new_content)
    return changes


def update_cargo_toml(
    repo: Path, old_ver: str, new_ver: str, dry_run: bool
) -> list[str]:
    """Auto-discover and update Cargo.toml files containing the old version.

    Scans all Cargo.toml files in the repo (excluding target/ and vendored
    directories) so newly added workspaces are picked up automatically.
    Uses semver format (0.9.0-post1) for Cargo files.
    """
    changes = []
    old_semver = to_semver(old_ver)
    new_semver = to_semver(new_ver)

    skip_dirs = {"target", ".git", "node_modules", "__pycache__", ".venv"}
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if "Cargo.toml" not in filenames:
            continue
        filepath = Path(dirpath) / "Cargo.toml"
        rel_path = str(filepath.relative_to(repo))
        content = filepath.read_text()
        new_content = re.sub(
            rf'(version\s*=\s*"){re.escape(old_semver)}"',
            rf'\g<1>{new_semver}"',
            content,
        )
        if new_content != content:
            changes.append(f"{rel_path}: version {old_semver} -> {new_semver}")
            if not dry_run:
                filepath.write_text(new_content)
    return changes


def update_gpu_memory_setup(
    repo: Path, old_ver: str, new_ver: str, dry_run: bool
) -> list[str]:
    """Update version in lib/gpu_memory_service/setup.py."""
    filepath = repo / "lib/gpu_memory_service/setup.py"
    if not filepath.exists():
        return []
    content = filepath.read_text()
    new_content = re.sub(
        rf'(version\s*=\s*"){re.escape(old_ver)}"',
        rf'\g<1>{new_ver}"',
        content,
    )
    if new_content != content:
        if not dry_run:
            filepath.write_text(new_content)
        return [f"lib/gpu_memory_service/setup.py: version {old_ver} -> {new_ver}"]
    return []


# ===================================================================
# Category 2: Helm charts (targeted edits)
# ===================================================================


def update_helm_charts(
    repo: Path, old_ver: str, new_ver: str, dry_run: bool
) -> list[str]:
    """Update version fields in Helm Chart.yaml files.

    Handles exact old version, dev-suffixed versions (1.0.0-dev), and
    post-release versions (0.9.0-post1). Uses semver format for Helm.
    """
    changes = []
    new_semver = to_semver(new_ver)
    ver_pattern = r"\d+\.\d+\.\d+(?:-(?:dev|post\d+))?"

    # Chart.yaml files: (relative_path, update_appVersion)
    chart_configs = [
        ("deploy/helm/charts/crds/Chart.yaml", False),
        ("deploy/helm/charts/platform/Chart.yaml", False),
        ("deploy/helm/charts/platform/components/operator/Chart.yaml", True),
        ("deploy/helm/charts/snapshot/Chart.yaml", True),
    ]

    for rel_path, update_app_version in chart_configs:
        filepath = repo / rel_path
        if not filepath.exists():
            continue
        content = filepath.read_text()
        new_content = re.sub(
            rf"(^version:\s*){ver_pattern}",
            rf"\g<1>{new_semver}",
            content,
            count=1,
            flags=re.MULTILINE,
        )
        if update_app_version:
            new_content = re.sub(
                rf'(^appVersion:\s*"){ver_pattern}"',
                rf'\g<1>{new_semver}"',
                new_content,
                flags=re.MULTILINE,
            )
        if new_content != content:
            changes.append(f"{rel_path}: version -> {new_semver}")
            if not dry_run:
                filepath.write_text(new_content)

    # Platform chart: dynamo-operator dependency version
    platform_chart = repo / "deploy/helm/charts/platform/Chart.yaml"
    if platform_chart.exists():
        content = platform_chart.read_text()
        new_content = re.sub(
            rf"(  - name: dynamo-operator\n    version:\s*){ver_pattern}",
            rf"\g<1>{new_semver}",
            content,
        )
        if new_content != content and not dry_run:
            platform_chart.write_text(new_content)

    # Snapshot values.yaml: image tag
    snapshot_values = repo / "deploy/helm/charts/snapshot/values.yaml"
    if snapshot_values.exists():
        content = snapshot_values.read_text()
        new_content = re.sub(
            rf"(repository:\s*nvcr\.io/nvidia/ai-dynamo/snapshot-agent\n\s*tag:\s*){ver_pattern}",
            rf"\g<1>{new_semver}",
            content,
        )
        if new_content != content:
            changes.append(
                "deploy/helm/charts/snapshot/values.yaml: tag -> " + new_semver
            )
            if not dry_run:
                snapshot_values.write_text(new_content)

    return changes


# ===================================================================
# Category 2b: Operator Go source and samples (targeted edits)
# ===================================================================


def update_operator_source(repo: Path, new_ver: str, dry_run: bool) -> list[str]:
    """Update version references in operator Go source and sample files."""
    changes = []

    targets = [
        "deploy/operator/api/v1alpha1/dynamographdeploymentrequest_types.go",
        "deploy/operator/config/samples/nvidia.com_v1alpha1_dynamographdeploymentrequest.yaml",
        "deploy/operator/config/samples/nvidia.com_v1alpha1_dynamocheckpoint.yaml",
    ]

    for rel in targets:
        filepath = repo / rel
        if not filepath.exists():
            continue
        content = filepath.read_text()
        new_content = content

        # Replace any Dynamo image tags
        new_content = IMAGE_TAG_RE.sub(rf"\1:{new_ver}", new_content)
        new_content = SHORT_IMAGE_TAG_RE.sub(rf"\1:{new_ver}", new_content)
        # Replace dynamoVersion: "X.Y.Z"
        new_content = DYNAMO_VERSION_FIELD_RE.sub(rf"\g<1>{new_ver}\3", new_content)

        if new_content != content:
            changes.append(f"{rel}: updated version references -> {new_ver}")
            if not dry_run:
                filepath.write_text(new_content)

    return changes


# ===================================================================
# Category 3: Reference documentation (targeted edits)
# ===================================================================


def update_feature_matrix(repo: Path, new_ver: str, dry_run: bool) -> list[str]:
    """Update the 'Updated for Dynamo vX.Y.Z' line in feature-matrix.md."""
    filepath = repo / "docs/reference/feature-matrix.md"
    if not filepath.exists():
        return []
    content = filepath.read_text()
    new_content = re.sub(
        r"\*Updated for Dynamo v[\d.]+(?:\.post\d+)?\*",
        f"*Updated for Dynamo v{new_ver}*",
        content,
    )
    if new_content != content:
        if not dry_run:
            filepath.write_text(new_content)
        return [f"feature-matrix.md: updated version tag -> v{new_ver}"]
    return []


def update_support_matrix(
    repo: Path,
    new_ver: str,
    dry_run: bool,
    backend_versions: dict[str, str | None] | None = None,
) -> list[str]:
    """Update support-matrix.md for a new release.

    - Removes '*(in progress)*' from the version row
    - Updates the "At a Glance" latest stable release line
    - Adds a new row to the Backend Dependencies table
    """
    filepath = repo / "docs/reference/support-matrix.md"
    if not filepath.exists():
        return []
    content = filepath.read_text()
    original = content
    changes = []
    bv = backend_versions or {}

    # Remove "*(in progress)*" from the released version row
    content = re.sub(
        rf"(\*\*v{re.escape(new_ver)}\*\*)\s*\*\(in progress\)\*",
        r"\1",
        content,
    )

    if all(bv.get(k) for k in ("sglang", "trtllm", "vllm", "nixl")):
        # Update "At a Glance" latest stable release line
        content = re.sub(
            r"(\*\*Latest stable release:\*\* \[v)[\d.]+\S*"
            r"(\]\(https://github\.com/ai-dynamo/dynamo/releases/tag/v)[\d.]+\S*"
            r"(\) --).*",
            rf"\g<1>{new_ver}\g<2>{new_ver}\3"
            rf' SGLang `{bv["sglang"]}` |'
            rf' TensorRT-LLM `{bv["trtllm"]}` |'
            rf' vLLM `{bv["vllm"]}` |'
            rf' NIXL `{bv["nixl"]}`',
            content,
        )
        changes.append(f"support-matrix.md: updated At a Glance for v{new_ver}")

        # Add new row to Backend Dependencies table (after main ToT row)
        if f"| **v{new_ver}**" not in content:
            new_row = (
                f"| **v{new_ver}** "
                f'| `{bv["sglang"]}` '
                f'| `{bv["trtllm"]}` '
                f'| `{bv["vllm"]}` '
                f'| `{bv["nixl"]}` |'
            )
            content = re.sub(
                r"(\| \*\*main \(ToT\)\*\* \|[^\n]+\n)",
                rf"\1{new_row}\n",
                content,
            )
            changes.append(
                f"support-matrix.md: added v{new_ver} to Backend Dependencies"
            )

    if content != original:
        if not dry_run:
            filepath.write_text(content)
    return changes


def update_release_artifacts(
    repo: Path,
    new_ver: str,
    release_date: str | None,
    dry_run: bool,
    backend_versions: dict[str, str | None] | None = None,
) -> list[str]:
    """Update the Current Release section and add history rows in release-artifacts.md."""
    filepath = repo / "docs/reference/release-artifacts.md"
    if not filepath.exists():
        return []
    content = filepath.read_text()
    original = content
    changes = []

    # Update "Current Release: Dynamo vX.Y.Z" header
    content = re.sub(
        r"(## Current Release: Dynamo v)[\d.]+\S*",
        rf"\g<1>{new_ver}",
        content,
    )

    # Update GitHub Release link in current release section
    content = re.sub(
        r"(\*\*GitHub Release:\*\* \[v)[\d.]+\S*(\]\(https://github\.com/ai-dynamo/dynamo/releases/tag/v)[\d.]+\S*(\))",
        rf"\g<1>{new_ver}\g<2>{new_ver}\3",
        content,
    )

    # Update Docs link in current release section
    ver_dashed = new_ver.replace(".", "-")
    content = re.sub(
        r"(\*\*Docs:\*\* \[v)[\d.]+\S*(\]\(https://docs\.nvidia\.com/dynamo/v-)[\d-]+\S*(/?\))",
        rf"\g<1>{new_ver}\g<2>{ver_dashed}\3",
        content,
    )

    # Add row to GitHub Releases table if not already present
    if f"| `v{new_ver}`" not in content:
        date_str = release_date or datetime.now().strftime("%b %d, %Y")
        new_row = (
            f"| `v{new_ver}` | {date_str} "
            f"| [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v{new_ver}) "
            f"| [Docs](https://docs.nvidia.com/dynamo/v-{ver_dashed}/) |"
        )
        # Insert after the table header row
        table_pattern = r"(### GitHub Releases\n\n\| Version \|.*\n\|[-| ]+\n)"
        if re.search(table_pattern, content):
            content = re.sub(
                table_pattern,
                rf"\1{new_row}\n",
                content,
            )
            changes.append(
                f"release-artifacts.md: added v{new_ver} to GitHub Releases table"
            )
        else:
            print(
                "WARNING: Could not find GitHub Releases table pattern in "
                "release-artifacts.md. Skipping table row insertion.",
                file=sys.stderr,
            )

    # Update backend versions in Current Release Container Images table
    # Scoped to "## Current Release" section to avoid touching history tables
    bv = backend_versions or {}
    backend_labels = {"vllm": "vLLM", "sglang": "SGLang", "trtllm": "TRT-LLM"}
    cr_start = content.find("## Current Release")
    if cr_start >= 0:
        # Find the next H2 section after Current Release
        next_h2 = content.find("\n## ", cr_start + 1)
        if next_h2 < 0:
            next_h2 = len(content)
        section = content[cr_start:next_h2]
        for key, label in backend_labels.items():
            ver = bv.get(key)
            if ver:
                section = re.sub(rf"({label} `)v[^`]+(`)", rf"\g<1>v{ver}\2", section)
        content = content[:cr_start] + section + content[next_h2:]
        if bv:
            changes.append(
                "release-artifacts.md: updated backend versions in Current Release"
            )

    if content != original:
        changes.append(f"release-artifacts.md: updated Current Release to v{new_ver}")
        if not dry_run:
            filepath.write_text(content)

    return changes


# ===================================================================
# Category 4: Discovery-based catch-all scan
# ===================================================================


def scan_and_replace(repo: Path, new_ver: str, dry_run: bool) -> list[str]:
    """Walk the repo and replace all Dynamo version patterns with new_ver."""
    changes = []
    for rel_path, filepath, content in _iter_version_files(
        repo, extra_skip=TARGETED_EDIT_FILES
    ):
        new_content = content
        for regex, repl_template, _, _ in _VERSION_PATTERNS:
            new_content = regex.sub(repl_template.format(ver=new_ver), new_content)
        if new_content != content:
            changes.append(f"{rel_path}: updated version references -> {new_ver}")
            if not dry_run:
                filepath.write_text(
                    new_content, encoding="utf-8", errors="surrogateescape"
                )
    return changes


# ===================================================================
# --check mode: scan for stale versions
# ===================================================================


def check_stale_versions(repo: Path, expected_ver: str) -> list[str]:
    """Scan repo for Dynamo version patterns that don't match expected_ver.

    Returns list of stale reference descriptions. Empty list = all clean.
    """
    stale = []
    skip_files = frozenset(
        {
            "docs/reference/release-artifacts.md",
            "docs/reference/support-matrix.md",
        }
    )
    for rel_path, _, content in _iter_version_files(repo, extra_skip=skip_files):
        for lineno, line in enumerate(content.splitlines(), 1):
            reported_spans: set[tuple[int, int]] = set()
            for regex, _, ver_group, track_spans in _VERSION_PATTERNS:
                for m in regex.finditer(line):
                    if track_spans and any(
                        m.start() >= s and m.end() <= e for s, e in reported_spans
                    ):
                        continue
                    ver = m.group(ver_group)
                    if ver != expected_ver and not ver.startswith(expected_ver + "."):
                        stale.append(
                            f"STALE: {rel_path}:{lineno} -- {m.group(0)} (expected {expected_ver})"
                        )
                    if track_spans:
                        reported_spans.add((m.start(), m.end()))
    return stale


# ===================================================================
# Auto-detect current version from pyproject.toml
# ===================================================================


def detect_current_version(repo: Path) -> str:
    """Read the current version from pyproject.toml."""
    pyproject = repo / "pyproject.toml"
    content = pyproject.read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not m:
        print("ERROR: Could not detect version from pyproject.toml", file=sys.stderr)
        sys.exit(1)
    return m.group(1)


# ===================================================================
# Main
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Dynamo release version bump script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--new-version", help="Target version (e.g., 1.0.0 or 0.9.0.post1)"
    )
    parser.add_argument(
        "--old-version", help="Current version to replace (auto-detected if omitted)"
    )
    parser.add_argument(
        "--repo-root", default=".", help="Repository root (default: cwd)"
    )

    # Backend metadata — accepted for forward-compatibility with the release
    # workflow (which always passes them) but not yet consumed by this script.
    # Future work: auto-populate support-matrix rows from these values.
    parser.add_argument("--vllm-version", help="vLLM version (reserved, not yet used)")
    parser.add_argument(
        "--sglang-version", help="SGLang version (reserved, not yet used)"
    )
    parser.add_argument(
        "--trtllm-version", help="TRT-LLM version (reserved, not yet used)"
    )
    parser.add_argument("--nixl-version", help="NIXL version (reserved, not yet used)")
    parser.add_argument(
        "--cuda-versions-vllm", help="CUDA versions for vLLM (reserved, not yet used)"
    )
    parser.add_argument(
        "--cuda-versions-sglang",
        help="CUDA versions for SGLang (reserved, not yet used)",
    )
    parser.add_argument(
        "--cuda-versions-trtllm",
        help="CUDA versions for TRT-LLM (reserved, not yet used)",
    )
    parser.add_argument("--release-date", help="Release date (e.g., 'Feb 15, 2026')")

    # Modes
    parser.add_argument(
        "--dry-run", action="store_true", help="Print changes without writing"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check for stale versions (CI mode)"
    )
    parser.add_argument("--expected-version", help="Expected version for --check mode")

    # Scope controls (for post-releases or selective bumps)
    parser.add_argument(
        "--skip-core",
        action="store_true",
        help="Skip core version files (pyproject.toml, Cargo.toml, setup.py)",
    )
    parser.add_argument(
        "--skip-containers",
        action="store_true",
        help="Skip discovery-based container image tag scan (docs, recipes, examples)",
    )
    parser.add_argument(
        "--skip-helm",
        action="store_true",
        help="Skip Helm Chart.yaml version updates",
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip reference doc updates (support-matrix, feature-matrix, release-artifacts)",
    )

    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()

    # --check mode
    if args.check:
        expected = args.expected_version
        if not expected:
            expected = detect_current_version(repo)
        print(f"Checking for stale version references (expected: {expected})...")
        stale = check_stale_versions(repo, expected)
        if stale:
            for s in stale:
                print(s)
            print(f"\nFound {len(stale)} stale version reference(s).")
            sys.exit(1)
        else:
            print("All version references are up to date.")
            sys.exit(0)

    # Bump mode
    if not args.new_version:
        parser.error("--new-version is required (unless using --check)")

    new_ver = args.new_version
    old_ver = args.old_version or detect_current_version(repo)

    if old_ver == new_ver:
        print(
            f"WARNING: old version ({old_ver}) == new version ({new_ver}). "
            "Nothing to bump.",
            file=sys.stderr,
        )
        sys.exit(0)

    if args.dry_run:
        print(f"DRY RUN: Would bump {old_ver} -> {new_ver}")
    else:
        print(f"Bumping version: {old_ver} -> {new_ver}")

    if is_post_release(new_ver):
        print(f"  Post-release detected: Python={new_ver}, Semver={to_semver(new_ver)}")

    skip_flags = []
    if args.skip_core:
        skip_flags.append("core")
    if args.skip_containers:
        skip_flags.append("containers")
    if args.skip_helm:
        skip_flags.append("helm")
    if args.skip_docs:
        skip_flags.append("docs")
    if skip_flags:
        print(f"  Skipping: {', '.join(skip_flags)}")

    backend_versions = {
        "vllm": args.vllm_version,
        "sglang": args.sglang_version,
        "trtllm": args.trtllm_version,
        "nixl": args.nixl_version,
    }

    all_changes: list[str] = []

    # Category 1: Core version files
    if not args.skip_core:
        print("\n=== Category 1: Core version files ===")
        all_changes.extend(update_pyproject_toml(repo, old_ver, new_ver, args.dry_run))
        all_changes.extend(update_cargo_toml(repo, old_ver, new_ver, args.dry_run))
        all_changes.extend(
            update_gpu_memory_setup(repo, old_ver, new_ver, args.dry_run)
        )
    else:
        print("\n=== Category 1: Core version files [SKIPPED] ===")

    # Category 2: Helm charts
    if not args.skip_helm:
        print("\n=== Category 2: Helm charts ===")
        all_changes.extend(update_helm_charts(repo, old_ver, new_ver, args.dry_run))
    else:
        print("\n=== Category 2: Helm charts [SKIPPED] ===")

    # Category 2b: Operator Go source and samples
    # Part of the CRD regen chain -- update image tags and dynamoVersion fields
    # in Go doc comments and sample YAMLs before CRD regeneration.
    if not args.skip_containers:
        print("\n=== Category 2b: Operator Go source and samples ===")
        all_changes.extend(update_operator_source(repo, new_ver, args.dry_run))
    else:
        print("\n=== Category 2b: Operator Go source and samples [SKIPPED] ===")

    # Category 3: Reference documentation
    if not args.skip_docs:
        print("\n=== Category 3: Reference documentation ===")
        all_changes.extend(update_feature_matrix(repo, new_ver, args.dry_run))
        all_changes.extend(
            update_support_matrix(repo, new_ver, args.dry_run, backend_versions)
        )
        all_changes.extend(
            update_release_artifacts(
                repo, new_ver, args.release_date, args.dry_run, backend_versions
            )
        )
    else:
        print("\n=== Category 3: Reference documentation [SKIPPED] ===")

    # Category 4: Discovery-based catch-all scan
    if not args.skip_containers:
        print("\n=== Category 4: Discovery-based scan (docs, recipes, examples) ===")
        all_changes.extend(scan_and_replace(repo, new_ver, args.dry_run))
    else:
        print("\n=== Category 4: Discovery-based scan [SKIPPED] ===")

    # Summary
    print(f"\n{'=' * 60}")
    print(
        f"{'DRY RUN ' if args.dry_run else ''}Summary: {len(all_changes)} file(s) changed"
    )
    print(f"{'=' * 60}")
    for change in all_changes:
        print(f"  {'[DRY] ' if args.dry_run else '✅ '}{change}")

    if not all_changes:
        print(
            "WARNING: No files were changed. This may indicate the old version "
            f"({old_ver}) was not found in any target files, or all categories "
            "were skipped.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
