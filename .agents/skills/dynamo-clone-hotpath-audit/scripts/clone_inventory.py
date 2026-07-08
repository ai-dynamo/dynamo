#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank Rust .clone() call sites for Dynamo hot-path clone audits."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_PATHS = ("lib", "crates", "components")

CLONE_RE = re.compile(r"(?:\bArc::clone\s*\(|\.clone\s*\()")

TEST_PATH_MARKERS = (
    "/tests/",
    "/benches/",
    "/examples/",
    "/bench/",
    "_test.rs",
    "tests.rs",
)

HOT_PATH_RULES = (
    ("lib/llm/src/backend", 10, "llm request backend"),
    ("lib/llm/src/preprocessor", 9, "llm request preprocessing"),
    ("lib/llm/src/http", 8, "http request path"),
    ("lib/llm/src/grpc", 8, "grpc request path"),
    ("lib/llm/src/protocols", 7, "protocol conversion"),
    ("lib/llm/src/migration", 7, "request retry/migration"),
    ("lib/llm/src/kv_router", 8, "llm kv routing"),
    ("lib/llm/src/block_manager", 8, "block manager path"),
    ("lib/bindings/kvbm/src/block_manager", 8, "kvbm binding block manager"),
    ("lib/kvbm-engine/src/offload", 8, "kvbm offload path"),
    ("lib/kvbm-logical/src", 7, "kvbm logical manager"),
    ("lib/kvbm-physical/src", 7, "kvbm physical manager"),
    ("lib/kv-router/src/scheduling", 9, "kv-router scheduler"),
    ("lib/kv-router/src/sequences", 8, "kv-router sequence tracker"),
    ("lib/kv-router/src/indexer", 8, "kv-router indexer"),
    ("lib/runtime/src/component", 7, "runtime component path"),
    ("lib/runtime/src/pipeline", 7, "runtime pipeline path"),
    ("lib/runtime/src/transports", 7, "runtime transport path"),
    ("lib/runtime/src/storage", 6, "runtime storage path"),
)

CHEAP_HINTS = (
    "Arc::clone",
    "CancellationToken",
    "cancel_token",
    "shutdown_token",
    "sender",
    "receiver",
    "_tx",
    "_rx",
    "watch::",
    "Handle",
    "span.clone",
    "metrics",
)

DEEP_VALUE_HINTS = (
    "tokens",
    "token_ids",
    "token_block",
    "token_chunks",
    "request",
    "response",
    "event",
    "payload",
    "metadata",
    "blocks",
    "block_hashes",
    "scores",
    "overlap",
    "runtime_data",
    "annotations",
    "messages",
    "content",
    "schema",
    "config",
)

BOUNDARY_HINTS = (
    "tokio::spawn",
    "spawn_blocking",
    ".send(",
    ".try_send(",
    ".instrument(",
    ".map_err(",
    "async move",
)


@dataclass(frozen=True)
class CloneCandidate:
    score: int
    tier: str
    category: str
    path: str
    line: int
    hot_path: str
    reasons: list[str]
    text: str


def require_under_root(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        msg = f"refusing to scan path outside repository root: {path}"
        raise SystemExit(msg) from exc
    return resolved_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root. Defaults to current working directory.",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=DEFAULT_PATHS,
        help="Files or directories to scan, relative to --root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=120,
        help="Maximum candidates to print. Use 0 for all.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--only-actionable",
        action="store_true",
        help="Hide cheap shared-ownership and test-only clones.",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Keep test, benchmark, and example clone sites in normal scoring.",
    )
    return parser.parse_args()


def rust_files(root: Path, paths: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for raw_path in paths:
        path = require_under_root(root, root / raw_path)
        if path.is_file() and path.suffix == ".rs":
            files.add(path)
        elif path.is_dir():
            files.update(p for p in path.rglob("*.rs") if p.is_file())
    return sorted(files)


def rel_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def is_test_path(path: str) -> bool:
    normalized = f"/{path}"
    return any(marker in normalized for marker in TEST_PATH_MARKERS)


def hot_path_score(path: str) -> tuple[int, str]:
    for prefix, score, label in HOT_PATH_RULES:
        if path.startswith(prefix):
            return score, label
    return 2, "not a known hot path"


def context_window(lines: list[str], index: int) -> str:
    start = max(0, index - 5)
    end = min(len(lines), index + 3)
    return "\n".join(lines[start:end])


def has_loop_context(window: str) -> bool:
    return bool(re.search(r"^\s*(for|while|loop)\b", window, re.MULTILINE))


def score_candidate(
    path: str, line: str, window: str, include_tests: bool
) -> CloneCandidate:
    base_score, hot_path = hot_path_score(path)
    score = base_score
    reasons = [hot_path]

    stripped = line.strip()
    path_is_test = is_test_path(path)
    is_cold = path_is_test and not include_tests
    is_cheap = any(hint in stripped for hint in CHEAP_HINTS)
    is_loop = has_loop_context(window)
    is_clone_into_arc = "Arc::new(" in stripped and ".clone" in stripped

    if is_cold:
        score -= 8
        reasons.append("test, benchmark, or example path")

    if is_cheap:
        score -= 5
        reasons.append("looks like shared-handle or control-plane clone")

    if any(hint in stripped for hint in DEEP_VALUE_HINTS):
        score += 3
        reasons.append("name suggests non-trivial owned data")

    if is_loop:
        score += 4
        reasons.append("clone appears inside or near a loop")

    if is_clone_into_arc:
        score += 5
        reasons.append("owned value may be cloned before Arc wrapping")

    if ".clone()" in stripped and any(hint in window for hint in BOUNDARY_HINTS):
        score -= 2
        reasons.append("near async/task/channel boundary; ownership may be required")

    if re.search(r"let\s+\w+\s*=\s*\w+\.clone\(\)\s*;", stripped):
        score += 2
        reasons.append("simple clone assignment; check whether final use can move")

    if ".clone()." in stripped or (
        ".clone()" in stripped and "unwrap_or_else" in stripped
    ):
        score += 1
        reasons.append("clone participates in expression chain")

    if is_cold:
        category = "cold_or_test"
    elif is_cheap:
        category = "cheap_shared"
    elif is_clone_into_arc:
        category = "clone_into_arc"
    elif is_loop:
        category = "loop_clone"
    elif score >= 8:
        category = "suspicious_hotpath"
    else:
        category = "needs_review"

    tier = "high" if score >= 12 else "medium" if score >= 8 else "low"
    return CloneCandidate(
        score=score,
        tier=tier,
        category=category,
        path=path,
        line=0,
        hot_path=hot_path,
        reasons=reasons,
        text=stripped,
    )


def scan_file(root: Path, path: Path, include_tests: bool) -> list[CloneCandidate]:
    rel = rel_path(root, path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    candidates: list[CloneCandidate] = []

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if not CLONE_RE.search(line):
            continue

        candidate = score_candidate(
            rel, line, context_window(lines, index), include_tests
        )
        candidates.append(
            CloneCandidate(
                score=candidate.score,
                tier=candidate.tier,
                category=candidate.category,
                path=rel,
                line=index + 1,
                hot_path=candidate.hot_path,
                reasons=candidate.reasons,
                text=candidate.text,
            )
        )

    return candidates


def scan(root: Path, paths: Iterable[str], include_tests: bool) -> list[CloneCandidate]:
    candidates: list[CloneCandidate] = []
    for path in rust_files(root, paths):
        candidates.extend(scan_file(root, path, include_tests))
    return sorted(candidates, key=lambda c: (-c.score, c.path, c.line))


def filtered(
    candidates: list[CloneCandidate], only_actionable: bool
) -> list[CloneCandidate]:
    if not only_actionable:
        return candidates
    ignored = {"cheap_shared", "cold_or_test"}
    return [candidate for candidate in candidates if candidate.category not in ignored]


def limited(candidates: list[CloneCandidate], limit: int) -> list[CloneCandidate]:
    if limit <= 0:
        return candidates
    return candidates[:limit]


def summarize(candidates: list[CloneCandidate]) -> dict[str, object]:
    by_tier: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for candidate in candidates:
        by_tier[candidate.tier] = by_tier.get(candidate.tier, 0) + 1
        by_category[candidate.category] = by_category.get(candidate.category, 0) + 1
    return {
        "total": len(candidates),
        "by_tier": dict(sorted(by_tier.items())),
        "by_category": dict(sorted(by_category.items())),
    }


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def print_markdown(
    candidates: list[CloneCandidate], all_candidates: list[CloneCandidate]
) -> None:
    summary = summarize(all_candidates)
    print("# Rust Clone Hotpath Inventory")
    print()
    print(f"Total candidates after filters: {summary['total']}")
    print(f"Tier counts: `{summary['by_tier']}`")
    print(f"Category counts: `{summary['by_category']}`")
    print()
    print("| Score | Tier | Category | Location | Reasons | Code |")
    print("|---:|---|---|---|---|---|")
    for candidate in candidates:
        location = f"{candidate.path}:{candidate.line}"
        reasons = "; ".join(candidate.reasons)
        print(
            "| "
            f"{candidate.score} | "
            f"{candidate.tier} | "
            f"{candidate.category} | "
            f"`{markdown_escape(location)}` | "
            f"{markdown_escape(reasons)} | "
            f"`{markdown_escape(candidate.text)}` |"
        )


def print_json(
    candidates: list[CloneCandidate], all_candidates: list[CloneCandidate]
) -> None:
    payload = {
        "summary": summarize(all_candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    all_candidates = filtered(
        scan(root, args.paths, args.include_tests),
        only_actionable=args.only_actionable,
    )
    candidates = limited(all_candidates, args.limit)

    if args.format == "json":
        print_json(candidates, all_candidates)
    else:
        print_markdown(candidates, all_candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
