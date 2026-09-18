# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line integration for the shared API-surface tracker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .ledger import (
    deprecations,
    load_ledger,
    load_provenance,
    merge_changes,
    save_ledger,
)
from .models import SurfaceChange, SurfaceSnapshot
from .pr_analysis import IMPACT_FORMATS, analyze_impact, render_impact
from .results import OpError
from .snapshot import (
    build_snapshot,
    default_ledger_path,
    default_provenance_path,
    default_snapshot_path,
    gather_changes,
    load_snapshot,
    save_snapshot,
)
from .suppressions import default_suppressions_path, load_suppressions
from .validate import validate_ledger


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Attach API-surface subcommands to the repository's root parser."""
    commands = parser.add_subparsers(dest="api_surface_command", required=True)

    extract = commands.add_parser("extract", help="capture a source tree's public API")
    extract.add_argument("--repo", type=Path, required=True)
    extract.add_argument("--release", required=True)
    extract.add_argument("--ref", default="")
    extract.add_argument("--repo-name", default="ai-dynamo/dynamo")
    extract.add_argument("--out", "--json", dest="out", type=Path)

    diff = commands.add_parser("diff", help="compare two API snapshots")
    diff.add_argument("--old", type=Path, required=True)
    diff.add_argument("--new", type=Path, required=True)
    diff.add_argument("--repo", type=Path)
    diff.add_argument("--release", default="")
    diff.add_argument("--no-markers", action="store_true")
    diff.add_argument("--json", type=Path)

    update = commands.add_parser(
        "update-ledger", help="merge a release into the ledger"
    )
    update.add_argument("--new", type=Path, required=True)
    update.add_argument("--release", required=True)
    update.add_argument("--old", type=Path)
    update.add_argument("--repo", type=Path)
    update.add_argument("--no-markers", action="store_true")
    update.add_argument("--ledger", type=Path)
    update.add_argument("--provenance", type=Path)
    update.add_argument("--dry-run", action="store_true")

    listed = commands.add_parser(
        "list-deprecations",
        help="list the ledger's current deprecations",
    )
    listed.add_argument("--repo", type=Path, default=Path("."))
    listed.add_argument("--ledger", type=Path)
    listed.add_argument("--include-unconfirmed", action="store_true")
    listed.add_argument("--json", type=Path)

    validate = commands.add_parser("validate", help="enforce stable API promises")
    validate.add_argument("--repo", type=Path, default=Path("."))
    validate.add_argument("--ledger", type=Path)
    validate.add_argument("--suppressions", type=Path)
    validate.add_argument("--json", type=Path)

    analyze = commands.add_parser(
        "analyze-pr", help="classify a base-to-head API delta"
    )
    analyze.add_argument("--base-repo", type=Path)
    analyze.add_argument("--head-repo", type=Path)
    analyze.add_argument("--base-snapshot", type=Path)
    analyze.add_argument("--head-snapshot", type=Path)
    analyze.add_argument("--base-ref", default="")
    analyze.add_argument("--head-ref", default="")
    analyze.add_argument("--release", default="0.0.0")
    analyze.add_argument("--format", choices=sorted(IMPACT_FORMATS), default="md")
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--markers", action="store_true")
    analyze.add_argument("--fail-on-breaking", action="store_true")
    analyze.add_argument("--suppressions", type=Path)

    render = commands.add_parser(
        "render", help="re-render a saved analyze-pr JSON payload"
    )
    render.add_argument("--from-json", type=Path, required=True)
    render.add_argument("--format", choices=["md", "slack"], default="md")


def _write_json(path: Path, value: Any) -> None:
    """Write deterministic, newline-terminated JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _print_errors(errors: list[OpError]) -> None:
    """Print extractor errors without hiding successfully covered surfaces."""
    for error in errors:
        print(f"warning: {error.operation}: {error.message}", file=sys.stderr)


def _extract(args: argparse.Namespace) -> int:
    """Capture one checked-out source tree as a persisted snapshot."""
    result = build_snapshot(
        args.repo,
        args.release,
        ref=args.ref,
        repo=args.repo_name,
    )
    if result.errors:
        _print_errors(result.errors)
        return 1
    snapshot = SurfaceSnapshot.from_dict(result.data["snapshot"])
    output = args.out or default_snapshot_path(args.release, args.repo)
    save_snapshot(snapshot, output)
    print(f"captured {len(snapshot.symbols)} symbols -> {output}")
    return 0


def _diff(args: argparse.Namespace) -> int:
    """Compare two snapshots and optionally persist the change list."""
    result = gather_changes(
        load_snapshot(args.old),
        load_snapshot(args.new),
        repo_path=args.repo,
        release=args.release,
        include_markers=not args.no_markers and args.repo is not None,
    )
    if result.errors:
        _print_errors(result.errors)
        return 1
    if args.json:
        _write_json(args.json, result.data)
    print(f"{result.metadata['change_count']} API-surface changes")
    for change_type, count in _counts_by_type(result.data["changes"]).items():
        print(f"  {change_type}: {count}")
    if result.data["coverage_gaps"]:
        print(f"  coverage gaps: {', '.join(result.data['coverage_gaps'])}")
    return 0


def _counts_by_type(changes: list[dict[str, Any]]) -> dict[str, int]:
    """Return deterministic change counts keyed by change type."""
    counts: dict[str, int] = {}
    for change in changes:
        kind = str(change["change_type"])
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def _update_ledger(args: argparse.Namespace) -> int:
    """Merge one release snapshot and its changes into the durable ledger."""
    root = args.repo or Path(".")
    ledger_path = args.ledger or default_ledger_path(root)
    provenance_path = args.provenance or default_provenance_path(root)
    new = load_snapshot(args.new)
    old = load_snapshot(args.old) if args.old else SurfaceSnapshot(release="", ref="")
    result = gather_changes(
        old,
        new,
        repo_path=args.repo,
        release=args.release,
        include_markers=not args.no_markers and args.repo is not None,
    )
    if result.errors:
        _print_errors(result.errors)
        return 1
    changes = [SurfaceChange.from_dict(item) for item in result.data["changes"]]
    ledger = merge_changes(
        load_ledger(ledger_path),
        changes,
        args.release,
        snapshot=new,
        provenance=load_provenance(provenance_path),
    )
    if args.dry_run:
        print(f"would write {len(ledger.entries)} entries -> {ledger_path}")
    else:
        save_ledger(ledger, ledger_path)
        print(f"merged {len(ledger.entries)} entries -> {ledger_path}")
    return 0


def _list_deprecations(args: argparse.Namespace) -> int:
    """Print currently deprecated ledger entries."""
    path = args.ledger or default_ledger_path(args.repo)
    entries = deprecations(
        load_ledger(path),
        include_unconfirmed=args.include_unconfirmed,
    )
    if args.json:
        _write_json(args.json, [entry.to_dict() for entry in entries])
    for entry in entries:
        target = f" -> {entry.removal_target}" if entry.removal_target else ""
        print(f"{entry.id}{target}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    """Validate the durable ledger and return nonzero on broken promises."""
    ledger_path = args.ledger or default_ledger_path(args.repo)
    suppressions_path = args.suppressions or default_suppressions_path(args.repo)
    result = validate_ledger(
        load_ledger(ledger_path),
        suppressions=load_suppressions(suppressions_path),
    )
    if args.json:
        _write_json(args.json, result.data)
    for violation in result.data["violations"]:
        print(
            f"{violation['kind']}: {violation['id']}: {violation['detail']}",
            file=sys.stderr,
        )
    print(
        f"{result.metadata['violation_count']} violation(s); "
        f"{result.metadata['waived_count']} waived"
    )
    return 1 if result.data["violations"] else 0


def _resolve_snapshot(
    repo: Path | None,
    snapshot: Path | None,
    ref: str,
    release: str,
) -> tuple[SurfaceSnapshot | None, list[OpError]]:
    """Load a supplied snapshot or capture the supplied checked-out tree."""
    if snapshot:
        return load_snapshot(snapshot), []
    if repo:
        result = build_snapshot(repo, release, ref=ref)
        return SurfaceSnapshot.from_dict(result.data["snapshot"]), list(result.errors)
    return None, []


def _analyze_pr(args: argparse.Namespace) -> int:
    """Render and optionally gate the public API delta between two trees."""
    base, base_errors = _resolve_snapshot(
        args.base_repo,
        args.base_snapshot,
        args.base_ref,
        args.release,
    )
    head, head_errors = _resolve_snapshot(
        args.head_repo,
        args.head_snapshot,
        args.head_ref,
        args.release,
    )
    if base is None or head is None:
        print("provide both a base and head repository or snapshot", file=sys.stderr)
        return 2
    extraction_errors = base_errors + head_errors
    if extraction_errors:
        _print_errors(extraction_errors)
        return 1
    marker_repo = args.head_repo if args.markers else None
    suppressions_path = args.suppressions
    if suppressions_path is None and args.head_repo is not None:
        suppressions_path = default_suppressions_path(args.head_repo)
    suppressions = load_suppressions(suppressions_path) if suppressions_path else None
    result = analyze_impact(
        base,
        head,
        repo_path=str(marker_repo) if marker_repo else None,
        release=args.release,
        include_markers=marker_repo is not None,
        suppressions=suppressions,
    )
    if result.errors:
        _print_errors(result.errors)
        return 1
    rendered = render_impact(result.data, args.format)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="" if rendered.endswith("\n") else "\n")
    return 1 if args.fail_on_breaking and result.metadata["has_breaking"] else 0


def _render(args: argparse.Namespace) -> int:
    """Re-render a saved ``analyze-pr`` JSON payload."""
    data = json.loads(args.from_json.read_text(encoding="utf-8"))
    print(render_impact(data, args.format), end="")
    return 0


def run(args: argparse.Namespace) -> int:
    """Dispatch one parsed API-surface command."""
    handlers = {
        "extract": _extract,
        "diff": _diff,
        "update-ledger": _update_ledger,
        "list-deprecations": _list_deprecations,
        "validate": _validate,
        "analyze-pr": _analyze_pr,
        "render": _render,
    }
    try:
        return handlers[args.api_surface_command](args)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m api_surface``."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="api_surface")
    configure_parser(parser)
    return run(parser.parse_args(args_list))
