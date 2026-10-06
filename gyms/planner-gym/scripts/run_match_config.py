#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate, inspect, and execute an Autoscaling Arena Match Config."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    from autoscaling_arena.match_config import MatchConfigError, load_match_config
    from autoscaling_arena.match_runner import (
        MatchPublishError,
        execute_match_config,
        format_match_matrix,
        match_replay_sha256,
        preflight_publish_destinations,
        publish_match_results,
    )

    parser = argparse.ArgumentParser(
        description="Run a versioned YAML Autoscaling Arena Match Config."
    )
    parser.add_argument("config", help="path to the Match Config YAML")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate without running DynoSim or AIPerf",
    )
    parser.add_argument(
        "--print-matrix",
        action="store_true",
        help="print every expanded run before exiting or executing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow configured JSON and HTML result files to be replaced",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="run only this expanded matrix cell (repeatable)",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="keep run artifacts but skip configured result destinations",
    )
    parser.add_argument(
        "--expect-config-sha256",
        help=(
            "refuse to run unless the replay-relevant resolved Match Config "
            "has this digest"
        ),
    )
    parser.add_argument(
        "--expect-trace-sha256",
        action="append",
        default=[],
        metavar="DIGEST",
        help=(
            "refuse to run unless selected source trace contents match this "
            "SHA-256 (repeat for sharded traces)"
        ),
    )
    args = parser.parse_args()

    try:
        config = load_match_config(args.config)
    except MatchConfigError as exc:
        parser.exit(2, f"Match Config error: {exc}\n")

    config_sha256 = match_replay_sha256(config)
    if (
        args.expect_config_sha256 is not None
        and args.expect_config_sha256 != config_sha256
    ):
        parser.error(
            "Match Config digest mismatch: expected "
            f"{args.expect_config_sha256}, got {config_sha256}"
        )

    matrix = list(config.iter_runs())
    selected_ids = set(args.run_id)
    available_ids = {item.run_id for item in matrix}
    unknown_ids = sorted(selected_ids - available_ids)
    if unknown_ids:
        parser.error("unknown --run-id: " + ", ".join(unknown_ids))
    selected_matrix = (
        [item for item in matrix if item.run_id in selected_ids]
        if selected_ids
        else matrix
    )
    if args.expect_trace_sha256:
        expected_trace_sha256s = [
            _validated_sha256(value, parser=parser)
            for value in args.expect_trace_sha256
        ]
        try:
            actual_trace_sha256s = [
                _sha256_file(path) for path in _selected_source_traces(selected_matrix)
            ]
        except OSError as exc:
            parser.error(f"cannot fingerprint selected source trace: {exc}")
        if actual_trace_sha256s != expected_trace_sha256s:
            parser.error(
                "source trace digest mismatch: expected "
                + ", ".join(expected_trace_sha256s)
                + "; got "
                + (", ".join(actual_trace_sha256s) or "no source traces")
            )
    selected_count = (
        sum(item.run_id in selected_ids for item in matrix)
        if selected_ids
        else len(matrix)
    )
    print(
        f"Valid Match Config: {config.name!r} — "
        f"{selected_count} {config.backend.type} run"
        f"{'s' if selected_count != 1 else ''} selected"
    )
    if args.print_matrix:
        print(format_match_matrix(config))
    if args.validate_only:
        return 0

    if not args.no_publish:
        try:
            preflight_publish_destinations(config, force_overwrite=args.overwrite)
        except MatchPublishError as exc:
            print(f"Publish error: {exc}", file=sys.stderr)
            return 2

    progress_index = 0

    def progress(phase, item, result):
        nonlocal progress_index
        if phase == "started":
            progress_index += 1
            matrix_position = (
                f"; matrix {item.index}/{config.expected_runs}" if selected_ids else ""
            )
            print(
                f"  [{progress_index}/{selected_count}{matrix_position}] "
                f"{item.autoscaler} × {item.workload} × {item.sla} ...",
                flush=True,
            )
        elif result is not None and result["status"] != "ok":
            error = result.get("error", {})
            print(
                f"    failed: {error.get('type', 'Error')}: "
                f"{error.get('message', '')}",
                flush=True,
            )

    try:
        report = execute_match_config(
            config,
            on_progress=progress,
            run_ids=selected_ids or None,
        )
    except ValueError as exc:
        parser.error(str(exc))
    except MatchPublishError as exc:
        print(f"Artifact error: {exc}", file=sys.stderr)
        return 2
    if args.no_publish:
        print(
            "Artifacts: "
            + str(config.publish.artifact_root / report["provenance"]["session_id"]),
            flush=True,
        )
        written = []
    else:
        try:
            written = publish_match_results(
                config, report, force_overwrite=args.overwrite
            )
        except MatchPublishError as exc:
            print(f"Publish error: {exc}", file=sys.stderr)
            return 2
    for path in written:
        kind = (
            "HTML report"
            if path.suffix.lower() in {".html", ".htm"}
            else "Results JSON"
        )
        print(f"{kind}: {path}")
    return 0 if report["summary"]["failed_runs"] == 0 else 1


def _validated_sha256(value: str, *, parser: argparse.ArgumentParser) -> str:
    digest = value.lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        parser.error(f"invalid SHA-256 digest: {value}")
    return digest


def _selected_source_traces(matrix) -> list[Path]:
    paths: list[Path] = []
    seen_groups: set[tuple[Path, ...]] = set()
    for item in matrix:
        raw_paths = (
            item.trace_paths
            if item.trace_format == "dynamo"
            else ((item.trace_path,) if item.trace_path is not None else ())
        )
        candidates = tuple(path.resolve() for path in raw_paths)
        if candidates and candidates not in seen_groups:
            seen_groups.add(candidates)
            paths.extend(candidates)
    return paths


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
