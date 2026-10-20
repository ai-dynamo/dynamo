# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render saved Arena results without replaying their workload.

Read the normalized dictionary produced by ``execute_match_config``.
Rendering saved results never reopens source traces or invokes Dynamo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any


class SavedReportError(ValueError):
    """A saved result cannot be safely loaded or adapted."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SavedReportError(f"{label} must be a JSON object")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SavedReportError(f"{label} must be a JSON array")
    return list(value)


def _load_json(path: Path, label: str) -> Any:
    try:
        text = path.read_text()
    except OSError as exc:
        raise SavedReportError(f"cannot read {label} {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SavedReportError(
            f"{label} {path} is not valid JSON: {exc.msg} at line {exc.lineno}"
        ) from exc


def _is_normalized_match_report(value: Mapping[str, Any]) -> bool:
    return (
        isinstance(value.get("match"), Mapping)
        and isinstance(value.get("summary"), Mapping)
        and isinstance(value.get("results"), list)
    )


def _apply_overrides(
    report: dict[str, Any],
    *,
    title: str | None,
    description: str | None,
    rank_by: str | None,
) -> dict[str, Any]:
    match = _mapping(report.get("match"), "match")
    summary = _mapping(report.get("summary"), "summary")
    results = _sequence(report.get("results"), "results")
    if any(not isinstance(result, Mapping) for result in results):
        raise SavedReportError("every results entry must be a JSON object")
    report["match"] = match
    report["summary"] = summary
    report["results"] = results
    if title is not None:
        match["name"] = title
    if description is not None:
        match["description"] = description
    if rank_by is not None:
        summary["rank_by"] = rank_by
        metrics = [str(value) for value in summary.get("metrics", [])]
        if rank_by not in metrics:
            metrics.insert(0, rank_by)
        summary["metrics"] = metrics
    return report


def load_saved_report(
    input_path: str | Path,
    *,
    title: str | None = None,
    description: str | None = None,
    rank_by: str | None = None,
) -> dict[str, Any]:
    """Load a normalized Match Config report."""

    path = Path(input_path).expanduser().resolve()
    payload = _mapping(_load_json(path, "saved results"), "saved results")
    if _is_normalized_match_report(payload):
        return _apply_overrides(
            payload,
            title=title,
            description=description,
            rank_by=rank_by,
        )
    schema = payload.get("schema")
    if schema is not None:
        raise SavedReportError(f"unsupported saved-results schema: {schema!r}")
    raise SavedReportError("input is not a normalized Match Config report")


def _preflight_output(path: Path, *, overwrite: bool) -> None:
    if path.is_symlink():
        raise SavedReportError(f"output must not be a symbolic link: {path}")
    if path.exists() and path.is_dir():
        raise SavedReportError(f"output is a directory: {path}")
    if path.exists() and not overwrite:
        raise SavedReportError(
            f"output already exists: {path}; pass --overwrite to replace it"
        )


def _write_text(path: Path, text: str, *, overwrite: bool) -> None:
    _preflight_output(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def render_saved_result(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    normalized_output_path: str | Path | None = None,
    title: str | None = None,
    description: str | None = None,
    rank_by: str | None = None,
    match_config_path: str | Path | None = None,
    replay_config_ref: str | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path | None]:
    """Load saved results and write the existing standalone HTML report."""

    source = Path(input_path).expanduser().resolve()
    destination = (
        Path(output_path).expanduser().absolute()
        if output_path is not None
        else source.with_suffix(".html")
    )
    normalized_destination = (
        Path(normalized_output_path).expanduser().absolute()
        if normalized_output_path is not None
        else None
    )
    if destination.resolve() == source:
        raise SavedReportError("HTML output must not replace the input results")
    _preflight_output(destination, overwrite=overwrite)
    if normalized_destination is not None:
        if normalized_destination.resolve() == source:
            raise SavedReportError(
                "normalized output must not replace the input results"
            )
        if normalized_destination.resolve() == destination.resolve():
            raise SavedReportError("HTML and normalized JSON outputs must be different")
        _preflight_output(normalized_destination, overwrite=overwrite)

    report = load_saved_report(
        source,
        title=title,
        description=description,
        rank_by=rank_by,
    )
    if replay_config_ref is not None and match_config_path is None:
        raise SavedReportError(
            "replay_config_ref requires a validated match_config_path"
        )
    replay_config_path: str | None = None
    if match_config_path is not None:
        from autoscaling_arena.match_config import load_match_config
        from autoscaling_arena.match_runner import (
            match_config_sha256,
            match_replay_sha256,
        )

        replay_config = load_match_config(match_config_path)
        saved_provenance = _mapping(report.get("provenance"), "provenance")
        expected_replay_sha256 = str(saved_provenance.get("replay_config_sha256") or "")
        expected_full_sha256 = str(saved_provenance.get("config_sha256") or "")
        actual_replay_sha256 = match_replay_sha256(replay_config)
        actual_full_sha256 = match_config_sha256(replay_config)
        expected_sha256 = expected_replay_sha256 or expected_full_sha256
        actual_sha256 = (
            actual_replay_sha256 if expected_replay_sha256 else actual_full_sha256
        )
        if not expected_sha256:
            raise SavedReportError("saved results do not contain a Match Config digest")
        if actual_sha256 != expected_sha256:
            raise SavedReportError(
                "Match Config digest does not match saved results: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        available_ids = {item.run_id for item in replay_config.iter_runs()}
        missing_ids = sorted(
            str(result.get("run_id") or "")
            for result in _sequence(report.get("results"), "results")
            if str(result.get("run_id") or "") not in available_ids
        )
        if missing_ids:
            raise SavedReportError(
                "saved run IDs are absent from the Match Config: "
                + ", ".join(missing_ids)
            )
        provenance = dict(saved_provenance)
        provenance["replay_config_sha256"] = actual_replay_sha256
        report["provenance"] = provenance
        replay_config_path = (
            _validated_replay_config_ref(replay_config_ref)
            if replay_config_ref is not None
            else "$MATCH_CONFIG"
        )
    from autoscaling_arena.html_report import render_match_report

    rendered = render_match_report(report, replay_config_path=replay_config_path)
    normalized_json = (
        json.dumps(report, indent=2, allow_nan=False) + "\n"
        if normalized_destination is not None
        else None
    )
    _write_text(destination, rendered, overwrite=overwrite)
    if normalized_destination is not None and normalized_json is not None:
        _write_text(
            normalized_destination,
            normalized_json,
            overwrite=overwrite,
        )
    return destination, normalized_destination


def _validated_replay_config_ref(value: str) -> str:
    """Validate a portable path reference before embedding it in HTML."""

    reference = value.strip()
    if not reference:
        raise SavedReportError("replay_config_ref must not be empty")
    if (
        Path(reference).is_absolute()
        or PureWindowsPath(reference).is_absolute()
        or reference.startswith("~")
        or "\x00" in reference
    ):
        raise SavedReportError("replay_config_ref must be a portable relative path")
    return reference


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Render a saved Match Config report without replaying requests.")
    )
    parser.add_argument("results", help="saved Match Config results.json")
    parser.add_argument(
        "--out",
        help="HTML destination (default: replace the input suffix with .html)",
    )
    parser.add_argument(
        "--normalized-out",
        help="optionally write the normalized Match Config report JSON",
    )
    parser.add_argument("--title", help="override the report title")
    parser.add_argument("--description", help="override the report description")
    parser.add_argument("--rank-by", help="override the leaderboard rank metric")
    parser.add_argument(
        "--match-config",
        help=(
            "validated source Match Config path used to enable per-run replay "
            "commands"
        ),
    )
    parser.add_argument(
        "--replay-config-ref",
        help=(
            "portable relative Match Config path to embed in replay buttons; "
            "defaults to $MATCH_CONFIG and requires --match-config"
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing output files"
    )
    args = parser.parse_args(argv)
    try:
        html_path, normalized_path = render_saved_result(
            args.results,
            output_path=args.out,
            normalized_output_path=args.normalized_out,
            title=args.title,
            description=args.description,
            rank_by=args.rank_by,
            match_config_path=args.match_config,
            replay_config_ref=args.replay_config_ref,
            overwrite=args.overwrite,
        )
    except (SavedReportError, ImportError, OSError, TypeError, ValueError) as exc:
        print(f"Saved report error: {exc}", file=sys.stderr)
        return 2
    print(f"HTML report: {html_path}")
    if normalized_path is not None:
        print(f"Normalized results JSON: {normalized_path}")
    return 0


__all__ = [
    "SavedReportError",
    "load_saved_report",
    "main",
    "render_saved_result",
]
