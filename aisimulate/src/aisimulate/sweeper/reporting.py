# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Human- and machine-readable Sweeper result reporting."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import Candidate, SmartSearchConfig

RESULT_SCHEMA_VERSION = 1


def serialize_sweep_results(
    config: SmartSearchConfig,
    candidates: Sequence[Candidate],
) -> dict[str, Any]:
    """Return the versioned, lossless result envelope used by CLI JSON output."""

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_type": (
            "pareto_front" if config.goal.is_pareto else "ranked_candidates"
        ),
        "goal": config.goal.model_dump(mode="json"),
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }


def format_sweep_results(
    config: SmartSearchConfig,
    candidates: Sequence[Candidate],
    *,
    top_n: int,
) -> str:
    """Format complete top-candidate summaries for terminal output."""

    shown = candidates if config.goal.is_pareto else candidates[:top_n]
    if config.goal.is_pareto:
        lines = [f"pareto front ({len(candidates)} non-dominated):"]
    else:
        lines = [f"top {min(top_n, len(candidates))} of {len(candidates)} candidates:"]

    for rank, candidate in enumerate(shown, start=1):
        headline = [
            f"{rank}.",
            f"score={candidate.score:.6g}",
            f"used_gpus={candidate.used_gpus}",
        ]
        if candidate.objectives:
            headline.append(
                "objectives="
                + json.dumps(
                    candidate.objectives, sort_keys=True, separators=(",", ":")
                )
            )
        lines.append(" ".join(headline))
        lines.append(
            "   config="
            + json.dumps(candidate.config, sort_keys=True, separators=(",", ":"))
        )
        lines.append(
            "   metrics="
            + json.dumps(candidate.metrics, sort_keys=True, separators=(",", ":"))
        )
    return "\n".join(lines)


def write_sweep_results(
    output_dir: str | Path,
    config: SmartSearchConfig,
    candidates: Sequence[Candidate],
    *,
    top_n: int,
) -> tuple[Path, ...]:
    """Persist a lossless JSON envelope plus legacy-compatible CSV summaries.

    Scalar searches write ``best_config_topn.csv``. Pareto searches write the complete
    non-dominated set to ``pareto.csv`` because a Pareto front has no canonical top-N
    ordering. Both modes always write ``sweep_results.json``.
    """

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    json_path = directory / "sweep_results.json"
    json_path.write_text(
        json.dumps(
            serialize_sweep_results(config, candidates),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_name = "pareto.csv" if config.goal.is_pareto else "best_config_topn.csv"
    csv_path = directory / csv_name
    selected = candidates if config.goal.is_pareto else candidates[:top_n]
    _write_candidates_csv(csv_path, selected)
    return json_path, csv_path


def _write_candidates_csv(path: Path, candidates: Sequence[Candidate]) -> None:
    rows = [
        _candidate_csv_row(rank, candidate)
        for rank, candidate in enumerate(candidates, 1)
    ]
    fieldnames = ["rank", "score", "used_gpus"]
    remaining = sorted({key for row in rows for key in row}.difference(fieldnames))
    fieldnames.extend(remaining)

    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_csv_row(rank: int, candidate: Candidate) -> dict[str, Any]:
    row: dict[str, Any] = {
        "rank": rank,
        "score": candidate.score,
        "used_gpus": candidate.used_gpus,
    }
    _flatten_json(row, "config", candidate.config)
    _flatten_json(row, "metrics", candidate.metrics)
    if candidate.objectives is not None:
        _flatten_json(row, "objectives", candidate.objectives)
    return row


def _flatten_json(output: dict[str, Any], prefix: str, value: Any) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten_json(output, child, value[key])
        return
    if isinstance(value, list):
        output[prefix] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return
    output[prefix] = value
