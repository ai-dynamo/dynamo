#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MODEL_DIR = "zai-org_GLM-5.2-FC"
OFFICIAL_MODEL = "GLM-5.2 Native FC OpenAI Chat Completions"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, list):
        for item in value:
            yield from flatten_numbers(item)


def category_from_path(path: Path, suffix: str) -> str:
    prefix = "BFCL_v4_"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected BFCL filename: {path}")
    return name[len(prefix) : -len(suffix)]


def result_stats(run_dir: Path) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    pattern = f"result/{MODEL_DIR}/**/BFCL_v4_*_result.json"
    for path in sorted(run_dir.glob(pattern)):
        category = category_from_path(path, "_result.json")
        entries = load_jsonl(path)
        input_tokens = sum(
            sum(flatten_numbers(entry.get("input_token_count", 0))) for entry in entries
        )
        output_tokens = sum(
            sum(flatten_numbers(entry.get("output_token_count", 0)))
            for entry in entries
        )
        query_latencies = [
            latency
            for entry in entries
            for latency in flatten_numbers(entry.get("latency", []))
        ]
        inference_errors = sum(
            1
            for entry in entries
            if "traceback" in entry
            or (
                isinstance(entry.get("result"), str)
                and entry["result"].startswith("Error during inference:")
            )
        )
        stats[category] = {
            "generated_count": len(entries),
            "inference_error_count": inference_errors,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "query_count": len(query_latencies),
            "query_latency_seconds_sum": sum(query_latencies),
            "query_latency_seconds_mean": (
                sum(query_latencies) / len(query_latencies) if query_latencies else None
            ),
        }
    return stats


def score_stats(
    run_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    stats: dict[str, dict[str, Any]] = {}
    all_failures: list[dict[str, Any]] = []
    error_types: Counter[str] = Counter()
    pattern = f"score/{MODEL_DIR}/**/BFCL_v4_*_score.json"
    for path in sorted(run_dir.glob(pattern)):
        category = category_from_path(path, "_score.json")
        entries = load_jsonl(path)
        if not entries:
            continue
        header, failures = entries[0], entries[1:]
        for failure in failures:
            enriched = {"category": category, "source_file": str(path), **failure}
            all_failures.append(enriched)
            error_type = failure.get("error_type", "unknown")
            if isinstance(error_type, list):
                error_types.update(str(item) for item in error_type)
            else:
                error_types[str(error_type)] += 1
        stats[category] = {
            "accuracy": header["accuracy"],
            "correct_count": header["correct_count"],
            "failure_count": header["total_count"] - header["correct_count"],
            "total_count": header["total_count"],
        }
    return stats, all_failures, error_types


def overall_row(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "score" / "data_overall.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def normalized(row: dict[str | None, Any]) -> dict[str, Any]:
        # BFCL's aggregate CSV can contain trailing unnamed columns. DictReader
        # exposes those values under a None key, which is not sortable beside
        # string keys when json.dumps(sort_keys=True) recurses into this row.
        return {"_extra" if key is None else key: value for key, value in row.items()}

    matches = [row for row in rows if row.get("Model") == OFFICIAL_MODEL]
    if len(matches) != 1:
        raise ValueError(
            "BFCL data_overall.csv must contain exactly one exact GLM-5.2 campaign "
            f"row; found {len(matches)}"
        )
    return normalized(matches[0])


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RUN_DIR")
    run_dir = Path(sys.argv[1]).resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    generated = result_stats(run_dir)
    scored, failures, error_types = score_stats(run_dir)
    categories = {}
    for category in sorted(set(generated) | set(scored)):
        categories[category] = {
            **generated.get(category, {}),
            **scored.get(category, {}),
        }

    summary = {
        "schema_version": 1,
        "variant": metadata.get("variant"),
        "mode": metadata.get("mode"),
        "campaign_phase": metadata.get("campaign_phase"),
        "run_name": metadata.get("run_name"),
        "bfcl_gorilla_commit": metadata.get("bfcl_gorilla_commit"),
        "categories": categories,
        "totals": {
            "generated_count": sum(
                v.get("generated_count", 0) for v in categories.values()
            ),
            "inference_error_count": sum(
                v.get("inference_error_count", 0) for v in categories.values()
            ),
            "correct_count": sum(
                v.get("correct_count", 0) for v in categories.values()
            ),
            "failure_count": sum(
                v.get("failure_count", 0) for v in categories.values()
            ),
            "scored_count": sum(v.get("total_count", 0) for v in categories.values()),
            "input_tokens": sum(v.get("input_tokens", 0) for v in categories.values()),
            "output_tokens": sum(
                v.get("output_tokens", 0) for v in categories.values()
            ),
            "query_count": sum(v.get("query_count", 0) for v in categories.values()),
            "query_latency_seconds_sum": sum(
                v.get("query_latency_seconds_sum", 0) for v in categories.values()
            ),
        },
        "failure_error_types": dict(sorted(error_types.items())),
        "official_overall_csv_row": overall_row(run_dir),
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (run_dir / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, sort_keys=True) + "\n")

    print(json.dumps(summary["totals"], indent=2, sort_keys=True))
    print(f"Summary:  {run_dir / 'summary.json'}")
    print(f"Failures: {run_dir / 'failures.jsonl'}")


if __name__ == "__main__":
    main()
