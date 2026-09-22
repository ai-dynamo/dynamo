# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a recent-cache policy comparison manifest without running replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

TWO_TIER_YAML = """worker_selection:
  aggregated: smg
  instances:
    - name: smg
      type: dynamo-two-tier-cost-fn
      parameters:
        cache_threshold: 0.5
        balance_abs_threshold: 32
        balance_rel_threshold: 1.1
"""


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def generate(args: argparse.Namespace) -> dict[str, Any]:
    for key in (
        "expected_requests",
        "workers",
        "ttl_seconds",
        "arrival_speedup",
        "threshold",
    ):
        value = vars(args)[key]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if args.trace_block_size is None:
        args.trace_block_size = 64 if args.format == "weka" else 512
    if args.trace_block_size <= 0:
        raise ValueError("trace_block_size must be positive")
    if args.agentic_lanes is not None:
        if args.format != "weka" or args.agentic_lanes <= 0:
            raise ValueError("agentic_lanes requires Weka and a positive lane count")
    if args.trace.is_symlink():
        raise ValueError("trace may not be a symlink")
    trace = args.trace.resolve()
    if not trace.is_file() and not (args.format == "weka" and trace.is_dir()):
        raise ValueError("trace must be a file, or a Weka directory")
    output = args.output_dir.resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("output-dir must be outside the repository")
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    if trace.is_dir() and output.is_relative_to(trace):
        raise ValueError("output-dir must not be inside the Weka input directory")

    engine_path = args.engine_args.resolve()
    engine_raw = engine_path.read_bytes()
    engine_args = json.loads(engine_raw)
    if not isinstance(engine_args, dict) or not engine_args:
        raise ValueError("engine-args must contain a nonempty JSON object")
    json.dumps(engine_args, allow_nan=False)
    if trace.is_dir():
        for path in trace.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Weka directory may not contain symlinks: {path}")
    source_files = sorted(trace.rglob("*.json")) if trace.is_dir() else [trace]
    if not source_files:
        raise ValueError("Weka directory contains no JSON files")
    provenance = []
    for path in source_files:
        if path.is_symlink():
            raise ValueError(f"trace file may not be a symlink: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        provenance.append(
            {
                "path": str(path),
                "sha256": digest.hexdigest(),
                "bytes": path.stat().st_size,
            }
        )

    common = {
        "trace_files": [str(trace)],
        "trace_format": args.format,
        "trace_block_size": args.trace_block_size,
        "num_workers": args.workers,
        "arrival_speedup_ratio": args.arrival_speedup,
        "engine_args": engine_args,
        "expected_requests": args.expected_requests,
    }
    if args.agentic_lanes is not None:
        common["agentic_lanes"] = args.agentic_lanes

    def case(name: str, mode: str, predict: bool, credit: float) -> dict[str, Any]:
        return {
            **common,
            "name": name,
            "router_config": {"overlap_score_credit": 1.0},
            "recent_cache": {
                "ttl_secs": args.ttl_seconds,
                "threshold": args.threshold,
                "boost_credit": credit,
                "mode": mode,
                "predict": predict,
                "trace_decisions": True,
            },
        }

    cases = [
        case("stock-default-credit1", "observe", False, 1.0),
        case("prediction-only-credit1", "observe", True, 1.0),
    ]
    cases.extend(
        case(f"static-credit{credit}", "fixed", True, credit) for credit in (4, 16, 64)
    )
    cases.extend(
        case(f"hybrid-credit{credit}", "adaptive", True, credit)
        for credit in (4, 16, 64)
    )
    smg = case("smg", "observe", False, 1.0)
    smg["router_config"]["router_policy_config"] = str(output / "two-tier.yaml")
    cases.append(smg)

    manifest = {"cases": cases}
    metadata = {
        "trace_files": provenance,
        "engine_args_source": {
            "path": str(engine_path),
            "sha256": hashlib.sha256(engine_raw).hexdigest(),
        },
        "configuration": {
            key: value
            for key, value in vars(args).items()
            if key not in ("trace", "engine_args", "output_dir")
        },
        "comparison": {
            "best_static_candidates": [
                "prediction-only-credit1",
                "static-credit4",
                "static-credit16",
                "static-credit64",
            ],
            "other_baselines": ["stock-default-credit1", "smg"],
            "adaptive_candidates": [
                "hybrid-credit4",
                "hybrid-credit16",
                "hybrid-credit64",
            ],
            "success_criterion": (
                "Evaluate hybrids against the best static credit (1/4/16/64) and "
                "SMG, with identical workload and engine settings; a gain over "
                "stock alone is insufficient."
            ),
            "static_prediction": (
                "Static credits and prediction-only credit1 use the same TTL "
                "overlap prediction as hybrids. Stock and SMG use observe mode "
                "with predict=false, which records telemetry without affecting "
                "routing."
            ),
            "seed": (
                "No seed is configured; the normal selector remains stochastic. "
                "Repeat close comparisons without enabling replay-bench."
            ),
        },
        "smg_parameters": {
            "cache_threshold": 0.5,
            "balance_abs_threshold": 32,
            "balance_rel_threshold": 1.1,
        },
        "build_features": ["aic-forward-pass", "replay-builtin"],
        "logging_environment": {
            "DYN_LOG": "warn,aisimulate_core::engine::scheduler::vllm::core=debug",
            "DYN_LOGGING_CONSOLE_FORMAT": "jsonl",
            "DYNAMO_SKIP_PYTHON_LOG_INIT": None,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "two-tier.yaml").open("x") as handle:
        handle.write(TWO_TIER_YAML)
    write_json(output / "engine-args.json", engine_args)
    write_json(output / "cases.json", manifest)
    write_json(output / "provenance.json", metadata)
    return {
        "manifest": str(output / "cases.json"),
        "cases": len(cases),
        "provenance": str(output / "provenance.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--format", required=True, choices=("mooncake", "weka"))
    parser.add_argument("--expected-requests", required=True, type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ttl-seconds", type=float, default=300)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--arrival-speedup", type=float, default=1)
    parser.add_argument("--trace-block-size", type=int)
    parser.add_argument("--engine-args", required=True, type=Path)
    parser.add_argument("--agentic-lanes", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = generate(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
