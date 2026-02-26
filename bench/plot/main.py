# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CLI entry-point for benchmark result plotting.

Usage::

    python -m bench.plot.main --run-dir bench_results/run_20260213_121845
    python -m bench.plot.main --run-dir bench_results/run_20260213_121845 --output-dir /tmp/plots

Data loading strategy:
    1. **Primary** -- per-concurrency aiperf JSON files at
       ``<scenario>/aiperf/concurrency_N/profile_export_aiperf.json``
       (available after the client.sh fix).
    2. **Fallback** -- parse the aiperf rich-text tables captured in
       ``<scenario>/logs/client.log`` (for older runs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from bench.plot import charts, parse_logs, parse_metrics

# Metrics we pull from aiperf JSON (top-level keys).
_AIPERF_METRIC_KEYS = (
    "time_to_first_token",
    "inter_token_latency",
    "request_latency",
    "output_token_throughput",
    "request_throughput",
    "request_count",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_aiperf_json(
    scenario_dir: Path,
    concurrencies: list[int],
) -> dict[int, dict[str, dict[str, float | None]]] | None:
    """Try to load per-concurrency aiperf JSON files (primary source).

    Returns None if the per-concurrency directories don't exist.
    """
    aiperf_dir = scenario_dir / "aiperf"
    result: dict[int, dict[str, dict[str, float | None]]] = {}

    for c in concurrencies:
        json_path = aiperf_dir / f"concurrency_{c}" / "profile_export_aiperf.json"
        if not json_path.exists():
            return None  # not a per-concurrency run; fall back
        with open(json_path) as f:
            raw = json.load(f)

        metrics: dict[str, dict[str, float | None]] = {}
        for key in _AIPERF_METRIC_KEYS:
            if key in raw:
                metrics[key] = {k: v for k, v in raw[key].items() if k != "unit"}
        result[c] = metrics

    return result


def _load_from_client_log(
    scenario_dir: Path,
) -> dict[int, dict[str, dict[str, float | None]]]:
    """Fallback: parse aiperf tables from client.log."""
    log_path = scenario_dir / "logs" / "client.log"
    if not log_path.exists():
        return {}
    return parse_logs.parse_client_log(log_path)


def load_scenario_data(
    scenario_dir: Path,
    concurrencies: list[int],
) -> dict[int, dict[str, dict[str, float | None]]]:
    """Load per-concurrency aiperf data for one scenario.

    Prefers JSON files; falls back to client.log parsing.
    """
    data = _load_aiperf_json(scenario_dir, concurrencies)
    if data is not None:
        print(f"  Loaded from per-concurrency JSON: {scenario_dir.name}")
        return data

    data = _load_from_client_log(scenario_dir)
    if data:
        print(f"  Loaded from client.log (fallback): {scenario_dir.name}")
    else:
        print(f"  WARNING: no aiperf data found for {scenario_dir.name}")
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from a bench run directory."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to the run directory (e.g. bench_results/run_20260213_121845)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output PNGs (default: <run-dir>/plots)",
    )
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Load config
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)

    concurrencies = [int(c) for c in str(config["concurrencies"]).split(",")]

    output_dir: Path = args.output_dir or run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir:    {run_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Concurrencies: {concurrencies}")
    print()

    # Discover scenario directories
    scenario_dirs = sorted(run_dir.glob("scenario_*"))
    if not scenario_dirs:
        print("ERROR: no scenario_* directories found", file=sys.stderr)
        sys.exit(1)

    # Load per-scenario aiperf data
    all_data: dict[str, dict[int, dict[str, Any]]] = {}
    for sd in scenario_dirs:
        all_data[sd.name] = load_scenario_data(sd, concurrencies)

    print()

    # -----------------------------------------------------------------------
    # Generate comparison charts (Graphs 1-5)
    # -----------------------------------------------------------------------
    generated: list[Path] = []

    out = charts.plot_ttft(all_data, concurrencies, config, output_dir)
    generated.append(out)
    print(f"  [1/6] {out.name}")

    out = charts.plot_itl(all_data, concurrencies, config, output_dir)
    generated.append(out)
    print(f"  [2/6] {out.name}")

    out = charts.plot_throughput(all_data, concurrencies, config, output_dir)
    generated.append(out)
    print(f"  [3/6] {out.name}")

    out = charts.plot_request_latency(all_data, concurrencies, config, output_dir)
    generated.append(out)
    print(f"  [4/6] {out.name}")

    out = charts.plot_request_throughput(all_data, concurrencies, config, output_dir)
    generated.append(out)
    print(f"  [5/6] {out.name}")

    # -----------------------------------------------------------------------
    # Graph 6: Tokenizer latency (Scenario 2 only, from Prometheus deltas)
    # -----------------------------------------------------------------------
    scenario2_dir = run_dir / "scenario_2_dynamo_rust"
    if scenario2_dir.is_dir():
        metrics_dir = scenario2_dir / "metrics"
        if metrics_dir.is_dir():
            prom = parse_metrics.load_concurrency_metrics(metrics_dir, concurrencies)
            tok_data = parse_metrics.compute_tokenizer_latency_deltas(
                prom, concurrencies
            )
            out = charts.plot_tokenizer_latency(
                tok_data, concurrencies, config, output_dir
            )
            if out:
                generated.append(out)
                print(f"  [6/6] {out.name}")
            else:
                print("  [6/6] skipped (no tokenizer latency data)")
        else:
            print("  [6/6] skipped (no metrics directory for scenario 2)")
    else:
        print("  [6/6] skipped (scenario 2 not found)")

    # -----------------------------------------------------------------------
    # Save summary.json with extracted tabular data
    # -----------------------------------------------------------------------
    summary = {
        "config": config,
        "concurrencies": concurrencies,
        "scenarios": {},
    }
    for scenario_key, scenario_data in all_data.items():
        # Convert to JSON-serializable (int keys -> str keys)
        summary["scenarios"][scenario_key] = {
            str(c): metrics for c, metrics in sorted(scenario_data.items())
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  summary.json written to {summary_path}")

    print(f"\nDone. {len(generated)} charts saved to {output_dir}")


if __name__ == "__main__":
    main()
