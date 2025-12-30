# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare pre-existing profiling results from different methods."""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from benchmarks.profiler.comparison.metrics import (
    ProfilingMetrics,
    ComparisonResult,
    load_profiling_results,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

BASELINE_METHODS = {
    "aic_offline": "AI Configurator simulation (offline)",
    "online_aiperf": "Real deployment with AIPerf (online)",
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare profiling results")
    parser.add_argument("--aic-results", type=str, help="AIC profiling results directory")
    parser.add_argument("--online-results", type=str, help="Online profiling results directory")
    parser.add_argument("--additional-results", type=str, action="append", default=[],
                        help="Additional method: 'name:path'")
    parser.add_argument("--validation-results", type=str, action="append", default=[],
                        help="Validation data: 'name:path/to/validation.json'")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--backend", type=str, default="")
    parser.add_argument("--isl", type=int, default=0)
    parser.add_argument("--osl", type=int, default=0)
    parser.add_argument("--ttft", type=float, default=0.0)
    parser.add_argument("--itl", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="comparison_results")
    return parser


def parse_log_timing(log_path: Path) -> float:
    """Extract duration from profiler log timestamps."""
    if not log_path.exists():
        return 0.0
    first_ts = last_ts = None
    ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    with open(log_path, "r") as f:
        for line in f:
            match = ts_pattern.search(line)
            if match:
                ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
    if first_ts and last_ts:
        return (last_ts - first_ts).total_seconds()
    return 0.0


def parse_log_deployments(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    count = 0
    with open(log_path, "r") as f:
        for line in f:
            if "Created client with service_name" in line:
                count += 1
    return count


def parse_log_recommendations(log_path: Path) -> dict:
    results = {}
    if not log_path.exists():
        return results
    with open(log_path, "r") as f:
        for line in f:
            if "Suggested prefill" in line:
                try:
                    if "TTFT" in line:
                        start = line.index("TTFT") + 5
                        end = line.index("ms", start)
                        results["prefill_ttft"] = float(line[start:end].strip())
                    if "throughput" in line:
                        start = line.index("throughput") + 11
                        end = line.index("tokens/s/GPU", start)
                        results["prefill_thpt"] = float(line[start:end].strip())
                    match = re.search(r"(\d+)\s*GPU\(s\)", line)
                    if match:
                        results["prefill_gpus"] = int(match.group(1))
                except (ValueError, IndexError):
                    pass
            if "Suggested decode" in line:
                try:
                    if "ITL" in line:
                        start = line.index("ITL") + 4
                        end = line.index("ms", start)
                        results["decode_itl"] = float(line[start:end].strip())
                    match = re.search(r"(\d+)\s*GPU\(s\)", line)
                    if match:
                        results["decode_gpus"] = int(match.group(1))
                except (ValueError, IndexError):
                    pass
    return results


def is_aic_run(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    with open(log_path, "r") as f:
        return "aiconfigurator" in f.read().lower()


def load_method_results(results_dir: Path, method_name: str) -> Optional[ProfilingMetrics]:
    if not results_dir.exists():
        logger.warning(f"Results not found: {results_dir}")
        return None

    log_path = results_dir / "profile_sla.log"
    duration = parse_log_timing(log_path)
    num_deployments = 0 if is_aic_run(log_path) else parse_log_deployments(log_path)
    recs = parse_log_recommendations(log_path)

    metrics = ProfilingMetrics(
        method_name=method_name,
        method_description=BASELINE_METHODS.get(method_name, ""),
        source_dir=str(results_dir),
        total_duration_seconds=duration,
        num_deployments_created=num_deployments,
        recommended_prefill_gpus=recs.get("prefill_gpus", 0),
        recommended_decode_gpus=recs.get("decode_gpus", 0),
        predicted_ttft=recs.get("prefill_ttft", 0.0),
        predicted_itl=recs.get("decode_itl", 0.0),
        predicted_prefill_thpt_per_gpu=recs.get("prefill_thpt", 0.0),
    )

    # Estimate GPU hours
    if is_aic_run(log_path):
        metrics.gpu_hours_consumed = 0.0
    else:
        metrics.gpu_hours_consumed = num_deployments * 5 * 60 / 3600 * 4  # ~5min/deploy, 4 GPUs avg

    # Count configs
    metrics.num_prefill_configs_tested = len(list(results_dir.glob("prefill_*gpus*")))
    metrics.num_decode_configs_tested = len(list(results_dir.glob("decode_*gpus*")))

    logger.info(f"Loaded {method_name}: duration={duration:.0f}s, deploys={num_deployments}")
    return metrics


def load_validation(path: Path, metrics: ProfilingMetrics) -> ProfilingMetrics:
    if not path.exists():
        return metrics
    try:
        with open(path, "r") as f:
            data = json.load(f)
        metrics.validated = True
        for result in data.get("load_level_results", []):
            if result.get("load_level") == "medium":
                metrics.ttft_error_at_medium = result.get("ttft_error_pct")
    except Exception as e:
        logger.warning(f"Failed to load validation: {e}")
    return metrics


def run_comparison(args: argparse.Namespace) -> ComparisonResult:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = ComparisonResult(
        model=args.model, backend=args.backend,
        target_ttft=args.ttft, target_itl=args.itl,
        isl=args.isl, osl=args.osl,
    )

    validation_lookup = {}
    for v in args.validation_results:
        if ":" in v:
            name, path = v.split(":", 1)
            validation_lookup[name] = Path(path)

    # Load baselines
    if args.aic_results:
        m = load_method_results(Path(args.aic_results), "aic_offline")
        if m:
            if "aic_offline" in validation_lookup:
                m = load_validation(validation_lookup["aic_offline"], m)
            comparison.add_method_result(m)

    if args.online_results:
        m = load_method_results(Path(args.online_results), "online_aiperf")
        if m:
            if "online_aiperf" in validation_lookup:
                m = load_validation(validation_lookup["online_aiperf"], m)
            comparison.add_method_result(m)

    # Load additional methods
    for additional in args.additional_results:
        if ":" not in additional:
            continue
        name, path = additional.split(":", 1)
        m = load_method_results(Path(path), name)
        if m:
            if name in validation_lookup:
                m = load_validation(validation_lookup[name], m)
            comparison.add_method_result(m)

    if not comparison.method_metrics:
        logger.error("No results to compare")
        sys.exit(1)

    comparison.compute_comparison()
    comparison.save(output_dir / "comparison_results.json")

    summary = comparison.generate_summary_table()
    print(summary)
    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary)

    # Generate visualizations
    try:
        from benchmarks.profiler.comparison.visualize import visualize_comparison
        visualize_comparison(output_dir / "comparison_results.json", output_dir)
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    return comparison


def main():
    parser = create_parser()
    args = parser.parse_args()
    if not args.aic_results and not args.online_results and not args.additional_results:
        parser.error("Provide at least one of --aic-results, --online-results, or --additional-results")
    run_comparison(args)


if __name__ == "__main__":
    main()
