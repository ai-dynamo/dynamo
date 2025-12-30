# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate deployment performance at multiple load levels."""

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class LoadLevelResult:
    load_level: str
    concurrency: int
    concurrency_pct: float
    ttft_p50: Optional[float] = None
    ttft_p99: Optional[float] = None
    itl_p50: Optional[float] = None
    itl_p99: Optional[float] = None
    goodput_rps: Optional[float] = None
    sla_hit_rate: Optional[float] = None
    ttft_error_pct: Optional[float] = None
    itl_error_pct: Optional[float] = None


@dataclass
class ValidationResult:
    timestamp: str
    model: str
    url: str
    ttft_target: float
    itl_target: float
    predicted_ttft: float
    predicted_itl: float
    max_batch_size: int
    isl: int
    osl: int
    load_level_results: list[LoadLevelResult]

    def save(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                **{k: v for k, v in self.__dict__.items() if k != "load_level_results"},
                "load_level_results": [asdict(r) for r in self.load_level_results],
            }, f, indent=2)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate deployment at multiple load levels")
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--isl", type=int, default=2048)
    parser.add_argument("--osl", type=int, default=256)
    parser.add_argument("--ttft-target", type=float, required=True)
    parser.add_argument("--itl-target", type=float, required=True)
    parser.add_argument("--predicted-ttft", type=float, default=0.0)
    parser.add_argument("--predicted-itl", type=float, default=0.0)
    parser.add_argument("--max-batch-size", type=int, required=True)
    parser.add_argument("--requests-per-level", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="validation_results")
    return parser


def parse_aiperf_output(artifact_dir: Path) -> dict:
    for pattern in ["**/summary.json", "**/profile_export.json"]:
        files = list(artifact_dir.glob(pattern))
        if files:
            try:
                with open(files[0], "r") as f:
                    data = json.load(f)
                metrics = {}
                if "time_to_first_token" in data:
                    metrics["ttft_p50"] = data["time_to_first_token"].get("p50")
                    metrics["ttft_p99"] = data["time_to_first_token"].get("p99")
                if "inter_token_latency" in data:
                    metrics["itl_p50"] = data["inter_token_latency"].get("p50")
                    metrics["itl_p99"] = data["inter_token_latency"].get("p99")
                if "goodput" in data:
                    metrics["goodput_rps"] = data["goodput"].get("request_goodput")
                    metrics["sla_hit_rate"] = (data["goodput"].get("goodput_ratio", 0)) * 100
                return metrics
            except Exception:
                pass
    return {}


def run_aiperf(url: str, model: str, isl: int, osl: int, concurrency: int,
               request_count: int, ttft_target: float, itl_target: float,
               artifact_dir: Path) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if not url.startswith("http"):
        url = f"http://{url}"

    cmd = [
        "aiperf", "profile",
        "--model", model, "--tokenizer", model,
        "--endpoint-type", "chat",
        "--url", url.replace("http://", "").replace("https://", ""),
        "--streaming",
        "--synthetic-input-tokens-mean", str(isl),
        "--synthetic-input-tokens-stddev", "0",
        "--output-tokens-mean", str(osl),
        "--output-tokens-stddev", "0",
        "--concurrency", str(concurrency),
        "--request-count", str(request_count),
        "--warmup-request-count", "10",
        "--artifact-dir", str(artifact_dir),
        "--goodput", f"time_to_first_token:{ttft_target} inter_token_latency:{itl_target}",
    ]

    logger.info(f"Running AIPerf: concurrency={concurrency}")
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as e:
        logger.error(f"AIPerf failed: {e}")
        return {}
    return parse_aiperf_output(artifact_dir)


def run_validation(args: argparse.Namespace) -> ValidationResult:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_bs = args.max_batch_size
    load_levels = {
        "idle": (1, 1 / max_bs),
        "medium": (max(1, int(max_bs * 0.5)), 0.5),
        "saturation": (max(1, int(max_bs * 0.9)), 0.9),
        "overload": (max(1, int(max_bs * 1.1)), 1.1),
    }

    results = []
    for level, (conc, pct) in load_levels.items():
        logger.info(f"Testing {level}: concurrency={conc}")
        metrics = run_aiperf(
            args.url, args.model, args.isl, args.osl, conc,
            args.requests_per_level, args.ttft_target, args.itl_target,
            output_dir / level,
        )

        result = LoadLevelResult(
            load_level=level, concurrency=conc, concurrency_pct=pct,
            ttft_p50=metrics.get("ttft_p50"),
            ttft_p99=metrics.get("ttft_p99"),
            itl_p50=metrics.get("itl_p50"),
            itl_p99=metrics.get("itl_p99"),
            goodput_rps=metrics.get("goodput_rps"),
            sla_hit_rate=metrics.get("sla_hit_rate"),
        )

        if args.predicted_ttft > 0 and result.ttft_p50:
            result.ttft_error_pct = (args.predicted_ttft - result.ttft_p50) / result.ttft_p50 * 100
        if args.predicted_itl > 0 and result.itl_p50:
            result.itl_error_pct = (args.predicted_itl - result.itl_p50) / result.itl_p50 * 100

        results.append(result)
        if result.ttft_p50:
            logger.info(f"  TTFT p50: {result.ttft_p50:.1f}ms, error: {result.ttft_error_pct:.1f}%"
                       if result.ttft_error_pct else f"  TTFT p50: {result.ttft_p50:.1f}ms")

    validation = ValidationResult(
        timestamp=datetime.now().isoformat(),
        model=args.model, url=args.url,
        ttft_target=args.ttft_target, itl_target=args.itl_target,
        predicted_ttft=args.predicted_ttft, predicted_itl=args.predicted_itl,
        max_batch_size=args.max_batch_size, isl=args.isl, osl=args.osl,
        load_level_results=results,
    )
    validation.save(output_dir / "validation.json")

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"{'Level':<12} {'Conc':<6} {'TTFT p50':<12} {'ITL p50':<12} {'Error%':<10}")
    print("-" * 80)
    for r in results:
        ttft = f"{r.ttft_p50:.1f}ms" if r.ttft_p50 else "N/A"
        itl = f"{r.itl_p50:.1f}ms" if r.itl_p50 else "N/A"
        err = f"{r.ttft_error_pct:.1f}%" if r.ttft_error_pct is not None else "N/A"
        print(f"{r.load_level:<12} {r.concurrency:<6} {ttft:<12} {itl:<12} {err:<10}")
    print("=" * 80)

    return validation


def main():
    args = create_parser().parse_args()
    run_validation(args)


if __name__ == "__main__":
    main()
