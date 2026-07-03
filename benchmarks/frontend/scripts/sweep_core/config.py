# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Typed SweepConfig construction from argparse Namespace.

Centralizes the parsing of CLI arguments into the SweepConfig data model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

from sweep_core.models import K8sConfig, SweepConfig

SCRIPT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_OSL = 256
DEFAULT_SPEEDUP = 1.0
DEFAULT_BENCHMARK_DURATION = 60
DEFAULT_MAX_CONSECUTIVE_FAILS = 2
DEFAULT_COOLDOWN = 3


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the argument parser for sweep_runner.py."""
    parser = argparse.ArgumentParser(
        description="Frontend performance sweep runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Local smoke test
  python3 sweep_runner.py --tokenizers hf,fastokens --concurrency 32 --isl 512 \\
      --benchmark-duration 30 --speedup-ratio 1000000

  # K8s sweep with DGD
  python3 sweep_runner.py --mode k8s --tokenizers hf,fastokens --concurrency 50,100 --isl 512

  # K8s with custom deploy template
  python3 sweep_runner.py --mode k8s --deploy-template dgd/templates/vllm.yaml \\
      --tokenizers hf --concurrency 128 --isl 1024

  # Transport saturation (high concurrency, vary workers)
  python3 sweep_runner.py --tokenizers hf --concurrency 4096 \\
      --num-requests 16384,32768 --workers 1,2,4,8 --speedup-ratio 1000000

  # Dry run
  python3 sweep_runner.py --dry-run --tokenizers hf,fastokens --concurrency 32,64 --isl 512,1024
""",
    )

    # Common options
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model path")
    parser.add_argument(
        "--model-name", default="", help="Served model name (default: same as --model)"
    )
    parser.add_argument(
        "--mode",
        choices=["local", "k8s"],
        default="local",
        help="Execution mode: local (run_perf.sh) or k8s (DGD + aiperf)",
    )
    parser.add_argument(
        "--backend",
        choices=["mocker", "vllm"],
        default="mocker",
        help="Engine backend: mocker (synthetic) or vllm (real inference)",
    )
    parser.add_argument(
        "--tokenizers",
        default="hf,fastokens",
        help="Comma-separated tokenizer backends (hf, fastokens)",
    )
    parser.add_argument(
        "--concurrency", default="50,100,200", help="Comma-separated concurrency levels"
    )
    parser.add_argument(
        "--isl", default="512,1024,2048", help="Comma-separated ISL values"
    )
    parser.add_argument(
        "--osl", type=int, default=DEFAULT_OSL, help="Output sequence length"
    )
    parser.add_argument(
        "--workers", default="2", help="Comma-separated worker counts per model"
    )
    parser.add_argument(
        "--num-models",
        type=int,
        default=1,
        help="Number of model instances",
    )
    parser.add_argument(
        "--aiperf-targets",
        choices=["first", "all"],
        default="first",
        help="'first': aiperf targets model-1 only. 'all': run aiperf for each model.",
    )
    parser.add_argument(
        "--speedup-ratio",
        type=float,
        default=DEFAULT_SPEEDUP,
        help="Mocker speedup (0=infinite)",
    )
    parser.add_argument(
        "--benchmark-duration",
        type=int,
        default=DEFAULT_BENCHMARK_DURATION,
        help="aiperf duration (seconds)",
    )
    parser.add_argument(
        "--num-requests",
        default=None,
        help="Comma-separated request counts (overrides --benchmark-duration)",
    )
    parser.add_argument(
        "--rps",
        default=None,
        help="Comma-separated target request rates (req/s)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto timestamped)",
    )
    parser.add_argument(
        "--max-consecutive-fails",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_FAILS,
    )
    parser.add_argument(
        "--cooldown", type=int, default=DEFAULT_COOLDOWN, help="Seconds between runs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without executing"
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Skip per-run report generation"
    )
    parser.add_argument(
        "--isolation",
        choices=["fresh_per_run", "reuse_by_deploy_key"],
        default="fresh_per_run",
        help="Isolation policy (default: fresh_per_run)",
    )

    # K8s-specific options
    k8s_group = parser.add_argument_group("K8s mode options")
    k8s_group.add_argument("--namespace", default="dynamo-bench", help="K8s namespace")
    k8s_group.add_argument(
        "--endpoint", default=None, help="K8s frontend endpoint (host:port)"
    )
    k8s_group.add_argument("--dgd-name", default="", help="DynamoGraphDeployment name")
    k8s_group.add_argument(
        "--image", default="", help="Container image for k8s deployment"
    )
    k8s_group.add_argument(
        "--deploy-template",
        default="",
        help="Path to deploy.yaml template (enables template-based deployment)",
    )
    k8s_group.add_argument(
        "--reset-strategy",
        choices=["none", "frontend", "graph"],
        default="graph",
        help="K8s reset strategy per run (default: graph)",
    )
    k8s_group.add_argument(
        "--deploy", action="store_true", help="Deploy infrastructure before sweeping"
    )
    k8s_group.add_argument(
        "--frontend-port", type=int, default=8000, help="Frontend HTTP port"
    )
    k8s_group.add_argument(
        "--worker-replicas", type=int, default=1, help="Number of worker pod replicas"
    )
    k8s_group.add_argument(
        "--frontend-replicas",
        type=int,
        default=1,
        help="Number of frontend pod replicas",
    )
    k8s_group.add_argument(
        "--request-plane", default="tcp", help="Request plane transport"
    )
    k8s_group.add_argument(
        "--event-plane", default="nats", help="Event plane transport"
    )
    k8s_group.add_argument(
        "--router-mode", default="round-robin", help="Frontend router mode"
    )
    k8s_group.add_argument("--hf-token", default="", help="HuggingFace token for k8s")
    k8s_group.add_argument(
        "--image-pull-secret", default="", help="Image pull secret name"
    )
    k8s_group.add_argument(
        "--export-level", default="summary", help="aiperf export level"
    )

    # Passthrough args for run_perf.sh
    parser.add_argument(
        "passthrough", nargs="*", help="Extra args passed to run_perf.sh (after --)"
    )

    return parser


def config_from_args(args: argparse.Namespace) -> SweepConfig:
    """Convert parsed argparse Namespace to SweepConfig."""
    # Parse comma-separated lists
    tokenizers = [t.strip() for t in args.tokenizers.split(",")]
    concurrencies = [int(c) for c in args.concurrency.split(",")]
    isls = [int(i) for i in args.isl.split(",")]
    worker_counts = [int(w) for w in args.workers.split(",")]
    num_requests_list: List[Optional[int]] = (
        [int(n) for n in args.num_requests.split(",")] if args.num_requests else [None]
    )
    rps_list: List[Optional[int]] = (
        [int(r) for r in args.rps.split(",")] if args.rps else [None]
    )

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        if args.mode == "k8s" and Path("/artifacts").is_dir():
            # Inside a k8s pod with /artifacts PVC mounted
            output_dir = f"/artifacts/sweep_{ts}"
        else:
            # Local or k8s-from-host: use repo artifacts directory
            output_dir = str(REPO_ROOT / "artifacts" / f"sweep_{ts}")

    # Build K8s config
    k8s_config = K8sConfig(
        namespace=args.namespace,
        dgd_name=args.dgd_name,
        image=args.image,
        frontend_port=args.frontend_port,
        worker_replicas=args.worker_replicas,
        frontend_replicas=args.frontend_replicas,
        deploy_template=args.deploy_template,
        reset_strategy=args.reset_strategy,
        request_plane=args.request_plane,
        event_plane=args.event_plane,
        router_mode=args.router_mode,
        deploy=args.deploy,
        hf_token=args.hf_token,
        image_pull_secret=args.image_pull_secret,
        export_level=args.export_level,
    )

    # Compute k8s endpoint
    if args.endpoint:
        k8s_config.endpoint = args.endpoint
    elif k8s_config.dgd_name:
        k8s_config.endpoint = (
            f"{k8s_config.dgd_name}-frontend:{k8s_config.frontend_port}"
        )
    else:
        k8s_config.endpoint = f"frontend:{k8s_config.frontend_port}"

    return SweepConfig(
        model=args.model,
        model_name=args.model_name or args.model,
        mode=args.mode,
        backend=args.backend,
        tokenizers=tokenizers,
        concurrencies=concurrencies,
        isls=isls,
        osl=args.osl,
        worker_counts=worker_counts,
        num_models=args.num_models,
        aiperf_targets=args.aiperf_targets,
        speedup_ratio=args.speedup_ratio,
        benchmark_duration=args.benchmark_duration,
        num_requests_list=num_requests_list,
        rps_list=rps_list,
        output_dir=output_dir,
        max_consecutive_fails=args.max_consecutive_fails,
        cooldown=args.cooldown,
        dry_run=args.dry_run,
        no_report=args.no_report,
        isolation_policy=args.isolation,
        passthrough_args=args.passthrough or [],
        k8s=k8s_config,
    )
