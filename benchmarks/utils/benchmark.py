#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import sys
from pathlib import Path

from benchmarks.utils.workflow import run_benchmark_workflow


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Orchestrator")
    parser.add_argument("--agg", help="Path to aggregated DGD manifest")
    parser.add_argument("--disagg", help="Path to disaggregated DGD manifest")
    parser.add_argument(
        "--vanilla",
        help="Path to vanilla backend manifest",
    )
    parser.add_argument(
        "--endpoint",
        help="Existing endpoint URL to benchmark (mutually exclusive with --agg/--disagg/--vanilla)",
    )
    parser.add_argument("--namespace", required=True, help="Kubernetes namespace")
    parser.add_argument("--isl", type=int, default=200, help="Input sequence length")
    parser.add_argument(
        "--std",
        type=int,
        default=10,
        help="Input sequence standard deviation",
    )
    parser.add_argument("--osl", type=int, default=200, help="Output sequence length")
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/results", help="Output directory"
    )
    args = parser.parse_args()

    # Check mutual exclusivity between endpoint and deployment manifests
    deployment_types = [args.agg, args.disagg, args.vanilla]
    has_deployment_manifests = any(deployment_types)
    has_endpoint = args.endpoint is not None

    if has_endpoint and has_deployment_manifests:
        print(
            "ERROR: --endpoint cannot be used together with --agg, --disagg, or --vanilla"
        )
        return 1

    if not has_endpoint and not has_deployment_manifests:
        print(
            "ERROR: Must specify either --endpoint OR at least one deployment type (--agg, --disagg, or --vanilla)"
        )
        return 1

    # Validate that specified manifest files exist
    for manifest_path in deployment_types:
        if manifest_path and not Path(manifest_path).is_file():
            print(f"ERROR: Manifest not found: {manifest_path}")
            return 1

    asyncio.run(
        run_benchmark_workflow(
            namespace=args.namespace,
            agg_manifest=args.agg,
            disagg_manifest=args.disagg,
            vanilla_manifest=args.vanilla,
            endpoint=args.endpoint,
            isl=args.isl,
            std=args.std,
            osl=args.osl,
            model=args.model,
            output_dir=args.output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
