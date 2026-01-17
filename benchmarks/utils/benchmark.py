#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
import sys
from urllib.parse import urlsplit

from benchmarks.utils.workflow import has_http_scheme, run_benchmark_workflow
from deploy.utils.kubernetes import is_running_in_cluster


def validate_endpoint(endpoint: str) -> None:
    """Validate that endpoint is HTTP endpoint or internal service URL when running in cluster"""
    v = endpoint.strip()
    if is_running_in_cluster():
        # Allow HTTP(S) or internal service URLs like host[:port][/path]
        if has_http_scheme(v):
            pass
        else:
            parts = urlsplit(f"//{v}")
            host_ok = bool(parts.hostname)
            port_ok = parts.port is None or (1 <= parts.port <= 65535)
            if not (host_ok and port_ok):
                raise ValueError(
                    f"Endpoint must be HTTP(S) or internal service URL. Got: {endpoint}"
                )
    else:
        if not has_http_scheme(v):
            raise ValueError(f"Endpoint must be HTTP endpoint. Got: {endpoint}")


def validate_benchmark_name(name: str) -> None:
    """Validate benchmark name"""
    if not name.strip():
        raise ValueError("Benchmark name cannot be empty")

    name = name.strip()

    # Validate name characters
    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        raise ValueError(f"Invalid benchmark name: {name}")

    # Validate reserved names
    if name.lower() == "plots":
        raise ValueError("Benchmark name 'plots' is reserved")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Orchestrator")
    parser.add_argument(
        "--benchmark-name",
        required=True,
        help="Name/label for this benchmark (used in plots and results)",
    )
    parser.add_argument(
        "--endpoint-url",
        required=True,
        help="Endpoint to benchmark: HTTP(S) URL (e.g., http://localhost:8000) or in-cluster service URL host[:port]",
    )
    parser.add_argument("--isl", type=int, default=2000, help="Input sequence length")
    parser.add_argument(
        "--std",
        type=int,
        default=10,
        help="Input sequence standard deviation",
    )
    parser.add_argument("--osl", type=int, default=256, help="Output sequence length")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Model name (must match the model deployed at the endpoint)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/results", help="Output directory"
    )
    args = parser.parse_args()

    # Validate inputs
    try:
        validate_benchmark_name(args.benchmark_name)
        validate_endpoint(args.endpoint_url)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # Run the benchmark workflow with the parsed inputs
    run_benchmark_workflow(
        inputs={args.benchmark_name: args.endpoint_url},
        isl=args.isl,
        std=args.std,
        osl=args.osl,
        model=args.model,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
