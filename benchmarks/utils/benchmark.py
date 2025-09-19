#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
import sys
from typing import Dict, Tuple
from urllib.parse import urlsplit

from benchmarks.utils.workflow import has_http_scheme, run_benchmark_workflow
from deploy.utils.kubernetes import is_running_in_cluster


def validate_inputs(inputs: Dict[str, str]) -> None:
    """Validate that all inputs are HTTP endpoints or internal service URLs when running in cluster"""
    for label, value in inputs.items():
        v = value.strip()
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
                        f"Input '{label}' must be HTTP(S) or internal service URL. Got: {value}"
                    )
        else:
            if not has_http_scheme(v):
                raise ValueError(f"Input '{label}' must be HTTP endpoint. Got: {value}")

        # Validate reserved labels
        if label.lower() == "plots":
            raise ValueError("Label 'plots' is reserved")


def parse_input(input_str: str) -> Tuple[str, str]:
    """Parse input string in format key=value with additional validation"""
    if "=" not in input_str:
        raise ValueError(f"Invalid input format: {input_str}")

    parts = input_str.split("=", 1)  # Split on first '=' only
    if len(parts) != 2:
        raise ValueError(f"Invalid input format: {input_str}")

    label, value = parts

    if not label.strip():
        raise ValueError("Empty label")
    if not value.strip():
        raise ValueError("Empty value")

    label = label.strip()
    value = value.strip()

    # Validate label characters
    if not re.match(r"^[a-zA-Z0-9_-]+$", label):
        raise ValueError(f"Invalid label: {label}")

    return label, value


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Orchestrator")
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        help="Input in format <label>=<endpoint>. Can be specified multiple times for comparisons.",
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
        help="Model name",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/results", help="Output directory"
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.inputs:
        print("ERROR: At least one --input must be specified")
        return 1

    # Parse inputs
    try:
        parsed_inputs = {}
        for input_str in args.inputs:
            label, value = parse_input(input_str)
            if label in parsed_inputs:
                print(
                    f"ERROR: Duplicate label '{label}' found. Each label must be unique."
                )
                return 1
            parsed_inputs[label] = value

        # Check for plotting limitations
        if len(parsed_inputs) > 12:
            print(
                f"WARNING: You provided {len(parsed_inputs)} inputs, but the plotting system supports up to 12 inputs."
            )
            print(
                "Consider running separate benchmark sessions or grouping related comparisons together."
            )
            print(
                "Continuing with benchmark, but some inputs may not appear in plots..."
            )
            print()

        # Validate that inputs are HTTP endpoints or in-cluster service URLs
        validate_inputs(parsed_inputs)

    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # Run the benchmark workflow with the parsed inputs
    run_benchmark_workflow(
        inputs=parsed_inputs,
        isl=args.isl,
        std=args.std,
        osl=args.osl,
        model=args.model,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
