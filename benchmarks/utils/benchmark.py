#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
import sys
from typing import Dict, Tuple

from benchmarks.utils.workflow import run_benchmark_workflow


def validate_inputs(inputs: Dict[str, str]) -> None:
    """Validate that all inputs are HTTP endpoints"""
    for label, value in inputs.items():
        if not value.lower().startswith(("http://", "https://")):
            raise ValueError(
                f"Input '{label}' must be an HTTP endpoint (starting with http:// or https://). Got: {value}"
            )

        # Validate reserved labels
        if label.lower() == "plots":
            raise ValueError(
                "Label 'plots' is reserved and cannot be used. Please choose a different label."
            )


def parse_input(input_str: str) -> Tuple[str, str]:
    """Parse input string in format key=value with additional validation"""
    if "=" not in input_str:
        raise ValueError(
            f"Invalid input format. Expected: <label>=<endpoint>, got: {input_str}"
        )

    parts = input_str.split("=", 1)  # Split on first '=' only
    if len(parts) != 2:
        raise ValueError(
            f"Invalid input format. Expected: <label>=<endpoint>, got: {input_str}"
        )

    label, value = parts

    if not label.strip():
        raise ValueError("Label cannot be empty")
    if not value.strip():
        raise ValueError("Value cannot be empty")

    label = label.strip()
    value = value.strip()

    # Validate label characters
    if not re.match(r"^[a-zA-Z0-9_-]+$", label):
        raise ValueError(
            f"Label must contain only letters, numbers, hyphens, and underscores. Invalid label: {label}"
        )

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

        # Validate that all inputs are HTTP endpoints
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
