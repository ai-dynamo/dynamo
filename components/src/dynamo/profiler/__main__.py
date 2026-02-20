# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Entry point for the Dynamo profiler.

Usage::

    python -m dynamo.profiler --config <json string or path to json/yaml>
    python -m dynamo.profiler --config '{"model": "Qwen/Qwen3-32B", ...}'
    python -m dynamo.profiler --config /path/to/dgdr_spec.yaml
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import yaml

from dynamo.profiler.utils.dgdr_v1beta1_types import DynamoGraphDeploymentRequestSpec

from .profile_sla import (
    DEFAULT_DECODE_INTERPOLATION_GRANULARITY,
    DEFAULT_DEPLOYMENT_TIMEOUT,
    DEFAULT_DRY_RUN,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PREFILL_INTERPOLATION_GRANULARITY,
    ProfilerOperationalConfig,
    run_profile,
)

logger = logging.getLogger(__name__)


def _parse_dgdr_spec(config_arg: str) -> DynamoGraphDeploymentRequestSpec:
    """Parse a DGDR spec from a CLI ``--config`` argument.

    Accepts a file path (JSON/YAML) or an inline JSON string.
    """
    path = Path(config_arg)
    if path.is_file():
        text = path.read_text()
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = yaml.safe_load(text)
        return DynamoGraphDeploymentRequestSpec.model_validate(data)

    try:
        data = json.loads(config_arg)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"--config value is neither a valid file path nor valid JSON. "
            f"File not found: '{config_arg}'. JSON parse error: {e}"
        ) from e

    return DynamoGraphDeploymentRequestSpec.model_validate(data)


def _parse_args() -> tuple[DynamoGraphDeploymentRequestSpec, ProfilerOperationalConfig]:
    parser = argparse.ArgumentParser(description="Dynamo Profiler")
    parser.add_argument(
        "--config",
        required=True,
        help="DynamoGraphDeploymentRequestSpec as JSON string or path to JSON/YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Path to the output results directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--deployment-timeout",
        type=int,
        default=DEFAULT_DEPLOYMENT_TIMEOUT,
        help=f"Max seconds to wait for deployment readiness (default: {DEFAULT_DEPLOYMENT_TIMEOUT})",
    )
    parser.add_argument(
        "--prefill-interpolation-granularity",
        type=int,
        default=DEFAULT_PREFILL_INTERPOLATION_GRANULARITY,
        help=f"Number of ISL samples for prefill interpolation (default: {DEFAULT_PREFILL_INTERPOLATION_GRANULARITY})",
    )
    parser.add_argument(
        "--decode-interpolation-granularity",
        type=int,
        default=DEFAULT_DECODE_INTERPOLATION_GRANULARITY,
        help=f"Number of samples for decode interpolation (default: {DEFAULT_DECODE_INTERPOLATION_GRANULARITY})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=DEFAULT_DRY_RUN,
        help="Skip deployments and benchmarking (dev mode)",
    )
    args = parser.parse_args()

    dgdr = _parse_dgdr_spec(args.config)
    ops = ProfilerOperationalConfig(
        output_dir=args.output_dir,
        deployment_timeout=args.deployment_timeout,
        prefill_interpolation_granularity=args.prefill_interpolation_granularity,
        decode_interpolation_granularity=args.decode_interpolation_granularity,
        dry_run=args.dry_run,
    )

    return dgdr, ops


def main():
    dgdr, ops = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(ops.output_dir, exist_ok=True)
    log_file_handler = logging.FileHandler(f"{ops.output_dir}/profile_sla.log")
    log_file_handler.setLevel(logging.INFO)
    log_file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(log_file_handler)

    asyncio.run(run_profile(dgdr, ops))


if __name__ == "__main__":
    main()
