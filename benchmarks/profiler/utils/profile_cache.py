# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import glob
import json
import logging
import os
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def check_prefill_results_exist(output_dir: str, tp_size: int, isl: int) -> bool:
    """Check if prefill results already exist for a given TP size."""
    work_dir = f"{output_dir}/prefill_tp{tp_size}"
    result_file = f"{work_dir}/aiperf_isl{isl}/*/profile_export_aiperf.json"

    # Check if the work directory exists
    if not os.path.exists(work_dir):
        return False

    # Look for the aiperf result file
    result_files = glob.glob(result_file)
    if not result_files:
        return False

    # Verify the result file has valid data
    try:
        with open(result_files[0], "r") as f:
            data = json.load(f)
            # Check if it has the required metrics
            if "time_to_first_token" in data and "avg" in data["time_to_first_token"]:
                logger.info(
                    f"Found existing prefill results for TP{tp_size} at {result_files[0]}"
                )
                return True
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        pass

    return False


def check_decode_results_exist(
    output_dir: str, tp_size: int, isl: int, osl: int
) -> bool:
    """Check if decode results already exist for a given TP size."""
    work_dir = f"{output_dir}/decode_tp{tp_size}"

    # Check if the work directory exists
    if not os.path.exists(work_dir):
        return False

    # Look for at least one decode result file
    result_pattern = (
        f"{work_dir}/aiperf_request*_isl{isl}_osl{osl}_n*/*/profile_export_aiperf.json"
    )
    result_files = glob.glob(result_pattern)

    if not result_files:
        return False

    # Verify at least one result file has valid data
    try:
        with open(result_files[0], "r") as f:
            data = json.load(f)
            # Check if it has the required metrics
            if "inter_token_latency" in data and "avg" in data["inter_token_latency"]:
                logger.info(
                    f"Found existing decode results for TP{tp_size} at {result_files[0]} (and {len(result_files)-1} others)"
                )
                return True
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        pass

    return False


def load_existing_prefill_results(
    output_dir: str, tp_size: int, isl: int
) -> Tuple[Optional[float], Optional[float]]:
    """Load existing prefill results from disk."""
    work_dir = f"{output_dir}/prefill_tp{tp_size}"
    result_file = f"{work_dir}/aiperf_isl{isl}/*/profile_export_aiperf.json"

    result_files = glob.glob(result_file)
    if result_files:
        try:
            with open(result_files[0], "r") as f:
                data = json.load(f)
                ttft = data["time_to_first_token"]["avg"]
                thpt_per_gpu = isl / ttft / tp_size * 1000
                return ttft, thpt_per_gpu
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            pass
    return None, None


def load_existing_decode_results(
    output_dir: str, tp_size: int, isl: int, osl: int
) -> List[Tuple[float, float, int]]:
    """Load existing decode results from disk."""
    work_dir = f"{output_dir}/decode_tp{tp_size}"

    result_pattern = (
        f"{work_dir}/aiperf_request*_isl{isl}_osl{osl}_n*/*/profile_export_aiperf.json"
    )
    result_files = glob.glob(result_pattern)

    decode_results = []
    for result_file in result_files:
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
                itl = data["inter_token_latency"]["avg"]
                thpt_per_gpu = data["output_token_throughput"]["avg"] / tp_size

                # Extract concurrency from filename
                match = re.search(r"aiperf_request(\d+)_", result_file)
                if match:
                    concurrency = int(match.group(1))
                    decode_results.append((itl, thpt_per_gpu, concurrency))
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            continue

    return decode_results
