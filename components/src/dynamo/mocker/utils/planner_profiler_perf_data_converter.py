#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
This module converts planner profiler's results for mocker to use.

Example prefill query:
    input:
        isl: 3000

    1. binary search prefill_isl to find isl_idx
    2. predicted TTFT is prefill_ttft_ms[isl_idx]
For chunked prefill, can ignore the KV cache read time and use ISL=prefill_tokens in this iteration.
This ignores the KV read time, which might leads to slightly lower latency..

Example decode query:
    input:
        active_kv_tokens: 10000
        batch_size: 100

    1. derive decode_context_length = active_kv_tokens / batch_size = 100
    2. binary search decode_active_kv_tokens to find kv_idx
    3. binary search decode_context_length to find context_idx
    4. predicted ITL is decode_itl[kv_idx, context_idx]

For aggregated engines, can separately query prefill and decode and use their sum as the total latency.
This ignores the fact that active tokens' up/down projection is usually combine in one kernel,
and might leads to slightly higher latency.
"""

import logging
from pathlib import Path

import numpy as np

from dynamo.planner.utils.perf_interpolation import (
    DecodeInterpolator,
    PrefillInterpolator,
)

logger = logging.getLogger(__name__)


def convert_profile_results_to_npz(
    profile_results_dir: str | Path,
    output_path: str | Path,
    resolution: int = 100,
) -> Path:
    """
    Convert planner profiler results directory to mocker-compatible NPZ format.

    Args:
        profile_results_dir: Path to directory containing selected_prefill_interpolation
            and selected_decode_interpolation subdirectories with raw_data.npz files.
        output_path: Full path where the output perf_data.npz will be written.
        resolution: Resolution for the interpolation grid (default: 100).

    Returns:
        Path to the generated NPZ file.
    """
    profile_results_dir = str(Path(profile_results_dir).resolve())
    output_path = Path(output_path)

    logger.info(f"Converting profile results from {profile_results_dir}...")

    # Convert prefill data
    prefill_interpolator = PrefillInterpolator(profile_results_dir)

    prefill_x = np.linspace(
        prefill_interpolator.ttft_interpolator.x.min(),
        prefill_interpolator.ttft_interpolator.x.max(),
        resolution,
    )
    prefill_y = prefill_interpolator.ttft_interpolator(prefill_x)

    result = {
        "prefill_isl": prefill_x.tolist(),
        "prefill_ttft_ms": prefill_y.tolist(),
    }

    # Convert decode data
    decode_interpolator = DecodeInterpolator(profile_results_dir, resolution=resolution)

    decode_active_kv_tokens = decode_interpolator.xi * decode_interpolator.max_kv_tokens
    decode_context_length = decode_interpolator.yi
    decode_itl = decode_interpolator.itl_interpolator.transpose()

    result["decode_active_kv_tokens"] = decode_active_kv_tokens.tolist()
    result["decode_context_length"] = decode_context_length.tolist()
    result["decode_itl"] = decode_itl.tolist()

    np.savez(output_path, **result)

    logger.info(f"Wrote perf data to {output_path}")
    return output_path


def is_profile_results_dir(path: Path) -> bool:
    """
    Check if the given path is a profile results directory (profiler-style format).

    A profile results directory contains:
    - selected_prefill_interpolation/raw_data.npz (or prefill_raw_data.json)
    - selected_decode_interpolation/raw_data.npz (or decode_raw_data.json)

    Args:
        path: Path to check.

    Returns:
        True if path is a profile results directory, False otherwise.
    """
    if not path.is_dir():
        return False

    has_prefill = (
        path / "selected_prefill_interpolation" / "raw_data.npz"
    ).exists() or (path / "prefill_raw_data.json").exists()

    has_decode = (path / "selected_decode_interpolation" / "raw_data.npz").exists() or (
        path / "decode_raw_data.json"
    ).exists()

    return has_prefill and has_decode


def is_mocker_format_npz(path: Path) -> bool:
    """
    Check if the given path is a mocker-format NPZ file.

    A mocker-format NPZ file contains:
    - prefill_isl, prefill_ttft_ms
    - decode_active_kv_tokens, decode_context_length, decode_itl

    Args:
        path: Path to check.

    Returns:
        True if path is a valid mocker-format NPZ file, False otherwise.
    """
    if not path.is_file():
        return False
    if path.suffix != ".npz":
        return False

    try:
        with np.load(path) as data:
            required_keys = {
                "prefill_isl",
                "prefill_ttft_ms",
                "decode_active_kv_tokens",
                "decode_context_length",
                "decode_itl",
            }
            return required_keys.issubset(data.keys())
    except Exception:
        return False


if __name__ == "__main__":
    import argparse

    from dynamo.runtime.logging import configure_dynamo_logging

    configure_dynamo_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_results_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    if not args.output_dir:
        output_dir = Path(args.profile_results_dir).resolve()
    else:
        output_dir = Path(args.output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "perf_data.npz"

    convert_profile_results_to_npz(
        args.profile_results_dir, output_path, args.resolution
    )
