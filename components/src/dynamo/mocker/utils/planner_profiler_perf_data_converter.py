#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
This script converts planner profiler's results for mocker to use.

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

import argparse
import logging
import os
from pathlib import Path

import numpy as np

from dynamo.planner.utils.perf_interpolation import (
    DecodeInterpolator,
    PrefillInterpolator,
)
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_results_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    # Convert to absolute paths to handle relative directories properly
    args.profile_results_dir = str(Path(args.profile_results_dir).resolve())

    if not args.output_dir:
        args.output_dir = args.profile_results_dir
    else:
        args.output_dir = str(Path(args.output_dir).resolve())

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Converting profile results from {args.profile_results_dir} to {args.output_dir}..."
    )

    # first convert prefill
    prefill_interpolator = PrefillInterpolator(args.profile_results_dir)

    prefill_x = np.linspace(
        prefill_interpolator.ttft_interpolator.x.min(),
        prefill_interpolator.ttft_interpolator.x.max(),
        args.resolution,
    )
    prefill_y = prefill_interpolator.ttft_interpolator(prefill_x)

    result = {
        "prefill_isl": prefill_x.tolist(),
        "prefill_ttft_ms": prefill_y.tolist(),
    }

    # then convert decode
    decode_interpolator = DecodeInterpolator(
        args.profile_results_dir, resolution=args.resolution
    )

    decode_active_kv_tokens = decode_interpolator.xi * decode_interpolator.max_kv_tokens
    decode_context_length = decode_interpolator.yi
    decode_itl = decode_interpolator.itl_interpolator.transpose()

    result["decode_active_kv_tokens"] = decode_active_kv_tokens.tolist()
    result["decode_context_length"] = decode_context_length.tolist()
    result["decode_itl"] = decode_itl.tolist()

    np.savez(os.path.join(args.output_dir, "perf_data.npz"), **result)

    logger.info(f"Wrote perf data to {os.path.join(args.output_dir, 'perf_data.npz')}")
