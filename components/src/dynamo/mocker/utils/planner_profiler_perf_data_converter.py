#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Convert planner profiler results to mocker-compatible NPZ format.

Uses the FPM-based regression models from ``dynamo.planner.core.perf_model``
to evaluate prefill TTFT and decode ITL on a regular grid, producing the
lookup tables that the mocker uses for latency simulation.

Example prefill query:
    input:
        isl: 3000

    1. binary search prefill_isl to find isl_idx
    2. predicted TTFT is prefill_ttft_ms[isl_idx]

Example decode query:
    input:
        active_kv_tokens: 10000
        batch_size: 100

    1. derive decode_context_length = active_kv_tokens / batch_size = 100
    2. binary search decode_active_kv_tokens to find kv_idx
    3. binary search decode_context_length to find context_idx
    4. predicted ITL is decode_itl[kv_idx, context_idx]
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from dynamo.planner.core.perf_model import DecodeRegressionModel, PrefillRegressionModel
from dynamo.planner.monitoring.perf_metrics import (
    _convert_decode_profiling,
    _convert_prefill_profiling,
)

logger = logging.getLogger(__name__)


def convert_profile_results_to_npz(
    profile_results_dir: str | Path,
    output_path: str | Path,
    resolution: int = 100,
) -> Path:
    """Convert planner profiler results to mocker-compatible NPZ format.

    Loads the profiler's raw data (npz or JSON), fits FPM regression
    models, and evaluates them on a regular grid to produce the lookup
    tables the mocker expects.

    Args:
        profile_results_dir: Path containing selected_prefill_interpolation
            and selected_decode_interpolation subdirectories with raw_data.
        output_path: Full path where the output perf_data.npz will be written.
        resolution: Resolution for the evaluation grid (default: 100).

    Returns:
        Path to the generated NPZ file.
    """
    profile_results_dir = str(Path(profile_results_dir).resolve())
    output_path = Path(output_path)

    logger.info(f"Converting profile results from {profile_results_dir}...")

    result: dict[str, Any] = {}

    # --- Prefill: fit 1D model, evaluate TTFT on ISL grid ---
    prefill_fpms = _convert_prefill_profiling(profile_results_dir)
    if not prefill_fpms:
        raise FileNotFoundError(
            f"No prefill profiling data found in {profile_results_dir}"
        )

    prefill_model = PrefillRegressionModel(
        max_num_fpm_samples=len(prefill_fpms) + 10,
        min_observations=1,
    )
    prefill_model.load_benchmark_fpms(prefill_fpms)
    if not prefill_model._ensure_fitted():
        raise RuntimeError("Failed to fit prefill regression from profiling data")

    isl_values = [
        float(fpm.scheduled_requests.sum_prefill_tokens) for fpm in prefill_fpms
    ]
    prefill_x = np.linspace(min(isl_values), max(isl_values), resolution)
    prefill_y = np.array(
        [prefill_model._predict_wall_time(isl) * 1000.0 for isl in prefill_x]
    )

    result["prefill_isl"] = prefill_x.tolist()
    result["prefill_ttft_ms"] = prefill_y.tolist()

    # --- Decode: fit 2D model, evaluate ITL on (kv_tokens, context_length) grid ---
    decode_fpms = _convert_decode_profiling(profile_results_dir)
    if not decode_fpms:
        raise FileNotFoundError(
            f"No decode profiling data found in {profile_results_dir}"
        )

    decode_model = DecodeRegressionModel(
        max_num_fpm_samples=len(decode_fpms) + 10,
        min_observations=1,
    )
    decode_model.load_benchmark_fpms(decode_fpms)
    if not decode_model._ensure_fitted():
        raise RuntimeError("Failed to fit decode regression from profiling data")

    max_kv = max(
        float(fpm.scheduled_requests.sum_decode_kv_tokens) for fpm in decode_fpms
    )
    ctx_values = [
        float(fpm.scheduled_requests.sum_decode_kv_tokens)
        / max(1, fpm.scheduled_requests.num_decode_requests)
        for fpm in decode_fpms
        if fpm.scheduled_requests.num_decode_requests > 0
    ]
    max_ctx = max(ctx_values) if ctx_values else 8192.0

    decode_active_kv_tokens = np.linspace(0, max_kv, resolution)
    decode_context_length = np.linspace(1, max_ctx, resolution)

    decode_itl = np.zeros((resolution, resolution))
    for i, kv in enumerate(decode_active_kv_tokens):
        for j, ctx in enumerate(decode_context_length):
            bs = max(1, kv / ctx) if ctx > 0 else 1
            decode_itl[i, j] = decode_model._predict_2d(bs, kv) * 1000.0

    result["decode_active_kv_tokens"] = decode_active_kv_tokens.tolist()
    result["decode_context_length"] = decode_context_length.tolist()
    result["decode_itl"] = decode_itl.tolist()

    np.savez(output_path, **result)

    logger.info(f"Wrote perf data to {output_path}")
    return output_path


def is_profile_results_dir(path: Path) -> bool:
    """Check if the given path is a profile results directory.

    A profile results directory contains:
    - selected_prefill_interpolation/raw_data.npz (or prefill_raw_data.json)
    - selected_decode_interpolation/raw_data.npz (or decode_raw_data.json)
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
    """Check if the given path is a mocker-format NPZ file.

    A mocker-format NPZ file contains:
    - prefill_isl, prefill_ttft_ms
    - decode_active_kv_tokens, decode_context_length, decode_itl
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
