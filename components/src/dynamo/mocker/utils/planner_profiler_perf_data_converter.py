#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Convert planner profiler results to mocker-compatible NPZ format.

Deprecated with the profiler results directory form of ``--planner-profile-data``
and removed with it: ``dynamo.mocker.utils.profile_results_ais_perf_model``
turns the same directory into an AIS perf model.

Fits one linear model to the prefill points and one to the decode points and
evaluates them on a regular grid, producing the lookup tables that the mocker
uses for latency simulation.

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

import numpy as np

logger = logging.getLogger(__name__)

# A slope of the decode batch size this far below zero is kept as noise; one
# further below is set to zero.
_BATCH_SLOPE_NOISE = 1e-6
# No prediction is below this many seconds.
_MIN_WALL_TIME = 1e-6


def _fit_wall_time(
    features: list[list[float]], wall_times: list[float], *, batch_column: int | None
) -> tuple[np.ndarray, float] | None:
    """Fit ``wall_time = features @ slopes + intercept`` by least squares.

    Returns ``(slopes, intercept)``, or ``None`` without a point or when the
    slope of a token count is negative: more tokens cannot take less time.
    The slope of ``batch_column``, the decode batch size, may be negative
    next to the KV token count; it is kept as noise or set to zero.
    """
    if not wall_times:
        return None
    x = np.array(features, dtype=float)
    y = np.array(wall_times, dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Profiling data contains NaN or infinity")
    x_mean = x.mean(axis=0)
    y_mean = y.mean()
    # The minimum-norm solution on centered data with the singular value
    # cutoff of the scikit-learn regression this replaced, so that the tables
    # keep their values, also for points on one line.
    cutoff = max(x.shape) * float(np.finfo(float).eps)
    slopes = np.linalg.lstsq(x - x_mean, y - y_mean, rcond=cutoff)[0]
    intercept = float(y_mean - np.dot(x_mean, slopes))
    for column in np.flatnonzero(slopes < 0):
        if column != batch_column:
            logger.warning(
                "Regression produced negative coefficients %s, model rejected",
                slopes.tolist(),
            )
            return None
        if slopes[column] < -_BATCH_SLOPE_NOISE:
            slopes[column] = 0.0
    return slopes, intercept


def convert_profile_results_to_npz(
    profile_results_dir: str | Path,
    output_path: str | Path,
    resolution: int = 100,
) -> Path:
    """Convert planner profiler results to mocker-compatible NPZ format.

    Loads the profiler's raw data (npz or JSON), fits one linear model for
    prefill and one for decode, and evaluates them on a regular grid to
    produce the lookup tables the mocker expects.

    Args:
        profile_results_dir: Path containing selected_prefill_interpolation
            and selected_decode_interpolation subdirectories with raw_data.
        output_path: Full path where the output perf_data.npz will be written.
        resolution: Resolution for the evaluation grid (default: 100).

    Returns:
        Path to the generated NPZ file.
    """
    from dynamo.planner.monitoring.perf_metrics import (
        _convert_decode_profiling,
        _convert_prefill_profiling,
    )

    profile_results_dir = str(Path(profile_results_dir).resolve())
    output_path = Path(output_path)

    logger.info(f"Converting profile results from {profile_results_dir}...")

    # --- Prefill: fit 1D model, evaluate TTFT on ISL grid ---
    prefill_fpms = _convert_prefill_profiling(profile_results_dir)
    if not prefill_fpms:
        raise FileNotFoundError(
            f"No prefill profiling data found in {profile_results_dir}"
        )

    # A point without a wall time is not an observation.
    timed = [fpm for fpm in prefill_fpms if fpm.wall_time != 0.0]
    prefill_fit = _fit_wall_time(
        [[float(fpm.scheduled_requests.sum_prefill_tokens)] for fpm in timed],
        [fpm.wall_time for fpm in timed],
        batch_column=None,
    )
    if prefill_fit is None:
        raise RuntimeError("Failed to fit prefill regression from profiling data")
    slopes, intercept = prefill_fit

    isl_values = [
        float(fpm.scheduled_requests.sum_prefill_tokens) for fpm in prefill_fpms
    ]
    prefill_isl = np.linspace(min(isl_values), max(isl_values), resolution)
    prefill_ttft_ms = (
        np.maximum(_MIN_WALL_TIME, prefill_isl * slopes[0] + intercept) * 1000.0
    )

    # --- Decode: fit 2D model, evaluate ITL on (kv_tokens, context_length) grid ---
    decode_fpms = _convert_decode_profiling(profile_results_dir)
    if not decode_fpms:
        raise FileNotFoundError(
            f"No decode profiling data found in {profile_results_dir}"
        )

    timed = [fpm for fpm in decode_fpms if fpm.wall_time != 0.0]
    decode_fit = _fit_wall_time(
        [
            [
                float(fpm.scheduled_requests.num_decode_requests),
                float(fpm.scheduled_requests.sum_decode_kv_tokens),
            ]
            for fpm in timed
        ],
        [fpm.wall_time for fpm in timed],
        batch_column=0,
    )
    if decode_fit is None:
        raise RuntimeError("Failed to fit decode regression from profiling data")
    slopes, intercept = decode_fit

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

    # Rows are KV token counts and columns context lengths. A context length
    # of zero, which only data without KV usage gives, counts as one request.
    kv_tokens = decode_active_kv_tokens[:, np.newaxis]
    context = decode_context_length[np.newaxis, :]
    batch_size = np.maximum(
        1,
        np.divide(
            kv_tokens,
            context,
            out=np.zeros((resolution, resolution)),
            where=context > 0,
        ),
    )
    decode_itl = (
        np.maximum(
            _MIN_WALL_TIME,
            batch_size * slopes[0] + kv_tokens * slopes[1] + intercept,
        )
        * 1000.0
    )

    np.savez(
        output_path,
        prefill_isl=prefill_isl,
        prefill_ttft_ms=prefill_ttft_ms,
        decode_active_kv_tokens=decode_active_kv_tokens,
        decode_context_length=decode_context_length,
        decode_itl=decode_itl,
    )

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

    from dynamo.common.ais_perf_model import AIS_OLD_INPUTS_REMOVE_IN
    from dynamo.common.configuration.utils import Deprecated
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
    Deprecated(
        "python -m dynamo.mocker.utils.profile_results_ais_perf_model",
        remove_in=AIS_OLD_INPUTS_REMOVE_IN,
    ).warn("python -m dynamo.mocker.utils.planner_profiler_perf_data_converter")
