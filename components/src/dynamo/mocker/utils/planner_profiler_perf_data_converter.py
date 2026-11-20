#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Turn a profiler results directory into an AIS perf model.

The profiler's thorough mode measures prefill time against input length and
decode time against KV usage and context length. Each measured point becomes
one forward pass, as the Planner reads the same directory. One AISimulate
``fpm_regression`` model per worker type is tuned with them and its state is
saved next to a file that names the three states.

The mocker does this at startup for ``--planner-profile-data DIRECTORY``. To
keep the result, or to give it to replay as ``ais_perf_model=``:

    python -m dynamo.mocker.utils.planner_profiler_perf_data_converter \\
        PROFILE_RESULTS_DIR OUTPUT_DIR --model MODEL
    python -m dynamo.mocker --ais-perf-model OUTPUT_DIR/ais_perf_model.json ...
"""

import json
import logging
import secrets
import shutil
from pathlib import Path
from typing import Any

import msgspec
import numpy as np

from dynamo.common.ais_perf_model import save_tuned_state_snapshot
from dynamo.common.forward_pass_metrics import FPM_VERSION, ForwardPassMetrics

logger = logging.getLogger(__name__)

DOCUMENT_NAME = "ais_perf_model.json"
# What AISimulate accepts for an identity fact that the producer does not know.
_UNKNOWN = "unknown"


def _iterations(fpms: list[ForwardPassMetrics]) -> list[list[dict[str, Any]]]:
    """Return one single-rank forward pass per measured point."""
    iterations = []
    for fpm in fpms:
        payload = msgspec.to_builtins(fpm)
        payload["version"] = FPM_VERSION
        iterations.append([payload])
    return iterations


def write_ais_perf_model(
    profile_results_dir: str | Path,
    output_dir: str | Path,
    *,
    model: str,
    backend: str = "vllm",
    nextn: int = 0,
) -> Path:
    """Write the AIS perf model of a profiler results directory.

    ``output_dir`` must not exist. It gets one state folder for each worker
    type (``prefill``, ``decode``, ``aggregated``) and the file
    ``ais_perf_model.json``, whose path is returned and is an AIS perf model
    value. The folder can be moved as a whole.

    ``model``, ``backend`` and ``nextn``, the speculative decoding depth of
    the engine that uses the models, are recorded as their identity. The
    hardware and the parallelism are recorded as unknown, and the KV capacity
    of the profiled decode engine as the deployment limit ``max_kv_tokens``,
    which an engine without its own KV capacity takes.

    Too few distinct points for a fit are a ``ValueError``, and
    ``output_dir`` is not created.
    """
    # Imported on first use: the Planner package needs AISimulate, and a
    # mocker with a lookup-table file imports this module without either.
    from dynamo.planner.monitoring.perf_metrics import (
        _convert_decode_profiling,
        _convert_prefill_profiling,
        _read_decode_profiling,
    )

    source = str(Path(profile_results_dir).resolve())
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"{output} already exists")

    prefill = _iterations(_convert_prefill_profiling(source))
    decode = _iterations(_convert_decode_profiling(source))
    max_kv_tokens = _read_decode_profiling(source)[3]
    identity = {
        "model": model,
        "backend": backend,
        "nextn": nextn,
        "system": _UNKNOWN,
        "tp": _UNKNOWN,
        "pp": _UNKNOWN,
        "attention_dp": _UNKNOWN,
        "estimation_mode": "fpm_regression",
        "deployment_limits": {"max_kv_tokens": max_kv_tokens},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    # Written next to the output and renamed, so a failure leaves nothing.
    staging = output.parent / f".{output.name}.{secrets.token_hex(8)}.tmp"
    staging.mkdir()
    written = False
    try:
        entries = []
        for worker_type, iterations in (
            ("prefill", prefill),
            ("decode", decode),
            ("aggregated", prefill + decode),
        ):
            save_tuned_state_snapshot(
                {**identity, "worker_type": worker_type},
                iterations,
                staging / worker_type,
            )
            entries.append(
                {
                    "worker_type": worker_type,
                    # Relative to the file, so the folder can be moved.
                    "from": {"state_snapshot": {"path": worker_type}},
                }
            )
        (staging / DOCUMENT_NAME).write_text(json.dumps(entries, indent=2) + "\n")
        staging.rename(output)
        written = True
    finally:
        if not written:
            shutil.rmtree(staging, ignore_errors=True)
    logger.info("Wrote the AIS perf model of %s to %s", source, output)
    return output / DOCUMENT_NAME


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

    parser = argparse.ArgumentParser(
        description="Turn a profiler results directory into an AIS perf model."
    )
    parser.add_argument(
        "profile_results_dir",
        type=Path,
        help="Directory with the profiler's prefill and decode data.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to create. It gets ais_perf_model.json, the value to pass "
        "as --ais-perf-model.",
    )
    parser.add_argument(
        "--model", required=True, help="Name of the profiled model, recorded as is."
    )
    parser.add_argument(
        "--backend",
        default="vllm",
        help="Backend of the profiled engine (default: vllm).",
    )
    parser.add_argument(
        "--nextn",
        type=int,
        default=0,
        help="Speculative decoding depth of the engine that uses the models, the "
        "mocker's --ais-nextn (default: 0).",
    )
    cli_args = parser.parse_args()
    print(
        write_ais_perf_model(
            cli_args.profile_results_dir,
            cli_args.output_dir,
            model=cli_args.model,
            backend=cli_args.backend,
            nextn=cli_args.nextn,
        )
    )
