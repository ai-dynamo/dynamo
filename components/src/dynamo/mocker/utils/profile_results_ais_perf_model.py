#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Turn a profiler results directory into an AIS perf model.

The profiler's thorough mode measures prefill time against input length and
decode time against KV usage and context length. Each measured point becomes
one forward pass, as the Planner reads the same directory. One AISimulate
``fpm_regression`` model per worker type is tuned with them and its state is
saved next to a file that names the three states:

    python -m dynamo.mocker.utils.profile_results_ais_perf_model \\
        PROFILE_RESULTS_DIR OUTPUT_DIR --model MODEL
    python -m dynamo.mocker --ais-perf-model OUTPUT_DIR/ais_perf_model.json ...

Replay takes the same file as ``ais_perf_model=``.
"""

import argparse
import json
import logging
import secrets
import shutil
from pathlib import Path
from typing import Any

import msgspec

from dynamo.common.ais_perf_model import save_tuned_state_snapshot
from dynamo.common.forward_pass_metrics import FPM_VERSION, ForwardPassMetrics
from dynamo.planner.monitoring.perf_metrics import (
    _convert_decode_profiling,
    _convert_prefill_profiling,
    _read_decode_profiling,
)
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)

# The file that names the saved states, in the output directory.
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
) -> Path:
    """Write the AIS perf model of a profiler results directory.

    ``output_dir`` must not exist. It gets one state folder for each worker
    type (``prefill``, ``decode``, ``aggregated``) and the file
    ``ais_perf_model.json``, whose path is returned and is an AIS perf model
    value. The folder can be moved as a whole.

    ``model`` and ``backend`` are recorded as the identity of the three
    models. The hardware and the parallelism are recorded as unknown, and
    the KV capacity of the profiled decode engine as the deployment limit
    ``max_kv_tokens``, which an engine without its own KV capacity takes.

    Too few distinct points for a fit are a ``ValueError``, and
    ``output_dir`` is not created.
    """
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


if __name__ == "__main__":
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
    cli_args = parser.parse_args()
    print(
        write_ais_perf_model(
            cli_args.profile_results_dir,
            cli_args.output_dir,
            model=cli_args.model,
            backend=cli_args.backend,
        )
    )
