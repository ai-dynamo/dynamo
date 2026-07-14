# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the nine-cell vLLM/native/custom Qwen3-VL image QPS sweep."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.multimodal.sweep.config import (  # noqa: E402
    BenchmarkConfig,
    SweepConfig,
)
from benchmarks.multimodal.sweep.orchestrator import run_sweep  # noqa: E402

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
RATES = (16, 24, 32)
REQUESTS = 1000
ENCODER_CLASS = "examples.custom_encoder.qwen3_vl_vision_encoder.Qwen3VLVisionEncoder"

AIPERF_EXTRA_ARGS = [
    "--endpoint-type",
    "chat",
    "--endpoint",
    "/v1/chat/completions",
    "--warmup-request-rate",
    "1000",
    "--warmup-arrival-pattern",
    "constant",
    "--random-seed",
    "42",
    "--workers-max",
    "20",
    "--record-processors",
    "32",
    "--use-server-token-count",
    "--request-timeout-seconds",
    "300",
]


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _metadata(
    model: str, rates: tuple[int, ...], smoke: bool, workload_dir: Path
) -> dict[str, Any]:
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = None
    try:
        aiperf_version = importlib.metadata.version("aiperf")
    except importlib.metadata.PackageNotFoundError:
        aiperf_version = _command_output(["aiperf", "--version"])
    manifest_path = workload_dir / "workload_manifest.json"
    return {
        "model": model,
        "rates": list(rates),
        "requests_per_cell": 1 if smoke else REQUESTS,
        "warmup_requests": 0 if smoke else 20,
        "osl": 1 if smoke else 70,
        "streaming": True,
        "aiperf_extra_args": AIPERF_EXTRA_ARGS,
        "custom_encoder_class": ENCODER_CLASS,
        "custom_encoder_load": (
            "AutoProcessor and Qwen3VLForConditionalGeneration are loaded in bf16; "
            "model.visual is retained and the remaining checkpoint is released"
        ),
        "dynamo_commit": os.environ.get("DYNAMO_BENCHMARK_COMMIT")
        or _command_output(["git", "rev-parse", "HEAD"]),
        "dynamo_branch": os.environ.get("DYNAMO_BENCHMARK_BRANCH")
        or _command_output(["git", "branch", "--show-current"]),
        "container_image": os.environ.get("DYNAMO_BENCHMARK_IMAGE"),
        "vllm_version": vllm_version,
        "aiperf_version": aiperf_version,
        "workload_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "gpu": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "host": platform.node(),
    }


def _config(
    model: str,
    input_file: Path,
    rate: int,
    output_dir: Path,
    configs: list[BenchmarkConfig],
    smoke: bool,
) -> SweepConfig:
    return SweepConfig(
        model=model,
        request_rates=[1 if smoke else rate],
        osl=1 if smoke else 70,
        conversation_num=1 if smoke else REQUESTS,
        warmup_count=0 if smoke else 20,
        port=8000,
        timeout=1200,
        input_files=[str(input_file.resolve())],
        configs=configs,
        output_dir=str(output_dir),
        skip_plots=True,
        restart_server_every_benchmark=True,
        env={
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DYN_MAX_MODEL_LEN": "2048",
            "DYN_MAX_NUM_SEQS": "64",
            "DYN_VLLM_GPU_MEMORY_UTILIZATION": "0.7",
            "DYN_QWEN3_VL_EMBEDDING_CACHE_BYTES": "0",
            "DYN_QWEN3_VL_PREPROCESS_CACHE_SIZE": "0",
            "DYN_CUSTOM_ENCODER_QUEUE_WAIT_MS": "1",
        },
        aiperf_extra_args=AIPERF_EXTRA_ARGS,
    )


def run_matrix(
    workload_dir: Path,
    output_dir: Path,
    model: str,
    rates: tuple[int, ...],
    smoke: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "benchmark_metadata.json"
    metadata = _metadata(model, rates, smoke, workload_dir)
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise RuntimeError(
                "refusing to resume benchmark with different provenance: "
                f"existing={existing}, requested={metadata}"
            )
    else:
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    native_workflow = (
        REPO_ROOT / "examples/custom_encoder/benchmark/workflows/dynamo_native.sh"
    )
    vllm_workflow = (
        REPO_ROOT / "examples/custom_encoder/benchmark/workflows/vllm_serve.sh"
    )
    custom_workflow = REPO_ROOT / "examples/custom_encoder/launch/agg_qwen3_vl.sh"

    for rate in rates:
        native_input = workload_dir / f"image_native_qps{rate}_1000_isl515.jsonl"
        custom_input = workload_dir / f"image_custom_qps{rate}_1000_isl515.jsonl"
        native_configs = [
            BenchmarkConfig(
                label="vllm-serve",
                workflow=str(vllm_workflow),
            ),
            BenchmarkConfig(
                label="dynamo-native",
                workflow=str(native_workflow),
            ),
        ]
        custom_configs = [
            BenchmarkConfig(
                label="dynamo-custom-encoder",
                workflow=str(custom_workflow),
                extra_args=["--max-num-seqs", "64"],
            )
        ]
        for config in (
            _config(model, native_input, rate, output_dir, native_configs, smoke),
            _config(model, custom_input, rate, output_dir, custom_configs, smoke),
        ):
            config.validate(repo_root=REPO_ROOT)
            run_sweep(config, repo_root=REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--rates", type=int, nargs="+", default=list(RATES))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_matrix(
        workload_dir=args.workload_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        model=args.model,
        rates=tuple(args.rates),
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
