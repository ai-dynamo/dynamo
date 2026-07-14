# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the three-cell custom-encoder Qwen2.5 concurrency sweep."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
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
from examples.custom_encoder.benchmark.generate_concurrency_workload import (  # noqa: E402
    CONCURRENCIES,
    DECODER_MODEL,
    ENCODER_MODEL,
    REQUESTS_PER_CONCURRENCY,
)
from examples.custom_encoder.benchmark.run_image_sweep import (  # noqa: E402
    AIPERF_EXTRA_ARGS,
    ENCODER_CLASS,
    _command_output,
    _gpu_metadata,
)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _metadata(
    decoder_model: str,
    encoder_model: str,
    concurrencies: tuple[int, ...],
    smoke: bool,
    workload_dir: Path,
) -> dict[str, Any]:
    manifest_path = workload_dir / "workload_manifest.json"
    return {
        "axis": "concurrency",
        "concurrencies": list(concurrencies),
        "runtime": "dynamo-custom-encoder",
        "decoder_model": decoder_model,
        "encoder_model": encoder_model,
        "requests_per_cell": 1 if smoke else REQUESTS_PER_CONCURRENCY,
        "warmup_requests": 1 if smoke else 20,
        "osl": 1 if smoke else 70,
        "streaming": True,
        "custom_encoder_class": ENCODER_CLASS,
        "performance_only_adapter": (
            "the complete native 2048-wide vision projector is computed, then "
            "the first 1536 columns are passed to the decoder; no quality or parity claim"
        ),
        "settings": {
            "preprocess_concurrency": 64,
            "max_batch_cost": 64,
            "graph_buckets": [1, 2, 4, 8, 16, 32, 64],
            "graph_image_sizes": ["500x500"],
            "preprocess_cache_size": 0,
            "queue_wait_ms": 1,
            "max_num_seqs": 64,
            "max_model_len": 2048,
            "vllm_gpu_memory_utilization": 0.4,
        },
        "reachability_note": (
            "one image has cost 1, so closed-loop client concurrency <=32 cannot "
            "produce a batch of 64; bucket 64 is captured but not expected to dispatch"
        ),
        "aiperf_extra_args": AIPERF_EXTRA_ARGS,
        "dynamo_commit": os.environ.get("DYNAMO_BENCHMARK_COMMIT")
        or _command_output(["git", "rev-parse", "HEAD"]),
        "dynamo_branch": os.environ.get("DYNAMO_BENCHMARK_BRANCH")
        or _command_output(["git", "branch", "--show-current"]),
        "container_image": os.environ.get("DYNAMO_BENCHMARK_IMAGE"),
        "vllm_version": _package_version("vllm"),
        "aiperf_version": _package_version("aiperf")
        or _command_output(["aiperf", "--version"]),
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "python_version": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "workload_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "gpu": _gpu_metadata(),
        "host": platform.node(),
    }


def _config(
    decoder_model: str,
    encoder_model: str,
    input_file: Path,
    concurrency: int,
    output_dir: Path,
    smoke: bool,
) -> SweepConfig:
    workflow = REPO_ROOT / "examples/custom_encoder/launch/agg_qwen2_vl_1_5b.sh"
    return SweepConfig(
        model=decoder_model,
        concurrencies=[1 if smoke else concurrency],
        osl=1 if smoke else 70,
        conversation_num=1 if smoke else REQUESTS_PER_CONCURRENCY,
        warmup_count=1 if smoke else 20,
        port=8000,
        timeout=1800,
        input_files=[str(input_file.resolve())],
        configs=[
            BenchmarkConfig(
                label="dynamo-custom-encoder",
                workflow=str(workflow),
            )
        ],
        output_dir=str(output_dir),
        skip_plots=True,
        restart_server_every_benchmark=True,
        env={
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DYN_MAX_MODEL_LEN": "2048",
            "DYN_MAX_NUM_SEQS": "64",
            "DYN_VLLM_GPU_MEMORY_UTILIZATION": "0.4",
            "DYN_QWEN2_VL_ENCODER_MODEL": encoder_model,
            "DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE": "1536",
            "DYN_QWEN2_VL_PREPROCESS_CONCURRENCY": "64",
            "DYN_QWEN2_VL_MAX_BATCH_COST": "64",
            "DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS": "1,2,4,8,16,32,64",
            "DYN_QWEN2_VL_GRAPH_IMAGE_SIZES": "500x500",
            "DYN_QWEN2_VL_PREPROCESS_CACHE_SIZE": "0",
            "DYN_CUSTOM_ENCODER_QUEUE_WAIT_MS": "1",
            "DYN_CUSTOM_ENCODER_TIMING": "1",
        },
        aiperf_extra_args=AIPERF_EXTRA_ARGS,
    )


def run_matrix(
    workload_dir: Path,
    output_dir: Path,
    decoder_model: str,
    encoder_model: str,
    concurrencies: tuple[int, ...],
    smoke: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(
        decoder_model, encoder_model, concurrencies, smoke, workload_dir
    )
    metadata_path = output_dir / "benchmark_metadata.json"
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

    for concurrency in concurrencies:
        input_file = workload_dir / (
            f"image_custom_concurrency{concurrency}_"
            f"{REQUESTS_PER_CONCURRENCY}_isl515.jsonl"
        )
        config = _config(
            decoder_model,
            encoder_model,
            input_file,
            concurrency,
            output_dir,
            smoke,
        )
        config.validate(repo_root=REPO_ROOT)
        run_sweep(config, repo_root=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decoder-model", default=DECODER_MODEL)
    parser.add_argument("--encoder-model", default=ENCODER_MODEL)
    parser.add_argument(
        "--concurrencies", type=int, nargs="+", default=list(CONCURRENCIES)
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_matrix(
        workload_dir=args.workload_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        decoder_model=args.decoder_model,
        encoder_model=args.encoder_model,
        concurrencies=tuple(args.concurrencies),
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
