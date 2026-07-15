# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the five-trial Qwen2.5 text engine-client benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
import vllm

from benchmarks.multimodal.sweep.experiments.engine_client.generate_text_workload import (
    sha256,
)
from benchmarks.multimodal.sweep.experiments.engine_client.text_config import (
    TextSweepConfig,
)
from benchmarks.multimodal.sweep.server import ServerManager

REPO_ROOT = Path(__file__).resolve().parents[5]
SMOKE_ORDER = ("vllm-serve", "dynamo-async", "dynamo-sync")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def source_revision() -> tuple[str, str]:
    requested_commit = os.environ.get("DYNAMO_BENCHMARK_COMMIT")
    requested_branch = os.environ.get("DYNAMO_BENCHMARK_BRANCH")
    try:
        git_commit = git_value("rev-parse", "HEAD")
        git_branch = git_value("branch", "--show-current")
        git_status = git_value("status", "--porcelain")
    except (FileNotFoundError, subprocess.CalledProcessError):
        if not requested_commit or not requested_branch:
            raise ValueError(
                "container has no Git metadata; set DYNAMO_BENCHMARK_COMMIT "
                "and DYNAMO_BENCHMARK_BRANCH from the clean source worktree"
            ) from None
        return requested_commit, requested_branch

    if git_status:
        raise ValueError("benchmark source worktree must be clean and committed")
    if requested_commit and requested_commit != git_commit:
        raise ValueError("DYNAMO_BENCHMARK_COMMIT does not match source HEAD")
    if requested_branch and git_branch and requested_branch != git_branch:
        raise ValueError("DYNAMO_BENCHMARK_BRANCH does not match the source branch")
    branch = requested_branch or git_branch
    if not branch:
        raise ValueError("set DYNAMO_BENCHMARK_BRANCH for a detached source revision")
    return requested_commit or git_commit, branch


def command_value(*args: str) -> str:
    return subprocess.check_output(list(args), text=True).strip()


def aiperf_command(
    config: TextSweepConfig,
    dataset_path: Path,
    artifact_dir: Path,
) -> list[str]:
    return [
        "aiperf",
        "profile",
        "--model",
        config.model,
        "--url",
        f"http://127.0.0.1:{config.port}",
        "--endpoint-type",
        "chat",
        "--streaming",
        "--request-count",
        str(config.request_count),
        "--warmup-request-count",
        str(config.warmup_count),
        "--concurrency",
        str(config.concurrency),
        "--osl",
        str(config.osl),
        "--extra-inputs",
        "ignore_eos:true",
        "--extra-inputs",
        "temperature:0.0",
        "--extra-inputs",
        "seed:0",
        "--random-seed",
        "42",
        "--input-file",
        str(dataset_path),
        "--custom-dataset-type",
        "single_turn",
        "--artifact-dir",
        str(artifact_dir),
        "--use-server-token-count",
        "--no-server-metrics",
        "--ui",
        "none",
    ]


def run_aiperf(
    config: TextSweepConfig,
    dataset_path: Path,
    artifact_dir: Path,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command = aiperf_command(config, dataset_path, artifact_dir)
    (artifact_dir / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    with (artifact_dir / "aiperf.log").open("w", encoding="utf-8") as output:
        subprocess.run(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def shared_engine_args(config: TextSweepConfig) -> list[str]:
    args = [
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-seqs",
        str(config.max_num_seqs),
    ]
    if config.prefix_caching:
        args.append("--enable-prefix-caching")
    return args


def benchmark_metadata(
    config: TextSweepConfig,
    dataset_path: Path,
    workload_manifest: dict[str, Any],
    *,
    commit: str,
    branch: str,
    smoke: bool,
) -> dict[str, Any]:
    return {
        "model": config.model,
        "commit": commit,
        "branch": branch,
        "container_image": os.environ.get("DYNAMO_BENCHMARK_IMAGE", "unknown"),
        "physical_gpu": os.environ.get("DYNAMO_BENCHMARK_PHYSICAL_GPU", "unknown"),
        "gpu_identity": command_value(
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ),
        "host": platform.node(),
        "versions": {
            "aiperf": command_value("aiperf", "--version"),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
        },
        "config": str(config.source_path),
        "config_sha256": config.source_sha256,
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256(dataset_path),
        "workload_manifest": workload_manifest,
        "smoke": smoke,
        "concurrency": config.concurrency,
        "request_count": config.request_count,
        "warmup_count": config.warmup_count,
        "target_isl": config.target_isl,
        "osl": config.osl,
        "repeats": config.repeats,
        "trial_order": config.trial_order,
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.max_num_seqs,
        "kv_cache_memory_bytes": config.kv_cache_memory_bytes,
        "prefix_caching": config.prefix_caching,
        "created_at": utc_now(),
    }


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    normalized = json.loads(json.dumps(metadata))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_keys = set(metadata) - {"created_at"}
        if any(existing.get(key) != normalized.get(key) for key in comparable_keys):
            raise ValueError("refusing to resume with different benchmark provenance")
        return
    path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")


def run_sweep(
    config: TextSweepConfig,
    dataset_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    smoke: bool = False,
) -> None:
    commit, branch = source_revision()
    workload_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if workload_manifest["model"] != config.model:
        raise ValueError("workload model does not match benchmark model")
    if sha256(dataset_path) != workload_manifest["dataset_sha256"]:
        raise ValueError("workload hash does not match its manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        config = replace(
            config,
            request_count=3,
            warmup_count=1,
            repeats=1,
            trial_order=(SMOKE_ORDER,),
        )
    metadata = benchmark_metadata(
        config,
        dataset_path,
        workload_manifest,
        commit=commit,
        branch=branch,
        smoke=smoke,
    )
    if "0.8.0" not in metadata["versions"]["aiperf"]:
        raise ValueError("text engine-client benchmark requires AIPerf 0.8.0")
    write_metadata(output_dir / "benchmark_metadata.json", metadata)

    environment = {
        "DYN_HTTP_PORT": str(config.port),
        "DYN_FRONTEND_SYSTEM_PORT": str(config.port + 1),
        "DYN_WORKER_SYSTEM_PORT": str(config.port + 2),
        "MAX_MODEL_LEN": str(config.max_model_len),
        "MAX_NUM_SEQS": str(config.max_num_seqs),
        "_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES": str(config.kv_cache_memory_bytes),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    }
    server = ServerManager(port=config.port, timeout=config.timeout)

    try:
        for trial_index, order in enumerate(config.trial_order, start=1):
            for order_index, runtime_label in enumerate(order, start=1):
                runtime = config.runtimes[runtime_label]
                run_dir = output_dir / f"trial-{trial_index:02d}" / runtime_label
                artifact_dir = run_dir / f"concurrency{config.concurrency}"
                result_path = artifact_dir / "profile_export_aiperf.json"
                if result_path.exists():
                    print(f"SKIP trial={trial_index} runtime={runtime_label}")
                    continue
                if run_dir.exists():
                    raise RuntimeError(
                        f"incomplete run directory must be removed before retry: {run_dir}"
                    )

                run_dir.mkdir(parents=True)
                workflow = (REPO_ROOT / runtime.workflow).resolve()
                extra_args = [*shared_engine_args(config), *runtime.extra_args]
                server_command = [
                    "bash",
                    str(workflow),
                    "--model",
                    config.model,
                    *extra_args,
                ]
                (run_dir / "server_command.txt").write_text(
                    shlex.join(server_command) + "\n", encoding="utf-8"
                )
                run_metadata = {
                    "trial": trial_index,
                    "order_index": order_index,
                    "runtime": runtime_label,
                    "started_at": utc_now(),
                    "config_sha256": config.source_sha256,
                    "dataset_sha256": sha256(dataset_path),
                }
                (run_dir / "run_metadata.json").write_text(
                    json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
                )

                print(
                    f"START trial={trial_index} order={order_index} "
                    f"runtime={runtime_label}",
                    flush=True,
                )
                server.start(
                    workflow_script=str(workflow),
                    model=config.model,
                    extra_args=extra_args,
                    env_overrides=environment,
                )
                try:
                    run_aiperf(config, dataset_path, artifact_dir)
                finally:
                    server.stop()
                print(f"DONE trial={trial_index} runtime={runtime_label}", flush=True)
    finally:
        if server.is_running:
            server.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run three measured requests plus one warmup for every runtime",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sweep(
        config=TextSweepConfig.load(args.config.resolve()),
        dataset_path=args.dataset.resolve(),
        manifest_path=args.manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
