#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

CONFIG="${1:-benchmarks/multimodal/sweep/experiments/embedding_cache/vllm_serve.yaml}"
OUTPUT_BASE="${2:-/dynamo-tmp/logs/09-15/qwen35-122b-vllm-ec-h2d-overlap}"
REPETITIONS="${3:-5}"
PYTHON_BIN="${DYN_PYTHON:-python}"
ORDER_SEED="${DYN_BENCHMARK_ORDER_SEED:-42}"

: "${VLLM_SOURCE_REVISION:?VLLM_SOURCE_REVISION must identify the tested vLLM commit}"
: "${VLLM_BASELINE_SOURCE_REVISION:?VLLM_BASELINE_SOURCE_REVISION must identify the baseline vLLM commit}"
: "${VLLM_BASELINE_PYTHONPATH:?VLLM_BASELINE_PYTHONPATH must select the baseline vLLM tree}"
: "${VLLM_PATCHED_PYTHONPATH:?VLLM_PATCHED_PYTHONPATH must select the patched vLLM tree}"
: "${CONTAINER_IMAGE:?CONTAINER_IMAGE must identify the tested runtime image}"
: "${CONTAINER_IMAGE_DIGEST:?CONTAINER_IMAGE_DIGEST must identify the tested image digest}"
: "${CONTAINER_IMAGE_FILE:?CONTAINER_IMAGE_FILE must identify the imported image file}"
: "${HARNESS_REVISION:?HARNESS_REVISION must identify the benchmark harness commit}"

if [[ ! "$REPETITIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPETITIONS must be a positive integer, got: $REPETITIONS" >&2
    exit 2
fi
if [[ ! "$ORDER_SEED" =~ ^[0-9]+$ ]]; then
    echo "DYN_BENCHMARK_ORDER_SEED must be a non-negative integer, got: $ORDER_SEED" >&2
    exit 2
fi

export VLLM_PATCHED_SOURCE_REVISION="${VLLM_PATCHED_SOURCE_REVISION:-$VLLM_SOURCE_REVISION}"

mkdir -p "$OUTPUT_BASE"
"$PYTHON_BIN" - "$OUTPUT_BASE/run_metadata.json" "$CONFIG" "$OUTPUT_BASE" "$REPETITIONS" "$ORDER_SEED" <<'PY'
import datetime
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version

import dynamo._core
import vllm
import yaml

from benchmarks.multimodal.sweep.repetition_plan import randomized_config_orders

output = pathlib.Path(sys.argv[1])
config_path = pathlib.Path(sys.argv[2])
output_base = pathlib.Path(sys.argv[3])
repetitions = int(sys.argv[4])
order_seed = int(sys.argv[5])
config = yaml.safe_load(config_path.read_text())
configs = config["configs"]
if not configs:
    raise ValueError(f"No configs found in {config_path}")
concurrencies = config.get("concurrencies")
sweep_mode = "concurrency" if concurrencies else "request_rate"
sweep_values = concurrencies or config.get("request_rates") or [4, 8, 16, 32, 64]


def command_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


gpu_inventory = command_output(
    [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
)
gpu_topology = command_output(["nvidia-smi", "topo", "-m"])

arm_orders = []
for iteration, ordered_configs in enumerate(
    randomized_config_orders(configs, repetitions, order_seed), start=1
):
    labels = [item["label"] for item in ordered_configs]
    arm_orders.append(labels)
    iteration_config = dict(config)
    iteration_config["configs"] = ordered_configs
    (output_base / f"config-rep-{iteration}.yaml").write_text(
        yaml.safe_dump(iteration_config, sort_keys=False)
    )

metadata = {
    "model": config["model"],
    "arms": [item["label"] for item in configs],
    "arm_orders": arm_orders,
    "arm_order_seed": order_seed,
    "tensor_parallel_sizes": {},
    "ec_cpu_capacity_bytes": {},
    "nvtx": config.get("env", {}).get(
        "DYN_DISABLE_NSYS", os.environ.get("DYN_DISABLE_NSYS", "1")
    )
    != "1",
    "uuid_and_strip": config.get("uuid_and_strip", False),
    "aiperf_version": subprocess.check_output(
        ["aiperf", "--version"], text=True
    ).strip(),
    "vllm_file": vllm.__file__,
    "vllm_version": vllm.__version__,
    "vllm_executable": shutil.which("vllm"),
    "vllm_source_revision": os.environ["VLLM_SOURCE_REVISION"],
    "vllm_source_revisions": {},
    "vllm_pythonpaths": {},
    "python_executable": sys.executable,
    "dynamo_core_file": dynamo._core.__file__,
    "dynamo_version": version("ai-dynamo"),
    "dynamo_runtime_version": version("ai-dynamo-runtime"),
    "container_image": os.environ["CONTAINER_IMAGE"],
    "container_image_digest": os.environ["CONTAINER_IMAGE_DIGEST"],
    "container_image_file": os.environ["CONTAINER_IMAGE_FILE"],
    "harness_revision": os.environ["HARNESS_REVISION"],
    "node": os.environ.get(
        "SLURMD_NODENAME", os.environ.get("SLURM_NODELIST")
    ),
    "platform_machine": platform.machine(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "gpu_inventory": gpu_inventory.splitlines() if gpu_inventory else [],
    "gpu_topology": gpu_topology,
    "datasets": config["input_files"],
    "sweep_mode": sweep_mode,
    "sweep_values": sweep_values,
    "utc_start_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
for arm in configs:
    args = arm.get("extra_args", [])
    label = arm["label"]
    arm_env = {
        key: os.path.expandvars(str(value))
        for key, value in arm.get("env", {}).items()
    }
    if "DYN_VLLM_SOURCE_REVISION" in arm_env:
        metadata["vllm_source_revisions"][label] = arm_env[
            "DYN_VLLM_SOURCE_REVISION"
        ]
    if "PYTHONPATH" in arm_env:
        metadata["vllm_pythonpaths"][label] = arm_env["PYTHONPATH"]
    if "--tensor-parallel-size" in args:
        index = args.index("--tensor-parallel-size")
        metadata["tensor_parallel_sizes"][label] = int(args[index + 1])
    if "--ec-transfer-config" in args:
        index = args.index("--ec-transfer-config")
        ec_config = json.loads(args[index + 1])
        capacity = ec_config.get("ec_connector_extra_config", {}).get("ec_cpu_bytes")
        if capacity is not None:
            metadata["ec_cpu_capacity_bytes"][label] = int(capacity)
output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

arms_raw="$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import pathlib
import sys

import yaml

config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
for item in config["configs"]:
    print(item["label"])
PY
)"
mapfile -t arms <<< "$arms_raw"
if [[ ${#arms[@]} -eq 0 || -z "${arms[0]}" ]]; then
    echo "No benchmark arms found in $CONFIG" >&2
    exit 2
fi

shape_raw="$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import pathlib
import sys

import yaml

from benchmarks.multimodal.sweep.config import input_file_tag
from benchmarks.multimodal.sweep.dataset_shape import count_uuid_expectations

config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
concurrencies = config.get("concurrencies")
sweep_mode = "concurrency" if concurrencies else "request_rate"
sweep_values = concurrencies or config.get("request_rates") or [4, 8, 16, 32, 64]
for dataset in config["input_files"]:
    content, stripped = count_uuid_expectations(
        dataset, conversation_num=config.get("conversation_num")
    )
    for value in sorted(sweep_values):
        print(
            "\t".join(
                (
                    input_file_tag(dataset),
                    f"{sweep_mode}{value}",
                    str(content),
                    str(stripped),
                )
            )
        )
PY
)"
mapfile -t sweep_shapes <<< "$shape_raw"

nsys_output_prefix_base="${DYN_NSYS_OUTPUT_PREFIX:-vllm}"
for ((iteration = 1; iteration <= REPETITIONS; iteration++)); do
    iteration_config="$OUTPUT_BASE/config-rep-$iteration.yaml"
    iteration_order="$("$PYTHON_BIN" - "$iteration_config" <<'PY'
import pathlib
import sys

import yaml

config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
print(" -> ".join(item["label"] for item in config["configs"]))
PY
    )"
    echo "[sweep] ITERATION_ORDER_${iteration}=${iteration_order}"
    echo "[sweep] CUDA_VISIBLE_DEVICES_${iteration}=${CUDA_VISIBLE_DEVICES:-<unset>}"
    export DYN_NSYS_OUTPUT_PREFIX="${nsys_output_prefix_base}-rep-${iteration}"
    "$PYTHON_BIN" -m benchmarks.multimodal.sweep \
        --config "$iteration_config" \
        --output-dir "$OUTPUT_BASE/rep-$iteration" \
        --skip-plots
    echo "[sweep] END_ITER_${iteration}"
    if [[ "$iteration" == "1" ]]; then
        for shape in "${sweep_shapes[@]}"; do
            IFS=$'\t' read -r dataset_tag sweep_tag expected_content expected_stripped \
                <<< "$shape"
            for arm in "${arms[@]}"; do
                artifact="$OUTPUT_BASE/rep-1/$dataset_tag/$arm/$sweep_tag"
                "$PYTHON_BIN" -m benchmarks.multimodal.jsonl.validate_uuid_transport \
                    "$artifact/inputs.json" \
                    --expect-content "$expected_content" \
                    --expect-stripped "$expected_stripped" \
                    --output "$artifact/uuid_transport_summary.json"
            done
        done
        echo "[sweep] UUID_TRANSPORT_VALIDATED"
    fi
done

echo "[sweep] END_ALL_ITERS"
