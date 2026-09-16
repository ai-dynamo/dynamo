#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

CONFIG="${1:-benchmarks/multimodal/sweep/experiments/embedding_cache/vllm_serve.yaml}"
OUTPUT_BASE="${2:-/dynamo-tmp/logs/09-15/qwen35-122b-vllm-ec-h2d-overlap}"
REPETITIONS="${3:-4}"
PYTHON_BIN="${DYN_PYTHON:-python}"

: "${VLLM_SOURCE_REVISION:?VLLM_SOURCE_REVISION must identify the tested vLLM commit}"

if [[ ! "$REPETITIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPETITIONS must be a positive integer, got: $REPETITIONS" >&2
    exit 2
fi

mkdir -p "$OUTPUT_BASE"
"$PYTHON_BIN" - "$OUTPUT_BASE/run_metadata.json" "$CONFIG" "$OUTPUT_BASE" "$REPETITIONS" <<'PY'
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
from importlib.metadata import version

import dynamo._core
import vllm
import yaml

from benchmarks.multimodal.sweep.repetition_plan import balanced_config_orders

output = pathlib.Path(sys.argv[1])
config_path = pathlib.Path(sys.argv[2])
output_base = pathlib.Path(sys.argv[3])
repetitions = int(sys.argv[4])
config = yaml.safe_load(config_path.read_text())
configs = config["configs"]
if not configs:
    raise ValueError(f"No configs found in {config_path}")

arm_orders = []
for iteration, ordered_configs in enumerate(
    balanced_config_orders(configs, repetitions), start=1
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
    "tensor_parallel_sizes": {},
    "ec_cpu_capacity_bytes": {},
    "nvtx": config.get("env", {}).get("DYN_DISABLE_NSYS", "1") != "1",
    "uuid_and_strip": config.get("uuid_and_strip", False),
    "aiperf_version": subprocess.check_output(
        ["aiperf", "--version"], text=True
    ).strip(),
    "vllm_file": vllm.__file__,
    "vllm_version": vllm.__version__,
    "vllm_executable": shutil.which("vllm"),
    "vllm_source_revision": os.environ["VLLM_SOURCE_REVISION"],
    "python_executable": sys.executable,
    "dynamo_core_file": dynamo._core.__file__,
    "dynamo_version": version("ai-dynamo"),
    "dynamo_runtime_version": version("ai-dynamo-runtime"),
    "container_image": os.environ["CONTAINER_IMAGE"],
    "container_image_digest": os.environ["CONTAINER_IMAGE_DIGEST"],
    "container_image_file": os.environ["CONTAINER_IMAGE_FILE"],
    "harness_revision": os.environ["HARNESS_REVISION"],
    "dataset": config["input_files"][0],
    "concurrency": config["concurrencies"][0],
    "utc_start_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
for arm in configs:
    args = arm.get("extra_args", [])
    label = arm["label"]
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
import json
import pathlib
import sys
from collections import defaultdict

import yaml

config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
dataset = pathlib.Path(config["input_files"][0])
seen = defaultdict(set)
included_sessions = set()
conversation_limit = config.get("conversation_num")
content = 0
stripped = 0
for line_index, line in enumerate(dataset.open()):
    item = json.loads(line)
    session_id = item.get("session_id", f"row-{line_index}")
    if session_id not in included_sessions:
        if (
            conversation_limit is not None
            and len(included_sessions) >= conversation_limit
        ):
            continue
        included_sessions.add(session_id)
    session_seen = seen[session_id]
    for image_uuid in item.get("image_uuids", []):
        if image_uuid in session_seen:
            stripped += 1
        else:
            session_seen.add(image_uuid)
            content += 1

print(dataset.stem.replace(" ", "_"))
print(f"concurrency{config['concurrencies'][0]}")
print(content)
print(stripped)
PY
)"
mapfile -t sweep_shape <<< "$shape_raw"
dataset_tag="${sweep_shape[0]}"
sweep_tag="${sweep_shape[1]}"
expected_content="${sweep_shape[2]}"
expected_stripped="${sweep_shape[3]}"

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
    "$PYTHON_BIN" -m benchmarks.multimodal.sweep \
        --config "$iteration_config" \
        --output-dir "$OUTPUT_BASE/rep-$iteration" \
        --skip-plots
    echo "[sweep] END_ITER_${iteration}"
    if [[ "$iteration" == "1" ]]; then
        for arm in "${arms[@]}"; do
            artifact="$OUTPUT_BASE/rep-1/$dataset_tag/$arm/$sweep_tag"
            "$PYTHON_BIN" -m benchmarks.multimodal.jsonl.validate_uuid_transport \
                "$artifact/inputs.json" \
                --expect-content "$expected_content" \
                --expect-stripped "$expected_stripped" \
                --output "$artifact/uuid_transport_summary.json"
        done
        echo "[sweep] UUID_TRANSPORT_VALIDATED"
    fi
done

echo "[sweep] END_ALL_ITERS"
