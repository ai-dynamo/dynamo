#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

CONFIG="${1:-benchmarks/multimodal/sweep/experiments/embedding_cache/vllm_serve.yaml}"
OUTPUT_BASE="${2:-/dynamo-tmp/logs/09-14/qwen35-122b-vllm-ec-uuid}"
REPETITIONS="${3:-3}"

if [[ ! "$REPETITIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPETITIONS must be a positive integer, got: $REPETITIONS" >&2
    exit 2
fi

mkdir -p "$OUTPUT_BASE"
python - "$OUTPUT_BASE/run_metadata.json" "$CONFIG" "$OUTPUT_BASE" "$REPETITIONS" <<'PY'
import datetime
import json
import os
import pathlib
import subprocess
import sys
from importlib.metadata import version

import dynamo._core
import vllm
import yaml

output = pathlib.Path(sys.argv[1])
config_path = pathlib.Path(sys.argv[2])
output_base = pathlib.Path(sys.argv[3])
repetitions = int(sys.argv[4])
config = yaml.safe_load(config_path.read_text())
configs = config["configs"]
if not configs:
    raise ValueError(f"No configs found in {config_path}")

rotations = [configs[index:] + configs[:index] for index in range(len(configs))]
balanced_orders = rotations + [list(reversed(order)) for order in rotations]
arm_orders = []
for iteration in range(1, repetitions + 1):
    ordered_configs = balanced_orders[(iteration - 1) % len(balanced_orders)]
    labels = [item["label"] for item in ordered_configs]
    arm_orders.append(labels)
    iteration_config = dict(config)
    iteration_config["configs"] = ordered_configs
    (output_base / f"config-rep-{iteration}.yaml").write_text(
        yaml.safe_dump(iteration_config, sort_keys=False)
    )

metadata = {
    "profile": "122b",
    "model": "Qwen/Qwen3.5-122B-A10B-FP8",
    "preset": "dl-H100x2",
    "tp": 2,
    "arms": ["vllm-serve", "vllm-serve-native-ec", "vllm-serve-dynamo-ec"],
    "arm_orders": arm_orders,
    "ec_capacity_gb": 4,
    "nvtx": False,
    "uuid_and_strip": True,
    "aiperf_version": subprocess.check_output(
        ["aiperf", "--version"], text=True
    ).strip(),
    "vllm_file": vllm.__file__,
    "vllm_version": vllm.__version__,
    "dynamo_core_file": dynamo._core.__file__,
    "dynamo_version": version("ai-dynamo"),
    "dynamo_runtime_version": version("ai-dynamo-runtime"),
    "container_image": os.environ["CONTAINER_IMAGE"],
    "container_image_digest": os.environ["CONTAINER_IMAGE_DIGEST"],
    "container_image_file": os.environ["CONTAINER_IMAGE_FILE"],
    "harness_revision": os.environ["HARNESS_REVISION"],
    "dataset": "/dynamo-tmp/data/30u_8t_5w_8000word_base64_uuid_seed42.jsonl",
    "concurrency": 30,
    "utc_start_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

for ((iteration = 1; iteration <= REPETITIONS; iteration++)); do
    iteration_config="$OUTPUT_BASE/config-rep-$iteration.yaml"
    iteration_order="$(python - "$iteration_config" <<'PY'
import sys

import yaml

config = yaml.safe_load(open(sys.argv[1]))
print(" -> ".join(item["label"] for item in config["configs"]))
PY
)"
    echo "[sweep] ITERATION_ORDER_${iteration}=${iteration_order}"
    python -m benchmarks.multimodal.sweep \
        --config "$iteration_config" \
        --output-dir "$OUTPUT_BASE/rep-$iteration" \
        --skip-plots
    echo "[sweep] END_ITER_${iteration}"
    if [[ "$iteration" == "1" ]]; then
        for arm in vllm-serve vllm-serve-native-ec vllm-serve-dynamo-ec; do
            artifact="$OUTPUT_BASE/rep-1/30u_8t_5w_8000word_base64_uuid_seed42/$arm/concurrency30"
            python -m benchmarks.multimodal.jsonl.validate_uuid_transport \
                "$artifact/inputs.json" \
                --expect-content 360 \
                --expect-stripped 840 \
                --output "$artifact/uuid_transport_summary.json"
        done
        echo "[sweep] UUID_TRANSPORT_VALIDATED"
    fi
done

echo "[sweep] END_ALL_ITERS"
