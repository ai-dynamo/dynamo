#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking runner for genai-perf")

    parser.add_argument("--tp", "--tensor-parallelism", dest="tp", type=int, default=0)
    parser.add_argument("--dp", "--data-parallelism", dest="dp", type=int, default=0)
    parser.add_argument(
        "--prefill-tp",
        "--prefill-tensor-parallelism",
        dest="prefill_tp",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--prefill-dp",
        "--prefill-data-parallelism",
        dest="prefill_dp",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--decode-tp",
        "--decode-tensor-parallelism",
        dest="decode_tp",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--decode-dp",
        "--decode-data-parallelism",
        dest="decode_dp",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--model", default="neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic"
    )
    parser.add_argument(
        "--isl", "--input-sequence-length", dest="isl", type=int, default=3000
    )
    parser.add_argument(
        "--osl", "--output-sequence-length", dest="osl", type=int, default=150
    )
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument(
        "--mode", choices=["aggregated", "disaggregated"], default="aggregated"
    )
    parser.add_argument(
        "--artifacts-root-dir", dest="artifacts_root_dir", default="artifacts_root"
    )
    parser.add_argument("--deployment-kind", dest="deployment_kind", default="dynamo")

    return parser.parse_args()


def validate_concurrency(concurrency_array):
    for val in concurrency_array:
        if not val.isdigit() or int(val) <= 0:
            sys.exit(
                f"Error: Invalid concurrency value '{val}'. Must be a positive integer."
            )


def create_artifact_dir(root_dir):
    os.makedirs(root_dir, exist_ok=True)
    index = 0
    while (Path(root_dir) / f"artifacts_{index}").exists():
        index += 1

    artifact_dir = Path(root_dir) / f"artifacts_{index}"
    artifact_dir.mkdir(parents=True)

    if index > 0:
        print("--------------------------------")
        print(f"WARNING: Found {index} existing artifacts directories:")
        for i in range(index):
            config_path = Path(root_dir) / f"artifacts_{i}" / "deployment_config.json"
            if config_path.exists():
                print(f"artifacts_{i}:")
                print(config_path.read_text())
                print("--------------------------------")
        print(f"Creating new artifacts directory: artifacts_{index}")
        print("--------------------------------")

    return artifact_dir


def dump_deployment_config(artifact_dir, config):
    config_path = artifact_dir / "deployment_config.json"
    if config_path.exists():
        print("Deployment configuration already exists. Overwriting...")
        config_path.unlink()
    config_path.write_text(json.dumps(config, indent=2))


def main():
    args = parse_args()

    concurrency_array = args.concurrency.split(",")
    validate_concurrency(concurrency_array)

    if args.mode == "aggregated":
        if args.tp == 0 and args.dp == 0:
            sys.exit(
                "--tensor-parallelism and --data-parallelism must be set for aggregated mode."
            )
        print("Starting benchmark with:")
        print(f"  - Tensor Parallelism: {args.tp}")
        print(f"  - Data Parallelism: {args.dp}")
    elif args.mode == "disaggregated":
        if all(
            x == 0
            for x in [args.prefill_tp, args.prefill_dp, args.decode_tp, args.decode_dp]
        ):
            sys.exit(
                "--prefill-tp, --prefill-dp, --decode-tp, --decode-dp must be set for disaggregated mode."
            )
        print("Starting benchmark with:")
        print(f"  - Prefill TP: {args.prefill_tp}")
        print(f"  - Prefill DP: {args.prefill_dp}")
        print(f"  - Decode TP: {args.decode_tp}")
        print(f"  - Decode DP: {args.decode_dp}")
    else:
        sys.exit(f"Unknown mode: {args.mode}")

    print("--------------------------------")
    print("WARNING: This script does not validate tp/dp settings against deployment.")
    print("--------------------------------")

    artifact_dir = create_artifact_dir(args.artifacts_root_dir)

    print("Running genai-perf with:")
    print(f"Model: {args.model}")
    print(f"ISL: {args.isl}")
    print(f"OSL: {args.osl}")
    print(f"Concurrency levels: {concurrency_array}")

    for concurrency in concurrency_array:
        print(f"Run concurrency: {concurrency}")
        cmd = [
            "genai-perf",
            "profile",
            "--model",
            args.model,
            "--tokenizer",
            args.model,
            "--endpoint-type",
            "chat",
            "--endpoint",
            "/v1/chat/completions",
            "--streaming",
            "--url",
            args.url,
            "--synthetic-input-tokens-mean",
            str(args.isl),
            "--synthetic-input-tokens-stddev",
            "0",
            "--output-tokens-mean",
            str(args.osl),
            "--output-tokens-stddev",
            "0",
            "--extra-inputs",
            f"max_tokens:{args.osl}",
            "--extra-inputs",
            f"min_tokens:{args.osl}",
            "--extra-inputs",
            "ignore_eos:true",
            "--extra-inputs",
            '{"nvext":{"ignore_eos":true}}',
            "--concurrency",
            concurrency,
            "--request-count",
            str(int(concurrency) * 10),
            "--warmup-request-count",
            str(int(concurrency) * 2),
            "--num-dataset-entries",
            str(int(concurrency) * 12),
            "--random-seed",
            "100",
            "--artifact-dir",
            str(artifact_dir),
            "--",
            "-v",
            "--max-threads",
            concurrency,
            "-H",
            "Authorization: Bearer NOT USED",
            "-H",
            "Accept: text/event-stream",
        ]
        subprocess.run(cmd, check=True)

    deployment_config = {
        "kind": args.deployment_kind,
        "model": args.model,
        "input_sequence_length": args.isl,
        "output_sequence_length": args.osl,
        "tensor_parallelism": args.tp,
        "data_parallelism": args.dp,
        "prefill_tensor_parallelism": args.prefill_tp,
        "prefill_data_parallelism": args.prefill_dp,
        "decode_tensor_parallelism": args.decode_tp,
        "decode_data_parallelism": args.decode_dp,
        "mode": args.mode,
    }

    dump_deployment_config(artifact_dir, deployment_config)

    print("Benchmarking Successful!!!")


if __name__ == "__main__":
    main()
