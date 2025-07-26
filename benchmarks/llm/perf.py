#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking CLI")

    parser.add_argument("--tp", "--tensor-parallelism", type=int, default=0, dest="tp")
    parser.add_argument("--dp", "--data-parallelism", type=int, default=0, dest="dp")
    parser.add_argument(
        "--prefill-tp",
        "--prefill-tensor-parallelism",
        type=int,
        default=0,
        dest="prefill_tp",
    )
    parser.add_argument(
        "--prefill-dp",
        "--prefill-data-parallelism",
        type=int,
        default=0,
        dest="prefill_dp",
    )
    parser.add_argument(
        "--decode-tp",
        "--decode-tensor-parallelism",
        type=int,
        default=0,
        dest="decode_tp",
    )
    parser.add_argument(
        "--decode-dp",
        "--decode-data-parallelism",
        type=int,
        default=0,
        dest="decode_dp",
    )

    parser.add_argument(
        "--model", default="neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic"
    )
    parser.add_argument(
        "--isl", "--input-sequence-length", type=int, default=3000, dest="isl"
    )
    parser.add_argument(
        "--osl", "--output-sequence-length", type=int, default=150, dest="osl"
    )
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument(
        "--mode", choices=["aggregated", "disaggregated"], default="aggregated"
    )
    parser.add_argument("--artifacts-root-dir", default="artifacts_root")
    parser.add_argument("--deployment-kind", default="dynamo")

    return parser.parse_args()


def validate_concurrency(concurrency_array):
    for val in concurrency_array:
        if not val.isdigit() or int(val) <= 0:
            raise ValueError(f"Invalid concurrency value: {val}")


def main():
    args = parse_args()
    concurrency_array = args.concurrency.split(",")
    validate_concurrency(concurrency_array)

    if args.mode == "aggregated":
        if args.tp == 0 and args.dp == 0:
            raise SystemExit("--tp and --dp must be set for aggregated mode.")
        print("Starting aggregated benchmark:")
        print(f"  - Tensor Parallelism: {args.tp}")
        print(f"  - Data Parallelism: {args.dp}")
    elif args.mode == "disaggregated":
        if all(
            x == 0
            for x in [args.prefill_tp, args.prefill_dp, args.decode_tp, args.decode_dp]
        ):
            raise SystemExit(
                "--prefill/decode TP and DP must be set for disaggregated mode."
            )
        print("Starting disaggregated benchmark:")
        print(f"  - Prefill TP: {args.prefill_tp}, DP: {args.prefill_dp}")
        print(f"  - Decode TP: {args.decode_tp}, DP: {args.decode_dp}")
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")

    print("WARNING: User must ensure TP/DP config matches deployment.")

    # Create artifact directory
    root = Path(args.artifacts_root_dir)
    root.mkdir(exist_ok=True)
    index = 0
    while (root / f"artifacts_{index}").exists():
        index += 1
    artifact_dir = root / f"artifacts_{index}"
    artifact_dir.mkdir()

    if index > 0:
        print(f"WARNING: Found {index} existing artifacts directories.")
        for i in range(index):
            path = root / f"artifacts_{i}/deployment_config.json"
            if path.exists():
                print(f"artifacts_{i}:")
                print(path.read_text())
                print("-" * 32)
        print(f"Creating new artifacts directory: artifacts_{index}")
        print("-" * 32)

    print(f"Model: {args.model}")
    print(f"ISL: {args.isl}, OSL: {args.osl}")
    print(f"Concurrency: {concurrency_array}")

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

    config = {
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

    config_path = artifact_dir / "deployment_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print("Benchmarking Successful!")


if __name__ == "__main__":
    main()
