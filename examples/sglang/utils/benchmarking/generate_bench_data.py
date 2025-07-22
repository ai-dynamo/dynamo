# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import random

import numpy as np
from sglang.bench_serving import sample_random_requests
from transformers import AutoTokenizer, PreTrainedTokenizerBase

"""
Helper script that uses SGLang's random request generator to sample ShareGPT data
and then converts it to a jsonl file that can be used by GenAI perf for benchmarking

Example usage:
python3 generate_bench_data.py --model deepseek-ai/DeepSeek-R1 --output data.jsonl
"""


def main():
    parser = argparse.ArgumentParser(
        description="Use sglang.sample_random_requests to generate token-based JSONL for GenAI-Perf"
    )
    parser.add_argument(
        "--dataset-path", type=str, default="", help="Path or URL to ShareGPT JSON"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output JSONL filename"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier for payloads and tokenizer name",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=8192, help="Total number of samples"
    )
    parser.add_argument(
        "--input-len", type=int, default=4096, help="Target input token length"
    )
    parser.add_argument(
        "--output-len", type=int, default=5, help="Target output token length"
    )
    parser.add_argument(
        "--range-ratio", type=float, default=1.0, help="Sampling length range ratio"
    )
    parser.add_argument(
        "--random-seed", type=int, default=1, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # this is what SGL uses in their benchmarking
    # https://github.com/sgl-project/sglang/blob/b783c1cb829ec451639d1a3ce68380fb7a7be4a3/python/sglang/bench_one_batch_server.py#L131
    # We return text instead of returning raw tokens as GenAI Perf expects text during benchmarking
    samples = sample_random_requests(
        input_len=args.input_len,
        output_len=args.output_len,
        num_prompts=args.num_prompts,
        range_ratio=args.range_ratio,
        tokenizer=tokenizer,
        dataset_path=args.dataset_path,
        random_sample=True,
        return_text=True,
    )

    with open(args.output, "w", encoding="utf-8") as fout:
        for row in samples:
            # genai-perf expects this format
            payload = {
                "text": row.prompt,
                "output_length": row.output_len,
            }
            fout.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
