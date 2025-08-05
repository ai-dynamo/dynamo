#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add the utils directory to the path so we can import genai_perf
sys.path.append(str(Path(__file__).parent / "utils"))
from utils.genai_perf import benchmark_prefill, benchmark_decode

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def main():
    parser = argparse.ArgumentParser(
        description="Profile endpoint using genai-perf for prefill and decode benchmarks"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Prefill benchmark subcommand
    prefill_parser = subparsers.add_parser(
        "benchmark_prefill", help="Run prefill benchmark"
    )
    prefill_parser.add_argument(
        "--isl", 
        type=int, 
        required=True, 
        help="Input sequence length (number of input tokens)"
    )
    prefill_parser.add_argument(
        "--artifact-dir", 
        type=str, 
        required=True, 
        help="Directory to store benchmark artifacts"
    )
    prefill_parser.add_argument(
        "--model-name", 
        type=str, 
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name to benchmark (default: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)"
    )
    prefill_parser.add_argument(
        "--base-url", 
        type=str, 
        default="http://localhost:8000",
        help="Base URL for the endpoint (default: http://localhost:8000)"
    )

    # Decode benchmark subcommand
    decode_parser = subparsers.add_parser(
        "benchmark_decode", help="Run decode benchmark"
    )
    decode_parser.add_argument(
        "--isl", 
        type=int, 
        required=True, 
        help="Input sequence length (number of input tokens)"
    )
    decode_parser.add_argument(
        "--osl", 
        type=int, 
        required=True, 
        help="Output sequence length (number of output tokens)"
    )
    decode_parser.add_argument(
        "--num-request", 
        type=int, 
        required=True, 
        help="Number of concurrent requests"
    )
    decode_parser.add_argument(
        "--artifact-dir", 
        type=str, 
        required=True, 
        help="Directory to store benchmark artifacts"
    )
    decode_parser.add_argument(
        "--model-name", 
        type=str, 
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name to benchmark (default: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)"
    )
    decode_parser.add_argument(
        "--base-url", 
        type=str, 
        default="http://localhost:8000",
        help="Base URL for the endpoint (default: http://localhost:8000)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Create artifact directory if it doesn't exist
    os.makedirs(args.artifact_dir, exist_ok=True)

    if args.command == "benchmark_prefill":
        logger.info("Starting prefill benchmark...")
        result = benchmark_prefill(
            isl=args.isl,
            genai_perf_artifact_dir=args.artifact_dir,
            model_name=args.model_name,
            base_url=args.base_url
        )
        
        if result:
            logger.info("Prefill benchmark completed successfully")
            # Save results to a JSON file
            result_file = os.path.join(args.artifact_dir, "prefill_benchmark_results.json")
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {result_file}")
        else:
            logger.error("Prefill benchmark failed")
            sys.exit(1)

    elif args.command == "benchmark_decode":
        logger.info("Starting decode benchmark...")
        result = benchmark_decode(
            isl=args.isl,
            osl=args.osl,
            num_request=args.num_request,
            genai_perf_artifact_dir=args.artifact_dir,
            model_name=args.model_name,
            base_url=args.base_url
        )
        
        if result:
            logger.info("Decode benchmark completed successfully")
            # Save results to a JSON file
            result_file = os.path.join(args.artifact_dir, "decode_benchmark_results.json")
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {result_file}")
        else:
            logger.error("Decode benchmark failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
