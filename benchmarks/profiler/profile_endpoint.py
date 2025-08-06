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

# This script is used to profile the endpoint as prefill worker or decode worker.
# It is used to generate the interpolation results for the SLA script.
# It is not used to generate the SLA results.

import argparse
import json
import logging
import os
import sys
import numpy as np
from pathlib import Path

# Add the utils directory to the path so we can import genai_perf
sys.path.append(str(Path(__file__).parent / "utils"))
from utils.genai_perf import benchmark_prefill, benchmark_decode
from utils.plot import plot_prefill_interpolation, plot_decode_3d_surface

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def generate_prefill_benchmarks(args):
    logger.info("Starting prefill benchmark in interpolation mode...")
        
    # Initialize lists to store results
    prefill_isl = []
    prefill_ttft = []
    prefill_thpt_per_gpu = []
    
    # Run benchmarks for different ISL values
    for isl in range(
        100,
        args.max_context_length,
        (args.max_context_length - 100) // args.prefill_interpolation_granularity,
    ):
        logger.info(f"Running prefill benchmark for ISL={isl}")
        genai_perf_artifact_dir = f"{args.artifact_dir}/gap_isl{isl}"
        
        gap_result = benchmark_prefill(
            isl, genai_perf_artifact_dir, args.model_name, base_url=args.base_url
        )
        
        if gap_result is not None:
            ttft = gap_result["time_to_first_token"]["avg"]
            prefill_isl.append(isl)
            prefill_ttft.append(ttft)
            # Calculate throughput per GPU (tokens per second per GPU)
            # Assuming single GPU for now - this might need adjustment based on your setup
            prefill_thpt_per_gpu.append(isl / ttft * 1000)  # Convert to tokens/s
            logger.info(f"ISL={isl}: TTFT={ttft:.2f}ms, Throughput={isl/ttft*1000:.2f} tokens/s")
        else:
            logger.warning(f"Benchmark failed for ISL={isl}")
    
    # Interpolate and plot results if we have enough data points
    if len(prefill_isl) > 2:
        logger.info("Interpolating prefill TTFT and throughput vs ISL...")
        
        # Convert to numpy arrays for easier manipulation
        prefill_isl_np = np.array(prefill_isl)
        prefill_ttft_np = np.array(prefill_ttft)
        prefill_thpt_per_gpu_np = np.array(prefill_thpt_per_gpu)
        
        # Save raw data
        save_path = f"{args.artifact_dir}/raw_data.npz"
        np.savez(
            save_path,
            prefill_isl=prefill_isl_np,
            prefill_ttft=prefill_ttft_np,
            prefill_thpt_per_gpu=prefill_thpt_per_gpu_np,
        )
        logger.info(f"Raw data saved to {save_path}")
        
        # Call the plotting function
        plot_prefill_interpolation(
            prefill_isl_np, prefill_ttft_np, prefill_thpt_per_gpu_np, args.artifact_dir
        )
        
        # Save results to a JSON file
        results = {
             "interpolation_mode": True,
             "max_context_length": args.max_context_length,
             "prefill_interpolation_granularity": args.prefill_interpolation_granularity,
             "data_points": len(prefill_isl),
             "isl_values": prefill_isl,
             "ttft_values": prefill_ttft,
             "throughput_values": prefill_thpt_per_gpu,
             "raw_data_file": save_path
         }
        result_file = os.path.join(args.artifact_dir, "prefill_interpolation_results.json")
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Interpolation results saved to {result_file}")


def generate_decode_benchmarks(args):
    logger.info("Starting decode benchmark in interpolation mode...")
        
    # Initialize lists to store results
    x_kv_usage = []
    y_context_length = []
    z_itl = []
    z_thpt_per_gpu = []
    
    # Use fixed OSL for interpolation mode
    osl = args.fixed_osl
    
    # Estimate max_kv_tokens if not provided
    max_kv_tokens = args.max_kv_tokens
    if max_kv_tokens is None:
        # Rough estimation - this might need adjustment based on your model
        max_kv_tokens = args.max_context_length * 2  # Conservative estimate
        logger.info(f"Estimated max_kv_tokens: {max_kv_tokens}")
    
    # Run benchmarks for different ISL values
    for isl in range(
        100,
        args.max_context_length - osl,
        (args.max_context_length - osl) // args.decode_interpolation_granularity,
    ):
        max_concurrency = max_kv_tokens // (isl + osl)
        if max_concurrency < 1:
            logger.warning(f"Skipping ISL={isl} as max_concurrency would be {max_concurrency}")
            continue
            
        step_size = max(1, max_concurrency // args.decode_interpolation_granularity)
        sweep_num_request = list(
            range(
                1,
                max_concurrency,
                step_size,
            )
        )
        
        for num_request in sweep_num_request:
            if num_request < 1:
                continue
                
            logger.info(f"Running decode benchmark for ISL={isl}, OSL={osl}, num_request={num_request}")
            genai_perf_artifact_dir = f"{args.artifact_dir}/gap_isl{isl}_osl{osl}_n{num_request}"
            
            gap_result = benchmark_decode(
                isl,
                osl,
                num_request,
                genai_perf_artifact_dir,
                args.model_name,
                base_url=args.base_url,
            )
            
            if gap_result is not None:
                itl = gap_result["inter_token_latency"]["avg"]
                kv_usage = (isl + osl / 2) * num_request / max_kv_tokens
                context_length = isl + osl / 2
                throughput_per_gpu = gap_result["output_token_throughput"]["avg"]  # Assuming single GPU
                
                x_kv_usage.append(kv_usage)
                y_context_length.append(context_length)
                z_itl.append(itl)
                z_thpt_per_gpu.append(throughput_per_gpu)
                
                logger.info(f"ISL={isl}, OSL={osl}, n={num_request}: ITL={itl:.2f}ms, KV_usage={kv_usage:.2f}, Throughput={throughput_per_gpu:.2f} tokens/s")
            else:
                logger.warning(f"Benchmark failed for ISL={isl}, OSL={osl}, num_request={num_request}")
    
    # Interpolate and plot results if we have enough data points
    if len(x_kv_usage) > 2:
        logger.info("Interpolating decode ITL vs KV usage and context length...")
        
        # Save raw data
        save_path = f"{args.artifact_dir}/raw_data.npz"
        np.savez(
            save_path,
            x_kv_usage=np.array(x_kv_usage),
            y_context_length=np.array(y_context_length),
            z_itl=np.array(z_itl),
            z_thpt_per_gpu=np.array(z_thpt_per_gpu),
            max_kv_tokens=np.array([max_kv_tokens]),
        )
        logger.info(f"Raw data saved to {save_path}")
        
        # Call the plotting function (assuming single GPU for now)
        tp_size = 1  # This might need adjustment based on your setup
        plot_decode_3d_surface(
            x_kv_usage, y_context_length, z_itl, tp_size, args.artifact_dir
        )
        
        # Save results to a JSON file
        results = {
            "interpolation_mode": True,
            "max_context_length": args.max_context_length,
            "decode_interpolation_granularity": args.decode_interpolation_granularity,
            "fixed_osl": osl,
            "max_kv_tokens": max_kv_tokens,
            "data_points": len(x_kv_usage),
            "kv_usage_values": x_kv_usage,
            "context_length_values": y_context_length,
            "itl_values": z_itl,
            "throughput_values": z_thpt_per_gpu,
            "raw_data_file": save_path
        }
        result_file = os.path.join(args.artifact_dir, "decode_interpolation_results.json")
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Interpolation results saved to {result_file}")
    else:
        logger.warning("Not enough data points to perform interpolation (need at least 3 points)")
        sys.exit(1)


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
    prefill_parser.add_argument(
        "--max-context-length",
        type=int,
        default=16384,
        help="Maximum context length supported by the served model (default: 16384)"
    )
    prefill_parser.add_argument(
        "--prefill-interpolation-granularity",
        type=int,
        default=16,
        help="Number of samples to benchmark for TTFT interpolation under different ISL (default: 16)"
    )

    # Decode benchmark subcommand
    decode_parser = subparsers.add_parser(
        "benchmark_decode", help="Run decode benchmark"
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
    decode_parser.add_argument(
        "--max-context-length",
        type=int,
        default=16384,
        help="Maximum context length supported by the served model (default: 16384)"
    )
    decode_parser.add_argument(
        "--decode-interpolation-granularity",
        type=int,
        default=6,
        help="Number of samples to benchmark for ITL interpolation (default: 6)"
    )
    decode_parser.add_argument(
        "--fixed-osl",
        type=int,
        default=500,
        help="Fixed output sequence length for interpolation mode (default: 500)"
    )
    decode_parser.add_argument(
        "--max-kv-tokens",
        type=int,
        help="Maximum KV cache tokens (if not provided, will be estimated)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Create artifact directory if it doesn't exist
    os.makedirs(args.artifact_dir, exist_ok=True)

    if args.command == "benchmark_prefill":
        generate_prefill_benchmarks(args)

    elif args.command == "benchmark_decode":
        generate_decode_benchmarks(args)


if __name__ == "__main__":
    main()
