#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
import subprocess
import threading
import time
from datetime import datetime

import requests

LOGGER = logging.getLogger(__name__)


def wait_for_server(url, model, timeout=300):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "What is the capital of France?",
            }
        ],
        "temperature": 0,
        "top_p": 0.95,
        "max_tokens": 25,
        "min_tokens": 25,
        "stream": True,
        "n": 1,
        "frequency_penalty": 0.0,
        "stop": [],
    }
    start = time.time()
    while True:
        try:
            response = requests.post(url, data=json.dumps(data), headers=headers)
            if response.status_code == 200:
                LOGGER.debug(response.content)
                return True
            else:
                if time.time() - start > timeout:
                    raise ValueError(
                        f"Server is not responding: {response.status_code}"
                    )
                LOGGER.warning(
                    f"Server is not responding: {response.status_code} URL: {url} DATA: {data} HEADERS: {headers}"
                )
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            LOGGER.warning(f"Server is not responding: {e}")
            time.sleep(5)
            if time.time() - start > timeout:
                raise ValueError(f"Server is not responding: {e}")


## Add output threads for stdout and stderr


def run_output_thread(output_file, log_file, print_func):
    """Thread for outputting the process stdout and stderr to a file."""
    if output_file is None:
        LOGGER.debug("No output file specified")
        return
    for line in output_file:
        decoded_line = line.decode()
        print_func(decoded_line)
        log_file.write(decoded_line)


def run_benchmark(args):
    """Waits for the server and then runs the benchmark command."""

    # Wait until server is responding
    wait_for_server(
        args.url + "/v1/chat/completions", args.model, args.benchmark_timeout
    )

    # Create a directory for the test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(
        args.artifact_dir,
        f"{args.load_type}_{args.load_value}_{timestamp}",
    )
    os.makedirs(run_folder, exist_ok=True)

    # Construct the benchmark command
    command = (
        f"genai-perf profile -m {args.model} --endpoint-type chat --streaming "
        f"--num-dataset-entries 1000 --service-kind openai --endpoint v1/chat/completions "
        f"--request-count {args.request_count} --warmup-request-count 10 --random-seed 123 "
        f"--synthetic-input-tokens-stddev 0 --output-tokens-stddev 0 "
        f"--tokenizer {args.tokenizer} --synthetic-input-tokens-mean {args.isl_uncached} "
        f"--output-tokens-mean {args.osl} --extra-inputs seed:100 "
        f"--extra-inputs min_tokens:{args.osl} --extra-inputs max_tokens:{args.osl} "
        f"--profile-export-file my_profile_export.json "
        f"--url {args.url} --artifact-dir {run_folder} "
        f"--num-prefix-prompts 1 --prefix-prompt-length {args.isl_cached} "
    )
    if args.load_type == "rps":
        command += f"--request-rate {args.load_value} "
    elif args.load_type == "concurrency":
        command += f"--concurrency {int(args.load_value)} "
    else:
        raise ValueError(f"Invalid load type: {args.load_type}")

    command += "-- -v --async "

    LOGGER.debug(command)

    # Print information about the run
    LOGGER.info(
        f"ISL cached: {args.isl_cached}, ISL uncached: {args.isl_uncached}, OSL: {args.osl}, "
        f"{args.load_type}: {args.load_value}, request-count: {args.request_count}"
    )
    LOGGER.info(f"Saving artifacts in: {args.artifact_dir}")
    LOGGER.info(f"Command: {command}")
    # Save the command to a file in the artifacts folder
    with open(os.path.join(args.artifact_dir, "run.sh"), "w") as f:
        f.write("#!/bin/bash\n")
        f.write(command)
    subprocess.run(["chmod", "+x", os.path.join(args.artifact_dir, "run.sh")])

    # Prepare output file
    output_file = os.path.join(args.artifact_dir, "output.txt")

    # Run the command and capture both stdout and stderr
    with open(output_file, "w") as f:
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        error_thread = threading.Thread(
            target=run_output_thread, args=(process.stderr, f, LOGGER.error)
        )
        error_thread.start()
        output_thread = threading.Thread(
            target=run_output_thread, args=(process.stdout, f, LOGGER.debug)
        )
        output_thread.start()
        error_thread.join()
        output_thread.join()

    process.wait()


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark for GenAI")
    parser.add_argument(
        "--isl-cached",
        type=int,
        default=0,
        help="Input sequence length (cached)",
    )
    parser.add_argument(
        "--isl-uncached",
        type=int,
        required=True,
        help="Input sequence length (uncached)",
    )
    parser.add_argument(
        "--osl",
        type=int,
        required=True,
        help="Output sequence length",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
        help="Tokenizer name",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL of the API server",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts/",
        help="Directory to save artifacts",
    )
    # Changed to default=None so we can compute if needed
    parser.add_argument(
        "--request-count",
        type=int,
        default=None,
        help="Number of requests to send. If not provided AND load-type=concurrency, it will be computed.",
    )
    parser.add_argument(
        "--load-type",
        type=str,
        required=True,
        help="Type of load: rps or concurrency",
    )
    parser.add_argument(
        "--load-value",
        type=float,
        required=True,
        help="Value of load",
    )
    parser.add_argument(
        "--benchmark-timeout",
        type=int,
        default=300,
        help="Benchmark timeout",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(filename)s: "
        "%(levelname)s: "
        "%(funcName)s(): "
        "%(lineno)d:\t"
        "%(message)s",
    )

    # If --request-count was not provided, compute it if load-type=concurrency
    if args.request_count is None:
        if args.load_type == "concurrency":
            concurrency = int(args.load_value)
            request_count = 10 * concurrency
            # Enforce at least 100
            if request_count < 100:
                request_count = 100
            # If concurrency is over 256, cap at 2×concurrency
            if concurrency > 256 and request_count > 2 * concurrency:
                request_count = 2 * concurrency

            LOGGER.info(
                f"No --request-count specified; computed request-count = {request_count}"
            )
            args.request_count = request_count
        else:
            raise ValueError(
                "No --request-count specified and load-type is 'rps'. "
                "Please specify --request-count explicitly for RPS usage."
            )

    run_benchmark(args)


if __name__ == "__main__":
    main()
