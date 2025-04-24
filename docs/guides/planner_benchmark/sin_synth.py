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
import math

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    output_data = []

    def get_isl_osl(t):
        isl_osl_ratio = (args.isl_osl_ratio_min + args.isl_osl_ratio_max) / 2 + (
            args.isl_osl_ratio_max - args.isl_osl_ratio_min
        ) / 2 * np.sin(2 * np.pi / args.isl_osl_ratio_period * t)
        logger.info(f"isl_osl_ratio at {t:.2f}: {isl_osl_ratio:.2f}")
        if np.random.uniform(0, 1) < isl_osl_ratio:
            return (args.isl1, args.osl1)
        else:
            return (args.isl2, args.osl2)

    total_hash_ids = np.arange(args.total_blocks)
    for t in tqdm(range(0, args.time_duration, args.process_interval)):
        t_e = min(t + args.process_interval, args.time_duration)
        request_rate = (args.request_rate_min + args.request_rate_max) / 2 + (
            args.request_rate_max - args.request_rate_min
        ) / 2 * np.sin(2 * np.pi / args.request_rate_period * t)
        logger.info(f"request_rate at {t:.2f}: {request_rate:.2f}")
        num_requests = np.random.poisson(request_rate * (t_e - t))
        for req_idx in range(num_requests):
            t_req = t + (t_e - t) * req_idx / num_requests
            isl, osl = get_isl_osl(t_req)
            output_data.append(
                {
                    "timestamp": t_req * 1000,  # in ms
                    "input_length": isl,
                    "output_length": osl,
                    "hash_ids": np.random.choice(
                        total_hash_ids, size=math.ceil(isl / args.block_size)
                    ).tolist(),
                }
            )

    with open(args.output_file, "w") as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset with sinusoidal request rate and isl/osl"
    )
    parser.add_argument(
        "--block-size", type=int, default=512, help="Block size for hashing"
    )
    parser.add_argument(
        "--total-blocks",
        type=int,
        default=10000,
        help="ISL prompt blocks are randomly sampled from this range",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file name (in jsonl format)",
    )
    parser.add_argument(
        "--time-duration",
        type=int,
        default=100,
        help="Time duration of the dataset in seconds",
    )
    parser.add_argument(
        "--process-interval",
        type=int,
        default=1,
        help="Sampling interval used to generate the dataset",
    )

    # request rate parameters
    # for the process interval at [t, t + process_interval), the number of requests to generate is sampled
    # from a poison distribution with the following parameters:
    # request_rate(t) = (min + max) / 2 + (max - min) / 2 * sin(2 * pi / period * t)
    # num_requests[t, t + process_interval) ~ Poisson(request_rate(t) * process_interval)
    # requests are uniformly distributed in the interval [t, t + process_interval)
    parser.add_argument(
        "--request-rate-min",
        type=float,
        default=5,
        help="Minimum request rate in requests per second",
    )
    parser.add_argument(
        "--request-rate-max",
        type=float,
        default=10,
        help="Maximum request rate in requests per second",
    )
    parser.add_argument(
        "--request-rate-period",
        type=float,
        default=10,
        help="Period of the sinusoidal request rate in seconds",
    )

    # isl/osl parameters
    # isl/osl is randomly sampled from two candidates following the isl-osl-ratio.
    # at time t, the isl-osl-ratio is calculated as:
    # isl-osl-ratio(t) = (min + max) / 2 + (max - min) / 2 * sin(2 * pi / period * t)
    # Then, we sample [isl1/osl1, isl2/osl2] from the distribution [isl-osl-ratio(t), 1 - isl-osl-ratio(t)]
    parser.add_argument(
        "--isl1", type=int, default=100, help="Minimum input sequence length"
    )
    parser.add_argument(
        "--osl1", type=int, default=2000, help="Minimum output sequence length"
    )
    parser.add_argument(
        "--isl2", type=int, default=5000, help="Maximum input sequence length"
    )
    parser.add_argument(
        "--osl2", type=int, default=100, help="Maximum output sequence length"
    )
    parser.add_argument(
        "--isl-osl-ratio-min",
        type=float,
        default=0.2,
        help="Minimum ratio of input sequence length to output sequence length",
    )
    parser.add_argument(
        "--isl-osl-ratio-max",
        type=float,
        default=0.8,
        help="Maximum ratio of input sequence length to output sequence length",
    )
    parser.add_argument(
        "--isl-osl-ratio-period",
        type=float,
        default=10,
        help="Period of the sinusoidal input/output sequence length ratio",
    )
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = f"sin_b{args.block_size}_t{args.time_duration}_rr{args.request_rate_min}-{args.request_rate_max}-{args.request_rate_period}_io{args.isl1}{args.osl1}-{args.isl2}{args.osl2}-{args.isl_osl_ratio_min}-{args.isl_osl_ratio_max}-{args.isl_osl_ratio_period}.jsonl"

    main(args)
