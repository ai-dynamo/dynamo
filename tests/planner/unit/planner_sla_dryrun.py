# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging

from dynamo.planner.utils.dryrun import run_sla_planner_dryrun
from dynamo.planner.utils.planner_config import PlannerConfig

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planner Dryrun")
    parser.add_argument(
        "--config",
        required=True,
        help="JSON string or path to a JSON/YAML config file",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the jsonl dataset file"
    )
    parser.add_argument(
        "--start-num-p",
        type=int,
        default=1,
        help="Number of prefill workers to start with",
    )
    parser.add_argument(
        "--start-num-d",
        type=int,
        default=1,
        help="Number of decode workers to start with",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="dryrun_plot.png",
        help="Path to the output plot file",
    )
    args = parser.parse_args()
    config = PlannerConfig.from_config_arg(args.config)

    run_sla_planner_dryrun(
        config,
        dataset=args.dataset,
        start_num_p=args.start_num_p,
        start_num_d=args.start_num_d,
        output_plot=args.output_plot,
    )
