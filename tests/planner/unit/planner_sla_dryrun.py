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

import logging

from dynamo.planner.utils.planner_argparse import create_sla_planner_parser
from dynamo.planner.utils.planner_core import Planner

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = create_sla_planner_parser()
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

    planner = Planner(None, args, dryrun=True)
    planner.dryrun_run()
