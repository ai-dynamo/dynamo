# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import subprocess


def gpu_product_name():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu",
                "name",
                "--format",
                "csv",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        result_values = [
            x.replace(", ", ",").split(",") for x in result.stdout.split("\n") if x
        ]
        if result_values[0][0] == "No devices were found":
            return None
        return result_values[1][0].strip().replace(" ", "_")
    except FileNotFoundError:
        return None


def gpu_count():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"], capture_output=True, text=True
        )

        return len(result.stdout.strip().split("\n"))
    except FileNotFoundError:
        return 0
