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

"""Deployment-owned tool catalog shared by the native model, worker and page."""

import secrets

TOOLS = [
    {
        "type": "function",
        "name": "generate_random_number",
        "description": "Generate a random integer between min and max (inclusive).",
        "parameters": {
            "type": "object",
            "properties": {
                "min": {"type": "integer", "description": "Minimum value (inclusive)"},
                "max": {"type": "integer", "description": "Maximum value (inclusive)"},
            },
            "required": ["min", "max"],
            "additionalProperties": False,
        },
    }
]


def generate_random_number(arguments):
    if set(arguments) != {"min", "max"}:
        raise ValueError("Provide exactly min and max")
    low, high = arguments["min"], arguments["max"]
    if (
        type(low) is not int
        or type(high) is not int
        or not (-1000000 <= low <= high <= 1000000)
    ):
        raise ValueError(
            "min and max must be integers between -1000000 and 1000000, with min <= max"
        )
    return {"result": low + secrets.randbelow(high - low + 1)}


def registry():
    return {
        tool["name"]: {
            "function": {key: value for key, value in tool.items() if key != "type"},
            "implementation": "demo_tools." + tool["name"],
        }
        for tool in TOOLS
    }
