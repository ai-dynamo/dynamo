# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Minimal configuration for LLM-powered error classification."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Environment-based configuration for LLM API access."""

    api_key: str
    model: str
    api_base_url: str
    max_rpm: int

    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables."""
        try:
            max_rpm = int(os.getenv("LLM_MAX_RPM", "10"))
        except ValueError:
            max_rpm = 10

        return cls(
            api_key=os.environ.get("LLM_API_KEY", ""),
            model=os.environ.get("LLM_MODEL", "nvidia/llama-3.3-70b-instruct"),
            api_base_url=os.environ.get(
                "LLM_API_BASE_URL", "https://integrate.api.nvidia.com/v1"
            ),
            max_rpm=max_rpm,
        )
