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

import typing as t
from dataclasses import dataclass, field


@dataclass
class Resources:
    """Resources for a service."""

    cpu: str | None = None  # example: "3", "300m"
    memory: str | None = None  # example: "10Gi", "1024Mi"
    gpu: str | None = None  # example: "4"

    def __post_init__(self):
        # Validate and normalize CPU format (e.g., "3", "300m")
        if self.cpu is not None:
            self.cpu = self.cpu.strip()
            if not (
                self.cpu.isdigit() or (self.cpu[:-1].isdigit() and self.cpu[-1] == "m")
            ):
                raise ValueError(
                    f"Invalid CPU format: {self.cpu}. Expected format like '3' or '300m'"
                )

        # Validate and normalize memory format (e.g., "10Gi", "1024Mi")
        if self.memory is not None:
            self.memory = self.memory.strip()
            valid_suffixes = [
                "Ki",
                "Mi",
                "Gi",
                "Ti",
                "Pi",
                "Ei",
                "K",
                "M",
                "G",
                "T",
                "P",
                "E",
            ]
            if not any(
                self.memory.endswith(suffix) and self.memory[: -len(suffix)].isdigit()
                for suffix in valid_suffixes
            ):
                if not self.memory.isdigit():
                    raise ValueError(
                        f"Invalid memory format: {self.memory}. Expected format like '10Gi' or '1024Mi'"
                    )

        # Validate and normalize GPU format (should be a number)
        if self.gpu is not None:
            self.gpu = self.gpu.strip()
            if not self.gpu.isdigit():
                raise ValueError(
                    f"Invalid GPU format: {self.gpu}. Expected a number like '1' or '4'"
                )


@dataclass
class Env:
    """Environment variable."""

    name: str
    value: str = ""


@dataclass
class Service:
    """The entry service of a deployment."""

    service_name: str
    name: str
    namespace: str
    version: str
    path: str
    cmd: t.List[str] = field(default_factory=list)
    resources: Resources | None = None
    envs: t.List[Env] = field(default_factory=list)
    secrets: t.List[str] = field(default_factory=list)
    apis: dict = field(default_factory=dict)
    size_bytes: int = 0
