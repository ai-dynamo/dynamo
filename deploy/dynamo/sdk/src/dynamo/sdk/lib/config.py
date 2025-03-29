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

import json
import os
from typing import Optional


class ServiceConfig(dict):
    """Configuration store that inherits from dict for simpler access patterns"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls._load_from_env()
        return cls._instance

    @classmethod
    def _load_from_env(cls):
        """Load config from environment variable"""
        configs = {}
        env_config = os.environ.get("DYNAMO_SERVICE_CONFIG")
        if env_config:
            try:
                configs = json.loads(env_config)
            except json.JSONDecodeError:
                print("Failed to parse DYNAMO_SERVICE_CONFIG")
        return cls(configs)  # Initialize dict subclass with configs

    def require(self, service_name, key):
        """Require a config value, raising error if not found"""
        if service_name not in self or key not in self[service_name]:
            raise ValueError(f"{service_name}.{key} must be specified in configuration")
        return self[service_name][key]

    def as_args(
        self, service_name, prefix="", common_configs: Optional[set[str]] = None
    ):
        """Extract configs as CLI args for a service, with optional prefix filtering.
        Every service will additionally have the args in the `Common` service name
        uniformly applied.
        """
        if service_name not in self:
            return []

        args = []

        def add_to_args(args: list[str], value):
            if prefix and not key.startswith(prefix):
                return

            # Strip prefix if needed
            arg_key = key[len(prefix) :] if prefix and key.startswith(prefix) else key

            # Convert to CLI format
            if isinstance(value, bool):
                if value:
                    args.append(f"--{arg_key}")
            elif isinstance(value, dict):
                args.extend([f"--{arg_key}", json.dumps(value)])
            else:
                args.extend([f"--{arg_key}", str(value)])

        if common_configs is not None and (common := self.get("Common")) is not None:
            for key in common_configs:
                if key in common and key not in self[service_name]:
                    add_to_args(args, common[key])

        for key, value in self[service_name].items():
            add_to_args(args, value)

        return args
