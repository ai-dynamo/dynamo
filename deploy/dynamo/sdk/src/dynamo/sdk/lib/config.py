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
import logging
import os

logger = logging.getLogger(__name__)


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

    def as_args(self, service_name, prefix=""):
        """Extract configs as CLI args for a service, with optional prefix filtering.

        Every component will additionally have the args in the `Common` configs
        applied if it has subscribed to that config key, i.e. the given key is provided in
        the component's `common-configs` setting, and that key has not been overriden by the
        component's config.
        """
        COMMON_CONFIG_SERVICE = "Common"
        COMMON_CONFIG_KEY = "common-configs"

        if service_name not in self:
            return []

        args: list[str] = []

        def add_to_args(args: list[str], key: str, value):
            if prefix and not key.startswith(prefix):
                return

            if key.endswith(COMMON_CONFIG_KEY):
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

        # Get service config excluding ServiceArgs if it exists
        # We never want args to be generated from the ServiceArgs
        service_config = self[service_name].copy()
        if "ServiceArgs" in service_config:
            del service_config["ServiceArgs"]

        if (common := self.get(COMMON_CONFIG_SERVICE)) is not None and (
            common_config_keys := service_config.get(COMMON_CONFIG_KEY)
        ) is not None:
            for key in common_config_keys:
                if key in common and key not in service_config:
                    add_to_args(args, key, common[key])

        for key, value in service_config.items():
            add_to_args(args, key, value)

        logger.info(f"Running {service_name} with {args=}")

        return args
