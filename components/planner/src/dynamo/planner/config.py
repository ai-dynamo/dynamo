# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os

logger = logging.getLogger(__name__)


class ServiceConfig(dict):
    """Configuration store that inherits from dict for simpler access patterns"""

    _instance = None
    COMMON_CONFIG_SERVICE = "Common"
    COMMON_CONFIG_KEY = "common-configs"

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

    @classmethod
    def get_parsed_config(cls, service_name):
        """Get parsed config for a service with common configs applied, returned as dict"""
        instance = cls.get_instance()

        if service_name not in instance:
            return {}

        # Get service config excluding ServiceArgs if it exists
        service_config = instance[service_name].copy()
        if "ServiceArgs" in service_config:
            del service_config["ServiceArgs"]

        # Apply common configs if they exist
        if (common := instance.get(cls.COMMON_CONFIG_SERVICE)) is not None and (
            common_config_keys := service_config.get(cls.COMMON_CONFIG_KEY)
        ) is not None:
            for key in common_config_keys:
                if key in common and key not in service_config:
                    service_config[key] = common[key]

        # Remove the common-configs key itself from the final config
        if cls.COMMON_CONFIG_KEY in service_config:
            del service_config[cls.COMMON_CONFIG_KEY]

        return service_config

    def as_args(self, service_name, prefix=""):
        """Extract configs as CLI args for a service, with optional prefix filtering.

        Every component will additionally have the args in the `Common` configs
        applied if it has subscribed to that config key, i.e. the given key is provided in
        the component's `common-configs` setting, and that key has not been overriden by the
        component's config.
        """

        if service_name not in self:
            return []

        args: list[str] = []

        def add_to_args(args: list[str], key: str, value):
            if prefix and not key.startswith(prefix):
                return

            if key.endswith(self.COMMON_CONFIG_KEY):
                return

            # Strip prefix if needed
            arg_key = key[len(prefix) :] if prefix and key.startswith(prefix) else key

            # vLLM arguments that default to True and need explicit false handling
            # Based on https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/config.py
            vllm_true_defaults = {
                "enable_prefix_caching": "no-enable-prefix-caching",
                "use_tqdm_on_load": "no-use-tqdm-on-load",
                "multi_step_stream_outputs": "no-multi-step-stream-outputs",
            }

            # Normalize key for comparison (replace hyphens with underscores)
            normalized_key = arg_key.replace("-", "_")

            # Convert to CLI format
            if isinstance(value, bool):
                if value:
                    # Always output true values as flags
                    args.append(f"--{arg_key}")
                else:
                    # For false values, check if this is a vLLM arg that defaults to True
                    if normalized_key in vllm_true_defaults:
                        # Use negative flag if available
                        args.append(f"--{vllm_true_defaults[normalized_key]}")
                    # For other false values, omit entirely (standard action="store_true" behavior)
            elif isinstance(value, dict):
                args.extend([f"--{arg_key}", json.dumps(value)])
            else:
                args.extend([f"--{arg_key}", str(value)])

        # Get service config excluding ServiceArgs if it exists
        # We never want args to be generated from the ServiceArgs
        service_config = self[service_name].copy()
        if "ServiceArgs" in service_config:
            del service_config["ServiceArgs"]

        if (common := self.get(self.COMMON_CONFIG_SERVICE)) is not None and (
            common_config_keys := service_config.get(self.COMMON_CONFIG_KEY)
        ) is not None:
            for key in common_config_keys:
                if key in common and key not in service_config:
                    add_to_args(args, key, common[key])

        for key, value in service_config.items():
            add_to_args(args, key, value)

        logger.info(f"Running {service_name} with {args=}")

        return args
