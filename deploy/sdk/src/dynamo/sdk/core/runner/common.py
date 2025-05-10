#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

import json
import logging
import os
from typing import Any, ClassVar, Dict, Optional

logger = logging.getLogger(__name__)


class ServiceMixin:
    """Mixin for Dynamo services to inject configuration from environment."""

    # Class variable to store service configurations
    _global_service_configs: ClassVar[Dict[str, Dict[str, Any]]] = {}

    def all_services(self) -> Dict[str, Any]:
        """Return all services in the dependency chain."""
        raise NotImplementedError("")

    def inject_config(self) -> None:
        """Inject configuration from environment into service configs.

        This reads from DYNAMO_SERVICE_CONFIG environment variable and merges
        the configuration with any existing service config.
        """
        # Get service configs from environment
        service_config_str = os.environ.get("DYNAMO_SERVICE_CONFIG")
        if not service_config_str:
            logger.debug("No DYNAMO_SERVICE_CONFIG found in environment")
            return

        try:
            service_configs = json.loads(service_config_str)
            logger.debug(f"Loaded service configs: {service_configs}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse DYNAMO_SERVICE_CONFIG: {e}")
            return

        cls = self.__class__
        # Store the entire config at class level
        if not hasattr(cls, "_global_service_configs"):
            setattr(cls, "_global_service_configs", {})
        cls._global_service_configs = service_configs

        # Process ServiceArgs for all services
        all_services = self.all_services()
        logger.debug(f"Processing configs for services: {list(all_services.keys())}")

        for name, svc in all_services.items():
            if name in service_configs:
                svc_config = service_configs[name]
                # Extract ServiceArgs if present
                if "ServiceArgs" in svc_config:
                    logger.debug(
                        f"Found ServiceArgs for {name}: {svc_config['ServiceArgs']}"
                    )
                    if not hasattr(svc, "_service_args"):
                        object.__setattr__(svc, "_service_args", {})
                    svc._service_args = svc_config["ServiceArgs"]
                else:
                    logger.debug(f"No ServiceArgs found for {name}")
                    # Set default config
                    if not hasattr(svc, "_service_args"):
                        object.__setattr__(svc, "_service_args", {"workers": 1})

    def get_service_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get the service configurations for resource allocation.

        Returns:
            Dict mapping service names to their configs
        """
        # Get all services in the dependency chain
        all_services = self.all_services()
        result = {}

        # If we have global configs, use them to build service configs
        cls = self.__class__
        if hasattr(cls, "_global_service_configs"):
            for name, svc in all_services.items():
                # Start with default config
                config = {"workers": 1}

                # If service has specific args, use them
                if hasattr(svc, "_service_args"):
                    config.update(svc._service_args)

                # If there are global configs for this service, get ServiceArgs
                if name in cls._global_service_configs:
                    svc_config = cls._global_service_configs[name]
                    if "ServiceArgs" in svc_config:
                        config.update(svc_config["ServiceArgs"])

                result[name] = config
                logger.debug(f"Built config for {name}: {config}")

        return result

    def _remove_service_args(self, service_name: str):
        """Remove ServiceArgs from the environment config after using them, preserving envs"""
        logger.debug(f"Removing service args for {service_name}")
        config_str = os.environ.get("DYNAMO_SERVICE_CONFIG")
        if config_str:
            config = json.loads(config_str)
            if service_name in config and "ServiceArgs" in config[service_name]:
                # Save envs to separate env var before removing ServiceArgs
                service_args = config[service_name]["ServiceArgs"]
                if "envs" in service_args:
                    service_envs = os.environ.get("DYNAMO_SERVICE_ENVS", "{}")
                    envs_config = json.loads(service_envs)
                    if service_name not in envs_config:
                        envs_config[service_name] = {}
                    envs_config[service_name]["ServiceArgs"] = {
                        "envs": service_args["envs"]
                    }
                    os.environ["DYNAMO_SERVICE_ENVS"] = json.dumps(envs_config)

    def _get_service_args(self, service_name: str) -> Optional[dict]:
        """Get ServiceArgs from environment config if specified"""
        config_str = os.environ.get("DYNAMO_SERVICE_CONFIG")
        if config_str:
            config = json.loads(config_str)
            service_config = config.get(service_name, {})
            return service_config.get("ServiceArgs")
        return None
