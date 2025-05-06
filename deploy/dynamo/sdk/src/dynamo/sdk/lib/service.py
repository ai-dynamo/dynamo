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
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union

# WARNING: internal
from _bentoml_sdk import Service, ServiceConfig
from _bentoml_sdk.images import Image
from _bentoml_sdk.service.config import validate
from fastapi import FastAPI

from dynamo.sdk.lib.decorators import AbstractDynamoService, DynamoEndpoint

T = TypeVar("T", bound=object)

logger = logging.getLogger(__name__)


class RuntimeLinkedServices:
    """
    A class to track the linked services in the runtime.
    """

    def __init__(self) -> None:
        self.edges: Dict[DynamoService, Set[DynamoService]] = defaultdict(set)

    def add(self, edge: Tuple[DynamoService, DynamoService]):
        src, dest = edge
        self.edges[src].add(dest.inner)
        # drawing an edge from src to dest.inner

    def remove_unused_edges(self):
        # this method is idempotent
        if not self.edges:
            return
        # remove edges that are not in the current service
        for u, vertices in self.edges.items():
            u.remove_unused_edges(used_edges=vertices)

    # the way 


LinkedServices = RuntimeLinkedServices()


@dataclass
class DynamoConfig:
    """Configuration for Dynamo components"""

    enabled: bool = False
    name: str | None = None
    namespace: str | None = None
    custom_lease: LeaseConfig | None = None


@dataclass
class LeaseConfig:
    """Configuration for custom dynamo leases"""

    ttl: int = 1  # seconds


def _is_dynamo(func) -> bool:
    """True if the function is a DynamoEndpoint instance."""
    return isinstance(func, DynamoEndpoint)


def _get_abstract_dynamo_endpoints(cls: type) -> Set[str]:
    """Get all abstract endpoint names from the class's MRO.

    Args:
        cls: The class to check for abstract endpoints

    Returns:
        Set of abstract endpoint names
    """
    return {
        name
        for base in cls.mro()
        for name, val in base.__dict__.items()
        if getattr(val, "__is_abstract_dynamo__", False)
    }


def _check_dynamo_endpoint_implemented(cls: type, name: str) -> bool:
    """Check if an endpoint is properly implemented.

    Args:
        cls: The class to check
        name: The name of the endpoint to check

    Returns:
        True if the endpoint is properly implemented, False otherwise
    """
    impl = getattr(cls, name, None)
    return impl is not None and _is_dynamo(impl)


def _validate_dynamo_interfaces(cls: type) -> None:
    """
    Validate that *cls* fully implements every @abstract_dynamo_endpoint
    declared in its ancestors and that each implementation is
    decorated with @dynamo_endpoint.
    """
    # TODO: Handle edge cases where there are conflicts in names between the interfaces
    required = _get_abstract_dynamo_endpoints(cls)

    missing, undecorated, not_callable = [], [], []

    for name in required:
        impl = getattr(cls, name, None)  # walk MRO (handles grand-parent impls)
        if impl is None:
            missing.append(name)
            continue

        # guard against attribute shadowing with a non-callable
        if not callable(impl):
            not_callable.append((name, type(impl).__name__))
            continue

        if not _is_dynamo(impl):
            undecorated.append(name)

    problems = []
    if missing:
        problems.append(f"missing implementation(s): {', '.join(missing)}")
    if undecorated:
        problems.append(
            f"method(s) not decorated with @dynamo_endpoint: {', '.join(undecorated)}"
        )
    if not_callable:
        problems.append(
            ", ".join(f"{n} must be callable, got {kind}" for n, kind in not_callable)
        )

    if problems:
        raise TypeError(
            f"{cls.__name__} violates Dynamo interface — " + "; ".join(problems)
        )


class DynamoService(Service[T]):
    """A custom service class that extends BentoML's base Service with Dynamo capabilities"""

    def __init__(
        self,
        config: ServiceConfig,
        inner: type[T],
        image: Optional[Image] = None,
        envs: Optional[list[dict[str, Any]]] = None,
        dynamo_config: Optional[DynamoConfig] = None,
        app: Optional[FastAPI] = None,
    ):
        service_name = inner.__name__
        service_args = self._get_service_args(service_name)
        self.app = app

        if service_args:
            # Validate and merge service args with existing config
            validated_args = validate(service_args)
            config.update(validated_args)
            self._remove_service_args(service_name)

        super().__init__(config=config, inner=inner, image=image, envs=envs or [])

        # Get any dynamo overrides from service_args
        dynamo_overrides = {}
        if service_args and "dynamo" in service_args:
            dynamo_overrides = service_args["dynamo"]
            logger.debug(f"Found dynamo overrides in service_args: {dynamo_overrides}")

        # Initialize Dynamo configuration with overrides
        base_config = (
            dynamo_config
            if dynamo_config
            else DynamoConfig(name=inner.__name__, namespace="default")
        )
        logger.debug(f"Initial base DynamoConfig: {asdict(base_config)}")

        # Apply overrides from service_args to base config
        for key, value in dynamo_overrides.items():
            if hasattr(base_config, key):
                logger.debug(f"Applying override: {key}={value}")
                setattr(base_config, key, value)

        self._dynamo_config = base_config
        if self._dynamo_config.name is None:
            self._dynamo_config.name = inner.__name__

        logger.debug(f"Final DynamoConfig: {asdict(self._dynamo_config)}")

        # Add dynamo configuration to the service config
        # this allows for the config to be part of the service in bento.yaml
        self.config["dynamo"] = asdict(self._dynamo_config)

        # Register Dynamo endpoints
        self._dynamo_endpoints: Dict[str, DynamoEndpoint] = {}
        for field in dir(inner):
            value = getattr(inner, field)
            if isinstance(value, DynamoEndpoint):
                self._dynamo_endpoints[value.name] = value

        self._linked_services: List[DynamoService] = []  # Track linked services

    def _get_service_args(self, service_name: str) -> Optional[dict]:
        """Get ServiceArgs from environment config if specified"""
        config_str = os.environ.get("DYNAMO_SERVICE_CONFIG")
        if config_str:
            config = json.loads(config_str)
            service_config = config.get(service_name, {})
            return service_config.get("ServiceArgs")
        return None

    def is_dynamo_component(self) -> bool:
        """Check if this service is configured as a Dynamo component"""
        return self._dynamo_config.enabled

    def dynamo_address(self) -> Tuple[Optional[str], Optional[str]]:
        """Get the Dynamo address for this component in namespace/name format"""
        if not self.is_dynamo_component():
            raise ValueError("Service is not configured as a Dynamo component")

        # Check if we have a runner map with Dynamo address
        runner_map = os.environ.get("BENTOML_RUNNER_MAP")
        if runner_map:
            try:
                runners = json.loads(runner_map)
                if self.name in runners:
                    address = runners[self.name]
                    if address.startswith("dynamo://"):
                        # Parse dynamo://namespace/name into (namespace, name)
                        _, path = address.split("://", 1)
                        namespace, name = path.split("/", 1)
                        logger.debug(
                            f"Resolved Dynamo address from runner map: {namespace}/{name}"
                        )
                        return (namespace, name)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Failed to parse BENTOML_RUNNER_MAP: {str(e)}") from e

        logger.debug(
            f"Using default Dynamo address: {self._dynamo_config.namespace}/{self._dynamo_config.name}"
        )
        return (self._dynamo_config.namespace, self._dynamo_config.name)

    def get_dynamo_endpoints(self) -> Dict[str, DynamoEndpoint]:
        """Get all registered Dynamo endpoints"""
        return self._dynamo_endpoints

    def get_dynamo_endpoint(self, name: str) -> DynamoEndpoint:
        """Get a specific Dynamo endpoint by name"""
        if name not in self._dynamo_endpoints:
            raise ValueError(f"No Dynamo endpoint found with name: {name}")
        return self._dynamo_endpoints[name]

    def list_dynamo_endpoints(self) -> List[str]:
        """List names of all registered Dynamo endpoints"""
        return list(self._dynamo_endpoints.keys())

    def remove_unused_edges(self, used_edges: Set[DynamoService]):
        """Remove a dependancy from the current service based on the key"""
        current_deps = dict(self.dependencies)
        for dep_key, dep_value in current_deps.items():
            if dep_value.on.inner not in used_edges:
                del self.dependencies[dep_key]

    def link(self, next_service: DynamoService):
        """Link this service to another service, creating a pipeline.

        This method allows linking a concrete service implementation to a service that depends on an interface.
        It will:
        1. Find all interface dependencies in the current service
        2. Check if the next_service implements any of those interfaces
        3. If exactly one match is found, override that dependency
        4. If no matches or multiple matches are found, raise an error

        Args:
            next_service: The concrete service implementation to link

        Raises:
            ValueError: If no matching interface is found or if multiple matches are found
        """
        print(f"\n[DEBUG] Starting link process:")
        print(f"[DEBUG] Current service: {self.name} ({self.inner.__name__})")
        print(f"[DEBUG] Next service: {next_service.name} ({next_service.inner.__name__})")

        # Get all dependencies that may be on interfaces, storing both interface and dep_key
        interface_deps = [
            (dep.on.inner, dep_key, dep)
            for dep_key, dep in self.dependencies.items()
            if issubclass(dep.on.inner, AbstractDynamoService)
        ]

        print(f"[DEBUG] Found {len(interface_deps)} interface dependencies:")
        for interface, dep_key, _ in interface_deps:
            print(f"  - {interface.__name__} (key: {dep_key})")

        if not interface_deps:
            print(f"[DEBUG] ERROR: No AbstractDynamoServices found to override")
            raise ValueError(
                f"No AbstractDynamoServices found to override in {self.name}"
            )

        curr_inner = next_service.inner
        print(f"[DEBUG] Checking if {curr_inner.__name__} implements any required interfaces")

        # Find interfaces that next_service implements
        matching_interfaces = []
        for interface, dep_key, original_dep in interface_deps:
            if issubclass(curr_inner, interface):
                print(f"[DEBUG] Found match: {curr_inner.__name__} implements {interface.__name__}")
                matching_interfaces.append((interface, dep_key, original_dep))

        if not matching_interfaces:
            print(f"[DEBUG] ERROR: No matching interfaces found")
            raise ValueError(
                f"{curr_inner.__name__} does not implement any interfaces required by {self.name}"
            )

        if len(matching_interfaces) > 1:
            interface_names = [interface.__name__ for interface, _, _ in matching_interfaces]
            print(f"[DEBUG] ERROR: Multiple matching interfaces found: {interface_names}")
            raise ValueError(
                f"{curr_inner.__name__} implements multiple interfaces required by {self.name}: {interface_names}"
            )

        # Get the matching interface, dep_key, and original dependency
        _, matching_dep_key, matching_dep = matching_interfaces[0]

        print(f"[DEBUG] Replacing dependency {matching_dep_key} with {next_service.name}")
        # Let's hot swap the on of the existing dependency with the new service
        matching_dep.on = next_service

        # Track the linked service
        self._linked_services.append(next_service)
        LinkedServices.add((self, next_service))
        print(f"[DEBUG] Successfully linked {self.name} to {next_service.name}")

        return next_service

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

        # Store the entire config at class level
        if not hasattr(DynamoService, "_global_service_configs"):
            setattr(DynamoService, "_global_service_configs", {})
        DynamoService._global_service_configs = service_configs

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
        if hasattr(DynamoService, "_global_service_configs"):
            for name, svc in all_services.items():
                # Start with default config
                config = {"workers": 1}

                # If service has specific args, use them
                if hasattr(svc, "_service_args"):
                    config.update(svc._service_args)

                # If there are global configs for this service, get ServiceArgs
                if name in DynamoService._global_service_configs:
                    svc_config = DynamoService._global_service_configs[name]
                    if "ServiceArgs" in svc_config:
                        config.update(svc_config["ServiceArgs"])

                result[name] = config
                logger.debug(f"Built config for {name}: {config}")

        return result

    def is_servable(self) -> bool:
        """Check if this service is ready to be served.

        A service is servable if:
        1. It is not a subclass of AbstractDynamoService (concrete service)
        2. If it is a subclass of AbstractDynamoService, all abstract methods are implemented
           with @dynamo_endpoint decorators

        Returns:
            bool: True if the service is ready to be served, False otherwise
        """
        from dynamo.sdk.lib.decorators import AbstractDynamoService

        # If not a AbstractDynamoService, it's servable
        if not issubclass(self.inner, AbstractDynamoService):
            return True

        # Get all abstract endpoints and check their implementations
        abstract_endpoints = _get_abstract_dynamo_endpoints(self.inner)
        return all(
            _check_dynamo_endpoint_implemented(self.inner, name)
            for name in abstract_endpoints
        )


def service(
    inner: Optional[type[T]] = None,
    /,
    *,
    image: Optional[str] = None,
    envs: Optional[list[dict[str, Any]]] = None,
    dynamo: Optional[Union[Dict[str, Any], DynamoConfig]] = None,
    app: Optional[FastAPI] = None,
    **kwargs: Any,
) -> Any:
    """Enhanced service decorator that supports Dynamo configuration

    Args:
        dynamo: Dynamo configuration, either as a DynamoConfig object or dict with keys:
            - enabled: bool (default True)
            - name: str (default: class name)
            - namespace: str (default: "default")
        **kwargs: Existing BentoML service configuration
    """
    config = kwargs

    # Parse dict into DynamoConfig object
    dynamo_config: Optional[DynamoConfig] = None
    if dynamo is not None:
        if isinstance(dynamo, dict):
            dynamo_config = DynamoConfig(**dynamo)
        else:
            dynamo_config = dynamo

    def decorator(inner: type[T]) -> DynamoService[T]:
        if isinstance(inner, Service):
            raise TypeError("service() decorator can only be applied once")

        # Validate that if the inner class is a subclass of 1 or more AbstractDynamoServices,
        # it implements all the abstract methods declared in the interfaces
        _validate_dynamo_interfaces(inner)

        return DynamoService(
            config=config,
            inner=inner,
            image=image,
            envs=envs or [],
            dynamo_config=dynamo_config,
            app=app,
        )

    return decorator(inner) if inner is not None else decorator
