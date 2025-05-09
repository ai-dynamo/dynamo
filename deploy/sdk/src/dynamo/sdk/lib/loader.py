#  SPDX-FileCopyrightText: Copyright (c) 2020 Atalaya Tech. Inc
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

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Optional, TypeVar

from dynamo.sdk.lib.service import DynamoService

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=object)


def find_and_load_service(
    import_str: str,
    working_dir: Optional[str] = None,
) -> DynamoService:
    """Load a DynamoService instance from source code by providing an import string.

    Args:
        import_str: String in format "module[:attribute]" or "path/to/file.py[:attribute]"
            Examples:
                "graphs:disagg:Frontend"
                "fraud_detector:svc"
                "./path/to/service.py:MyService"
                "fraud_detector"  # Will find the root service if only one exists
        working_dir: Optional directory to use as base for imports. Defaults to cwd.

    Returns:
        The loaded DynamoService instance

    Raises:
        ImportError: If module cannot be imported
        ValueError: If service cannot be found or multiple root services exist
    """
    logger.info(f"Loading service from import string: {import_str}")
    logger.info(f"Working directory: {working_dir or os.getcwd()}")

    sys_path_modified = False
    prev_cwd = None

    if working_dir is not None:
        prev_cwd = os.getcwd()
        working_dir = os.path.realpath(os.path.expanduser(working_dir))
        logger.info(f"Changing working directory to: {working_dir}")
        os.chdir(working_dir)
    else:
        working_dir = os.getcwd()

    if working_dir not in sys.path:
        logger.info(f"Adding {working_dir} to sys.path")
        sys.path.insert(0, working_dir)
        sys_path_modified = True

    try:
        return _do_import(import_str, working_dir)
    finally:
        if sys_path_modified and working_dir:
            logger.info(f"Removing {working_dir} from sys.path")
            sys.path.remove(working_dir)
        if prev_cwd is not None:
            logger.info(f"Restoring working directory to: {prev_cwd}")
            os.chdir(prev_cwd)


def _do_import(import_str: str, working_dir: str) -> DynamoService:
    """Internal function to handle the actual import logic"""
    import_path, _, attrs_str = import_str.partition(":")
    logger.info(f"Parsed import string - path: {import_path}, attributes: {attrs_str}")

    if not import_path:
        raise ValueError(
            f'Invalid import string "{import_str}", must be in format '
            '"<module>:<attribute>" or "<module>"'
        )

    # Handle file path vs module name imports
    if os.path.isfile(import_path):
        logger.info(f"Importing from file path: {import_path}")
        import_path = os.path.realpath(import_path)
        if not import_path.startswith(working_dir):
            raise ImportError(
                f'Module "{import_path}" not found in working directory "{working_dir}"'
            )

        file_name, ext = os.path.splitext(import_path)
        if ext != ".py":
            raise ImportError(
                f'Invalid module extension "{ext}", only ".py" files are supported'
            )

        # Build module name from path components
        module_parts = []
        path = file_name
        while True:
            path, name = os.path.split(path)
            module_parts.append(name)
            if (
                not os.path.exists(os.path.join(path, "__init__.py"))
                or path == working_dir
            ):
                break
        module_name = ".".join(module_parts[::-1])
        logger.info(f"Constructed module name from path: {module_name}")
    else:
        logger.info(f"Importing from module name: {import_path}")
        module_name = import_path

    try:
        logger.info(f"Attempting to import module: {module_name}")
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f'Failed to import module "{module_name}": {e}')

    # If no specific attribute given, find the root service
    if not attrs_str:
        logger.info("No attributes specified, searching for root service")
        services = [
            (name, obj)
            for name, obj in module.__dict__.items()
            if isinstance(obj, DynamoService)
        ]
        logger.info(f"Found {len(services)} DynamoService instances")

        if not services:
            raise ValueError(
                f"No DynamoService instances found in module '{module_name}'"
            )

        # Find root services (those that aren't dependencies of other services)
        dependents = set()
        for _, svc in services:
            for dep in svc.dependencies.values():
                if dep.on is not None:
                    dependents.add(dep.on)

        root_services = [(n, s) for n, s in services if s not in dependents]
        logger.info(f"Found {len(root_services)} root services")

        if not root_services:
            raise ValueError(
                f"No root DynamoService found in module '{module_name}'. "
                "All services are dependencies of other services."
            )
        if len(root_services) > 1:
            names = [n for n, _ in root_services]
            raise ValueError(
                f"Multiple root services found in module '{module_name}': {names}. "
                "Please specify which service to use with '<module>:<service_name>'"
            )

        _, instance = root_services[0]
        logger.info(f"Selected root service: {instance}")
    else:
        # Navigate through dot-separated attributes
        logger.info(f"Navigating attributes: {attrs_str}")
        instance = module
        for attr in attrs_str.split("."):
            try:
                if isinstance(instance, DynamoService):
                    logger.info(f"Following dependency link: {attr}")
                    instance = instance.dependencies[attr].on
                else:
                    logger.info(f"Getting attribute: {attr}")
                    instance = getattr(instance, attr)
            except (AttributeError, KeyError):
                raise ValueError(f'Attribute "{attr}" not found in "{module_name}"')

    # Set import string for debugging/logging
    if not hasattr(instance, "_import_str"):
        import_str_val = f"{module_name}:{attrs_str}" if attrs_str else module_name
        logger.info(f"Setting _import_str to: {import_str_val}")
        object.__setattr__(instance, "_import_str", import_str_val)

    return instance
