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

import argparse
import json
import typing
from typing import Any, get_args, get_origin

from dynamo.planner.utils.planner_config import PlannerConfig


def _pydantic_default(field_name: str) -> Any:
    """Return the static default for a PlannerConfig field, or None for factory defaults."""
    field_info = PlannerConfig.model_fields[field_name]
    default = field_info.default
    try:
        from pydantic_core import PydanticUndefinedType

        if isinstance(default, PydanticUndefinedType):
            return None
    except ImportError:
        pass
    return default


def _unwrap_optional(annotation: Any) -> Any:
    """Unwrap Optional[X] / Union[X, None] to X; return annotation unchanged otherwise."""
    if get_origin(annotation) is typing.Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def add_planner_arguments_to_parser(
    parser: argparse.ArgumentParser, prefix: str = "planner-"
):
    """Add PlannerConfig fields as prefixed CLI args to *parser*.

    Skips any field already present in the parser (shared args such as
    ``backend``, ``itl``, ``ttft``, ``namespace``).  The resulting namespace
    attributes are read back by :func:`build_planner_args_from_namespace` and
    forwarded to the planner as ``--config '<json>'``.
    """
    existing_dests = {action.dest for action in parser._actions}
    dest_prefix = prefix.replace("-", "_")

    planner_group = parser.add_argument_group(
        "planner arguments",
        "Arguments forwarded to the planner service "
        "(only showing args not already in profile_sla)",
    )

    for field_name, field_info in PlannerConfig.model_fields.items():
        if field_name in existing_dests:
            continue

        dest = f"{dest_prefix}{field_name}"
        opt = f"--{prefix}{field_name.replace('_', '-')}"
        default = _pydantic_default(field_name)
        inner = _unwrap_optional(field_info.annotation)

        if inner is bool:
            # Boolean flags: store_true / store_false based on default
            if not default:
                planner_group.add_argument(
                    opt, dest=dest, action="store_true", default=False
                )
            else:
                planner_group.add_argument(
                    opt, dest=dest, action="store_false", default=True
                )
        else:
            # All other types: accept as string; PlannerConfig coerces on
            # model_validate so we don't need strict type parsing here.
            planner_group.add_argument(opt, dest=dest, type=str, default=None)


def build_planner_args_from_namespace(
    args: argparse.Namespace, prefix: str = "planner_"
) -> list[str]:
    """Build a ``--config '<json>'`` argument list for the planner process.

    For each PlannerConfig field the function checks:
    1. A prefixed namespace attribute (``planner_<field>``), for planner-specific
       overrides added by :func:`add_planner_arguments_to_parser`.
    2. A shared namespace attribute (``<field>`` without prefix), for args that
       profile_sla and the planner share (``backend``, ``itl``, ``ttft``, …).

    Only non-``None`` values are included in the config dict, so PlannerConfig
    defaults apply for everything not explicitly set.
    """
    config_dict: dict[str, Any] = {}

    for field_name in PlannerConfig.model_fields:
        value = None

        # Prefixed attr takes priority (planner-specific override)
        prefixed = f"{prefix}{field_name}"
        if hasattr(args, prefixed):
            candidate = getattr(args, prefixed)
            if candidate is not None:
                value = candidate

        # Fall back to shared attr (e.g. backend, ttft, itl, namespace)
        if value is None and hasattr(args, field_name):
            candidate = getattr(args, field_name)
            if candidate is not None:
                value = candidate

        if value is not None:
            config_dict[field_name] = value

    return ["--config", json.dumps(config_dict)]
