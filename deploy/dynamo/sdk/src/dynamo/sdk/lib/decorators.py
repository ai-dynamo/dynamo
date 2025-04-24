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

import typing as t
from functools import wraps
from typing import Any, get_type_hints

import bentoml
from pydantic import BaseModel


class DynamoEndpoint:
    """Decorator class for Dynamo endpoints"""

    def __init__(self, func: t.Callable, name: str | None = None):
        self.func = func
        self.name = name or func.__name__
        self.is_dynamo_endpoint = True

        # Extract request type from hints
        hints = get_type_hints(func)
        args = list(hints.items())

        # Skip self/cls argument
        if args[0][0] in ("self", "cls"):
            args = args[1:]

        # Get request type from first arg
        self.request_type = args[0][1]
        wraps(func)(self)

    async def __call__(self, *args: t.Any, **kwargs: t.Any) -> Any:
        # Validate request
        if len(args) > 1 and issubclass(self.request_type, BaseModel):
            args = list(args)  # type: ignore
            if isinstance(args[1], (str, dict)):
                args[1] = self.request_type.parse_obj(args[1])  # type: ignore

        # Convert Pydantic model to dict before passing to dynamo
        if len(args) > 1 and isinstance(args[1], BaseModel):
            args = list(args)  # type: ignore
            args[1] = args[1].model_dump()  # type: ignore

        return await self.func(*args, **kwargs)


def dynamo_endpoint(
    name: str | None = None,
) -> t.Callable[[t.Callable], DynamoEndpoint]:
    """Decorator for Dynamo endpoints.

    Args:
        name: Optional name for the endpoint. Defaults to function name.

    Example:
        @dynamo_endpoint()
        def my_endpoint(self, input: str) -> str:
            return input

        @dynamo_endpoint(name="custom_name")
        def another_endpoint(self, input: str) -> str:
            return input
    """

    def decorator(func: t.Callable) -> DynamoEndpoint:
        return DynamoEndpoint(func, name)

    return decorator


def dynamo_api(func: t.Callable) -> t.Callable:
    """Decorator for BentoML API endpoints.

    Args:
        func: The function to be decorated.

    Returns:
        The decorated function.
    """
    return bentoml.api(func)


def async_on_start(func: t.Callable) -> t.Callable:
    """Decorator for async onstart functions."""
    # Mark the function as a startup hook
    setattr(func, "__bentoml_startup_hook__", True)
    return bentoml.on_startup(func)
