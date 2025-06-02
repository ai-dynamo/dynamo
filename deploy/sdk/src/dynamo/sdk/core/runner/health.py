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


from typing import Any, Awaitable, Union

from fastapi import FastAPI, Response


# TODO: These defaults should be set by the provider. For now, I'm just adding them so that something is exposed when we do --use-default-health-checks
def default_liveness_check() -> bool:
    """Default liveness check that always returns True."""
    return True


def default_readiness_check() -> bool:
    """Default readiness check that always returns True."""
    return True


def register_liveness_probe(
    app: FastAPI, instance: Any, route: str = "/healthz", use_default: bool = False
) -> None:
    """Registers /healthz endpoint.

    If a method decorated with @liveness is found, uses that.
    Otherwise, if use_default is True, uses a default check that always returns 200.
    Does nothing if neither condition is met.

     Args:
        app (FastAPI): The FastAPI application to register the liveness route on.
        instance (Any): The service or component instance to inspect for a @liveness-decorated method.
        route (str, optional): The URL path to register the liveness endpoint under.
                               Defaults to "/healthz".
        use_default (bool, optional): Whether to use default health check if no decorated method is found.
                                    Defaults to False.
    """

    # Find the decorated method.
    decorated_method = None
    for attr in dir(instance):
        method = getattr(instance, attr)
        if callable(method) and getattr(method, "__is_liveness_probe__", False):
            decorated_method = method
            break

    if not decorated_method and not use_default:
        # Do nothing if no @liveness() decorator found and default not requested
        return

    @app.get(route)
    async def liveness_check():
        try:
            # Use decorated method if available, otherwise use default
            check_method = (
                decorated_method if decorated_method else default_liveness_check
            )
            # self needs to be bound so we need to use the instance of the inner
            result: Union[bool, Awaitable[bool]] = check_method()
            if isinstance(result, Awaitable):
                result = await result
            return Response(status_code=200 if result else 503)
        except Exception as e:
            return Response(content=str(e), status_code=500)


def register_readiness_probe(
    app: FastAPI, instance: Any, route: str = "/readyz", use_default: bool = False
) -> None:
    """Registers /readyz endpoint.

    If a method decorated with @readiness is found, uses that.
    Otherwise, if use_default is True, uses a default check that always returns 200.
    Does nothing if neither condition is met.

     Args:
        app (FastAPI): The FastAPI application to register the readiness route on.
        instance (Any): The service or component instance to inspect for a @readiness-decorated method.
        route (str, optional): The URL path to register the readiness endpoint under.
                               Defaults to "/readyz".
        use_default (bool, optional): Whether to use default health check if no decorated method is found.
                                    Defaults to False.
    """

    # Find the decorated method.
    decorated_method = None
    for attr in dir(instance):
        method = getattr(instance, attr)
        if callable(method) and getattr(method, "__is_readiness_probe__", False):
            decorated_method = method
            break

    if not decorated_method and not use_default:
        # Do nothing if no @readiness() decorator found and default not requested
        return

    @app.get(route)
    async def readiness_check():
        try:
            # Use decorated method if available, otherwise use default
            check_method = (
                decorated_method if decorated_method else default_readiness_check
            )
            # self needs to be bound so we need to use the instance of the inner
            result: Union[bool, Awaitable[bool]] = check_method()
            if isinstance(result, Awaitable):
                result = await result
            return Response(status_code=200 if result else 503)
        except Exception as e:
            return Response(content=str(e), status_code=500)
