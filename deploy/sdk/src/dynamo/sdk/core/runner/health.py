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


from typing import Any

from fastapi import FastAPI, Response


def register_liveness_probe(
    app: FastAPI, instance: Any, route: str = "/healthz/liveness"
) -> None:
    """Registers /healthz/liveness only if a method decorated with @liveness is found.

    Does nothing otherwise (no fallback, no error).

     Args:
        app (FastAPI): The FastAPI application to register the liveness route on.
        instance (Any): The service or component instance to inspect for a @liveness-decorated method.
        route (str, optional): The URL path to register the liveness endpoint under.
                               Defaults to "/healthz/liveness".
    """

    # Find the decorated method.
    for attr in dir(instance):
        decorated_method = getattr(instance, attr)
        if callable(decorated_method) and getattr(
            decorated_method, "__is_liveness_probe__", False
        ):
            break
    else:
        # Do nothing if no @liveness() decorator found.
        return

    @app.get(route)
    async def liveness_check():
        try:
            result = decorated_method()
            if callable(getattr(result, "__await__", None)):
                result = await result
            return Response(status_code=200 if result else 503)
        except Exception as e:
            return Response(content=str(e), status_code=500)
