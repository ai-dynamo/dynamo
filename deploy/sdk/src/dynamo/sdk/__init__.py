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

from typing import Any

from bentoml import on_shutdown as async_on_shutdown

from dynamo.sdk.core.decorators.endpoint import api, endpoint
from dynamo.sdk.core.lib import DYNAMO_IMAGE, depends, liveness, readiness, service
from dynamo.sdk.lib.decorators import async_on_start
from dynamo.sdk.request_tracing import (
    RequestTracingMixin,
    auto_trace_endpoints,
    extract_or_generate_request_id,
    get_current_request_id,
    trace_frontend_endpoint,
    trace_processor_method,
    with_request_tracing,
)

dynamo_context: dict[str, Any] = {}

__all__ = [
    "DYNAMO_IMAGE",
    "async_on_shutdown",
    "async_on_start",
    "auto_trace_endpoints",
    "depends",
    "dynamo_context",
    "endpoint",
    "api",
    "dynamo_endpoint",
    "extract_or_generate_request_id",
    "get_current_request_id",
    "RequestTracingMixin",
    "service",
    "trace_frontend_endpoint",
    "trace_processor_method",
    "with_request_tracing",
    "liveness",
    "readiness",
]
