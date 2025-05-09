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

from dynamo.sdk.core.decorators.endpoint import dynamo_api, dynamo_endpoint
from dynamo.sdk.core.lib import DYNAMO_IMAGE, depends, service
from dynamo.sdk.lib.decorators import async_on_start

dynamo_context: dict[str, Any] = {}

__all__ = [
    "DYNAMO_IMAGE",
    "async_on_shutdown",
    "async_on_start",
    "depends",
    "dynamo_context",
    "dynamo_endpoint",
    "dynamo_api",
    "service",
]
