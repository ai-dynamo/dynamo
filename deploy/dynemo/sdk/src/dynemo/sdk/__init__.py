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

from bentoml import api  # type: ignore
from bentoml._internal.context import server_context  # type: ignore

from dynemo.sdk.lib.decorators import async_onstart, dynemo_api, dynemo_endpoint
from dynemo.sdk.lib.dependency import depends
from dynemo.sdk.lib.image import DYNEMO_IMAGE
from dynemo.sdk.lib.service import service

dynemo_context = {}

__all__ = [
    "api",
    "server_context",
    "async_onstart",
    "dynemo_api",
    "dynemo_endpoint",
    "depends",
    "DYNEMO_IMAGE",
    "service",
    "dynemo_context",
]
