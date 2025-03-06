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

from bentoml import api
from bentoml._internal.context import server_context
from compoundai.sdk.decorators import async_onstart, nova_api, nova_endpoint
from compoundai.sdk.dependency import depends
from compoundai.sdk.image import NOVA_IMAGE
from compoundai.sdk.service import service

tdist_context = {}

__all__ = [
    "api",
    "server_context",
    "async_onstart",
    "nova_api",
    "nova_endpoint",
    "depends",
    "NOVA_IMAGE",
    "service",
    "tdist_context",
]
