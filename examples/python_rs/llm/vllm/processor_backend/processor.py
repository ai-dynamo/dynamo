# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import asyncio

import uvloop
from processor_backend.common import parse_vllm_args

from triton_distributed.runtime import (
    DistributedRuntime,
    ModelDeploymentCard,
    OAIChatPreprocessor,
    triton_worker,
)

uvloop.install()


@triton_worker()
async def preprocessor(runtime: DistributedRuntime, model_name: str, model_path: str):
    # create model deployment card
    mdc = await ModelDeploymentCard.from_local_path(model_path, model_name)
    # create preprocessor endpoint
    component = runtime.namespace("triton-init").component("preprocessor")
    await component.create_service()
    endpoint = component.endpoint("generate")

    # create backend endpoint
    backend = runtime.namespace("triton-init").component("backend").endpoint("generate")

    # start preprocessor service with next backend
    chat = OAIChatPreprocessor(mdc, endpoint, next=backend)
    await chat.start()


if __name__ == "__main__":
    args = parse_vllm_args()
    asyncio.run(preprocessor(args.model, args.model_path))
