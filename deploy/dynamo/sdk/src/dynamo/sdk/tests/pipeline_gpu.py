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

import os

from pydantic import BaseModel

from dynamo.sdk import (
    DYNAMO_IMAGE,
    api,
    depends,
    dynamo_endpoint,
    server_context,
    service,
)

"""
Pipeline Architecture:

Users/Clients (HTTP)
      │
      ▼
┌─────────────┐
│  Frontend   │  HTTP API endpoint (/generate)
└─────────────┘
      │ dynamo/runtime
      ▼
┌─────────────┐
│   Middle    │
└─────────────┘
      │ dynamo/runtime
      ▼
┌─────────────┐
│  Backend    │
└─────────────┘
"""


class RequestType(BaseModel):
    text: str


class ResponseType(BaseModel):
    devices: list[int]


@service(
    resources={"gpu": 2, "cpu": "1", "memory": "20Gi"},
    workers=1,
    traffic={"timeout": 30},
    dynamo={
        "enabled": True,
        "namespace": "inference",
    },
    image=DYNAMO_IMAGE,
)
class Backend:
    def __init__(self) -> None:
        print(
            f"Starting backend index: {server_context.worker_index}, os.environ['CUDA_VISIBLE_DEVICES']: {os.environ['CUDA_VISIBLE_DEVICES']}"
        )
        self.devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Generate tokens."""
        yield ResponseType(devices=self.devices)


@service(
    resources={"gpu": 2, "cpu": "1", "memory": "20Gi"},
    workers=1,
    traffic={"timeout": 30},
    dynamo={"enabled": True, "namespace": "inference"},
    image=DYNAMO_IMAGE,
)
class Middle:
    def __init__(self) -> None:
        print(
            f"Starting middle index: {server_context.worker_index}, os.environ['CUDA_VISIBLE_DEVICES']: {os.environ['CUDA_VISIBLE_DEVICES']}"
        )
        self.devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Forward requests to backend."""
        yield ResponseType(devices=self.devices)


@service(
    resources={"gpu": 2, "cpu": "1", "memory": "20Gi"},
    workers=1,
    traffic={"timeout": 60},
    image=DYNAMO_IMAGE,
)  # Regular HTTP API
class Frontend:
    middle = depends(Middle)
    backend = depends(Backend)

    def __init__(self) -> None:
        print(
            f"Starting frontend index: {server_context.worker_index}, os.environ['CUDA_VISIBLE_DEVICES']: {os.environ['CUDA_VISIBLE_DEVICES']}"
        )
        self.devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

    @api
    async def generate(self, text):
        """Stream results from the pipeline."""
        txt = RequestType(text=text)
        resp = {"frontend": self.devices}
        async for response in self.middle.generate(txt.model_dump_json()):
            resp["middle"] = response.devices
            async for response in self.backend.generate(txt.model_dump_json()):
                resp["backend"] = response.devices
                yield resp
