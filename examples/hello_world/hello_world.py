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

from pydantic import BaseModel
from dynamo.sdk import DYNAMO_IMAGE, api, depends, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

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
    text: str


@service(
    dynamo={
        "enabled": True,
        "namespace": "inference",
    },
    image=DYNAMO_IMAGE,
)
class Backend:
    def __init__(self) -> None:
        print("Starting backend")
        config = ServiceConfig.get_instance()
        self.message = config.get("Backend", {}).get("message", "Default Backend Message")
        print(f"Backend config message: {self.message}")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Generate tokens."""
        req_text = req.text
        print(f"Backend received: {req_text}")
        text = f"{req_text}-back"
        for token in text.split():
            yield f"Backend: {token}"


@service(
    dynamo={"enabled": True, "namespace": "inference"},
    image=DYNAMO_IMAGE,
)
class Middle:
    backend = depends(Backend)

    def __init__(self) -> None:
        print("Starting middle")
        config = ServiceConfig.get_instance()
        self.message = config.get("Middle", {}).get("message", "Default Middle Message")
        print(f"Middle config message: {self.message}")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Forward requests to backend."""
        req_text = req.text
        print(f"Middle received: {req_text}")
        text = f"{req_text}-mid"
        next_request = RequestType(text=text).model_dump_json()
        async for response in self.backend.generate(next_request):
            print(f"Middle received response: {response}")
            yield f"Middle: {response}"


@service(
    image=DYNAMO_IMAGE,
)  # Regular HTTP API
class Frontend:
    middle = depends(Middle)

    def __init__(self) -> None:
        print("Starting frontend")
        config = ServiceConfig.get_instance()
        self.message = config.get("Frontend", {}).get("message", "Default Frontend Message")
        self.port = config.get("Frontend", {}).get("port", 8000)
        print(f"Frontend config message: {self.message}")
        print(f"Frontend config port: {self.port}")

    @api
    async def generate(self, text):
        """Stream results from the pipeline."""
        print(f"Frontend received: {text}")
        print(f"Frontend received type: {type(text)}")
        txt = RequestType(text=text)
        print(f"Frontend sending: {type(txt)}")
        async for response in self.middle.generate(txt.model_dump_json()):
            yield f"Frontend: {response}"
