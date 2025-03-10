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

# This is a simple example of a pipeline that uses Dynamo to deploy a backend, middle, and frontend service. Use this to test
# changes made to CLI, SDK, etc

from pydantic import BaseModel

from dynamo.sdk import api, depends, dynamo_endpoint, service
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
    resources={"cpu": "2"},
    traffic={"timeout": 30},
    dynamo={
        "enabled": True,
        "namespace": "inference",
    },
    workers=3,
)
class Backend:
    def __init__(self) -> None:
        print("Starting backend")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Generate tokens."""
        req_text = req.text
        print(f"Backend received: {req_text}")
        text = f"{req_text}-back"
        for token in text.split():
            yield f"Backend: {token}"


@service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
    dynamo={"enabled": True, "namespace": "inference"},
)
class Middle:
    backend = depends(Backend)

    def __init__(self) -> None:
        print("Starting middle")

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


@service(resources={"cpu": "1"}, traffic={"timeout": 60})  # Regular HTTP API
class Frontend:
    middle = depends(Middle)

    def __init__(self) -> None:
        print("Starting frontend")
        self.config = ServiceConfig.get_instance()

        # Get required configuration
        self.model = self.config.require_config("Frontend", "model")
        
        # Get optional configurations with defaults
        self.temperature = self.config.get_config("Frontend", "temperature", 0.7)
        self.max_tokens = self.config.get_config("Frontend", "max_tokens", 1024)
        self.stream = self.config.get_config("Frontend", "stream", True)
        
        print(f"Frontend initialized with model={self.model}, "
              f"temp={self.temperature}, max_tokens={self.max_tokens}")
        
        # You can get all configs for debugging
        all_frontend_configs = self.config.get_service_config("Frontend")
        print(f"All Frontend configs: {all_frontend_configs}")
        
        # You can also check if other services have specific configs
        if self.config.get_config("Middle", "special_mode") == "fast":
            print("Using Middle service in fast mode")
    @api
    async def generate(self, text):
        """Stream results from the pipeline."""
        print(f"Frontend received: {text}")
        print(f"Frontend received type: {type(text)}")
        txt = RequestType(text=text)
        print(f"Frontend sending: {type(txt)}")
        async for response in self.middle.generate(txt.model_dump_json()):
            yield f"Frontend: {response}"
