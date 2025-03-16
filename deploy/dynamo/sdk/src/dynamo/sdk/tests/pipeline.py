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


GPU_ENABLED = False


@service(
    resources={"cpu": "1"},
    traffic={"timeout": 30},
    dynamo={
        "enabled": True,
        "namespace": "inference",
    },
    workers=1,
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
class Backend2:
    backend = depends(Backend)

    def __init__(self) -> None:
        print("Starting middle2")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Forward requests to backend."""

        req_text = req.text
        print(f"Middle2 received: {req_text}")
        text = f"{req_text}-mid2"
        next_request = RequestType(text=text).model_dump_json()
        print(next_request)


@service(
    resources={"cpu": "1"},
    traffic={"timeout": 30},
    dynamo={"enabled": True, "namespace": "inference"},
)
class Middle:
    backend = depends(Backend)
    backend2 = depends(Backend2)

    def __init__(self) -> None:
        print("Starting middle")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Forward requests to backend."""
        req_text = req.text
        print(f"Middle received: {req_text}")
        text = f"{req_text}-mid"
        for token in text.split():
            yield f"Mid: {token}"


@service(resources={"cpu": "1"}, traffic={"timeout": 60})
class Frontend:
    middle = depends(Middle)
    backend = depends(Backend)

    def __init__(self) -> None:
        print("Starting frontend")

    @api
    async def generate(self, text):
        """Stream results from the pipeline."""
        print(f"Frontend received: {text}")
        print(f"Frontend received type: {type(text)}")
        txt = RequestType(text=text)
        print(f"Frontend sending: {type(txt)}")
        if self.backend:
            async for back_resp in self.backend.generate(txt.model_dump_json()):
                print(f"Frontend received back_resp: {back_resp}")
                yield f"Frontend: {back_resp}"
        else:
            async for mid_resp in self.middle.generate(txt.model_dump_json()):
                print(f"Frontend received mid_resp: {mid_resp}")
                yield f"Frontend: {mid_resp}"
