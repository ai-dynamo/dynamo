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

# This is a simple example of a pipeline that uses Dynamo to deploy a backend, middle, and frontend service.
# Use this to test changes made to CLI, SDK, etc


from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dynamo.sdk import depends, dynamo_endpoint, service

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


app = FastAPI(title="Hello World!")


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

    @dynamo_endpoint()
    async def generate_v2(self, req: RequestType):
        """Generate tokens."""
        req_text = req.text
        print(f"Backend received: {req_text}")
        text = f"{req_text}-back"
        for token in text.split():
            yield f"Backend generate_v2: {token}"


@service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
    dynamo={"enabled": True, "namespace": "inference"},
)
class Backend2:
    backend = depends(Backend)

    def __init__(self) -> None:
        print("Starting backend2")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Forward requests to backend."""

        req_text = req.text
        print(f"Backend2 received: {req_text}")
        text = f"{req_text}-back2"
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

        txt = RequestType(text=text)

        if self.backend:
            async for back_resp in self.backend.generate(txt.model_dump_json()):
                print(f"Frontend received back_resp: {back_resp}")
                yield f"Frontend: {back_resp}"
            async for back_resp in self.backend.generate_v2(txt.model_dump_json()):
                print(f"Frontend received back_resp: {back_resp}")
                yield f"Frontend: {back_resp}"
        else:
            async for back_resp in self.backend2.generate(txt.model_dump_json()):
                print(f"Frontend received back_resp: {back_resp}")
                yield f"Frontend: {back_resp}"


@service(
    resources={"cpu": "1"},
    traffic={"timeout": 60},
    dynamo={"enabled": True, "namespace": "inference"},
    app=app,
)
class Frontend:
    middle = depends(Middle)
    backend = depends(Backend)

    def __init__(self) -> None:
        print("Starting frontend")

    @dynamo_endpoint(is_api=True)
    async def generate(self, request: RequestType):
        """Stream results from the pipeline."""
        print(f"Frontend received: {request.text}")

        async def content_generator():
            async for response in self.middle.generate(request.model_dump_json()):
                yield f"Frontend: {response}"

        return StreamingResponse(content_generator())
