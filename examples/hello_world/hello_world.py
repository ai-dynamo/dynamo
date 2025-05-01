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

import logging

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sdk import DYNAMO_IMAGE, depends, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)

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
        logger.info("Starting backend")
        config = ServiceConfig.get_instance()
        self.message = config.get("Backend", {}).get("message", "back")
        logger.info(f"Backend config message: {self.message}")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Generate tokens."""
        req_text = req.text
        logger.info(f"Backend received: {req_text}")
        text = f"{req_text}-{self.message}"
        for token in text.split():
            yield f"Backend: {token}"


@service(
    dynamo={"enabled": True, "namespace": "inference"},
    image=DYNAMO_IMAGE,
)
class Middle:
    backend = depends(Backend)

    def __init__(self) -> None:
        logger.info("Starting middle")
        config = ServiceConfig.get_instance()
        self.message = config.get("Middle", {}).get("message", "mid")
        logger.info(f"Middle config message: {self.message}")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Forward requests to backend."""
        req_text = req.text
        logger.info(f"Middle received: {req_text}")
        text = f"{req_text}-{self.message}"
        next_request = RequestType(text=text).model_dump_json()
        async for response in self.backend.generate(next_request):
            logger.info(f"Middle received response: {response}")
            yield f"Middle: {response}"


app = FastAPI(title="Hello World!")


@service(
    dynamo={"enabled": True, "namespace": "inference"},
    image=DYNAMO_IMAGE,
    app=app,
)
class Frontend:
    """A simple frontend HTTP API that forwards requests to the dynamo graph."""

    middle = depends(Middle)

    def __init__(self) -> None:
        # Configure logging
        configure_dynamo_logging(service_name="Frontend")

        logger.info("Starting frontend")
        config = ServiceConfig.get_instance()
        self.message = config.get("Frontend", {}).get("message", "front")
        self.port = config.get("Frontend", {}).get("port", 8000)
        logger.info(f"Frontend config message: {self.message}")
        logger.info(f"Frontend config port: {self.port}")

    @dynamo_endpoint(is_api=True)
    async def generate(self, request: RequestType):
        """Stream results from the pipeline."""
        logger.info(f"Frontend received: {request.text}")

        async def content_generator():
            async for response in self.middle.generate(request.model_dump_json()):
                yield f"Frontend: {response}"

        return StreamingResponse(content_generator())
