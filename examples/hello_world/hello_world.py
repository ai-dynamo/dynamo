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

from pydantic import BaseModel

from dynamo.sdk import DYNAMO_IMAGE, api, depends, dynamo_endpoint, service

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

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Generate tokens."""
        req_text = req.text
        logger.info(f"Backend received: {req_text}")
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
        logger.info("Starting middle")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Forward requests to backend."""
        req_text = req.text
        logger.info(f"Middle received: {req_text}")
        text = f"{req_text}-mid"
        next_request = RequestType(text=text).model_dump_json()
        async for response in self.backend.generate(next_request):
            logger.info(f"Middle received response: {response}")
            yield f"Middle: {response}"


@service(
    image=DYNAMO_IMAGE,
)  # Regular HTTP API
class Frontend:
    middle = depends(Middle)

    def __init__(self) -> None:
        print("Starting frontend")

    @api
    async def generate(self, text):
        """Stream results from the pipeline."""
        print(f"Frontend received: {text}")
        print(f"Frontend received type: {type(text)}")
        txt = RequestType(text=text)
        print(f"Frontend sending: {type(txt)}")
        async for response in self.middle.generate(txt.model_dump_json()):
            yield f"Frontend: {response}"
