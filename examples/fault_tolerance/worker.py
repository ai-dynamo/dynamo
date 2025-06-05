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

from dynamo.sdk import endpoint, service

logger = logging.getLogger(__name__)


class RequestType(BaseModel):
    text: str


class ResponseType(BaseModel):
    text: str


# TODO: Can we adjust worker id printed on log?
@service(
    dynamo={"namespace": "FaultTolerance"},
    workers=1,
)
class Worker:
    def __init__(self) -> None:
        logger.info("Starting Worker...")

    @endpoint(name="generate")
    async def generate(self, request: RequestType):
        logger.info(f"Worker received: {request.text}")

        for c in request.text.split(" "):
            # logger.info(f"Worker sending: {c}")
            yield ResponseType(
                text=c,
            ).model_dump_json()
