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

from components.processor import Processor
from components.utils import GeneralRequest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from dynamo.sdk import DYNAMO_IMAGE, depends, dynamo_endpoint, service

logger = logging.getLogger(__name__)

app = FastAPI(title="Hello World!")


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo-demo",
    },
    image=DYNAMO_IMAGE,
    app=app,
)
class Frontend:
    processor = depends(Processor)

    @dynamo_endpoint(is_api=True)
    async def generate(self, request: GeneralRequest):  # from request body keys
        """Stream results from the pipeline."""
        logger.info(f"-Frontend layer received: {request=}")

        async def content_generator():
            async for response in self.processor.generate(request.model_dump_json()):
                yield f"Frontend: {response}"

        return StreamingResponse(content_generator())
