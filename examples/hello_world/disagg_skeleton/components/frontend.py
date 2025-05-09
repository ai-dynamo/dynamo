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
import signal
import sys

from components.processor import Processor
from components.utils import GeneralRequest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from dynamo.sdk import DYNAMO_IMAGE, depends, dynamo_api, service

logger = logging.getLogger(__name__)

app = FastAPI(title="Hello World LLM")


@service(
    dynamo={"namespace": "dynamo-demo"},
    image=DYNAMO_IMAGE,
    app=app,
)
class Frontend:
    processor = depends(Processor)

    def __init__(self):
        signal.signal(signal.SIGTERM, self.handle_exit)
        signal.signal(signal.SIGINT, self.handle_exit)

    def handle_exit(self, signum, frame):
        logger.debug(f"Received signal {signum}, shutting down...")
        sys.exit(0)

    @dynamo_api()
    async def generate(self, prompt, request_id):  # from request body keys
        """Stream results from the pipeline."""
        logger.info(f"Received: {prompt=},{request_id=}")

        async def content_generator():
            frontend_request = GeneralRequest(
                prompt=prompt, request_id=request_id
            ).model_dump_json()
            async for response in self.processor.processor_generate(frontend_request):
                yield f"Response: {response}\n"

        return StreamingResponse(content_generator())
