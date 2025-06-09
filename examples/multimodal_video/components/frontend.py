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

import json
import logging

from components.processor import Processor
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from utils.protocol import MultiModalRequest

from dynamo.sdk import DYNAMO_IMAGE, api, depends, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
    image=DYNAMO_IMAGE,
    app=FastAPI(title="Multimodal Example"),
)
class Frontend:
    processor = depends(Processor)

    @api(name="v1/chat/completions")
    async def generate(self, request: MultiModalRequest):
        async def content_generator():
            async for response in self.processor.generate(request.model_dump_json()):
                s = json.loads(response)
                yield s

        return StreamingResponse(content_generator(), media_type="text/event-stream")
