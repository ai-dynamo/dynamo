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

from dynemo.sdk import DYNEMO_IMAGE, api, depends, service
from sdk_kv_router.processor import Processor


@service(traffic={"timeout": 10000}, image=DYNEMO_IMAGE)
class Frontend:
    processor = depends(Processor)

    def __init__(self):
        print("frontend init")

    @api
    async def chat_completion(self, msg: str):
        # Call the generate method
        generator = self.processor.generate(
            {
                "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "messages": [{"role": "user", "content": msg}],
                "stream": True,
                "max_tokens": 10,
            }
        )

        # Now iterate over the async generator
        async for response in generator:
            print("client response_data:", response)
            yield response
