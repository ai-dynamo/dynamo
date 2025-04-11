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


from typing import List, Optional

import msgspec
import torch
from utils.nats_queue import NATSQueue
from vllm.remote_prefill import RemotePrefillRequest

class EncodeRequest(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    """The request data of one remote prefill output of a request.
       Args:
        request_id: The unique ID of the request.
        image_url: The url of the image.
        embedding: The embedding of the image. A list of torch.Tensor.
    """
    request_id: str
    image_url: str
    embedding: List[torch.Tensor]

class EncodeQueue(NATSQueue):
    """
    A wrapper of NATSQueue for EncodeRequest.
    The stream name is forced to be "encode_queue".
    """
    def __init__(
        self,
        stream_name="encode_queue",
        nats_server: str = "nats://localhost:4222",
        dequeue_timeout: float = 1,
    ):
        super().__init__(
            stream_name=stream_name,
            nats_server=nats_server,
            dequeue_timeout=dequeue_timeout,
        )

    async def enqueue_encode_request(
        self, encode_request: EncodeRequest
    ) -> None:
        encoded_request = msgspec.json.encode(encode_request)
        await self.enqueue_task(encoded_request)

    async def dequeue_encode_request(self) -> Optional[EncodeRequest]:
        encoded_request = await self.dequeue_task()
        if encoded_request is not None:
            encode_request = msgspec.json.decode(
                encoded_request, type=EncodeRequest
            )
            return encode_request
        else:
            return None
