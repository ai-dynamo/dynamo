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


from typing import Optional

import msgspec
from utils.nats_queue import NATSQueue
from vllm.remote_prefill import RemotePrefillRequest


class PrefillQueue(NATSQueue):
    """
    A wrapper of NATSQueue for PrefillRequest.
    The stream name is forced to be "prefill_queue".
    """

    def __init__(
        self,
        stream_name="prefill_queue",
        nats_server: str = "nats://localhost:4222",
        dequeue_timeout: float = 1,
    ):
        super().__init__(
            stream_name=stream_name,
            nats_server=nats_server,
            dequeue_timeout=dequeue_timeout,
        )

    async def enqueue_prefill_request(
        self, prefill_request: RemotePrefillRequest
    ) -> None:
        encoded_request = msgspec.json.encode(prefill_request)
        await self.enqueue_task(encoded_request)

    async def dequeue_prefill_request(self) -> Optional[RemotePrefillRequest]:
        encoded_request = await self.dequeue_task()
        if encoded_request is not None:
            prefill_request = msgspec.json.decode(
                encoded_request, type=RemotePrefillRequest
            )
            return prefill_request
        else:
            return None
