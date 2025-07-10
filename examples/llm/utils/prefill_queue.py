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

        self.pending = None

    async def enqueue_prefill_request(
        self, prefill_request: RemotePrefillRequest
    ) -> None:
        encoded_request = msgspec.json.encode(prefill_request)
        await self.enqueue_task(encoded_request)

    async def dequeue_prefill_request(
        self, timeout: Optional[float] = None
    ) -> Optional[RemotePrefillRequest]:
        encoded_request = await self.dequeue_task(timeout)
        if encoded_request is not None:
            prefill_request = msgspec.json.decode(
                encoded_request, type=RemotePrefillRequest
            )
            return prefill_request
        else:
            return None

    async def dequeue_prefill_request_batch(
        self, max_batched_prefill_tokens: int, block_size: int
    ) -> Optional[List[RemotePrefillRequest]]:
        def num_new_tokens(req: RemotePrefillRequest) -> int:
            return len(req.prompt_token_ids) - len(req.computed_block_ids) * block_size

        req = (
            self.pending
            if self.pending is not None
            else await self.dequeue_prefill_request()
        )

        if req is None:
            return None

        reqs = [req]

        # Reset the pending request (if any).
        self.pending = None
        # Determine how much margin we have for more requests in the same batch.
        remaining_prefill_tokens = max_batched_prefill_tokens - num_new_tokens(req)

        if remaining_prefill_tokens < 0:
            return reqs

        # TODO: We might want to double-buffer this process
        # to avoid the overhead of dequeuing from nats
        prefill_queue_size = await self.get_queue_size()
        for _ in range(prefill_queue_size):
            # This should be immediate, hence the zero timeout.
            req = await self.dequeue_prefill_request(0)

            if req is None:
                break

            if num_new_tokens(req) <= remaining_prefill_tokens:
                reqs.append(req)
                remaining_prefill_tokens -= num_new_tokens(req)
            else:
                # We need to save this request for the next batch.
                self.pending = req
                break

        return reqs
