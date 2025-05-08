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

import asyncio
import logging
import uuid
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Union

from components.worker import SGLangWorker
from sglang.srt.openai_api.protocol import ChatCompletionRequest, CompletionRequest
from utils.chat_processor import ChatProcessor, CompletionsProcessor, ProcessMixIn
from utils.protocol import SGLangGenerateRequest
from utils.sglang import parse_sglang_args

from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service

logger = logging.getLogger(__name__)


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    workers=1,
)
class Processor(ProcessMixIn):
    """
    SGLang pre and post processing
    """

    sglang_worker = depends(SGLangWorker)

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_sglang_args(class_name, "")
        self.chat_processor = ChatProcessor()
        self.completions_processor = CompletionsProcessor()
        # test
        self.request_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.request_futures: Dict[str, asyncio.Future] = {}
        self.num_worker_tasks = 4
        self.worker_tasks: List[asyncio.Task] = []
        # test
        # init tokenizer manager
        super().__init__()

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        comp_ns, comp_name = SGLangWorker.dynamo_address()  # type: ignore
        self.worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("generate")
            .client()
        )
        logger.warning("processor initialized")

        # Start multiple worker tasks to process the queue
        self._start_worker_tasks()

    def _start_worker_tasks(self):
        """Start multiple worker tasks to process the queue concurrently"""
        # Clear any existing worker tasks
        for task in self.worker_tasks:
            if not task.done():
                task.cancel()

        self.worker_tasks = []

        # Create new worker tasks
        for i in range(self.num_worker_tasks):
            task = asyncio.create_task(self._process_queue(worker_id=i))
            self.worker_tasks.append(task)

        logger.info(f"Started {self.num_worker_tasks} queue worker tasks")

    async def _process_queue(self, worker_id: int):
        """Background task to process the request queue"""
        logger.info(f"Queue worker {worker_id} started")
        while True:
            try:
                # Get the next request from the queue
                request_data = await self.request_queue.get()

                # Process the request
                try:
                    await self._process_request(request_data)
                except Exception as e:
                    logger.error(f"Worker {worker_id}: Error processing request: {e}")
                finally:
                    # Mark the task as done
                    self.request_queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"Queue worker {worker_id} was cancelled")
                break
            except Exception as e:
                logger.error(
                    f"Worker {worker_id}: Unexpected error in queue processing: {e}"
                )
                # Sleep briefly to avoid tight error loops
                await asyncio.sleep(0.1)

    async def _process_request(self, request_data: Dict[str, Any]):
        """Process a single request from the queue"""
        request_id = request_data["request_id"]
        raw_request = request_data["raw_request"]
        # request_type = request_data["request_type"]

        try:
            # Parse the raw request
            request, sampling_params = await self._parse_raw_request(raw_request)

            # Create an async generator function to process this request
            async def process_and_stream():
                request_obj = SGLangGenerateRequest(
                    request_id=request_id,
                    input_ids=request.input_ids,
                    sampling_params=sampling_params,
                ).model_dump_json()

                engine_generator = await self.worker_client.generate(request_obj)

                # Stream responses directly to the caller
                async for result in engine_generator:
                    yield result

            # Set the future result to our async generator
            if request_id in self.request_futures:
                self.request_futures[request_id].set_result(process_and_stream())

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            # Set exception on the future if it still exists
            if (
                request_id in self.request_futures
                and not self.request_futures[request_id].done()
            ):
                self.request_futures[request_id].set_exception(e)

    async def _generate(
        self,
        raw_request: Union[CompletionRequest, ChatCompletionRequest],
        request_type: RequestType,
    ):
        request_id = str(uuid.uuid4())
        logger.debug(f"Got raw request: {raw_request}")

        # Create a future for this request
        future: asyncio.Future[AsyncIterator[Any]] = asyncio.Future()
        self.request_futures[request_id] = future

        # Enqueue the request with minimal processing
        await self.request_queue.put(
            {
                "request_id": request_id,
                "raw_request": raw_request,
                "request_type": request_type,
            }
        )

        try:
            # Wait for the future to complete and yield the results
            generator = await future
            async for result in generator:
                yield result
        finally:
            # Clean up the future when done
            if request_id in self.request_futures:
                del self.request_futures[request_id]

    @dynamo_endpoint(name="chat/completions")
    async def chat_completions(self, raw_request: ChatCompletionRequest):
        sglang_generator = self._generate(raw_request, RequestType.CHAT)

        async for response in self.chat_processor.generate_stream_response(
            raw_request, sglang_generator, self.tokenizer_manager
        ):
            yield response

    @dynamo_endpoint(name="completions")
    async def completions(self, raw_request: CompletionRequest):
        sglang_generator = self._generate(raw_request, RequestType.COMPLETION)

        async for response in self.completions_processor.generate_stream_response(
            raw_request, sglang_generator, self.tokenizer_manager
        ):
            yield response
