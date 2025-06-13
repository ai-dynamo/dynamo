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
import os
import signal
from typing import Optional

import connect
import torch
from components.video_encode_worker import VllmEncodeWorker
from pydantic import BaseModel
from utils.logging import check_required_workers
from utils.nixl import NixlMetadataStore
from utils.prefill_queue import PrefillQueue
from utils.protocol import EncodeRequest
from utils.vllm import parse_vllm_args
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest

from dynamo.sdk import async_on_start, depends, dynamo_context, endpoint, service

logger = logging.getLogger(__name__)

# Constants for the shape and dtype of the INCOMING FRAMES tensor.
# Other constants taken from yaml as they are model dependent.
INCOMING_FRAMES_DTYPE = torch.uint8
INCOMING_FRAMES_DEVICE = "cpu"


class RequestType(BaseModel):
    text: str


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmPrefillWorker:
    encode_worker = depends(VllmEncodeWorker)

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.model_path = self.engine_args.model  # Store model path for AutoProcessor
        self.num_sampled_frames = getattr(self.engine_args, "num_sampled_frames", 8)
        self.frame_height = getattr(self.engine_args, "frame_height", 336)
        self.frame_width = getattr(self.engine_args, "frame_width", 336)
        self.frame_channels = getattr(self.engine_args, "frame_channels", 3)
        self.dummy_token_id = getattr(self.engine_args, "dummy_token_id", 0)
        self.video_token_id = getattr(self.engine_args, "video_token_id", 32000)
        self.dummy_tokens_per_frame = getattr(
            self.engine_args, "dummy_tokens_per_frame", 144
        )
        self._loaded_metadata = set()
        self.initialized = False
        self.min_workers = 1

        # IMPORTANT: PrefillWorker MUST remove dummy tokens before passing to vLLM
        # Only the actual video tokens (32000) should remain as placeholders for multimodal embeddings

        if self.engine_args.enable_chunked_prefill is not False:
            logger.info("Chunked prefill is not supported yet, setting to False")
            self.engine_args.enable_chunked_prefill = False

        if self.engine_args.pipeline_parallel_size != 1:
            logger.info("Pipeline parallel size is not supported yet, setting to 1")
            self.engine_args.pipeline_parallel_size = 1

        if self.engine_args.disable_async_output_proc is not True:
            logger.info("Async output processing is not supported yet, setting to True")
            self.engine_args.disable_async_output_proc = True

        if self.engine_args.enforce_eager is not True:
            logger.info("Prefill must be done eagerly, setting to True")
            self.engine_args.enforce_eager = True

        if self.engine_args.enable_prefix_caching is not False:
            logger.info(
                "Prefix caching is not supported yet in prefill worker, setting to False"
            )
            self.engine_args.enable_prefix_caching = False

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

    @async_on_start
    async def async_init(self):
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to initialize engine client")

        # NOTE: PrefillWorker no longer needs AutoProcessor since it uses the original
        # tokenized prompt from DecodeWorker instead of creating its own.
        logger.info(
            "PrefillWorker: Skipping AutoProcessor initialization - using original tokens from DecodeWorker"
        )

        runtime = dynamo_context["runtime"]

        enc_comp_ns, enc_comp_name = VllmEncodeWorker.dynamo_address()  # type: ignore
        self.encode_worker_client = (
            await runtime.namespace(enc_comp_ns)
            .component(enc_comp_name)
            .endpoint("encode")
            .client()
        )

        # Initialize the connector for RDMA transfers within the specified namespace.
        self._connector = connect.Connector(runtime=runtime, namespace=enc_comp_ns)
        await self._connector.initialize()

        incoming_frames_shape = (
            self.num_sampled_frames,
            self.frame_height,
            self.frame_width,
            self.frame_channels,
        )

        # Pre-allocate a tensor on the CPU to receive frame data.
        frames_tensor = torch.empty(
            incoming_frames_shape,
            dtype=INCOMING_FRAMES_DTYPE,
            device=INCOMING_FRAMES_DEVICE,
        )
        # Create a descriptor for the tensor to make it available for remote access.
        descriptor = connect.Descriptor(frames_tensor)
        # Register the memory with the connector, making it discoverable.
        descriptor.register_memory(self._connector)
        self._frames_descriptor = (frames_tensor, descriptor)

        await check_required_workers(self.encode_worker_client, self.min_workers)

        metadata = self.engine_client.nixl_metadata
        self._metadata_store = NixlMetadataStore("dynamo", runtime)
        await self._metadata_store.put(metadata.engine_id, metadata)

        logger.info("PrefillWorker: Creating prefill_queue_handler task.")
        task = asyncio.create_task(self.prefill_queue_handler())

        def prefill_queue_handler_cb(fut):
            try:
                fut.result()
                logger.info(
                    "PrefillWorker: prefill_queue_handler task exited successfully."
                )
            except asyncio.CancelledError:
                logger.info("PrefillWorker: prefill_queue_handler task was cancelled.")
            except Exception as e:
                logger.error(
                    f"PrefillWorker: prefill_queue_handler task failed with exception: {e!r}",
                    exc_info=True,
                )

        task.add_done_callback(prefill_queue_handler_cb)
        logger.info("PrefillWorker: async_init complete.")

    def shutdown_vllm_engine(self, signum, frame):
        """Shutdown the background loop"""
        logger.info(f"Shutdown started, signal {signum} received.")
        loop = asyncio.get_event_loop()
        try:
            self.engine_client.close()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            loop.stop()
        logger.info("Shutdown complete.")

    async def prefill_queue_handler(self):
        logger.info("PrefillWorker: Prefill queue handler task started.")
        prefill_queue_nats_server = os.getenv("NATS_SERVER", "nats://localhost:4222")
        prefill_queue_stream_name = (
            self.engine_args.served_model_name
            if self.engine_args.served_model_name is not None
            else "vllm"
        )
        logger.info(
            f"PrefillWorker: Connecting to prefill queue: {prefill_queue_nats_server}, stream: '{prefill_queue_stream_name}'"
        )
        self.initialized = True
        try:
            async with PrefillQueue.get_instance(
                nats_server=prefill_queue_nats_server,
                stream_name=prefill_queue_stream_name,
            ) as prefill_queue:
                logger.info(
                    f"PrefillWorker: Entering dequeue loop for stream '{prefill_queue_stream_name}'."
                )
                while True:
                    prefill_request: Optional[RemotePrefillRequest] = None
                    try:
                        prefill_request = await prefill_queue.dequeue_prefill_request()
                    except Exception as e:
                        logger.error(
                            f"PrefillWorker: Exception during dequeue_prefill_request: {e}",
                            exc_info=True,
                        )
                        await asyncio.sleep(5)
                        continue

                    if prefill_request is not None:
                        logger.info(
                            f"PrefillWorker: Dequeued prefill request: {prefill_request.request_id}"
                        )
                        try:
                            async for _ in self.generate(prefill_request):
                                pass
                            logger.info(
                                f"PrefillWorker: Successfully processed prefill request {prefill_request.request_id}."
                            )
                        except Exception as e:
                            logger.error(
                                f"PrefillWorker: Error processing prefill request {prefill_request.request_id} in self.generate: {e}",
                                exc_info=True,
                            )
                    else:
                        await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(
                f"PrefillWorker: Prefill queue handler CRASHED: {e}", exc_info=True
            )

    async def generate(self, request: RemotePrefillRequest):
        video_url = request.multimodal_data_source.get("video_url")

        if video_url is None:
            raise ValueError(
                "No video_url provided in multimodal_data_source for prefill request"
            )

        request_id = request.request_id
        engine_id = request.engine_id

        logger.info(
            f"PrefillWorker {request_id}: Received prefill request for video_url: {video_url}."
        )

        raw_frames_tensor, descriptor = self._frames_descriptor

        logger.debug(
            f"PrefillWorker {request_id}: Requesting frames from EncodeWorker for {video_url}"
        )
        # Create a writable operation handle for the remote EncodeWorker.
        # This allows the EncodeWorker to write directly into this worker's `frames_tensor`.
        with self._connector.create_writable(descriptor) as writable:
            encode_generator = await self.encode_worker_client.round_robin(
                EncodeRequest(
                    request_id=request_id,
                    video_url=video_url,
                    # Serialize the writable handle to send it to the EncodeWorker.
                    serialized_request=writable.to_serialized(),
                ).model_dump_json()
            )
            async for _ in encode_generator:
                pass
            # Wait for the remote write from the EncodeWorker to complete.
            await writable.wait_for_completion()
            logger.debug(
                f"PrefillWorker {request_id}: Frames received from EncodeWorker, shape: {raw_frames_tensor.shape}"
            )

        if not request.prompt_token_ids:
            logger.error(
                f"PrefillWorker {request_id}: No prompt_token_ids provided in request!"
            )
            raise ValueError(
                "PrefillWorker requires prompt_token_ids from DecodeWorker"
            )

        # Constants for token manipulation
        DUMMY_TOKEN_ID = self.dummy_token_id
        VIDEO_TOKEN_ID = self.video_token_id

        # Step 1: Find all video token positions
        video_token_positions = [
            i
            for i, token in enumerate(request.prompt_token_ids)
            if token == VIDEO_TOKEN_ID
        ]
        logger.debug(
            f"PrefillWorker {request_id}: Found {len(video_token_positions)} video tokens at positions: {video_token_positions}"
        )

        # Step 2: Process tokens from end to start to avoid position shifting
        processed_tokens = list(request.prompt_token_ids)
        for pos in reversed(video_token_positions):
            # Calculate range of tokens to remove (dummy tokens after this video token)
            start_idx = pos + 1
            end_idx = start_idx + self.dummy_tokens_per_frame

            # Check if we have enough tokens to remove
            if end_idx > len(processed_tokens):
                logger.warning(
                    f"PrefillWorker {request_id}: Not enough tokens to remove at position {pos}"
                )
                continue

            # Remove the dummy tokens
            processed_tokens = processed_tokens[:start_idx] + processed_tokens[end_idx:]

        # Step 3: Verify we have exactly one video token left
        final_video_count = sum(
            1 for token in processed_tokens if token == VIDEO_TOKEN_ID
        )
        if final_video_count != 1:
            logger.error(
                f"PrefillWorker {request_id}: Wrong number of video tokens! Expected 1, got {final_video_count}"
            )

        # Step 4: Check for any remaining dummy tokens (should be none)
        remaining_dummies = sum(
            1 for token in processed_tokens if token == DUMMY_TOKEN_ID
        )
        if remaining_dummies > 0:
            logger.warning(
                f"PrefillWorker {request_id}: Found {remaining_dummies} remaining dummy tokens!"
            )

        # Create the input for vLLM
        prefill_vllm_input = TokensPrompt(
            prompt_token_ids=processed_tokens,
            multi_modal_data={"video": raw_frames_tensor.numpy()},
        )

        sampling_params = request.sampling_params
        sampling_params.max_tokens = 1
        sampling_params.min_tokens = 1

        remote_prefill_params = RemotePrefillParams(
            is_remote_decode=True,
            decode_block_ids=request.block_ids,
            decode_engine_id=engine_id,
            decode_computed_block_ids=request.computed_block_ids,
        )

        if engine_id not in self._loaded_metadata:
            remote_metadata = await self._metadata_store.get(request.engine_id)
            await self.engine_client.add_remote_nixl_metadata(remote_metadata)
            logger.info(
                f"Loaded nixl metadata from engine {engine_id} into "
                f"engine {self.engine_client.nixl_metadata.engine_id}"
            )
            self._loaded_metadata.add(engine_id)

        logger.debug(
            f"PrefillWorker {request_id}: Calling engine_client.generate for prefill."
        )

        async for _ in self.engine_client.generate(
            prefill_vllm_input,
            sampling_params=sampling_params,
            request_id=request_id,
            remote_prefill_params=remote_prefill_params,
        ):
            yield

        logger.info(f"PrefillWorker {request_id}: Finished processing prefill request.")

    @endpoint()
    async def mock(self, req: RequestType):
        yield f"mock_response: {req}"
