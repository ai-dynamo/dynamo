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

import argparse
import asyncio
import base64
import binascii
import logging
import os
import signal
import sys
from io import BytesIO
from queue import Queue
from typing import AsyncIterator, Optional, Tuple
from urllib.parse import urlparse

import av
import httpx
import numpy as np
import torch
import uvloop
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser

import dynamo.nixl_connect as connect
from dynamo.runtime import Client, DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.args import Config, base_parse_args, parse_endpoint
from utils.protocol import MyRequestOutput, vLLMMultimodalRequest
from utils.video_processor import (
    calculate_frame_sampling_indices,
    get_video_metadata,
    open_video_container,
    prepare_tensor_for_rdma,
    resize_video_frames,
)

configure_dynamo_logging()
logger = logging.getLogger(__name__)

try:
    import cupy as array_module

    if not array_module.cuda.is_available():
        raise ImportError("CUDA is not available.")
    DEVICE = "cuda"
    logger.info("Using cupy for array operations (GPU mode).")
except ImportError as e:
    logger.warning(f"Failed to import cupy, falling back to numpy: {e}.")
    import numpy as array_module

    DEVICE = "cpu"

CACHE_SIZE_MAXIMUM = 8


class VllmEncodeWorker:
    def __init__(
        self,
        args: argparse.Namespace,
        engine_args: AsyncEngineArgs,
        pd_worker_client: Client,
    ) -> None:
        self.pd_worker_client = pd_worker_client
        self.engine_args = engine_args
        self.model = self.engine_args.model
        self.min_workers = 1

        # Video processing parameters
        self.num_frames_to_sample = 8
        self.frame_height = 336
        self.frame_width = 336
        self.frame_channels = 3
        self._video_content_cache: dict[str, BytesIO] = {}
        self._cache_queue: Queue[str] = Queue(maxsize=CACHE_SIZE_MAXIMUM)

        self._http_client: Optional[httpx.AsyncClient] = None
        self._http_timeout = 60.0

    def cleanup(self):
        pass

    async def _read_video_pyav(
        self, container: av.container.InputContainer, indices: np.ndarray
    ) -> np.ndarray:
        """
        Decode the video with PyAV decoder. Async wrapper.
        """

        def blocking_decode():
            container.seek(0)  # Reset container for decoding
            processed_indices = set(indices)

            # Determine min/max index to optimize decoding loop slightly
            min_idx = 0
            max_idx = -1
            if len(indices) > 0:
                min_idx = np.min(indices)
                max_idx = np.max(indices)

            if (
                not processed_indices
                and container.streams.video
                and container.streams.video[0].frames > 0
            ):
                logger.warning(
                    "_read_video_pyav called with empty indices for a non-empty video, attempting to read first frame."
                )
                try:
                    frame = next(container.decode(video=0))
                    return np.stack([frame.to_ndarray(format="rgb24")])
                except StopIteration:
                    logger.error(
                        "Failed to read even the first frame despite non-empty indices check."
                    )
                    return np.array([])

            decoded_frames_list = []
            for i, frame in enumerate(container.decode(video=0)):
                if i > max_idx and max_idx != -1:  # max_idx is -1 if indices is empty
                    break
                if i >= min_idx and i in processed_indices:
                    decoded_frames_list.append(frame)

            if not decoded_frames_list and len(processed_indices) > 0:
                actual_decoded_count = 0
                try:
                    container.seek(0)  # Reset for counting
                    for _ in container.decode(video=0):
                        actual_decoded_count += 1
                except Exception:  # Handle cases where re-decoding/counting fails
                    pass  # Keep original error message
                raise ValueError(
                    f"Could not decode any frames for the given indices: {indices.tolist()}. "
                    f"Video might be shorter than expected or indices out of bounds. "
                    f"Actual decodable frames in container (approx): {actual_decoded_count}."
                )

            return (
                np.stack([x.to_ndarray(format="rgb24") for x in decoded_frames_list])
                if decoded_frames_list
                else np.array([])
            )

        return await asyncio.to_thread(blocking_decode)

    async def _load_video_content(self, video_url: str) -> BytesIO:
        parsed_url = urlparse(video_url)
        video_url_lower = video_url.lower()

        if parsed_url.scheme in ("http", "https"):
            if video_url_lower in self._video_content_cache:
                logger.info(f"Video content found in cache for URL: {video_url}")
                cached_content = self._video_content_cache[video_url_lower]
                cached_content.seek(0)
                return cached_content

        try:
            video_data: BytesIO
            if parsed_url.scheme == "data":
                if not parsed_url.path.startswith(
                    ("video/", "application/octet-stream")
                ):
                    raise ValueError("Data URL must be a video type or octet-stream")

                media_type_and_data = parsed_url.path.split(",", 1)
                if len(media_type_and_data) != 2:
                    raise ValueError("Invalid Data URL format: missing comma separator")

                media_type, data_segment = media_type_and_data
                if ";base64" not in media_type:
                    raise ValueError("Video Data URL currently must be base64 encoded")

                try:
                    video_bytes = base64.b64decode(data_segment)
                    video_data = BytesIO(video_bytes)
                except binascii.Error as e:
                    raise ValueError(
                        f"Invalid base64 encoding for video data: {e}"
                    ) from e

            elif parsed_url.scheme in ("http", "https"):
                if not self._http_client:
                    await self._init_http_client()
                    if not self._http_client:  # Double check after initialization
                        raise RuntimeError("Failed to initialize HTTP client")

                logger.info(f"Downloading video from URL: {video_url}")
                response = await self._http_client.get(
                    video_url, timeout=self._http_timeout
                )
                response.raise_for_status()

                if not response.content:
                    raise ValueError(
                        f"Empty response content from video URL: {video_url}"
                    )
                video_data = BytesIO(response.content)
                video_data.seek(0)
                logger.info(
                    f"Video downloaded from {video_url}, size: {len(response.content)} bytes."
                )

            elif parsed_url.scheme == "file" or not parsed_url.scheme:
                file_path = parsed_url.path if parsed_url.scheme else video_url
                # Ensure path is absolute or resolve relative to a known base if necessary
                # For simplicity, assuming it's an accessible path.
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Error reading file: {file_path}")

                with open(file_path, "rb") as f:
                    video_bytes = f.read()
                video_data = BytesIO(video_bytes)
            else:
                raise ValueError(
                    f"Unsupported video source scheme: {parsed_url.scheme} for URL {video_url}"
                )

            if parsed_url.scheme in (
                "http",
                "https",
            ):  # Cache successfully downloaded content
                if self._cache_queue.full():
                    oldest_url = self._cache_queue.get_nowait()
                    if oldest_url in self._video_content_cache:
                        del self._video_content_cache[oldest_url]

                # Store the BytesIO object directly; it will be seek(0)'d when retrieved
                self._video_content_cache[video_url_lower] = video_data
                self._cache_queue.put(video_url_lower)

            return video_data

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error {e.response.status_code} loading video {video_url}: {e.response.text[:200]}"
            )
            raise ValueError(
                f"Failed to download video {video_url}: HTTP {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            logger.error(f"Request error loading video {video_url}: {e}")
            raise ValueError(f"Network request failed for video {video_url}") from e
        except FileNotFoundError as e:
            logger.error(f"File error loading video {video_url}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Error loading video content from {video_url}: {type(e).__name__} - {e}"
            )
            raise ValueError(f"Failed to load video content: {e}") from e

    async def generate(
        self, request: vLLMMultimodalRequest
    ) -> AsyncIterator[MyRequestOutput]:
        logger.debug(f"Got raw request: {request}")
        if not isinstance(request, vLLMMultimodalRequest):
            if isinstance(request, str):
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)
        logger.debug(f"Received encode request: {{ id: {request.request_id} }}.")

        request_id = request.request_id
        video_url = request.multimodal_input.video_url

        if video_url is None:
            raise ValueError("Video URL is required.")

        container: Optional[av.container.InputContainer] = None

        try:
            video_content_stream = await self._load_video_content(video_url)

            # Open video container using utility function
            container = await open_video_container(video_content_stream, video_url)

            if not container or not container.streams.video:
                logger.error(f"No video stream found in {video_url}.")
                raise ValueError(f"No video stream in {video_url}.")

            # Get video metadata using utility function
            total_frames, duration_sec = get_video_metadata(container)

            # Calculate frame sampling indices using utility function
            indices = calculate_frame_sampling_indices(
                total_frames, self.num_frames_to_sample, duration_sec, video_url
            )

            if not container:
                raise ValueError(f"Container is None for {video_url}")

            # Decode video frames
            clip_np: np.ndarray = await self._read_video_pyav(container, indices)

            if clip_np.size == 0:
                raise ValueError(
                    f"Failed to extract any video frames from {video_url} for indices {indices.tolist()}. Clip is empty."
                )

            logger.info(
                f"Successfully extracted {len(clip_np) if clip_np.ndim > 1 and clip_np.shape[0] > 0 else 0} frames for {video_url} with original shape {clip_np.shape}."
            )

            # Convert the NumPy array from the video decoder into a PyTorch tensor.
            # This is a required step to use PyTorch functions for GPU-accelerated image processing.
            frames_tensor_orig_res = torch.from_numpy(clip_np)  # Shape: (T, H, W, C)

            # Resize frames using utility function
            resized_frames_tensor_hwc = resize_video_frames(
                frames_tensor_orig_res, self.frame_height, self.frame_width
            )

            # Prepare tensor for RDMA using utility function
            tensor_for_descriptor = prepare_tensor_for_rdma(
                resized_frames_tensor_hwc, request_id
            )

            request.embeddings_shape = tuple(tensor_for_descriptor.shape)
            descriptor = connect.Descriptor(tensor_for_descriptor)

            with self._connector.create_readable(descriptor) as readable:
                request.serialized_request = readable.metadata()
                # Clear the image URL as hint that the image is passed as embeddings.
                request.multimodal_input.video_url = None

                logger.debug(f"Request: {request.model_dump_json()}")

                # Get the response generator
                response_generator = await self.pd_worker_client.round_robin(
                    request.model_dump_json()
                )
                await readable.wait_for_completion()

                async for response in response_generator:
                    output = MyRequestOutput.model_validate_json(response.data())
                    yield MyRequestOutput(
                        request_id=output.request_id,
                        prompt=output.prompt,
                        prompt_token_ids=output.prompt_token_ids,
                        prompt_logprobs=output.prompt_logprobs,
                        outputs=output.outputs,
                        finished=output.finished,
                    ).model_dump_json()
        except (
            FileNotFoundError,
            av.FFmpegError,
            ValueError,
        ) as e:
            logger.error(
                f"Error processing request {request_id} ({video_url[:100]}...): {type(e).__name__} - {e}"
            )
            raise  # Re-raise to be handled by the service framework
        except Exception as e:
            logger.exception(
                f"Unexpected error processing request {request_id} ({video_url[:100]}...): {e}"
            )
            raise
        finally:
            if container:
                await asyncio.to_thread(container.close)

    async def _init_http_client(self):
        if (
            not self._http_client or self._http_client.is_closed
        ):  # Check if closed as well
            self._http_client = httpx.AsyncClient(
                timeout=self._http_timeout, follow_redirects=True
            )
            logger.info("HTTP client (re)initialized.")

    async def async_init(self, runtime: DistributedRuntime):
        logger.info("Startup started.")
        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()
        await self._connector.initialize()
        await self._init_http_client()

        logger.info("Startup completed.")

    @classmethod
    def parse_args(cls) -> Tuple[argparse.Namespace, Config]:
        DEFAULT_ENDPOINT = "dyn://dynamo.encoder.generate"
        DEFAULT_DOWNSTREAM_ENDPOINT = "dyn://dynamo.llm.generate"

        parser = FlexibleArgumentParser(
            description="vLLM based encoder for Dynamo LLM."
        )
        parser.add_argument(
            "--endpoint",
            type=str,
            default=DEFAULT_ENDPOINT,
            help=f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Default: '{DEFAULT_ENDPOINT}'",
        )
        parser.add_argument(
            "--downstream-endpoint",
            type=str,
            default=DEFAULT_DOWNSTREAM_ENDPOINT,
            help=f"The endpoint string of the downstream LLM in 'dyn://namespace.component.endpoint' format. Default: '{DEFAULT_DOWNSTREAM_ENDPOINT}'",
        )

        args, config = base_parse_args(parser)

        return args, config


async def graceful_shutdown(runtime):
    """
    By calling `runtime.shutdown()`, the endpoints will immediately be unavailable.
    However, in-flight requests will still be processed until they are finished.
    After all in-flight requests are finished, the `serve_endpoint` functions will return
    and the engine will be shutdown by Python's garbage collector.
    """
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    # Runtime setup
    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    # worker setup
    args, config = VllmEncodeWorker.parse_args()
    await init(runtime, args, config)


async def init(runtime: DistributedRuntime, args: argparse.Namespace, config: Config):
    """
    Instantiate and serve
    """

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)

    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        args.downstream_endpoint
    )
    pd_worker_client = (
        await runtime.namespace(parsed_namespace)
        .component(parsed_component_name)
        .endpoint(parsed_endpoint_name)
        .client()
    )

    handler = VllmEncodeWorker(args, config.engine_args, pd_worker_client)
    await handler.async_init(runtime)

    logger.info("Waiting for PD Worker Instances ...")
    await pd_worker_client.wait_for_instances()

    logger.info(f"Starting to serve the {args.endpoint} endpoint...")

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(handler.generate),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
