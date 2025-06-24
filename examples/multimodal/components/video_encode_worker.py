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
import base64
import binascii
import json
import logging
import os
from io import BytesIO
from queue import Queue
from typing import AsyncIterator, Optional
from urllib.parse import urlparse

import av
import connect
import httpx
import numpy as np
import torch
import torch.nn.functional as F
from utils.protocol import EncodeRequest
from utils.vllm import parse_vllm_args

from dynamo.sdk import async_on_start, endpoint, service

logger = logging.getLogger(__name__)

CACHE_SIZE_MAXIMUM = 8


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmEncodeWorker:
    def __init__(self) -> None:
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.num_frames_to_sample = getattr(self.engine_args, "num_sampled_frames", 8)
        self.frame_height = getattr(self.engine_args, "frame_height", 336)
        self.frame_width = getattr(self.engine_args, "frame_width", 336)
        self.frame_channels = getattr(self.engine_args, "frame_channels", 3)
        self.dummy_token_id = getattr(self.engine_args, "dummy_token_id", 0)
        self.video_token_id = getattr(self.engine_args, "video_token_id", 32000)
        self.dummy_tokens_per_frame = getattr(
            self.engine_args, "dummy_tokens_per_frame", 144
        )
        self._video_content_cache: dict[str, BytesIO] = {}
        self._cache_queue: Queue[str] = Queue(maxsize=CACHE_SIZE_MAXIMUM)

        self._http_client: Optional[httpx.AsyncClient] = None
        self._http_timeout = 60.0

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

    @endpoint()
    async def encode(self, request: EncodeRequest) -> AsyncIterator[str]:
        request_id = request.request_id
        video_url = getattr(request, "video_url", None)
        if not video_url:
            logger.error(f"Request {request_id}: 'video_url' not provided.")
            raise ValueError("'video_url' is required for encoding.")
        if request.serialized_request is None:
            logger.error(
                f"Request serialized_request is None for request: {{ id: {request_id} }}."
            )
            raise ValueError("'serialized_request' is required for encoding.")

        logger.info(
            f"Received encode request: {{ id: {request_id}, video_url: '{video_url[:100]}...' }}"
        )

        container: Optional[av.container.InputContainer] = None
        try:
            video_content_stream = await self._load_video_content(video_url)

            def open_video_container_sync():
                try:
                    return av.open(video_content_stream, mode="r")
                except av.FFmpegError as ave:
                    logger.error(
                        f"PyAV error opening video stream from {video_url}: {ave}"
                    )
                    raise ValueError(
                        f"Invalid video format or corrupted data from {video_url}."
                    ) from ave
                except Exception as e:
                    logger.error(
                        f"Unexpected error opening video stream from {video_url} with PyAV: {e}"
                    )
                    raise ValueError(
                        f"Unexpected error opening video from {video_url}."
                    ) from e

            container = await asyncio.to_thread(open_video_container_sync)

            if not container or not container.streams.video:
                logger.error(f"No video stream found in {video_url}.")
                raise ValueError(f"No video stream in {video_url}.")

            stream_info = container.streams.video[0]
            total_frames = stream_info.frames
            # Duration can be useful for streams where total_frames is 0
            if stream_info.duration and stream_info.time_base:
                duration_sec = float(stream_info.duration * stream_info.time_base)
            else:
                duration_sec = 0

            if total_frames == 0 and duration_sec == 0:
                logger.error(f"Video file '{video_url}' has 0 frames and 0 duration.")
                raise ValueError(f"Video {video_url} has 0 frames and 0 duration.")
            if total_frames == 0 and duration_sec > 0:
                logger.warning(
                    f"Video {video_url} reports 0 frames but has duration {duration_sec:.2f}s. Frame sampling may be based on requested count directly."
                )

            logger.debug(
                f"Video {video_url} has {total_frames} frames (duration: {duration_sec:.2f}s). Sampling {self.num_frames_to_sample} frames."
            )
            indices: np.ndarray
            if total_frames > 0:
                if total_frames < self.num_frames_to_sample:
                    logger.warning(
                        f"Video frames ({total_frames}) < samples ({self.num_frames_to_sample}). Using all {total_frames} available frames."
                    )
                    indices = np.arange(0, total_frames).astype(int)
                else:
                    indices = np.linspace(
                        0, total_frames - 1, self.num_frames_to_sample, dtype=int
                    )
            else:  # total_frames is 0 (likely a stream), sample by count.
                logger.warning(
                    f"Video {video_url} frame count is 0. Attempting to sample {self.num_frames_to_sample} frames by index. This might fail if stream is too short."
                )
                indices = np.arange(0, self.num_frames_to_sample).astype(int)

            # Ensure indices are unique, especially after linspace for small numbers.
            indices = np.unique(indices)

            if (
                len(indices) == 0 and total_frames > 0
            ):  # Safety for linspace oddities with few frames
                # If unique resulted in empty but there are frames, sample at least one or up to num_frames_to_sample
                actual_samples = min(self.num_frames_to_sample, total_frames)
                indices = np.arange(0, actual_samples).astype(int)
            elif len(indices) == 0 and total_frames == 0:
                # If indices is empty and total_frames is 0, this means num_frames_to_sample might be 0 or indices logic failed.
                # This case implies we might not be able to sample any frames.
                # _read_video_pyav handles empty indices with non-empty video by trying to read the first frame.
                # If indices is empty here due to num_frames_to_sample=0, _read_video_pyav will return empty.
                pass  # Let _read_video_pyav handle this.

            logger.info(f"Selected frame indices for {video_url}: {indices.tolist()}")

            if not container:
                raise ValueError(f"Container is None for {video_url}")

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

            # Permute to (T, C, H, W) for interpolate
            frames_tensor_chw = frames_tensor_orig_res.permute(
                0, 3, 1, 2
            ).float()  # Ensure float for interpolate

            # Resize
            resized_frames_tensor_chw = F.interpolate(
                frames_tensor_chw,
                size=(self.frame_height, self.frame_width),
                mode="bilinear",
                align_corners=False,
            )

            # Permute back to (T, H_new, W_new, C)
            resized_frames_tensor_hwc = resized_frames_tensor_chw.permute(0, 2, 3, 1)

            logger.debug(f"Resized frames to shape: {resized_frames_tensor_hwc.shape}")

            # Ensure the tensor is contiguous, on CUDA and uint8 for the NIXL buffer.
            tensor_for_descriptor: torch.Tensor = resized_frames_tensor_hwc.to(
                device="cpu", dtype=torch.uint8
            ).contiguous()

            logger.info(
                f"Req {request_id}: Preparing raw frames tensor (shape: {tensor_for_descriptor.shape}, "
                f"dtype: {tensor_for_descriptor.dtype}, device: {tensor_for_descriptor.device}, "
                f"contiguous: {tensor_for_descriptor.is_contiguous()}) for RDMA."
            )

            # Create a descriptor for the tensor to be sent via the connector.
            descriptor = connect.Descriptor(tensor_for_descriptor)
            logger.info(f"Req {request_id}: Beginning connector write operation.")
            # Pass the remote worker's SerializedRequest (representing its WritableOperation) to begin_write.
            # This initiates the data transfer to the memory buffer on the other worker.
            write_op = await self._connector.begin_write(
                descriptor, request.serialized_request
            )
            # Wait for the RDMA/transfer operation to complete.
            await write_op.wait_for_completion()
            logger.info(f"Req {request_id}: Connector write operation completed.")

            # Yield a simplified response, assuming EncodeResponse in protocol is adapted
            final_response_data = {
                "request_id": request.request_id,
            }
            yield json.dumps(final_response_data)
            logger.info(f"Encode request {request_id} processed successfully.")

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
            self._http_client = httpx.AsyncClient(timeout=self._http_timeout)
            logger.info("HTTP client (re)initialized.")

    @async_on_start
    async def async_init(self):
        logger.info(f"{self.__class__.__name__} async_init started.")
        # Initialize the connector for RDMA transfers.
        self._connector = connect.Connector()
        await self._connector.initialize()
        logger.info("Dynamo connector initialized.")
        await self._init_http_client()
        logger.info(
            f"{self.__class__.__name__} async_init completed. Ready to encode video frames."
        )
