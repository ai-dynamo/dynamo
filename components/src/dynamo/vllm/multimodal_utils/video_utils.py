# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
import os
from io import BytesIO
from queue import Queue
from typing import Tuple
from urllib.parse import urlparse

import av
import httpx
import numpy as np
import torch
import torch.nn.functional as F

from .http_client import get_http_client

logger = logging.getLogger(__name__)


async def load_video_content(
    video_url: str,
    video_content_cache: dict[str, BytesIO],
    cache_queue: Queue[str],
    http_timeout: float = 60.0,
) -> BytesIO:
    """
    Load video content from various sources (URL, data URI, file).

    Args:
        video_url: The video URL or path
        video_content_cache: Cache dictionary for storing downloaded content
        cache_queue: Queue for managing cache eviction
        http_timeout: Timeout for HTTP requests

    Returns:
        BytesIO stream containing video data

    Raises:
        ValueError: If video source is unsupported or invalid
        FileNotFoundError: If local file doesn't exist
        RuntimeError: If HTTP client initialization fails
    """
    parsed_url = urlparse(video_url)
    video_url_lower = video_url.lower()

    if parsed_url.scheme in ("http", "https"):
        if video_url_lower in video_content_cache:
            logger.debug(f"Video content found in cache for URL: {video_url}")
            cached_content = video_content_cache[video_url_lower]
            cached_content.seek(0)
            return cached_content

    try:
        video_data: BytesIO
        if parsed_url.scheme == "data":
            if not parsed_url.path.startswith(("video/", "application/octet-stream")):
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
                raise ValueError(f"Invalid base64 encoding for video data: {e}") from e

        elif parsed_url.scheme in ("http", "https"):
            http_client = get_http_client(http_timeout)

            logger.debug(f"Downloading video from URL: {video_url}")
            response = await http_client.get(video_url, timeout=http_timeout)
            response.raise_for_status()

            if not response.content:
                raise ValueError(f"Empty response content from video URL: {video_url}")
            video_data = BytesIO(response.content)
            video_data.seek(0)
            logger.debug(
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
            if cache_queue.full():
                oldest_url = cache_queue.get_nowait()
                if oldest_url in video_content_cache:
                    del video_content_cache[oldest_url]

            # Store the BytesIO object directly; it will be seek(0)'d when retrieved
            video_content_cache[video_url_lower] = video_data
            cache_queue.put(video_url_lower)

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


async def open_video_container(
    video_content_stream: BytesIO, video_url: str
) -> av.container.InputContainer:
    """
    Open a video container from a BytesIO stream using PyAV.

    Args:
        video_content_stream: BytesIO stream containing video data
        video_url: Original video URL for error reporting

    Returns:
        Opened PyAV container

    Raises:
        ValueError: If video format is invalid or corrupted
    """

    def open_video_container_sync():
        try:
            return av.open(video_content_stream, mode="r")
        except av.FFmpegError as ave:
            logger.error(f"PyAV error opening video stream from {video_url}: {ave}")
            raise ValueError(
                f"Invalid video format or corrupted data from {video_url}."
            ) from ave
        except Exception as e:
            logger.error(
                f"Unexpected error opening video stream from {video_url} with PyAV: {e}"
            )
            raise ValueError(f"Unexpected error opening video from {video_url}.") from e

    return await asyncio.to_thread(open_video_container_sync)


def get_video_metadata(container: av.container.InputContainer) -> Tuple[int, float]:
    """
    Extract metadata from video container.

    Args:
        container: Opened PyAV container

    Returns:
        Tuple of (total_frames, duration_in_seconds)
    """
    if not container or not container.streams.video:
        return 0, 0.0

    stream_info = container.streams.video[0]
    total_frames = stream_info.frames

    # Duration can be useful for streams where total_frames is 0
    if stream_info.duration and stream_info.time_base:
        duration_sec = float(stream_info.duration * stream_info.time_base)
    else:
        duration_sec = 0.0

    return total_frames, duration_sec


async def read_video_pyav(
    container: av.container.InputContainer, indices: np.ndarray
) -> np.ndarray:
    """
    Decode the video with PyAV decoder. Async wrapper.

    Args:
        container: The video container to decode from
        indices: Frame indices to extract

    Returns:
        NumPy array of decoded frames

    Raises:
        ValueError: If no frames could be decoded for the given indices
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
                "read_video_pyav called with empty indices for a non-empty video, attempting to read first frame."
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


def calculate_frame_sampling_indices(
    total_frames: int,
    num_frames_to_sample: int,
    duration_sec: float = 0,
    video_url: str = "",
) -> np.ndarray:
    """
    Calculate frame indices to sample from a video.

    Args:
        total_frames: Total number of frames in the video
        num_frames_to_sample: Number of frames to sample
        duration_sec: Duration of video in seconds (for logging)
        video_url: Video URL for logging purposes

    Returns:
        Array of frame indices to sample

    Raises:
        ValueError: If video has 0 frames and 0 duration
    """
    if total_frames == 0 and duration_sec == 0:
        logger.error(f"Video file '{video_url}' has 0 frames and 0 duration.")
        raise ValueError(f"Video {video_url} has 0 frames and 0 duration.")

    if total_frames == 0 and duration_sec > 0:
        logger.warning(
            f"Video {video_url} reports 0 frames but has duration {duration_sec:.2f}s. "
            "Frame sampling may be based on requested count directly."
        )

    logger.debug(
        f"Video {video_url} has {total_frames} frames (duration: {duration_sec:.2f}s). "
        f"Sampling {num_frames_to_sample} frames."
    )

    indices: np.ndarray
    if total_frames > 0:
        if total_frames < num_frames_to_sample:
            logger.warning(
                f"Video frames ({total_frames}) < samples ({num_frames_to_sample}). "
                f"Using all {total_frames} available frames."
            )
            indices = np.arange(0, total_frames).astype(int)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
    else:  # total_frames is 0 (likely a stream), sample by count.
        logger.warning(
            f"Video {video_url} frame count is 0. Attempting to sample {num_frames_to_sample} "
            "frames by index. This might fail if stream is too short."
        )
        indices = np.arange(0, num_frames_to_sample).astype(int)

    # Ensure indices are unique, especially after linspace for small numbers.
    indices = np.unique(indices)

    # Safety checks for edge cases
    if len(indices) == 0 and total_frames > 0:
        # If unique resulted in empty but there are frames, sample at least one
        actual_samples = min(num_frames_to_sample, total_frames)
        indices = np.arange(0, actual_samples).astype(int)
    elif len(indices) == 0 and total_frames == 0:
        # If indices is empty and total_frames is 0, let downstream handle this case
        pass

    logger.debug(f"Selected frame indices for {video_url}: {indices.tolist()}")
    return indices


def resize_video_frames(
    frames_tensor: torch.Tensor, target_height: int, target_width: int
) -> torch.Tensor:
    """
    Resize video frames using PyTorch interpolation.

    Args:
        frames_tensor: Input tensor with shape (T, H, W, C)
        target_height: Target frame height
        target_width: Target frame width

    Returns:
        Resized tensor with shape (T, target_height, target_width, C)
    """
    # Permute to (T, C, H, W) for interpolate
    frames_tensor_chw = frames_tensor.permute(0, 3, 1, 2).float()

    # Resize
    resized_frames_tensor_chw = F.interpolate(
        frames_tensor_chw,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    # Permute back to (T, H_new, W_new, C)
    resized_frames_tensor_hwc = resized_frames_tensor_chw.permute(0, 2, 3, 1)

    logger.debug(f"Resized frames to shape: {resized_frames_tensor_hwc.shape}")
    return resized_frames_tensor_hwc


def prepare_tensor_for_rdma(
    frames_tensor: torch.Tensor, request_id: str
) -> torch.Tensor:
    """
    Prepare video frames tensor for RDMA transfer.

    Args:
        frames_tensor: Input frames tensor
        request_id: Request ID for logging

    Returns:
        Tensor prepared for RDMA (CPU, uint8, contiguous)
    """
    # Ensure the tensor is contiguous, on CPU and uint8 for the NIXL buffer.
    tensor_for_descriptor = frames_tensor.to(
        device="cpu", dtype=torch.uint8
    ).contiguous()

    logger.debug(
        f"Req {request_id}: Preparing raw frames tensor (shape: {tensor_for_descriptor.shape}, "
        f"dtype: {tensor_for_descriptor.dtype}, device: {tensor_for_descriptor.device}, "
        f"contiguous: {tensor_for_descriptor.is_contiguous()}) for RDMA."
    )

    return tensor_for_descriptor
