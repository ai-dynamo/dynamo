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
import logging
import os
from io import BytesIO
from queue import Queue
from urllib.parse import urlparse

import av
import httpx
import numpy as np

from .http_client import get_http_client

logger = logging.getLogger(__name__)


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
            logger.info(f"Video content found in cache for URL: {video_url}")
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

            logger.info(f"Downloading video from URL: {video_url}")
            response = await http_client.get(video_url, timeout=http_timeout)
            response.raise_for_status()

            if not response.content:
                raise ValueError(f"Empty response content from video URL: {video_url}")
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
