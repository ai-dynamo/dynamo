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
from typing import Any, Dict, Final, List
from urllib.parse import urlparse

import httpx
import numpy as np

import dynamo.nixl_connect as nixl_connect
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.common.utils.media_nixl import read_decoded_media_via_nixl
from dynamo.common.utils.runtime import run_async

from .http_client import get_http_client

logger = logging.getLogger(__name__)


URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"


def _require_av() -> Any:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is required to decode `video_url` inputs in the vLLM backend. "
            "Install the `av` package with FFmpeg support, or enable frontend "
            "media decoding so requests arrive as decoded video frames."
        ) from exc
    return av


class VideoLoader:
    CACHE_SIZE_MAXIMUM = int(os.environ.get("DYN_MM_VIDEO_CACHE_SIZE", "8"))
    NUM_FRAMES_DEFAULT = int(os.environ.get("DYN_MM_VIDEO_NUM_FRAMES", "8"))

    def __init__(
        self,
        cache_size: int = CACHE_SIZE_MAXIMUM,
        http_timeout: float = 60.0,
        num_frames: int = NUM_FRAMES_DEFAULT,
        enable_frontend_decoding: bool = False,
    ) -> None:
        self._http_timeout = http_timeout
        self._num_frames = num_frames
        self._video_content_cache: dict[str, bytes] = {}
        self._cache_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=cache_size)
        self._enable_frontend_decoding = enable_frontend_decoding
        self._nixl_connector = None
        if self._enable_frontend_decoding:
            self._nixl_connector = nixl_connect.Connector()
            run_async(self._nixl_connector.initialize)

    @staticmethod
    def _get_stream_fps(stream_info: Any) -> float:
        for attr in ("average_rate", "base_rate", "guessed_rate"):
            rate = getattr(stream_info, attr, None)
            if rate:
                try:
                    fps = float(rate)
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
                if fps > 0:
                    return fps
        return 0.0

    async def _read_local_video_file(self, file_path: str) -> bytes:
        def _read() -> bytes:
            with open(file_path, "rb") as f:
                return f.read()

        return await asyncio.to_thread(_read)

    async def _load_video_content(self, video_url: str) -> BytesIO:
        parsed_url = urlparse(video_url)
        video_url_lower = video_url.lower()

        if parsed_url.scheme in ("http", "https"):
            cached_bytes = self._video_content_cache.get(video_url_lower)
            if cached_bytes is not None:
                logger.debug("Video content found in cache for URL: %s", video_url)
                return BytesIO(cached_bytes)

        try:
            video_bytes: bytes
            if parsed_url.scheme == "data":
                if not parsed_url.path.startswith(("video/", "application/octet-stream")):
                    raise ValueError("Data URL must be a video type or octet-stream")
                media_type_and_data = parsed_url.path.split(",", 1)
                if len(media_type_and_data) != 2:
                    raise ValueError("Invalid data URL format: missing comma separator")
                media_type, data_segment = media_type_and_data
                if ";base64" not in media_type:
                    raise ValueError("Video data URLs must be base64 encoded")
                try:
                    video_bytes = base64.b64decode(data_segment)
                except binascii.Error as exc:
                    raise ValueError(
                        f"Invalid base64 encoding for video data: {exc}"
                    ) from exc
            elif parsed_url.scheme in ("http", "https"):
                http_client = get_http_client(self._http_timeout)
                response = await http_client.get(video_url, timeout=self._http_timeout)
                response.raise_for_status()
                if not response.content:
                    raise ValueError(f"Empty response content from video URL: {video_url}")
                video_bytes = response.content
            elif parsed_url.scheme in ("", "file"):
                file_path = video_url if parsed_url.scheme == "" else parsed_url.path
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Error reading file: {file_path}")
                video_bytes = await self._read_local_video_file(file_path)
            else:
                raise ValueError(
                    f"Unsupported video source scheme: {parsed_url.scheme} for URL {video_url}"
                )

            if parsed_url.scheme in ("http", "https"):
                if self._cache_queue.full():
                    oldest_url = await self._cache_queue.get()
                    self._video_content_cache.pop(oldest_url, None)
                self._video_content_cache[video_url_lower] = video_bytes
                await self._cache_queue.put(video_url_lower)

            return BytesIO(video_bytes)
        except httpx.HTTPStatusError as exc:
            logger.error(
                "HTTP error %s loading video %s: %s",
                exc.response.status_code,
                video_url,
                exc.response.text[:200],
            )
            raise ValueError(
                f"Failed to download video {video_url}: HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error("Request error loading video %s: %s", video_url, exc)
            raise ValueError(f"Network request failed for video {video_url}") from exc
        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.error(
                "Error loading video content from %s: %s - %s",
                video_url,
                type(exc).__name__,
                exc,
            )
            raise ValueError(f"Failed to load video content: {exc}") from exc

    async def _open_video_container(
        self, video_content_stream: BytesIO, video_url: str
    ) -> Any:
        av = _require_av()

        def _open() -> Any:
            try:
                return av.open(video_content_stream, mode="r")
            except av.FFmpegError as exc:
                logger.error("PyAV error opening video stream from %s: %s", video_url, exc)
                raise ValueError(
                    f"Invalid video format or corrupted data from {video_url}."
                ) from exc
            except Exception as exc:
                logger.error(
                    "Unexpected error opening video stream from %s with PyAV: %s",
                    video_url,
                    exc,
                )
                raise ValueError(f"Unexpected error opening video from {video_url}.") from exc

        return await asyncio.to_thread(_open)

    @staticmethod
    def _get_video_metadata(container: Any) -> tuple[int, float, float]:
        if not container or not container.streams.video:
            return 0, 0.0, 0.0

        stream_info = container.streams.video[0]
        total_frames = int(stream_info.frames or 0)
        if stream_info.duration and stream_info.time_base:
            duration_sec = float(stream_info.duration * stream_info.time_base)
        else:
            duration_sec = 0.0
        fps = VideoLoader._get_stream_fps(stream_info)
        if fps <= 0 and duration_sec > 0 and total_frames > 0:
            fps = total_frames / duration_sec

        return total_frames, duration_sec, fps

    @staticmethod
    def _calculate_frame_sampling_indices(
        total_frames: int,
        num_frames_to_sample: int,
        duration_sec: float = 0,
        video_url: str = "",
    ) -> np.ndarray:
        if total_frames == 0 and duration_sec == 0:
            logger.error("Video file '%s' has 0 frames and 0 duration.", video_url)
            raise ValueError(f"Video {video_url} has 0 frames and 0 duration.")

        if total_frames > 0:
            actual_samples = min(num_frames_to_sample, total_frames)
            indices = np.linspace(0, total_frames - 1, actual_samples, dtype=int)
        else:
            logger.warning(
                "Video %s frame count is 0. Attempting to sample %d frames by index.",
                video_url,
                num_frames_to_sample,
            )
            indices = np.arange(0, num_frames_to_sample).astype(int)

        return np.unique(indices)

    async def _read_video_frames(self, container: Any, indices: np.ndarray) -> np.ndarray:
        def _decode() -> np.ndarray:
            container.seek(0)
            processed_indices = set(indices.tolist())
            if not processed_indices:
                return np.array([])

            min_idx = min(processed_indices)
            max_idx = max(processed_indices)
            decoded_frames = []
            for frame_idx, frame in enumerate(container.decode(video=0)):
                if frame_idx > max_idx:
                    break
                if frame_idx >= min_idx and frame_idx in processed_indices:
                    decoded_frames.append(frame.to_ndarray(format="rgb24"))

            if not decoded_frames:
                raise ValueError(
                    f"Could not decode any frames for indices: {indices.tolist()}."
                )

            return np.stack(decoded_frames)

        return await asyncio.to_thread(_decode)

    @_nvtx.annotate("mm:video:load_video", color="cyan")
    @staticmethod
    def _build_video_metadata(
        *,
        total_frames: int,
        duration_sec: float,
        fps: float,
        indices: np.ndarray,
        decoded_num_frames: int,
    ) -> Dict[str, Any]:
        if total_frames <= 0:
            total_frames = max(decoded_num_frames, int(indices[-1]) + 1 if len(indices) else 0)
        if fps <= 0:
            if duration_sec > 0 and total_frames > 0:
                fps = total_frames / duration_sec
            else:
                # Qwen3-VL requires positive fps metadata for timestamp calculation.
                fps = 1.0
                logger.warning(
                    "Could not determine video FPS from container metadata; falling back to 1.0 FPS"
                )
        if duration_sec <= 0 and total_frames > 0 and fps > 0:
            duration_sec = total_frames / fps

        return {
            "fps": float(fps),
            "duration": float(duration_sec),
            "total_num_frames": int(total_frames),
            "frames_indices": [int(idx) for idx in indices.tolist()],
            "video_backend": "pyav",
            "do_sample_frames": False,
        }

    async def load_video(self, video_url: str) -> tuple[np.ndarray, Dict[str, Any]]:
        container = None
        try:
            video_content_stream = await self._load_video_content(video_url)
            container = await self._open_video_container(video_content_stream, video_url)

            if not container or not container.streams.video:
                raise ValueError(f"No video stream in {video_url}.")

            total_frames, duration_sec, fps = self._get_video_metadata(container)
            indices = self._calculate_frame_sampling_indices(
                total_frames, self._num_frames, duration_sec, video_url
            )
            frames = await self._read_video_frames(container, indices)
            if frames.size == 0:
                raise ValueError(
                    f"Failed to extract video frames from {video_url}. Decoded clip is empty."
                )

            frames = np.ascontiguousarray(frames)
            metadata = self._build_video_metadata(
                total_frames=total_frames,
                duration_sec=duration_sec,
                fps=fps,
                indices=indices,
                decoded_num_frames=frames.shape[0],
            )
            return frames, metadata
        finally:
            if container is not None:
                await asyncio.to_thread(container.close)

    async def load_video_batch(
        self,
        video_mm_items: List[Dict[str, Any]],
    ) -> List[tuple[np.ndarray, Dict[str, Any]]]:
        video_futures = []

        for item in video_mm_items:
            if isinstance(item, dict) and URL_VARIANT_KEY in item:
                url = item[URL_VARIANT_KEY]
                video_futures.append(self.load_video(url))
                logger.debug("Preparing to load video from URL: %s...", url[:80])
            elif isinstance(item, dict) and DECODED_VARIANT_KEY in item:
                if self._enable_frontend_decoding:
                    metadata = item[DECODED_VARIANT_KEY]
                    if self._nixl_connector is None:
                        raise RuntimeError("NIXL connector is not initialized")
                    video_futures.append(
                        read_decoded_media_via_nixl(self._nixl_connector, metadata)
                    )
                else:
                    raise ValueError(
                        "Received decoded video data but enable_frontend_decoding=False. "
                        "Enable frontend decoding to transfer decoded video frames via NIXL."
                    )

        results = await asyncio.gather(*video_futures, return_exceptions=True)
        loaded_videos: list[tuple[np.ndarray, Dict[str, Any]]] = []
        collective_exceptions = ""
        for media_item, result in zip(video_mm_items, results):
            if isinstance(result, Exception):
                source = media_item.get(URL_VARIANT_KEY, "decoded")
                logger.error("Failed to load video from %s...: %s", source[:80], result)
                collective_exceptions += (
                    f"Failed to load video from {source[:80]}...: {result}\n"
                )
                continue
            frames, metadata = result
            loaded_videos.append((np.ascontiguousarray(frames), metadata))

        if collective_exceptions:
            raise Exception(collective_exceptions)

        return loaded_videos
