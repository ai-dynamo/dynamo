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
import logging
import os
from typing import Any, Awaitable, Dict, Final, List
from urllib.parse import urlparse

import numpy as np

from dynamo.common.http import fetch_bytes
from dynamo.common.http.url_validator import UrlValidationPolicy, validate_media_url
from dynamo.common.utils.runtime import run_async

logger = logging.getLogger(__name__)


URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"
UUID_ONLY_VARIANT_KEY: Final = "UuidOnly"


def _create_nixl_connector() -> Any:
    try:
        import dynamo.nixl_connect as nixl_connect
    except ImportError as exc:
        raise RuntimeError(
            "NIXL is required for frontend video decoding; install "
            "dynamo.nixl_connect to enable decoded video transfers."
        ) from exc

    return nixl_connect.Connector()


async def read_decoded_media_via_nixl(*args: Any, **kwargs: Any) -> Any:
    try:
        from dynamo.common.utils.media_nixl import (
            read_decoded_media_via_nixl as _read_decoded_media_via_nixl,
        )
    except ImportError as exc:
        raise RuntimeError(
            "NIXL media utilities are required for frontend video decoding."
        ) from exc

    return await _read_decoded_media_via_nixl(*args, **kwargs)


def _require_vllm_video_media() -> tuple[Any, Any, Any]:
    try:
        from vllm.multimodal.media import MediaConnector, VideoMediaIO
        from vllm.multimodal.media.image import ImageMediaIO
    except ImportError as exc:
        raise RuntimeError(
            "vLLM multimodal media components are required to decode `video_url` "
            "inputs in the vLLM backend."
        ) from exc
    return MediaConnector, VideoMediaIO, ImageMediaIO


class VideoLoader:
    NUM_FRAMES_DEFAULT = int(os.environ.get("DYN_MM_VIDEO_NUM_FRAMES", "32"))

    def __init__(
        self,
        http_timeout: float = 60.0,
        num_frames: int = NUM_FRAMES_DEFAULT,
        enable_frontend_decoding: bool = False,
        url_policy: UrlValidationPolicy | None = None,
    ) -> None:
        self._http_timeout = int(http_timeout)
        self._num_frames = num_frames
        self._enable_frontend_decoding = enable_frontend_decoding
        self._url_policy = url_policy or UrlValidationPolicy.from_env()
        self._nixl_connector = None
        self._vllm_media_connector = None
        if self._enable_frontend_decoding:
            self._nixl_connector = _create_nixl_connector()
            run_async(self._nixl_connector.initialize)

    def _get_vllm_media_connector(self) -> Any:
        if self._vllm_media_connector is None:
            MediaConnector, _, _ = _require_vllm_video_media()
            # Confine vLLM's own local-path access to the same prefix we enforce.
            # Empty string matches vLLM's secure default (no local access).
            allowed = self._url_policy.allowed_local_path or ""
            self._vllm_media_connector = MediaConnector(
                allowed_local_media_path=allowed
            )

        return self._vllm_media_connector

    def _create_vllm_video_io(self) -> Any:
        _, VideoMediaIO, ImageMediaIO = _require_vllm_video_media()
        return VideoMediaIO(
            ImageMediaIO(image_mode="RGB"),
            num_frames=self._num_frames,
        )

    async def _load_video_with_vllm(
        self, video_url: str
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        normalized_url = await validate_media_url(video_url, self._url_policy)
        media_io = self._create_vllm_video_io()

        # HTTP(S) goes through our SSRF-safe fetcher so each redirect hop is
        # revalidated; vLLM's own fetcher honors redirects without re-checking.
        # data: and file:// never touch the network, so vLLM can handle them.
        if urlparse(normalized_url).scheme in ("http", "https"):
            content = await fetch_bytes(
                normalized_url, self._http_timeout, policy=self._url_policy
            )
            return await asyncio.to_thread(media_io.load_bytes, content)

        connector = self._get_vllm_media_connector()
        return await connector.load_from_url_async(
            normalized_url, media_io, fetch_timeout=self._http_timeout
        )

    async def load_video(self, video_url: str) -> tuple[np.ndarray, Dict[str, Any]]:
        try:
            frames, metadata = await self._load_video_with_vllm(video_url)
            if frames.size == 0:
                raise ValueError(
                    f"Failed to extract video frames from {video_url}. Decoded clip is empty."
                )
            return np.ascontiguousarray(frames), metadata
        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.error("Error loading video from %s: %s", video_url, exc)
            raise ValueError(f"Failed to load video from {video_url}: {exc}") from exc

    async def _load_decoded_video(
        self, decoded_metadata: Dict[str, Any]
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        if self._nixl_connector is None:
            raise RuntimeError("NIXL connector is not initialized")

        frames, metadata = await read_decoded_media_via_nixl(
            self._nixl_connector,
            decoded_metadata,
            return_metadata=True,
        )
        if metadata is None:
            raise ValueError("Decoded video metadata is required")

        return np.ascontiguousarray(frames), metadata

    async def load_video_batch(
        self,
        video_mm_items: List[Dict[str, Any]],
    ) -> List[tuple[np.ndarray, Dict[str, Any]] | None]:
        video_futures: List[Awaitable[tuple[np.ndarray, Dict[str, Any]]]] = []
        slot_to_future_idx: list[int | None] = []

        for idx, item in enumerate(video_mm_items):
            if isinstance(item, dict) and URL_VARIANT_KEY in item:
                url = item[URL_VARIANT_KEY]
                slot_to_future_idx.append(len(video_futures))
                video_futures.append(self.load_video(url))
                logger.debug("Preparing to load video from URL: %s...", url[:80])
            elif isinstance(item, dict) and DECODED_VARIANT_KEY in item:
                if self._enable_frontend_decoding:
                    metadata = item[DECODED_VARIANT_KEY]
                    slot_to_future_idx.append(len(video_futures))
                    video_futures.append(self._load_decoded_video(metadata))
                else:
                    raise ValueError(
                        "Received decoded video data but enable_frontend_decoding=False. "
                        "Enable frontend decoding to transfer decoded video frames via NIXL."
                    )
            elif isinstance(item, dict) and UUID_ONLY_VARIANT_KEY in item:
                slot_to_future_idx.append(None)
            else:
                raise ValueError(
                    f"Invalid video multimodal item at index {idx}. "
                    "Expected dict with 'Url', 'Decoded', or 'UuidOnly' key."
                )

        results = await asyncio.gather(*video_futures, return_exceptions=True)
        loaded_videos: list[tuple[np.ndarray, Dict[str, Any]] | None] = []
        collective_exceptions: list[str] = []
        for media_item, future_idx in zip(
            video_mm_items, slot_to_future_idx, strict=True
        ):
            if future_idx is None:
                loaded_videos.append(None)
                continue
            result = results[future_idx]
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                source = media_item.get(URL_VARIANT_KEY, "decoded")
                logger.error("Failed to load video from %s...: %s", source[:80], result)
                collective_exceptions.append(
                    f"Failed to load video from {source[:80]}...: {result}\n"
                )
                continue
            frames, metadata = result
            loaded_videos.append((np.ascontiguousarray(frames), metadata))

        if collective_exceptions:
            raise Exception("".join(collective_exceptions))

        return loaded_videos
