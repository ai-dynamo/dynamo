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

from dynamo.common.http import HttpStatusError, fetch_bytes
from dynamo.common.http.url_validator import (
    UrlValidationError,
    UrlValidationPolicy,
    validate_media_url,
)
from dynamo.common.multimodal.codec_errors import (
    MissingMediaDecoderError,
    video_decoder_missing,
)
from dynamo.common.multimodal.media_source import (
    is_local_media_url,
    read_local_media_bytes,
)
from dynamo.common.multimodal.nvdec_decoder import (
    decode_video_nvdec,
    probe_video_codec,
    should_use_nvdec,
)
from dynamo.common.utils.runtime import run_async

logger = logging.getLogger(__name__)


URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"


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

    def _create_vllm_video_io(
        self, media_io_kwargs: Dict[str, Any] | None = None
    ) -> Any:
        _, VideoMediaIO, ImageMediaIO = _require_vllm_video_media()
        video_io_kwargs = VideoMediaIO.merge_kwargs(
            {"num_frames": self._num_frames}, media_io_kwargs or {}
        )
        return VideoMediaIO(
            ImageMediaIO(image_mode="RGB"),
            **video_io_kwargs,
        )

    async def _load_video_with_vllm(
        self, video_url: str, media_io_kwargs: Dict[str, Any] | None = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        normalized_url = await validate_media_url(video_url, self._url_policy)
        media_io = self._create_vllm_video_io(media_io_kwargs)

        # HTTP(S) goes through our SSRF-safe fetcher so each redirect hop is
        # revalidated; vLLM's own fetcher honors redirects without re-checking.
        # data: and file:// never touch the network, so vLLM can handle them.
        if urlparse(normalized_url).scheme in ("http", "https"):
            content = await fetch_bytes(
                normalized_url, self._http_timeout, policy=self._url_policy
            )
            return await self._decode_video_bytes(content, media_io)

        # file:// and data: never touch the network, but they still deserve
        # hardware decode: without this they reach only the software decoder,
        # which the codec-compliant images do not ship, so H.264/H.265 from a
        # local file or data URI would fail despite NVDEC being available and
        # able to decode it. Reading is gated by the same url policy the vLLM
        # connector below uses, so this adds no local-read surface.
        if is_local_media_url(normalized_url):
            content = await read_local_media_bytes(normalized_url, self._url_policy)
            return await self._decode_video_bytes(content, media_io)

        connector = self._get_vllm_media_connector()
        return await connector.load_from_url_async(
            normalized_url, media_io, fetch_timeout=self._http_timeout
        )

    async def _decode_video_bytes(
        self, content: bytes, media_io: Any
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Decode video bytes: H.264/H.265 on NVDEC, all else software.

        The runtime images purge the software decode wheels (opencv/av/decord/
        torchcodec) for codec compliance, so the software fallback only
        resolves where a decoder was installed separately. When it is absent,
        vLLM's lazy import surfaces a bare ``No module named 'cv2'`` with no
        codec and no remedy -- convert that into the actionable
        unsupported-codec error, which can name the codec because the probe
        already ran here.

        Request ``media_io_kwargs`` do not apply on the NVDEC path: it
        implements uniform ``num_frames`` sampling only, so ``fps`` and any
        backend-specific keys are ignored for H.264/H.265. Only vLLM's video
        backend owns the full ``media_io_kwargs`` contract, so it is reached
        for royalty-free codecs, or for every clip when
        ``DYN_DISABLE_NVDEC=1``.
        """
        codec = probe_video_codec(content)
        if should_use_nvdec(codec):
            try:
                return await asyncio.to_thread(
                    decode_video_nvdec, content, self._num_frames
                )
            except Exception as exc:  # noqa: BLE001 - fall back to software decode
                logger.warning(
                    "NVDEC decode failed for a %s clip (%s); using software decode",
                    codec,
                    exc,
                )
        try:
            return await asyncio.to_thread(media_io.load_bytes, content)
        except ImportError as exc:
            raise video_decoder_missing(
                "vllm", "opencv-python-headless", "cv2", codec, cause=str(exc)
            ) from exc

    async def load_video(
        self, video_url: str, media_io_kwargs: Dict[str, Any] | None = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        try:
            frames, metadata = await self._load_video_with_vllm(
                video_url, media_io_kwargs
            )
            if frames.size == 0:
                raise ValueError(
                    f"Failed to extract video frames from {video_url}. Decoded clip is empty."
                )
            return np.ascontiguousarray(frames), metadata
        except FileNotFoundError:
            raise
        except (UrlValidationError, HttpStatusError):
            # Preserve deliberate client-error verdicts. UrlValidationError is
            # a ValueError, so the generic handler below would otherwise erase
            # its type and prevent the frontend from returning a 4xx.
            logger.error("URL rejected loading video: '%s'", video_url)
            raise
        except MissingMediaDecoderError:
            # Already actionable (names the codec and the install); a missing
            # decoder is deployment configuration, not a bad request, so keep
            # the type instead of degrading it to the ValueError below.
            logger.error("No decoder available for video: '%s'", video_url)
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
        media_io_kwargs: Dict[str, Any] | None = None,
    ) -> List[tuple[np.ndarray, Dict[str, Any]]]:
        video_futures: List[Awaitable[tuple[np.ndarray, Dict[str, Any]]]] = []

        for item in video_mm_items:
            if isinstance(item, dict) and URL_VARIANT_KEY in item:
                url = item[URL_VARIANT_KEY]
                video_futures.append(self.load_video(url, media_io_kwargs))
                logger.debug("Preparing to load video from URL: %s...", url[:80])
            elif isinstance(item, dict) and DECODED_VARIANT_KEY in item:
                if self._enable_frontend_decoding:
                    metadata = item[DECODED_VARIANT_KEY]
                    video_futures.append(self._load_decoded_video(metadata))
                else:
                    raise ValueError(
                        "Received decoded video data but enable_frontend_decoding=False. "
                        "Enable frontend decoding to transfer decoded video frames via NIXL."
                    )

        results = await asyncio.gather(*video_futures, return_exceptions=True)
        loaded_videos: list[tuple[np.ndarray, Dict[str, Any]]] = []
        collective_exceptions: list[str] = []
        status_error: HttpStatusError | None = None
        url_error: UrlValidationError | None = None
        decoder_error: MissingMediaDecoderError | None = None
        for media_item, result in zip(video_mm_items, results):
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                source = media_item.get(URL_VARIANT_KEY, "decoded")
                logger.error("Failed to load video from %s...: %s", source[:80], result)
                collective_exceptions.append(
                    f"Failed to load video from {source[:80]}...: {result}\n"
                )
                if status_error is None and isinstance(result, HttpStatusError):
                    status_error = result
                elif url_error is None and isinstance(result, UrlValidationError):
                    url_error = result
                elif decoder_error is None and isinstance(
                    result, MissingMediaDecoderError
                ):
                    decoder_error = result
                continue
            frames, metadata = result
            loaded_videos.append((np.ascontiguousarray(frames), metadata))

        if status_error is not None:
            raise status_error
        if url_error is not None:
            raise url_error

        if decoder_error is not None:
            # Keep the actionable type: the generic aggregate below would erase
            # it, and a missing decoder is deployment configuration handlers
            # must be able to distinguish from a bad request.
            raise decoder_error

        if collective_exceptions:
            raise Exception("".join(collective_exceptions))

        return loaded_videos
