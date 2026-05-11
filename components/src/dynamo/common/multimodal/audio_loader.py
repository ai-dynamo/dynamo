# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any, Awaitable, Dict, Final, List
from urllib.parse import urlparse

import numpy as np

import dynamo.nixl_connect as nixl_connect
from dynamo.common.multimodal.http_client import get_http_client, get_http_semaphore
from dynamo.common.multimodal.url_validator import (
    UrlValidationPolicy,
    fetch_with_revalidation,
    validate_media_url,
)
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.common.utils.media_nixl import read_decoded_media_via_nixl
from dynamo.common.utils.runtime import run_async

logger = logging.getLogger(__name__)

# Constants for multimodal data variants
URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"


try:
    from vllm.multimodal.media import MediaConnector
    from vllm.multimodal.media.audio import AudioMediaIO
except ImportError:
    MediaConnector = None  # type: ignore[assignment]
    AudioMediaIO = None  # type: ignore[assignment]


def _require_vllm_audio_media() -> tuple[Any, Any]:
    """Return vLLM's audio media components, raising if not installed."""
    if MediaConnector is None or AudioMediaIO is None:
        raise RuntimeError(
            "vLLM multimodal media components are required to decode `audio_url` "
            "inputs in the vLLM backend."
        )
    return MediaConnector, AudioMediaIO


class AudioLoader:
    """Async audio loader for multimodal pipelines.

    Delegates URL fetching and decoding to vLLM's ``MediaConnector`` +
    ``AudioMediaIO`` so that the exact same loading logic runs whether the
    request arrives via ``vllm serve`` or through Dynamo.  Returns
    ``(waveform, sample_rate)`` tuples at the native sample rate — vLLM's
    model-specific ``MultiModalDataParser`` handles resampling and channel
    normalization downstream.

    Also supports the NIXL decoded variant for frontend-decoded audio
    transferred via RDMA.
    """

    def __init__(
        self,
        http_timeout: float = 30.0,
        enable_frontend_decoding: bool = False,
        url_policy: UrlValidationPolicy | None = None,
    ) -> None:
        if http_timeout <= 0:
            raise ValueError(f"http_timeout must be positive, got {http_timeout}")
        self._http_timeout = http_timeout
        self._enable_frontend_decoding = enable_frontend_decoding
        self._url_policy = url_policy or UrlValidationPolicy.from_env()
        self._nixl_connector = None
        self._vllm_media_connector = None
        if self._enable_frontend_decoding:
            self._nixl_connector = nixl_connect.Connector()
            run_async(self._nixl_connector.initialize)

    def _get_vllm_media_connector(self) -> Any:
        if self._vllm_media_connector is None:
            MediaConnector, _ = _require_vllm_audio_media()
            # Confine vLLM's own local-path access to the same prefix we enforce.
            # Empty string matches vLLM's secure default (no local access).
            allowed = self._url_policy.allowed_local_path or ""
            self._vllm_media_connector = MediaConnector(
                allowed_local_media_path=allowed
            )

        return self._vllm_media_connector

    def _create_vllm_audio_io(self) -> Any:
        _, AudioMediaIO = _require_vllm_audio_media()
        return AudioMediaIO()

    @_nvtx.annotate("mm:audio:load_with_vllm", color="cyan")
    async def _load_audio_with_vllm(self, audio_url: str) -> tuple[np.ndarray, float]:
        normalized_url = await validate_media_url(audio_url, self._url_policy)
        media_io = self._create_vllm_audio_io()

        # HTTP(S) goes through our SSRF-safe fetcher so each redirect hop is
        # revalidated; vLLM's own fetcher honors redirects without re-checking.
        # data: and file:// never touch the network, so vLLM can handle them.
        if urlparse(normalized_url).scheme in ("http", "https"):
            http_client = get_http_client(self._http_timeout)
            async with get_http_semaphore():
                response = await fetch_with_revalidation(
                    http_client, normalized_url, self._url_policy
                )
                response.raise_for_status()
                content = response.content
            return await asyncio.to_thread(media_io.load_bytes, content)

        connector = self._get_vllm_media_connector()
        return await connector.load_from_url_async(
            normalized_url, media_io, fetch_timeout=self._http_timeout
        )

    @_nvtx.annotate("mm:audio:load_audio", color="cyan")
    async def load_audio(self, audio_url: str) -> tuple[np.ndarray, float]:
        """Load audio from a URL and return a (waveform, sample_rate) tuple.

        Supports http(s), data: URIs, file:// paths, and bare filesystem paths.
        Audio is loaded at the native sample rate — no resampling is performed.
        """
        try:
            waveform, sr = await self._load_audio_with_vllm(audio_url)
            if waveform.size == 0:
                raise ValueError(
                    f"Failed to decode audio from {audio_url}. Decoded waveform is empty."
                )
            return waveform, sr
        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.error("Error loading audio from %s: %s", audio_url, exc)
            raise ValueError(f"Failed to load audio from {audio_url}: {exc}") from exc

    async def _load_decoded_audio(
        self, decoded_metadata: Dict[str, Any]
    ) -> tuple[np.ndarray, float]:
        """Read pre-decoded audio via NIXL RDMA."""
        if self._nixl_connector is None:
            raise RuntimeError("NIXL connector is not initialized")

        result = await read_decoded_media_via_nixl(
            self._nixl_connector,
            decoded_metadata,
            return_metadata=True,
        )
        frames, metadata = result
        if metadata is None:
            metadata = {}
        sr = metadata.get("sample_rate", 16000)
        return frames, float(sr)

    async def load_audio_batch(
        self,
        audio_mm_items: List[Dict[str, Any]],
    ) -> List[tuple[np.ndarray, float]]:
        """Load a batch of audio files from multimodal data items.

        Supports two paths:
        1. Url variant: Download and decode audio via vLLM's MediaConnector
        2. Decoded variant: Read pre-decoded audio via NIXL RDMA
           (requires enable_frontend_decoding=True)

        Returns:
            List of (waveform, sample_rate) tuples.
        """
        audio_futures: List[Awaitable[tuple[np.ndarray, float]]] = []

        for idx, item in enumerate(audio_mm_items):
            if isinstance(item, dict) and URL_VARIANT_KEY in item:
                url = item[URL_VARIANT_KEY]
                audio_futures.append(self.load_audio(url))
                logger.debug("Preparing to load audio from URL: %s...", url[:80])
            elif isinstance(item, dict) and DECODED_VARIANT_KEY in item:
                if self._enable_frontend_decoding:
                    metadata = item[DECODED_VARIANT_KEY]
                    audio_futures.append(self._load_decoded_audio(metadata))
                else:
                    raise ValueError(
                        "Received decoded audio data but enable_frontend_decoding=False. "
                        "Enable frontend decoding to transfer decoded audio via NIXL."
                    )
            else:
                raise ValueError(
                    f"Invalid audio multimodal item at index {idx}. "
                    "Expected dict with 'Url' or 'Decoded' key."
                )

        results = await asyncio.gather(*audio_futures, return_exceptions=True)
        loaded_audio: list[tuple[np.ndarray, float]] = []
        collective_exceptions: list[str] = []
        for media_item, result in zip(audio_mm_items, results, strict=True):
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                source = media_item.get(URL_VARIANT_KEY, "decoded")
                logger.error("Failed to load audio from %s...: %s", source[:80], result)
                collective_exceptions.append(
                    f"Failed to load audio from {source[:80]}...: {result}\n"
                )
                continue
            loaded_audio.append(result)

        if collective_exceptions:
            raise Exception("".join(collective_exceptions))

        return loaded_audio
