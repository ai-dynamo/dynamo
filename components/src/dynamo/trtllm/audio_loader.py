# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
from typing import Any, Sequence
from urllib.parse import urlparse

import soundfile as sf

from dynamo.common.http import (
    HttpConnectionError,
    HttpStatusError,
    HttpTimeoutError,
    fetch_bytes,
)
from dynamo.common.http.url_validator import (
    UrlValidationError,
    UrlValidationPolicy,
    validate_media_url,
)
from dynamo.common.multimodal.media_source import decode_data_uri, describe_media_source


class AudioLoader:
    """Fetch and decode TRT-LLM audio inputs with bounded concurrency."""

    def __init__(self, max_bytes: int, url_policy: UrlValidationPolicy) -> None:
        self._max_bytes = max_bytes
        self._url_policy = url_policy
        self._semaphore = asyncio.Semaphore(4)

    async def load_audio_batch(self, items: Sequence[Any]) -> list[tuple[Any, int]]:
        """Load clips in input order, with at most four active loads per loader."""

        async def load_one(item: Any) -> tuple[Any, int]:
            async with self._semaphore:
                return await self._load_audio(item)

        tasks = [asyncio.create_task(load_one(item)) for item in items]
        try:
            return await asyncio.gather(*tasks)
        except (Exception, asyncio.CancelledError):
            # gather does not cancel siblings when one clip fails.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _load_audio(self, item: Any) -> tuple[Any, int]:
        url = item.get("Url") if isinstance(item, dict) else item
        source = describe_media_source(url) if isinstance(url, str) else "audio item"
        if not isinstance(url, str):
            raise HttpStatusError(400, "Malformed audio_url item", source)
        try:
            normalized_url = await validate_media_url(url, self._url_policy)
            scheme = urlparse(normalized_url).scheme
            if scheme == "data":
                content = decode_data_uri(normalized_url, max_bytes=self._max_bytes)
            elif scheme in ("http", "https"):
                content = await fetch_bytes(
                    normalized_url,
                    30.0,
                    policy=self._url_policy,
                    max_bytes=self._max_bytes,
                )
            else:
                raise HttpStatusError(400, "Unsupported audio URL scheme", source)
        except UrlValidationError as exc:
            raise HttpStatusError(400, str(exc), source) from exc
        except HttpTimeoutError as exc:
            raise HttpStatusError(408, "Timed out loading audio", source) from exc
        except HttpConnectionError as exc:
            raise HttpStatusError(400, "Could not load audio URL", source) from exc

        try:
            waveform, sample_rate = await asyncio.to_thread(
                sf.read, io.BytesIO(content), dtype="float32"
            )
            if waveform.size == 0 or sample_rate <= 0 or waveform.ndim not in (1, 2):
                raise ValueError("Audio has no decodable samples")
            return waveform, sample_rate
        except (RuntimeError, ValueError) as exc:
            raise HttpStatusError(400, "Invalid audio data", source) from exc
