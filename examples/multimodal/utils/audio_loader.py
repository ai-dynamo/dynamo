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
from io import BytesIO
from typing import Tuple
from urllib.parse import urlparse

import httpx
import librosa
import numpy as np

logger = logging.getLogger(__name__)


class AudioLoader:
    CACHE_SIZE_MAXIMUM = 8

    def __init__(self, cache_size: int = CACHE_SIZE_MAXIMUM):
        self._http_timeout = 30.0
        self._http_client = httpx.AsyncClient(timeout=self._http_timeout)
        self._audio_cache: dict[str, Tuple[np.ndarray, float]] = {}
        self._cache_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=cache_size)

    async def load_audio(
        self, audio_url: str, sampling_rate: int = 16000
    ) -> Tuple[np.ndarray, float]:
        parsed_url = urlparse(audio_url)

        # For HTTP(S) URLs, check cache first
        if parsed_url.scheme in ("http", "https"):
            audio_url_lower = audio_url.lower()
            if audio_url_lower in self._audio_cache:
                logger.debug(f"Audio found in cache for URL: {audio_url}")
                return self._audio_cache[audio_url_lower]

        try:
            if parsed_url.scheme in ("http", "https"):
                if not self._http_client:
                    raise RuntimeError("HTTP client not initialized")

                response = await self._http_client.get(audio_url)
                response.raise_for_status()

                if not response.content:
                    raise ValueError("Empty response content from audio URL")

                audio_data_stream = BytesIO(response.content)
            else:
                raise ValueError(f"Invalid audio source scheme: {parsed_url.scheme}")

            # librosa.load is sync, so offload to a thread to avoid blocking the event loop
            def _load_audio():
                return librosa.load(audio_data_stream, sr=16000)

            audio_data, sr = await asyncio.to_thread(_load_audio)

            # Cache HTTP(S) URLs
            if parsed_url.scheme in ("http", "https"):
                audio_url_lower = audio_url.lower()
                # Cache the audio for future use, and evict the oldest audio if the cache is full
                if self._cache_queue.full():
                    oldest_audio_url = await self._cache_queue.get()
                    del self._audio_cache[oldest_audio_url]

                self._audio_cache[audio_url_lower] = (audio_data, sr)
                await self._cache_queue.put(audio_url_lower)

            return audio_data, sr

        except httpx.HTTPError as e:
            logger.error(f"HTTP error loading audio: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise ValueError(f"Failed to load audio: {e}")
