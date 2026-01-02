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
import functools
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
        # functools.lru_cache is not directly compatible with async methods.
        # We create a synchronous method for fetching and processing audio,
        # and then apply lru_cache to it. This cached synchronous method
        # is then called from our async method using asyncio.to_thread.
        self._load_and_process_audio_cached = functools.lru_cache(maxsize=cache_size)(
            self._load_and_process_audio
        )

    def _load_and_process_audio(
        self, audio_url: str, sampling_rate: int
    ) -> Tuple[np.ndarray, float]:
        """
        Synchronously loads and processes audio from a URL.
        This method is memoized using lru_cache.
        """
        with httpx.Client(timeout=self._http_timeout) as client:
            response = client.get(audio_url)
            response.raise_for_status()

            if not response.content:
                raise ValueError("Empty response content from audio URL")

            audio_data_stream = BytesIO(response.content)
            audio_data, sr = librosa.load(audio_data_stream, sr=sampling_rate)
            return audio_data, sr

    async def load_audio(
        self, audio_url: str, sampling_rate: int = 16000
    ) -> Tuple[np.ndarray, float]:
        parsed_url = urlparse(audio_url)

        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"Invalid audio source scheme: {parsed_url.scheme}")

        try:
            # Offload the synchronous, cached function to a separate thread
            # to avoid blocking the asyncio event loop.
            return await asyncio.to_thread(
                self._load_and_process_audio_cached, audio_url, sampling_rate
            )
        except httpx.HTTPError as e:
            logger.error(f"HTTP error loading audio: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise ValueError(f"Failed to load audio: {e}")
