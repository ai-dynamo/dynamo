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

"""
Shared protocol definitions for the streaming ASR pipeline.

This module defines the Pydantic models used for communication between
the audio_chunker and asr_inference workers.
"""

import base64

import numpy as np
from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Metadata describing the chunking configuration for streaming ASR."""

    left_context_samples: int
    chunk_samples: int
    right_context_samples: int
    sample_rate: int


class ASRRequest(BaseModel):
    """Request sent from audio_chunker to asr_inference worker."""

    audio_b64: str  # base64-encoded audio tensor bytes (float32)
    audio_length: int  # actual audio length in samples
    metadata: ChunkMetadata

    @classmethod
    def from_audio(
        cls, audio: np.ndarray, audio_length: int, metadata: ChunkMetadata
    ) -> "ASRRequest":
        """Create an ASRRequest from a numpy audio array."""
        audio_bytes = audio.astype(np.float32).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return cls(audio_b64=audio_b64, audio_length=audio_length, metadata=metadata)

    def decode_audio(self) -> np.ndarray:
        """Decode the base64 audio back to a numpy array."""
        audio_bytes = base64.b64decode(self.audio_b64)
        return np.frombuffer(audio_bytes, dtype=np.float32)


class ASRResponse(BaseModel):
    """Response sent from asr_inference worker back to audio_chunker."""

    text: str
    is_final: bool
