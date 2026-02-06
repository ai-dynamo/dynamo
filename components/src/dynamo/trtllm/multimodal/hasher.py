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

"""Simple multimodal content hasher based on raw bytes using BLAKE3."""

from blake3 import blake3


class MultimodalHasher:
    """Hashes multimodal content (images, videos, etc.) based on raw bytes.

    Fast and deterministic - no decoding overhead. Uses BLAKE3 for cryptographic
    hashing of raw file bytes.

    Note: Different file formats of the same visual content will produce different
    hashes. This is by design - the hasher operates on raw bytes, not semantic content.
    """

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        """Hash raw bytes using BLAKE3.

        Args:
            data: Raw bytes to hash

        Returns:
            Hex digest string (64 characters for BLAKE3)

        Example:
            >>> hasher = MultimodalHasher()
            >>> hash_result = hasher.hash_bytes(b"hello world")
            >>> isinstance(hash_result, str)
            True
            >>> len(hash_result)
            64
        """
        return blake3(data).hexdigest()
