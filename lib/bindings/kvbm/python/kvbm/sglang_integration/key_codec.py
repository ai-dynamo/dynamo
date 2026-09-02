# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang adapter for Dynamo's canonical PLH page-key codec."""

from __future__ import annotations

from collections.abc import Sequence

from sglang.srt.mem_cache.unified_cache.unified_cache_linker import LinkerKeyDomain

# isort: split

from kvbm._core import DynamoPlhKeyCodec as _NativeDynamoPlhKeyCodec


class DynamoPlhKeyCodec:
    codec_id = "dynamo-plh-v1"

    def __init__(self, manager_namespace: bytes):
        self._native = _NativeDynamoPlhKeyCodec(manager_namespace)

    def extend_pages(
        self,
        *,
        parent_key: bytes | None,
        page_tokens: Sequence[int],
        page_size: int,
        key_domain: LinkerKeyDomain,
    ) -> list[bytes]:
        tokens = []
        for token in page_tokens:
            if not isinstance(token, int) or token < 0 or token > 0xFFFFFFFF:
                raise ValueError(
                    "dynamo-plh-v1 only accepts unsigned 32-bit token IDs."
                )
            tokens.append(token)
        return list(
            self._native.extend_pages(
                parent_key,
                tokens,
                page_size,
                key_domain.cache_salt,
                key_domain.extra_key,
            )
        )
