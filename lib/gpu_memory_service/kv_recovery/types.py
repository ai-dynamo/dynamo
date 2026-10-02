# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum


class RecoveryResult(str, Enum):
    RECOVERED = "RECOVERED"
    FRESH = "FRESH"


@dataclass(frozen=True)
class KVBlockRecord:
    block_id: int
    block_hash: bytes  # Native vLLM hash including cache group ID.
    num_tokens: int


@dataclass(frozen=True)
class KVTensorLayout:
    name: str
    allocation_id: str
    offset_bytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class KVRecoveryManifest:
    model: str
    num_blocks: int
    block_size: int
    hash_algorithm: str
    tensors: tuple[KVTensorLayout, ...]
    engine_version: str
