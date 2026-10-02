# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Physical allocation identity, independent of prefix-cache semantics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PersistentAllocation:
    allocation_id: str
    size_bytes: int
    server_nonce: str
    gpu_uuid: str
