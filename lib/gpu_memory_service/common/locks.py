# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class RequestedLockType(str, Enum):
    RW = "rw"
    RO = "ro"
    RW_OR_RO = "rw_or_ro"
    # Persistent RW: allocations survive client disconnect and can be
    # re-attached by a new client with the same (engine_id, tag) key.
    # Used for KV-cache pools where the engine writes every forward pass
    # and must survive engine restart without losing bytes.
    RW_PERSISTENT = "rw_persistent"


class GrantedLockType(str, Enum):
    RW = "rw"
    RO = "ro"
    RW_PERSISTENT = "rw_persistent"
