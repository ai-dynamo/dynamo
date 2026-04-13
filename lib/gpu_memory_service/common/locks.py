# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class RequestedLockType(str, Enum):
    RW = "rw"
    RO = "ro"
    AUTO = "auto"


class GrantedLockType(str, Enum):
    RW = "rw"
    RO = "ro"
