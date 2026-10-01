# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class RequestedLockType(str, Enum):
    RW = "rw"
    RO = "ro"
    RW_OR_RO = "rw_or_ro"
    # "Adopt if possible": grant RW_DATA when a committed layout exists (reattach the
    # same pages), otherwise RW so the caller can build one. The requested mode is how
    # a client expresses adopt-vs-replace: asking for RW means "wipe it, I'll allocate
    # my own", asking for RW_DATA means "I only intend to write bytes into what's there".
    RW_DATA_OR_RW = "rw_data_or_rw"


class GrantedLockType(str, Enum):
    RW = "rw"
    RO = "ro"
    # Exclusive writer over a committed layout: may write bytes into the existing
    # allocations, may not change their shape (no allocate/free/metadata mutation).
    # Granted in place of RW while the layout is committed.
    RW_DATA = "rw_data"
