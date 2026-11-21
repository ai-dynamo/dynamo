# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility re-export; the implementation lives in ``dynamo.common.backend.push_egress``."""

from dynamo.common.backend.push_egress import (
    drive_push_egress,
    drive_push_egress_stream,
    push_egress_capable,
)

__all__ = ["drive_push_egress", "drive_push_egress_stream", "push_egress_capable"]
