# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test harness for Dynamo.

Tier 0 is values: stdlib only, no pytest, no HTTP, no Kubernetes, and — by
design — no ``dynamo.*``. Welding the harness to the runtime would make it
impossible to point one suite at an older release.
"""

from .argv import ArgForm, ArgV, is_shell_command_flag
from .facts import Fact, FactNotKnown, Status

__all__ = [
    "ArgForm",
    "ArgV",
    "Fact",
    "FactNotKnown",
    "Status",
    "is_shell_command_flag",
]
