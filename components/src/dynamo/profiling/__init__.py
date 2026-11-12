# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-demand capture coordination, independent of frontend and HTTP hosting."""

from typing import TYPE_CHECKING

from .errors import CoordinatorError
from .models import Config, Plan, RuntimeSnapshot, Target, Worker

if TYPE_CHECKING:
    from .coordinator import Coordinator


def __getattr__(name: str):
    # Worker metadata queries do not need session management or trace parsing.
    if name == "Coordinator":
        from .coordinator import Coordinator

        return Coordinator
    raise AttributeError(name)


__all__ = [
    "Config",
    "Coordinator",
    "CoordinatorError",
    "Plan",
    "RuntimeSnapshot",
    "Target",
    "Worker",
]
